/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdio>
#include <cstdlib>
#include "base/arch.h"
#include "base/logging.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "guest_abi.h"

#include <sys/select.h>
#include <unistd.h>

#include "kern/lv2/sys_event.h"
#include "kern/lv2/wait_probe.h"
#include "kern/process.h"
#include "kern/ps4/dev/socket_dev.h"

#include "base/atomic.h"
#include "base/containers/pair.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "kern/crash.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kEventTrace, "DELTA_EVENT_TRACE", false);
DELTA_OPTION(bool, kKeventTrace, "DELTA_KEVENT_TRACE", false);
DELTA_OPTION(long, kEventStallSecs, "DELTA_EVENT_STALL", 0);
DELTA_OPTION(long, kEopPumpMs, "DELTA_PS5_EOPPUMP", 0);
DELTA_OPTION(bool, kIdent0Vblank, "DELTA_PS5_IDENT0_VBLANK", false);
}  // namespace

namespace krnl {
// All live equeues, so the vblank pump can fan flip events to every one of them
// without knowing which equeue a given flip event was registered on.
static base::Mutex g_eq_reg_m;
static base::Vector<Equeue*> g_equeues;

// EVFILT_DISPLAY (-13) and Sony's videoout filter (-14) carry real vblank/flip
// waits; with no display hardware, a synthetic 60 Hz tick keeps them from
// blocking forever.
static constexpr i16 kEVFILT_READ = -1;
static constexpr i16 kEVFILT_USER = -11;
static constexpr u32 kNOTE_TRIGGER = 0x01000000;
static void WatchSocket(u32 fd);
static constexpr i16 kEVFILT_DISPLAY = -13;
static constexpr i16 kEVFILT_VIDEOOUT = -14;
// Filter -14 carries two things: vblank waiters put their event id in the HIGH
// bits of ident (0x6 << 48 up), sceGnmAddEqEvent registers under the bare id.
// This bound tells them apart.
static constexpr u64 kGnmIdentMax = 0x10000;
static base::Atomic<bool> g_vblank_started{false};

// Flips the title has actually submitted. The display event's data>>16 carries
// this (not the vblank tick) so render-frame pacing tracks real flips.
static base::Atomic<u64> g_flip_count{0};
u64 FlipCount() {
  return g_flip_count.load();
}

// Low-bit TSC nonce for the display event's bits 0..11 so a polling title sees
// each event as new; real rdtsc on native, monotonic clock on FEX. See
// dce_dev.cc.
static u64 TscNonce() {
#if defined(DELTA_BACKEND_NATIVE)
  return __builtin_ia32_rdtsc();
#else
  return static_cast<u64>(base::TickClock::NowNs());
#endif
}

// Guest fds with a live EVFILT_READ knote + the thread selecting on their host
// sockets. A read knote's source is outside the guest, so nothing here would
// ever mark it active; without this, Minecraft's rtc::PhysicalSocketServer
// never wakes.
static base::Mutex g_watch_m;
static base::Set<u32> g_watched;
static base::Atomic<bool> g_watch_started{false};

static void WatchSocket(u32 fd) {
  if (!FdToSocket(fd))
    return;
  {
    base::LockGuard<base::Mutex> lk(g_watch_m);
    if (!g_watched.insert(fd).second)
      return;
  }
  bool expected = false;
  if (!g_watch_started.compare_exchange_strong(expected, true))
    return;
  BASE_LOGI("kevent", "socket read-poll started");
  base::SpawnDetachedThread("sys_event", [] {
    for (;;) {
      fd_set rd;
      FD_ZERO(&rd);
      int max_fd = -1;
      base::Vector<base::Pair<u32, int>> live;
      {
        base::LockGuard<base::Mutex> lk(g_watch_m);
        for (u32 g : g_watched)
          if (auto* s = FdToSocket(g)) {
            live.emplace_back(g, s->HostFd());
            FD_SET(s->HostFd(), &rd);
            max_fd = base::Max(max_fd, s->HostFd());
          }
      }
      timeval tv{0, 20000};  // 20 ms; also the retry tick when nothing is live
      if (max_fd < 0 || ::select(max_fd + 1, &rd, nullptr, nullptr, &tv) <= 0) {
        base::SleepForMilliseconds(5);
        continue;
      }
      for (auto& [guestFd, HostFd] : live)
        if (FD_ISSET(HostFd, &rd))
          TriggerAllEqueues(guestFd, kEVFILT_READ, 1);
    }
  });
}

// Start the 60 Hz EVFILT_DISPLAY pump once, on the first vblank registration,
// so the timer thread only runs when something waits on it.
static void StartVblankPump() {
  bool expected = false;
  if (!g_vblank_started.compare_exchange_strong(expected, true))
    return;
  BASE_LOGI("vblank", "pump started (60 Hz, EVFILT_DISPLAY/VIDEOOUT)");
  base::SpawnDetachedThread("sys_event", [] {
    u64 count = 0;
    for (;;) {
      base::SleepForMicroseconds(16667);  // ~60 Hz
      ++count;
      // data>>16 = counter, bits 12..15 = 1..14 per-event sequence the title
      // polls for "new", bits 0..11 = TSC nonce. Packing only count<<16 left
      // the sequence 0: the title woke every tick but saw "no new event".
      u64 seq = (count - 1) % 14 + 1;  // 1..14
      u64 tsc = TscNonce() & 0xFFF;
      // Vblank (-14): a free-running tick for vblank waiters / frame timing.
      i64 vdata = static_cast<i64>((count << 16) | (seq << 12) | tsc);
      TriggerAllEqueues(-1, kEVFILT_VIDEOOUT, vdata);
      // Flip (-13): the engine reads data>>16 as the LAST flipped frame index
      // and asserts unless the sim has produced it. Never post before the first
      // real flip (during loading, any value is "out of range"); afterwards it
      // rides g_flipCount.
      u64 flips = g_flip_count.load();
      if (flips > 0) {
        u64 idx = flips - 1;
        i64 fdata =
            static_cast<i64>((idx << 16) | ((idx % 14 + 1) << 12) | tsc);
        TriggerAllEqueues(-1, kEVFILT_DISPLAY, fdata);
      }
    }
  });
}

// A GPU end-of-pipe interrupt: RELEASE_MEM/EVENT_WRITE_EOP's INT_SEL asks the
// CP to raise one when the label write lands; libSceGnmDriver turns it into an
// event on the equeue sceGnmAddEqEvent registered (filter -14, small ident,
// e.g. GTA:SA's 0x5/0x40). Deliberately NOT the 60 Hz pump's business: that
// tick serves the vblank waiters sharing the filter, and firing Gnm events on
// it says "GPU done" when it did not. EOP interrupts since boot; a knote
// compares so none is lost.
static base::Atomic<u64> g_eop_seq{0};
u64 GpuEndOfPipeCount() {
  return g_eop_seq.load(base::memory_order_relaxed);
}

void NoteGpuEndOfPipe() {
  const u64 n = g_eop_seq.fetch_add(1) + 1;
  if (kEventTrace && (n == 1 || (n % 2000) == 0))
    BASE_LOGI("gpueop", "end-of-pipe interrupt #{}", (unsigned long long)n);
  // Same packing as the display events: the counter above bit 16, a 1..14
  // sequence so a poller can tell a new event from a repeat, a TSC nonce below.
  const i64 data = static_cast<i64>((n << 16) | (((n - 1) % 14 + 1) << 12) |
                                    (TscNonce() & 0xFFF));
  base::LockGuard<base::Mutex> lk(g_eq_reg_m);
  for (auto* eq : g_equeues)
    eq->TriggerGnm(data);
}

// The id of the most recent completion, for a knote that was not listening when
// it happened: re-arming it with a counter would hand the title an id it can
// never match.
static base::Atomic<u64> g_last_eop_ctx{0};

void NoteGpuEndOfPipeCtx(u64 context_id) {
  g_eop_seq.fetch_add(1, base::memory_order_relaxed);
  g_last_eop_ctx.store(context_id, base::memory_order_relaxed);
  base::LockGuard<base::Mutex> lk(g_eq_reg_m);
  for (auto* eq : g_equeues)
    eq->TriggerGnm(static_cast<i64>(context_id), /*raw_data=*/true);
}

void NoteFlip() {
  u64 idx = g_flip_count.fetch_add(1);  // index of the flip that just completed
  // Post the flip event immediately so a blocked waiter wakes now, not on the
  // next pump tick; data>>16 is the LAST completed flip index, which the sim
  // has produced.
  u64 seq = idx % 14 + 1;
  u64 tsc = TscNonce() & 0xFFF;
  i64 data = static_cast<i64>((idx << 16) | (seq << 12) | tsc);
  TriggerAllEqueues(-1, kEVFILT_DISPLAY, data);
}

Equeue::Equeue(ObjectTable& objects, const char* nm)
    : Object(objects, OType::kEqueue) {
  if (nm)
    name_ = nm;
  base::LockGuard<base::Mutex> lk(g_eq_reg_m);
  g_equeues.push_back(this);
}

Equeue::~Equeue() {
  base::LockGuard<base::Mutex> lk(g_eq_reg_m);
  for (size_t i = 0; i < g_equeues.size(); i++) {
    if (g_equeues[i] == this) {
      g_equeues.erase(g_equeues.begin() + i);
      break;
    }
  }
}

Equeue::Knote* Equeue::Find(u64 ident, i16 filter) {
  for (auto& k : notes_)
    if (k.ev.ident == ident && k.ev.filter == filter)
      return &k;
  return nullptr;
}

int Equeue::Kevent(const kevent_t* changes,
                   int nchanges,
                   kevent_t* out,
                   int nout,
                   const ktimespec* to) {
  base::UniqueLock<base::Mutex> lk(m_);

  // 1) apply the changelist.
  for (int i = 0; i < nchanges; i++) {
    const auto& c = changes[i];
    BASE_LOGI("kevent",
              "change ident={:#x} filter={} flags={:#x} fflags={:#x} "
              "data={:#x} udata={:p}",
              (unsigned long long)c.ident, c.filter, c.flags, c.fflags,
              (unsigned long long)c.data, c.udata);
    if (c.flags & kEV_DELETE) {
      for (size_t j = 0; j < notes_.size(); j++)
        if (notes_[j].ev.ident == c.ident && notes_[j].ev.filter == c.filter) {
          notes_.erase(notes_.begin() + j);
          break;
        }
      continue;
    }
    // EVFILT_USER + NOTE_TRIGGER is one thread poking another awake, not a
    // registration: fire the existing knote, don't replace and clear it (this
    // is how WebRTC's SocketServer::WakeUp reaches a parked kevent; dropping it
    // hangs every rtc::Thread::BlockingCall).
    if (c.filter == kEVFILT_USER && (c.fflags & kNOTE_TRIGGER)) {
      if (auto* k = Find(c.ident, c.filter)) {
        k->active = true;
        k->ev.data = c.data;
        // The trigger carries the udata, the registration doesn't; keeping null
        // made sceKernelGetEventUserData return null and Minecraft deref it
        // into a vcall.
        if (c.udata)
          k->ev.udata = c.udata;
        cv_.NotifyAll();
      }
      continue;
    }
    // EV_ADD (and plain enable): register/replace.
    if (auto* k = Find(c.ident, c.filter)) {
      k->ev = c;
      k->active = false;
    } else {
      notes_.push_back({c, false});
    }
    // First vblank registration kicks off the synthetic 60 Hz pump.
    if (c.filter == kEVFILT_DISPLAY || c.filter == kEVFILT_VIDEOOUT)
      StartVblankPump();
    // A read knote on a socket is the only knote whose source lives outside the
    // guest, so nothing here can set it active. Watch the host fd instead.
    if (c.filter == kEVFILT_READ)
      WatchSocket(static_cast<u32>(c.ident));
  }

  // 2) collect ready events, waiting if asked.
  auto collect = [&]() -> int {
    int n = 0;
    for (auto& k : notes_) {
      if (n >= nout)
        break;
      if (!k.active)
        continue;
      out[n++] = k.ev;
      // edge semantics: a fired knote is consumed until its source fires again.
      k.active = false;
      k.eop_seen = GpuEndOfPipeCount();
    }
    return n;
  };

  if (nout <= 0)
    return 0;

  // DELTA_PS5_EOPPUMP: diagnostic. Treat a Gnm knote as due again after this
  // many milliseconds of silence, to tell "the title is one event short" apart
  // from "the title is stuck on something else entirely".
  if (kEopPumpMs > 0) {
    const auto now = base::TimeTicks::Now();
    for (auto& k : notes_) {
      if (k.active || k.ev.filter != kEVFILT_VIDEOOUT ||
          k.ev.ident >= kGnmIdentMax)
        continue;
      k.active = true;
      k.ev.data =
          static_cast<i64>(g_last_eop_ctx.load(base::memory_order_relaxed));
    }
    (void)now;
  }

  // Re-arm Gnm knotes the GPU has run past: our submits finish inside the
  // submit call, so an edge raised between kevents would be lost with nothing
  // to re-raise it.
  {
    const u64 eop = GpuEndOfPipeCount();
    for (auto& k : notes_) {
      if (k.active || k.ev.filter != kEVFILT_VIDEOOUT ||
          k.ev.ident >= kGnmIdentMax || k.eop_seen >= eop)
        continue;
      k.active = true;
      k.ev.data =
          static_cast<i64>(g_last_eop_ctx.load(base::memory_order_relaxed));
    }
  }

  int got = collect();
  if (got > 0) {
    if (kEventTrace)
      BASE_LOGI("kevent",
                "return immediate n={} filter={} ident={:#x} data={:#x}", got,
                out[0].filter, (unsigned long long)out[0].ident,
                (unsigned long long)out[0].data);
    return got;
  }

  bool ready = false;
  auto pred = [&] {
    for (auto& k : notes_)
      if (k.active)
        return true;
    return false;
  };

  // An untimed wait on a knote-less queue can never return; report once per
  // queue, it always means a missing registration on our side.
  if (!to && notes_.empty() && !warned_empty_wait_) {
    warned_empty_wait_ = true;
    BASE_LOGI("kevent", "tid={} waits forever on '{}' (fd={}): no knotes",
              (long)gettid(), name_.empty() ? "(unnamed)" : name_.c_str(),
              handle());
  }
  if (!to) {
    // Report once what an untimed wait is registered for if it goes long: a
    // queue with knotes that never go active is a source we do not drive.
    if (kEventStallSecs > 0) {
      const auto until =
          base::TimeTicks::Now() + base::Seconds(kEventStallSecs);
      if (!cv_.WaitUntil(lk, until, pred)) {
        ReportRegistrationsLocked(handle());
        cv_.Wait(lk, pred);
      }
    } else {
      cv_.Wait(lk, pred);
    }
    ready = true;
  } else {
    auto dur =
        base::Seconds(to->tv_sec) + base::Microseconds((to->tv_nsec) / 1000);
    ready = cv_.WaitFor(lk, dur, pred);
  }
  if (!ready) {
    if (kEventTrace)
      BASE_LOGI("kevent", "timeout nout={}", nout);
    const auto now = base::TimeTicks::Now();
    if (idle_since_ == base::TimeTicks{})
      idle_since_ = now;
    if (kEventStallSecs > 0 && !reported_idle_ &&
        now - idle_since_ >= base::Seconds(kEventStallSecs)) {
      reported_idle_ = true;
      ReportRegistrationsLocked(handle());
      // On the waiting thread itself, so the walk is this call's own stack:
      // which of the title's calls is parked here.
      GuestStackTrace("kevent-stall", 12);
    }
    return 0;
  }
  idle_since_ = {};
  reported_idle_ = false;
  got = collect();
  if (kEventTrace && got > 0)
    BASE_LOGI("kevent", "return wait n={} filter={} ident={:#x} data={:#x}",
              got, out[0].filter, (unsigned long long)out[0].ident,
              (unsigned long long)out[0].data);
  return got;
}

void Equeue::ReportRegistrations(int fd) {
  base::LockGuard<base::Mutex> lk(m_);
  ReportRegistrationsLocked(fd);
}

// Caller holds m.
void Equeue::ReportRegistrationsLocked(int fd) {
  base::String line;
  base::FormatTo(line, "kq={} ({}) long wait, registered:", fd, name_.c_str());
  for (const auto& k : notes_)
    base::FormatTo(line, " [ident={:#x} filter={} active={}]",
                   (unsigned long long)k.ev.ident, (int)k.ev.filter,
                   (int)k.active);
  BASE_LOGI("kevent", "{}", line.c_str());
}

void Equeue::AddEvent(u64 ident, i16 filter, void* udata) {
  base::LockGuard<base::Mutex> lk(m_);
  kevent_t ev{};
  ev.ident = ident;
  ev.filter = filter;
  ev.flags = kEV_CLEAR;
  ev.udata = udata;
  if (auto* k = Find(ident, filter)) {
    k->ev = ev;
    k->active = false;
  } else {
    notes_.push_back({ev, false});
  }
  if (filter == kEVFILT_DISPLAY || filter == kEVFILT_VIDEOOUT)
    StartVblankPump();
}

bool Equeue::RemoveEvent(u64 ident, i16 filter) {
  base::LockGuard<base::Mutex> lk(m_);
  for (size_t j = 0; j < notes_.size(); j++)
    if (notes_[j].ev.ident == ident && notes_[j].ev.filter == filter) {
      notes_.erase(notes_.begin() + j);
      return true;
    }
  return false;
}

// The Gnm half of filter -14: every knote registered under a small ident gets
// its own event id. sceGnmGetEqEventType reads data whole and GetEqTimeStamp
// reads data >> 16, so the id must survive in the low bits.
void Equeue::TriggerGnm(i64 data, bool raw_data) {
  base::LockGuard<base::Mutex> lk(m_);
  bool any = false;
  for (auto& k : notes_) {
    if (k.ev.filter != kEVFILT_VIDEOOUT || k.ev.ident >= kGnmIdentMax)
      continue;
    k.active = true;
    // AGC: the data IS the context id. Gnm (PS4): the id has to survive in the
    // low bits, because sceGnmGetEqEventType reads the data there.
    k.ev.data = raw_data ? data
                         : ((data & ~static_cast<i64>(0xFFF)) |
                            static_cast<i64>(k.ev.ident));
    any = true;
  }
  if (any)
    cv_.NotifyAll();
}

void Equeue::Trigger(i64 ident, i16 filter, i64 data) {
  base::LockGuard<base::Mutex> lk(m_);
  bool any = false;
  for (auto& k : notes_) {
    // filter==0 is a wildcard (no real EVFILT is 0); ident<0 matches any.
    if (filter != 0 && k.ev.filter != filter)
      continue;
    if (ident >= 0 && k.ev.ident != static_cast<u64>(ident))
      continue;
    // A Gnm registration is not a vblank waiter and must not be told the GPU
    // finished on a timer. DELTA_PS5_IDENT0_VBLANK probes whether ident 0 is
    // one of those or a videoout FLIP registration (event id 0), which lands on
    // the same ident.
    if (filter == kEVFILT_VIDEOOUT && k.ev.ident < kGnmIdentMax &&
        !(kIdent0Vblank && k.ev.ident == 0))
      continue;
    k.active = true;
    k.ev.data = data;
    any = true;
  }
  if (any)
    cv_.NotifyAll();
}

void TriggerAllEqueues(i64 ident, i16 filter, i64 data) {
  base::LockGuard<base::Mutex> lk(g_eq_reg_m);
  for (auto* eq : g_equeues)
    eq->Trigger(ident, filter, data);
}

int PS4ABI sys_kqueue() {
  auto* eq = new Equeue(Proc::GetActive()->GetObjTable(), nullptr);
  BASE_LOGI("kqueue", "-> fd={}", eq->handle());
  return eq->handle();
}

int PS4ABI sys_kqueueex(const char* name, int flags) {
  auto* eq = new Equeue(Proc::GetActive()->GetObjTable(), name);
  BASE_LOGI("kqueueex", "name={} flags={:#x} -> fd={}", name ? name : "(null)",
            flags, eq->handle());
  return eq->handle();
}

int PS4ABI sys_kevent(int kq,
                      const kevent_t* changelist,
                      int nchanges,
                      kevent_t* eventlist,
                      int nevents,
                      const ktimespec* to) {
  auto* obj = Proc::GetActive()->GetObjTable().Get(kq);
  if (!obj || obj->type() != Object::OType::kEqueue) {
    BASE_LOGI("kevent", "bad kq fd={}", kq);
    return -SysError::eBADF;
  }
  // A thread blocked here is waiting for an event that may never be posted –
  // the same "wedged title" question the umtx/semaphore probes answer, and it
  // was the one wait they could not see.
  WaitProbe wp("kevent", (long)kq, (long)nevents);
  auto* eq = static_cast<Equeue*>(obj);
  const auto t0 = base::TimeTicks::Now();
  int r = eq->Kevent(changelist, nchanges, eventlist, nevents, to);
  // A wait this long is a title that is not going to wake up on its own. What
  // it registered says which event never arrived, and that is the only thing
  // the queue can tell us from the outside.
  if (kEventStallSecs > 0 &&
      base::TimeTicks::Now() - t0 >= base::Seconds(kEventStallSecs))
    eq->ReportRegistrations(kq);
  if (kKeventTrace) {
    base::String line;
    base::FormatTo(line, "kq={} nchanges={} -> {}", kq, nchanges, r);
    for (int i = 0; i < nchanges && changelist && i < 4; i++)
      base::FormatTo(line, " chg[ident={:#x} filter={} flags={:#x}]",
                     (unsigned long long)changelist[i].ident,
                     (int)changelist[i].filter, (unsigned)changelist[i].flags);
    BASE_LOGI("kevent", "{}", line.c_str());
  }
  return r;
}
}  // namespace krnl
