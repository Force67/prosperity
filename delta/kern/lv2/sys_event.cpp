/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include <sys/select.h>
#include <unistd.h>

#include "kern/proc.h"
#include "wait_probe.h"
#include "kern/ps4/dev/socket_dev.h"
#include "sys_event.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kEventTrace, "DELTA_EVENT_TRACE", false);
DELTA_OPTION(bool, kKeventTrace, "DELTA_KEVENT_TRACE", false);
DELTA_OPTION(long, kEventStallSecs, "DELTA_EVENT_STALL", 0);
}  // namespace

namespace krnl {
// All live equeues, so the vblank pump can fan flip events to every one of them
// without knowing which equeue a given flip event was registered on.
static std::mutex g_eqRegM;
static base::Vector<equeue *> g_equeues;

// EVFILT_DISPLAY (-13) and Sony's videoout event filter (-14) are both used by
// real system modules for vblank/flip waits. A thread that waits on either
// blocks in kevent until a display event arrives. With no real display hardware,
// a synthetic 60 Hz tick keeps those waits from blocking forever.
static constexpr i16 kEVFILT_READ = -1;
static constexpr i16 kEVFILT_USER = -11;
static constexpr u32 kNOTE_TRIGGER = 0x01000000;
static void watchSocket(u32 fd);
static constexpr i16 kEVFILT_DISPLAY = -13;
static constexpr i16 kEVFILT_VIDEOOUT = -14;
// Filter -14 carries two unrelated things. libSceVideoOut's vblank waiters put
// their event id in the HIGH bits of ident (0x6 << 48 and up); sceGnmAddEqEvent
// registers the GPU's graphics-core events under the bare id. This is where the
// two are told apart.
static constexpr u64 kGnmIdentMax = 0x10000;
static std::atomic<bool> g_vblankStarted{false};

// Flips the title has actually submitted. The display event's data>>16 carries
// this (not the vblank tick) so render-frame pacing tracks real flips.
static std::atomic<u64> g_flipCount{0};
u64 flipCount() { return g_flipCount.load(); }

// A low-bit TSC nonce for the display event's bits 0..11, so a polling title
// sees each event as new. On the native x86 backend that's the real rdtsc; the
// aarch64/FEX host has no rdtsc intrinsic, so fall back to a monotonic wall
// clock -- only the low 12 bits are used. Mirrors dce_dev.cpp::guestTsc.
static u64 tscNonce() {
#if defined(DELTA_BACKEND_NATIVE)
  return __builtin_ia32_rdtsc();
#else
  using namespace std::chrono;
  return static_cast<u64>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
#endif
}

// Guest fds with a live EVFILT_READ knote, and the one thread that selects on
// their host sockets. A read knote's source is outside the guest, so nothing in
// here would ever mark it active; without this a title that waits on socket
// readability (Minecraft's rtc::PhysicalSocketServer) never wakes.
static std::mutex g_watchM;
static std::set<u32> g_watched;
static std::atomic<bool> g_watchStarted{false};

static void watchSocket(u32 fd) {
  if (!fdToSocket(fd))
    return;
  {
    std::lock_guard<std::mutex> lk(g_watchM);
    if (!g_watched.insert(fd).second)
      return;
  }
  bool expected = false;
  if (!g_watchStarted.compare_exchange_strong(expected, true))
    return;
  BASE_LOGI("kevent", "socket read-poll started");
  std::thread([] {
    for (;;) {
      fd_set rd;
      FD_ZERO(&rd);
      int maxFd = -1;
      std::vector<std::pair<u32, int>> live;
      {
        std::lock_guard<std::mutex> lk(g_watchM);
        for (u32 g : g_watched)
          if (auto *s = fdToSocket(g)) {
            live.emplace_back(g, s->hostFd());
            FD_SET(s->hostFd(), &rd);
            maxFd = std::max(maxFd, s->hostFd());
          }
      }
      timeval tv{0, 20000};  // 20 ms; also the retry tick when nothing is live
      if (maxFd < 0 || ::select(maxFd + 1, &rd, nullptr, nullptr, &tv) <= 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      for (auto &[guestFd, hostFd] : live)
        if (FD_ISSET(hostFd, &rd))
          triggerAllEqueues(guestFd, kEVFILT_READ, 1);
    }
  }).detach();
}

// Start the 60 Hz EVFILT_DISPLAY pump once, on the first vblank registration, so
// the timer thread only runs when something waits on it.
static void startVblankPump() {
  bool expected = false;
  if (!g_vblankStarted.compare_exchange_strong(expected, true))
    return;
  BASE_LOGI("vblank", "pump started (60 Hz, EVFILT_DISPLAY/VIDEOOUT)");
  std::thread([] {
    u64 count = 0;
    for (;;) {
      std::this_thread::sleep_for(std::chrono::microseconds(16667));  // ~60 Hz
      ++count;
      // Event data layout (read as data>>16 for the counter, bits 12..15 a 1..14
      // per-event sequence the title polls to detect a NEW event, bits 0..11 a
      // TSC nonce). Packing only count<<16 left bits 12..15 = 0, so the title
      // woke every tick but saw "no new event".
      u64 seq = (count - 1) % 14 + 1;                    // 1..14
      u64 tsc = tscNonce() & 0xFFF;
      // Vblank (-14): a free-running tick for vblank waiters / frame timing.
      i64 vdata = static_cast<i64>((count << 16) | (seq << 12) | tsc);
      triggerAllEqueues(-1, kEVFILT_VIDEOOUT, vdata);
      // Flip (-13): the engine's flip handler reads data>>16 as the index of the
      // last flipped frame and asserts unless the sim has produced it. Never post
      // it before the first real flip (during loading the last produced frame is
      // -1, so any flip event is "out of range"); once flipping it rides the real
      // flip count and noteFlip already posts each flip immediately.
      u64 flips = g_flipCount.load();
      if (flips > 0) {
        u64 idx = flips - 1;
        i64 fdata =
            static_cast<i64>((idx << 16) | ((idx % 14 + 1) << 12) | tsc);
        triggerAllEqueues(-1, kEVFILT_DISPLAY, fdata);
      }
    }
  }).detach();
}

// A GPU end-of-pipe interrupt. RELEASE_MEM/EVENT_WRITE_EOP carry an INT_SEL
// field asking the CP to raise one once the label write lands, and
// libSceGnmDriver turns that interrupt into an event on whatever equeue
// sceGnmAddEqEvent registered -- filter -14, under the small ident the caller
// chose (GTA:SA asks for 0x5 and 0x40). We used to write the label and drop the
// interrupt, so a title driving its fences off the event never saw one.
//
// Deliberately NOT the 60 Hz pump's business: that tick exists for the videoout
// vblank waiters that share this filter, and firing a Gnm event on it says "the
// GPU finished" at moments when it did not. Small idents are the Gnm
// registrations (videoout puts its event id in the high bits of ident), so the
// pump leaves those alone and they arrive from here instead.
// End-of-pipe interrupts since boot. A Gnm knote compares against this so an
// interrupt raised between two arming windows is not lost.
static std::atomic<u64> g_eopSeq{0};
u64 gpuEndOfPipeCount() { return g_eopSeq.load(std::memory_order_relaxed); }

void noteGpuEndOfPipe() {
  const u64 n = g_eopSeq.fetch_add(1) + 1;
  if (kEventTrace && (n == 1 || (n % 2000) == 0))
    BASE_LOGI("gpueop", "end-of-pipe interrupt #{}", (unsigned long long)n);
  // Same packing as the display events: the counter above bit 16, a 1..14
  // sequence so a poller can tell a new event from a repeat, a TSC nonce below.
  const i64 data =
      static_cast<i64>((n << 16) | (((n - 1) % 14 + 1) << 12) | (tscNonce() & 0xFFF));
  std::lock_guard<std::mutex> lk(g_eqRegM);
  for (auto *eq : g_equeues)
    eq->triggerGnm(data);
}

void noteFlip() {
  u64 idx = g_flipCount.fetch_add(1);  // index of the flip that just completed
  // Post the flip (-13) event immediately so a thread blocked waiting for this
  // flip wakes now instead of on the next 60 Hz pump tick. data>>16 is the index
  // of the LAST completed flip (not the count): the engine's flip handler then
  // processes frames up to and including that index, which the sim has produced.
  u64 seq = idx % 14 + 1;
  u64 tsc = tscNonce() & 0xFFF;
  i64 data = static_cast<i64>((idx << 16) | (seq << 12) | tsc);
  triggerAllEqueues(-1, kEVFILT_DISPLAY, data);
}

equeue::equeue(proc *p, const char *nm) : kObject(p, oType::equeue) {
  if (nm)
    name = nm;
  std::lock_guard<std::mutex> lk(g_eqRegM);
  g_equeues.push_back(this);
}

equeue::~equeue() {
  std::lock_guard<std::mutex> lk(g_eqRegM);
  for (size_t i = 0; i < g_equeues.size(); i++) {
    if (g_equeues[i] == this) {
      g_equeues.erase(g_equeues.begin() + i);
      break;
    }
  }
}

equeue::knote *equeue::find(u64 ident, i16 filter) {
  for (auto &k : notes)
    if (k.ev.ident == ident && k.ev.filter == filter)
      return &k;
  return nullptr;
}

int equeue::kevent(const kevent_t *changes, int nchanges, kevent_t *out,
                   int nout, const ktimespec *to) {
  std::unique_lock<std::mutex> lk(m);

  // 1) apply the changelist.
  for (int i = 0; i < nchanges; i++) {
    const auto &c = changes[i];
    BASE_LOGI("kevent",
              "change ident={:#x} filter={} flags={:#x} fflags={:#x} "
              "data={:#x} udata={:p}",
              (unsigned long long)c.ident, c.filter, c.flags, c.fflags,
              (unsigned long long)c.data, c.udata);
    if (c.flags & kEV_DELETE) {
      for (size_t j = 0; j < notes.size(); j++)
        if (notes[j].ev.ident == c.ident && notes[j].ev.filter == c.filter) {
          notes.erase(notes.begin() + j);
          break;
        }
      continue;
    }
    // EVFILT_USER + NOTE_TRIGGER is one thread poking another awake, not a
    // registration: it must fire the existing knote, not replace and clear it.
    // This is how WebRTC's SocketServer::WakeUp reaches a thread parked in
    // kevent, so dropping it leaves every rtc::Thread::BlockingCall hung.
    if (c.filter == kEVFILT_USER && (c.fflags & kNOTE_TRIGGER)) {
      if (auto *k = find(c.ident, c.filter)) {
        k->active = true;
        k->ev.data = c.data;
        // The trigger carries the udata, not the registration: sceKernelAddUserEvent
        // registers with none and sceKernelTriggerUserEvent supplies it per poke.
        // Keeping the registration's null made sceKernelGetEventUserData return
        // null, and Minecraft's handler dereferences it straight into a vcall.
        if (c.udata)
          k->ev.udata = c.udata;
        cv.notify_all();
      }
      continue;
    }
    // EV_ADD (and plain enable): register/replace.
    if (auto *k = find(c.ident, c.filter)) {
      k->ev = c;
      k->active = false;
    } else {
      notes.push_back({c, false});
    }
    // First vblank registration kicks off the synthetic 60 Hz pump.
    if (c.filter == kEVFILT_DISPLAY || c.filter == kEVFILT_VIDEOOUT)
      startVblankPump();
    // A read knote on a socket is the only knote whose source lives outside the
    // guest, so nothing here can set it active. Watch the host fd instead.
    if (c.filter == kEVFILT_READ)
      watchSocket(static_cast<u32>(c.ident));
  }

  // 2) collect ready events, waiting if asked.
  auto collect = [&]() -> int {
    int n = 0;
    for (auto &k : notes) {
      if (n >= nout)
        break;
      if (!k.active)
        continue;
      out[n++] = k.ev;
      // edge semantics: a fired knote is consumed until its source fires again.
      k.active = false;
      k.eop_seen = gpuEndOfPipeCount();
    }
    return n;
  };

  if (nout <= 0)
    return 0;

  // Re-arm any Gnm knote the GPU has run past since this queue last looked.
  // The interrupt is an edge on real hardware, but our submits finish inside
  // the submit call, so an edge raised while the title was between kevents
  // would otherwise be lost with nothing left to raise another one.
  {
    const u64 eop = gpuEndOfPipeCount();
    for (auto &k : notes) {
      if (k.active || k.ev.filter != kEVFILT_VIDEOOUT ||
          k.ev.ident >= kGnmIdentMax || k.eop_seen >= eop)
        continue;
      k.active = true;
      k.ev.data = static_cast<i64>((eop << 16) | k.ev.ident);
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
    for (auto &k : notes)
      if (k.active)
        return true;
    return false;
  };

  // An untimed wait on a queue with no knotes can never return: nothing has a
  // source that could set one active. Report it once per queue -- it is always a
  // missing registration on our side, and the symptom (a wedged render thread)
  // otherwise looks like the title hanging on its own.
  if (!to && notes.empty() && !warnedEmptyWait) {
    warnedEmptyWait = true;
    BASE_LOGI("kevent", "tid={} waits forever on '{}' (fd={}): no knotes",
              (long)gettid(), name.empty() ? "(unnamed)" : name.c_str(),
              handle());
  }
  if (!to) {
    // Report once what an untimed wait is registered for if it goes long: a
    // queue with knotes that never go active is a source we do not drive.
    if (kEventStallSecs > 0) {
      const auto until =
          std::chrono::steady_clock::now() + std::chrono::seconds(kEventStallSecs);
      if (!cv.wait_until(lk, until, pred)) {
        reportRegistrationsLocked(handle());
        cv.wait(lk, pred);
      }
    } else {
      cv.wait(lk, pred);
    }
    ready = true;
  } else {
    auto dur = std::chrono::seconds(to->tv_sec) +
               std::chrono::nanoseconds(to->tv_nsec);
    ready = cv.wait_for(lk, dur, pred);
  }
  if (!ready) {
    if (kEventTrace)
      BASE_LOGI("kevent", "timeout nout={}", nout);
    const auto now = std::chrono::steady_clock::now();
    if (idleSince == std::chrono::steady_clock::time_point{})
      idleSince = now;
    if (kEventStallSecs > 0 && !reportedIdle &&
        now - idleSince >= std::chrono::seconds(kEventStallSecs)) {
      reportedIdle = true;
      reportRegistrationsLocked(handle());
    }
    return 0;
  }
  idleSince = {};
  reportedIdle = false;
  got = collect();
  if (kEventTrace && got > 0)
    BASE_LOGI("kevent", "return wait n={} filter={} ident={:#x} data={:#x}",
              got, out[0].filter, (unsigned long long)out[0].ident,
              (unsigned long long)out[0].data);
  return got;
}

void equeue::reportRegistrations(int fd) {
  std::lock_guard<std::mutex> lk(m);
  reportRegistrationsLocked(fd);
}

// Caller holds m.
void equeue::reportRegistrationsLocked(int fd) {
  base::String line;
  base::FormatTo(line, "kq={} ({}) long wait, registered:", fd, name.c_str());
  for (const auto &k : notes)
    base::FormatTo(line, " [ident={:#x} filter={} active={}]",
                   (unsigned long long)k.ev.ident, (int)k.ev.filter,
                   (int)k.active);
  BASE_LOGI("kevent", "{}", line.c_str());
}

void equeue::addEvent(u64 ident, i16 filter, void *udata) {
  std::lock_guard<std::mutex> lk(m);
  kevent_t ev{};
  ev.ident = ident;
  ev.filter = filter;
  ev.flags = kEV_CLEAR;
  ev.udata = udata;
  if (auto *k = find(ident, filter)) {
    k->ev = ev;
    k->active = false;
  } else {
    notes.push_back({ev, false});
  }
  if (filter == kEVFILT_DISPLAY || filter == kEVFILT_VIDEOOUT)
    startVblankPump();
}

bool equeue::removeEvent(u64 ident, i16 filter) {
  std::lock_guard<std::mutex> lk(m);
  for (size_t j = 0; j < notes.size(); j++)
    if (notes[j].ev.ident == ident && notes[j].ev.filter == filter) {
      notes.erase(notes.begin() + j);
      return true;
    }
  return false;
}

// The Gnm half of filter -14: every knote registered under a small ident, each
// one told its own event id. sceGnmGetEqEventType reads the delivered data
// whole and sceGnmGetEqTimeStamp reads data >> 16, so the id has to survive in
// the low bits or the title cannot tell which of its events arrived.
void equeue::triggerGnm(i64 data) {
  std::lock_guard<std::mutex> lk(m);
  bool any = false;
  for (auto &k : notes) {
    if (k.ev.filter != kEVFILT_VIDEOOUT || k.ev.ident >= kGnmIdentMax)
      continue;
    k.active = true;
    k.ev.data = (data & ~static_cast<i64>(0xFFF)) | static_cast<i64>(k.ev.ident);
    any = true;
  }
  if (any)
    cv.notify_all();
}

void equeue::trigger(i64 ident, i16 filter, i64 data) {
  std::lock_guard<std::mutex> lk(m);
  bool any = false;
  for (auto &k : notes) {
    // filter==0 is a wildcard (no real EVFILT is 0); ident<0 matches any.
    if (filter != 0 && k.ev.filter != filter)
      continue;
    if (ident >= 0 && k.ev.ident != static_cast<u64>(ident))
      continue;
    // See triggerGnm: a Gnm graphics-core registration is not a vblank waiter
    // and must not be told the GPU finished on a timer.
    if (filter == kEVFILT_VIDEOOUT && k.ev.ident < kGnmIdentMax)
      continue;
    k.active = true;
    k.ev.data = data;
    any = true;
  }
  if (any)
    cv.notify_all();
}

void triggerAllEqueues(i64 ident, i16 filter, i64 data) {
  std::lock_guard<std::mutex> lk(g_eqRegM);
  for (auto *eq : g_equeues)
    eq->trigger(ident, filter, data);
}

int PS4ABI sys_kqueue() {
  auto *eq = new equeue(proc::getActive(), nullptr);
  BASE_LOGI("kqueue", "-> fd={}", eq->handle());
  return eq->handle();
}

int PS4ABI sys_kqueueex(const char *name, int flags) {
  auto *eq = new equeue(proc::getActive(), name);
  BASE_LOGI("kqueueex", "name={} flags={:#x} -> fd={}",
            name ? name : "(null)", flags, eq->handle());
  return eq->handle();
}

int PS4ABI sys_kevent(int kq, const kevent_t *changelist, int nchanges,
                      kevent_t *eventlist, int nevents, const ktimespec *to) {
  auto *obj = proc::getActive()->getObjTable().get(kq);
  if (!obj || obj->type() != kObject::oType::equeue) {
    BASE_LOGI("kevent", "bad kq fd={}", kq);
    return -SysError::eBADF;
  }
  // A thread blocked here is waiting for an event that may never be posted --
  // the same "wedged title" question the umtx/semaphore probes answer, and it
  // was the one wait they could not see.
  WaitProbe _wp("kevent", (long)kq, (long)nevents);
  auto *eq = static_cast<equeue *>(obj);
  const auto t0 = std::chrono::steady_clock::now();
  int r = eq->kevent(changelist, nchanges, eventlist, nevents, to);
  // A wait this long is a title that is not going to wake up on its own. What
  // it registered says which event never arrived, and that is the only thing
  // the queue can tell us from the outside.
  if (kEventStallSecs > 0 &&
      std::chrono::steady_clock::now() - t0 >=
          std::chrono::seconds(kEventStallSecs))
    eq->reportRegistrations(kq);
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
