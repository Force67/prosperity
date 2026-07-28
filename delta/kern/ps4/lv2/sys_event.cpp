/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
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

#include "kern/proc.h"
#include "kern/ps4/dev/socket_dev.h"
#include "sys_event.h"

namespace krnl {
// All live equeues, so the vblank pump can fan flip events to every one of them
// without knowing which equeue a given flip event was registered on.
static std::mutex g_eqRegM;
static base::Vector<equeue *> g_equeues;

// EVFILT_DISPLAY (-13) and Sony's videoout event filter (-14) are both used by
// real system modules for vblank/flip waits. A thread that waits on either
// blocks in kevent until a display event arrives. With no real display hardware,
// a synthetic 60 Hz tick keeps those waits from blocking forever.
static constexpr int16_t kEVFILT_READ = -1;
static constexpr int16_t kEVFILT_USER = -11;
static constexpr uint32_t kNOTE_TRIGGER = 0x01000000;
static void watchSocket(uint32_t fd);
static constexpr int16_t kEVFILT_DISPLAY = -13;
static constexpr int16_t kEVFILT_VIDEOOUT = -14;
static std::atomic<bool> g_vblankStarted{false};

// Flips the title has actually submitted. The display event's data>>16 carries
// this (not the vblank tick) so render-frame pacing tracks real flips.
static std::atomic<uint64_t> g_flipCount{0};
uint64_t flipCount() { return g_flipCount.load(); }

// A low-bit TSC nonce for the display event's bits 0..11, so a polling title
// sees each event as new. On the native x86 backend that's the real rdtsc; the
// aarch64/FEX host has no rdtsc intrinsic, so fall back to a monotonic wall
// clock -- only the low 12 bits are used. Mirrors dce_dev.cpp::guestTsc.
static uint64_t tscNonce() {
#if defined(DELTA_BACKEND_NATIVE)
  return __builtin_ia32_rdtsc();
#else
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
#endif
}

// Guest fds with a live EVFILT_READ knote, and the one thread that selects on
// their host sockets. A read knote's source is outside the guest, so nothing in
// here would ever mark it active; without this a title that waits on socket
// readability (Minecraft's rtc::PhysicalSocketServer) never wakes.
static std::mutex g_watchM;
static std::set<uint32_t> g_watched;
static std::atomic<bool> g_watchStarted{false};

static void watchSocket(uint32_t fd) {
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
  std::printf("[kevent] socket read-poll started\n");
  std::thread([] {
    for (;;) {
      fd_set rd;
      FD_ZERO(&rd);
      int maxFd = -1;
      std::vector<std::pair<uint32_t, int>> live;
      {
        std::lock_guard<std::mutex> lk(g_watchM);
        for (uint32_t g : g_watched)
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
  std::printf("[vblank] pump started (60 Hz, EVFILT_DISPLAY/VIDEOOUT)\n");
  std::thread([] {
    uint64_t count = 0;
    for (;;) {
      std::this_thread::sleep_for(std::chrono::microseconds(16667));  // ~60 Hz
      ++count;
      // Event data layout (read as data>>16 for the counter, bits 12..15 a 1..14
      // per-event sequence the title polls to detect a NEW event, bits 0..11 a
      // TSC nonce). Packing only count<<16 left bits 12..15 = 0, so the title
      // woke every tick but saw "no new event".
      uint64_t seq = (count - 1) % 14 + 1;                    // 1..14
      uint64_t tsc = tscNonce() & 0xFFF;
      // Vblank (-14): a free-running tick for vblank waiters / frame timing.
      int64_t vdata = static_cast<int64_t>((count << 16) | (seq << 12) | tsc);
      triggerAllEqueues(-1, kEVFILT_VIDEOOUT, vdata);
      // Flip (-13): the engine's flip handler reads data>>16 as the index of the
      // last flipped frame and asserts unless the sim has produced it. Never post
      // it before the first real flip (during loading the last produced frame is
      // -1, so any flip event is "out of range"); once flipping it rides the real
      // flip count and noteFlip already posts each flip immediately.
      uint64_t flips = g_flipCount.load();
      if (flips > 0) {
        uint64_t idx = flips - 1;
        int64_t fdata =
            static_cast<int64_t>((idx << 16) | ((idx % 14 + 1) << 12) | tsc);
        triggerAllEqueues(-1, kEVFILT_DISPLAY, fdata);
      }
    }
  }).detach();
}

void noteFlip() {
  uint64_t idx = g_flipCount.fetch_add(1);  // index of the flip that just completed
  // Post the flip (-13) event immediately so a thread blocked waiting for this
  // flip wakes now instead of on the next 60 Hz pump tick. data>>16 is the index
  // of the LAST completed flip (not the count): the engine's flip handler then
  // processes frames up to and including that index, which the sim has produced.
  uint64_t seq = idx % 14 + 1;
  uint64_t tsc = tscNonce() & 0xFFF;
  int64_t data = static_cast<int64_t>((idx << 16) | (seq << 12) | tsc);
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

equeue::knote *equeue::find(uint64_t ident, int16_t filter) {
  for (auto &k : notes)
    if (k.ev.ident == ident && k.ev.filter == filter)
      return &k;
  return nullptr;
}

int equeue::kevent(const kevent_t *changes, int nchanges, kevent_t *out,
                   int nout, const ktimespec *to) {
  static const bool traceReturns = std::getenv("DELTA_EVENT_TRACE") != nullptr;
  std::unique_lock<std::mutex> lk(m);

  // 1) apply the changelist.
  for (int i = 0; i < nchanges; i++) {
    const auto &c = changes[i];
    std::printf("[kevent] change ident=%#llx filter=%d flags=%#x fflags=%#x "
                "data=%#llx udata=%p\n",
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
      watchSocket(static_cast<uint32_t>(c.ident));
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
    }
    return n;
  };

  if (nout <= 0)
    return 0;

  int got = collect();
  if (got > 0) {
    if (traceReturns)
      std::printf("[kevent] return immediate n=%d filter=%d ident=%#llx data=%#llx\n",
                  got, out[0].filter, (unsigned long long)out[0].ident,
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

  if (!to) {
    cv.wait(lk, pred);
    ready = true;
  } else {
    auto dur = std::chrono::seconds(to->tv_sec) +
               std::chrono::nanoseconds(to->tv_nsec);
    ready = cv.wait_for(lk, dur, pred);
  }
  if (!ready) {
    if (traceReturns)
      std::printf("[kevent] timeout nout=%d\n", nout);
    return 0;
  }
  got = collect();
  if (traceReturns && got > 0)
    std::printf("[kevent] return wait n=%d filter=%d ident=%#llx data=%#llx\n",
                got, out[0].filter, (unsigned long long)out[0].ident,
                (unsigned long long)out[0].data);
  return got;
}

void equeue::addEvent(uint64_t ident, int16_t filter, void *udata) {
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

bool equeue::removeEvent(uint64_t ident, int16_t filter) {
  std::lock_guard<std::mutex> lk(m);
  for (size_t j = 0; j < notes.size(); j++)
    if (notes[j].ev.ident == ident && notes[j].ev.filter == filter) {
      notes.erase(notes.begin() + j);
      return true;
    }
  return false;
}

void equeue::trigger(int64_t ident, int16_t filter, int64_t data) {
  std::lock_guard<std::mutex> lk(m);
  bool any = false;
  for (auto &k : notes) {
    // filter==0 is a wildcard (no real EVFILT is 0); ident<0 matches any.
    if (filter != 0 && k.ev.filter != filter)
      continue;
    if (ident >= 0 && k.ev.ident != static_cast<uint64_t>(ident))
      continue;
    k.active = true;
    k.ev.data = data;
    any = true;
  }
  if (any)
    cv.notify_all();
}

void triggerAllEqueues(int64_t ident, int16_t filter, int64_t data) {
  std::lock_guard<std::mutex> lk(g_eqRegM);
  for (auto *eq : g_equeues)
    eq->trigger(ident, filter, data);
}

int PS4ABI sys_kqueue() {
  auto *eq = new equeue(proc::getActive(), nullptr);
  std::printf("[kqueue] -> fd=%u\n", eq->handle());
  return eq->handle();
}

int PS4ABI sys_kqueueex(const char *name, int flags) {
  auto *eq = new equeue(proc::getActive(), name);
  std::printf("[kqueueex] name=%s flags=%#x -> fd=%u\n", name ? name : "(null)",
              flags, eq->handle());
  return eq->handle();
}

int PS4ABI sys_kevent(int kq, const kevent_t *changelist, int nchanges,
                      kevent_t *eventlist, int nevents, const ktimespec *to) {
  auto *obj = proc::getActive()->getObjTable().get(kq);
  if (!obj || obj->type() != kObject::oType::equeue) {
    std::printf("[kevent] bad kq fd=%d\n", kq);
    return -SysError::eBADF;
  }
  int r = static_cast<equeue *>(obj)->kevent(changelist, nchanges, eventlist,
                                             nevents, to);
  if (std::getenv("DELTA_KEVENT_TRACE")) {
    std::printf("[kevent] kq=%d nchanges=%d -> %d", kq, nchanges, r);
    for (int i = 0; i < nchanges && changelist && i < 4; i++)
      std::printf(" chg[ident=%#llx filter=%d flags=%#x]",
                  (unsigned long long)changelist[i].ident,
                  (int)changelist[i].filter, (unsigned)changelist[i].flags);
    std::printf("\n");
  }
  return r;
}
}  // namespace krnl
