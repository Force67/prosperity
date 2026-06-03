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
#include <cstdio>
#include <thread>

#include "kern/proc.h"
#include "sys_event.h"

namespace krnl {
// All live equeues, so the vblank pump can fan flip events to every one of them
// without knowing which equeue a given flip event was registered on.
static std::mutex g_eqRegM;
static base::Vector<equeue *> g_equeues;

// EVFILT_DISPLAY (-13) is the videoout vblank/flip filter. A thread that waits
// on it blocks in kevent until a vblank arrives. With no real display hardware,
// a synthetic 60 Hz tick keeps that wait from blocking forever.
static constexpr int16_t kEVFILT_DISPLAY = -13;
static std::atomic<bool> g_vblankStarted{false};

// Start the 60 Hz EVFILT_DISPLAY pump once, on the first vblank registration, so
// the timer thread only runs when something waits on it.
static void startVblankPump() {
  bool expected = false;
  if (!g_vblankStarted.compare_exchange_strong(expected, true))
    return;
  std::printf("[vblank] pump started (60 Hz, EVFILT_DISPLAY)\n");
  std::thread([] {
    uint64_t count = 0;
    for (;;) {
      std::this_thread::sleep_for(std::chrono::microseconds(16667));  // ~60 Hz
      triggerAllEqueues(-1, kEVFILT_DISPLAY, static_cast<int64_t>(++count));
    }
  }).detach();
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
    // EV_ADD (and plain enable): register/replace.
    if (auto *k = find(c.ident, c.filter)) {
      k->ev = c;
      k->active = false;
    } else {
      notes.push_back({c, false});
    }
    // First vblank registration kicks off the synthetic 60 Hz pump.
    if (c.filter == kEVFILT_DISPLAY)
      startVblankPump();
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
  if (got > 0)
    return got;

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
  if (!ready)
    return 0;
  return collect();
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
  if (filter == kEVFILT_DISPLAY)
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
  return static_cast<equeue *>(obj)->kevent(changelist, nchanges, eventlist,
                                            nevents, to);
}
}  // namespace krnl
