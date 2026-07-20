/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <unordered_map>

#include "kern/proc.h"
#include "sys_event_flag.h"

namespace krnl {
// SCE_KERNEL_EVF_WAITMODE_*
enum { kEvfAnd = 0x01, kEvfOr = 0x02, kEvfClearAll = 0x10, kEvfClearPat = 0x20 };

// Named event flags, so evf_open(name) finds the one evf_create(name) made.
static std::mutex g_efRegM;
static std::unordered_map<std::string, eventFlag *> g_efByName;

eventFlag::eventFlag(proc *p, const char *nm, uint64_t init, uint64_t sticky_)
    : kObject(p, oType::eventflag), bits(init), sticky(sticky_) {
  if (nm && *nm) {
    name = nm;
    std::lock_guard<std::mutex> lk(g_efRegM);
    g_efByName[nm] = this;
  }
}

bool eventFlag::satisfied(uint64_t pattern, uint32_t mode) const {
  return (mode & kEvfOr) ? (bits & pattern) != 0 : (bits & pattern) == pattern;
}

int eventFlag::take(uint64_t pattern, uint32_t mode, uint64_t *result) {
  if (result)
    *result = bits;
  if (mode & kEvfClearAll)
    bits = 0;
  else if (mode & kEvfClearPat)
    bits &= ~pattern;
  bits |= sticky;  // system focus/ready flags stay asserted (no ShellCore here)
  return 0;
}

int eventFlag::wait(uint64_t pattern, uint32_t mode, uint64_t *result,
                    uint32_t *timeoutUs) {
  std::unique_lock<std::mutex> lk(m);
  if (satisfied(pattern, mode))
    return take(pattern, mode, result);
  if (timeoutUs) {
    if (!cv.wait_for(lk, std::chrono::microseconds(*timeoutUs),
                     [&] { return satisfied(pattern, mode); }))
      return -SysError::eTIMEDOUT;
  } else {
    cv.wait(lk, [&] { return satisfied(pattern, mode); });
  }
  return take(pattern, mode, result);
}

int eventFlag::trywait(uint64_t pattern, uint32_t mode, uint64_t *result) {
  std::unique_lock<std::mutex> lk(m);
  if (!satisfied(pattern, mode))
    return -SysError::eBUSY;
  return take(pattern, mode, result);
}

void eventFlag::set(uint64_t b) {
  std::lock_guard<std::mutex> lk(m);
  bits |= b;
  lastSetTid.store((long)gettid(), std::memory_order_relaxed);
  cv.notify_all();
}

void eventFlag::clear(uint64_t b) {
  std::lock_guard<std::mutex> lk(m);
  bits &= b;  // SCE clear keeps the bits set in b
}

static eventFlag *fromId(int id) {
  auto *obj = proc::getActive()->getObjTable().get(id);
  if (!obj || obj->type() != kObject::oType::eventflag)
    return nullptr;
  return static_cast<eventFlag *>(obj);
}

// DELTA_EVF_TRACE[=substr]: log every evf op (optionally only for flags whose
// name contains substr) with tid + bits, to reconstruct producer/consumer
// interleavings (e.g. SOTTR's file-I/O channel handshake).
static bool evfTraceOn(const eventFlag *ef) {
  static const char *filt = std::getenv("DELTA_EVF_TRACE");
  if (!filt)
    return false;
  if (!*filt || std::strcmp(filt, "1") == 0)
    return true;
  return ef && std::strstr(ef->fname().c_str(), filt) != nullptr;
}

static void evfTrace(const char *op, int id, const eventFlag *ef,
                     uint64_t pattern, uint32_t mode, int ret, uint64_t res) {
  if (!evfTraceOn(ef))
    return;
  std::fprintf(stderr, "[evf] t=%ld %s id=%d '%s' pat=%#llx mode=%#x -> ret=%d res=%#llx\n",
               (long)gettid(), op, id, ef ? ef->fname().c_str() : "?",
               (unsigned long long)pattern, mode, ret,
               (unsigned long long)res);
}

int PS4ABI sys_evf_create(const char *name, uint32_t attr,
                          uint64_t initPattern) {
  auto *ef = new eventFlag(proc::getActive(), name, initPattern);
  std::printf("[evf] create '%s' attr=%#x init=%#llx -> id=%u\n",
              name ? name : "", attr, (unsigned long long)initPattern,
              ef->handle());
  return ef->handle();
}

// Some system-service event flags gate the game on state the ShellCore would
// publish (app focus granted, power normal, system running). With no ShellCore
// the flag stays 0 and the game's "wait for focus/ready" (EVF OR-wait for any
// bit) blocks forever. Seed those flags as "focused/ready" so the game proceeds.
static uint64_t systemFlagInit(const char *name) {
  if (!name)
    return 0;
  base::StringRef n(name);
  if (n.find("AppFocus", 0, 8) != base::StringRef::npos ||
      n.find("CtrlFocus", 0, 9) != base::StringRef::npos ||
      n.find("PowerControl", 0, 12) != base::StringRef::npos ||
      n.find("SystemStateMgr", 0, 14) != base::StringRef::npos)
    return 0x1;  // bit0 = focused / powered / running
  // ShellCore publishes boot progress here bit-by-bit (SotC waits for 0x400);
  // with no ShellCore, report every boot stage as already complete.
  if (n.find("BootStatus", 0, 10) != base::StringRef::npos)
    return ~0ull;
  return 0;
}

int PS4ABI sys_evf_open(const char *name) {
  {
    std::lock_guard<std::mutex> lk(g_efRegM);
    auto it = name ? g_efByName.find(name) : g_efByName.end();
    if (it != g_efByName.end())
      return it->second->handle();
  }
  // Auto-create unknown named flags: on real hw a system service creates them;
  // here both producer and consumer just open by name, so creating on first
  // open gives them a shared flag and the sync actually works.
  uint64_t seed = systemFlagInit(name);
  auto *ef = new eventFlag(proc::getActive(), name, seed, seed);
  std::printf("[evf] open '%s' (auto-created) -> id=%u\n", name ? name : "",
              ef->handle());
  return ef->handle();
}

int PS4ABI sys_evf_delete(int id) {
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  {
    std::lock_guard<std::mutex> lk(g_efRegM);
    if (!ef->fname().empty())
      g_efByName.erase(ef->fname().c_str());
  }
  proc::getActive()->getObjTable().release(id);
  return 0;
}

int PS4ABI sys_evf_close(int id) { return sys_evf_delete(id); }

int PS4ABI sys_evf_wait(int id, uint64_t pattern, uint32_t mode,
                        uint64_t *result, uint32_t *timeoutUs) {
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  uint64_t res = 0;
  int r = ef->wait(pattern, mode, &res, timeoutUs);
  if (result)
    *result = res;
  evfTrace("wait", id, ef, pattern, mode, r, res);
  return r;
}

int PS4ABI sys_evf_trywait(int id, uint64_t pattern, uint32_t mode,
                           uint64_t *result) {
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  uint64_t res = 0;
  int r = ef->trywait(pattern, mode, &res);
  // Handshake grace: when the polling thread itself posted the last set() on
  // this flag, it is the requester of a request/response channel (it set the
  // "request" bit and now polls for the responder's "done" bit). On the real
  // console the responder runs at higher SCE priority, so the set() preempts
  // the requester and the response is already posted by the time it polls;
  // engines rely on that ordering (Shadow of the Tomb Raider's file-I/O
  // channel streams garbage progress if the poll misses). Emulate it with a
  // bounded wait for the response. Pure status pollers never set the flag
  // themselves, so they keep true poll semantics and pay nothing.
  // DELTA_NO_EVF_GRACE disables for A/B testing.
  static const bool noGrace = std::getenv("DELTA_NO_EVF_GRACE") != nullptr;
  if (r == -SysError::eBUSY && !noGrace &&
      ef->lastSetTid.load(std::memory_order_relaxed) == (long)gettid()) {
    uint32_t toUs = 250000;
    r = ef->wait(pattern, mode, &res, &toUs);
    if (r == -SysError::eTIMEDOUT)
      r = -SysError::eBUSY;
  }
  if (result)
    *result = res;
  evfTrace("poll", id, ef, pattern, mode, r, res);
  return r;
}

int PS4ABI sys_evf_set(int id, uint64_t bits) {
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  ef->set(bits);
  evfTrace("set", id, ef, bits, 0, 0, 0);
  return 0;
}

int PS4ABI sys_evf_clear(int id, uint64_t bits) {
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  ef->clear(bits);
  return 0;
}

int PS4ABI sys_evf_cancel(int id, uint64_t pattern, int *numWaiters) {
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  ef->set(pattern);  // wake waiters with the pattern (best-effort)
  if (numWaiters)
    *numWaiters = 0;
  return 0;
}
}  // namespace krnl
