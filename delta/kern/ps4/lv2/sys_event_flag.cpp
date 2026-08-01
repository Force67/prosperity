/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>

#include <mutex>
#include "wait_probe.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <unordered_map>

#include "kern/crash.h"
#include "kern/ipmi/services.h"
#include "kern/proc.h"
#include "sys_event_flag.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(const char *, kEvfTrace, "DELTA_EVF_TRACE", nullptr);
DELTA_OPTION(long, kAudioMixAck, "DELTA_AUDIOMIX_ACK", -1);
DELTA_OPTION(bool, kNoEvfGrace, "DELTA_NO_EVF_GRACE", false);
DELTA_OPTION(bool, kWaitProbe, "DELTA_WAIT_PROBE", false);
DELTA_OPTION(bool, kEvfStack, "DELTA_EVF_STACK", false);
}  // namespace

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

void eventFlag::removeWaiter(Waiter *waiter) {
  for (auto it = waiters.begin(); it != waiters.end(); ++it) {
    if (*it == waiter) {
      waiters.erase(it);
      return;
    }
  }
}

int eventFlag::wait(uint64_t pattern, uint32_t mode, uint64_t *result,
                    uint32_t *timeoutUs) {
  std::unique_lock<std::mutex> lk(m);
  if (satisfied(pattern, mode))
    return take(pattern, mode, result);

  Waiter waiter{pattern, mode};
  waiters.push_back(&waiter);
  if (timeoutUs) {
    if (!cv.wait_for(lk, std::chrono::microseconds(*timeoutUs),
                     [&] { return waiter.done; })) {
      removeWaiter(&waiter);
      return -SysError::eTIMEDOUT;
    }
  } else {
    cv.wait(lk, [&] { return waiter.done; });
  }
  removeWaiter(&waiter);
  if (result)
    *result = waiter.result;
  return 0;
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
  // A kernel event flag commits satisfied queued waits during set(). Keeping
  // that result on the waiter prevents a later clear from revoking the wake
  // before the host thread gets scheduled and reacquires this mutex.
  for (auto *waiter : waiters) {
    if (waiter->done || !satisfied(waiter->pattern, waiter->mode))
      continue;
    take(waiter->pattern, waiter->mode, &waiter->result);
    waiter->done = true;
  }
  cv.notify_all();
}

void eventFlag::clear(uint64_t b) {
  std::lock_guard<std::mutex> lk(m);
  bits &= b;  // SCE clear keeps the bits set in b
}

// Name-keyed set for host-side subsystems that stand in for an absent system
// service (see sys_event_flag.h). The registry lock is held across set() on
// purpose: dropping it first would leave a window in which sys_evf_delete frees
// the flag under us. Nothing takes g_efRegM while holding a flag's own mutex, so
// this nesting cannot deadlock.
bool evfSetByNameSubstr(const char *substr, uint64_t bits) {
  if (!substr)
    return false;
  std::lock_guard<std::mutex> lk(g_efRegM);
  for (auto &kv : g_efByName) {
    if (kv.first.find(substr) == std::string::npos)
      continue;
    kv.second->set(bits);
    return true;
  }
  return false;
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
static bool evfTraceOn(const eventFlag *ef, int id) {
  const char *filt = kEvfTrace;
  if (!filt)
    return false;
  if (!*filt || std::strcmp(filt, "1") == 0)
    return true;
  // "id:<n>[,<n>...]" filters by handle (names like PS4SyncEvent repeat dozens
  // of times; the handle is the only unique identity). A list is what a
  // handshake needs: SotC's world-load coordinator clears one flag, sets three
  // others, then waits on the first, and only seeing all of them together says
  // which side of that exchange never happens. Values may be decimal or 0x hex.
  if (std::strncmp(filt, "id:", 3) == 0) {
    for (const char *p = filt + 3; *p;) {
      const int want = (int)std::strtol(p, const_cast<char **>(&p), 0);
      if (id == want)
        return true;
      while (*p == ',' || *p == ' ')
        p++;
    }
    return false;
  }
  return ef && std::strstr(ef->fname().c_str(), filt) != nullptr;
}

static void evfTrace(const char *op, int id, const eventFlag *ef,
                     uint64_t pattern, uint32_t mode, int ret, uint64_t res) {
  if (!evfTraceOn(ef, id))
    return;
  // us timestamp from the same steady_clock the shm-audio dumper stamps its
  // snapshots with, so an evf signal can be placed against a cursor movement.
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();
  std::fprintf(stderr,
               "[evf] us=%lld tid=%ld %s id=%d '%s' pat=%#llx mode=%#x -> ret=%d res=%#llx\n",
               (long long)us, (long)gettid(), op, id,
               ef ? ef->fname().c_str() : "?",
               (unsigned long long)pattern, mode, ret,
               (unsigned long long)res);
  // DELTA_EVF_STACK: name the guest code on both sides of a handshake. Which
  // function waits or signals is what a trace of ids alone cannot say.
  if (kEvfStack)
    guestStackTrace("evfstk", 6);
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
  // ShellCore keeps the focus flags' pattern equal to the appId of the app
  // that holds focus; libSceSystemService's GetStatus compares it against the
  // title's own appId (from SceLncService GetAppStatus) and any other value
  // reads as backgrounded/overlaid. Same value also satisfies OR-waits.
  if (n.find("AppFocus", 0, 8) != base::StringRef::npos ||
      n.find("CtrlFocus", 0, 9) != base::StringRef::npos)
    return ipmi::kForegroundAppId;
  if (n.find("PowerControl", 0, 12) != base::StringRef::npos ||
      n.find("SystemStateMgr", 0, 14) != base::StringRef::npos)
    return 0x1;  // bit0 = powered / running
  // ShellCore publishes boot progress here bit-by-bit (SotC waits for 0x400);
  // with no ShellCore, report every boot stage as already complete.
  if (n.find("BootStatus", 0, 10) != base::StringRef::npos)
    return ~0ull;
  // The settings service raises bit 32 after publishing /SceAvSetting. We
  // provide that shared-memory block in sys_shm_open, so publish its matching
  // ready state as well; otherwise the real libSceVideoOut blocks during open.
  if (n == "SceAvSettingEvf")
    return 1ull << 32;
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

// RESEARCH INSTRUMENTATION, default OFF.
//   DELTA_AUDIOMIX_ACK=<us>
// The LLE libSceAudioOut's per-port mixer thread submits a block into its
// "/shm_<pid>_<port>_A" region and then waits on bit <port> of the named event
// flag "sceAudioOutMix<pid>" for the console's audio daemon to say "block
// taken". We host no daemon, so that wait never returns and the port produces
// exactly one block, ever. Setting this makes any wait on that flag succeed
// after <us> microseconds WITHOUT consuming anything -- just enough to keep the
// mixer thread cycling so its ring can be observed. 0 = free-run. This is an
// observation aid for reverse-engineering the ring, NOT a daemon.
static long audioMixAckUs() {
  return kAudioMixAck;
}

int PS4ABI sys_evf_wait(int id, uint64_t pattern, uint32_t mode,
                        uint64_t *result, uint32_t *timeoutUs) {
  WaitProbe _wp("evf_wait", (long)id, (long)pattern);
  auto *ef = fromId(id);
  if (!ef)
    return -SysError::eSRCH;
  // Trace the ENTRY too: a wait that never satisfies never reaches the exit
  // trace, which is exactly the wait one is usually hunting.
  evfTrace("waitE", id, ef, pattern, mode, 0, 0);
  if (audioMixAckUs() >= 0 &&
      ef->fname().find("sceAudioOutMix") != std::string::npos) {
    uint32_t to = static_cast<uint32_t>(audioMixAckUs());
    uint64_t ares = 0;
    int ar = ef->wait(pattern, mode, &ares, &to);
    if (ar == -SysError::eTIMEDOUT) {  // nobody signalled: fake the daemon's ack
      ar = 0;
      ares = pattern;
    }
    if (result)
      *result = ares;
    evfTrace("ackwait", id, ef, pattern, mode, ar, ares);
    return ar;
  }
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
  if (r == -SysError::eBUSY && !kNoEvfGrace &&
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

// DELTA_WAIT_PROBE also tallies which flags are ever SET. A flag that threads
// park on but nobody signals is the stall; comparing the two lists names it.
static void evfSetTally(int id) {
  if (!kWaitProbe)
    return;
  static std::mutex m;
  static std::unordered_map<int, uint64_t> hist;
  static auto last = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(m);
  hist[id]++;
  const auto now = std::chrono::steady_clock::now();
  if (now - last < std::chrono::seconds(10))
    return;
  last = now;
  std::fprintf(stderr, "[evfset] ids ever signalled:");
  for (const auto &[k, v] : hist) std::fprintf(stderr, " %d(x%llu)", k,
                                               (unsigned long long)v);
  std::fprintf(stderr, "\n");
}

int PS4ABI sys_evf_set(int id, uint64_t bits) {
  evfSetTally(id);
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
  evfTrace("clear", id, ef, bits, 0, 0, 0);
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
