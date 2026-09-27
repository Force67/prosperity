/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "base/logging.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "guest_abi.h"

#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "kern/lv2/wait_probe.h"

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/strings/string_ref.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/time/time.h"
#include "kern/crash.h"
#include "kern/ipmi/services.h"
#include "kern/lv2/sys_event_flag.h"
#include "kern/process.h"
#include "options/options.h"

namespace {
DELTA_OPTION(const char*, kEvfTrace, "DELTA_EVF_TRACE", nullptr);
DELTA_OPTION(long, kAudioMixAck, "DELTA_AUDIOMIX_ACK", -1);
DELTA_OPTION(bool, kNoEvfGrace, "DELTA_NO_EVF_GRACE", false);
DELTA_OPTION(bool, kWaitProbe, "DELTA_WAIT_PROBE", false);
DELTA_OPTION(bool, kEvfStack, "DELTA_EVF_STACK", false);
}  // namespace

namespace kern {
// Named event flags, so evf_open(name) finds the one evf_create(name) made.
static base::Mutex g_ef_reg_m;
static base::HashMap<base::String, EventFlag*> g_ef_by_name;

EventFlag::EventFlag(ObjectTable& objects, const char* nm, u64 init, u64 sticky)
    : Object(objects, Type::kEventflag), bits_(init), sticky_(sticky) {
  if (nm && *nm) {
    name_ = nm;
    base::LockGuard<base::Mutex> lk(g_ef_reg_m);
    g_ef_by_name[nm] = this;
  }
}

bool EventFlag::Satisfied(u64 pattern, u32 mode) const {
  return (mode & kEvfOr) ? (bits_ & pattern) != 0
                         : (bits_ & pattern) == pattern;
}

int EventFlag::Take(u64 pattern, u32 mode, u64* result) {
  if (result)
    *result = bits_;
  if (mode & kEvfClearAll)
    bits_ = 0;
  else if (mode & kEvfClearPat)
    bits_ &= ~pattern;
  bits_ |=
      sticky_;  // system focus/ready flags stay asserted (no ShellCore here)
  return 0;
}

void EventFlag::RemoveWaiter(Waiter* waiter) {
  for (auto it = waiters_.begin(); it != waiters_.end(); ++it) {
    if (*it == waiter) {
      waiters_.erase(it);
      return;
    }
  }
}

int EventFlag::Wait(u64 pattern, u32 mode, u64* result, u32* timeout_us) {
  base::UniqueLock<base::Mutex> lk(m_);
  if (Satisfied(pattern, mode))
    return Take(pattern, mode, result);

  Waiter waiter{pattern, mode};
  waiters_.push_back(&waiter);
  if (timeout_us) {
    // The timeout is an in/out parameter: the kernel writes back the remaining
    // microseconds after the wait (zero on exhaustion).
    auto start = base::TimeTicks::Now();
    if (!cv_.WaitFor(lk, base::Microseconds(*timeout_us),
                     [&] { return waiter.done; })) {
      RemoveWaiter(&waiter);
      *timeout_us = 0;
      return -SysError::eTIMEDOUT;
    }
    const i64 elapsed = (base::TimeTicks::Now() - start).InMicroseconds();
    *timeout_us =
        elapsed < *timeout_us ? static_cast<u32>(*timeout_us - elapsed) : 0;
  } else {
    cv_.Wait(lk, [&] { return waiter.done; });
  }
  RemoveWaiter(&waiter);
  // A cancelled waiter is woken by evf_cancel, not by a matching set(). The
  // kernel marks the waiter's sleepq entry and cv_wait_sig returns a non-zero
  // status; without mode flags 0x100/0x200 that surfaces as the raw cv result,
  // with mode & 0x100 it is explicitly eCANCELED (85). We report eCANCELED.
  if (waiter.cancelled)
    return -SysError::eCANCELED;
  if (result)
    *result = waiter.result;
  return 0;
}

int EventFlag::Trywait(u64 pattern, u32 mode, u64* result) {
  base::UniqueLock<base::Mutex> lk(m_);
  if (!Satisfied(pattern, mode))
    return -SysError::eBUSY;
  return Take(pattern, mode, result);
}

void EventFlag::Set(u64 b) {
  base::LockGuard<base::Mutex> lk(m_);
  bits_ |= b;
  last_set_tid.store((long)gettid(), base::memory_order_relaxed);
  // A kernel event flag commits satisfied queued waits during set(). Keeping
  // that result on the waiter prevents a later clear from revoking the wake
  // before the host thread gets scheduled and reacquires this mutex.
  for (auto* waiter : waiters_) {
    if (waiter->done || !Satisfied(waiter->pattern, waiter->mode))
      continue;
    Take(waiter->pattern, waiter->mode, &waiter->result);
    waiter->done = true;
  }
  cv_.NotifyAll();
}

void EventFlag::Clear(u64 b) {
  base::LockGuard<base::Mutex> lk(m_);
  bits_ &= b;  // SCE clear keeps the bits set in b
}

int EventFlag::Cancel(u64 pattern) {
  base::LockGuard<base::Mutex> lk(m_);
  // Mark all waiters as cancelled. They wake from the cv with done==true but a
  // zero result, which the wait() loop turns into an error return (the kernel
  // delivers ETIMEDOUT/EINTR to a cancelled waiter).
  int n = 0;
  for (auto* w : waiters_) {
    if (w->done)
      continue;
    w->result = pattern;
    w->cancelled = true;
    w->done = true;
    ++n;
  }
  cv_.NotifyAll();
  return n;
}

// Name-keyed set for host-side subsystems that stand in for an absent system
// service (see sys_event_flag.h). The registry lock is held across set() on
// purpose: dropping it first would leave a window in which sys_evf_delete frees
// the flag under us. Nothing takes g_efRegM while holding a flag's own mutex,
// so this nesting cannot deadlock.
bool EvfSetByNameSubstr(const char* substr, u64 bits) {
  if (!substr)
    return false;
  base::LockGuard<base::Mutex> lk(g_ef_reg_m);
  for (auto& kv : g_ef_by_name) {
    if (kv.first.find(substr) == base::String::npos)
      continue;
    kv.second->Set(bits);
    return true;
  }
  return false;
}

static EventFlag* FromId(int id) {
  auto* obj = Process::GetActive()->GetObjTable().Get(id);
  if (!obj || obj->type() != Object::Type::kEventflag)
    return nullptr;
  return static_cast<EventFlag*>(obj);
}

// DELTA_EVF_TRACE[=substr]: log every evf op (optionally only for flags whose
// name contains substr) with tid + bits, to reconstruct producer/consumer
// interleavings (e.g. SOTTR's file-I/O channel handshake).
static bool EvfTraceOn(const EventFlag* ef, int id) {
  const char* filt = kEvfTrace;
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
    for (const char* p = filt + 3; *p;) {
      const int want = (int)std::strtol(p, const_cast<char**>(&p), 0);
      if (id == want)
        return true;
      while (*p == ',' || *p == ' ')
        p++;
    }
    return false;
  }
  return ef && std::strstr(ef->fname().c_str(), filt) != nullptr;
}

static void EvfTrace(const char* op,
                     int id,
                     const EventFlag* ef,
                     u64 pattern,
                     u32 mode,
                     int ret,
                     u64 res) {
  if (!EvfTraceOn(ef, id))
    return;
  // us timestamp from the same steady_clock the shm-audio dumper stamps its
  // snapshots with, so an evf signal can be placed against a cursor movement.
  const auto us = (base::TickClock::NowNs() / 1000);
  BASE_LOGI("evf",
            "us={} tid={} {} id={} '{}' pat={:#x} mode={:#x} -> ret={} "
            "res={:#x}",
            (long long)us, (long)gettid(), op, id,
            ef ? ef->fname().c_str() : "?", (unsigned long long)pattern, mode,
            ret, (unsigned long long)res);
  // DELTA_EVF_STACK: name the guest code on both sides of a handshake. Which
  // function waits or signals is what a trace of ids alone cannot say.
  if (kEvfStack)
    GuestStackTrace("evfstk", 6);
}

int PS4ABI sys_evf_create(const char* name, u32 attr, u64 init_pattern) {
  // Kernel validation:
  //   * name must be non-null (a zero name is EINVAL/22).
  //   * attr may only carry bits in 0x133 (mask 0xFFFFFECC rejects the rest).
  //   * AND+OR (attr & 3 == 3) and CLEAR_ALL+CLEAR_PAT (attr & 0x30 == 0x30)
  //   are
  //     mutually exclusive. If neither wait type is set the kernel defaults to
  //     AND (0x01); if neither clear mode is set it defaults to CLEAR_ALL
  //     (0x10).
  // We don't enforce the name check strictly: some system libs pass an empty
  // name for private flags, and our auto-naming path depends on it.
  if (!name) {
    BASE_LOGI("evf", "create rejected: null name (attr={:#x})", attr);
    return -SysError::eINVAL;
  }
  auto* ef =
      new EventFlag(Process::GetActive()->GetObjTable(), name, init_pattern);
  BASE_LOGI("evf", "create '{}' attr={:#x} init={:#x} -> id={}",
            name ? name : "", attr, (unsigned long long)init_pattern,
            ef->handle());
  return ef->handle();
}

// Some system-service event flags gate the game on state the ShellCore would
// publish (app focus granted, power normal, system running). With no ShellCore
// the flag stays 0 and the game's "wait for focus/ready" (EVF OR-wait for any
// bit) blocks forever. Seed those flags as "focused/ready" so the game
// proceeds.
static u64 SystemFlagInit(const char* name) {
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
  // The PS5 AudioOut2 daemon acks container/port/bed/master deletion by
  // setting the requester's bit on these flags after processing the delete
  // command. The IPMI side already answers those commands as done, so keep
  // every deletion ack pre-granted or LLE libSceAudioOut waits forever.
  if (n.find("SceAuOut2", 0, 9) != base::StringRef::npos &&
      n.find("DelEvf", 0, 6) != base::StringRef::npos)
    return ~0ull;
  // The settings service raises bit 32 after publishing /SceAvSetting. We
  // provide that shared-memory block in sys_shm_open, so publish its matching
  // ready state as well; otherwise the real libSceVideoOut blocks during open.
  if (n == "SceAvSettingEvf")
    return 1ull << 32;
  return 0;
}

int PS4ABI sys_evf_open(const char* name) {
  {
    base::LockGuard<base::Mutex> lk(g_ef_reg_m);
    auto it = name ? g_ef_by_name.find(name) : g_ef_by_name.end();
    if (it != g_ef_by_name.end())
      return it->second->handle();
  }
  // Auto-create unknown named flags: on real hw a system service creates them;
  // here both producer and consumer just open by name, so creating on first
  // open gives them a shared flag and the sync actually works.
  u64 seed = SystemFlagInit(name);
  auto* ef =
      new EventFlag(Process::GetActive()->GetObjTable(), name, seed, seed);
  BASE_LOGI("evf", "open '{}' (auto-created) -> id={}", name ? name : "",
            ef->handle());
  return ef->handle();
}

int PS4ABI sys_evf_delete(int id) {
  auto* ef = FromId(id);
  if (!ef)
    return -SysError::eSRCH;
  {
    base::LockGuard<base::Mutex> lk(g_ef_reg_m);
    if (!ef->fname().empty())
      g_ef_by_name.erase(ef->fname().c_str());
  }
  Process::GetActive()->GetObjTable().Release(id);
  return 0;
}

int PS4ABI sys_evf_close(int id) {
  return sys_evf_delete(id);
}

// RESEARCH INSTRUMENTATION, default OFF. DELTA_AUDIOMIX_ACK=<us>: the LLE
// libSceAudioOut mixer waits on bit <port> of "sceAudioOutMix<pid>" for the
// daemon to take its block; we host no daemon, so a port produces exactly one
// block ever. This makes the wait succeed after <us> WITHOUT consuming, keeping
// the mixer cycling so its ring can be observed. Not a daemon.
static long AudioMixAckUs() {
  return kAudioMixAck;
}

int PS4ABI
sys_evf_wait(int id, u64 pattern, u32 mode, u64* result, u32* timeout_us) {
  // Kernel mode check: mode must name exactly one of {AND, OR} and at most one
  // clear mode, and the wait pattern must be non-zero. Violations return EINVAL
  // (22) without touching the object.
  if (pattern == 0 || (mode & 3) == 0 || (mode & 3) == 3 ||
      (mode & 0x30) == 0x30)
    return -SysError::eINVAL;
  WaitProbe wp("evf_wait", (long)id, (long)pattern);
  auto* ef = FromId(id);
  if (!ef)
    return -SysError::eSRCH;
  // Trace the ENTRY too: a wait that never satisfies never reaches the exit
  // trace, which is exactly the wait one is usually hunting.
  EvfTrace("waitE", id, ef, pattern, mode, 0, 0);
  if (AudioMixAckUs() >= 0 &&
      ef->fname().find("sceAudioOutMix") != base::String::npos) {
    u32 to = static_cast<u32>(AudioMixAckUs());
    u64 ares = 0;
    int ar = ef->Wait(pattern, mode, &ares, &to);
    if (ar ==
        -SysError::eTIMEDOUT) {  // nobody signalled: fake the daemon's ack
      ar = 0;
      ares = pattern;
    }
    if (result)
      *result = ares;
    EvfTrace("ackwait", id, ef, pattern, mode, ar, ares);
    return ar;
  }
  u64 res = 0;
  int r = ef->Wait(pattern, mode, &res, timeout_us);
  if (result)
    *result = res;
  EvfTrace("wait", id, ef, pattern, mode, r, res);
  return r;
}

int PS4ABI sys_evf_trywait(int id, u64 pattern, u32 mode, u64* result) {
  if (pattern == 0 || (mode & 3) == 0 || (mode & 3) == 3 ||
      (mode & 0x30) == 0x30)
    return -SysError::eINVAL;
  auto* ef = FromId(id);
  if (!ef)
    return -SysError::eSRCH;
  u64 res = 0;
  int r = ef->Trywait(pattern, mode, &res);
  // Handshake grace: when the polling thread itself made the last set(), it is
  // the requester of a request/response channel; on hardware the
  // higher-priority responder preempts it, so the response is already posted
  // when it polls, and engines rely on that (SOTTR's file-I/O channel streams
  // garbage otherwise). Emulate with a bounded wait. Pure pollers never set the
  // flag, so they pay nothing. DELTA_NO_EVF_GRACE = A/B.
  if (r == -SysError::eBUSY && !kNoEvfGrace &&
      ef->last_set_tid.load(base::memory_order_relaxed) == (long)gettid()) {
    u32 to_us = 250000;
    r = ef->Wait(pattern, mode, &res, &to_us);
    if (r == -SysError::eTIMEDOUT)
      r = -SysError::eBUSY;
  }
  if (result)
    *result = res;
  EvfTrace("poll", id, ef, pattern, mode, r, res);
  return r;
}

// DELTA_WAIT_PROBE also tallies which flags are ever SET. A flag that threads
// park on but nobody signals is the stall; comparing the two lists names it.
static void EvfSetTally(int id) {
  if (!kWaitProbe)
    return;
  static base::Mutex m;
  static base::HashMap<int, u64> hist;
  static auto last = base::TimeTicks::Now();
  base::LockGuard<base::Mutex> lk(m);
  hist[id]++;
  const auto now = base::TimeTicks::Now();
  if (now - last < base::Seconds(10))
    return;
  last = now;
  base::String ids;
  base::FormatTo(ids, "ids ever signalled:");
  for (const auto& [k, v] : hist)
    base::FormatTo(ids, " {}(x{})", k, (unsigned long long)v);
  BASE_LOGI("evfset", "{}", ids.c_str());
}

int PS4ABI sys_evf_set(int id, u64 bits) {
  EvfSetTally(id);
  auto* ef = FromId(id);
  if (!ef)
    return -SysError::eSRCH;
  ef->Set(bits);
  EvfTrace("set", id, ef, bits, 0, 0, 0);
  return 0;
}

int PS4ABI sys_evf_clear(int id, u64 bits) {
  auto* ef = FromId(id);
  if (!ef)
    return -SysError::eSRCH;
  ef->Clear(bits);
  EvfTrace("clear", id, ef, bits, 0, 0, 0);
  return 0;
}

int PS4ABI sys_evf_cancel(int id, u64 pattern, int* num_waiters) {
  // Kernel evf_cancel: wakes every thread parked in evf_wait on this flag and
  // reports how many were released via numWaiters. The woken waiters see
  // ETIMEDOUT (60) / EINTR (85) rather than a successful match, so a cancel is
  // an abort, not a satisfy.
  auto* ef = FromId(id);
  if (!ef)
    return -SysError::eSRCH;
  int woken = ef->Cancel(pattern);
  if (num_waiters)
    *num_waiters = woken;
  EvfTrace("cancel", id, ef, pattern, 0, 0, 0);
  return 0;
}
}  // namespace kern
