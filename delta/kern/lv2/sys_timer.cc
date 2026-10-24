/*
 * PS4Delta : PS4 emulation and research project
 *
 * Interval timers (setitimer) and POSIX timers (ktimer_*), fired by one
 * service thread into the guest's signals.
 */

#include <cstring>
#include <ctime>
#include "base/algorithm.h"
#include "base/arch.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "guest/session.h"
#include "guest_abi.h"
#include "host_memory/host_memory.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_signal.h"
#include "kern/lv2/sys_time.h"

namespace kern {
namespace {
constexpr int kSigAlrm = 14, kSigVtalrm = 26, kSigProf = 27;
constexpr int kItimerReal = 0, kItimerVirtual = 1, kItimerProf = 2;
constexpr int kSigevNone = 0, kSigevSignal = 1, kSigevThreadId = 4;
constexpr int kTimerAbstime = 1;
constexpr int kTimerFirst = 3;  // 0..2 name the interval timers
constexpr int kTimerMax = 32;
constexpr int kClockRealtime = 0;
constexpr u64 kSecond = 1'000'000'000;

u64 Now(clockid_t clock) {
  timespec ts{};
  ::clock_gettime(clock, &ts);
  return u64(ts.tv_sec) * kSecond + u64(ts.tv_nsec);
}

struct Timer {
  bool used = false;
  clockid_t clock = CLOCK_MONOTONIC;  // the host clock the deadline counts on
  int notify = kSigevSignal;
  int signo = kSigAlrm;
  u64 value = 0;
  u32 tid = 0;     // 0: the process
  u64 expiry = 0;  // 0: disarmed
  u64 interval = 0;
  int overrun = 0;       // expirations the pending signal absorbed
  int last_overrun = 0;  // what the last delivered expiration absorbed
};

base::Mutex g_mutex;
base::ConditionVariable g_cv;
Timer g_timers[kTimerMax];
bool g_service = false;

// The interval timers live in slots 0..2 of the same table.
Timer& Itimer(int which) {
  return g_timers[which];
}

clockid_t ItimerClock(int which) {
  return which == kItimerReal ? CLOCK_MONOTONIC : CLOCK_PROCESS_CPUTIME_ID;
}

void Fire(int id, Timer& t, u64 now) {
  if (t.notify != kSigevNone) {
    if (IsSignalPending(t.tid, t.signo)) {
      t.overrun++;
    } else {
      SignalInfo info{};
      info.signo = t.signo;
      info.code = id >= kTimerFirst ? kSiTimer : kSiKernel;
      info.value = t.value;
      info.timerid = id;
      info.overrun = t.overrun;
      t.last_overrun = t.overrun;
      t.overrun = 0;
      PostSignal(t.tid, info);
    }
  }
  if (!t.interval) {
    t.expiry = 0;
    return;
  }
  // Expirations missed while late count as overruns.
  const u64 late = now - t.expiry;
  const u64 missed = late / t.interval;
  t.overrun += static_cast<int>(missed);
  t.expiry += (missed + 1) * t.interval;
}

void Service() {
  base::UniqueLock<base::Mutex> lk(g_mutex);
  while (!guest::Stopping()) {
    u64 wait_ns = 50'000'000;
    for (int id = 0; id < kTimerMax; id++) {
      Timer& t = g_timers[id];
      if (!t.used || !t.expiry)
        continue;
      const u64 now = Now(t.clock);
      if (now >= t.expiry)
        Fire(id, t, now);
      if (t.expiry) {
        const u64 left = t.expiry > now ? t.expiry - now : 0;
        // CPU-time clocks advance slower than the wall: poll them.
        wait_ns = base::Min<u64>(
            wait_ns, t.clock == CLOCK_MONOTONIC || t.clock == CLOCK_REALTIME
                         ? left
                         : base::Min<u64>(left, 10'000'000));
      }
    }
    g_cv.WaitFor(lk, base::Microseconds(static_cast<i64>(wait_ns / 1000) + 1));
  }
  g_service = false;
}

// Under g_mutex.
void Wake() {
  if (!g_service) {
    g_service = true;
    guest::SpawnThread("guest_timers", [] { Service(); });
  }
  g_cv.NotifyAll();
}

bool ValidSpec(const sce_timespec& ts) {
  return ts.tv_sec >= 0 && ts.tv_nsec >= 0 && ts.tv_nsec < i64(kSecond);
}

u64 ToNs(const sce_timespec& ts) {
  return u64(ts.tv_sec) * kSecond + u64(ts.tv_nsec);
}

sce_timespec FromNs(u64 ns) {
  return {i64(ns / kSecond), i64(ns % kSecond)};
}

// Arms (or with a zero value disarms) `t`, reporting the old setting.
void Arm(Timer& t, u64 value_ns, u64 interval_ns, sce_itimerspec* old) {
  const u64 now = Now(t.clock);
  if (old) {
    old->it_interval = FromNs(t.interval);
    old->it_value = FromNs(t.expiry > now ? t.expiry - now : t.expiry ? 1 : 0);
  }
  t.expiry = value_ns ? now + value_ns : 0;
  t.interval = value_ns ? interval_ns : 0;
  t.overrun = 0;
  Wake();
}

Timer* FindTimer(int id) {
  if (id < kTimerFirst || id >= kTimerMax || !g_timers[id].used)
    return nullptr;
  return &g_timers[id];
}

const guest::SessionReset g_reset([] {
  base::LockGuard<base::Mutex> lk(g_mutex);
  for (auto& t : g_timers)
    t = Timer{};
});
}  // namespace

int PS4ABI sys_getitimer(int which, sce_itimerval* val) {
  if (which < kItimerReal || which > kItimerProf)
    return -SysError::eINVAL;
  if (!host_memory::IsMemoryRangeMapped(val, sizeof(*val)))
    return -SysError::eFAULT;
  base::LockGuard<base::Mutex> lk(g_mutex);
  const Timer& t = Itimer(which);
  const u64 now = Now(ItimerClock(which));
  const u64 left = t.used && t.expiry > now ? t.expiry - now : 0;
  val->it_interval = {i64(t.interval / kSecond),
                      i64(t.interval % kSecond / 1000)};
  val->it_value = {i64(left / kSecond), i64(left % kSecond / 1000)};
  return 0;
}

int PS4ABI sys_setitimer(int which,
                         const sce_itimerval* val,
                         sce_itimerval* oval) {
  if (which < kItimerReal || which > kItimerProf)
    return -SysError::eINVAL;
  if (oval) {
    if (const int err = sys_getitimer(which, oval))
      return err;
  }
  if (!val)
    return 0;
  if (!host_memory::IsMemoryRangeMapped(val, sizeof(*val)))
    return -SysError::eFAULT;
  auto valid = [](const sce_timeval& tv) {
    return tv.tv_sec >= 0 && tv.tv_usec >= 0 && tv.tv_usec < 1'000'000;
  };
  if (!valid(val->it_value) || !valid(val->it_interval))
    return -SysError::eINVAL;
  auto ns = [](const sce_timeval& tv) {
    return u64(tv.tv_sec) * kSecond + u64(tv.tv_usec) * 1000;
  };
  base::LockGuard<base::Mutex> lk(g_mutex);
  Timer& t = Itimer(which);
  t.used = true;
  t.clock = ItimerClock(which);
  t.notify = kSigevSignal;
  t.signo = which == kItimerReal      ? kSigAlrm
            : which == kItimerVirtual ? kSigVtalrm
                                      : kSigProf;
  t.tid = 0;
  Arm(t, ns(val->it_value), ns(val->it_interval), nullptr);
  return 0;
}

// sigevent: notify, signo, value, then the target thread for THREAD_ID.
int PS4ABI sys_ktimer_create(u32 clock_id, void* evp, int* timerid) {
  clockid_t clock;
  switch (clock_id) {
    case kClockRealtime:
    case 9:   // CLOCK_REALTIME_PRECISE
    case 10:  // CLOCK_REALTIME_FAST
      clock = CLOCK_REALTIME;
      break;
    case 4:   // CLOCK_MONOTONIC
    case 5:   // CLOCK_UPTIME
    case 7:   // CLOCK_UPTIME_PRECISE
    case 8:   // CLOCK_UPTIME_FAST
    case 11:  // CLOCK_MONOTONIC_PRECISE
    case 12:  // CLOCK_MONOTONIC_FAST
      clock = CLOCK_MONOTONIC;
      break;
    default:
      return -SysError::eINVAL;
  }
  if (!host_memory::IsMemoryRangeMapped(timerid, sizeof(int)))
    return -SysError::eFAULT;
  int notify = kSigevSignal, signo = kSigAlrm, tid = 0;
  u64 value = 0;
  bool default_value = true;
  if (evp) {
    if (!host_memory::IsMemoryRangeMapped(evp, 0x14))
      return -SysError::eFAULT;
    const auto* ev = static_cast<const u8*>(evp);
    std::memcpy(&notify, ev, 4);
    std::memcpy(&signo, ev + 4, 4);
    std::memcpy(&value, ev + 8, 8);
    std::memcpy(&tid, ev + 0x10, 4);
    default_value = false;
    if (notify != kSigevNone && notify != kSigevSignal &&
        notify != kSigevThreadId)
      return -SysError::eINVAL;
    if (notify != kSigevNone && (signo < 1 || signo > kGuestNsig))
      return -SysError::eINVAL;
    if (notify != kSigevThreadId)
      tid = 0;
  }
  base::LockGuard<base::Mutex> lk(g_mutex);
  int id = kTimerFirst;
  while (id < kTimerMax && g_timers[id].used)
    id++;
  if (id == kTimerMax)
    return -SysError::eAGAIN;
  Timer& t = g_timers[id];
  t = Timer{};
  t.used = true;
  t.clock = clock;
  t.notify = notify;
  t.signo = signo;
  t.value = default_value ? u64(id) : value;
  t.tid = static_cast<u32>(tid);
  *timerid = id;
  return 0;
}

int PS4ABI sys_ktimer_delete(int timerid) {
  base::LockGuard<base::Mutex> lk(g_mutex);
  Timer* t = FindTimer(timerid);
  if (!t)
    return -SysError::eINVAL;
  *t = Timer{};
  return 0;
}

int PS4ABI sys_ktimer_settime(int timerid,
                              int flags,
                              const sce_itimerspec* newval,
                              sce_itimerspec* oldval) {
  if (!host_memory::IsMemoryRangeMapped(newval, sizeof(*newval)))
    return -SysError::eFAULT;
  if (!ValidSpec(newval->it_value) || !ValidSpec(newval->it_interval))
    return -SysError::eINVAL;
  base::LockGuard<base::Mutex> lk(g_mutex);
  Timer* t = FindTimer(timerid);
  if (!t)
    return -SysError::eINVAL;
  if (oldval && !host_memory::IsMemoryRangeMapped(oldval, sizeof(*oldval)))
    return -SysError::eFAULT;
  u64 value = ToNs(newval->it_value);
  if (value && (flags & kTimerAbstime)) {
    const u64 now = Now(t->clock);
    value = value > now ? value - now : 1;  // a past deadline fires at once
  }
  Arm(*t, value, ToNs(newval->it_interval), oldval);
  return 0;
}

int PS4ABI sys_ktimer_gettime(int timerid, sce_itimerspec* cur) {
  if (!host_memory::IsMemoryRangeMapped(cur, sizeof(*cur)))
    return -SysError::eFAULT;
  base::LockGuard<base::Mutex> lk(g_mutex);
  const Timer* t = FindTimer(timerid);
  if (!t)
    return -SysError::eINVAL;
  const u64 now = Now(t->clock);
  cur->it_interval = FromNs(t->interval);
  cur->it_value = FromNs(t->expiry > now ? t->expiry - now : t->expiry ? 1 : 0);
  return 0;
}

int PS4ABI sys_ktimer_getoverrun(int timerid) {
  base::LockGuard<base::Mutex> lk(g_mutex);
  const Timer* t = FindTimer(timerid);
  if (!t)
    return -SysError::eINVAL;
  return t->last_overrun;
}

}  // namespace kern
