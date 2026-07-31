/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "error_table.h"
#include "sys_time.h"
#include "sys_time_ext.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(long, kTimeScaleRt, "DELTA_TIMESCALE_RT", 1);
}  // namespace

namespace krnl {
// gettimeofday: wall-clock time as seconds + microseconds. We pull from
// CLOCK_REALTIME (same source as sys_clock_gettime's REALTIME path) and convert
// the nanosecond field down to microseconds for the timeval.
int PS4ABI sys_gettimeofday(sce_timeval *tv, sce_timezone *tz) {
  if (tv) {
    struct timespec ts {};
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t now = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
    // DIAGNOSTIC (DELTA_TIMESCALE_RT=N): advance the wall clock Nx from a fixed
    // baseline. Classifies a busy-wait gated on a gettimeofday-based TIMEOUT
    // (Doom64's ~1fps main loop hammers gettimeofday, not monotonic, so the old
    // DELTA_TIMESCALE which only scaled monotonic had no effect): if fps rises the
    // loop is timeout-bound (it gives up sooner); if unchanged it is work-bound.
    const long rtScale = kTimeScaleRt;
    if (rtScale > 1) {
      static const uint64_t base = now;
      now = base + (now - base) * static_cast<uint64_t>(rtScale);
    }
    tv->tv_sec = static_cast<long>(now / 1000000000ull);
    tv->tv_usec = static_cast<long>((now % 1000000000ull) / 1000);
  }
  if (tz) {
    // We report UTC with no DST: the guest's notion of "local time" is handled
    // by the Sony utc<->local helpers below, not by the kernel timezone.
    tz->tz_minuteswest = 0;
    tz->tz_dsttime = 0;
  }
  return 0;
}

// We can't (and shouldn't) reprogram the host wall clock from the guest, so the
// time-setting syscalls are accepted and ignored.
int PS4ABI sys_settimeofday(const sce_timeval *tv, const sce_timezone *tz) {
  return 0;
}

int PS4ABI sys_clock_settime(uint32_t clock_id, const sce_timespec *tp) {
  return 0;
}

// Report a 1ns clock resolution. The host clocks we back these with are far
// finer than anything the guest schedules on, so advertising 1ns avoids the
// guest rounding its waits up to a coarser (and slower) granularity.
int PS4ABI sys_clock_getres(uint32_t clock_id, sce_timespec *res) {
  if (res) {
    res->tv_sec = 0;
    res->tv_nsec = 1;
  }
  return 0;
}

// Map any (pid, which) request onto a single cpu-time clock id. 2 corresponds
// to CLOCK_PROCESS_CPUTIME_ID on FreeBSD; callers only ever feed the result
// straight back into clock_gettime, which our handler treats as monotonic.
int PS4ABI sys_clock_getcpuclockid2(uint64_t id, int which, int *out) {
  if (out)
    *out = 2;
  return 0;
}

// We don't run real interval timers; report "disarmed" (all zero) and accept
// arming requests as no-ops so guest code that merely sets/clears an itimer
// during init keeps going.
int PS4ABI sys_getitimer(int which, sce_itimerval *val) {
  if (val)
    std::memset(val, 0, sizeof(sce_itimerval));
  return 0;
}

int PS4ABI sys_setitimer(int which, const sce_itimerval *val,
                         sce_itimerval *oval) {
  if (oval)
    std::memset(oval, 0, sizeof(sce_itimerval));
  return 0;
}

// POSIX per-process timers. We hand out unique ids but never actually fire the
// associated signals/events; this keeps create/delete/arm bookkeeping in the
// guest happy without us needing a timer thread.
int PS4ABI sys_ktimer_create(uint32_t clock_id, void *evp, int *timerid) {
  static std::atomic<int> next{1};
  if (timerid)
    *timerid = next.fetch_add(1, std::memory_order_relaxed);
  return 0;
}

int PS4ABI sys_ktimer_delete(int timerid) { return 0; }

int PS4ABI sys_ktimer_settime(int timerid, int flags,
                              const sce_itimerspec *newval,
                              sce_itimerspec *oldval) {
  // Report the previous setting as disarmed; we never armed anything.
  if (oldval)
    std::memset(oldval, 0, sizeof(sce_itimerspec));
  return 0;
}

int PS4ABI sys_ktimer_gettime(int timerid, sce_itimerspec *cur) {
  if (cur)
    std::memset(cur, 0, sizeof(sce_itimerspec));
  return 0;
}

// No timer ever fires, so it can never overrun.
int PS4ABI sys_ktimer_getoverrun(int timerid) { return 0; }

// Feed-forward clock counter. We don't model an ffclock, so just expose a
// monotonically increasing nanosecond count, which is all the counter is used
// for in practice.
int PS4ABI sys_ffclock_getcounter(uint64_t *ffcount) {
  if (ffcount) {
    struct timespec ts {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    *ffcount = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
               static_cast<uint64_t>(ts.tv_nsec);
  }
  return 0;
}

int PS4ABI sys_ffclock_setestimate(void *cest) { return 0; }

int PS4ABI sys_ffclock_getestimate(void *cest) { return 0; }

// ntp_gettime fills a struct ntptimeval. We don't run an NTP discipline, so
// zero the whole buffer (0x48 bytes on LP64: a timespec plus several long/int
// fields) to report an unsynchronised but valid state.
int PS4ABI sys_ntp_gettime(void *ntv) {
  if (ntv)
    std::memset(ntv, 0, 0x48);
  return 0;
}

// Sony timezone-info setter; we keep a single UTC=local view, so ignore it.
int PS4ABI sys_set_timezone_info(void *info) { return 0; }

// Sony UTC<->local conversion helpers. We treat the guest as running in UTC
// (zero offset, no DST), so the conversion is the identity: copy the input
// time_t straight to the output if one was supplied. This keeps timestamps
// self-consistent rather than introducing a spurious offset.
int PS4ABI sys_utc_to_localtime(int64_t utc, void *out, void *tzinfo,
                                void *dst) {
  if (out)
    *reinterpret_cast<int64_t *>(out) = utc;
  return 0;
}

int PS4ABI sys_localtime_to_utc(int64_t local, void *out, void *a, void *b,
                                void *c) {
  if (out)
    *reinterpret_cast<int64_t *>(out) = local;
  return 0;
}
}  // namespace krnl
