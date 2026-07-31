/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <cstdlib>
#include <ctime>

#include "error_table.h"
#include "sys_time.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(long, kTimeScale, "DELTA_TIMESCALE", 1);
}  // namespace

namespace krnl {
// PS4 timespec is {int64 tv_sec; int64 tv_nsec}, same layout as the host's on
// x86-64. Map the guest clock id onto a host clock; unknown ids fall back to
// monotonic (good enough for the deltas most callers want).
int PS4ABI sys_clock_gettime(uint32_t clock_id, sce_timespec *tp) {
  if (!tp)
    return -SysError::eINVAL;

  // Map the guest clock id onto the closest host clock. The ids follow FreeBSD's
  // (kern_clock_gettime): 0/9/10 realtime, 13 second (coarse realtime), 14/15
  // thread/process cpu time, the rest monotonic/uptime.
  clockid_t host;
  switch (clock_id) {
  case 0:   // CLOCK_REALTIME
  case 9:   // CLOCK_REALTIME_PRECISE
  case 10:  // CLOCK_REALTIME_FAST
  case 13:  // CLOCK_SECOND (coarse wall clock)
    host = CLOCK_REALTIME;
    break;
  case 1:   // CLOCK_VIRTUAL (user cpu time) -> process cpu time
  case 2:   // CLOCK_PROF (user+sys cpu time)
  case 15:  // CLOCK_PROCESS_CPUTIME_ID
    host = CLOCK_PROCESS_CPUTIME_ID;
    break;
  case 14:  // CLOCK_THREAD_CPUTIME_ID
    host = CLOCK_THREAD_CPUTIME_ID;
    break;
  default:  // 4/5/7/8/11/12: monotonic & uptime variants
    host = CLOCK_MONOTONIC;
    break;
  }

  struct timespec ts {};
  clock_gettime(host, &ts);
  // DIAGNOSTIC (DELTA_TIMESCALE=N): make the guest's monotonic clock advance N x
  // faster from a fixed baseline. If a title's ~1fps spin is a frame-rate LIMITER
  // (`while(now < target) yield`), this makes it exit Nx sooner (fps up); if it's
  // a work loop drowning in clock-syscall overhead, fps is unchanged. Classifies
  // the bottleneck. Realtime is left alone (only monotonic/uptime ids scale).
  const long scale = kTimeScale;
  if (scale > 1 && host == CLOCK_MONOTONIC) {
    static const uint64_t base = (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
    uint64_t now = (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
    uint64_t scaled = base + (now - base) * (uint64_t)scale;
    ts.tv_sec = scaled / 1000000000ull;
    ts.tv_nsec = scaled % 1000000000ull;
  }
  tp->tv_sec = ts.tv_sec;
  tp->tv_nsec = ts.tv_nsec;
  return 0;
}

// Actually sleep. Stubbing this to return immediately turns guest polling loops
// (which sleep between checks) into full-speed busy-waits that starve the very
// threads they're waiting on. The PS4 timespec matches the host layout.
int PS4ABI sys_nanosleep(const sce_timespec *rqtp, sce_timespec *rmtp) {
  if (!rqtp)
    return -SysError::eINVAL;

  struct timespec req {};
  req.tv_sec = static_cast<time_t>(rqtp->tv_sec);
  req.tv_nsec = static_cast<long>(rqtp->tv_nsec);

  struct timespec rem {};
  int r = ::nanosleep(&req, &rem);
  if (r != 0 && rmtp) {
    rmtp->tv_sec = rem.tv_sec;
    rmtp->tv_nsec = rem.tv_nsec;
  }
  return r == 0 ? 0 : -SysError::eINTR;
}
}  // namespace krnl
