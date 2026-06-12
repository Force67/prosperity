/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <ctime>

#include "error_table.h"
#include "sys_time.h"

namespace krnl {
// PS4 timespec is {int64 tv_sec; int64 tv_nsec}, same layout as the host's on
// x86-64. Map the guest clock id onto a host clock; unknown ids fall back to
// monotonic (good enough for the deltas most callers want).
int PS4ABI sys_clock_gettime(uint32_t clock_id, sce_timespec *tp) {
  if (!tp)
    return -SysError::eINVAL;

  clockid_t host;
  switch (clock_id) {
  case 0:  // SCE_KERNEL_CLOCK_REALTIME
    host = CLOCK_REALTIME;
    break;
  default:  // monotonic / proctime / threadtime
    host = CLOCK_MONOTONIC;
    break;
  }

  struct timespec ts {};
  clock_gettime(host, &ts);
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
