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
    return SysError::eINVAL;

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
}  // namespace krnl
