#pragma once

// Copyright (C) Force67 2019

#include <base.h>

#include "sys_time.h"

namespace krnl {
// timeval / timezone use the LP64 FreeBSD layout. On PS4 (always LP64) tv_sec
// and tv_usec are 64-bit, unlike the host's struct timeval where they may be
// 32/64-bit depending on platform, so we mirror them explicitly here.
struct sce_timeval {
  int64_t tv_sec;
  int64_t tv_usec;
};

struct sce_timezone {
  int tz_minuteswest;
  int tz_dsttime;
};

// itimerval / itimerspec are just pairs of timeval / timespec.
struct sce_itimerval {
  sce_timeval it_interval;
  sce_timeval it_value;
};

struct sce_itimerspec {
  sce_timespec it_interval;
  sce_timespec it_value;
};

int PS4ABI sys_gettimeofday(sce_timeval *tv, sce_timezone *tz);
int PS4ABI sys_settimeofday(const sce_timeval *tv, const sce_timezone *tz);
int PS4ABI sys_clock_settime(uint32_t clock_id, const sce_timespec *tp);
int PS4ABI sys_clock_getres(uint32_t clock_id, sce_timespec *res);
int PS4ABI sys_clock_getcpuclockid2(uint64_t id, int which, int *out);

int PS4ABI sys_getitimer(int which, sce_itimerval *val);
int PS4ABI sys_setitimer(int which, const sce_itimerval *val,
                         sce_itimerval *oval);

int PS4ABI sys_ktimer_create(uint32_t clock_id, void *evp, int *timerid);
int PS4ABI sys_ktimer_delete(int timerid);
int PS4ABI sys_ktimer_settime(int timerid, int flags,
                              const sce_itimerspec *newval,
                              sce_itimerspec *oldval);
int PS4ABI sys_ktimer_gettime(int timerid, sce_itimerspec *cur);
int PS4ABI sys_ktimer_getoverrun(int timerid);

int PS4ABI sys_ffclock_getcounter(uint64_t *ffcount);
int PS4ABI sys_ffclock_setestimate(void *cest);
int PS4ABI sys_ffclock_getestimate(void *cest);

int PS4ABI sys_ntp_gettime(void *ntv);

int PS4ABI sys_set_timezone_info(void *info);
int PS4ABI sys_utc_to_localtime(int64_t utc, void *out, void *tzinfo, void *dst);
int PS4ABI sys_localtime_to_utc(int64_t local, void *out, void *a, void *b,
                                void *c);
}  // namespace krnl
