#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <guest_abi.h>
#include "base/arch.h"

namespace krnl {
// Thread lifecycle / signalling.
int PS4ABI sys_thr_exit(i64 *state);
int PS4ABI sys_thr_kill(u32 tid, int sig);
int PS4ABI sys_thr_kill2(u32 pid, u32 tid, int sig);
int PS4ABI sys_thr_suspend(const void *timeout);
int PS4ABI sys_thr_wake(u32 tid);
int PS4ABI sys_thr_set_name(u32 tid, const char *name);
int PS4ABI sys_thr_get_name(u32 tid, char *buf);

// Scheduling.
int PS4ABI sys_yield();
int PS4ABI sys_sched_yield();
int PS4ABI sys_sched_get_priority_max(int policy);
int PS4ABI sys_sched_get_priority_min(int policy);
int PS4ABI sys_sched_setparam(int pid, const void *param);
int PS4ABI sys_sched_getparam(int pid, void *param);
int PS4ABI sys_sched_setscheduler(int pid, int policy, const void *param);
int PS4ABI sys_sched_getscheduler(int pid);
int PS4ABI sys_sched_rr_get_interval(int pid, void *interval);

// cpuset (484 sys_cpuset is owned elsewhere; not declared here).
int PS4ABI sys_cpuset_setid(int which, int level, void *id, int setid);
int PS4ABI sys_cpuset_getid(int level, int which, i64 id, int *setid);
int PS4ABI sys_cpuset_setaffinity(int level, int which, i64 id, size_t cpusetsize,
                                  const void *mask);

// ucontext-based suspend/resume (not exposed).
int PS4ABI sys_thr_suspend_ucontext(u32 tid);
int PS4ABI sys_thr_resume_ucontext(u32 tid);
int PS4ABI sys_thr_get_ucontext(u32 tid, void *ucontext);
int PS4ABI sys_thr_set_ucontext(u32 tid, void *ucontext);
}
