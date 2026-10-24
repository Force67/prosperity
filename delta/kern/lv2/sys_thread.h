#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstddef>
#include "base/arch.h"
#include "guest_abi.h"

namespace kern {
struct thread_prio {  // NOLINT(readability-identifier-naming): guest ABI name
  u16 type;
  u16 prio;
};

// FreeBSD thr_param (rdi layout, 0x68 bytes); +40 tls_size / +64 flags exist
// for the user-mode libthr wrapper, untouched by the kernel path.
struct thr_param {  // NOLINT(readability-identifier-naming): guest ABI name
  void(PS4ABI* start_func)(void*);  // +0x00
  void* arg;                        // +0x08
  u8* stack_base;                   // +0x10
  size_t stack_size;                // +0x18
  u8* tls_base;                     // +0x20
  size_t tls_size;                  // +0x28  (not read by kern_thr_new)
  i64* child_tid;                   // +0x30
  i64* parent_tid;                  // +0x38
  i32 flags;                        // +0x40  (not read by kern_thr_new)
  i32 pad;
  void* rtp;         // +0x48  rtprio (4-byte struct copied in)
  const char* name;  // +0x50  thread name (max 32 chars)
  void* spare[2];    // +0x58
};
int PS4ABI sys_thr_new(thr_param* p, int size);
int PS4ABI sys_thr_self(i64* tid);
int PS4ABI sys_rtprio_thread(int, u64, thread_prio*);
int PS4ABI sys_umtx_op(void*, int, u64, void*, void*);
}  // namespace kern

namespace kern {
// Thread lifecycle / signalling.
int PS4ABI sys_thr_exit(i64* state);
int PS4ABI sys_thr_set_name(u32 tid, const char* name);
int PS4ABI sys_thr_get_name(u32 tid, char* buf);

// Scheduling.
int PS4ABI sys_yield();
int PS4ABI sys_sched_yield();
int PS4ABI sys_sched_get_priority_max(int policy);
int PS4ABI sys_sched_get_priority_min(int policy);
int PS4ABI sys_sched_setparam(int pid, const void* param);
int PS4ABI sys_sched_getparam(int pid, void* param);
int PS4ABI sys_sched_setscheduler(int pid, int policy, const void* param);
int PS4ABI sys_sched_getscheduler(int pid);
int PS4ABI sys_sched_rr_get_interval(int pid, void* interval);

// cpuset (484 sys_cpuset is owned elsewhere; not declared here).
int PS4ABI sys_cpuset_setid(int which, int level, void* id, int setid);
int PS4ABI sys_cpuset_getid(int level, int which, i64 id, int* setid);
int PS4ABI sys_cpuset_setaffinity(int level,
                                  int which,
                                  i64 id,
                                  size_t cpusetsize,
                                  const void* mask);

int PS4ABI sys_sblock_create();
int PS4ABI sys_sblock_delete();
int PS4ABI sys_sblock_enter();
int PS4ABI sys_sblock_exit();
int PS4ABI sys_sblock_xenter();
int PS4ABI sys_sblock_xexit();
int PS4ABI sys_setpriority();
int PS4ABI sys_getpriority();
int PS4ABI sys_thr_sleep();
int PS4ABI sys_thr_wakeup();

int PS4ABI sys_sysarch(int num, void* args);

// The calling guest thread's tid slot in host TLS.
const u32* CurrentGuestTidPtr();

}  // namespace kern
