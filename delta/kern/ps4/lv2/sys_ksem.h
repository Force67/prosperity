#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>

namespace krnl {
// FreeBSD POSIX kernel semaphores (ksem_*). These are the libc/pthread-backing
// counting semaphores keyed by an integer semid_t handed out by ksem_init /
// ksem_open, distinct from the SCE "osem" syscalls (549+). They were
// null_handler stubs, so sem_wait() never blocked and consumers raced their
// producers. We back them with a self-contained registry (see the .cpp); the
// kObject path is intentionally avoided since ksem ids are plain ints, not
// object-table handles.

// timespec as the guest passes it: two 64-bit fields. Declared here so the
// timedwait prototype is well-formed; layout matches FreeBSD's struct timespec
// on the 64-bit ABI.
struct ksem_timespec {
  int64_t tv_sec;
  int64_t tv_nsec;
};

int PS4ABI sys_ksem_init(int *idp, unsigned value);
int PS4ABI sys_ksem_open(int *idp, const char *name, int oflag, uint16_t mode,
                         unsigned value);
int PS4ABI sys_ksem_unlink(const char *name);
int PS4ABI sys_ksem_close(int id);
int PS4ABI sys_ksem_post(int id);
int PS4ABI sys_ksem_wait(int id);
int PS4ABI sys_ksem_trywait(int id);
int PS4ABI sys_ksem_timedwait(int id, const struct ksem_timespec *abstime);
int PS4ABI sys_ksem_getvalue(int id, int *val);
int PS4ABI sys_ksem_destroy(int id);
}  // namespace krnl
