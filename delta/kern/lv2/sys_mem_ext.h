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
/* POSIX-ish memory lifecycle (we run on a flat host-backed VM; teardown is a
 * no-op since the host reclaims everything at process exit) */
int PS4ABI sys_munmap(void *addr, size_t len);
int64_t PS4ABI sys_obreak(void *addr);
int64_t PS4ABI sys_sbrk(intptr_t incr);
int PS4ABI sys_msync(void *addr, size_t len, int flags);
int PS4ABI sys_madvise(void *addr, size_t len, int behav);
int PS4ABI sys_mincore(void *addr, size_t len, char *vec);
int PS4ABI sys_mlock(const void *addr, size_t len);
int PS4ABI sys_munlock(const void *addr, size_t len);
int PS4ABI sys_mlockall(int how);
int PS4ABI sys_munlockall();
int PS4ABI sys_minherit(void *addr, size_t len, int inherit);

/* Sony memory-introspection syscalls */
int PS4ABI sys_query_memory_protection(void *addr, void *info);
int PS4ABI sys_virtual_query(const void *addr, int flags, void *info,
                             size_t infoSize);

/* Sony direct-memory / vm-map management (largely stubbed) */
int PS4ABI sys_batch_map(uint32_t handle, uint32_t flags, void *entries,
                         int count, int *processed);
int PS4ABI sys_set_vm_container(uint32_t container);
int64_t PS4ABI sys_mmap_dmem(void *addr, size_t len, int memType, int prot,
                             int flags, int64_t directMemoryStart);
int PS4ABI sys_cpuset(void *out, int level, int which, int64_t id, size_t size,
                      void *mask);
int PS4ABI sys_extend_page_table_pool();
int64_t PS4ABI sys_get_vm_map_timestamp();
int PS4ABI sys_get_map_statistics(void *info);
int PS4ABI sys_free_stack(void *addr, size_t len);
}
