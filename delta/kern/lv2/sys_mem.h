#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "guest_abi.h"

namespace kern {
// BSD/PS4 mmap prot bits (matches PROT_* and host_memory::PageProtection's bit
// layout).
enum MprotFlags : u32 { kNone = 0, kRead = 1, kWrite = 2, kExec = 4 };

enum MFlags : u32 {
  kFixed = 0x10,
  kStack = 0x400,
  kNoextend = 0x100,
  kAnon = 0x1000,
};

// Reserve `size` bytes of zero-filled guest memory below the 2^40 user ceiling
// (bump-allocated, never freed). Used for kernel-side guest allocations that
// must be guest-dereferenceable (identity-mapped) and pass PS4 pointer checks.
u8* AllocLowGuest(size_t size, size_t align = 0);

// mmap-family handlers return either a guest pointer or a negative errno
// encoded as a pointer (e.g. -ENOMEM as 0xFFFF...FF4). Distinguishes the two:
// every error pointer is negative when viewed as an integer, while guest
// pointers always stay below the 2^40 user ceiling.
inline bool IsErrnoPtr(const u8* p) {
  return reinterpret_cast<intptr_t>(p) < 0;
}

// VA the guest itself unmapped. sys_munmap keeps the host pages (see there), so
// a later mmap HINT at the same address would otherwise be refused and
// relocated, which breaks any allocator that frees a probe mapping and then
// asks for an exact sub-range of it.
void NoteGuestReleased(u8* ptr, size_t size);
bool WasGuestReleased(u8* ptr, size_t size);
// ...and VA that has been handed out again since. Direct-memory maps bypass the
// VMA (they mmap the shared store themselves), so without this a later hint
// could be honoured straight over a live one.
void NoteGuestTaken(u8* ptr, size_t size);

u8* PS4ABI
sys_mmap(void* addr, size_t size, u32 prot, u32 flags, u32 fd, size_t offset);
int PS4ABI sys_mname(u8*, size_t len, const char* name, void*);
int PS4ABI sys_mprotect(u8*, size_t len, int prot);
int PS4ABI sys_mdbg_service(u32 op, void*, void*, void*);

// True for the shm or the named semaphore of a system service we do not host.
bool IsAbsentServiceChannel(const char* name);

/*POSIX shared memory*/
int PS4ABI sys_shm_open(const char* path, u32 flags, u16 mode);
int PS4ABI sys_shm_unlink(const char* path);
int PS4ABI sys_ftruncate(u32 fd, i64 length);
// Size of a shm fd's backing for fstat (SIZE_MAX if fd isn't a shm).
size_t ShmFstatSize(u32 fd);

/*direct memory access*/
int PS4ABI sys_dmem_container(u32 op);
}  // namespace kern

namespace kern {
/* POSIX-ish memory lifecycle (we run on a flat host-backed VM; teardown is a
 * no-op since the host reclaims everything at process exit) */
int PS4ABI sys_munmap(void* addr, size_t len);
i64 PS4ABI sys_obreak(void* addr);
i64 PS4ABI sys_sbrk(intptr_t incr);
int PS4ABI sys_msync(void* addr, size_t len, int flags);
int PS4ABI sys_madvise(void* addr, size_t len, int behav);
int PS4ABI sys_mincore(void* addr, size_t len, char* vec);
int PS4ABI sys_mlock(const void* addr, size_t len);
int PS4ABI sys_munlock(const void* addr, size_t len);
int PS4ABI sys_mlockall(int how);
int PS4ABI sys_munlockall();
int PS4ABI sys_minherit(void* addr, size_t len, int inherit);

/* Sony memory-introspection syscalls */
int PS4ABI sys_query_memory_protection(void* addr, void* info);
int PS4ABI sys_virtual_query(const void* addr,
                             int flags,
                             void* info,
                             size_t info_size);

/* Sony direct-memory / vm-map management (largely stubbed) */
int PS4ABI
sys_batch_map(u32 handle, u32 flags, void* entries, int count, int* processed);
int PS4ABI sys_set_vm_container(u32 container);
i64 PS4ABI sys_mmap_dmem(void* addr,
                         size_t len,
                         int prot,
                         int flags,
                         i64 packed,
                         i64 phys_offset);
int PS4ABI
sys_cpuset(void* out, int level, int which, i64 id, size_t size, void* mask);
int PS4ABI sys_extend_page_table_pool();
i64 PS4ABI sys_get_vm_map_timestamp();
int PS4ABI sys_get_map_statistics(void* info);
int PS4ABI sys_free_stack(void* addr, size_t len);

i64 PS4ABI sys_jitshm_create(size_t len, u32 flags);
int PS4ABI sys_jitshm_alias();
int PS4ABI sys_get_paging_stats_of_all_threads();
int PS4ABI sys_get_paging_stats_of_all_objects();
int PS4ABI sys_get_resident_count();
int PS4ABI sys_get_resident_fmem_count();
int PS4ABI sys_physhm_open();
int PS4ABI sys_physhm_unlink();
int PS4ABI sys_set_phys_fmem_limit();
int PS4ABI sys_get_kernel_mem_statistics(void* out);
i64 PS4ABI sys_blockpool_map(i64 pool, size_t len, u32 prot, u32 flags);
int PS4ABI sys_blockpool_unmap();
i64 PS4ABI sys_blockpool_batch(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5);
int PS4ABI sys_get_page_table_stats();
int PS4ABI sys_getpagesize();

int PS4ABI sys_blockpool_open();

}  // namespace kern
