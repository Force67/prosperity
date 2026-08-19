#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "base/arch.h"

namespace krnl {
// BSD/PS4 mmap prot bits (matches PROT_* and utl::pageProtection's bit layout).
enum mprotFlags : u32 { none = 0, read = 1, write = 2, exec = 4 };

enum mFlags : u32 {
  fixed = 0x10,
  stack = 0x400,
  noextend = 0x100,
  anon = 0x1000,
};

// Reserve `size` bytes of zero-filled guest memory below the 2^40 user ceiling
// (bump-allocated, never freed). Used for kernel-side guest allocations that
// must be guest-dereferenceable (identity-mapped) and pass PS4 pointer checks.
u8 *allocLowGuest(size_t size, size_t align = 0);

// mmap-family handlers return either a guest pointer or a negative errno
// encoded as a pointer (e.g. -ENOMEM as 0xFFFF...FF4). Distinguishes the two:
// every error pointer is negative when viewed as an integer, while guest
// pointers always stay below the 2^40 user ceiling.
inline bool isErrnoPtr(const u8 *p) {
  return reinterpret_cast<intptr_t>(p) < 0;
}

// VA the guest itself unmapped. sys_munmap keeps the host pages (see there), so
// a later mmap HINT at the same address would otherwise be refused and
// relocated -- which breaks any allocator that frees a probe mapping and then
// asks for an exact sub-range of it.
void noteGuestReleased(u8 *ptr, size_t size);
bool wasGuestReleased(u8 *ptr, size_t size);
// ...and VA that has been handed out again since. Direct-memory maps bypass the
// VMA (they mmap the shared store themselves), so without this a later hint
// could be honoured straight over a live one.
void noteGuestTaken(u8 *ptr, size_t size);

u8 *PS4ABI sys_mmap(void *addr, size_t size, u32 prot, u32 flags,
                         u32 fd, size_t offset);
int PS4ABI sys_mname(u8 *, size_t len, const char *name, void *);
int PS4ABI sys_mprotect(u8 *, size_t len, int prot);
int PS4ABI sys_mdbg_service(u32 op, void *, void *, void *);

// True for the shm or the named semaphore of a system service we do not host.
bool isAbsentServiceChannel(const char *name);

/*POSIX shared memory*/
int PS4ABI sys_shm_open(const char *path, u32 flags, u16 mode);
int PS4ABI sys_shm_unlink(const char *path);
int PS4ABI sys_ftruncate(u32 fd, i64 length);
// Size of a shm fd's backing for fstat (SIZE_MAX if fd isn't a shm).
size_t shmFstatSize(u32 fd);

/*direct memory access*/
int PS4ABI sys_dmem_container(u32 op);
}