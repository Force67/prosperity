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
// BSD/PS4 mmap prot bits (matches PROT_* and utl::pageProtection's bit layout).
enum mprotFlags : uint32_t { none = 0, read = 1, write = 2, exec = 4 };

enum mFlags : uint32_t {
  fixed = 0x10,
  stack = 0x400,
  noextend = 0x100,
  anon = 0x1000,
};

// Reserve `size` bytes of zero-filled guest memory below the 2^40 user ceiling
// (bump-allocated, never freed). Used for kernel-side guest allocations that
// must be guest-dereferenceable (identity-mapped) and pass PS4 pointer checks.
uint8_t *allocLowGuest(size_t size, size_t align = 0);

// mmap-family handlers return either a guest pointer or a negative errno
// encoded as a pointer (e.g. -ENOMEM as 0xFFFF...FF4). Distinguishes the two:
// every error pointer is negative when viewed as an integer, while guest
// pointers always stay below the 2^40 user ceiling.
inline bool isErrnoPtr(const uint8_t *p) {
  return reinterpret_cast<intptr_t>(p) < 0;
}

// VA the guest itself unmapped. sys_munmap keeps the host pages (see there), so
// a later mmap HINT at the same address would otherwise be refused and
// relocated -- which breaks any allocator that frees a probe mapping and then
// asks for an exact sub-range of it.
void noteGuestReleased(uint8_t *ptr, size_t size);
bool wasGuestReleased(uint8_t *ptr, size_t size);
// ...and VA that has been handed out again since. Direct-memory maps bypass the
// VMA (they mmap the shared store themselves), so without this a later hint
// could be honoured straight over a live one.
void noteGuestTaken(uint8_t *ptr, size_t size);

uint8_t *PS4ABI sys_mmap(void *addr, size_t size, uint32_t prot, uint32_t flags,
                         uint32_t fd, size_t offset);
int PS4ABI sys_mname(uint8_t *, size_t len, const char *name, void *);
int PS4ABI sys_mprotect(uint8_t *, size_t len, int prot);
int PS4ABI sys_mdbg_service(uint32_t op, void *, void *, void *);

/*POSIX shared memory*/
int PS4ABI sys_shm_open(const char *path, uint32_t flags, uint16_t mode);
int PS4ABI sys_shm_unlink(const char *path);
int PS4ABI sys_ftruncate(uint32_t fd, int64_t length);
// Size of a shm fd's backing for fstat (SIZE_MAX if fd isn't a shm).
size_t shmFstatSize(uint32_t fd);

/*direct memory access*/
int PS4ABI sys_dmem_container(uint32_t op);
}