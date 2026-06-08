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
uint8_t *allocLowGuest(size_t size);

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