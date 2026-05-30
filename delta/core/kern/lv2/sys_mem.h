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
enum mprotFlags : uint32_t { none, exec = 1, write = 2, read = 4 };

enum mFlags : uint32_t {
  fixed = 0x10,
  stack = 0x400,
  noextend = 0x100,
  anon = 0x1000,
};

uint8_t *PS4ABI sys_mmap(void *addr, size_t size, uint32_t prot, uint32_t flags,
                         uint32_t fd, size_t offset);
int PS4ABI sys_mname(uint8_t *, size_t len, const char *name, void *);
int PS4ABI sys_mprotect(uint8_t *, size_t len, int prot);
int PS4ABI sys_mdbg_service(uint32_t op, void *, void *, void *);

/*direct memory access*/
int PS4ABI sys_dmem_container(uint32_t op);
}