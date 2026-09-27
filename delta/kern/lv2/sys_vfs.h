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

namespace krnl {
enum FcFlags {
  // FreeBSD open(2) flag spelling.
  // NOLINTBEGIN(readability-identifier-naming)
  /*open only*/
  O_RDONLY,
  O_WRONLY,
  O_RDWR,
  O_ACCMODE,

  // FreeBSD/Orbis open() flag bits.
  O_APPEND = 0x00000008,
  O_CREAT = 0x00000200,
  O_TRUNC = 0x00000400,
  O_EXCL = 0x00000800,

  O_EXEC = 0x00040000,
  O_DIRECTORY = 0x00020000,
  // NOLINTEND(readability-identifier-naming)
};

int PS4ABI sys_open(const char* path, u32 flags, u32 mode);
int PS4ABI sys_close(u32 fd);
i64 PS4ABI sys_read(u32 fd, void* buf, size_t nbytes);
void FdReadStat(u32 fd, i64 n);
i64 PS4ABI sys_lseek(u32 fd, i64 offset, int whence);
int PS4ABI sys_fstat(u32 fd, void* stat);
int PS4ABI sys_stat(const char* path, void* stat);
int PS4ABI sys_statfs(const char* path, void* buf);
int PS4ABI sys_fstatfs(u32 fd, void* buf);
i64 PS4ABI sys_getdents(u32 fd, void* buf, size_t nbytes);
}  // namespace krnl