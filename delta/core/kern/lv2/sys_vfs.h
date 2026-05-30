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
enum fcFlags {
  /*open only*/
  O_RDONLY,
  O_WRONLY,
  O_RDWR,
  O_ACCMODE,

  O_EXEC = 0x00040000,
};

int PS4ABI sys_open(const char *path, uint32_t flags, uint32_t mode);
int PS4ABI sys_close(uint32_t fd);
} // namespace krnl