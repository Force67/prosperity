#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <vector>

#include "kern/vfs.h"
#include "device.h"

namespace krnl {
// A directory opened by the guest (e.g. /app0/resources). Holds a snapshot of
// its immediate children and serves them through getdents; fstat reports it as
// a directory. Games open a dir then enumerate it to find their resources.
class dirDevice : public device {
public:
  dirDevice(proc *p, std::vector<vfs::DirEntry> &&entries);

  int64_t getdents(void *buf, size_t len) override;
  int fstat(void *stat) override;
  int64_t read(void *, size_t) override;

private:
  std::vector<vfs::DirEntry> entries_;
  size_t cursor_ = 0;
};
}  // namespace krnl
