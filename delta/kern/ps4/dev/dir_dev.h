#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"

#include "base/containers/vector.h"
#include "kern/ps4/dev/device.h"
#include "kern/vfs.h"

namespace kern {
// A directory opened by the guest (e.g. /app0/resources). Holds a snapshot of
// its immediate children and serves them through getdents; fstat reports it as
// a directory. Games open a dir then enumerate it to find their resources.
class DirDevice : public Device {
 public:
  DirDevice(ObjectTable& objects, base::Vector<vfs::DirEntry>&& entries);

  i64 Getdents(void* buf, size_t len) override;
  int Fstat(void* stat) override;
  i64 Read(void*, size_t) override;

 private:
  base::Vector<vfs::DirEntry> entries_;
  size_t cursor_ = 0;
};
}  // namespace kern
