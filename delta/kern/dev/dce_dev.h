#pragma once

// Copyright (C) Force67 2019

#include "device.h"

namespace krnl {
class proc;

// /dev/dce: the Display Control Engine (framebuffer scanout). libSceVideoOut
// opens it and ioctls it for display/buffer info. We don't drive a real GPU
// yet; the point is to return *defined* output so the guest doesn't size
// allocations from uninitialized stack data.
class dceDevice : public device {
public:
  dceDevice(proc *);

  bool init(const char *, uint32_t, uint32_t) override;
  int32_t ioctl(uint32_t command, void *args) override;
};
}  // namespace krnl
