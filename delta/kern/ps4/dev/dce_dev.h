#pragma once

// Copyright (C) Force67 2019

#include "device.h"

namespace krnl {
class proc;

// /dev/dce: the Display Control Engine (framebuffer scanout). The real
// libSceVideoOut.sprx (LLE) opens it and drives the whole display pipe through
// it: a multiplexed control ioctl (0xc0308203, sub-op in arg[0]), scanout-buffer
// registration (0xc0308207) and flip submit (0xc0488204). We emulate the client
// contract the 11.00 module relies on: hand back a display handle, a real mmap-
// able scanout pool, and flip completion, so the module runs unmodified.
class dceDevice : public device {
public:
  dceDevice(proc *);

  bool init(const char *, uint32_t, uint32_t) override;
  int32_t ioctl(uint32_t command, void *args) override;
  // The module mmaps this fd at the offset sub-op 9 handed back to get its
  // scanout/control pool; return the matching slice of our pool.
  uint8_t *map(void *, size_t, uint32_t, uint32_t, size_t) override;

private:
  // One contiguous guest-addressable pool, bump-allocated by sub-op 9 (and the
  // 0xc0588212 variant). map(offset) returns poolBase + offset.
  uint8_t *poolBase = nullptr;
  uint64_t poolSize = 0;
  uint64_t poolUsed = 0;
  uint64_t nextHandle = 1;  // opaque display handle handed out by sub-op 0

  // Allocate `bytes` from the pool (lazily creating it); returns the offset, or
  // UINT64_MAX on failure.
  uint64_t poolAlloc(uint64_t bytes);
};
}  // namespace krnl
