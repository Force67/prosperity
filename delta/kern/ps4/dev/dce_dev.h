#pragma once

// Copyright (C) Force67 2019

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

u32 dceCurrentBuffer();
i64 dceCurrentFlipArg();
u64 dceScanoutBuffer(u32 index);

// /dev/dce: the Display Control Engine (framebuffer scanout). The real
// libSceVideoOut.sprx (LLE) opens it and drives the whole display pipe through
// it: a multiplexed control ioctl (0xc0308203, sub-op in arg[0]), scanout-buffer
// registration (0xc0308207) and flip submit (0xc0488204). We emulate the client
// contract the 11.00 module relies on: hand back a display handle, a real mmap-
// able scanout pool, and flip completion, so the module runs unmodified.
class dceDevice : public device {
public:
  dceDevice(objectTable &);

  bool init(const char *, u32, u32) override;
  i32 ioctl(u32 command, void *args) override;
  // The module mmaps this fd at the offset sub-op 9 handed back to get its
  // scanout/control pool; return the matching slice of our pool.
  u8 *map(void *, size_t, u32, u32, size_t) override;

private:
  // One contiguous guest-addressable pool, bump-allocated by sub-op 9 (and the
  // 0xc0588212 variant). map(offset) returns poolBase + offset.
  u8 *poolBase = nullptr;
  u64 poolSize = 0;
  u64 poolUsed = 0;
  u64 nextHandle = 1;  // opaque display handle handed out by sub-op 0

  // Allocate `bytes` from the pool (lazily creating it); returns the offset, or
  // UINT64_MAX on failure.
  u64 poolAlloc(u64 bytes);
};
}  // namespace krnl
