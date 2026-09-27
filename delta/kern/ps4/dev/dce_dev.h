#pragma once

// Copyright (C) Force67 2019

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {
class Process;

u32 DceCurrentBuffer();
i64 DceCurrentFlipArg();
u64 DceScanoutBuffer(u32 index);

// /dev/dce: the Display Control Engine (framebuffer scanout). The real
// libSceVideoOut.sprx (LLE) opens it and drives the whole display pipe through
// it: a multiplexed control ioctl (0xc0308203, sub-op in arg[0]),
// scanout-buffer registration (0xc0308207) and flip submit (0xc0488204). We
// emulate the client contract the 11.00 module relies on: hand back a display
// handle, a real mmap- able scanout pool, and flip completion, so the module
// runs unmodified.
class DceDevice : public Device {
 public:
  DceDevice(ObjectTable&);

  bool Init(const char*, u32, u32) override;
  i32 Ioctl(u32 command, void* args) override;
  // The module mmaps this fd at the offset sub-op 9 handed back to get its
  // scanout/control pool; return the matching slice of our pool.
  u8* Map(void*, size_t, u32, u32, size_t) override;

 private:
  // One contiguous guest-addressable pool, bump-allocated by sub-op 9 (and the
  // 0xc0588212 variant). map(offset) returns poolBase + offset.
  u8* pool_base_ = nullptr;
  u64 pool_size_ = 0;
  u64 pool_used_ = 0;
  u64 next_handle_ = 1;  // opaque display handle handed out by sub-op 0

  // Allocate `bytes` from the pool (lazily creating it); returns the offset, or
  // UINT64_MAX on failure.
  u64 PoolAlloc(u64 bytes);
};
}  // namespace kern
