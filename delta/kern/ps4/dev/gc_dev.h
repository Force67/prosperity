#pragma once

// Copyright (C) Force67 2019

#include "device.h"
#include "base/arch.h"

#include <array>
#include <mutex>

namespace krnl {
class proc;

class gcDevice : public device {
public:
  gcDevice(objectTable &);

  bool init(const char *, u32, u32) override;
  i32 ioctl(u32 command, void *args) override;
  u8 *map(void *, size_t, u32, u32, size_t) override;

  // Kernel /dev/gc maps GPU-visible memory at a fixed base+offset; mirror that
  // with a lazily-allocated pool (identity range, CP-renderable).
  u8 *poolBase = nullptr;
  u64 poolSize = 0;

  // Execute the complete indirect-buffer packets sitting in every mapped
  // compute queue, up to `budget_dw` dwords per queue. Static because the
  // queues are hardware state and the per-frame drain runs outside any ioctl.
  // The CALLER must hold computeMutex (the doorbell handler already does).
  static void drainQueues(u32 budget_dw);
  // A doorbell ring: execute queue `ringId` up to the write pointer the guest
  // just published. See prosperity_gc_dingdong.
  static void ringDoorbell(u32 ringId, u32 writeOffsetDw);

  struct ComputeQueue {
    u32 me = 0;
    u32 pipe = 0;
    u32 queue = 0;
    u32 vqueue = 0;
    u64 ringBase = 0;
    u64 readPtr = 0;
    u64 state = 0;
    u32 ringSizeDw = 0;
    u32 readOffsetDw = 0;
    bool mapped = false;
  };

  // The mapped compute queues are hardware, not per-descriptor state: sys_open
  // news a gcDevice per open, and a queue mapped through one fd has to be
  // visible to anything that drains it (including the per-frame drain, which
  // runs outside any ioctl). Shared for the same reason the /dev/gc mapping
  // pool is.
  static std::array<ComputeQueue, 64> computeQueues;
  static std::mutex computeMutex;
};
}
