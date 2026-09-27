#pragma once

// Copyright (C) Force67 2019

#include "base/arch.h"
#include "base/containers/array.h"
#include "base/threading/mutex.h"
#include "kern/ps4/dev/device.h"

namespace kern {
class Process;

class GcDevice : public Device {
 public:
  GcDevice(ObjectTable&);

  bool Init(const char*, u32, u32) override;
  i32 Ioctl(u32 command, void* args) override;
  u8* Map(void*, size_t, u32, u32, size_t) override;

  // Kernel /dev/gc maps GPU-visible memory at a fixed base+offset; mirror that
  // with a lazily-allocated pool (identity range, CP-renderable).
  u8* pool_base = nullptr;
  u64 pool_size = 0;

  // Execute the complete indirect-buffer packets sitting in every mapped
  // compute queue, up to `budget_dw` dwords per queue. Static because the
  // queues are hardware state and the per-frame drain runs outside any ioctl.
  // The CALLER must hold g_compute_mutex (the doorbell handler already does).
  static void DrainQueues(u32 budget_dw);
  // A doorbell ring: execute queue `ringId` up to the write pointer the guest
  // just published. See prosperity_gc_dingdong.
  static void RingDoorbell(u32 ring_id, u32 write_offset_dw);

  struct ComputeQueue {
    u32 me = 0;
    u32 pipe = 0;
    u32 queue = 0;
    u32 vqueue = 0;
    u64 ring_base = 0;
    u64 read_ptr = 0;
    u64 state = 0;
    u32 ring_size_dw = 0;
    u32 read_offset_dw = 0;
    bool mapped = false;
  };

  // The mapped compute queues are hardware, not per-descriptor state: sys_open
  // news a GcDevice per open, and a queue mapped through one fd has to be
  // visible to anything that drains it (including the per-frame drain, which
  // runs outside any ioctl). Shared for the same reason the /dev/gc mapping
  // pool is.
  static base::Array<ComputeQueue, 64> g_compute_queues;
  static base::Mutex g_compute_mutex;
};
}  // namespace kern
