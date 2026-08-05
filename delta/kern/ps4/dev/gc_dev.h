#pragma once

// Copyright (C) Force67 2019

#include "device.h"

#include <array>
#include <mutex>

namespace krnl {
class proc;

class gcDevice : public device {
public:
  gcDevice(proc *);

  bool init(const char *, uint32_t, uint32_t) override;
  int32_t ioctl(uint32_t command, void *args) override;
  uint8_t *map(void *, size_t, uint32_t, uint32_t, size_t) override;

  // Kernel /dev/gc maps GPU-visible memory at a fixed base+offset; mirror that
  // with a lazily-allocated pool (identity range, CP-renderable).
  uint8_t *poolBase = nullptr;
  uint64_t poolSize = 0;

private:
  struct ComputeQueue {
    uint32_t me = 0;
    uint32_t pipe = 0;
    uint32_t queue = 0;
    uint32_t vqueue = 0;
    uint64_t ringBase = 0;
    uint64_t readPtr = 0;
    uint64_t state = 0;
    uint32_t ringSizeDw = 0;
    uint32_t readOffsetDw = 0;
    bool mapped = false;
  };

  std::array<ComputeQueue, 64> computeQueues{};
  std::mutex computeMutex;
};
}
