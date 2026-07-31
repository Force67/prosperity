#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "device.h"

namespace krnl {
class proc;

class dmaDevice : public device {
public:
  dmaDevice(proc *);

  int32_t ioctl(uint32_t command, void *args) override;
  uint8_t *map(void *, size_t, uint32_t, uint32_t, size_t) override;
  bool isDirectMemory() const override { return true; }

private:
  int32_t ioctlImpl(uint32_t command, void *args);
};

// Shared physical-dmem backing store (a memfd) so that every virtual address
// mapping the same physical offset aliases the same bytes -- the direct-memory
// coherency the AGC/GNM command buffers rely on (a CPU-written command buffer
// and the GPU's view of it map one physOffset at possibly-different VAs). The
// dmem physical-offset bump-allocator lives in the /dev/dmem ioctls; the mapper
// (syscall 628) commits VA -> backing_fd@physOffset via MAP_SHARED. Returns -1
// if the backing could not be created. See ps5-boot-progress memory.
int dmemBackingFd();
uint64_t dmemBackingSize();
}