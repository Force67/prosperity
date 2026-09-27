#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace krnl {
class Proc;

class DmaDevice : public Device {
 public:
  DmaDevice(ObjectTable&);

  i32 Ioctl(u32 command, void* args) override;
  u8* Map(void*, size_t, u32, u32, size_t) override;
  bool IsDirectMemory() const override { return true; }

 private:
  i32 IoctlImpl(u32 command, void* args);
};

// Shared physical-dmem backing store (a memfd) so that every virtual address
// mapping the same physical offset aliases the same bytes, the direct-memory
// coherency the AGC/GNM command buffers rely on (a CPU-written command buffer
// and the GPU's view of it map one physOffset at possibly-different VAs). The
// dmem physical-offset bump-allocator lives in the /dev/dmem ioctls; the mapper
// (syscall 628) commits VA -> backing_fd@physOffset via MAP_SHARED. Returns -1
// if the backing could not be created. See ps5-boot-progress memory.
int DmemBackingFd();

// Memory type (SCE_KERNEL_WB_ONION / WC_GARLIC / ...) of the direct-memory
// reservation covering a physical offset, or -1 when none does.
int DmemTypeForOffset(u64 off);
u64 DmemBackingSize();
}  // namespace krnl