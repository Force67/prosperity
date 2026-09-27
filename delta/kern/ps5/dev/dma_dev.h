#pragma once

// PS5 /dev/dmem device. The physical-offset bump-allocator ioctls are identical
// to PS4 (both platforms call sceKernelAllocateDirectMemory), so this inherits
// the shared DmaDevice; only the mapping diverges: PS5 backs each mapping with
// the shared physical-dmem memfd (MAP_SHARED) so a CPU-written GPU command
// buffer and the command processor's view of it alias one physOffset. PS4 has
// no such backing (its map falls back to the anonymous path).

#include "base/arch.h"
#include "kern/ps4/dev/dma_dev.h"

namespace krnl {
class Proc;

// Forget the direct-memory mapping recorded at this VA range. sys_munmap calls
// it so an explicit unmap-then-map reads as a fresh allocation, while a map
// straight onto a live mapping's base stays a re-point and keeps its contents.
void ForgetDmemVa(u8* ptr, size_t size);

class DmaDevicePs5 : public DmaDevice {
 public:
  DmaDevicePs5(ObjectTable& objects) : DmaDevice(objects) {}

  u8* Map(void*, size_t, u32, u32, size_t) override;
};
}  // namespace krnl
