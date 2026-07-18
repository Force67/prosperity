#pragma once

// PS5 /dev/dmem device. The physical-offset bump-allocator ioctls are identical
// to PS4 (both platforms call sceKernelAllocateDirectMemory), so this inherits
// the shared dmaDevice; only the mapping diverges: PS5 backs each mapping with the
// shared physical-dmem memfd (MAP_SHARED) so a CPU-written GPU command buffer and
// the command processor's view of it alias one physOffset. PS4 has no such backing
// (its map falls back to the anonymous path).

#include "kern/ps4/dev/dma_dev.h"

namespace krnl {
class proc;

class dmaDevicePs5 : public dmaDevice {
public:
  dmaDevicePs5(proc *p) : dmaDevice(p) {}

  uint8_t *map(void *, size_t, uint32_t, uint32_t, size_t) override;
};
}  // namespace krnl
