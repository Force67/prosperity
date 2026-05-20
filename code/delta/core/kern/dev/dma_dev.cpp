
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "dma_dev.h"

namespace krnl {
dmaDevice::dmaDevice(proc *p) : device(p) {}

/* dmem_ioctl */
int32_t dmaDevice::ioctl(uint32_t cmd, void *data) {
  switch (cmd) {
  case 0x4008800A: {

    /*FIXME: validate _real_ dma size*/
    *static_cast<size_t *>(data) = 1024;
    return 0;
  }
  }

  return 0;
}

uint8_t *dmaDevice::map(void *addr, size_t, uint32_t, uint32_t, size_t) {
  //__debugbreak();
  return reinterpret_cast<uint8_t *>(-1);
}
} // namespace krnl
