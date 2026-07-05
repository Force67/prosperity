
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <cstdio>
#include <cstring>
#include "dipsw_dev.h"

namespace krnl {
dipswDevice::dipswDevice(proc *p) : device(p) {}

/* dipsw_dev_ioctl */
int32_t dipswDevice::ioctl(uint32_t cmd, void *data) {
  switch (cmd) {
  case 0x40048806: /*sceKernelCheckDipsw*/
    *static_cast<uint32_t *>(data) = 1;
    break;
  /* dont seem to be implemented ? */
  case 0x40048807:
  case 0x40088808:
  case 0x40088809:
    *static_cast<uint32_t *>(data) = 0;
    break;
  default:
    // Unknown dipsw command (e.g. PS5-era 0x40080002, an 8-byte IOC_OUT read).
    // Soft-succeed: zero-fill the OUT buffer by the ioctl size so the guest
    // reads a benign 0 instead of trapping.
    std::printf("[dipsw] UNHANDLED ioctl(%x) data=%p\n", cmd, data);
    if (data && (cmd & 0x40000000u)) {
      uint32_t len = (cmd >> 16) & 0x1fff;
      if (len)
        std::memset(data, 0, len);
    }
    break;
  }

  return 0;
}
} // namespace krnl
