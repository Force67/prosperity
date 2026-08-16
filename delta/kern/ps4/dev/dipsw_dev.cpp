
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <cstdio>
#include <cstring>
#include "dipsw_dev.h"

namespace krnl {
dipswDevice::dipswDevice(proc *p) : device(p) {}

/* dipsw_dev_ioctl */
i32 dipswDevice::ioctl(u32 cmd, void *data) {
  switch (cmd) {
  case 0x40048806: /*sceKernelCheckDipsw*/
    *static_cast<u32 *>(data) = 1;
    break;
  case 0x8010880a:
    // A 16-byte IOC_IN pair the PS5 libkernel writes while it caches the dip
    // switches; there is no reply and nothing of ours reads them back.
    break;
  /* dont seem to be implemented ? */
  case 0x40048807:
  case 0x40088808:
  case 0x40088809:
    *static_cast<u32 *>(data) = 0;
    break;
  default:
    // Unknown dipsw command (e.g. PS5-era 0x40080002, an 8-byte IOC_OUT read).
    // Soft-succeed: zero-fill the OUT buffer by the ioctl size so the guest
    // reads a benign 0 instead of trapping.
    BASE_LOGI("dipsw", "UNHANDLED ioctl({:x}) data={:p}", cmd, data);
    if (data && (cmd & 0x40000000u)) {
      u32 len = (cmd >> 16) & 0x1fff;
      if (len)
        std::memset(data, 0, len);
    }
    break;
  }

  return 0;
}
} // namespace krnl
