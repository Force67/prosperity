#include <base.h>
#include <cstdio>
#include <cstring>

#include "file_dev.h"
#include "scegp_dev.h"

namespace krnl {
sceGpDevice::sceGpDevice(proc *p) : device(p) {}

int32_t sceGpDevice::ioctl(uint32_t cmd, void *data) {
  std::printf("[sceGp] UNHANDLED ioctl(%#x)\n", cmd);
  if (data && (cmd & 0x40000000u)) {
    const uint32_t len = (cmd >> 16) & 0x1fff;
    if (len)
      std::memset(data, 0, len);
  }
  return 0;
}

int64_t sceGpDevice::lseek(int64_t, int) { return 0; }

int sceGpDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl
