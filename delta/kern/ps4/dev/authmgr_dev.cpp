#include <base.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "authmgr_dev.h"
#include "file_dev.h"

namespace krnl {
authmgrDevice::authmgrDevice(proc *p) : device(p) {}

int32_t authmgrDevice::ioctl(uint32_t cmd, void *data) {
  if (!data)
    return 0;
  uint8_t *args = static_cast<uint8_t *>(data);

  switch (cmd) {
  default:
    std::printf("[authmgr] UNHANDLED ioctl(%#x)\n", cmd);
    if (cmd & 0x40000000u) {
      const uint32_t len = (cmd >> 16) & 0x1fff;
      if (len)
        std::memset(args, 0, len);
    }
    break;
  }

  return 0;
}

int64_t authmgrDevice::lseek(int64_t, int) { return 0; }

int authmgrDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl
