#include <base.h>
#include <cstdio>
#include <cstring>

#include "av_control_dev.h"
#include "file_dev.h"

namespace krnl {
avControlDevice::avControlDevice(proc *p) : device(p) {}

int32_t avControlDevice::ioctl(uint32_t cmd, void *data) {
  // The full set (crtc/pll/dp/fmt/blnd/dvo) configures display hardware the
  // emulator fakes on the gc device. Soft-succeed, zeroing any OUT payload so
  // the caller reads a benign result instead of stale stack bytes.
  std::printf("[av_control] UNHANDLED ioctl(%#x)\n", cmd);
  if (data && (cmd & 0x40000000u)) {
    const uint32_t len = (cmd >> 16) & 0x1fff;
    if (len)
      std::memset(data, 0, len);
  }
  return 0;
}

int64_t avControlDevice::lseek(int64_t, int) { return 0; }

int avControlDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl
