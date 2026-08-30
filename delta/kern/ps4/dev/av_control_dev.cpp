#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <cstdio>
#include <cstring>

#include "av_control_dev.h"
#include "file_dev.h"

namespace krnl {
avControlDevice::avControlDevice(objectTable &objects) : device(objects) {}

i32 avControlDevice::ioctl(u32 cmd, void *data) {
  // The full set (crtc/pll/dp/fmt/blnd/dvo) configures display hardware the
  // emulator fakes on the gc device. Soft-succeed, zeroing any OUT payload so
  // the caller reads a benign result instead of stale stack bytes.
  BASE_LOGI("av_control", "UNHANDLED ioctl({:#x})", cmd);
  if (data && (cmd & 0x40000000u)) {
    const u32 len = (cmd >> 16) & 0x1fff;
    if (len)
      std::memset(data, 0, len);
  }
  return 0;
}

i64 avControlDevice::lseek(i64, int) { return 0; }

int avControlDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl
