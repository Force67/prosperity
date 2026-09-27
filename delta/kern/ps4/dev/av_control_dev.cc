#include <cstdio>
#include <cstring>
#include "base/arch.h"
#include "base/logging.h"

#include "kern/ps4/dev/av_control_dev.h"
#include "kern/ps4/dev/file_dev.h"

namespace krnl {
AvControlDevice::AvControlDevice(ObjectTable& objects) : Device(objects) {}

i32 AvControlDevice::Ioctl(u32 cmd, void* data) {
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

i64 AvControlDevice::Lseek(i64, int) {
  return 0;
}

int AvControlDevice::Fstat(void* stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  FillStat(*reinterpret_cast<SceKernelStat*>(stat), 0x2000, 0);
  return 0;
}
}  // namespace krnl
