#include <cstdio>
#include <cstring>
#include "base/arch.h"
#include "base/logging.h"

#include "kern/ps4/dev/file_dev.h"
#include "kern/ps4/dev/usbctl_dev.h"

namespace krnl {
UsbctlDevice::UsbctlDevice(ObjectTable& objects) : Device(objects) {}

i32 UsbctlDevice::Ioctl(u32 cmd, void* data) {
  BASE_LOGI("usbctl", "UNHANDLED ioctl({:#x})", cmd);
  if (data && (cmd & 0x40000000u)) {
    const u32 len = (cmd >> 16) & 0x1fff;
    if (len)
      std::memset(data, 0, len);
  }
  return 0;
}

i64 UsbctlDevice::Lseek(i64, int) {
  return 0;
}

int UsbctlDevice::Fstat(void* stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  FillStat(*reinterpret_cast<SceKernelStat*>(stat), 0x2000, 0);
  return 0;
}
}  // namespace krnl
