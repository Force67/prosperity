#include <cstdio>
#include <cstring>
#include "base/arch.h"
#include "base/logging.h"

#include "kern/ps4/dev/file_dev.h"
#include "kern/ps4/dev/hdmi_dev.h"

namespace kern {
HdmiDevice::HdmiDevice(ObjectTable& objects) : Device(objects) {}

i32 HdmiDevice::Ioctl(u32 cmd, void* data) {
  BASE_LOGI("hdmi", "UNHANDLED ioctl({:#x})", cmd);
  if (data && (cmd & 0x40000000u)) {
    const u32 len = (cmd >> 16) & 0x1fff;
    if (len)
      std::memset(data, 0, len);
  }
  return 0;
}

i64 HdmiDevice::Lseek(i64, int) {
  return 0;
}

int HdmiDevice::Fstat(void* stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  FillStat(*reinterpret_cast<SceKernelStat*>(stat), 0x2000, 0);
  return 0;
}
}  // namespace kern
