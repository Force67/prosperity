#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <cstdio>
#include <cstring>

#include "file_dev.h"
#include "srtc_dev.h"

namespace krnl {
srtcDevice::srtcDevice(objectTable &objects) : device(objects) {}

i32 srtcDevice::ioctl(u32 cmd, void *data) {
  BASE_LOGI("srtc", "UNHANDLED ioctl({:#x})", cmd);
  if (data && (cmd & 0x40000000u)) {
    const u32 len = (cmd >> 16) & 0x1fff;
    if (len)
      std::memset(data, 0, len);
  }
  return 0;
}

i64 srtcDevice::lseek(i64, int) { return 0; }

int srtcDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl
