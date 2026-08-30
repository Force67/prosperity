#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <cstdio>
#include <cstring>

#include "authmgr_dev.h"
#include "file_dev.h"

namespace krnl {
authmgrDevice::authmgrDevice(objectTable &objects) : device(objects) {}

i32 authmgrDevice::ioctl(u32 cmd, void *data) {
  if (!data)
    return 0;
  u8 *args = static_cast<u8 *>(data);

  switch (cmd) {
  default:
    BASE_LOGI("authmgr", "UNHANDLED ioctl({:#x})", cmd);
    if (cmd & 0x40000000u) {
      const u32 len = (cmd >> 16) & 0x1fff;
      if (len)
        std::memset(args, 0, len);
    }
    break;
  }

  return 0;
}

i64 authmgrDevice::lseek(i64, int) { return 0; }

int authmgrDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl
