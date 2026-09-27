#include <cstdio>
#include <cstring>
#include "base/arch.h"
#include "base/logging.h"

#include "kern/ps4/dev/authmgr_dev.h"
#include "kern/ps4/dev/file_dev.h"

namespace kern {
AuthmgrDevice::AuthmgrDevice(ObjectTable& objects) : Device(objects) {}

i32 AuthmgrDevice::Ioctl(u32 cmd, void* data) {
  if (!data)
    return 0;
  u8* args = static_cast<u8*>(data);

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

i64 AuthmgrDevice::Lseek(i64, int) {
  return 0;
}

int AuthmgrDevice::Fstat(void* stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  FillStat(*reinterpret_cast<SceKernelStat*>(stat), 0x2000, 0);
  return 0;
}
}  // namespace kern
