#include <cstring>
#include "base/arch.h"

#include "kern/ps4/dev/file_dev.h"
#include "kern/ps4/dev/zero_dev.h"

namespace kern {
ZeroDevice::ZeroDevice(ObjectTable& objects) : Device(objects) {}

i64 ZeroDevice::Read(void* buf, size_t len) {
  if (!buf)
    return -SysError::eFAULT;
  std::memset(buf, 0, len);
  return static_cast<i64>(len);
}
i64 ZeroDevice::Write(const void*, size_t n) {
  return static_cast<i64>(n);
}
i64 ZeroDevice::Lseek(i64, int) {
  return 0;
}

int ZeroDevice::Fstat(void* stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  FillStat(*reinterpret_cast<SceKernelStat*>(stat), 0x2000, 0);
  return 0;
}
}  // namespace kern