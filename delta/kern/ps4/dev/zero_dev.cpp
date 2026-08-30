#include <base.h>
#include "base/arch.h"
#include <cstring>

#include "file_dev.h"
#include "zero_dev.h"

namespace krnl {
zeroDevice::zeroDevice(objectTable &objects) : device(objects) {}

i64 zeroDevice::read(void *buf, size_t len) {
  if (!buf)
    return -SysError::eFAULT;
  std::memset(buf, 0, len);
  return static_cast<i64>(len);
}
i64 zeroDevice::write(const void *, size_t n) {
  return static_cast<i64>(n);
}
i64 zeroDevice::lseek(i64, int) { return 0; }

int zeroDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl