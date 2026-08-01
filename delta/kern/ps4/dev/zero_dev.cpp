#include <base.h>
#include <cstring>

#include "file_dev.h"
#include "zero_dev.h"

namespace krnl {
zeroDevice::zeroDevice(proc *p) : device(p) {}

int64_t zeroDevice::read(void *buf, size_t len) {
  if (!buf)
    return -SysError::eFAULT;
  std::memset(buf, 0, len);
  return static_cast<int64_t>(len);
}
int64_t zeroDevice::write(const void *, size_t n) {
  return static_cast<int64_t>(n);
}
int64_t zeroDevice::lseek(int64_t, int) { return 0; }

int zeroDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl