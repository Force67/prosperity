#include <base.h>
#include <cstring>

#include "file_dev.h"
#include "null_dev.h"

namespace krnl {
nullDevice::nullDevice(proc *p) : device(p) {}

int64_t nullDevice::read(void *, size_t) { return 0; }
int64_t nullDevice::write(const void *, size_t n) {
  return static_cast<int64_t>(n);
}
int64_t nullDevice::lseek(int64_t, int) { return 0; }

int nullDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl