#include <base.h>
#include "base/arch.h"
#include <cstring>

#include "file_dev.h"
#include "null_dev.h"

namespace krnl {
nullDevice::nullDevice(objectTable &objects) : device(objects) {}

i64 nullDevice::read(void *, size_t) { return 0; }
i64 nullDevice::write(const void *, size_t n) {
  return static_cast<i64>(n);
}
i64 nullDevice::lseek(i64, int) { return 0; }

int nullDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl