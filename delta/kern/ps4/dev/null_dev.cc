#include <cstring>
#include "base/arch.h"

#include "kern/ps4/dev/file_dev.h"
#include "kern/ps4/dev/null_dev.h"

namespace krnl {
NullDevice::NullDevice(ObjectTable& objects) : Device(objects) {}

i64 NullDevice::Read(void*, size_t) {
  return 0;
}
i64 NullDevice::Write(const void*, size_t n) {
  return static_cast<i64>(n);
}
i64 NullDevice::Lseek(i64, int) {
  return 0;
}

int NullDevice::Fstat(void* stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  FillStat(*reinterpret_cast<SceKernelStat*>(stat), 0x2000, 0);
  return 0;
}
}  // namespace krnl