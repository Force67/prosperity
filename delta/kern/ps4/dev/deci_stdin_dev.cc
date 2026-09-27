#include "base/arch.h"

#include "kern/ps4/dev/deci_stdin_dev.h"
#include "kern/ps4/dev/file_dev.h"

namespace krnl {
DeciStdinDevice::DeciStdinDevice(ObjectTable& objects) : Device(objects) {}

i64 DeciStdinDevice::Read(void*, size_t) {
  return 0;
}
i64 DeciStdinDevice::Write(const void*, size_t n) {
  return static_cast<i64>(n);
}
i64 DeciStdinDevice::Lseek(i64, int) {
  return 0;
}

int DeciStdinDevice::Fstat(void* stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  FillStat(*reinterpret_cast<SceKernelStat*>(stat), 0x2000, 0);
  return 0;
}
}  // namespace krnl
