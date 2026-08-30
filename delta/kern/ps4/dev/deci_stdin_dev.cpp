#include <base.h>
#include "base/arch.h"

#include "deci_stdin_dev.h"
#include "file_dev.h"

namespace krnl {
deciStdinDevice::deciStdinDevice(objectTable &objects) : device(objects) {}

i64 deciStdinDevice::read(void *, size_t) { return 0; }
i64 deciStdinDevice::write(const void *, size_t n) {
  return static_cast<i64>(n);
}
i64 deciStdinDevice::lseek(i64, int) { return 0; }

int deciStdinDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl
