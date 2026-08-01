#include <base.h>

#include "deci_stdin_dev.h"
#include "file_dev.h"

namespace krnl {
deciStdinDevice::deciStdinDevice(proc *p) : device(p) {}

int64_t deciStdinDevice::read(void *, size_t) { return 0; }
int64_t deciStdinDevice::write(const void *, size_t n) {
  return static_cast<int64_t>(n);
}
int64_t deciStdinDevice::lseek(int64_t, int) { return 0; }

int deciStdinDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}
} // namespace krnl
