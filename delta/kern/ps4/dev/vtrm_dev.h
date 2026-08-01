#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/vtrm: the secure VM/trusted-runner interface (system ucred only).
// Games never open it. Registers so an open succeeds; commands soft-succeed.
class vtrmDevice : public device {
public:
  vtrmDevice(proc *p);

  int32_t ioctl(uint32_t command, void *args) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
