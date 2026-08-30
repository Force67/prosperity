#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/vtrm: the secure VM/trusted-runner interface (system ucred only).
// Games never open it. Registers so an open succeeds; commands soft-succeed.
class vtrmDevice : public device {
public:
  vtrmDevice(objectTable &objects);

  i32 ioctl(u32 command, void *args) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
