#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {
class Process;

// /dev/vtrm: the secure VM/trusted-runner interface (system ucred only).
// Games never open it. Registers so an open succeeds; commands soft-succeed.
class VtrmDevice : public Device {
 public:
  VtrmDevice(ObjectTable& objects);

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace kern
