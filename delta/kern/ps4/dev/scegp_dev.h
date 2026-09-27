#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {
class Process;

// /dev/sceGp: the general-purpose accelerator (GPU) media queue. System-only;
// games reach the GPU through the gc device. Registers so an open succeeds;
// commands soft-succeed.
class SceGpDevice : public Device {
 public:
  SceGpDevice(ObjectTable& objects);

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace kern
