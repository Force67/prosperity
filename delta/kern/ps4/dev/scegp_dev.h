#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/sceGp: the general-purpose accelerator (GPU) media queue. System-only;
// games reach the GPU through the gc device. Registers so an open succeeds;
// commands soft-succeed.
class sceGpDevice : public device {
public:
  sceGpDevice(objectTable &objects);

  i32 ioctl(u32 command, void *args) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
