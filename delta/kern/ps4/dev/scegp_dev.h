#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/sceGp: the general-purpose accelerator (GPU) media queue. System-only;
// games reach the GPU through the gc device. Registers so an open succeeds;
// commands soft-succeed.
class sceGpDevice : public device {
public:
  sceGpDevice(proc *p);

  int32_t ioctl(uint32_t command, void *args) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
