#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/hid: the human-interface-device (keyboard/mouse) channel. System-only;
// games reach input through pad/libkernel. Registers so an open succeeds;
// commands soft-succeed.
class hidDevice : public device {
public:
  hidDevice(proc *p);

  int32_t ioctl(uint32_t command, void *args) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
