#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/usbctl: the USB host-controller control channel. System-only; games
// reach input through pad/libkernel. Registers so an open succeeds; commands
// soft-succeed.
class usbctlDevice : public device {
public:
  usbctlDevice(proc *p);

  int32_t ioctl(uint32_t command, void *args) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
