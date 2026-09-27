#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace krnl {
class Proc;

// /dev/usbctl: the USB host-controller control channel. System-only; games
// reach input through pad/libkernel. Registers so an open succeeds; commands
// soft-succeed.
class UsbctlDevice : public Device {
 public:
  UsbctlDevice(ObjectTable& objects);

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace krnl
