#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {
class Process;

// /dev/hid: the human-interface-device channel. The real device is system-only
// and games reach input through pad/libkernel, so by default commands
// soft-succeed. With DELTA_HID_PASSTHROUGH=1 the guest read commands pull the
// host's evdev state and the write commands drive a uinput device instead.
class HidDevice : public Device {
 public:
  HidDevice(ObjectTable& objects);

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace kern
