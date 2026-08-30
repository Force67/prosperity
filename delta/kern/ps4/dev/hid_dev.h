#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/hid: the human-interface-device channel. The real device is system-only
// and games reach input through pad/libkernel, so by default commands
// soft-succeed. With DELTA_HID_PASSTHROUGH=1 the guest read commands pull the
// host's evdev state and the write commands drive a uinput device instead.
class hidDevice : public device {
public:
  hidDevice(objectTable &objects);

  i32 ioctl(u32 command, void *args) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
