#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace krnl {
class Proc;

// /dev/av_control: the A/V controller (crtc/pll/dp/fmt/blnd/dvo). System-only
// in the kernel; games reach it through the gc device. Registers so an open
// succeeds; unknown commands soft-succeed.
class AvControlDevice : public Device {
 public:
  AvControlDevice(ObjectTable& objects);

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace krnl
