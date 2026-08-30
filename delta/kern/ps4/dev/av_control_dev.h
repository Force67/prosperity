#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/av_control: the A/V controller (crtc/pll/dp/fmt/blnd/dvo). System-only
// in the kernel; games reach it through the gc device. Registers so an open
// succeeds; unknown commands soft-succeed.
class avControlDevice : public device {
public:
  avControlDevice(objectTable &objects);

  i32 ioctl(u32 command, void *args) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
