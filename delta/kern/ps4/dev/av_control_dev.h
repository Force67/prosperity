#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/av_control: the A/V controller (crtc/pll/dp/fmt/blnd/dvo). System-only
// in the kernel; games reach it through the gc device. Registers so an open
// succeeds; unknown commands soft-succeed.
class avControlDevice : public device {
public:
  avControlDevice(proc *p);

  int32_t ioctl(uint32_t command, void *args) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
