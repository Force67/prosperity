#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/hdmi: the HDMI output controller (EDID/HDCP link state). System-only;
// games negotiate display through the gc device. Registers so an open
// succeeds; commands soft-succeed.
class hdmiDevice : public device {
public:
  hdmiDevice(proc *p);

  int32_t ioctl(uint32_t command, void *args) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
