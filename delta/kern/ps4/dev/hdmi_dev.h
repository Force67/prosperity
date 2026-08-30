#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/hdmi: the HDMI output controller (EDID/HDCP link state). System-only;
// games negotiate display through the gc device. Registers so an open
// succeeds; commands soft-succeed.
class hdmiDevice : public device {
public:
  hdmiDevice(objectTable &objects);

  i32 ioctl(u32 command, void *args) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
