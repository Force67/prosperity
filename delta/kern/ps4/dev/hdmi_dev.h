#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {
class Process;

// /dev/hdmi: the HDMI output controller (EDID/HDCP link state). System-only;
// games negotiate display through the gc device. Registers so an open
// succeeds; commands soft-succeed.
class HdmiDevice : public Device {
 public:
  HdmiDevice(ObjectTable& objects);

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace kern
