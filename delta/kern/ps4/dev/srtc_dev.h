#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/srtc: the secure real-time clock. System-only (ShellUI/Diag); games read
// time through clock_gettime/gettimeofday. Registers so an open succeeds;
// commands soft-succeed.
class srtcDevice : public device {
public:
  srtcDevice(objectTable &objects);

  i32 ioctl(u32 command, void *args) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
