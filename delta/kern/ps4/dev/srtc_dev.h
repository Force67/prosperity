#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {
class Process;

// /dev/srtc: the secure real-time clock. System-only (ShellUI/Diag); games read
// time through clock_gettime/gettimeofday. Registers so an open succeeds;
// commands soft-succeed.
class SrtcDevice : public Device {
 public:
  SrtcDevice(ObjectTable& objects);

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace kern
