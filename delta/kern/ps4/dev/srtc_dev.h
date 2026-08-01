#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/srtc: the secure real-time clock. System-only (ShellUI/Diag); games read
// time through clock_gettime/gettimeofday. Registers so an open succeeds;
// commands soft-succeed.
class srtcDevice : public device {
public:
  srtcDevice(proc *p);

  int32_t ioctl(uint32_t command, void *args) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
