#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/pfsctldev: the PFS filesystem control channel (format/compact/backup).
// System-only; games never open it. Registers so an open succeeds; commands
// soft-succeed.
class pfsctlDevice : public device {
public:
  pfsctlDevice(proc *p);

  int32_t ioctl(uint32_t command, void *args) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
