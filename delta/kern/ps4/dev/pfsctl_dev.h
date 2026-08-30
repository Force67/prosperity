#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/pfsctldev: the PFS filesystem control channel (format/compact/backup).
// System-only; games never open it. Registers so an open succeeds; commands
// soft-succeed.
class pfsctlDevice : public device {
public:
  pfsctlDevice(objectTable &objects);

  i32 ioctl(u32 command, void *args) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
