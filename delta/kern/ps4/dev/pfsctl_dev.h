#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace krnl {
class Proc;

// /dev/pfsctldev: the PFS filesystem control channel (format/compact/backup).
// System-only; games never open it. Registers so an open succeeds; commands
// soft-succeed.
class PfsctlDevice : public Device {
 public:
  PfsctlDevice(ObjectTable& objects);

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace krnl
