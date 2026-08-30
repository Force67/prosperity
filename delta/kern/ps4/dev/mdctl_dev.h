#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/mdctl: the memory-disk controller. The kernel opens it only to attach
// ".md" root-filesystem images at boot; games never do. The emulator has no
// memory disks, so queries/detaches report "not found" and list is empty.
class mdctlDevice : public device {
public:
  mdctlDevice(objectTable &objects);

  i32 ioctl(u32 command, void *args) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
