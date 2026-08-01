#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/mdctl: the memory-disk controller. The kernel opens it only to attach
// ".md" root-filesystem images at boot; games never do. The emulator has no
// memory disks, so queries/detaches report "not found" and list is empty.
class mdctlDevice : public device {
public:
  mdctlDevice(proc *p);

  int32_t ioctl(uint32_t command, void *args) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
