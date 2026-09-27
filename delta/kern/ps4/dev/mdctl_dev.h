#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {
class Process;

// /dev/mdctl: the memory-disk controller. The kernel opens it only to attach
// ".md" root-filesystem images at boot; games never do. The emulator has no
// memory disks, so queries/detaches report "not found" and list is empty.
class MdctlDevice : public Device {
 public:
  MdctlDevice(ObjectTable& objects);

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace kern
