#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace krnl {
class Proc;

// /dev/npdrm: the NPDRM license/decryption manager. System-only; games reach
// license checks through the SBL libs. Registers so an open succeeds;
// commands soft-succeed.
class NpdrmDevice : public Device {
 public:
  NpdrmDevice(ObjectTable& objects);

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace krnl
