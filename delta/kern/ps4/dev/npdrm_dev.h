#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/npdrm: the NPDRM license/decryption manager. System-only; games reach
// license checks through the SBL libs. Registers so an open succeeds;
// commands soft-succeed.
class npdrmDevice : public device {
public:
  npdrmDevice(objectTable &objects);

  i32 ioctl(u32 command, void *args) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
