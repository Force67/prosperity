#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/npdrm: the NPDRM license/decryption manager. System-only; games reach
// license checks through the SBL libs. Registers so an open succeeds;
// commands soft-succeed.
class npdrmDevice : public device {
public:
  npdrmDevice(proc *p);

  int32_t ioctl(uint32_t command, void *args) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
