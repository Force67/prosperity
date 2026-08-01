#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/authmgr: the authentication manager. Maintains the EE-kc key table
// (content-id -> 32-byte key) the secure processor uses for license checks,
// with add/read/delete ioctls and SBL-style status codes. Games normally route
// through the npdrm device, but register a functional table regardless.
class authmgrDevice : public device {
public:
  authmgrDevice(proc *p);

  int32_t ioctl(uint32_t command, void *args) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
