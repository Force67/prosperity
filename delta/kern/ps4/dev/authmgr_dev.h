#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/authmgr: the authentication manager. Maintains the EE-kc key table
// (content-id -> 32-byte key) the secure processor uses for license checks,
// with add/read/delete ioctls and SBL-style status codes. Games normally route
// through the npdrm device, but register a functional table regardless.
class authmgrDevice : public device {
public:
  authmgrDevice(objectTable &objects);

  i32 ioctl(u32 command, void *args) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
