#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {
class Process;

// /dev/authmgr: the authentication manager. Maintains the EE-kc key table
// (content-id -> 32-byte key) the secure processor uses for license checks,
// with add/read/delete ioctls and SBL-style status codes. Games normally route
// through the npdrm device, but register a functional table regardless.
class AuthmgrDevice : public Device {
 public:
  AuthmgrDevice(ObjectTable& objects);

  i32 Ioctl(u32 command, void* args) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace kern
