#pragma once

// PS5 /dev/gc device (libSceAgc / libSceAgcDriver AGC protocol). Kept fully
// separate from the PS4 GcDevice (kern/ps4/dev/gc_dev.h), which speaks the GNM
// PM4 ioctl protocol; the two command sets share no opcodes, so mixing them in
// one switch invited bugs. The device factory (sys_vfs make_device)
// instantiates this for /dev/gc when the active process is a PS5 title.

#include "base/arch.h"
#include "kern/ps4/dev/device.h"  // shared abstract device base

namespace kern {
class Process;

class GcDevicePs5 : public Device {
 public:
  GcDevicePs5(ObjectTable&);

  bool Init(const char*, u32, u32) override;
  i32 Ioctl(u32 command, void* args) override;
  u8* Map(void*, size_t, u32, u32, size_t) override;
};
}  // namespace kern
