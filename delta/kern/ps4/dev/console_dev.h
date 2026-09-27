#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace krnl {
class Proc;

// /dev/console: the system debug tty. A title's debug output lands here the
// way it would on the host console: writes go to stdout, reads report EOF
// (the emulator has no input source), and the ioctl set is the tty one the
// kernel answers.
class ConsoleDevice : public Device {
 public:
  ConsoleDevice(ObjectTable&);

  bool Init(const char*, u32, u32) override;
  i64 Read(void*, size_t) override;
  i64 Write(const void*, size_t n) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
  i32 Ioctl(u32 cmd, void* data) override;
};
}  // namespace krnl
