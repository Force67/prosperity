#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/console: the system debug tty. A title's debug output lands here the
// way it would on the host console: writes go to stdout, reads report EOF
// (the emulator has no input source), and the ioctl set is the tty one the
// kernel answers.
class consoleDevice : public device {
public:
  consoleDevice(objectTable &);

  bool init(const char *, u32, u32) override;
  i64 read(void *, size_t) override;
  i64 write(const void *, size_t n) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
  i32 ioctl(u32 cmd, void *data) override;
};
} // namespace krnl
