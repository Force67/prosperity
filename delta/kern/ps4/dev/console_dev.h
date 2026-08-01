#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/console: the system debug tty. A title's debug output lands here the
// way it would on the host console: writes go to stdout, reads report EOF
// (the emulator has no input source), and the ioctl set is the tty one the
// kernel answers.
class consoleDevice : public device {
public:
  consoleDevice(proc *);

  bool init(const char *, uint32_t, uint32_t) override;
  int64_t read(void *, size_t) override;
  int64_t write(const void *, size_t n) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
  int32_t ioctl(uint32_t cmd, void *data) override;
};
} // namespace krnl
