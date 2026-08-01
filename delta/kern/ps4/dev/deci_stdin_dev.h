#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/deci_stdin: the debugger tty input channel. Each opener gets a private
// line buffer that a privileged writer can fill; reads drain it. The emulator
// has no writer, so reads report EOF and writes are discarded.
class deciStdinDevice : public device {
public:
  deciStdinDevice(proc *p);

  int64_t read(void *, size_t) override;
  int64_t write(const void *, size_t n) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
