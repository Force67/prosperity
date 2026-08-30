#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/deci_stdin: the debugger tty input channel. Each opener gets a private
// line buffer that a privileged writer can fill; reads drain it. The emulator
// has no writer, so reads report EOF and writes are discarded.
class deciStdinDevice : public device {
public:
  deciStdinDevice(objectTable &objects);

  i64 read(void *, size_t) override;
  i64 write(const void *, size_t n) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl
