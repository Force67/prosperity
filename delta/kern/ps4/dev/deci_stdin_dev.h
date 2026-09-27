#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace krnl {
class Proc;

// /dev/deci_stdin: the debugger tty input channel. Each opener gets a private
// line buffer that a privileged writer can fill; reads drain it. The emulator
// has no writer, so reads report EOF and writes are discarded.
class DeciStdinDevice : public Device {
 public:
  DeciStdinDevice(ObjectTable& objects);

  i64 Read(void*, size_t) override;
  i64 Write(const void*, size_t n) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace krnl
