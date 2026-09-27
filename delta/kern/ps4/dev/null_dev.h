#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {
class Process;

// /dev/null: reads return EOF, writes discard. The kernel uses this as the
// discard sink for fds redirected from stdin/stdout/stderr, and games open it
// to discard output.
class NullDevice : public Device {
 public:
  NullDevice(ObjectTable& objects);

  i64 Read(void*, size_t) override;
  i64 Write(const void*, size_t) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace kern