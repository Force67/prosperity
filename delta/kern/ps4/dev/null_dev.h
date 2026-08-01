#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/null: reads return EOF, writes discard. The kernel uses this as the
// discard sink for fds redirected from stdin/stdout/stderr, and games open it
// to discard output.
class nullDevice : public device {
public:
  nullDevice(proc *p);

  int64_t read(void *, size_t) override;
  int64_t write(const void *, size_t) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl