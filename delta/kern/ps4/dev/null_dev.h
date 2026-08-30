#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/null: reads return EOF, writes discard. The kernel uses this as the
// discard sink for fds redirected from stdin/stdout/stderr, and games open it
// to discard output.
class nullDevice : public device {
public:
  nullDevice(objectTable &objects);

  i64 read(void *, size_t) override;
  i64 write(const void *, size_t) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl