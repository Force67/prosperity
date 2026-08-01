#pragma once

#include "device.h"

namespace krnl {
class proc;

// /dev/zero: reads return zero-filled buffers, writes discard. Used to back
// anonymous mmap fallbacks and as a source of zero pages.
class zeroDevice : public device {
public:
  zeroDevice(proc *p);

  int64_t read(void *buf, size_t len) override;
  int64_t write(const void *, size_t n) override;
  int64_t lseek(int64_t, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl