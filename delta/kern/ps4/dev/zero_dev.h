#pragma once

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

// /dev/zero: reads return zero-filled buffers, writes discard. Used to back
// anonymous mmap fallbacks and as a source of zero pages.
class zeroDevice : public device {
public:
  zeroDevice(objectTable &objects);

  i64 read(void *buf, size_t len) override;
  i64 write(const void *, size_t n) override;
  i64 lseek(i64, int) override;
  int fstat(void *stat) override;
};
} // namespace krnl