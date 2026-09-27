#pragma once

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {
class Process;

// /dev/zero: reads return zero-filled buffers, writes discard. Used to back
// anonymous mmap fallbacks and as a source of zero pages.
class ZeroDevice : public Device {
 public:
  ZeroDevice(ObjectTable& objects);

  i64 Read(void* buf, size_t len) override;
  i64 Write(const void*, size_t n) override;
  i64 Lseek(i64, int) override;
  int Fstat(void* stat) override;
};
}  // namespace kern