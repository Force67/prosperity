
// Copyright (C) Force67 2019

#include <base.h>
#include "base/arch.h"
#include "tty6_dev.h"

namespace krnl {
tty6Device::tty6Device(objectTable &objects) : device(objects) {}

bool tty6Device::init(const char *, u32, u32) { return true; }

u8 *tty6Device::map(void *addr, size_t, u32, u32, size_t) {
  __debugbreak();
  return reinterpret_cast<u8 *>(-1);
  // return SysError::SUCCESS;
}
} // namespace krnl
