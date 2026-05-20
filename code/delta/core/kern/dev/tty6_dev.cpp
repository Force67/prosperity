
// Copyright (C) Force67 2019

#include <base.h>
#include "tty6_dev.h"

namespace krnl {
tty6Device::tty6Device(proc *p) : device(p) {}

bool tty6Device::init(const char *, uint32_t, uint32_t) { return true; }

uint8_t *tty6Device::map(void *addr, size_t, uint32_t, uint32_t, size_t) {
  __debugbreak();
  return reinterpret_cast<uint8_t *>(-1);
  // return SysError::SUCCESS;
}
} // namespace krnl
