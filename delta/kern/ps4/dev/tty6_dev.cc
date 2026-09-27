
// Copyright (C) Force67 2019

#include "kern/ps4/dev/tty6_dev.h"
#include "base/arch.h"

namespace krnl {
Tty6Device::Tty6Device(ObjectTable& objects) : Device(objects) {}

bool Tty6Device::Init(const char*, u32, u32) {
  return true;
}

u8* Tty6Device::Map(void* addr, size_t, u32, u32, size_t) {
  __builtin_trap();
  return reinterpret_cast<u8*>(-1);
  // return SysError::SUCCESS;
}
}  // namespace krnl
