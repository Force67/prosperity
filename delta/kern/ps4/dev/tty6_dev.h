#pragma once

// Copyright (C) Force67 2019

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace krnl {
class Tty6Device : public Device {
 public:
  Tty6Device(ObjectTable&);

  bool Init(const char*, u32, u32) override;
  u8* Map(void*, size_t, u32, u32, size_t) override;
};
}  // namespace krnl