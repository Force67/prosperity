#pragma once

// Copyright (C) Force67 2019

#include "device.h"
#include "base/arch.h"

namespace krnl {
class tty6Device : public device {
public:
  tty6Device(objectTable &);

  bool init(const char *, u32, u32) override;
  u8 *map(void *, size_t, u32, u32, size_t) override;
};
}