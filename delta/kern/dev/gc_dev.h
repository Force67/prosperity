#pragma once

// Copyright (C) Force67 2019

#include "device.h"

namespace krnl {
class proc;

class gcDevice : public device {
public:
  gcDevice(proc *);

  bool init(const char *, uint32_t, uint32_t) override;
  int32_t ioctl(uint32_t command, void *args) override;
  uint8_t *map(void *, size_t, uint32_t, uint32_t, size_t) override;
};
}