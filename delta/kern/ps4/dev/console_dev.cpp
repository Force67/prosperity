
// Copyright (C) Force67 2019

#include "console_dev.h"

namespace krnl {
consoleDevice::consoleDevice(proc *p) : device(p) {}

bool consoleDevice::init(const char *, uint32_t, uint32_t) { return true; }
} // namespace krnl