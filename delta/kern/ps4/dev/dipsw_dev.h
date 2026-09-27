#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace krnl {
class Proc;

class DipswDevice : public Device {
 public:
  DipswDevice(ObjectTable&);

  i32 Ioctl(u32 command, void* args) override;
};
}  // namespace krnl