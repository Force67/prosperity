#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

class dipswDevice : public device {
public:
  dipswDevice(objectTable &);

  i32 ioctl(u32 command, void *args) override;
};
}