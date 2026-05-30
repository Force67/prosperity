#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "device.h"

namespace krnl {
class proc;

class dipswDevice : public device {
public:
  dipswDevice(proc *);

  int32_t ioctl(uint32_t command, void *args) override;
};
}