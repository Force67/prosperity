/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <cstring>

#include "dce_dev.h"

namespace krnl {
dceDevice::dceDevice(proc *p) : device(p) {}

bool dceDevice::init(const char *, uint32_t, uint32_t) { return true; }

int32_t dceDevice::ioctl(uint32_t cmd, void *data) {
  // TODO: implement the display ioctls. libSceVideoOut's framebuffer setup
  // passes output buffers (by pointer, inside the arg struct) that these
  // ioctls must fill with real framebuffer info; until then the guest reads
  // those buffers uninitialized and derives a garbage framebuffer size, which
  // throws bad_alloc deep in video init. Returning success alone is not enough.
  printf("[dce] ioctl(%#x) -> 0 (display unimplemented)\n", cmd);
  return 0;
}
}  // namespace krnl
