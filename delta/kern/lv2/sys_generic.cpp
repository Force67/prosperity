

// Copyright (C) Force67 2019

#include <guest_abi.h>
#include "base/arch.h"
#include <base/logging.h>
#include <cstring>
#include <cstdlib>
#include "sys_generic.h"
#include "kern/proc.h"
#include <options/options.h>

namespace {
DELTA_OPTION(bool, kIoctlTrace, "DELTA_IOCTL_TRACE", false);
}  // namespace

namespace krnl {
int PS4ABI sys_ioctl(u32 fd, u32 cmd, void *data) {
  auto *proc = proc::getActive();
  if (!proc)
    return -1;

  auto *obj = proc->getObjTable().get(fd);
  if (obj)
    return static_cast<device *>(obj)->ioctl(cmd, data);

  // Unknown fd (e.g. a stubbed socket from sys_socketex, or an ioctl probe on
  // stdio). Soft-succeed with a zeroed out-buffer rather than returning EBADF:
  // some middleware treats an ioctl error on its setup probe as fatal (DOOM's
  // bundled FMOD aborts audio init on the EBADF). Zero the IOC_OUT payload so
  // the caller reads a benign result instead of stack garbage.
  if (kIoctlTrace)
    BASE_LOGI("ioctl", "soft-ok: fd={} cmd={:#x}", fd, cmd);
  if (data) {
    u32 sz = (cmd >> 16) & 0x1fff;
    if (cmd & 0x40000000u) // IOC_OUT
      std::memset(data, 0, sz);
  }
  return 0;
}
} // namespace krnl
