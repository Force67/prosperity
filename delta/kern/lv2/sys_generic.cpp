

// Copyright (C) Force67 2019

#include <base.h>
#include "sys_generic.h"
#include "kern/proc.h"

namespace krnl {
int PS4ABI sys_ioctl(uint32_t fd, uint32_t cmd, void *data) {
  auto *proc = proc::getActive();
  if (!proc)
    return -1;

  auto *obj = proc->getObjTable().get(fd);
  if (obj)
    return static_cast<device *>(obj)->ioctl(cmd, data);

  // Unknown fd (e.g. a stubbed socket from sys_socketex). Soft-fail instead of
  // trapping so the guest can cope; matches sys_open / gc ioctl / sysctl.
  std::printf("[ioctl] EBADF: fd=%u cmd=%#x\n", fd, cmd);
  return -SysError::eBADF;
}
} // namespace krnl
