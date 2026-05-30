// Copyright (C) Force67 2019

#include <base.h>
#include "sys_net.h"
#include "error_table.h"

namespace krnl {
int PS4ABI sys_netcontrol(uint32_t fd, uint32_t op, void* buffer,
        uint32_t size) {

    if (size > 160)
    return SysError::eINVAL;

    if (op == 20) {
      *static_cast<uint32_t *>(buffer) = 0xF00D;
      return 0;
    }

    return SysError::eINVAL;
}

int PS4ABI sys_socketex(const char* name, int32_t domain, int32_t type,
    int32_t protocol) {
    // TOO lazy for now
  return 0;
}
}