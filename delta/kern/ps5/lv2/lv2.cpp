/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <cstdint>
#include <logger/logger.h>

#include "kern/ps4/lv2/error_table.h"
#include "lv2.h"

namespace krnl {
static int PS4ABI ps5_stub() {
  LOG_ERROR("unimplemented ps5 syscall");
  return -SysError::eNOSYS;
}

uintptr_t lv2_get_ps5(uint32_t sid) {
  LOG_WARNING("ps5 syscall {} routed to stub", sid);
  return reinterpret_cast<uintptr_t>(&ps5_stub);
}
} // namespace krnl
