
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <logger/logger.h>

#include <kern/proc.h>

namespace krnl {
using namespace krnl;

int PS4ABI sys_budget_get_ptype() {
  /*proc type, related to telemetry etc*/
  return 1;
}
} // namespace krnl