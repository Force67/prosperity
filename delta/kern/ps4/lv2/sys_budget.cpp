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

// sys_budget_get_ptype: looks up a process by pid, reads the budget id stored
// on the proc, then returns that budget's process-type field. The ptype is set
// at budget_create time and ranges 0..3. The kernel requires the system ucred
// (returns 78/ENOSYS otherwise) and returns 2/ESRCH if the pid has no budget.
// We always report ptype 1, the value a normal game title carries.
int PS4ABI sys_budget_get_ptype() {
  return 1;
}
} // namespace krnl