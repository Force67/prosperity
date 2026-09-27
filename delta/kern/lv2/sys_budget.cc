/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "guest_abi.h"
#include "logger/logger.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/stub_log.h"
#include "kern/lv2/sys_budget.h"
#include "kern/process.h"

namespace kern {
using namespace kern;

// sys_budget_get_ptype: looks up a process by pid, reads the budget id stored
// on the proc, then returns that budget's process-type field. The ptype is set
// at budget_create time and ranges 0..3. The kernel requires the system ucred
// (returns 78/ENOSYS otherwise) and returns 2/ESRCH if the pid has no budget.
// We always report ptype 1, the value a normal game title carries.
int PS4ABI sys_budget_get_ptype() {
  return 1;
}

// Budget objects gate flexible memory / resource pools; unenforced, but the
// guest stores the id and passes it back, so hand out a fixed non-zero id.
// Logged once: an overcommit shows up here first. Args {name@0, ptype@8 (0..3),
// res@16, nres@24 (0..10), resOut@32}; object = 64 bytes; system ucred only
// (else 78).
int PS4ABI sys_budget_create() {
  static base::Atomic<bool> once{false};
  LogOnce(once, "budget_create granted unconditionally (no enforcement)");
  return 0x2001;
}
int PS4ABI sys_budget_delete() {
  return 0;
}
// The libkernel_sys wrappers forward args straight through; the kernel fills a
// buffer no libkernel function reads back (layout unrecoverable), and callers
// pre-zero it, so success reads as an all-zero budget. Return success, not an
// errno: with carry clear, a negative return would drive the wrapper's errno
// path on a stale errno.
int PS4ABI sys_budget_get() {
  return 0;
}
int PS4ABI sys_budget_set() {
  return 0;
}
// sys_budget_getid: system-ucred only (a game gets 78 on hardware), but the
// game's wrapper passes the id back to budget_get/delete, so keep the benign
// fixed id rather than an error the title was never coded to handle.
int PS4ABI sys_budget_getid() {
  return 0x2001;
}
int PS4ABI sys_budget_get_ptype_of_budget() {
  return sys_budget_get_ptype();
}

}  // namespace kern