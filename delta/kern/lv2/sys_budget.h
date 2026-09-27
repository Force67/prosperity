#pragma once

#include "base/arch.h"
#include "guest_abi.h"

namespace kern {

// The proc telemetry "process type" sys_get_proc_type_info reports.
int PS4ABI sys_budget_get_ptype();

int PS4ABI sys_budget_create();
int PS4ABI sys_budget_delete();
int PS4ABI sys_budget_get();
int PS4ABI sys_budget_set();
int PS4ABI sys_budget_getid();
int PS4ABI sys_budget_get_ptype_of_budget();

}  // namespace kern
