#pragma once

#include "base/arch.h"
#include "guest_abi.h"

namespace kern {

int PS4ABI sys_debug_init(const int* version);
int PS4ABI sys_opmc_enable();
int PS4ABI sys_opmc_disable();
int PS4ABI sys_opmc_set_ctl();
int PS4ABI sys_opmc_set_ctr();
int PS4ABI sys_opmc_get_ctr();
int PS4ABI sys_opmc_set_hw();
int PS4ABI sys_opmc_get_hw();
int PS4ABI sys_mdbg_call();
int PS4ABI sys_set_gpo();
int PS4ABI sys_get_gpo();
int PS4ABI sys_test_debug_rwmem();
int PS4ABI sys_get_cpu_usage_all();
int PS4ABI sys_get_cpu_usage_proc();
int PS4ABI sys_set_uevt();
int PS4ABI sys_set_chicken_switches();
int PS4ABI sys_unk645();

int PS4ABI sys_namedobj_create(const char* name, void* arg2, u32 arg3);
int PS4ABI sys_namedobj_delete();
int PS4ABI sys_workaround8849();

}  // namespace kern
