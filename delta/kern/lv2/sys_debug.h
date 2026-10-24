#pragma once

#include "base/arch.h"
#include "guest_abi.h"

namespace kern {

int PS4ABI sys_debug_init(const int* version);
int PS4ABI sys_opmc_enable(u32 id);
int PS4ABI sys_opmc_disable(u32 id);
int PS4ABI sys_opmc_set_ctl(u32 id, u32 index, u64 ctl);
int PS4ABI sys_opmc_set_ctr(u32 id, u32 index, u64 value);
int PS4ABI sys_opmc_get_ctr(u32 id, u32 index, u64* value);
int PS4ABI sys_opmc_set_hw(u32 op, u64 a, u32 b, u64 c);
int PS4ABI sys_opmc_get_hw(u32 op, u64 a, u32 index, u64* out);
int PS4ABI sys_mdbg_call();
int PS4ABI sys_set_gpo();
int PS4ABI sys_get_gpo();
int PS4ABI sys_test_debug_rwmem();
int PS4ABI sys_get_cpu_usage_all();
int PS4ABI sys_get_cpu_usage_proc();
int PS4ABI sys_set_uevt();
int PS4ABI sys_set_chicken_switches();
int PS4ABI sys_unk645();

int PS4ABI sys_namedobj_create(const char* name, void* data, u32 type);
int PS4ABI sys_namedobj_delete(u32 id, u32 type);
int PS4ABI sys_workaround8849();

}  // namespace kern
