#pragma once

#include "base/arch.h"
#include "guest_abi.h"

namespace kern {

int PS4ABI sys_regmgr_call(u32 op, u32 id, void* result, void* value, u64 type);

}  // namespace kern
