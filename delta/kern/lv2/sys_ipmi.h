#pragma once

#include "base/arch.h"
#include "guest_abi.h"

namespace kern {

int PS4ABI
sys_ipmimgr_call(u32 op, u32 kid, void* out, void* in, u64 insize, u64 arg6);

}  // namespace kern
