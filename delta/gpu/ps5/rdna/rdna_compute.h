#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "base/arch.h"

#include "gpu/gcn/gcn_translate.h"

namespace gpu::rdna {

u32 ComputeCodeDwords(const u32* code);

gpu::gcn::RecompiledCs RecompileCompute(const u32* cs_code,
                                        u32 num_thread_x,
                                        u32 num_thread_y,
                                        u32 num_thread_z,
                                        u32 user_sgpr,
                                        u32 tgid_enable,
                                        u32 lds_dwords,
                                        bool trap_present = false,
                                        bool wave32 = false,
                                        u64 native_mask = 0,
                                        u64 uint_mask = 0,
                                        bool plan_only = false);

}  // namespace gpu::rdna
