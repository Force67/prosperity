#pragma once

#include "base/arch.h"

#include "gpu/gcn/gcn_translate.h"

namespace gpu::gcn {

bool RecompileSpirv(const u32* vs_code,
                    const u32* ps_code,
                    const u32* vs_user_data,
                    const u32* ps_user_data,
                    u32 ps_input_ena,
                    const u32* ps_in_cntl,
                    u32 ps_num_interp,
                    u32 tex_3d_mask,
                    u32 tex_1d_mask,
                    u32 tex_uint_mask,
                    u32 mrt_uint_mask,
                    u32 mrt_bound_mask,
                    bool gl_clip_space,
                    const GsPipeline* gs,
                    u32 int_attr_mask,
                    u32 col_format,
                    u32 tex_cube_mask,
                    Recompiled& r);

bool RecompileComputeSpirv(const u32* cs_code,
                           u32 num_thread_x,
                           u32 num_thread_y,
                           u32 num_thread_z,
                           u32 user_sgpr,
                           u32 tgid_enable,
                           u32 lds_dwords,
                           u64 native_mask,
                           u64 uint_mask,
                           u64 shared_mask,
                           RecompiledCs& r);

}  // namespace gpu::gcn
