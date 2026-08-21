#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Direct GCN -> SPIR-V backend entry points. Emits SPIR-V via spv_emit and
 * cleans it up with a SPIRV-Tools optimize pass (spv_post). Only
 * gcn_translate.cc calls these; everything else goes through the
 * gcn_translate.h facade.
 */

#include "base/arch.h"

#include "gpu/gcn/gcn_translate.h"

namespace gpu::gcn {

// Translate a VS+PS pair into r (fills the SPIR-V binaries + binding plan,
// sets r.ok). Returns r.ok. When the backend is compiled out
// (no SPIRV-Tools/Headers) this always declines.
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
                     Recompiled& r);

// Translate a compute shader into r (GLCompute SPIR-V + resource plan).
// lds_dwords is the raw COMPUTE_PGM_RSRC2.LDS_SIZE field (128-dword granules).
bool RecompileComputeSpirv(const u32* cs_code,
                           u32 num_thread_x,
                           u32 num_thread_y,
                           u32 num_thread_z,
                           u32 user_sgpr,
                           u32 tgid_enable,
                           u32 lds_dwords,
                           RecompiledCs& r);

}  // namespace gpu::gcn
