#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Direct GCN -> SPIR-V backend entry points. Emits SPIR-V via spv_emit and
 * cleans it up with a SPIRV-Tools optimize pass (spv_post). Only
 * gcn_translate.cc calls these; everything else goes through the
 * gcn_translate.h facade.
 */

#include <cstdint>

#include "gpu/ps4/gcn/gcn_translate.h"

namespace gpu::gcn {

// Translate a VS+PS pair into r (fills the SPIR-V binaries + binding plan,
// sets r.ok). Returns r.ok. When the backend is compiled out
// (no SPIRV-Tools/Headers) this always declines.
bool RecompileSpirv(const uint32_t* vs_code,
                    const uint32_t* ps_code,
                    const uint32_t* vs_user_data,
                    const uint32_t* ps_user_data,
                    Recompiled& r);

// Translate a compute shader into r (GLCompute SPIR-V + resource plan).
// lds_dwords is the raw COMPUTE_PGM_RSRC2.LDS_SIZE field (128-dword granules).
bool RecompileComputeSpirv(const uint32_t* cs_code,
                           uint32_t num_thread_x,
                           uint32_t num_thread_y,
                           uint32_t num_thread_z,
                           uint32_t user_sgpr,
                           uint32_t tgid_enable,
                           uint32_t lds_dwords,
                           RecompiledCs& r);

}  // namespace gpu::gcn
