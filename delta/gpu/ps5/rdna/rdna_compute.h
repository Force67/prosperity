#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) compute: recompiles an AGC compute shader to GLCompute
 * SPIR-V, producing the same gpu::gcn::RecompiledCs the shared Vulkan compute
 * backend (rhi::Dispatch) already consumes.
 *
 * The resource model, the SPIR-V translator and the MUBUF/MTBUF/MIMG/DS
 * emitters are shared with the GCN path (gpu/gcn): those encodings are
 * unchanged on RDNA2. Only SMEM was re-laid-out, and the control flow comes
 * from this directory's own decoder/CFG, so that is what lives here.
 */

#include "base/arch.h"

#include "gpu/gcn/gcn_translate.h"

namespace gpu::rdna {

// Readable shader scan budget, up to 64 KiB. Decoder kernels exceed 16 KiB.
u32 ComputeCodeDwords(const u32* code);

// Recompile an RDNA2 compute shader. cs_code is a guest pointer to the shader;
// num_thread_* the workgroup size (COMPUTE_NUM_THREAD_*); user_sgpr the number
// of user-data SGPRs seeded into s0.. (COMPUTE_PGM_RSRC2.user_sgpr);
// tgid_enable which workgroup-id dimensions land in the SGPRs after the user
// data; lds_dwords the raw RSRC2 LDS_SIZE field (128-dword granules).
// trap_present is RSRC2.TRAP_PRESENT; without a handler S_TRAP is a NOP.
//
// Returns ok=false when the shader uses something the compute backend cannot
// model, so the caller skips the dispatch loudly instead of running a wrong one
// (a compute dispatch writes guest memory; a wrong one corrupts it silently).
gpu::gcn::RecompiledCs RecompileCompute(const u32* cs_code,
                                        u32 num_thread_x,
                                        u32 num_thread_y,
                                        u32 num_thread_z,
                                        u32 user_sgpr,
                                        u32 tgid_enable,
                                        u32 lds_dwords,
                                        bool trap_present = false);

}  // namespace gpu::rdna
