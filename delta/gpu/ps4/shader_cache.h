#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Which recompiled SPIR-V module a given piece of guest state needs.
 *
 * A guest shader is not its code alone: the translator bakes in register state
 * the code does not carry (which PS inputs are enabled, which sampled images
 * are volumes or integers, the clip convention), so the same bytes under
 * different state are a different module. Everything that decides that lives
 * here, behind two lookups, along with the keying: by code CONTENT rather than
 * address, because titles stream shaders to fresh addresses and an address key
 * then misses forever.
 *
 * Modules are cached for the life of the process. Recompiling costs
 * milliseconds and the working set is bounded by the title's shader count.
 */

#include "base/arch.h"

#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps4/liverpool.h"

namespace gpu::ps4 {

// The state one VS/PS pair is compiled against. Addresses are guest addresses;
// `fetch_addr` is zero when the VS calls no fetch shader.
struct GraphicsShaderState {
  u64 vs_addr = 0;
  u64 ps_addr = 0;
  u64 fetch_addr = 0;
  u32 ps_input_ena = 0;
  // SPI_PS_INPUT_CNTL_0..31: the VS parameter export each PS input slot reads.
  // Always part of the module's identity; only handed to the translator when
  // `honour_ps_in_cntl` is set (see DELTA_GPU_PSCNTL_APPLY).
  const u32* ps_in_cntl = nullptr;
  bool honour_ps_in_cntl = false;
  u32 ps_num_interp = 0;
  // Bit n set = sampler binding n reads a volume / 1D / integer-format image.
  // A 3D or integer descriptor is indistinguishable in the code, so the
  // dimensionality and the texel type belong to the module's identity.
  u32 tex_3d_mask = 0;
  u32 tex_1d_mask = 0;
  u32 tex_uint_mask = 0;
  u32 mrt_uint_mask = 0;  // colour targets with an integer texel format
  // Bit n set = the pass binds colour attachment n. The module drops exports to
  // the targets that are clear, so the same code used in a pass that binds them
  // is a different module.
  u32 mrt_bound_mask = 0xFF;
  // PA_CL_CLIP_CNTL.DX_CLIP_SPACE_DEF == 0: the VS bakes the z remap in.
  bool gl_clip = false;
};

// The module for `state`, recompiled on first use. Never null; check .ok, which
// is false for a shader pair the translator declined.
const gcn::Recompiled& GetGraphicsShader(const Regs& regs,
                                         const GraphicsShaderState& state);

// The workgroup shape and RSRC2 state a compute module is built with. The same
// CS can legally be re-dispatched with a different workgroup size, so the code
// address alone does not identify the module.
struct ComputeShaderState {
  u64 cs_addr = 0;
  u32 thread_x = 0, thread_y = 0, thread_z = 0;
  u32 user_sgpr = 0, tgid_enable = 0, lds_dwords = 0;
};

// The compute module for `state`, recompiled on first use. Never null; .ok is
// false for a shader using something the compute backend does not implement,
// which the caller must skip loudly rather than run.
const gcn::RecompiledCs& GetComputeShader(const ComputeShaderState& state);

}  // namespace gpu::ps4
