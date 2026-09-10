#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Which recompiled SPIR-V module a given piece of guest state needs.
 *
 * A guest shader is not its code alone: the translator bakes in register state
 * the code does not carry (which PS inputs are enabled, how many user SGPRs the
 * stage was launched with, the clip convention), so the same bytes under
 * different state are a different module. A vertex program also depends on the
 * fetch program it calls, which is a separate address. Everything that decides
 * that lives here, behind two lookups.
 *
 * Keyed by code CONTENT, like the PS4 path: AGC hands its shaders out of a pool
 * the title refills, so the same program arrives at a fresh address constantly
 * and an address key misses forever.
 *
 * Modules are cached for the life of the process. Recompiling costs
 * milliseconds and the working set is bounded by the title's shader count.
 */

#include "base/arch.h"

#include "gpu/gcn/gcn_translate.h"

namespace gpu::ps5 {

// The state one VS/PS pair is compiled against. Addresses are guest addresses;
// `ps_addr` is zero for a depth-only pass and `fetch_addr` zero when the vertex
// program fetches nothing. The user-data windows are not part of the identity:
// the translator only reads the fetch pointer out of them, which `fetch_addr`
// already carries.
struct GraphicsShaderState {
  u64 vs_addr = 0;
  u64 ps_addr = 0;
  u64 fetch_addr = 0;
  u32 vs_user_sgprs = 0;
  u32 ps_user_sgprs = 0;
  u32 ps_input_ena = 0;
  // SPI_PS_INPUT_CNTL_0..31 and SPI_PS_IN_CONTROL.NUM_INTERP. Which VS export
  // each PS input slot reads is register state the code does not carry, so it
  // is part of the module's identity like ps_input_ena is.
  const u32* ps_in_cntl = nullptr;
  u32 ps_num_interp = 0;
  // PA_CL_CLIP_CNTL.DX_CLIP_SPACE_DEF == 0: the VS bakes the z remap in.
  bool gl_clip = false;
  const u32* vs_user_data = nullptr;
  const u32* ps_user_data = nullptr;
};

// The module for `state`, recompiled on first use. Never null; check .ok, which
// is false for a shader pair the translator declined.
const gcn::Recompiled& GetGraphicsShader(const GraphicsShaderState& state);

// The workgroup shape and RSRC2 state a compute module is built with. The same
// CS can legally be re-dispatched with a different workgroup size, so the code
// address alone does not identify the module.
struct ComputeShaderState {
  u64 cs_addr = 0;
  u32 thread_x = 0, thread_y = 0, thread_z = 0;
  u32 user_sgpr = 0, tgid_enable = 0, lds_dwords = 0;
  bool trap_present = false;
  bool wave32 = false;  // COMPUTE_DISPATCH_INITIATOR.CS_W32_EN
};

// The compute module for `state`, recompiled on first use. Never null; .ok is
// false for a shader using something the compute backend does not implement,
// which the caller must skip loudly rather than run.
const gcn::RecompiledCs& GetComputeShader(const ComputeShaderState& state);

}  // namespace gpu::ps5
