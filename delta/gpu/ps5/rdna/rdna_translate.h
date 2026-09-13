#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) shader recompiler facade. Translates PS5 guest shaders
 * (decoded by rdna_decode) directly to SPIR-V and returns the same
 * gpu::gcn::Recompiled binding plan the shared Vulkan renderer consumes, so the
 * whole gpu/rhi + gpu/vulkan path is reused unchanged.
 *
 * The translator reuses the shared gpu::gcn SPIR-V backend: the register-file
 * model (gpu::gcn::Translator), the scalar/vector ALU emitters, exports, and
 * constant-buffer plumbing. Only the RDNA2-specific per-instruction dispatch
 * (field layouts + opcode remap) and the SMEM constant-buffer decode live in
 * rdna_translate.cc; everything downstream is the GFX7 path's code.
 */

#include "base/arch.h"

#include "gpu/gcn/gcn_translate.h"

namespace gpu::rdna {

struct NggConfig {
  const u32* gs_code = nullptr;
  u32 threads = 0;
  u32 input_primitives = 0;
  u32 max_vertices = 0;
  u32 max_primitives = 0;
  u32 lds_dwords = 0;
  // Per-draw system SGPR s0:s1, separate from the user window at s8.
  // Used for resource replay; generated shaders load it from draw data.
  u64 gs_user_data_addr = 0;
  bool separate_es = true;  // false when one program contains both NGG halves
};

// Whether the ES entry explicitly transfers to the separately bound GS.
bool HasNggTransfer(const u32* code);
bool HasNggPrimitiveExports(const u32* code);

// Recompile an RDNA2 VS+PS pair (guest pointers + per-stage user SGPRs, used for the
// fetch-shader pointer). On gfx10.3 the "VS" is the merged ES/GS NGG program (GS SH
// block). ps_input_ena fixes the PS input-VGPR layout (frag-coord/face) read directly,
// not via v_interp; gl_clip_space selects the guest's clip convention (the VS remaps z
// from [-w,w] to Vulkan's [0,w]). r.ok = false on unsupported features.
gpu::gcn::Recompiled Recompile(const u32* vs_code,
                               const u32* ps_code,
                               const u32* vs_user_data,
                               const u32* ps_user_data,
                               u32 ps_input_ena = 0,
                               bool gl_clip_space = false,
                               u32 vs_user_sgprs = 32,
                               u32 ps_user_sgprs = 32,
                               // SPI_PS_INPUT_CNTL_0..31 and NUM_INTERP: which
                               // VS parameter export each PS input slot reads.
                               const u32* ps_in_cntl = nullptr,
                               u32 ps_num_interp = 0,
                               const NggConfig* ngg = nullptr);

// What the fetch pointer contributes to module identity: a hash of the attribute plan,
// 0 when it parses to none. NOT a code hash: the two user-data dwords it comes from
// hold a plain per-draw constant in some titles (Dead Cells: fading alpha), so hashing
// bytes makes every draw a fresh module and the title recompiles forever.
u64 FetchPlanHash(u64 fetch_addr);

}  // namespace gpu::rdna
