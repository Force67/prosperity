#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"
#include "base/functional/function.h"

#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps4/liverpool.h"

namespace gpu::ps4 {

struct GraphicsShaderState {
  u64 vs_addr = 0;
  u64 ps_addr = 0;
  u64 fetch_addr = 0;
  u32 ps_input_ena = 0;
  const u32* ps_in_cntl = nullptr;
  bool honour_ps_in_cntl = false;
  u32 ps_num_interp = 0;
  u32 tex_3d_mask = 0;
  u32 tex_1d_mask = 0;
  u32 tex_cube_mask = 0;
  u32 tex_uint_mask = 0;
  u32 mrt_uint_mask = 0;
  u32 mrt_bound_mask = 0xFF;
  bool gl_clip = false;
  const gcn::GsPipeline* gs = nullptr;
  u32 int_attr_mask = 0;
  u32 col_format = gcn::kColFormatUnknown;
};

const gcn::Recompiled& GetGraphicsShader(const Regs& regs,
                                         const GraphicsShaderState& state);

void PrefetchGraphicsShader(const Regs& regs, const GraphicsShaderState& state);

struct ComputeShaderState {
  u64 cs_addr = 0;
  u32 thread_x = 0, thread_y = 0, thread_z = 0;
  u32 user_sgpr = 0, tgid_enable = 0, lds_dwords = 0;
  u64 native_mask = 0, uint_mask = 0, shared_mask = 0;
  // `binding << 16 | T# dword1 bits 20..29` per staged image whose
  // descriptor the dispatch resolved; their decode is built for it.
  base::Vector<u32> formats;
};

const gcn::RecompiledCs& GetComputeShader(const ComputeShaderState& state);

// `variant` picks, from the recompiled `state`, the state the dispatch will
// run, so only that one gets a pipeline.
void PrefetchComputeShader(
    const ComputeShaderState& state,
    base::Function<ComputeShaderState(const gcn::RecompiledCs&)> variant);

}  // namespace gpu::ps4
