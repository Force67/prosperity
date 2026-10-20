#pragma once

#include "base/arch.h"

#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps5/rdna/rdna_translate.h"

namespace gpu::ps5 {

struct GraphicsShaderState {
  u64 vs_addr = 0;
  u64 ps_addr = 0;
  u64 fetch_addr = 0;
  u32 vs_user_sgprs = 0;
  u32 ps_user_sgprs = 0;
  u32 ps_input_ena = 0;
  const u32* ps_in_cntl = nullptr;
  u32 ps_num_interp = 0;
  bool gl_clip = false;
  const u32* vs_user_data = nullptr;
  const u32* ps_user_data = nullptr;
  rdna::NggConfig ngg;
};

const gcn::Recompiled& GetGraphicsShader(const GraphicsShaderState& state);

struct ComputeShaderState {
  u64 cs_addr = 0;
  u32 thread_x = 0, thread_y = 0, thread_z = 0;
  u32 user_sgpr = 0, tgid_enable = 0, lds_dwords = 0;
  bool trap_present = false;
  bool wave32 = false;
};

// `formats` holds `binding << 16 | T# dword1 bits 20..29` for staged images
// whose descriptor the dispatch resolved; their decode is built for that
// format.
const gcn::RecompiledCs& GetComputeShader(
    const ComputeShaderState& state,
    u64 native_mask = 0,
    u64 uint_mask = 0,
    bool plan_only = false,
    const base::Vector<u32>& formats = {});

}  // namespace gpu::ps5
