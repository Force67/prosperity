#pragma once

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
  u64 gs_user_data_addr = 0;
  bool separate_es = true;
};

bool HasNggTransfer(const u32* code);
bool HasNggPrimitiveExports(const u32* code);

gpu::gcn::Recompiled Recompile(const u32* vs_code,
                               const u32* ps_code,
                               const u32* vs_user_data,
                               const u32* ps_user_data,
                               u32 ps_input_ena = 0,
                               bool gl_clip_space = false,
                               u32 vs_user_sgprs = 32,
                               u32 ps_user_sgprs = 32,
                               const u32* ps_in_cntl = nullptr,
                               u32 ps_num_interp = 0,
                               const NggConfig* ngg = nullptr);

u64 FetchPlanHash(u64 fetch_addr);

}  // namespace gpu::rdna
