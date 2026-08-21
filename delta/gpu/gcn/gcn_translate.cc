/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN shader recompiler entry point. Recompilation goes GCN -> SPIR-V directly
 * (spirv/), with a SPIRV-Tools optimize pass. This file is just the public
 * facade over that backend.
 */

#include "gpu/gcn/gcn_translate.h"
#include "base/arch.h"

#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_disasm.h"
#include "gpu/gcn/spirv/gcn_spirv.h"

#include <algorithm>
#include <chrono>

namespace gpu::gcn {

u64 g_ns_recomp = 0;
u32 g_recomp_n = 0;
u64 g_ns_spv_val = 0, g_ns_spv_opt = 0;
u32 g_spv_hit_n = 0, g_spv_miss_n = 0;

namespace {
u64 NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
// Set-2 raw-buffer bindings the planner may use. Starts at the Vulkan floor so
// a shader planned before the renderer reports the device limit is still valid
// everywhere. Written exactly once, by vk_upload_ring during device init,
// before the command processor issues its first recompile.
u32 g_max_gfx_buffers = kMinGfxBuffers;

// Device push-constant budget. Written once at renderer init (before the
// command processor issues any recompile), read on the recompile path.
u32 g_push_budget = 128;  // the Vulkan floor: no room for code addresses

// Host subgroup width, same write-once-at-init discipline. Defaults to a full
// GCN wave so a shader planned before the renderer reports it assumes the
// lockstep the guest compiler assumed.
u32 g_host_subgroup = kGcnWave;

}  // namespace

u32 MaxGfxBuffers() {
  return g_max_gfx_buffers;
}

void SetMaxGfxBuffers(u32 n) {
  g_max_gfx_buffers = std::clamp(n, kMinGfxBuffers, kMaxGfxBuffers);
}

bool PushCodeBase() {
  return g_push_budget >= 144;
}

void SetPushBudget(u32 bytes) {
  g_push_budget = bytes;
}

u32 HostSubgroupSize() {
  return g_host_subgroup;
}

void SetHostSubgroupSize(u32 lanes) {
  g_host_subgroup = lanes ? lanes : kGcnWave;
}

Recompiled Recompile(const u32* vs_code,
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
                      bool gl_clip_space) {
  Recompiled r;
  if (!vs_code || !vs_user_data || !ps_user_data)
    return r;
  const u64 t0 = NowNs();
  RecompileSpirv(vs_code, ps_code, vs_user_data, ps_user_data, ps_input_ena,
                 ps_in_cntl, ps_num_interp,
                 tex_3d_mask, tex_1d_mask, tex_uint_mask, mrt_uint_mask,
                 mrt_bound_mask,
                 gl_clip_space, r);
  g_ns_recomp += NowNs() - t0;
  g_recomp_n++;
  return r;
}

RecompiledCs RecompileCompute(const u32* cs_code,
                              u32 num_thread_x,
                              u32 num_thread_y,
                              u32 num_thread_z,
                              u32 user_sgpr,
                              u32 tgid_enable,
                              u32 lds_dwords) {
  RecompiledCs r;
  if (!cs_code)
    return r;
  const u64 t0 = NowNs();
  RecompileComputeSpirv(cs_code, num_thread_x, num_thread_y, num_thread_z,
                        user_sgpr, tgid_enable, lds_dwords, r);
  g_ns_recomp += NowNs() - t0;
  g_recomp_n++;
  return r;
}

void DisassembleAt(u64 code_address, const char* tag) {
  if (!code_address)
    return;
  const auto* code = reinterpret_cast<const u32*>(code_address);
  const u32 words = CodeLength(code, 4096);
  Disassemble(code, words ? words : 512, tag);
}

}  // namespace gpu::gcn
