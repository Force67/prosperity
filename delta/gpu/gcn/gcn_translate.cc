/*
 * PS4Delta : PS4 emulation and research project
 */

#include "gpu/gcn/gcn_translate.h"
#include "base/arch.h"
#include "memory_debug/memory_debug.h"

#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_disasm.h"
#include "gpu/gcn/spirv/gcn_spirv.h"
#include "ui/shader_activity.h"

#include "base/algorithm.h"
#include "base/time/time.h"
#include "options/options.h"
#include "profile/profile.h"

namespace gpu::gcn {

u64 g_ns_recomp = 0;
u32 g_recomp_n = 0;
u64 g_ns_spv_val = 0, g_ns_spv_opt = 0;
u32 g_spv_hit_n = 0, g_spv_miss_n = 0;

namespace {
DELTA_OPTION(bool, kDirectGcn, "DELTA_GPU_DIRECT_GCN", false);

u64 NowNs() {
  return (base::TickClock::NowNs());
}
u32 g_max_gfx_buffers = kMinGfxBuffers;

u32 g_push_budget = 128;

u32 g_host_subgroup = kGcnWave;

}  // namespace

bool DirectGcnEnabled() {
  return kDirectGcn;
}

u32 MaxGfxBuffers() {
  return g_max_gfx_buffers;
}

void SetMaxGfxBuffers(u32 n) {
  g_max_gfx_buffers = base::Clamp(n, kMinGfxBuffers, kMaxGfxBuffers);
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
                     bool gl_clip_space,
                     const GsPipeline* gs,
                     u32 int_attr_mask,
                     u32 col_format,
                     u32 tex_cube_mask) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kShaderCache);
  DELTA_ZONE("gcn.recompile");
  const ui::ShaderCompilation compilation;
  Recompiled r;
  if (!vs_code || !vs_user_data || !ps_user_data)
    return r;
  const u64 t0 = NowNs();
  RecompileSpirv(vs_code, ps_code, vs_user_data, ps_user_data, ps_input_ena,
                 ps_in_cntl, ps_num_interp, tex_3d_mask, tex_1d_mask,
                 tex_uint_mask, mrt_uint_mask, mrt_bound_mask, gl_clip_space,
                 gs, int_attr_mask, col_format, tex_cube_mask, r);
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
                              u32 lds_dwords,
                              u64 native_mask,
                              u64 uint_mask) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kShaderCache);
  DELTA_ZONE("gcn.recompile_cs");
  const ui::ShaderCompilation compilation;
  RecompiledCs r;
  if (!cs_code)
    return r;
  const u64 t0 = NowNs();
  RecompileComputeSpirv(cs_code, num_thread_x, num_thread_y, num_thread_z,
                        user_sgpr, tgid_enable, lds_dwords, native_mask,
                        uint_mask, r);
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

base::String ListingAt(u64 code_address, u32 max_dwords) {
  base::String out;
  const auto* code = reinterpret_cast<const u32*>(code_address);
  for (const Inst& inst : DecodeShader(code, max_dwords)) {
    out += DisasmLine(inst);
    out += '\n';
  }
  return out;
}

}  // namespace gpu::gcn
