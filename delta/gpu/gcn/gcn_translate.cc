/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN shader recompiler entry point. Recompilation goes GCN -> SPIR-V directly
 * (spirv/), with a SPIRV-Tools optimize pass. This file is just the public
 * facade over that backend.
 */

#include "gpu/gcn/gcn_translate.h"

#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_disasm.h"
#include "gpu/gcn/spirv/gcn_spirv.h"

#include <chrono>

namespace gpu::gcn {

uint64_t g_ns_recomp = 0;
uint32_t g_recomp_n = 0;

namespace {
uint64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
}  // namespace

Recompiled Recompile(const uint32_t* vs_code,
                      const uint32_t* ps_code,
                      const uint32_t* vs_user_data,
                      const uint32_t* ps_user_data,
                      uint32_t ps_input_ena,
                      uint32_t tex_3d_mask,
                      uint32_t tex_1d_mask) {
  Recompiled r;
  if (!vs_code || !vs_user_data || !ps_user_data)
    return r;
  const uint64_t t0 = NowNs();
  RecompileSpirv(vs_code, ps_code, vs_user_data, ps_user_data, ps_input_ena,
                 tex_3d_mask, tex_1d_mask, r);
  g_ns_recomp += NowNs() - t0;
  g_recomp_n++;
  return r;
}

RecompiledCs RecompileCompute(const uint32_t* cs_code,
                              uint32_t num_thread_x,
                              uint32_t num_thread_y,
                              uint32_t num_thread_z,
                              uint32_t user_sgpr,
                              uint32_t tgid_enable,
                              uint32_t lds_dwords) {
  RecompiledCs r;
  if (!cs_code)
    return r;
  const uint64_t t0 = NowNs();
  RecompileComputeSpirv(cs_code, num_thread_x, num_thread_y, num_thread_z,
                        user_sgpr, tgid_enable, lds_dwords, r);
  g_ns_recomp += NowNs() - t0;
  g_recomp_n++;
  return r;
}

void DisassembleAt(uint64_t code_address, const char* tag) {
  if (!code_address)
    return;
  const auto* code = reinterpret_cast<const uint32_t*>(code_address);
  const uint32_t words = CodeLength(code, 4096);
  Disassemble(code, words ? words : 512, tag);
}

}  // namespace gpu::gcn
