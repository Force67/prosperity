/*
 * PS4Delta : PS4 emulation and research project
 *
 * Which recompiled module a given piece of guest state needs. See
 * shader_cache.h.
 */

#include "gpu/ps4/shader_cache.h"
#include "base/arch.h"

#include <functional>
#include <unordered_map>

#include "gpu/gcn/gcn_decode.h"
#include "gpu/ps4/cmd_trace.h"

namespace gpu::ps4 {
namespace {

constexpr u64 kGoldenRatio64 = 0x9e3779b97f4a7c15ull;

void MixHash(u64& hash, u64 value) {
  hash ^= value + kGoldenRatio64 + (hash << 6) + (hash >> 2);
}

// Two draws share a module iff their keys match. `vs`/`ps` are code content
// hashes rather than addresses (see GraphicsKeyOf).
struct GraphicsKey {
  u64 vs = 0, ps = 0, fetch = 0;
  u32 ps_input_ena = 0;
  u32 ps_in_cntl_hash = 0;
  u32 tex_3d_mask = 0;
  u32 tex_1d_mask = 0;
  u32 tex_uint_mask = 0;
  u32 mrt_uint_mask = 0;
  u32 mrt_bound_mask = 0xFF;
  bool gl_clip = false;
  bool neo = false;

  bool operator==(const GraphicsKey& other) const = default;
};

struct GraphicsKeyHash {
  size_t operator()(const GraphicsKey& k) const {
    u64 h = k.vs ^ (k.ps + kGoldenRatio64 + (k.vs << 6) + (k.vs >> 2));
    MixHash(h, k.fetch);
    MixHash(h, k.ps_input_ena);
    MixHash(h, k.ps_in_cntl_hash);
    MixHash(h, k.tex_3d_mask);
    MixHash(h, k.tex_1d_mask);
    MixHash(h, k.tex_uint_mask);
    MixHash(h, k.mrt_uint_mask);
    MixHash(h, k.mrt_bound_mask);
    h ^= k.gl_clip ? kGoldenRatio64 : 0ull;
    h ^= static_cast<u64>(k.neo) << 63;
    return static_cast<size_t>(h);
  }
};

struct ComputeKey {
  u64 address = 0;
  u32 thread_x = 0, thread_y = 0, thread_z = 0;
  u32 user_sgpr = 0, tgid_enable = 0, lds_dwords = 0;
  bool neo = false;

  bool operator==(const ComputeKey& other) const = default;
};

struct ComputeKeyHash {
  size_t operator()(const ComputeKey& key) const {
    u64 h = std::hash<u64>{}(key.address);
    MixHash(h, key.thread_x);
    MixHash(h, key.thread_y);
    MixHash(h, key.thread_z);
    MixHash(h, key.user_sgpr);
    MixHash(h, key.tgid_enable);
    MixHash(h, key.lds_dwords);
    MixHash(h, key.neo);
    return static_cast<size_t>(h);
  }
};

bool UsesGetPc(const gcn::Program& program) {
  for (const gcn::Inst& inst : program)
    if (inst.enc == gcn::Enc::kSop1 && inst.opcode == 0x1f)
      return true;
  return false;
}

// FNV-1a over the OFFSET field of every input-control slot plus NUM_INTERP: the
// module bakes the mapping into its input Locations, so the same code under a
// different mapping is another module.
u32 PsInputCntlHash(const u32* ps_in_cntl, u32 num_interp) {
  u32 hash = 2166136261u;
  for (u32 i = 0; i < 32; i++)
    hash = (hash ^ (ps_in_cntl[i] & 0x3F)) * 16777619u;
  return (hash ^ (num_interp & 0x3F)) * 16777619u;
}

GraphicsKey GraphicsKeyOf(const GraphicsShaderState& state) {
  GraphicsKey key;
  // The VS/PS by CONTENT, like the fetch shader: SotC streams shader code to
  // fresh guest addresses as the title progresses, so an address key misses
  // forever and the frame drowns in recompiles. s_getpc_b64 stays sound because
  // the module reads its own address from the push range per draw
  // (PushCodeBase); at the 128-byte push floor, where the address is baked into
  // the module instead, getpc programs keep the address key.
  key.vs = gcn::CachedCodeHash(state.vs_addr, 4096);
  key.ps = state.ps_addr ? gcn::CachedCodeHash(state.ps_addr, 4096) : 0;
  if (!gcn::PushCodeBase() &&
      (UsesGetPc(*gcn::CachedProgram(state.vs_addr, 4096)) ||
       (state.ps_addr &&
        UsesGetPc(*gcn::CachedProgram(state.ps_addr, 4096))))) {
    key.vs = state.vs_addr;
    key.ps = state.ps_addr;
  }
  // The fetch shader by CONTENT too: titles that generate one per draw into
  // scratch memory (Tomb Raider does) would otherwise miss on every draw.
  key.fetch = gcn::CachedCodeHash(state.fetch_addr, 64);
  key.ps_input_ena = state.ps_input_ena;
  key.ps_in_cntl_hash = PsInputCntlHash(state.ps_in_cntl, state.ps_num_interp);
  key.tex_3d_mask = state.tex_3d_mask;
  key.tex_1d_mask = state.tex_1d_mask;
  key.tex_uint_mask = state.tex_uint_mask;
  key.mrt_uint_mask = state.mrt_uint_mask;
  key.mrt_bound_mask = state.mrt_bound_mask;
  key.gl_clip = state.gl_clip;
  key.neo = gcn::DefaultIsaMode() == gcn::IsaMode::kNeo;
  return key;
}

}  // namespace

const gcn::Recompiled& GetGraphicsShader(const Regs& regs,
                                         const GraphicsShaderState& state) {
  static std::unordered_map<GraphicsKey, gcn::Recompiled, GraphicsKeyHash>
      cache;
  const GraphicsKey key = GraphicsKeyOf(state);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;

  TraceShaderCacheMiss(
      state.vs_addr, state.ps_addr, gcn::CachedCodeHash(state.vs_addr, 4096),
      state.ps_addr ? gcn::CachedCodeHash(state.ps_addr, 4096) : 0, key.fetch,
      state.ps_input_ena, state.tex_3d_mask, state.tex_1d_mask);
  return cache
      .emplace(key,
               gcn::Recompile(
                   reinterpret_cast<const u32*>(state.vs_addr),
                   reinterpret_cast<const u32*>(state.ps_addr),
                   regs.At(mmSPI_SHADER_USER_DATA_VS_0),
                   regs.At(mmSPI_SHADER_USER_DATA_PS_0), state.ps_input_ena,
                   state.honour_ps_in_cntl ? state.ps_in_cntl : nullptr,
                   state.ps_num_interp, state.tex_3d_mask, state.tex_1d_mask,
                   state.tex_uint_mask, state.mrt_uint_mask,
                   state.mrt_bound_mask, state.gl_clip))
      .first->second;
}

const gcn::RecompiledCs& GetComputeShader(const ComputeShaderState& state) {
  static std::unordered_map<ComputeKey, gcn::RecompiledCs, ComputeKeyHash>
      cache;
  const ComputeKey key{
      state.cs_addr,    state.thread_x,
      state.thread_y,   state.thread_z,
      state.user_sgpr,  state.tgid_enable,
      state.lds_dwords, gcn::DefaultIsaMode() == gcn::IsaMode::kNeo};
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  return cache
      .emplace(key, gcn::RecompileCompute(
                        reinterpret_cast<const u32*>(state.cs_addr),
                        state.thread_x, state.thread_y, state.thread_z,
                        state.user_sgpr, state.tgid_enable, state.lds_dwords))
      .first->second;
}

}  // namespace gpu::ps4
