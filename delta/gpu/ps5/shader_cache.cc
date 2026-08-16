/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Which recompiled module a given piece of guest state needs. See
 * shader_cache.h.
 */

#include "gpu/ps5/shader_cache.h"
#include "base/arch.h"

#include <functional>
#include <unordered_map>

#include "gpu/ps5/cmd_trace.h"
#include "gpu/ps5/rdna/rdna_compute.h"
#include "gpu/ps5/rdna/rdna_translate.h"

namespace gpu::ps5 {
namespace {

constexpr u64 kGoldenRatio64 = 0x9e3779b97f4a7c15ull;

void MixHash(u64& hash, u64 value) {
  hash ^= value + kGoldenRatio64 + (hash << 6) + (hash >> 2);
}

// Two draws share a module iff their keys match. Attribute translation depends
// on the fetch program, so the triple is the identity, not the VS/PS pair.
struct GraphicsKey {
  u64 vs = 0, ps = 0, fetch = 0;
  u32 vs_user_sgprs = 0;
  u32 ps_user_sgprs = 0;
  u32 ps_input_ena = 0;
  bool gl_clip = false;

  bool operator==(const GraphicsKey& other) const = default;
};

struct GraphicsKeyHash {
  size_t operator()(const GraphicsKey& k) const {
    u64 h = k.vs ^ (k.ps + kGoldenRatio64 + (k.vs << 6) + (k.vs >> 2));
    MixHash(h, k.fetch);
    MixHash(h, k.vs_user_sgprs);
    MixHash(h, k.ps_user_sgprs);
    MixHash(h, k.ps_input_ena);
    h ^= k.gl_clip ? kGoldenRatio64 : 0ull;
    return static_cast<size_t>(h);
  }
};

struct ComputeKey {
  u64 address = 0;
  u32 thread_x = 0, thread_y = 0, thread_z = 0;
  u32 user_sgpr = 0, tgid_enable = 0, lds_dwords = 0;

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
    return static_cast<size_t>(h);
  }
};

}  // namespace

const gcn::Recompiled& GetGraphicsShader(const GraphicsShaderState& state) {
  static std::unordered_map<GraphicsKey, gcn::Recompiled, GraphicsKeyHash>
      cache;
  const GraphicsKey key{state.vs_addr,       state.ps_addr,
                        state.fetch_addr,    state.vs_user_sgprs,
                        state.ps_user_sgprs, state.ps_input_ena,
                        state.gl_clip};
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;

  TraceRecompile(state.vs_addr, state.ps_addr, state.ps_input_ena);
  const gcn::Recompiled& rc =
      cache
          .emplace(key,
                   rdna::Recompile(
                       reinterpret_cast<const u32*>(state.vs_addr),
                       state.ps_addr
                           ? reinterpret_cast<const u32*>(state.ps_addr)
                           : nullptr,
                       state.vs_user_data, state.ps_user_data,
                       state.ps_input_ena, state.gl_clip, state.vs_user_sgprs,
                       state.ps_user_sgprs))
          .first->second;
  TraceRecompileDone(rc.ok);
  // A shader we cannot recompile drops its draw entirely, which is
  // indistinguishable from "the title never issued it" unless it is said out
  // loud.
  if (!rc.ok)
    TraceRecompileFailed(state.vs_addr, state.ps_addr);
  return rc;
}

const gcn::RecompiledCs& GetComputeShader(const ComputeShaderState& state) {
  static std::unordered_map<ComputeKey, gcn::RecompiledCs, ComputeKeyHash>
      cache;
  const ComputeKey key{state.cs_addr,   state.thread_x,    state.thread_y,
                       state.thread_z,  state.user_sgpr,   state.tgid_enable,
                       state.lds_dwords};
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  return cache
      .emplace(key, rdna::RecompileCompute(
                        reinterpret_cast<const u32*>(state.cs_addr),
                        state.thread_x, state.thread_y, state.thread_z,
                        state.user_sgpr, state.tgid_enable, state.lds_dwords))
      .first->second;
}

}  // namespace gpu::ps5
