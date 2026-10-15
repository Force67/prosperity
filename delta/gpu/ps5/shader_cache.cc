/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Which recompiled module a given piece of guest state needs. See
 * shader_cache.h.
 */

#include "gpu/ps5/shader_cache.h"
#include "base/arch.h"
#include "guest/session.h"
#include "ui/shader_activity.h"

#include "gpu/gcn/gcn_resource.h"
#include "gpu/gpu_perf.h"
#include "gpu/ps5/cmd_trace.h"
#include "gpu/ps5/rdna/rdna_compute.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/rdna/rdna_translate.h"

#include "base/containers/array.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/hashing/hash.h"
#include "base/logging.h"
#include "options/options.h"

namespace gpu::ps5 {
namespace {

constexpr u64 kGoldenRatio64 = 0x9e3779b97f4a7c15ull;

void MixHash(u64& hash, u64 value) {
  hash ^= value + kGoldenRatio64 + (hash << 6) + (hash >> 2);
}

struct GraphicsKey {
  u64 vs = 0, ps = 0, fetch = 0;
  u32 vs_user_sgprs = 0;
  u32 ps_user_sgprs = 0;
  u32 ps_input_ena = 0;
  u32 ps_num_interp = 0;
  base::Array<u16, 32> ps_param_slot{};
  bool gl_clip = false;
  u64 gs = 0;
  base::Array<u32, 6> ngg{};
  u64 direct_table = 0;

  bool operator==(const GraphicsKey& other) const = default;
};

struct GraphicsKeyHash {
  size_t operator()(const GraphicsKey& k) const {
    u64 h = k.vs ^ (k.ps + kGoldenRatio64 + (k.vs << 6) + (k.vs >> 2));
    MixHash(h, k.fetch);
    MixHash(h, k.gs);
    MixHash(h, k.direct_table);
    for (u32 value : k.ngg)
      MixHash(h, value);
    MixHash(h, k.vs_user_sgprs);
    MixHash(h, k.ps_user_sgprs);
    MixHash(h, k.ps_input_ena);
    MixHash(h, k.ps_num_interp);
    for (u16 slot : k.ps_param_slot)
      MixHash(h, slot);
    h ^= k.gl_clip ? kGoldenRatio64 : 0ull;
    return static_cast<size_t>(h);
  }
};

struct ComputeKey {
  u64 code = 0;
  u32 thread_x = 0, thread_y = 0, thread_z = 0;
  u32 user_sgpr = 0, tgid_enable = 0, lds_dwords = 0;
  bool trap_present = false;
  bool wave32 = false;
  u64 native_mask = 0, uint_mask = 0;
  bool plan_only = false;

  bool operator==(const ComputeKey& other) const = default;
};

struct ComputeKeyHash {
  size_t operator()(const ComputeKey& key) const {
    u64 h = base::Hash<u64>{}(key.code);
    MixHash(h, key.thread_x);
    MixHash(h, key.thread_y);
    MixHash(h, key.thread_z);
    MixHash(h, key.user_sgpr);
    MixHash(h, key.tgid_enable);
    MixHash(h, key.lds_dwords);
    MixHash(h, key.trap_present);
    MixHash(h, key.wave32);
    MixHash(h, key.native_mask);
    MixHash(h, key.uint_mask);
    MixHash(h, key.plan_only);
    return static_cast<size_t>(h);
  }
};

struct RecompTimer {
  ui::ShaderCompilation compilation;
  u64 t0 = NowNs();
  ~RecompTimer() {
    g_stats.ns_recomp += NowNs() - t0;
    g_stats.recomp_n++;
  }
};

u64 CodeHash(u64 addr, u32 max_dwords) {
  return rdna::CachedCodeHash(reinterpret_cast<const u32*>(addr), max_dwords);
}

DELTA_OPTION(bool, kShMiss, "DELTA_GPU_SHMISS", false);

void ReportMiss(const GraphicsKey& key,
                const GraphicsShaderState& state,
                size_t cache_size) {
  if (!kShMiss)
    return;
  static u32 n = 0;
  if (n++ >= 64)
    return;
  u64 slots = 0;
  for (u16 s : key.ps_param_slot)
    MixHash(slots, s);
  BASE_LOGI("shmiss",
            "n={} cache={} vs={:#x}@{:#x} ps={:#x}@{:#x} fetch={:#x}@{:#x} "
            "vsUd={} psUd={} ena={:#x} interp={} slots={:#x} clip={}",
            n, cache_size, key.vs, state.vs_addr, key.ps, state.ps_addr,
            key.fetch, state.fetch_addr, key.vs_user_sgprs, key.ps_user_sgprs,
            key.ps_input_ena, key.ps_num_interp, slots, (int)key.gl_clip);
  if (state.ps_in_cntl)
    for (u32 i = 0; i < state.ps_num_interp && i < 32; ++i)
      BASE_LOGI("shmiss", "  input[{}]={:#x}", i, state.ps_in_cntl[i]);
}

}  // namespace

const gcn::Recompiled& GetGraphicsShader(const GraphicsShaderState& state) {
  static base::HashMap<GraphicsKey, gcn::Recompiled, GraphicsKeyHash> cache;
  static const guest::SessionReset reset_cache(
      [] { guest::ResetResource(cache); });
  GraphicsKey key{CodeHash(state.vs_addr, 4096),
                  CodeHash(state.ps_addr, 4096),
                  rdna::FetchPlanHash(state.fetch_addr),
                  state.vs_user_sgprs,
                  state.ps_user_sgprs,
                  state.ps_input_ena,
                  state.ps_num_interp};
  key.gl_clip = state.gl_clip;
  key.direct_table = gcn::g_direct_table;
  if (state.ngg.gs_code) {
    key.gs = CodeHash(reinterpret_cast<u64>(state.ngg.gs_code), 4096);
    key.ngg = {state.ngg.threads,      state.ngg.input_primitives,
               state.ngg.max_vertices, state.ngg.max_primitives,
               state.ngg.lds_dwords,   state.ngg.separate_es};
  }
  if (state.ps_in_cntl)
    for (u32 i = 0; i < state.ps_num_interp && i < 32; i++)
      key.ps_param_slot[i] = static_cast<u16>(state.ps_in_cntl[i] & 0x43F);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;

  TraceRecompile(state.vs_addr, state.ps_addr, state.ps_input_ena);
  ReportMiss(key, state, cache.size());
  const RecompTimer timer;
  const gcn::Recompiled& rc =
      cache
          .emplace(
              key,
              rdna::Recompile(
                  reinterpret_cast<const u32*>(state.vs_addr),
                  state.ps_addr ? reinterpret_cast<const u32*>(state.ps_addr)
                                : nullptr,
                  state.vs_user_data, state.ps_user_data, state.ps_input_ena,
                  state.gl_clip, state.vs_user_sgprs, state.ps_user_sgprs,
                  state.ps_in_cntl, state.ps_num_interp,
                  state.ngg.gs_code ? &state.ngg : nullptr))
          .first->second;
  TraceRecompileDone(rc.ok);
  if (!rc.ok)
    TraceRecompileFailed(state.vs_addr, state.ps_addr);
  return rc;
}

const gcn::RecompiledCs& GetComputeShader(const ComputeShaderState& state,
                                          u64 native_mask,
                                          u64 uint_mask,
                                          bool plan_only) {
  static base::HashMap<ComputeKey, gcn::RecompiledCs, ComputeKeyHash> cache;
  static const guest::SessionReset reset_cache(
      [] { guest::ResetResource(cache); });
  const ComputeKey key{
      CodeHash(state.cs_addr, rdna::ComputeCodeDwords(
                                  reinterpret_cast<const u32*>(state.cs_addr))),
      state.thread_x,
      state.thread_y,
      state.thread_z,
      state.user_sgpr,
      state.tgid_enable,
      state.lds_dwords,
      state.trap_present,
      state.wave32,
      native_mask,
      uint_mask,
      plan_only};
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  const RecompTimer timer;
  return cache
      .emplace(key,
               rdna::RecompileCompute(
                   reinterpret_cast<const u32*>(state.cs_addr), state.thread_x,
                   state.thread_y, state.thread_z, state.user_sgpr,
                   state.tgid_enable, state.lds_dwords, state.trap_present,
                   state.wave32, native_mask, uint_mask, plan_only))
      .first->second;
}

}  // namespace gpu::ps5
