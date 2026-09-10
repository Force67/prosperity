/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Which recompiled module a given piece of guest state needs. See
 * shader_cache.h.
 */

#include "gpu/ps5/shader_cache.h"
#include "base/arch.h"

#include <functional>
#include <array>
#include <unordered_map>

#include "gpu/ps5/cmd_trace.h"
#include "gpu/ps5/rdna/rdna_compute.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/rdna/rdna_translate.h"
#include "gpu/gpu_perf.h"

#include <base/logging.h>
#include <utl/options.h>

namespace gpu::ps5 {
namespace {

constexpr u64 kGoldenRatio64 = 0x9e3779b97f4a7c15ull;

void MixHash(u64& hash, u64 value) {
  hash ^= value + kGoldenRatio64 + (hash << 6) + (hash >> 2);
}

// Two draws share a module iff their keys match. Attribute translation depends
// on the fetch program, so the triple is the identity, not the VS/PS pair.
// `vs`/`ps`/`fetch` are hashes of the code, not the addresses it happened to be
// at (see GraphicsKeyOf).
struct GraphicsKey {
  u64 vs = 0, ps = 0, fetch = 0;
  u32 vs_user_sgprs = 0;
  u32 ps_user_sgprs = 0;
  u32 ps_input_ena = 0;
  u32 ps_num_interp = 0;
  // The OFFSET fields that are defined for this draw, in slot order.
  std::array<u8, 32> ps_param_slot{};
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
    MixHash(h, k.ps_num_interp);
    for (u8 slot : k.ps_param_slot)
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

  bool operator==(const ComputeKey& other) const = default;
};

struct ComputeKeyHash {
  size_t operator()(const ComputeKey& key) const {
    u64 h = std::hash<u64>{}(key.code);
    MixHash(h, key.thread_x);
    MixHash(h, key.thread_y);
    MixHash(h, key.thread_z);
    MixHash(h, key.user_sgpr);
    MixHash(h, key.tgid_enable);
    MixHash(h, key.lds_dwords);
    MixHash(h, key.trap_present);
    return static_cast<size_t>(h);
  }
};

// The PS4 path times its recompiles inside gcn::Recompile; the RDNA entry
// points are reached directly, so without this `sh=` on the FPS line reads zero
// on PS5 no matter how long a burst of first-use shaders takes.
struct RecompTimer {
  u64 t0 = NowNs();
  ~RecompTimer() {
    gcn::g_ns_recomp += NowNs() - t0;
    gcn::g_recomp_n++;
  }
};

// The code identifies the shader, not the address holding it. AGC hands
// programs out of a pool the title refills, so Dead Cells reaches the same
// dozen shaders through a fresh address almost every draw: keyed by address
// this cache missed ~500 times every two seconds and spent a whole core
// recompiling shaders it already had, forever, with 200-600 ms frames where the
// misses clustered. Nothing in an RDNA module depends on where the code sat --
// the address reaches the translator only as a readability check and a log
// line -- so the content hash is the whole identity.
u64 CodeHash(u64 addr, u32 max_dwords) {
  return rdna::CachedCodeHash(reinterpret_cast<const u32*>(addr), max_dwords);
}

// DELTA_GPU_SHMISS=1: what a module cache miss was keyed on. A miss rate that
// does not fall when the key changes means the key is not the churning field,
// and only the fields themselves say which one is.
DELTA_OPTION(bool, kShMiss, "DELTA_GPU_SHMISS", false);

void ReportMiss(const GraphicsKey& key, const GraphicsShaderState& state,
                size_t cache_size) {
  if (!kShMiss)
    return;
  static u32 n = 0;
  if (n++ >= 64)
    return;
  u64 slots = 0;
  for (u8 s : key.ps_param_slot)
    MixHash(slots, s);
  BASE_LOGI("shmiss",
            "n={} cache={} vs={:#x}@{:#x} ps={:#x}@{:#x} fetch={:#x}@{:#x} "
            "vsUd={} psUd={} ena={:#x} interp={} slots={:#x} clip={}",
            n, cache_size, key.vs, state.vs_addr, key.ps, state.ps_addr,
            key.fetch, state.fetch_addr, key.vs_user_sgprs, key.ps_user_sgprs,
            key.ps_input_ena, key.ps_num_interp, slots, (int)key.gl_clip);
}

}  // namespace

const gcn::Recompiled& GetGraphicsShader(const GraphicsShaderState& state) {
  static std::unordered_map<GraphicsKey, gcn::Recompiled, GraphicsKeyHash>
      cache;
  GraphicsKey key{CodeHash(state.vs_addr, 4096),
                  CodeHash(state.ps_addr, 4096),
                  rdna::FetchPlanHash(state.fetch_addr),
                  state.vs_user_sgprs,
                  state.ps_user_sgprs,
                  state.ps_input_ena,
                  state.ps_num_interp};
  key.gl_clip = state.gl_clip;
  if (state.ps_in_cntl)
    for (u32 i = 0; i < state.ps_num_interp && i < 32; i++)
      key.ps_param_slot[i] = static_cast<u8>(state.ps_in_cntl[i] & 0x1F);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;

  TraceRecompile(state.vs_addr, state.ps_addr, state.ps_input_ena);
  ReportMiss(key, state, cache.size());
  const RecompTimer timer;
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
                       state.ps_user_sgprs, state.ps_in_cntl,
                       state.ps_num_interp))
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
  const ComputeKey key{CodeHash(state.cs_addr, rdna::ComputeCodeDwords(reinterpret_cast<const u32*>(state.cs_addr))),
                       state.thread_x,
                       state.thread_y,
                       state.thread_z,
                       state.user_sgpr,
                       state.tgid_enable,
                       state.lds_dwords,
                       state.trap_present};
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  const RecompTimer timer;
  return cache
      .emplace(key,
               rdna::RecompileCompute(
                   reinterpret_cast<const u32*>(state.cs_addr), state.thread_x,
                   state.thread_y, state.thread_z, state.user_sgpr,
                   state.tgid_enable, state.lds_dwords, state.trap_present))
      .first->second;
}

}  // namespace gpu::ps5
