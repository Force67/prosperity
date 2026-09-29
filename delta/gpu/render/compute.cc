/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// Compute dispatches. Each guest range a CS touches gets a persistent
// host-visible storage buffer keyed by its base address; dispatches are
// recorded into one batched command buffer, and their writes land back in guest
// memory lazily, only when something needs guest memory to be current.

#include "gpu/render/compute.h"
#include "base/arch.h"
#include "gpu/render/guest_memory_table.h"
#include "gpu/render/renderer.h"

#include "gpu/gcn/gcn_detile.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gpu_check.h"
#include "gpu/gpu_perf.h"
#include "gpu/guest_memory.h"
#include "gpu/guest_page_table.h"
#include "gpu/render/capture.h"
#include "gpu/render/compute_hazard.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "gpu/render/guest_format.h"
#include "gpu/render/hash.h"
#include "gpu/render/perf.h"
#include "gpu/render/render_target.h"
#include "gpu/render/renderer_state.h"
#include "gpu/render/texture_cache.h"
#include "gpu/render/tiling.h"
#include "gpu/render/trace.h"
#include "gpu/write_tracker.h"

#include <sys/mman.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "base/logging.h"
#include "base/memory/shared_pointer.h"
#include "base/strings/format.h"
#include "base/strings/to_string.h"
#include "base/strings/xstring.h"
#include "options/options.h"

#define BCDECDEF static inline
#define BCDEC_IMPLEMENTATION
#include <bcdec.h>
#include "base/algorithm.h"
#include "base/containers/array.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/pair.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/bit_cast.h"
#include "base/memory/move.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/time/time.h"
#include "profile/profile.h"
#include "gpu/render/gpu_timeline.h"

namespace {
DELTA_OPTION(u64, kDetileDump, "DELTA_GPU_DETILEDUMP", 0);
DELTA_OPTION(bool, kCsList, "DELTA_GPU_CSLIST", false);
DELTA_OPTION(int, kCsHist, "DELTA_GPU_CSHIST", 0);
DELTA_OPTION(bool, kCsRtTrace, "DELTA_GPU_CSRT", false);
DELTA_OPTION(bool, kCsRename, "DELTA_GPU_CSRENAME", false);
// Re-bridge a render target into its compute staging buffer only when
// something has rendered into it since the last copy. =0 restores the
// every-frame copy if a title ever writes a target by a path last_frame does
// not see.
DELTA_OPTION(bool, kCsRtCache, "DELTA_GPU_CS_RT_CACHE", true);
// Reuse a staged plain buffer when a later dispatch binds a smaller window of
// it instead of restaging the range. =0 restores the exact-size match.
DELTA_OPTION(bool, kCsSubset, "DELTA_GPU_CS_SUBSET", true);
// A compute result whose address is a live GPU image is refreshed on the GPU by
// UploadCsRangeToRt; retiling the same bytes into guest memory as well is only
// there for readers that go through guest memory. Skipping it is the single
// biggest CPU saving in a compute-heavy frame (Astro Bot: ~50 ms).
DELTA_OPTION(bool, kCsSkipRetile, "DELTA_GPU_CS_SKIP_RETILE", false);
// Skip staging in a compute range the shader never reads (see the use below).
DELTA_OPTION(bool, kCsSkipUpload, "DELTA_GPU_CS_SKIP_UPLOAD", false);
u64 g_cs_skip_n = 0;
DELTA_OPTION(bool, kCsImport, "DELTA_GPU_CSIMPORT", false);
// Revert to copying every staged byte back to guest memory, shader-written or
// not (the behaviour before the write-coverage merge). Kept for A/B only: it
// reverts CPU writes that share a staged range and corrupts SotC's heap.
DELTA_OPTION(bool, kCsWbFull, "DELTA_GPU_CS_WB_FULL", false);
// Report every writeback whose dispatch did not write the whole range: the
// difference is guest memory we would have reverted.
DELTA_OPTION(bool, kCsWbAudit, "DELTA_GPU_CS_WB_AUDIT", false);
// Put compute range buffers in VRAM rather than system RAM, with a host-cached
// mirror for the staging edges. The GPU reads and writes these every dispatch,
// and doing that across PCIe was 282 ms of a 457 ms SotC frame. No-op on a UMA
// part, where one buffer is already both (see SplitVram).
DELTA_OPTION(bool, kCsVram, "DELTA_GPU_CSVRAM", true);
DELTA_OPTION(bool, kCsGpuTiling, "DELTA_GPU_CS_GPU_TILING", true);
// Bind the guest pages of a tiled surface straight into the tiling shader
// instead of copying them through a staging buffer both ways. =0 restores the
// copies for A/B.
DELTA_OPTION(bool, kCsTileImport, "DELTA_GPU_CS_TILE_IMPORT", true);
// A draw sampling a surface a dispatch wrote copies it VRAM->image on the
// queue instead of writing it back to guest memory, retiling, hashing and
// re-uploading it. =0 restores the guest round trip.
DELTA_OPTION(bool, kCsTexBridge, "DELTA_GPU_CS_TEX_BRIDGE", true);
// An image range a draw took straight from VRAM this frame (CsSupplyTexture)
// stays there at frame end instead of being retiled into guest memory; a
// guest reader that does turn up (CP DMA, a reshaped dispatch, a vertex
// fetch) still gets the writeback on demand. Ranges with any other consumer
// are written back at frame end as before: one batch wait then covers them
// all, where deferring them costs a wait per reader next frame (measured 24 ->
// 32 syncs a frame in Demon's Souls). The risk is a guest CPU read through
// none of those hooks.
DELTA_OPTION(bool, kCsImageLazyWb, "DELTA_GPU_CS_IMAGE_LAZY_WB", true);
// A range holding a gfx10 image footprint keeps the guest bytes as they are
// (tiled). An image binding reads a detiled scratch the tiling shader fills
// inside the batch, and what it writes is retiled back the same way; a plain
// buffer view of the same memory binds the same bytes. No CPU ever tiles, a
// writeback is a copy, and a descriptor's shape stops mattering. =0 restores
// per-shape linear staging.
DELTA_OPTION(bool, kCsTruth, "DELTA_GPU_CS_TRUTH", true);
DELTA_OPTION(bool, kCsMergeOverlap, "DELTA_GPU_CS_MERGE_OVERLAP", true);
DELTA_OPTION(bool, kCsKeepOverlap, "DELTA_GPU_CS_KEEP_OVERLAP", true);
// Dispatches per batch submit. Small batches are what makes the asynchronous
// ring pay: a reader then waits for a few dispatches' worth of GPU work,
// usually already done, instead of a frame's worth.
DELTA_OPTION(u32, kCsBatchCap, "DELTA_GPU_CS_BATCH_CAP", 8);
// Why a sampled surface a dispatch wrote still takes the guest round trip.
DELTA_OPTION(bool, kCsTexBridgeTrace, "DELTA_GPU_CS_TEXBRIDGE_TRACE", false);
// A compute bridge to or from a target drawn into this frame submits the
// frame's open chunk first (SubmitFrameChunk), so it executes after those
// draws instead of a frame early. =0 restores the old one-frame latency.
DELTA_OPTION(bool, kFrameChunks, "DELTA_GPU_FRAME_CHUNKS", true);
DELTA_OPTION(u64, kCsFlushTrace, "DELTA_GPU_CSFLUSHTRACE", 0);
// DELTA_GPU_CSOVERLAP=<frame>: from that frame on, name the binding and the
// dirty range behind the next "overlap" writebacks (a dispatch input
// straddling another's unflushed output).
DELTA_OPTION(int, kCsOverlapTrace, "DELTA_GPU_CSOVERLAP", 0);
DELTA_OPTION(bool, kCsSyncReport, "DELTA_GPU_CSSYNC", false);
DELTA_OPTION(bool, kCsEagerReadback, "DELTA_GPU_CS_EAGER_READBACK", true);
// Off by default: with the CP DMA flush narrowed, ranges a dispatch rewrites
// every frame stay dirty indefinitely, and GTA:SA then shows ghosted
// post-processing (DELTA_GPU_CS_AGED lists those ranges). Some reader of them
// still misses its flush; the global DMA flush hides it.
DELTA_OPTION(bool, kCsTrack, "DELTA_GPU_CS_TRACK", false);
DELTA_OPTION(bool, kCsAged, "DELTA_GPU_CS_AGED", false);
DELTA_OPTION(bool, kGpuCsgpuVerbose, "DELTA_GPU_CSGPU_VERBOSE", false);
u64 g_cs_image_staged = 0;
}  // namespace

namespace gpu::render {
namespace {
// Writeback-half accounting (DELTA_GPU_CSSYNC); defined here because the
// aliased-image bridge below is the thing being counted.
u64 g_out_retile_ns = 0, g_out_rt_ns = 0, g_out_tail_ns = 0;
u64 g_out_buffer_ns = 0;
u64 g_tex_bridge_n = 0, g_tex_bridge_bytes = 0, g_cs_chunk_splits = 0;
u64 g_buf_bridge_n = 0, g_buf_bridge_bytes = 0, g_buf_bridge_declined = 0;
u64 g_vtx_bridge_n = 0;
u64 g_stage_ro_bytes = 0, g_stage_rw_bytes = 0, g_stage_img_bytes = 0;
u64 g_stage_guest_detile_bytes = 0, g_stage_guest_detile_n = 0;
// Staging-in halves: hashing guest memory to decide validity, CPU detiling,
// the render-target bridge (its own submit+wait), and the plain copy.
u64 g_in_hash_ns = 0, g_in_detile_ns = 0, g_in_rt_ns = 0, g_in_copy_ns = 0;
u64 g_in_hash_n = 0, g_in_rt_n = 0;
// The rest of a stage-in: writebacks forced by an overlapping dirty range or a
// reshape (each a GPU wait plus a retile), buffer allocation, the plain guest
// memcpy and its shadow copy. Without these `in=` is mostly unattributed.
u64 g_in_overlap_ns = 0, g_in_reshape_ns = 0, g_in_alloc_ns = 0;
u64 g_in_membuf_ns = 0, g_in_shadow_ns = 0;
u64 g_in_overlap_n = 0, g_in_reshape_n = 0, g_in_alloc_n = 0;
u64 g_truth_stage_bytes = 0, g_truth_import_n = 0, g_truth_detile_n = 0,
    g_truth_retile_n = 0, g_truth_parent_n = 0, g_truth_fold_n = 0,
    g_truth_rt_n = 0;
// Who forced each writeback: the reader tag of FlushCsWritesRange, or one of
// the internal sites.
base::Map<base::String, u64> g_wb_why;
u64 g_wb_why_bytes = 0;
u64 g_lazy_kept_n = 0, g_lazy_kept_bytes = 0;
u64 g_out_retile_n = 0, g_out_rt_submits = 0;
u64 g_gpu_tiling_ns[2] = {}, g_gpu_tiling_n[2] = {};
// Inside one GPU conversion: the host copies either side of it, the table
// build and command recording, and the submit+fence wait. 34 conversions a
// frame is 125 ms in Demon's Souls, and which of these four it is decides
// whether to batch the submits or to stop copying.
u64 g_tile_hin_ns = 0, g_tile_sync_ns = 0, g_tile_hout_ns = 0;
u64 g_tile_hin_bytes = 0, g_tile_hout_bytes = 0;

// Staging a 4K surface is tens of megabytes, and one core copies ~4 GB/s of
// memory the GPU has just written. Split anything big enough to be worth the
// handoff across the detiler's pool; below that the call is a plain memcpy.
constexpr u64 kParallelCopyMin = 512 * 1024;

void ParallelCopy(void* dst, const void* src, u64 bytes) {
  if (bytes < kParallelCopyMin) {
    std::memcpy(dst, src, bytes);
    return;
  }
  constexpr u64 kChunk = 256 * 1024;
  const u32 chunks = static_cast<u32>((bytes + kChunk - 1) / kChunk);
  auto* d = static_cast<u8*>(dst);
  const auto* s = static_cast<const u8*>(src);
  gcn::DetileParallelWork(chunks, bytes, [&](u32 c0, u32 c1) {
    const u64 off = u64(c0) * kChunk;
    std::memcpy(d + off, s + off,
                base::Min<u64>(u64(c1 - c0) * kChunk, bytes - off));
  });
}

// Per-range staging accounting (DELTA_GPU_CSSYNC). A frame that stages half a
// gigabyte says nothing about what to fix until the bytes are attributed to a
// base and a REASON: guest memory changed, the same base was rebound
// with a different shape, or a render target had to be bridged.
struct StageStat {
  u64 bytes = 0;
  u32 n = 0, first = 0, shape = 0, hash = 0, rt = 0, wait = 0;
  u64 wait_ns = 0;
};
base::HashMap<u64, StageStat> g_stage_stats;

using render::ComputeInfo;

static_assert(ComputeInfo::kMaxResources == gcn::kMaxCsResources);

// A recompiled compute pipeline, cached by CS address: the SPIR-V + binding
// layout depend only on the code, so only the descriptor set + push constants +
// storage buffers are rebuilt per dispatch.
struct CsPipe {
  rhi::Pipeline* pipe = nullptr;
  rhi::PipelineLayout* layout = nullptr;
  rhi::BindGroupLayout* set_layout = nullptr;
  u32 num_res = 0;
  int gds_binding = -1;
};

// The GDS scratchpad: one small buffer for the whole device, which is what the
// hardware's global data share is. Created on first use and left alone
// afterwards; the counters in it are the title's to manage.
struct GdsBuffer {
  rhi::Buffer* buf = nullptr;
  void* map = nullptr;
  static constexpr u64 kBytes = 64 * 1024;
};
GdsBuffer g_gds;

base::HashMap<u64, CsPipe> g_cs_pipes;

// Every read and write: the dependency the old ALL_COMMANDS/MEMORY barriers
// expressed.
constexpr u32 kAccessAll = rhi::kAccessAllRead | rhi::kAccessAllWrite;
constexpr u32 kAccessComputeRW =
    rhi::kAccessComputeRead | rhi::kAccessComputeWrite;
constexpr u32 kAccessCopyRW = rhi::kAccessCopyRead | rhi::kAccessCopyWrite;

// A buffer the GPU alone touches goes to VRAM, with a separate host mirror,
// only when the device heap is NOT already cheap for the CPU to read. On a UMA
// part one host-cached buffer serves both sides.
bool SplitVram() {
  const bool split = !Device().caps().unified_memory;
  static bool logged = false;
  if (!logged) {
    logged = true;
    BASE_LOGI("gpuvk", "cs buffers: {}",
              split ? "VRAM with a host-cached mirror"
                    : "one host-cached buffer (unified memory)");
  }
  return split;
}

// kReadback is the host-cached memory the CPU side of compute wants: reading
// back from a write-combined heap is ~100 MB/s, which is invisible in the code
// and obvious in the numbers.
rhi::Buffer* CreateCsBuffer(u64 size, u32 usage, rhi::MemoryKind memory) {
  rhi::BufferDesc desc;
  desc.size = size;
  desc.usage = usage;
  desc.memory = memory;
  return Device().CreateBuffer(desc);
}

// The set-0 layout every compute pipeline over `num_res` guest ranges uses.
rhi::BindGroupLayout* CsSetLayout(u32 num_res,
                                  int gds_binding,
                                  int guest_memory_binding) {
  rhi::BindGroupLayoutDesc desc;
  for (u32 i = 0; i < num_res; i++)
    desc.bindings.push_back(
        {i, rhi::BindingType::kStorageBuffer, rhi::kStageCompute});
  // The GDS scratchpad sits past the resources; it is ours, not a guest range.
  if (gds_binding >= 0)
    desc.bindings.push_back({static_cast<u32>(gds_binding),
                             rhi::BindingType::kStorageBuffer,
                             rhi::kStageCompute});
  if (guest_memory_binding >= 0)
    desc.bindings.push_back({static_cast<u32>(guest_memory_binding),
                             rhi::BindingType::kStorageBuffer,
                             rhi::kStageCompute});
  return Device().CreateBindGroupLayout(desc);
}

// 16 user-data dwords, then one bound (in dwords) per SSBO binding. The bound
// is what the emitted SSBO accesses clamp to; the SPIR-V block for a compute
// stage declares the same 16 + 48 shape, 256 B total, the driver max this
// backend runs against.
rhi::PipelineLayout* CsPipelineLayout(rhi::BindGroupLayout* set_layout) {
  rhi::PipelineLayoutDesc desc;
  desc.groups = {set_layout};
  desc.push_constant_bytes = 64 * 4;
  desc.push_constant_stages = rhi::kStageCompute;
  return Device().CreatePipelineLayout(desc);
}

rhi::Pipeline* CreateCsPipeline(rhi::PipelineLayout* layout,
                                const base::Vector<u32>& spirv) {
  rhi::ComputePipelineDesc desc;
  desc.layout = layout;
  desc.code = rhi::Code(spirv);
  desc.dispatch_base = true;
  return Device().CreateComputePipeline(desc);
}

// Pipelines compiled ahead of their first dispatch
// (render::PrebuildComputePipeline), by module content and layout shape.
struct PrebuiltCs {
  rhi::BindGroupLayout* set_layout = nullptr;
  rhi::PipelineLayout* layout = nullptr;
  rhi::Pipeline* pipe = nullptr;
};
// Filled in once by the thread building it; a consumer waits for `ready`.
struct PrebuiltSlot {
  base::Mutex mutex;
  base::ConditionVariable built;
  bool ready = false;
  PrebuiltCs value;
};
base::Mutex g_prebuilt_mutex;
base::HashMap<u64, base::SharedPointer<PrebuiltSlot>> g_prebuilt;

u64 PrebuiltKey(const base::Vector<u32>& spirv,
                u32 num_res,
                int guest_memory_binding) {
  u64 h = 1469598103934665603ull;
  for (u32 w : spirv)
    h = (h ^ w) * 1099511628211ull;
  return HashWord(HashWord(h, num_res), static_cast<u32>(guest_memory_binding));
}

CsPipe* GetCsPipe(const ComputeInfo& ci) {
  const u64 key = reinterpret_cast<uintptr_t>(ci.recomp);
  auto it = g_cs_pipes.find(key);
  if (it != g_cs_pipes.end())
    return it->second.num_res == ci.num_res ? &it->second : nullptr;
  CsPipe cp;
  cp.num_res = ci.num_res;
  cp.gds_binding = ci.gds_binding;
  if (ci.gds_binding < 0) {
    base::SharedPointer<PrebuiltSlot> slot;
    {
      base::LockGuard<base::Mutex> lock(g_prebuilt_mutex);
      const auto found = g_prebuilt.find(PrebuiltKey(
          ci.recomp->spirv, ci.num_res, ci.recomp->guest_memory_binding));
      if (found != g_prebuilt.end()) {
        slot = found->second;
        g_prebuilt.erase(found);
      }
    }
    PrebuiltCs prebuilt;
    if (slot) {
      base::UniqueLock<base::Mutex> lock(slot->mutex);
      slot->built.Wait(lock, [&] { return slot->ready; });
      prebuilt = slot->value;
    }
    if (prebuilt.pipe) {
      const u64 started = NowNs();
      cp.set_layout = prebuilt.set_layout;
      cp.layout = prebuilt.layout;
      cp.pipe = prebuilt.pipe;
      g_ns_pipe_build += NowNs() - started;
      g_pipe_build_n++;
      g_cs_pipes[key] = cp;
      return &g_cs_pipes[key];
    }
  }
  cp.set_layout =
      CsSetLayout(ci.num_res, ci.gds_binding, ci.recomp->guest_memory_binding);
  if (!cp.set_layout)
    return nullptr;
  cp.layout = CsPipelineLayout(cp.set_layout);
  if (!cp.layout) {
    Device().Destroy(cp.set_layout);
    return nullptr;
  }
  if (kCsSyncReport)
    BASE_LOGI("cspipe", "compile cs={:#x} words={}", ci.cs_addr,
              ci.recomp->spirv.size());
  const u64 started = NowNs();
  cp.pipe = CreateCsPipeline(cp.layout, ci.recomp->spirv);
  g_ns_pipe_build += NowNs() - started;
  if (kCsSyncReport)
    BASE_LOGI("cspipe", "compiled cs={:#x} ms={:.3f}", ci.cs_addr,
              double(NowNs() - started) / 1e6);

  g_pipe_build_n++;
  if (!cp.pipe) {
    BASE_LOGI("gpuvk", "compute pipeline failed");
    Device().Destroy(cp.layout);
    Device().Destroy(cp.set_layout);
    return nullptr;
  }
  if (Device().caps().debug_labels) {
    char name[32];
    std::snprintf(name, sizeof(name), "cs %#llx",
                  (unsigned long long)ci.cs_addr);
    Device().SetName(cp.pipe, name);
  }
  g_cs_pipes[key] = cp;
  return &g_cs_pipes[key];
}

// Persistent compute staging. Dispatches are serialized behind the batch, so
// one set of staging buffers (+ command list) is reused across every
// dispatch: this avoids the per-dispatch buffer/memory/command-buffer churn
// (~3ms/frame). The buffers are host-cached so the copy-BACK read after the
// dispatch hits cache instead of stalling on write-combined memory (which was
// ~25ms/frame for Doom64's 8 MB atlas: the dominant compute cost).
struct CsStage {
  rhi::Buffer* buf = nullptr;
  void* map = nullptr;
  u64 cap = 0;
};

CsStage g_cs_stage[ComputeInfo::kMaxResources];
// The open batch's command list.
rhi::CommandList* g_cs_list = nullptr;

bool BuildCsImageLayouts(const ComputeInfo::Res& res,
                         gcn::TextureLayout32& tiled,
                         gcn::TextureLayout32& linear) {
  const u32 stage_tiling = res.tiling_idx == 31 ? 31 : 8;
  const bool compressed = res.dfmt == 35 || res.dfmt == 40;
  return res.image_staging &&
         gcn::BuildTextureLayout32(
             tiled, compressed ? (res.width + 3) / 4 : res.width,
             compressed ? (res.height + 3) / 4 : res.height,
             compressed ? (res.pitch + 3) / 4 : res.pitch, res.layers,
             res.mip_levels, res.tiling_idx, res.pow2_pad, res.elem_bytes) &&
         gcn::BuildTextureLayout32(linear, res.width, res.height, res.pitch,
                                   res.layers, res.mip_levels, stage_tiling,
                                   res.pow2_pad, res.stage_elem_bytes) &&
         tiled.size == res.guest_size && linear.size == res.size;
}

float UnpackUnsignedFloat(u32 value, u32 mantissa_bits) {
  const u32 mantissa_mask = (1u << mantissa_bits) - 1;
  const u32 mantissa = value & mantissa_mask;
  const u32 exponent = (value >> mantissa_bits) & 0x1F;
  if (!exponent)
    return static_cast<float>(mantissa) *
           base::BitCast<float>((127u - 14u - mantissa_bits) << 23);
  if (exponent == 0x1F)
    return mantissa ? std::numeric_limits<float>::quiet_NaN()
                    : std::numeric_limits<float>::infinity();
  return base::BitCast<float>((exponent + 112) << 23 |
                              mantissa << (23 - mantissa_bits));
}

// Bit arithmetic rather than frexp/lround: this packs every texel of a
// float-staged R11G11B10 target on each writeback. Rounds to nearest, ties
// away, as lround did.
u32 PackUnsignedFloat(float value, u32 mantissa_bits) {
  const u32 bits = base::BitCast<u32>(value);
  const u32 inf = 0x1Fu << mantissa_bits;
  const u32 exp32 = (bits >> 23) & 0xFF;
  const u32 mant23 = bits & 0x7FFFFF;
  if (exp32 == 0xFF)
    return mant23 ? inf | 1u : (bits >> 31 ? 0 : inf);
  if (bits >> 31)
    return 0;
  const u32 drop = 23 - mantissa_bits;
  if (exp32 > 112) {
    const u32 packed =
        (((exp32 - 112) << 23 | mant23) + (1u << (drop - 1))) >> drop;
    return base::Min(packed, inf);
  }
  // Denormal in the target: the implicit bit joins the shifted mantissa.
  const u32 shift = drop + 113 - (exp32 ? exp32 : 1);
  const u32 m = exp32 ? mant23 | 0x800000 : mant23;
  return shift < 32 ? (m + (1u << (shift - 1))) >> shift : 0;
}

void UnpackR11G11B10(u32 packed, u8* dst) {
  const float value[4] = {
      UnpackUnsignedFloat(packed, 6),
      UnpackUnsignedFloat(packed >> 11, 6),
      UnpackUnsignedFloat(packed >> 22, 5),
      1.f,
  };
  std::memcpy(dst, value, sizeof(value));
}

u32 PackR11G11B10(const u8* src) {
  float value[4];
  std::memcpy(value, src, sizeof(value));
  return PackUnsignedFloat(value[0], 6) |
         (PackUnsignedFloat(value[1], 6) << 11) |
         (PackUnsignedFloat(value[2], 5) << 22);
}

bool StageCsImage(const ComputeInfo::Res& res, void* dst) {
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(res, tiled, linear))
    return false;
  const bool bc6 = res.dfmt == 40;
  const bool bc1 = res.dfmt == 35;
  const bool direct = !bc6 && !bc1 && res.elem_bytes == res.stage_elem_bytes;
  bool fills_complete_layout = direct;
  u64 filled_bytes = 0;
  for (u32 mip = 0; fills_complete_layout && mip < linear.mip_levels; ++mip) {
    const auto& level = linear.mips[mip];
    const u64 logical_bytes = static_cast<u64>(level.width) * level.height *
                              linear.layers * res.stage_elem_bytes;
    fills_complete_layout =
        level.offset == filled_bytes && level.pitch == level.width &&
        level.stored_height == level.height && level.size == logical_bytes;
    filled_bytes = level.offset + level.size;
  }
  fills_complete_layout = fills_complete_layout && filled_bytes == res.size;
  if (!fills_complete_layout)
    std::memset(dst, 0, res.size);
  base::Vector<u8> tight;
  if (!direct)
    tight.resize(static_cast<size_t>(tiled.mips[0].width) *
                 tiled.mips[0].height * res.elem_bytes);
  // DELTA_GPU_DETILEDUMP=<base>: write the de-tiled level-0 bytes of that guest
  // surface to <dumpdir>/detiled.bin once, so the swizzle can be checked
  // against an offline decode of the same texture.
  static bool detile_dumped = false;
  for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& src_level = tiled.mips[mip];
    const auto& dst_level = linear.mips[mip];
    for (u32 layer = 0; layer < tiled.layers; layer++) {
      u8* level_dst = static_cast<u8*>(dst) + dst_level.offset +
                      static_cast<u64>(layer) * dst_level.pitch *
                          dst_level.stored_height * res.stage_elem_bytes;
      if (direct) {
        if (!gcn::DetileTextureMip32Pitched(
                reinterpret_cast<const void*>(res.base), level_dst,
                static_cast<size_t>(dst_level.pitch) * res.stage_elem_bytes,
                tiled, mip, layer))
          return false;
        if (kDetileDump && res.base == kDetileDump && !mip && !layer &&
            !detile_dumped) {
          detile_dumped = true;
          char p[256];
          std::snprintf(p, sizeof p, "%s/detiled.bin", DumpDir());
          if (FILE* f = std::fopen(p, "wb")) {
            std::fwrite(level_dst, 1,
                        static_cast<size_t>(dst_level.pitch) *
                            dst_level.stored_height * res.stage_elem_bytes,
                        f);
            std::fclose(f);
            BASE_LOGI("detiledump", "{:#x} pitch={} h={} elem={}",
                      (unsigned long)res.base, dst_level.pitch,
                      dst_level.stored_height, res.stage_elem_bytes);
          }
        }
        continue;
      }
      if (!gcn::DetileTextureMip32(reinterpret_cast<const void*>(res.base),
                                   tight.data(), tiled, mip, layer))
        return false;
      if (bc1) {
        gcn::DetileParallelRows(src_level.height, [&](u32 y0, u32 y1) {
          for (u32 by = y0; by < y1; ++by)
            for (u32 bx = 0; bx < src_level.width; ++bx) {
              u8 rgba[4][4][4];
              const u8* block =
                  tight.data() +
                  (static_cast<size_t>(by) * src_level.width + bx) * 8;
              bcdec_bc1(block, rgba, 16);
              for (u32 y = 0; y < 4 && by * 4 + y < dst_level.height; ++y) {
                const size_t offset =
                    (static_cast<size_t>(by * 4 + y) * dst_level.pitch +
                     bx * 4) *
                    4;
                std::memcpy(level_dst + offset, rgba[y],
                            base::Min(4u, dst_level.width - bx * 4) * 4);
              }
            }
        });
        continue;
      }
      if (bc6) {
        gcn::DetileParallelRows(src_level.height, [&](u32 y0, u32 y1) {
          for (u32 by = y0; by < y1; ++by) {
            for (u32 bx = 0; bx < src_level.width; ++bx) {
              float rgb[4][4][3];
              const u8* block =
                  tight.data() +
                  (static_cast<size_t>(by) * src_level.width + bx) * 16;
              bcdec_bc6h_float(block, rgb, 12, res.nfmt == 1);
              for (u32 y = 0; y < 4 && by * 4 + y < dst_level.height; ++y) {
                for (u32 x = 0; x < 4 && bx * 4 + x < dst_level.width; ++x) {
                  const float rgba[4] = {rgb[y][x][0], rgb[y][x][1],
                                         rgb[y][x][2], 1.f};
                  const size_t offset =
                      (static_cast<size_t>(by * 4 + y) * dst_level.pitch +
                       bx * 4 + x) *
                      16;
                  std::memcpy(level_dst + offset, rgba, sizeof(rgba));
                }
              }
            }
          }
        });
        continue;
      }
      gcn::DetileParallelRows(src_level.height, [&](u32 y0, u32 y1) {
        for (u32 y = y0; y < y1; y++) {
          u8* dst_row = level_dst + static_cast<size_t>(y) * dst_level.pitch *
                                        res.stage_elem_bytes;
          const u8* src_row = tight.data() + static_cast<size_t>(y) *
                                                 src_level.width *
                                                 res.elem_bytes;
          if (res.dfmt == 6) {
            for (u32 x = 0; x < src_level.width; x++) {
              u32 packed;
              std::memcpy(&packed, src_row + static_cast<size_t>(x) * 4, 4);
              UnpackR11G11B10(packed, dst_row + static_cast<size_t>(x) * 16);
            }
          } else if (res.elem_bytes == 1) {
            for (u32 x = 0; x < src_level.width; x++) {
              const u32 expanded = src_row[x];
              std::memcpy(dst_row + static_cast<size_t>(x) * 4, &expanded, 4);
            }
          } else {
            for (u32 x = 0; x < src_level.width; x++) {
              u16 value;
              std::memcpy(&value, src_row + static_cast<size_t>(x) * 2, 2);
              const u32 expanded = value;
              std::memcpy(dst_row + static_cast<size_t>(x) * 4, &expanded, 4);
            }
          }
        }
      });
    }
  }
  return true;
}

bool WritebackCsImage(const ComputeInfo::Res& res, const void* src) {
  if (res.dfmt == 35 || res.dfmt == 40)
    return false;  // Compressed textures are sampled, never image-stored.
  gcn::TextureLayout32 tiled, linear;
  // A bail here leaves the destination holding whatever it held before the
  // dispatch, which for a first upload is zeros, indistinguishable from a
  // dispatch that never ran unless it says so (DELTA_GPU_CS_WB_AUDIT).
  if (!BuildCsImageLayouts(res, tiled, linear)) {
    if (kCsWbAudit) {
      static int n = 0;
      if (n++ < 16) {
        gcn::TextureLayout32 t2, l2;
        const u32 stage_tiling = res.tiling_idx == 31 ? 31 : 8;
        const bool tok = gcn::BuildTextureLayout32(
            t2, res.width, res.height, res.pitch, res.layers, res.mip_levels,
            res.tiling_idx, res.pow2_pad, res.elem_bytes);
        const bool lok = gcn::BuildTextureLayout32(
            l2, res.width, res.height, res.pitch, res.layers, res.mip_levels,
            stage_tiling, res.pow2_pad, res.stage_elem_bytes);
        BASE_LOGI("cswb",
                  "image layout REJECT base={:#x} {}x{} layers={} mips={} "
                  "tiling={} elem={}/{} tiledOk={}({} vs guest {}) "
                  "linearOk={}({} vs size {})",
                  (unsigned long long)res.base, res.width, res.height,
                  res.layers, res.mip_levels, res.tiling_idx, res.elem_bytes,
                  res.stage_elem_bytes, (int)tok,
                  (unsigned long long)(tok ? t2.size : 0),
                  (unsigned long long)res.guest_size, (int)lok,
                  (unsigned long long)(lok ? l2.size : 0),
                  (unsigned long long)res.size);
      }
    }
    return false;
  }
  const bool direct = res.elem_bytes == res.stage_elem_bytes;
  base::Vector<u8> tight;
  if (!direct)
    tight.resize(static_cast<size_t>(res.width) * res.height * res.elem_bytes);
  for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& dst_level = tiled.mips[mip];
    const auto& src_level = linear.mips[mip];
    for (u32 layer = 0; layer < tiled.layers; layer++) {
      const u8* level_src = static_cast<const u8*>(src) + src_level.offset +
                            static_cast<u64>(layer) * src_level.pitch *
                                src_level.stored_height * res.stage_elem_bytes;
      if (direct) {
        if (!gcn::RetileTextureMip32Pitched(
                level_src,
                static_cast<size_t>(src_level.pitch) * res.stage_elem_bytes,
                reinterpret_cast<void*>(res.base), tiled, mip, layer)) {
          if (kCsWbAudit) {
            static int n = 0;
            if (n++ < 16)
              BASE_LOGI("cswb",
                        "image retile REJECT base={:#x} mip={} "
                        "layer={} tiling={}",
                        (unsigned long long)res.base, mip, layer,
                        res.tiling_idx);
          }
          return false;
        }
        continue;
      }
      gcn::DetileParallelRows(dst_level.height, [&](u32 y0, u32 y1) {
        for (u32 y = y0; y < y1; y++) {
          u8* dst_row = tight.data() + static_cast<size_t>(y) *
                                           dst_level.width * res.elem_bytes;
          const u8* src_row = level_src + static_cast<size_t>(y) *
                                              src_level.pitch *
                                              res.stage_elem_bytes;
          if (res.dfmt == 6) {
            for (u32 x = 0; x < dst_level.width; x++) {
              const u32 packed =
                  PackR11G11B10(src_row + static_cast<size_t>(x) * 16);
              std::memcpy(dst_row + static_cast<size_t>(x) * 4, &packed, 4);
            }
          } else if (res.elem_bytes == 1) {
            for (u32 x = 0; x < dst_level.width; x++) {
              u32 expanded;
              std::memcpy(&expanded, src_row + static_cast<size_t>(x) * 4, 4);
              dst_row[x] = static_cast<u8>(expanded);
            }
          } else {
            for (u32 x = 0; x < dst_level.width; x++) {
              u32 expanded;
              std::memcpy(&expanded, src_row + static_cast<size_t>(x) * 4, 4);
              const u16 value = static_cast<u16>(expanded);
              std::memcpy(dst_row + static_cast<size_t>(x) * 2, &value, 2);
            }
          }
        }
      });
      if (!gcn::RetileTextureMip32(tight.data(),
                                   reinterpret_cast<void*>(res.base), tiled,
                                   mip, layer))
        return false;
    }
  }
  return true;
}

// GPU-resident compute working set. Each guest range a CS touches gets a
// persistent host-visible storage buffer keyed by its base address. Staged
// content persists across dispatches and frames: a range the GPU wrote
// (gpu_dirty) is the newest copy and is bound directly with no re-staging;

// guest-sourced ranges revalidate against a content hash at most once per
// frame. Writebacks to guest memory (the expensive image retile) happen
// LAZILY, only when a draw / DMA / frame boundary needs guest memory to be
// current (FlushCsWrites), not after every dispatch.
struct CsRange {
  rhi::Buffer* buf = nullptr;
  void* map = nullptr;
  u64 cap = 0;
  // DELTA_GPU_CSVRAM: `buf` is in VRAM (what the shaders bind) and `map` is a
  // separate host-cached mirror, moved across by DMA at the staging edges. The
  // shaders then read and write at VRAM bandwidth instead of over PCIe, which
  // is worth ~6x of the GPU time in a SotC frame; the CPU keeps a cached
  // pointer, which the mapped-VRAM alternative does not.
  rhi::Buffer* host_buf = nullptr;
  bool device_local = false;
  bool readback_pending = false;  // copy recorded, not yet waited on
  bool mirror_current = false;    // map holds the buffer's contents
  u64 size = 0;                   // active staged (linear) byte size
  u64 guest_bytes = 0;            // guest footprint (hash + overlap checks)
  u64 hash = 0;                   // TexHash of guest content when last in sync
  int last_validated_frame = -1;
  int last_used_frame = -1;
  bool gpu_dirty = false;      // buffer newer than guest memory
  bool pending_batch = false;  // referenced by a batch not yet seen complete
  u64 batch_id = 0;            // that batch (see CsBatchWaitId)
  bool image_staging = false;
  // Buffer memory IS the guest pages (an imported host allocation): no
  // staging copy in, no writeback out. See CsRangeImportGuest.
  bool imported = false;
  u64 imported_base = 0;
  u64 imported_offset = 0;  // base - page-aligned import base
  // Staged from a live render-target image rather than guest memory; content
  // changes every frame regardless of the guest bytes, so validity is
  // per-frame (last_rt_frame), never the guest content hash.
  bool rt_sourced = false;
  int last_rt_frame = -1;
  u64 rt_serial = 0;  // the live target's render_serial the staged copy holds
  ComputeInfo::Res res;  // writeback needs the full layout description
  // A copy of the guest bytes as they were staged IN, kept so the writeback can
  // tell "the shader wrote this word" from "the shader never touched it".
  // Without it the writeback has to assume the whole range is GPU output and
  // copies the stale snapshot back over anything the CPU changed meanwhile –
  // see CsRangeFlushOne.
  base::Vector<u8> shadow;
  bool shadow_valid = false;
  // Bumped whenever the buffer's contents change (a dispatch wrote it, or it
  // was staged in again), so a texture bridged from it knows when to re-copy.
  u64 write_seq = 1;
  // The last frame whose command buffer reads `buf` (CsSupplyTexture): the
  // buffer outlives that frame. `chunk_ref` is the chunk the read was
  // recorded into; a rewrite while that chunk is still open submits it first.
  int frame_ref = -1;
  u64 chunk_ref = 0;
  // The buffer holds the guest bytes of a gfx10 image footprint as they are
  // (DELTA_GPU_CS_TRUTH); `size` == `guest_bytes`, and image bindings read a
  // detiled scratch instead of it.
  bool truth = false;
  u64 rt_seq = 0;  // write_seq the aliased render target was refreshed at
  // A guest-memory reader other than the frame end has pulled this range back
  // before (a CMASK or HTILE the RT bind inspects): its readback is recorded
  // with the dispatch that writes it, so the reader finds the mirror current
  // instead of paying a submit and a wait for the copy alone.
  bool cpu_reader = false;
  // Guest CPU writes into the range while it held dispatch output that is
  // not written back (CsNoteGuestWrites), page aligned and coalesced. They
  // came after the dispatch, so they win over its output.
  base::Vector<base::Pair<u64, u64>> cpu_writes;
  int dirty_frame = -1;  // frame the range last went dirty
};

bool CsSplitFrameChunk();

// The tiled-truth conversions (defined with the tiling shader, below).
struct TileTable;
// `packed`: narrow texels stay packed on the linear side (a texture upload)
// instead of being expanded to dwords (a shader view).
const TileTable* GetTileTable(const gcn::TextureLayout32& tiled,
                              const gcn::TextureLayout32& linear,
                              bool packed = false);
void RecordImageTiling(rhi::CommandList* c,
                       const TileTable& t,
                       rhi::Buffer* tiled,
                       u64 tiled_off,
                       u64 tiled_bytes,
                       rhi::Buffer* linear,
                       u64 linear_bytes,
                       bool detile,
                       u32 mips = 0,
                       u32 layers = 0,
                       bool depth16 = false);
rhi::Buffer* AcquireScratch(u64 bytes);
// A scratch that already holds revision `seq` of the bytes at `base`+`off`
// converted with `table`, or a fresh one (`*fresh` says which).
rhi::Buffer* AcquireView(u64 bytes,
                         u64 base,
                         u64 off,
                         const TileTable* table,
                         u64 seq,
                         bool* fresh);
// What a scratch holds once the conversion recorded into it has run.
void StampView(rhi::Buffer* view,
               u64 base,
               u64 off,
               const TileTable* table,
               u64 seq);
bool RecordTruthImport(CsRange& e, u64 base, u64 bytes);
bool CsRangeEnsureBuffer(CsRange& e, u64 size);
void CsBatchBeginImpl();
void MarkPending(CsRange& e);
// The linear side of a bridge copy for a truth range (its own buffer holds
// tiled bytes), used only inside the bridge's synchronous submit.
CsRange g_truth_bridge_scratch;
u64 g_cs_scratch_owner = 0;  // the dispatch or bridge being recorded
u64 g_cs_scratch_bytes = 0;

// Buffers a frame command list reads: destroyed two BeginFrames after
// retirement, like ReleaseRetiredTextures, once every command list that
// could reference them has retired.
base::Vector<rhi::Buffer*> g_cs_frame_retired;

bool CsFrameReferenced(const CsRange& e) {
  return e.frame_ref >= 0 && e.frame_ref + 2 > g_frame.num;
}

// Evicted range buffers kept by capacity for the next range of that size.
// The guest hands compute a fresh address for its per-frame data every
// frame, so without this every frame was ~50 buffer+memory pairs allocated
// and 20-60 ms of driver time.
struct CsFreeBuffer {
  rhi::Buffer* buf = nullptr;
  rhi::Buffer* host_buf = nullptr;
  void* map = nullptr;
  u64 cap = 0;
  bool device_local = false;
};
base::Vector<CsFreeBuffer> g_cs_free;
u64 g_cs_free_bytes = 0, g_cs_recycled_n = 0;

// Move a split range between its host mirror and its VRAM buffer, bracketed by
// barriers against the dispatches on either side. Recorded rather than
// submitted, so the caller decides which command list carries it.
void RecordStagingCopy(rhi::CommandList* c,
                       CsRange& e,
                       u64 bytes,
                       bool to_device) {
  if (!e.device_local || !e.host_buf || !e.buf || !bytes)
    return;
  if (bytes > e.cap)
    bytes = e.cap;
  // TRANSFER on both sides, not just compute and host. A batch stages the same
  // range more than once (GTA:SA restages some buffers ~170 times in one cs
  // batch), and a barrier that names only COMPUTE|HOST as its source leaves
  // copy-after-copy on the same buffer unordered. The later copy may then land
  // first and the dispatch reads the OLDER guest snapshot.
  c->Barrier(kAccessComputeRW | rhi::kAccessHostWrite | kAccessCopyRW,
             to_device ? rhi::kAccessCopyWrite : rhi::kAccessCopyRead);
  if (to_device)
    c->CopyBuffer(e.buf, 0, e.host_buf, 0, bytes);
  else
    c->CopyBuffer(e.host_buf, 0, e.buf, 0, bytes);
  // Global, so it also covers host_buf, which the readback direction writes.
  c->Barrier(rhi::kAccessCopyWrite,
             kAccessComputeRW | rhi::kAccessHostRead | kAccessCopyRW);
}

bool SameCsResourceShape(const ComputeInfo::Res& a, const ComputeInfo::Res& b) {
  if (a.image_staging != b.image_staging)
    return false;
  if (!a.image_staging)
    return true;
  // RGBA8 sRGB conversion happens in the shader, so both views stage the
  // same bytes when a mip generator reads sRGB and writes through UNORM.
  const bool rgba8_views = a.dfmt == 10 && b.dfmt == 10 &&
                           (a.nfmt == 0 || a.nfmt == 9) &&
                           (b.nfmt == 0 || b.nfmt == 9);
  // A raw-staged surface holds the guest bytes as they are, so descriptors
  // that only disagree on how to read them (a float target bound as 32_32
  // uint for a copy) stage identical bytes. Unpacked, widened and decoded
  // stagings depend on the format.
  auto raw = [](const ComputeInfo::Res& r) {
    return r.elem_bytes == r.stage_elem_bytes && r.dfmt != 6 && r.dfmt != 35 &&
           r.dfmt != 40;
  };
  const bool raw_views = raw(a) && raw(b);
  return a.width == b.width && a.height == b.height && a.pitch == b.pitch &&
         a.layers == b.layers && a.mip_levels == b.mip_levels &&
         a.tiling_idx == b.tiling_idx && a.elem_bytes == b.elem_bytes &&
         a.stage_elem_bytes == b.stage_elem_bytes &&
         (a.dfmt == b.dfmt || raw_views) &&
         (a.nfmt == b.nfmt || rgba8_views || raw_views) &&
         a.pow2_pad == b.pow2_pad;
}

// A live image (color RT or depth target) aliasing a CS resource's guest
// range. Both bridge directions (StageCsRangeFromRt / UploadCsRangeToRt) use
// the same lookup, shape checks and barrier recipe.
struct CsAliasedImage {
  rhi::Texture* texture = nullptr;
  u32 w = 0, h = 0;
  u32 elem_bytes = 0;
  u8 aspect = rhi::kAspectColor;
  rhi::TextureState submitted = rhi::TextureState::kUndefined;
  bool is_depth = false;
  bool is_stencil = false;
  u32 layers = 1;
};

// True when `base` names a live target the compute bridges apply to. The
// UNDEFINED-submitted-layout case (target created this frame, no submission
// yet) reports false: there is nothing real to copy either way yet.
// `for_write`: the dispatch produces the surface, so an image nothing has
// rendered into yet is still the one to fill. Its layout anchor may be missing:
// a parked variant (see ActivateRtVariant) has no submitted-layout stamp, but
// one untouched this frame has nothing recorded against it either, so the
// layout its last frame left it in IS the one on the GPU.
// `prefer_depth`: the resource is a single 32-bit channel, which is what a
// depth surface looks like to a dispatch. One address can hold a colour target
// AND a depth target (Astro Bot's bloom pyramid and its half-res scene depth
// share 0x53a500000 in different passes); the colour one answered first, so the
// depth downsample landed in the bloom image.
bool FindCsAliasedImage(u64 base,
                        CsAliasedImage& out,
                        bool for_write = false,
                        bool prefer_depth = false) {
  if (prefer_depth) {
    auto depth_it = g_depths.find(base);
    if (depth_it != g_depths.end() && depth_it->second.texture) {
      DepthTarget& depth = depth_it->second;
      rhi::TextureState anchor = depth.submitted_layout;
      if (anchor == rhi::TextureState::kUndefined && for_write &&
          depth.last_frame != g_frame.num)
        anchor = depth.layout;
      out = {depth.texture, depth.w, depth.h, 4,           rhi::kAspectDepth,
             anchor,        true,    false,   depth.layers};
      return out.submitted != rhi::TextureState::kUndefined;
    }
  }
  auto rt_it = g_rts.find(base);
  if (rt_it != g_rts.end()) {
    RTarget& rt = rt_it->second;
    if (!rt.texture || rt.is_depth || (!for_write && !rt.ever_rendered))
      return false;
    rhi::TextureState anchor = rt.submitted_layout;
    if (anchor == rhi::TextureState::kUndefined && for_write &&
        rt.last_frame != g_frame.num)
      anchor = rt.layout;
    out = {rt.texture,        rt.w,   rt.h,  FormatBytes(rt.fmt),
           rhi::kAspectColor, anchor, false, false};
    return out.submitted != rhi::TextureState::kUndefined;
  }
  auto depth_it = g_depths.find(base);
  if (depth_it != g_depths.end()) {
    DepthTarget& depth = depth_it->second;
    if (!depth.texture)
      return false;
    out = {depth.texture,
           depth.w,
           depth.h,
           4,  // kDepthFormat == D32_SFLOAT
           rhi::kAspectDepth,
           depth.submitted_layout,
           true,
           false,
           depth.layers};
    return out.submitted != rhi::TextureState::kUndefined;
  }
  for (auto& [depth_base, depth] : g_depths) {
    (void)depth_base;
    if (depth.stencil_base != base || !depth.texture)
      continue;
    out = {depth.texture,
           depth.w,
           depth.h,
           1,
           rhi::kAspectStencil,
           depth.submitted_stencil_layout,
           false,
           true};
    return out.submitted != rhi::TextureState::kUndefined;
  }
  return false;
}

bool LiveTargetOfSize(u64 base, u32 w, u32 h) {
  const auto rt = g_rts.find(base);
  if (rt != g_rts.end()) {
    if (rt->second.w == w && rt->second.h == h)
      return true;
    const auto v = g_rt_variants.find(base);
    if (v != g_rt_variants.end())
      for (const RTarget& t : v->second)
        if (t.w == w && t.h == h)
          return true;
  }
  const auto d = g_depths.find(base);
  if (d != g_depths.end() && d->second.w == w && d->second.h == h)
    return true;
  return base::AnyOf(
      g_depths.begin(), g_depths.end(),
      [base](const auto& entry) { return entry.second.stencil_base == base; });
}

bool CsAliasedBase(u64 base) {
  if (g_rts.find(base) != g_rts.end() || g_depths.find(base) != g_depths.end())
    return true;
  return base::AnyOf(
      g_depths.begin(), g_depths.end(),
      [base](const auto& entry) { return entry.second.stencil_base == base; });
}

// The frame a live image at this address was last rendered into. Unknown
// addresses report "now", so a caller asking "has it changed since?" re-reads
// rather than trusting a stale copy.
u64 AliasedImageRenderSerial(u64 base) {
  auto rt = g_rts.find(base);
  if (rt != g_rts.end())
    return rt->second.render_serial;
  auto d = g_depths.find(base);
  if (d != g_depths.end())
    return d->second.render_serial;
  for (const auto& [depth_base, depth] : g_depths) {
    (void)depth_base;
    if (depth.stencil_base == base)
      return depth.render_serial;
  }
  return UINT64_MAX;
}

int AliasedImageLastRender(u64 base) {
  auto rt = g_rts.find(base);
  if (rt != g_rts.end())
    return rt->second.last_frame;
  auto d = g_depths.find(base);
  if (d != g_depths.end())
    return d->second.last_frame;
  return INT_MAX;
}

// Draws into the target at `base` are still in the frame's open chunk: a
// bridge submitted now would execute before them.
bool RenderedInOpenChunk(u64 base) {
  return kFrameChunks && AliasedImageLastRender(base) == g_frame.num &&
         g_frame.draws != g_frame.draws_at_chunk;
}

// A barrier on the aliased image's aspect over every layer. Global ALL/ALL
// dependency: these bridge copies run where precision buys nothing.
void AliasedImageBarrier(rhi::CommandList* c,
                         const CsAliasedImage& img,
                         rhi::TextureState from,
                         rhi::TextureState to) {
  rhi::TextureBarrier b;
  b.texture = img.texture;
  b.before = from;
  b.after = to;
  b.range.aspect = img.aspect;
  b.range.layers = img.texture->desc().layers;
  c->Barrier(kAccessAll, kAccessAll, &b, 1);
}

// How a bridge copy moves texels between a live image and the CS staging
// buffer. The two descriptions of one surface disagree routinely and for
// reasons that are not defects: the IMAGE is sized from CB_COLOR's
// pitch/slice while the CS descriptor carries the surface's own width and
// height, and a target the shader reads as R11G11B10F is staged as four
// floats per texel because that is what StageCsImage produces on the
// guest-memory path. Demanding exact agreement rejected GTA:SA's scene
// target on both counts, so its whole compute post chain read the guest
// bytes under the target, which draws never write, i.e. black.
struct AliasedCopyPlan {
  u32 w = 0;
  u32 h = 0;
  bool unpack = false;   // image holds packed 11/11/10, staging holds float4
  bool widen = false;    // image holds 8/16-bit texels, staging holds u32
  bool depth16 = false;  // image holds float depth, the surface UNORM16
  u32 layers = 1;
};

bool PlanAliasedCopy(const CsAliasedImage& img,
                     const ComputeInfo::Res& res,
                     const char* dir,
                     AliasedCopyPlan& plan) {
  const u32 want_elem = img.is_stencil ? res.elem_bytes : res.stage_elem_bytes;
  const bool unpack = res.dfmt == 6 && !img.is_depth && !img.is_stencil &&
                      img.elem_bytes == 4 && want_elem == 16;
  const bool widen =
      !img.is_depth && !img.is_stencil && img.elem_bytes == res.elem_bytes &&
      (img.elem_bytes == 1 || img.elem_bytes == 2) && want_elem == 4;
  // A tile row of slack in either direction: an image padded up to its tile
  // height and a descriptor rounded to the surface's own are the same
  // surface, and the copy takes the overlap.
  constexpr u32 kExtentSlack = 8;
  // An array copies whole from a layered target; a 16-bit depth surface
  // takes its texels from the float depth image through the tiling shader.
  const bool layered = res.layers > 1 || img.layers > 1;
  const bool depth16 = img.is_depth && res.elem_bytes == 2 && want_elem == 4;
  if (res.mip_levels == 1 && (!layered || img.is_depth) &&
      img.w + kExtentSlack >= res.width && img.h + kExtentSlack >= res.height &&
      (img.elem_bytes == want_elem || unpack || widen)) {
    plan.w = base::Min(img.w, res.width);
    plan.h = base::Min(img.h, res.height);
    plan.unpack = unpack;
    plan.widen = widen;
    plan.depth16 = depth16;
    // Layers the target has not grown to yet were never drawn.
    plan.layers = base::Max(base::Min(res.layers, img.layers), 1u);
    return true;
  }
  // One line per address and direction. A flat cap spends itself on the level
  // load and then says nothing at all about the steady state.
  static base::HashSet<u64> warned;
  const u64 key = (res.base << 1) | (dir[0] == 'r' ? 1u : 0u);
  if (warned.size() < 256 && warned.insert(key).second)
    BASE_LOGI(
        "gpuvk",
        "cs {} live {} target {:#x} shape mismatch: image {}x{} {}B vs "
        "cs {}x{} pitch={} mips={} layers={} dfmt={} elem={}/{}B tiling={} -> "
        "falling back to guest memory",
        dir,
        img.is_depth     ? "depth"
        : img.is_stencil ? "stencil"
                         : "color",
        (unsigned long long)res.base, img.w, img.h, img.elem_bytes, res.width,
        res.height, res.pitch, res.mip_levels, res.layers, res.dfmt,
        res.elem_bytes, res.stage_elem_bytes, res.tiling_idx);
  return false;
}

// A host-visible scratch the format-converting bridge copies through: the
// image side is packed at the image's own element size, the staging side
// holds the unpacked floats, and no image<->buffer copy converts between the
// two in one step.
struct CsBridgeScratch {
  rhi::Buffer* buf = nullptr;
  void* map = nullptr;
  u64 cap = 0;
};
CsBridgeScratch g_bridge;

bool EnsureBridgeScratch(u64 bytes) {
  if (g_bridge.cap >= bytes)
    return true;
  Device().Destroy(g_bridge.buf);
  g_bridge = CsBridgeScratch{};
  const u64 cap = (bytes + 0xFFFF) & ~u64(0xFFFF);
  g_bridge.buf = CreateCsBuffer(cap, rhi::kBufferCopySrc | rhi::kBufferCopyDst,
                                rhi::MemoryKind::kReadback);
  if (!g_bridge.buf)
    return false;
  g_bridge.map = g_bridge.buf->mapped();
  g_bridge.cap = cap;
  return true;
}

void CsCopyStaging(CsRange& e, u64 bytes, bool to_device);

// Record one bridge copy (image->buffer or buffer->image), submit and wait.
// Device is lost: every later submit fails regardless of what it records, so
// work that only waits must not pretend its failure is a per-staging miss and
// keep recording. Declared before its first possible setter (RunAliasedCopy).
bool g_cs_failed = false;

bool RunAliasedCopy(const CsAliasedImage& img,
                    const ComputeInfo::Res& res,
                    CsRange& e,
                    bool to_image,
                    const AliasedCopyPlan& plan) {
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(res, tiled, linear))
    return false;
  const auto& level = linear.mips[0];
  const u64 texels = static_cast<u64>(level.pitch) * plan.h;
  const u64 copy_bytes = texels * res.stage_elem_bytes;
  if (level.offset + copy_bytes > e.cap)
    return false;
  // Layers sit one stored slice apart in the linear staging layout.
  const u64 layer_bytes = static_cast<u64>(level.pitch) * level.stored_height *
                          res.stage_elem_bytes;
  if (res.layers > 1 && (level.size != layer_bytes * res.layers ||
                         level.offset + level.size > e.cap || plan.unpack ||
                         plan.widen || img.is_stencil))
    return false;
  // The converting path copies at the IMAGE's element size, through a scratch
  // the CPU then unpacks into (or packs out of) the staged layout.
  const u64 packed_bytes = texels * img.elem_bytes;
  const bool convert = plan.unpack || plan.widen;
  if (convert && !EnsureBridgeScratch(packed_bytes))
    return false;
  // A truth range holds no linear pixels: the copy goes through the tiling
  // scratch, detiled from the truth before an upload and retiled into it
  // after a readback. Only the direct copies are bridged that way.
  const bool truth = e.truth;
  if (truth && convert)
    return false;
  // A stencil truth goes through the tiling shader's packed form: one byte
  // per texel on the linear side, which is what the stencil aspect copies.
  const bool stencil_truth = truth && img.is_stencil;
  // Only the tiling shader converts float depth to UNORM16.
  if (plan.depth16 && !truth)
    return false;
  gcn::TextureLayout32 t_tiled, t_linear, s_linear;
  const TileTable* table = nullptr;
  if (truth && !BuildCsImageLayouts(res, t_tiled, t_linear))
    return false;
  if (stencil_truth &&
      (res.layers > 1 ||
       !gcn::BuildTextureLayout32(s_linear, res.width, res.height, res.pitch, 1,
                                  1, 8, res.pow2_pad, 1) ||
       !(table = GetTileTable(t_tiled, s_linear, /*packed=*/true))))
    return false;
  if (truth && !stencil_truth && !(table = GetTileTable(t_tiled, t_linear)))
    return false;
  if (truth && !CsRangeEnsureBuffer(g_truth_bridge_scratch, res.size))
    return false;
  rhi::Buffer* const copy_buf = convert ? g_bridge.buf
                                : truth ? g_truth_bridge_scratch.buf
                                        : e.buf;
  auto* scratch = static_cast<u32*>(g_bridge.map);
  if (convert && to_image) {
    gcn::DetileParallelRows(plan.h, [&](u32 y0, u32 y1) {
      for (u32 y = y0; y < y1; y++) {
        const u8* row =
            static_cast<const u8*>(e.map) + level.offset +
            static_cast<u64>(y) * level.pitch * res.stage_elem_bytes;
        for (u32 x = 0; x < plan.w; x++) {
          const u64 i = static_cast<u64>(y) * level.pitch + x;
          const u8* texel = row + static_cast<u64>(x) * res.stage_elem_bytes;
          if (plan.unpack)
            scratch[i] = PackR11G11B10(texel);
          else
            std::memcpy(static_cast<u8*>(g_bridge.map) + i * img.elem_bytes,
                        texel, img.elem_bytes);
        }
      }
    });
  }
  // Host-zero any padding an image->buffer copy does not cover (host writes
  // are made available by the submission).
  if (!to_image && (convert || level.offset != 0 || copy_bytes < res.size))
    std::memset(e.map, 0, res.size);
  if (img.is_stencil && !truth) {
    if (level.offset || texels > e.cap)
      return false;
    if (!to_image)
      std::memset(e.map, 0, res.size);
    else {
      auto* packed = static_cast<u8*>(e.map);
      auto* expanded = static_cast<u32*>(e.map);
      for (u64 i = 0; i < texels; i++)
        packed[i] = static_cast<u8>(expanded[i]);
    }
  }

  base::String where;
  base::FormatTo(where, "bridge base={:#x}", (unsigned long long)res.base);
  if (!QueueCheck(where.c_str())) {
    BASE_LOGI("gpuvk", "cs {} bridge copy failed: queue check (base={:#x})",
              to_image ? "upload" : "staging", (unsigned long long)res.base);
    g_cs_failed = true;
    Device().ReportDeviceLoss();
    return false;
  }
  rhi::CommandList* c = BeginImmediate();
  if (!c)
    return false;
  // A frame chunk submitted just before this may still read the buffer.
  c->Barrier(kAccessAll, kAccessAll);
  // The dispatch result already lives in VRAM. An ordinary buffer->image copy
  // must read it there, without uploading the identical host readback again.
  // Stencil packing changes the host bytes, and image->buffer staging may have
  // zeroed padding, so those directions still need the host preparation copy.
  // Format conversion uses the separate scratch buffer.
  if (!truth && !convert && (!to_image || img.is_stencil))
    RecordStagingCopy(c, e, e.cap, /*to_device=*/true);
  if (truth && to_image)
    RecordImageTiling(c, *table, e.buf, 0, e.guest_bytes, copy_buf, res.size,
                      /*detile=*/true, 1, plan.layers, plan.depth16);
  if (truth && !to_image &&
      (plan.w < res.width || plan.h < res.height || plan.layers < res.layers ||
       stencil_truth)) {
    // The retile below covers the whole mip; texels the copy leaves alone
    // must not carry a previous conversion's bytes into the truth.
    if (stencil_truth)
      c->FillBuffer(copy_buf, 0, (s_linear.size + 3) & ~u64(3), 0);
    else
      c->FillBuffer(copy_buf, level.offset,
                    layer_bytes * base::Max(res.layers, 1u), 0);
    c->Barrier(rhi::kAccessCopyWrite, rhi::kAccessCopyWrite);
  }
  // Chain from, and restore, the SUBMITTED layout: this copy executes
  // before the current frame's still-recording barriers, whose layout chain
  // must stay intact.
  const rhi::TextureState transfer =
      to_image ? rhi::TextureState::kCopyDst : rhi::TextureState::kCopySrc;
  if (to_image) {
    // The buffer was last written by the dispatch (already waited) or the
    // host; make those writes available to the transfer.
    c->Barrier(rhi::kAccessComputeWrite | rhi::kAccessHostWrite,
               rhi::kAccessCopyRead);
  }
  AliasedImageBarrier(c, img, img.submitted, transfer);
  rhi::BufferTextureCopy copy;
  copy.buffer_offset = (img.is_stencil || convert) ? 0 : level.offset;
  copy.row_length = level.pitch;
  copy.image_height = level.stored_height;
  if (stencil_truth) {
    copy.buffer_offset = s_linear.mips[0].offset;
    copy.row_length = s_linear.mips[0].pitch;
    copy.image_height = s_linear.mips[0].stored_height;
  }
  copy.region.aspect = img.aspect;
  copy.region.layers = plan.layers;
  copy.region.width = plan.w;
  copy.region.height = plan.h;
  if (to_image)
    c->CopyBufferToTexture(img.texture, copy_buf, &copy, 1);
  else
    c->CopyTextureToBuffer(copy_buf, img.texture, &copy, 1);
  if (truth && !to_image) {
    RecordImageTiling(c, *table, e.buf, 0, e.guest_bytes, copy_buf, res.size,
                      /*detile=*/false, 1, base::Max(res.layers, 1u),
                      plan.depth16);
    e.mirror_current = false;
  }
  AliasedImageBarrier(c, img, transfer, img.submitted);
  if (!to_image)
    c->Barrier(rhi::kAccessCopyWrite,
               rhi::kAccessComputeRead | rhi::kAccessHostRead);
  const bool ok = EndImmediate(c);
  g_out_rt_submits++;
  if (!ok) {
    // A device loss is not a "shape mismatch": the queue is dead and every
    // fallback the caller would keep recording with is already doomed. Latch
    // the failure so the caller stops instead of building more work on a
    // lost device.
    BASE_LOGI("gpuvk", "cs {} bridge copy failed (base={:#x})",
              to_image ? "upload" : "staging", (unsigned long long)res.base);
    g_cs_failed = true;
    Device().ReportDeviceLoss();
    return false;
  }
  if (trace::Recording())
    trace::RecordBridge(to_image ? "upload" : "stage", res.base,
                        reinterpret_cast<u64>(img.texture), plan.w, plan.h);
  if (img.is_stencil && !truth) {
    auto* packed = static_cast<u8*>(e.map);
    auto* expanded = static_cast<u32*>(e.map);
    for (u64 i = texels; i-- > 0;)
      expanded[i] = packed[i];
  }
  if (convert && !to_image) {
    gcn::DetileParallelRows(plan.h, [&](u32 y0, u32 y1) {
      for (u32 y = y0; y < y1; y++) {
        u8* row = static_cast<u8*>(e.map) + level.offset +
                  static_cast<u64>(y) * level.pitch * res.stage_elem_bytes;
        for (u32 x = 0; x < plan.w; x++) {
          const u64 i = static_cast<u64>(y) * level.pitch + x;
          u8* texel = row + static_cast<u64>(x) * res.stage_elem_bytes;
          if (plan.unpack)
            UnpackR11G11B10(scratch[i], texel);
          else {
            u32 expanded = 0;
            std::memcpy(
                &expanded,
                static_cast<const u8*>(g_bridge.map) + i * img.elem_bytes,
                img.elem_bytes);
            std::memcpy(texel, &expanded, 4);
          }
        }
      }
    });
    CsCopyStaging(e, e.cap, /*to_device=*/true);
  }
  return true;
}

// A float scratch image the packed-float bridge blits through: no
// buffer<->image copy converts R11G11B10 to the float4 staging layout, and a
// blit converts any colour format to any other.
struct BridgeFloatImage {
  rhi::Texture* texture = nullptr;
  u32 w = 0, h = 0;
};
base::Vector<BridgeFloatImage> g_bridge_float_images;
u64 g_in_frame_bridge_n = 0;

bool BlitsBetween(rhi::Format a, rhi::Format b) {
  static base::HashMap<u64, bool> known;
  const u64 key = (u64(a) << 32) | u64(b);
  auto it = known.find(key);
  if (it != known.end())
    return it->second;
  return known[key] = Device().SupportsBlit(a, b);
}

rhi::Texture* GetBridgeFloatImage(u32 w, u32 h) {
  for (const BridgeFloatImage& b : g_bridge_float_images)
    if (b.w == w && b.h == h)
      return b.texture;
  if (g_bridge_float_images.size() >= 8)
    return nullptr;
  rhi::TextureDesc desc;
  desc.format = rhi::Format::kRGBA32Float;
  desc.width = w;
  desc.height = h;
  desc.usage = rhi::kTextureCopySrc | rhi::kTextureCopyDst;
  BridgeFloatImage b;
  b.texture = Device().CreateTexture(desc);
  if (!b.texture)
    return nullptr;
  b.w = w;
  b.h = h;
  g_bridge_float_images.push_back(b);
  return b.texture;
}

// The draw -> compute bridge, recorded at the current point of the frame
// command list instead of submitted and waited on its own: the target's
// texels go straight into the range's VRAM buffer, which holds linear texels
// already, so this is one copy (a packed-float target goes through a blit)
// and nothing touches the host. A truth range's buffer holds tiled bytes: the
// copy lands in a scratch and the tiling shader retiles it into the truth.
// The chunk is then submitted so the dispatch that reads the buffer, recorded
// into the compute batch, runs after it.
// Returns false for the shapes it does not cover; the caller then takes the
// synchronous path.
DELTA_OPTION(bool, kCsInFrameBridge, "DELTA_GPU_CS_INFRAME_BRIDGE", true);

// A live target as the in-frame bridge sees it: the aspect it copies and the
// layout the recording has it in.
struct InFrameTarget {
  CsAliasedImage img;
  rhi::TextureState layout = rhi::TextureState::kUndefined;
  rhi::Format fmt = rhi::Format::kUndefined;  // the target's format key
};

bool RecordRtStageInFrame(const InFrameTarget& t,
                          const ComputeInfo::Res& res,
                          CsRange& e,
                          const AliasedCopyPlan& plan) {
  const CsAliasedImage& img = t.img;
  if (!kCsInFrameBridge || !g_frame.recording || !e.buf || plan.widen ||
      plan.depth16 || plan.layers != 1 || res.layers > 1 || img.layers != 1 ||
      t.layout == rhi::TextureState::kUndefined ||
      (e.truth && (plan.unpack || img.is_stencil)))
    return false;
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(res, tiled, linear))
    return false;
  const auto& level = linear.mips[0];
  const u64 copy_bytes =
      static_cast<u64>(level.pitch) * plan.h * res.stage_elem_bytes;
  const TileTable* table = nullptr;
  rhi::Buffer* linear_buf = e.buf;
  if (e.truth) {
    if (!(table = GetTileTable(tiled, linear)) ||
        !(linear_buf = AcquireScratch(res.size)))
      return false;
  } else if (level.offset + copy_bytes > e.cap || res.size > e.cap) {
    return false;
  }
  rhi::Texture* float_image = nullptr;
  if (plan.unpack && (!BlitsBetween((t.fmt), rhi::Format::kRGBA32Float) ||
                      !(float_image = GetBridgeFloatImage(plan.w, plan.h))))
    return false;
  EndRegion();
  rhi::CommandList* const c = g_frame.list;
  c->Barrier(kAccessAll, kAccessCopyRW);
  constexpr rhi::TextureState kSrc = rhi::TextureState::kCopySrc;
  AliasedImageBarrier(c, img, t.layout, kSrc);
  // Staging bytes the copy does not reach read as zero, as the host path's
  // cleared mirror did.
  if (level.offset || copy_bytes < res.size || plan.w < level.pitch) {
    c->FillBuffer(linear_buf, 0, res.size & ~u64(3), 0);
    c->Barrier(rhi::kAccessCopyWrite, rhi::kAccessCopyWrite);
  }
  rhi::BufferTextureCopy copy;
  copy.buffer_offset = level.offset;
  copy.row_length = level.pitch;
  copy.image_height = level.stored_height;
  copy.region.aspect = img.aspect;
  copy.region.width = plan.w;
  copy.region.height = plan.h;
  if (plan.unpack) {
    rhi::TextureRegion region;
    region.width = plan.w;
    region.height = plan.h;
    rhi::TextureBarrier fb;
    fb.texture = float_image;
    fb.before = rhi::TextureState::kUndefined;
    fb.after = rhi::TextureState::kCopyDst;
    c->Barrier(0, 0, &fb, 1);
    c->BlitTexture(float_image, region, img.texture, region,
                   rhi::Filter::kNearest);
    fb.before = rhi::TextureState::kCopyDst;
    fb.after = kSrc;
    c->Barrier(0, 0, &fb, 1);
    c->CopyTextureToBuffer(linear_buf, float_image, &copy, 1);
  } else {
    c->CopyTextureToBuffer(linear_buf, img.texture, &copy, 1);
  }
  if (table)
    RecordImageTiling(c, *table, e.buf, 0, e.guest_bytes, linear_buf, res.size,
                      /*detile=*/false, 1, 1);
  AliasedImageBarrier(c, img, kSrc, t.layout);
  c->Barrier(rhi::kAccessCopyWrite, kAccessAll);
  e.frame_ref = g_frame.num;
  e.mirror_current = false;
  e.readback_pending = false;
  g_in_frame_bridge_n++;
  return CsSplitFrameChunk();
}

// The live colour or depth target at `base` the bridge would pick (the
// preference FindCsAliasedImage applies), in its recording-time layout.
bool LiveInFrameTarget(u64 base, bool prefer_depth, InFrameTarget& t) {
  const auto use_depth = [&](const DepthTarget& depth) {
    if (!depth.texture || depth.clear_pending)
      return false;
    t.layout = depth.layout;
    t.img = {depth.texture, depth.w, depth.h, 4,           rhi::kAspectDepth,
             t.layout,      true,    false,   depth.layers};
    return true;
  };
  if (prefer_depth) {
    auto it = g_depths.find(base);
    if (it != g_depths.end())
      return use_depth(it->second);
  }
  auto rt_it = g_rts.find(base);
  if (rt_it != g_rts.end()) {
    const RTarget& rt = rt_it->second;
    if (!rt.texture || rt.is_depth || rt.depth > 1 || !rt.ever_rendered)
      return false;
    t.layout = rt.layout;
    t.img = {rt.texture,        rt.w,    rt.h, FormatBytes(rt.fmt),
             rhi::kAspectColor, t.layout};
    t.fmt = rt.fmt;
    return true;
  }
  auto it = g_depths.find(base);
  return it != g_depths.end() && use_depth(it->second);
}

bool StageCsRangeFromRt(const ComputeInfo::Res& res, CsRange& e) {
  DELTA_ZONE("gpu.cs_stage_from_rt");
  InFrameTarget live;
  AliasedCopyPlan live_plan;
  if (LiveInFrameTarget(res.base, /*prefer_depth=*/res.dfmt == 4, live) &&
      PlanAliasedCopy(live.img, res, "reads", live_plan) &&
      RecordRtStageInFrame(live, res, e, live_plan))
    return true;
  // The draws that produced it this frame must be on the queue before the
  // copy is, or the dispatch reads last frame's pixels.
  if (RenderedInOpenChunk(res.base) && !CsSplitFrameChunk())
    return false;
  CsAliasedImage img;
  if (!FindCsAliasedImage(res.base, img, /*for_write=*/false,
                          /*prefer_depth=*/res.dfmt == 4))
    return false;
  AliasedCopyPlan plan;
  if (!PlanAliasedCopy(img, res, "reads", plan))
    return false;
  return RunAliasedCopy(img, res, e, /*to_image=*/false, plan);
}

// The reverse: a CS result written to a range that a live render/depth target
// aliases must also land in the image, because draws sample the image,
// never guest memory (SotC's exposure/bloom compute writes the adapted scene
// into an RT the tonemap then samples; its depth downsample writes the
// half-res depth pyramid). e.buf already holds the linear pixel data the
// dispatch produced, so upload straight from it. Guest memory was refreshed
// by the caller either way; a shape mismatch just leaves the image stale.
bool UploadCsRangeToRt(u64 base, CsRange& e) {
  CsAliasedImage img;
  if (!e.image_staging)
    return true;
  // The frame list already copied this revision into the target
  // (CsRefreshRtInFrame); uploading it again would paint over whatever was
  // rendered into it since.
  if (e.rt_seq == e.write_seq)
    return true;
  // A base rendered at several geometries: the dispatch names which one it
  // writes, and only the live image answers to the address.
  ActivateWrittenRtVariant(base, e.res.width, e.res.height);
  // Draws into the target earlier this frame would otherwise execute after
  // this upload and paint over the dispatch's result.
  if (RenderedInOpenChunk(base) && !CsSplitFrameChunk())
    return false;
  if (!FindCsAliasedImage(base, img, /*for_write=*/true,
                          /*prefer_depth=*/e.res.dfmt == 4))
    return true;  // nothing to refresh
  AliasedCopyPlan plan;
  if (!PlanAliasedCopy(img, e.res, "writes", plan)) {
    // The guest reused this address with an incompatible image layout. The
    // compute result is current in guest memory, while the old image can no
    // longer represent it; stop resolving subsequent samples to that image.
    if (!img.is_depth) {
      // Losing this flag makes every later sample of that address fall back to
      // guest memory, which draws never write, i.e. the target silently reads
      // black from then on. Worth seeing.
      if (kCsRtTrace)
        BASE_LOGI("csrt", "shape mismatch on {:#x} -> ever_rendered=false",
                  (unsigned long)base);
      g_rts[base].ever_rendered = false;
    }
    return true;
  }
  if (!RunAliasedCopy(img, e.res, e, /*to_image=*/true, plan))
    return false;
  if (!img.is_depth) {
    RTarget& rt = g_rts[base];
    rt.ever_rendered = true;  // CS content is real content
    rt.last_frame = g_frame.num;
    // The copy chained from and restored the anchor, so that is the layout
    // on the GPU now whether or not a frame end had stamped it.
    if (rt.submitted_layout == rhi::TextureState::kUndefined)
      rt.submitted_layout = img.submitted;
  } else if (!img.is_stencil) {
    // A depth surface a dispatch produced is this frame's content: the first
    // pass to bind it must LOAD it, where an untouched depth target is
    // cleared on its first bind of the frame. Astro Bot downsamples its scene
    // depth with a CS and tests its fog volumes against the result; clearing
    // it on the bind rejected every fog fragment.
    auto it = g_depths.find(base);
    if (it != g_depths.end()) {
      it->second.used_this_frame = true;
      it->second.last_frame = g_frame.num;
    }
  }
  return true;
}

base::HashMap<u64, CsRange> g_cs_ranges;
// Ranges MarkPending flagged, so retiring a batch visits those instead of
// every range. Map nodes do not move; CsRangeDestroy (which precedes every
// erase) drops its entry. May hold ranges no longer pending, or twice.
base::Vector<CsRange*> g_cs_pending;
u64 g_cs_range_bytes = 0;
constexpr u32 kCsDirtyPageShift = 16;
base::HashMap<u64, base::Vector<u64>> g_cs_dirty_pages;
// How many dirty ranges touch each 64 KiB of guest memory lives in the guest
// page table: a zero answers "nothing dirty here" without a single hash map
// lookup, and without the false alarms of a hashed table. Atomic because the
// command processor's walk reads it without owning the renderer
// (CsRangeMaybeDirty); only the owner writes it.
void NoteDirtyFilter(u64 base, u64 end, int delta) {
  GuestPages().AddCsDirty(base, end, delta);
}

bool DirtyFilterClear(u64 base, u64 end) {
  return GuestPages().CsClear(base, end);
}
// The same bases, once each: what the readback staging walks instead of every
// range (thousands of them against a few dozen dirty ones).
// Dirty range base -> end of the footprint its index entries cover.
base::HashMap<u64, u64> g_cs_dirty_bases;
// Bumped whenever compute results land in guest memory (writeback or an
// executed batch of importing dispatches). The draw path's per-frame staging
// caches stamp entries with this and re-copy when it has moved: a cached
// window of guest bytes is only as fresh as the last compute visibility point.
u64 g_cs_writeback_gen = 1;

u64 RangeEnd(u64 base, u64 bytes) {
  return bytes > UINT64_MAX - base ? UINT64_MAX : base + bytes;
}

void IndexDirtyRange(u64 base, u64 bytes) {
  BumpTextureEpoch();
  if (!bytes)
    return;
  const u64 end = RangeEnd(base, bytes);
  u64& indexed = g_cs_dirty_bases[base];
  if (indexed)
    NoteDirtyFilter(base, indexed, -1);
  indexed = base::Max(indexed, end);
  NoteDirtyFilter(base, indexed, 1);
  for (u64 page = base >> kCsDirtyPageShift;
       page <= (end - 1) >> kCsDirtyPageShift; page++)
    g_cs_dirty_pages[page].push_back(base);
}

void UnindexDirtyRange(u64 base, u64 bytes) {
  BumpTextureEpoch();
  if (!bytes)
    return;
  u64 end = RangeEnd(base, bytes);
  if (auto found = g_cs_dirty_bases.find(base);
      found != g_cs_dirty_bases.end()) {
    NoteDirtyFilter(base, found->second, -1);
    end = base::Max(end, found->second);
    g_cs_dirty_bases.erase(found);
  }
  for (u64 page = base >> kCsDirtyPageShift;
       page <= (end - 1) >> kCsDirtyPageShift; page++) {
    auto found = g_cs_dirty_pages.find(page);
    if (found == g_cs_dirty_pages.end())
      continue;
    auto& bases = found->second;
    base::EraseIf(bases, [&](u64 b) { return b == base; });
    if (bases.empty())
      g_cs_dirty_pages.erase(found);
  }
}

// Every range by the 1 MiB blocks its guest footprint covers, dirty or not:
// what a CPU write to guest memory looks up instead of walking every range.
constexpr u32 kCsRangeBlockShift = 20;
base::HashMap<u64, base::Vector<u64>> g_cs_range_blocks;

void IndexCsRange(u64 base, u64 bytes) {
  if (!bytes)
    return;
  const u64 end = RangeEnd(base, bytes);
  for (u64 b = base >> kCsRangeBlockShift; b <= (end - 1) >> kCsRangeBlockShift;
       b++)
    g_cs_range_blocks[b].push_back(base);
}

void UnindexCsRange(u64 base, u64 bytes) {
  if (!bytes)
    return;
  const u64 end = RangeEnd(base, bytes);
  for (u64 b = base >> kCsRangeBlockShift; b <= (end - 1) >> kCsRangeBlockShift;
       b++) {
    auto found = g_cs_range_blocks.find(b);
    if (found == g_cs_range_blocks.end())
      continue;
    base::EraseIf(found->second, [base](u64 b) { return b == base; });
    if (found->second.empty())
      g_cs_range_blocks.erase(found);
  }
}

// Calls fn(other) for every dirty range overlapping [base, base+bytes) until
// it returns true; a range spanning several index pages may be offered more
// than once. Allocation free, for the per-draw paths.
template <typename Fn>
bool AnyDirtyOverlapping(u64 base, u64 bytes, Fn&& fn) {
  if (g_cs_dirty_pages.empty() || !bytes)
    return false;
  const u64 end = RangeEnd(base, bytes);
  if (DirtyFilterClear(base, end))
    return false;
  const auto hits = [&](u64 other) {
    auto range = g_cs_ranges.find(other);
    return range != g_cs_ranges.end() && range->second.gpu_dirty &&
           other < end && base < RangeEnd(other, range->second.guest_bytes) &&
           fn(other);
  };
  // Same smaller-side walk as DirtyRangesOverlapping, and safe for the same
  // reason: every candidate's overlap is rechecked exactly.
  const u64 first_page = base >> kCsDirtyPageShift;
  const u64 last_page = (end - 1) >> kCsDirtyPageShift;
  if (last_page - first_page + 1 > g_cs_dirty_pages.size()) {
    for (const auto& [page, bases] : g_cs_dirty_pages)
      for (u64 other : bases)
        if (hits(other))
          return true;
    return false;
  }
  for (u64 page = first_page; page <= last_page; page++) {
    auto found = g_cs_dirty_pages.find(page);
    if (found == g_cs_dirty_pages.end())
      continue;
    for (u64 other : found->second)
      if (hits(other))
        return true;
  }
  return false;
}

base::Vector<u64> DirtyRangesOverlapping(u64 base,
                                         u64 bytes,
                                         u64 exclude = UINT64_MAX) {
  base::Vector<u64> candidates;
  if (!bytes)
    return candidates;
  const u64 end = RangeEnd(base, bytes);
  if (DirtyFilterClear(base, end))
    return candidates;
  // Walking the QUERY's pages costs a lookup per 64 KB of it, and a texture is
  // hundreds of pages while the whole dirty index is a handful of entries.
  // Whichever side is smaller gives the same answer, because the filter below
  // rechecks every candidate's overlap exactly, so a superset is safe here.
  const u64 first_page = base >> kCsDirtyPageShift;
  const u64 last_page = (end - 1) >> kCsDirtyPageShift;
  if (last_page - first_page + 1 > g_cs_dirty_pages.size()) {
    for (const auto& [page, bases] : g_cs_dirty_pages)
      candidates.insert(candidates.end(), bases.begin(), bases.end());
  } else {
    for (u64 page = first_page; page <= last_page; page++) {
      auto found = g_cs_dirty_pages.find(page);
      if (found != g_cs_dirty_pages.end())
        candidates.insert(candidates.end(), found->second.begin(),
                          found->second.end());
    }
  }
  base::Sort(candidates.begin(), candidates.end());
  candidates.erase(base::Unique(candidates.begin(), candidates.end()),
                   candidates.end());
  candidates.erase(
      base::RemoveIf(candidates.begin(), candidates.end(),
                     [&](u64 other) {
                       if (other == exclude)
                         return true;
                       auto found = g_cs_ranges.find(other);
                       return found == g_cs_ranges.end() ||
                              !found->second.gpu_dirty || other >= end ||
                              base >=
                                  RangeEnd(other, found->second.guest_bytes);
                     }),
      candidates.end());
  return candidates;
}

// DELTA_GPU_CSIMPORT: back the range's buffer with the GUEST PAGES themselves
// instead of a staging copy. Everything a compute dispatch reads then costs
// nothing to get in, and anything it writes is already in guest memory when the
// dispatch retires. SotC spends ~550 ms of a 2 fps frame on that copy in and
// back out. Needs the guest range page-aligned to the driver's import
// granularity; callers fall back to staging when this returns false.
bool CsRangeImportGuest(CsRange& e, u64 base, u64 size) {
  const rhi::Caps& caps = Device().caps();
  const u64 align = caps.host_import_alignment;
  if (!caps.host_import || !align)
    return false;
  const u64 lo = base & ~(align - 1);
  const u64 hi = (base + size + align - 1) & ~(align - 1);
  // The shader indexes from the binding, so an unaligned base is fine as long
  // as the descriptor can carry the difference.
  const u64 off = base - lo;
  if (off % caps.storage_offset_alignment)
    return false;
  rhi::BufferDesc desc;
  desc.size = hi - lo;
  desc.usage = rhi::kBufferStorage | rhi::kBufferCopyDst;
  desc.host_pointer = reinterpret_cast<void*>(lo);
  rhi::Buffer* buf = Device().CreateBuffer(desc);
  if (!buf)
    return false;
  e.buf = buf;
  e.map = reinterpret_cast<void*>(lo);  // already the guest pages
  e.cap = hi - lo;
  e.imported = true;
  e.imported_base = lo;
  e.imported_offset = off;
  return true;
}

// The GDS scratchpad, created once. Zeroed at creation; after that it is the
// title's own counter store, so nothing here resets it.
bool EnsureGdsBuffer() {
  if (g_gds.buf)
    return true;
  g_gds.buf = CreateCsBuffer(GdsBuffer::kBytes,
                             rhi::kBufferStorage | rhi::kBufferCopyDst,
                             rhi::MemoryKind::kReadback);
  if (!g_gds.buf)
    return false;
  g_gds.map = g_gds.buf->mapped();
  if (!g_gds.map) {
    Device().Destroy(g_gds.buf);
    g_gds = {};
    return false;
  }
  std::memset(g_gds.map, 0, GdsBuffer::kBytes);
  return true;
}

void CsRangeDestroy(CsRange& e);

base::HashMap<u64, CsRange>::iterator EraseCsRange(
    base::HashMap<u64, CsRange>::iterator it) {
  UnindexCsRange(it->first, it->second.guest_bytes);
  CsRangeDestroy(it->second);
  return g_cs_ranges.erase(it);
}

// Device memory ran out: give back the recycled buffers and every range no
// dispatch has touched for a few frames, instead of dropping the dispatch.
// `keep` is the range being allocated for. True when anything was released.
bool CsReleaseMemory(const CsRange& keep) {
  const auto drop_free_list = [] {
    const bool any = !g_cs_free.empty();
    for (const CsFreeBuffer& f : g_cs_free) {
      Device().Destroy(f.buf);
      Device().Destroy(f.host_buf);
    }
    g_cs_free.clear();
    g_cs_free_bytes = 0;
    return any;
  };
  const bool released = drop_free_list();
  u64 evicted = 0;
  for (auto it = g_cs_ranges.begin(); it != g_cs_ranges.end();) {
    CsRange& r = it->second;
    if (&r == &keep || r.gpu_dirty || r.pending_batch || CsFrameReferenced(r) ||
        r.last_used_frame + 2 >= g_frame.num) {
      ++it;
      continue;
    }
    evicted += r.cap;
    it = EraseCsRange(it);
  }
  // CsRangeDestroy parks what it frees for reuse; that is the opposite of
  // what is wanted here.
  drop_free_list();
  static int logged = 0;
  if (logged++ < 16)
    BASE_LOGI("csgpu", "device memory full: evicted {} MB of idle ranges",
              evicted >> 20);
  return released || evicted;
}

bool CsRangeEnsureBuffer(CsRange& e, u64 size) {
  if (e.buf && e.cap >= size)
    return true;
  e.map = nullptr;
  if (CsFrameReferenced(e) || e.pending_batch) {
    g_cs_frame_retired.push_back(e.buf);
    g_cs_frame_retired.push_back(e.host_buf);
    e.frame_ref = -1;
    e.pending_batch = false;
  } else {
    Device().Destroy(e.buf);
    Device().Destroy(e.host_buf);
  }
  e.buf = nullptr;
  e.host_buf = nullptr;
  e.device_local = false;
  e.imported = false;
  if (!e.imported)
    g_cs_range_bytes -= e.cap;
  e.cap = 0;
  u64 cap = (size + 0xFFFF) & ~u64(0xFFFF);
  for (size_t i = g_cs_free.size(); i-- > 0;) {
    const CsFreeBuffer& f = g_cs_free[i];
    if (f.cap != cap || f.device_local != static_cast<bool>(kCsVram))
      continue;
    e.buf = f.buf;
    e.host_buf = f.host_buf;
    e.map = f.map;
    e.device_local = f.device_local;
    e.cap = cap;
    g_cs_free_bytes -= cap;
    g_cs_free.erase(g_cs_free.begin() + i);
    g_cs_range_bytes += cap;
    g_cs_recycled_n++;
    return true;
  }
  // CopyDst: RT-backed inputs are staged by an image->buffer copy on the
  // queue (StageCsRangeFromRt) instead of a CPU memcpy from guest memory.
  // Vertex: a draw may fetch a dispatch's output in place (CsVertexBuffer).
  u32 usage = rhi::kBufferStorage | rhi::kBufferCopyDst | rhi::kBufferVertex;
  if (kCsVram)
    usage |= rhi::kBufferCopySrc;  // readback of results
  const bool split = kCsVram && SplitVram();
  const rhi::MemoryKind memory =
      split ? rhi::MemoryKind::kDevice : rhi::MemoryKind::kReadback;
  e.buf = CreateCsBuffer(cap, usage, memory);
  if (!e.buf &&
      (!CsReleaseMemory(e) || !(e.buf = CreateCsBuffer(cap, usage, memory))))
    return false;
  // VRAM cannot be mapped usefully (uncached reads are ~100 MB/s), so the CPU
  // side gets its own host-cached buffer and the two are joined by DMA.
  if (split) {
    e.host_buf = CreateCsBuffer(cap, rhi::kBufferCopySrc | rhi::kBufferCopyDst,
                                rhi::MemoryKind::kReadback);
    if (!e.host_buf)
      return false;
    e.map = e.host_buf->mapped();
    e.device_local = true;
  } else {
    e.map = e.buf->mapped();
  }
  e.cap = cap;
  g_cs_range_bytes += cap;
  return true;
}

// DELTA_GPU_CSRENAME: a CPU write into a range the open batch already reads has
// to flush that batch and wait on the GPU. SotC's streaming does that 37 times
// a frame, and staging then costs 272 ms of a 2 fps frame. Giving the range a
// FRESH buffer and retiring the old one until the batch completes takes that to
// 24 ms: the recorded descriptors keep reading the contents they were built
// against, and this dispatch binds the new one. Only safe while nothing is
// waiting to read results back OUT of the old buffer, hence the caller's
// gpu_dirty/written check. Default OFF: it moves cost into the writeback rather
// than removing it (SotC gains ~7% overall), and no title that currently
// renders through the compute path could be used to gate the change.
bool CsRangeRename(CsRange& e, u64 size) {
  e.map = nullptr;
  if (e.buf)
    g_cs_frame_retired.push_back(e.buf);
  if (e.host_buf)
    g_cs_frame_retired.push_back(e.host_buf);
  e.frame_ref = -1;
  e.pending_batch = false;
  g_cs_range_bytes -= e.cap;
  e.buf = nullptr;
  e.host_buf = nullptr;
  e.device_local = false;
  e.cap = 0;
  return CsRangeEnsureBuffer(e, size);
}

void DropTiledImport(u64 base);

void CsRangeDestroy(CsRange& e) {
  base::EraseIf(g_cs_pending, [&e](CsRange* r) { return r == &e; });
  if (e.res.base)
    DropTiledImport(e.res.base);
  if (!e.imported && !CsFrameReferenced(e) && !e.pending_batch && e.buf &&
      e.cap && g_cs_free_bytes + e.cap <= (256ull << 20)) {
    g_cs_free.push_back({e.buf, e.host_buf, e.map, e.cap, e.device_local});
    g_cs_free_bytes += e.cap;
    g_cs_range_bytes -= e.cap;
    e = CsRange{};
    return;
  }
  if (CsFrameReferenced(e) || e.pending_batch) {
    g_cs_frame_retired.push_back(e.buf);
    g_cs_frame_retired.push_back(e.host_buf);
  } else {
    Device().Destroy(e.buf);
    Device().Destroy(e.host_buf);
  }
  g_cs_range_bytes -= e.cap;
  e = CsRange{};
}

#ifdef DELTA_HAVE_SPIRV_BACKEND
// A guest range mapped into the device so the GPU addresses it in place.
struct ImportedRange {
  rhi::Buffer* buf = nullptr;
  u64 offset = 0;  // base - the alignment floor the import starts at
  u64 bytes = 0;   // usable bytes from `offset`
};
base::HashMap<u64, ImportedRange> g_tiled_imports;

// The tiled side of a layout conversion is guest memory the CPU otherwise
// copies twice a frame: in on the way to the shader, out again on the way back.
// Importing those pages lets the tiling shader read and write them where they
// already live, which is the whole of `host-in` and `host-out` in the
// DELTA_GPU_CSSYNC report. Cached: the allocate/create pair is far too
// expensive to repeat per dispatch.
const ImportedRange* ImportTiledRange(u64 base, u64 bytes) {
  const rhi::Caps& caps = Device().caps();
  const u64 align = caps.host_import_alignment;
  if (!caps.host_import || !align || !base || !bytes)
    return nullptr;
  auto it = g_tiled_imports.find(base);
  if (it != g_tiled_imports.end()) {
    // The import pins host pages by address. If the title has since unmapped
    // that surface, the mapping still points at pages that now belong to
    // something else, and a conversion would write through it, so confirm
    // the range before every reuse, not only when it is created.
    if (it->second.bytes >= bytes && gpu::IsReadableRangeCached(base, bytes))
      return &it->second;
    // A bigger view of the same base needs a bigger mapping.
    Device().Destroy(it->second.buf);
    g_tiled_imports.erase(it);
  }
  // Unbounded caching would pin every surface a title ever converts; these are
  // a handful of large render targets, so a small cap is plenty.
  if (g_tiled_imports.size() >= 64)
    return nullptr;
  if (!gpu::IsReadableRange(base, bytes))
    return nullptr;
  const u64 lo = base & ~(align - 1);
  const u64 hi = (base + bytes + align - 1) & ~(align - 1);
  const u64 off = base - lo;
  if (off % caps.storage_offset_alignment)
    return nullptr;
  // The guest CPU and our own readers touch these pages without any map/unmap
  // of their own; an imported buffer's memory is coherent, or the import is
  // declined.
  rhi::BufferDesc desc;
  desc.size = hi - lo;
  desc.usage = rhi::kBufferStorage | rhi::kBufferCopySrc | rhi::kBufferCopyDst;
  desc.host_pointer = reinterpret_cast<void*>(lo);
  rhi::Buffer* buf = Device().CreateBuffer(desc);
  if (!buf)
    return nullptr;
  ImportedRange& out = g_tiled_imports[base];
  out.buf = buf;
  out.offset = off;
  out.bytes = hi - lo - off;
  return &out;
}

struct TileTable {
  rhi::Buffer* buf = nullptr;
  u64 bytes = 0;
  base::Vector<gcn::ImageTilingParams> params;  // per mip; `detile` patched
  u32 layers = 0;
  bool expanded = false;
};
base::HashMap<u64, TileTable> g_tile_tables;

// The tiling shader, its descriptors pushed into whichever command list
// records a conversion.
struct TilingState {
  rhi::BindGroupLayout* set_layout = nullptr;
  rhi::PipelineLayout* layout = nullptr;
  rhi::Pipeline* pipe = nullptr;
  CsRange buffers[3];
};
TilingState g_tiling;

bool InitTiling() {
  auto& s = g_tiling;
  if (s.pipe)
    return true;
  static bool attempted = false;
  if (attempted)
    return false;
  attempted = true;
  const auto code = gcn::BuildImageTilingShader();
  if (code.empty())
    return false;
  rhi::BindGroupLayoutDesc set;
  for (u32 i = 0; i < 3; i++)
    set.bindings.push_back(
        {i, rhi::BindingType::kStorageBuffer, rhi::kStageCompute});
  set.push = true;
  s.set_layout = Device().CreateBindGroupLayout(set);
  if (!s.set_layout)
    return false;
  rhi::PipelineLayoutDesc layout;
  layout.groups = {s.set_layout};
  layout.push_constant_bytes = sizeof(gcn::ImageTilingParams);
  layout.push_constant_stages = rhi::kStageCompute;
  s.layout = Device().CreatePipelineLayout(layout);
  if (!s.layout)
    return false;
  rhi::ComputePipelineDesc pipe;
  pipe.layout = s.layout;
  pipe.code = rhi::Code(code);
  s.pipe = Device().CreateComputePipeline(pipe);
  return s.pipe != nullptr;
}

void DropTiledImportImpl(u64 base) {
  auto it = g_tiled_imports.find(base);
  if (it == g_tiled_imports.end())
    return;
  Device().Destroy(it->second.buf);
  g_tiled_imports.erase(it);
}

bool ConvertGfx10ImageImpl(const gcn::TextureLayout32& tiled,
                           const gcn::TextureLayout32& linear,
                           const void* src,
                           void* dst,
                           bool detile,
                           rhi::Buffer* linear_buffer = nullptr) {
  const bool expanded = (tiled.elem_bytes == 1 || tiled.elem_bytes == 2) &&
                        linear.elem_bytes == 4;
  if (!g_backend.device || g_cs_failed)
    return false;
  const u64 max_range = Device().caps().max_storage_buffer_range;
  if ((!expanded &&
       (tiled.elem_bytes < 4 || tiled.elem_bytes != linear.elem_bytes)) ||
      linear.tiling_idx != 8 || tiled.layers != linear.layers ||
      tiled.mip_levels != linear.mip_levels || tiled.size > max_range ||
      linear.size > max_range || tiled.size > UINT32_MAX ||
      linear.size > UINT32_MAX)
    return false;
  base::Vector<u32> table, terms;
  base::Vector<gcn::ImageTilingParams> params;
  for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& t = tiled.mips[mip];
    const auto& l = linear.mips[mip];
    u32 mask;
    u64 stride;
    if (t.width != l.width || t.height != l.height ||
        !gcn::BuildSeparableAddressTable(tiled, mip, terms, mask, stride))
      return false;
    params.push_back({t.width, t.height, linear.elem_bytes / 4,
                      static_cast<u32>(table.size()), mask,
                      static_cast<u32>(t.offset), static_cast<u32>(stride),
                      static_cast<u32>(l.offset / 4), l.pitch,
                      l.pitch * l.stored_height * (linear.elem_bytes / 4),
                      detile ? 1u : 0u, tiled.elem_bytes});
    table.insert(table.end(), terms.begin(), terms.end());
  }
  if (params.empty() || !InitTiling())
    return false;
  auto& s = g_tiling;
  const u64 sizes[] = {tiled.size, linear.size, table.size() * 4};
  // Bind the guest pages themselves for the tiled half when they can be
  // imported. Only the direct-store forms qualify: the expanded (1/2-byte)
  // path clears its destination and merges texels with an atomic OR, which
  // over guest memory would zero the padding a copy leaves alone.
  const ImportedRange* imported =
      kCsTileImport && !expanded && src == static_cast<const void*>(dst)
          ? ImportTiledRange(reinterpret_cast<u64>(src), tiled.size)
          : nullptr;
  // The tiled half stays in VRAM even when its guest pages are imported: the
  // shader's addressing is scattered, and scattered access over PCIe runs at a
  // fraction of the rate a linear DMA of the same bytes does. The import is
  // used as a DMA endpoint instead, which keeps the scatter in VRAM and still
  // spares the CPU both copies.
  const bool dma_tiled = imported != nullptr;
  rhi::BindingWrite writes[3];
  for (u32 i = 0; i < 3; i++) {
    const bool bound = i == 1 && linear_buffer;
    if (!bound &&
        (!CsRangeEnsureBuffer(s.buffers[i], sizes[i]) || !s.buffers[i].map))
      return false;
    writes[i].binding = i;
    writes[i].buffer = bound ? linear_buffer : s.buffers[i].buf;
    writes[i].range = sizes[i];
  }
  const u32 input = detile ? 0 : 1, output = detile ? 1 : 0;
  // Whichever half is bound to memory the shader reaches directly needs no
  // copy: the linear side is the caller's own buffer, the tiled side is guest
  // memory.
  const bool input_bound =
      (input == 1 && linear_buffer) || (input == 0 && dma_tiled);
  const bool output_bound =
      (output == 1 && linear_buffer) || (output == 0 && dma_tiled);
  const u64 t_hin = NowNs();
  if (!input_bound) {
    ParallelCopy(s.buffers[input].map, src, sizes[input]);
    g_tile_hin_bytes += sizes[input];
  }
  if (!output_bound && !s.buffers[output].device_local)
    std::memset(s.buffers[output].map, 0, sizes[output]);
  std::memcpy(s.buffers[2].map, table.data(), sizes[2]);
  g_tile_hin_ns += NowNs() - t_hin;
  rhi::CommandList* c = BeginImmediate();
  if (!c)
    return false;
  c->Barrier(kAccessAll, rhi::kAccessCopyWrite);
  if (dma_tiled)
    c->CopyBuffer(s.buffers[0].buf, 0, imported->buf, imported->offset,
                  tiled.size);
  else if (!input_bound)
    RecordStagingCopy(c, s.buffers[input], sizes[input], true);
  RecordStagingCopy(c, s.buffers[2], sizes[2], true);
  // A tiled destination seeded from guest memory must not be cleared after it.
  if (!(output == 0 && dma_tiled) &&
      (output_bound || s.buffers[output].device_local))
    c->FillBuffer(writes[output].buffer, writes[output].offset, sizes[output],
                  0);
  c->Barrier(rhi::kAccessAllWrite, kAccessComputeRW);
  c->SetPipeline(s.pipe);
  c->PushBindGroup(0, s.set_layout, writes, 3);
  for (const auto& p : params) {
    c->SetPushConstants(0, sizeof(p), &p);
    c->Dispatch((p.width * p.words + 63) / 64, p.height, tiled.layers);
  }
  if (dma_tiled && !detile) {
    NoteGuestWrite(reinterpret_cast<u64>(dst), tiled.size);
    c->Barrier(rhi::kAccessComputeWrite, rhi::kAccessCopyRead);
    c->CopyBuffer(imported->buf, imported->offset, s.buffers[0].buf, 0,
                  tiled.size);
    g_tile_hout_bytes += tiled.size;
  } else if (!output_bound) {
    RecordStagingCopy(c, s.buffers[output], sizes[output], false);
  }
  c->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
             rhi::kAccessHostRead | kAccessComputeRW);
  const u64 t_sync = NowNs();
  const bool ok = EndImmediate(c);
  g_tile_sync_ns += NowNs() - t_sync;
  if (!ok) {
    BASE_LOGI("gpuvk", "image conversion failed");
    g_cs_failed = true;
    return false;
  }
  const u64 t_hout = NowNs();
  if (detile && !linear_buffer) {
    ParallelCopy(dst, s.buffers[output].map, sizes[output]);
    g_tile_hout_bytes += sizes[output];
  } else if (!detile && !dma_tiled) {
    gcn::CopyImageContents(tiled, s.buffers[output].map, dst);
    g_tile_hout_bytes += tiled.size;
  }
  g_tile_hout_ns += NowNs() - t_hout;
  return true;
}

// ---- Tiled-truth ranges (DELTA_GPU_CS_TRUTH) ----

u64 TileTableKey(const gcn::TextureLayout32& t,
                 const gcn::TextureLayout32& l,
                 bool packed) {
  u64 h = 1469598103934665603ull;
  const u64 words[] = {t.tiling_idx,    t.mips[0].width, t.mips[0].height,
                       t.mips[0].pitch, t.layers,        t.mip_levels,
                       t.elem_bytes,    t.size,          l.elem_bytes,
                       l.size,          l.mips[0].pitch, packed ? 1u : 0u};
  for (u64 w : words)
    h = (h ^ w) * 1099511628211ull;
  return h;
}

// The address terms of one surface layout, built once. The shader reads
// three of them per texel, so they live where it can reach them quickly.
const TileTable* GetTileTable(const gcn::TextureLayout32& tiled,
                              const gcn::TextureLayout32& linear,
                              bool packed) {
  const bool expanded = !packed &&
                        (tiled.elem_bytes == 1 || tiled.elem_bytes == 2) &&
                        linear.elem_bytes == 4;
  if (packed &&
      (tiled.elem_bytes >= 4 || tiled.elem_bytes != linear.elem_bytes))
    return nullptr;
  const u64 max_range = Device().caps().max_storage_buffer_range;
  if (!InitTiling() ||
      (!expanded && !packed &&
       (tiled.elem_bytes < 4 || tiled.elem_bytes != linear.elem_bytes)) ||
      linear.tiling_idx != 8 || tiled.layers != linear.layers ||
      tiled.mip_levels != linear.mip_levels || tiled.size > max_range ||
      linear.size > max_range || tiled.size > UINT32_MAX ||
      linear.size > UINT32_MAX)
    return nullptr;
  const u64 key = TileTableKey(tiled, linear, packed);
  auto it = g_tile_tables.find(key);
  if (it != g_tile_tables.end())
    return it->second.buf ? &it->second : nullptr;
  if (g_tile_tables.size() >= 1024)
    return nullptr;
  TileTable& t = g_tile_tables[key];  // a failed build stays empty: no retry
  base::Vector<u32> table, terms;
  for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& tm = tiled.mips[mip];
    const auto& lm = linear.mips[mip];
    u32 mask;
    u64 stride;
    if (tm.width != lm.width || tm.height != lm.height ||
        !gcn::BuildSeparableAddressTable(tiled, mip, terms, mask, stride))
      return nullptr;
    const u32 words = packed ? 1u : linear.elem_bytes / 4;
    t.params.push_back(
        {tm.width, tm.height, words, static_cast<u32>(table.size()), mask,
         static_cast<u32>(tm.offset), static_cast<u32>(stride),
         static_cast<u32>(packed ? lm.offset : lm.offset / 4), lm.pitch,
         packed ? lm.pitch * lm.stored_height * linear.elem_bytes
                : lm.pitch * lm.stored_height * words,
         1u, tiled.elem_bytes, packed ? 1u : 0u});
    table.insert(table.end(), terms.begin(), terms.end());
  }
  if (table.empty())
    return nullptr;
  t.layers = tiled.layers;
  t.expanded = expanded;
  t.bytes = table.size() * 4;
  // Written once by the host, read by every conversion: device-local
  // host-visible memory where the heap offers it, plain host memory otherwise.
  t.buf = CreateCsBuffer(t.bytes, rhi::kBufferStorage,
                         rhi::MemoryKind::kUploadDevice);
  if (t.buf && !t.buf->mapped()) {
    Device().Destroy(t.buf);
    t.buf = nullptr;
  }
  if (!t.buf)
    return nullptr;
  std::memcpy(t.buf->mapped(), table.data(), t.bytes);
  return &t;
}

// Record one layout conversion between two device buffers: `tiled` holds the
// guest bytes (a range's truth), `linear` the shader's view of them. `mips`
// and `layers` limit the pass to the leading levels and slices (a bridge
// copies mip 0 of layer 0 only); 0 means all of them.
void RecordImageTiling(rhi::CommandList* c,
                       const TileTable& t,
                       rhi::Buffer* tiled,
                       u64 tiled_off,
                       u64 tiled_bytes,
                       rhi::Buffer* linear,
                       u64 linear_bytes,
                       bool detile,
                       u32 mips,
                       u32 layers,
                       bool depth16) {
  auto& s = g_tiling;
  const u32 mip_count =
      mips ? base::Min<u32>(mips, t.params.size()) : t.params.size();
  const u32 layer_count = layers ? base::Min(layers, t.layers) : t.layers;
  c->Barrier(kAccessComputeRW | kAccessCopyRW | rhi::kAccessHostWrite,
             kAccessComputeRW | rhi::kAccessCopyWrite);
  c->SetPipeline(s.pipe);
  rhi::BindingWrite writes[3];
  writes[0].binding = 0;
  writes[0].buffer = tiled;
  writes[0].offset = tiled_off;
  writes[0].range = tiled_bytes;
  writes[1].binding = 1;
  writes[1].buffer = linear;
  writes[1].range = linear_bytes;
  writes[2].binding = 2;
  writes[2].buffer = t.buf;
  writes[2].range = t.bytes;
  c->PushBindGroup(0, s.set_layout, writes, 3);
  for (u32 m = 0; m < mip_count; m++) {
    gcn::ImageTilingParams p = t.params[m];
    p.detile = detile ? 1u : 0u;
    p.depth16 = depth16 ? 1u : 0u;
    c->SetPushConstants(0, sizeof(p), &p);
    c->Dispatch((p.width * p.words + 63) / 64, p.height, layer_count);
  }
  c->Barrier(rhi::kAccessComputeWrite,
             kAccessComputeRW | kAccessCopyRW | rhi::kAccessHostRead);
}

// Transient detiled views. A scratch is reused within the frame as soon as
// the pass that consumed it has been recorded (the conversions' own
// barriers order the accesses) and never by the dispatch being recorded.
// A scratch that goes cold is retired through the frame ring.
struct CsScratch {
  rhi::Buffer* buf = nullptr;
  u64 cap = 0;
  u64 owner = 0;
  int last_frame = -1;
  // The view it holds: which bytes, converted how, at which revision. A
  // later binding of the same unchanged truth reuses it as it is.
  u64 view_base = 0;
  u64 view_off = 0;
  const TileTable* view_table = nullptr;
  u64 view_seq = 0;
};
base::Vector<CsScratch> g_cs_scratch;
u64 g_truth_view_hits = 0;

rhi::Buffer* AcquireView(u64 bytes,
                         u64 base,
                         u64 off,
                         const TileTable* table,
                         u64 seq,
                         bool* fresh) {
  for (auto& s : g_cs_scratch) {
    if (s.view_base == base && s.view_off == off && s.view_table == table &&
        s.view_seq == seq && s.cap >= bytes) {
      s.owner = g_cs_scratch_owner;
      s.last_frame = g_frame.num;
      g_truth_view_hits++;
      *fresh = false;
      return s.buf;
    }
  }
  *fresh = true;
  return AcquireScratch(bytes);
}

void StampView(rhi::Buffer* view,
               u64 base,
               u64 off,
               const TileTable* table,
               u64 seq) {
  for (auto& s : g_cs_scratch)
    if (s.buf == view) {
      s.view_base = base;
      s.view_off = off;
      s.view_table = table;
      s.view_seq = seq;
      return;
    }
}

rhi::Buffer* AcquireScratch(u64 bytes) {
  int best = -1;
  for (size_t i = 0; i < g_cs_scratch.size(); i++) {
    const auto& s = g_cs_scratch[i];
    if (s.owner == g_cs_scratch_owner || s.cap < bytes)
      continue;
    if (best < 0 || s.cap < g_cs_scratch[best].cap ||
        (s.cap == g_cs_scratch[best].cap &&
         s.last_frame < g_cs_scratch[best].last_frame))
      best = static_cast<int>(i);
  }
  if (best >= 0) {
    auto& s = g_cs_scratch[best];
    s.owner = g_cs_scratch_owner;
    s.last_frame = g_frame.num;
    s.view_base = 0;
    s.view_table = nullptr;
    return s.buf;
  }
  // One dispatch's views are all that has to coexist; everything colder is
  // recycled before the ring is allowed to grow past its budget.
  constexpr u64 kScratchBudget = 512ull << 20;
  CsScratch s;
  s.cap = (bytes + 0x3FFFFF) & ~u64(0x3FFFFF);
  while (!g_cs_scratch.empty() &&
         (g_cs_scratch.size() >= 24 ||
          g_cs_scratch_bytes + s.cap > kScratchBudget)) {
    size_t coldest = SIZE_MAX;
    for (size_t i = 0; i < g_cs_scratch.size(); i++)
      if (g_cs_scratch[i].owner != g_cs_scratch_owner &&
          (coldest == SIZE_MAX ||
           g_cs_scratch[i].last_frame < g_cs_scratch[coldest].last_frame))
        coldest = i;
    if (coldest == SIZE_MAX)
      break;
    g_cs_frame_retired.push_back(g_cs_scratch[coldest].buf);
    g_cs_scratch_bytes -= g_cs_scratch[coldest].cap;
    g_cs_scratch.erase(g_cs_scratch.begin() + coldest);
  }
  s.buf = CreateCsBuffer(
      s.cap, rhi::kBufferStorage | rhi::kBufferCopySrc | rhi::kBufferCopyDst,
      SplitVram() ? rhi::MemoryKind::kDevice : rhi::MemoryKind::kReadback);
  if (!s.buf)
    return nullptr;
  s.owner = g_cs_scratch_owner;
  s.last_frame = g_frame.num;
  if (Device().caps().debug_labels) {
    char name[48];
    std::snprintf(name, sizeof(name), "cs scratch %llu KB",
                  (unsigned long long)(s.cap / 1024));
    Device().SetName(s.buf, name);
  }
  if (s.cap >= (64u << 20)) {
    static int n = 0;
    if (n++ < 32)
      BASE_LOGI("csgpu", "scratch {} MB for a {} MB view (ring {} x {} MB)",
                (unsigned long long)(s.cap >> 20),
                (unsigned long long)(bytes >> 20), g_cs_scratch.size(),
                (unsigned long long)(g_cs_scratch_bytes >> 20));
  }
  g_cs_scratch_bytes += s.cap;
  g_cs_scratch.push_back(s);
  return s.buf;
}

// DMA the guest pages of a truth range straight into its buffer. The pages
// are read when the batch executes, not now: the same window the tiled
// import already accepts for conversions.
bool RecordTruthImport(CsRange& e, u64 base, u64 bytes) {
  const ImportedRange* imp = ImportTiledRange(base, bytes);
  if (!imp || !e.buf)
    return false;
  CsBatchBeginImpl();
  g_cs_list->Barrier(kAccessComputeRW | kAccessCopyRW, rhi::kAccessCopyWrite);
  g_cs_list->CopyBuffer(e.buf, 0, imp->buf, imp->offset, bytes);
  g_cs_list->Barrier(rhi::kAccessCopyWrite, kAccessComputeRW | kAccessCopyRW);
  MarkPending(e);
  e.mirror_current = false;
  g_truth_import_n++;
  return true;
}

bool TruthAvailable() {
  return InitTiling();
}
#else
const TileTable* GetTileTable(const gcn::TextureLayout32&,
                              const gcn::TextureLayout32&,
                              bool) {
  return nullptr;
}
rhi::Buffer* AcquireView(u64, u64, u64, const TileTable*, u64, bool* fresh) {
  *fresh = true;
  return nullptr;
}
void StampView(rhi::Buffer*, u64, u64, const TileTable*, u64) {}
void RecordImageTiling(rhi::CommandList*,
                       const TileTable&,
                       rhi::Buffer*,
                       u64,
                       u64,
                       rhi::Buffer*,
                       u64,
                       bool,
                       u32,
                       u32,
                       bool) {}
rhi::Buffer* AcquireScratch(u64) {
  return nullptr;
}
bool RecordTruthImport(CsRange&, u64, u64) {
  return false;
}
bool TruthAvailable() {
  return false;
}
#endif

// The tiling shader handles every gfx10 swizzle, and a Liverpool layout once
// its address terms are shown to be separable (4-byte and wider texels).
bool GpuTileable(const ComputeInfo::Res& r) {
  if (r.tiling_idx >= gcn::kGfx10TilingBase)
    return true;
  if (r.elem_bytes < 4 || r.elem_bytes != r.stage_elem_bytes)
    return false;
  static base::HashMap<u64, bool> known;
  u64 key = 1469598103934665603ull;
  for (u64 v : {u64(r.tiling_idx), u64(r.width), u64(r.height), u64(r.pitch),
                u64(r.layers), u64(r.mip_levels), u64(r.elem_bytes),
                u64(r.dfmt), u64(r.pow2_pad), r.guest_size, r.size})
    key = (key ^ v) * 1099511628211ull;
  auto [it, inserted] = known.try_emplace(key, false);
  if (!inserted)
    return it->second;
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(r, tiled, linear))
    return false;
  base::Vector<u32> terms;
  u32 mask;
  u64 stride;
  for (u32 mip = 0; mip < tiled.mip_levels; mip++)
    if (!gcn::BuildSeparableAddressTable(tiled, mip, terms, mask, stride))
      return false;
  it->second = true;
  return true;
}

// A resource whose range can be a tiled truth: a surface the tiling shader
// handles, with the shader reading its texels directly or expanded.
bool TruthEligible(const ComputeInfo::Res& r) {
  return kCsTruth && kCsVram && r.image_staging && !r.zero_fill && r.size &&
         r.guest_size &&
         ((r.elem_bytes >= 4 && r.elem_bytes == r.stage_elem_bytes) ||
          ((r.elem_bytes == 1 || r.elem_bytes == 2) &&
           r.stage_elem_bytes == 4)) &&
         r.dfmt != 6 && r.dfmt != 35 && r.dfmt != 40 &&
         !gcn::TilingIsLinear(r.tiling_idx) &&
         r.guest_size <= Device().caps().max_storage_buffer_range &&
         r.size <= Device().caps().max_storage_buffer_range &&
         r.guest_size <= (768ull << 20) && r.size <= (768ull << 20) &&
         TruthAvailable() && GpuTileable(r);
}

// Defined either side of the backend guard: CsRangeDestroy runs whether or not
// the tiling shader was compiled in.
void DropTiledImport(u64 base) {
#ifdef DELTA_HAVE_SPIRV_BACKEND
  DropTiledImportImpl(base);
#else
  (void)base;
#endif
}

bool CanGpuTile(const ComputeInfo::Res& res, const CsRange& e) {
  return kCsGpuTiling && !kDetileDump && !kCsWbAudit && e.device_local &&
         res.image_staging && res.size >= 256 * 1024 &&
         ((res.elem_bytes >= 4 && res.elem_bytes == res.stage_elem_bytes) ||
          ((res.elem_bytes == 1 || res.elem_bytes == 2) &&
           res.stage_elem_bytes == 4)) &&
         res.dfmt != 35 && res.dfmt != 40 &&
         !gcn::TilingIsLinear(res.tiling_idx) && GpuTileable(res);
}

bool ConvertCsRangeImage(const ComputeInfo::Res& res, CsRange& e, bool detile) {
#ifdef DELTA_HAVE_SPIRV_BACKEND
  gcn::TextureLayout32 tiled, linear;
  if (!CanGpuTile(res, e) || !BuildCsImageLayouts(res, tiled, linear))
    return false;
  const u64 started = NowNs();
  if (!ConvertGfx10ImageImpl(tiled, linear, reinterpret_cast<void*>(res.base),
                             reinterpret_cast<void*>(res.base), detile, e.buf))
    return false;
  if (kCsSyncReport) {
    g_gpu_tiling_ns[detile] += NowNs() - started;
    g_gpu_tiling_n[detile]++;
  }
  e.mirror_current = false;
  return true;
#else
  return false;
#endif
}

// Dispatch batching: dispatches are recorded into one command buffer and
// submitted/waited only when something needs their results (a flush point,
// a staging hazard, or the batch cap). 228 individual submit+fence round
// trips per frame were ~40% of the whole compute cost.
bool g_cs_batch_open = false;
u32 g_cs_batch_count = 0;
// Batches are submitted without waiting and retired in order as their
// submissions complete; a reader waits for the one batch its range was last
// recorded into (CsBatchWaitId), and only then. The GPU then executes batch N
// while the walk records N+1, where a synchronous flush idled both in turn.
constexpr u32 kCsBatchRing = 6;
u64 g_cs_batch_id = 0;  // the open batch
u64 g_cs_batch_next_id = 1;
u64 g_cs_batch_done = 0;  // every batch with an id <= this has completed
u32 g_cs_batch_cur = 0;   // ring slot the open batch records into
u64 g_cs_submit_n = 0;
// What the open batch contains. A device loss names no dispatch on its own, and
// a batch is up to 128 of them, so the shader that killed the queue is
// otherwise unattributable.
struct BatchedDispatch {
  u64 cs_addr;
  u32 groups[3];
  u32 num_res;
  u64 res_base[4];
  u64 res_size[4];
  u8 res_img;    // bit i: resource i is an image
  u8 res_write;  // bit i: resource i is written
  bool traced = false;  // also a zone on the Tracy GPU timeline
  u16 query = 0;
};
base::Vector<BatchedDispatch> g_cs_batch_log;
rhi::TimestampPool* g_cs_timestamps = nullptr;
struct CsGpuTime {
  double ns = 0;
  u64 count = 0;
};
base::HashMap<u64, CsGpuTime> g_cs_gpu_times;
base::HashMap<rhi::Buffer*, ComputeBufferAccess> g_cs_batch_access;

struct CsBatch {
  rhi::CommandList* list = nullptr;
  rhi::TimestampPool* timestamps = nullptr;
  u64 submission = 0;  // the device submission carrying it
  u64 id = 0;
  bool submitted = false;
  u32 count = 0;
  base::Vector<BatchedDispatch> log;
};
CsBatch g_cs_batches[kCsBatchRing];

// The batch a range referenced now belongs to: the open one, or the one that
// opens next when nothing is recording yet.
void MarkPending(CsRange& e) {
  if (!e.pending_batch)
    g_cs_pending.push_back(&e);
  e.pending_batch = true;
  e.batch_id = g_cs_batch_open ? g_cs_batch_id : g_cs_batch_next_id;
}

bool CsBatchInit() {
  if (g_cs_batches[0].list)
    return true;
  for (auto& b : g_cs_batches) {
    b.list = Device().CreateCommandList();
    if (!b.list)
      return false;
    // Two timestamps for each of at most 128 dispatches.
#if defined(DELTA_TRACY)
    constexpr bool kTimeline = true;  // dispatches on the Tracy GPU timeline
#else
    constexpr bool kTimeline = false;
#endif
    if ((kCsSyncReport || kTimeline) && Device().caps().timestamps)
      b.timestamps = Device().CreateTimestampPool(256);
  }
  g_cs_list = g_cs_batches[0].list;
  return true;
}

void CsBatchLogFault(const CsBatch& b, const char* what) {
  BASE_LOGI("gpuvk", "cs batch DEVICE FAULT: {} failed id={} n={}", what,
            (unsigned long long)b.id, b.count);
  for (const auto& d : b.log) {
    base::String res;
    for (u32 i = 0; i < d.num_res && i < 4; i++)
      base::FormatTo(res, " [{}]{:#x}+{:#x}{}{}", i,
                     (unsigned long)d.res_base[i], (unsigned long)d.res_size[i],
                     (d.res_img >> i) & 1 ? " img" : " buf",
                     (d.res_write >> i) & 1 ? " w" : "");
    BASE_LOGI("gpuvk", "  batched cs={:#x} groups=[{} {} {}] res={}{}",
              (unsigned long)d.cs_addr, d.groups[0], d.groups[1], d.groups[2],
              d.num_res, res.c_str());
  }
  Device().ReportDeviceLoss();
  g_cs_failed = true;
}

// The batch's submission has completed: what it produced is real now.
void CsBatchFinalize(CsBatch& b) {
  if (b.timestamps && !b.log.empty()) {
    base::Array<u64, 256> stamps{};
    const u32 count = static_cast<u32>(b.log.size());
    const bool any_traced = base::AnyOf(
        b.log.begin(), b.log.end(),
        [](const BatchedDispatch& d) { return d.traced; });
    const bool read =
        (kCsSyncReport || any_traced) &&
        Device().ReadTimestamps(b.timestamps, 0, count * 2, stamps.data());
    for (u32 i = 0; i < count; ++i) {
      if (!b.log[i].traced)
        continue;
      timeline::Time(b.log[i].query, read ? stamps[i * 2] : 0);
      timeline::Time(b.log[i].query + 1, read ? stamps[i * 2 + 1] : 0);
    }
    if (read && kCsSyncReport) {
      const rhi::Caps& caps = Device().caps();
      const u64 mask = UINT64_MAX >> (64 - caps.timestamp_bits);
      for (u32 i = 0; i < count; ++i) {
        auto& time = g_cs_gpu_times[b.log[i].cs_addr];
        time.ns += ((stamps[i * 2 + 1] - stamps[i * 2]) & mask) *
                   caps.timestamp_period_ns;
        ++time.count;
      }
    }
  }
  g_cs_batch_done = base::Max(g_cs_batch_done, b.id);
  base::EraseIf(g_cs_pending, [](CsRange* range) {
    CsRange& e = *range;
    if (!e.pending_batch)
      return true;
    if (e.batch_id > g_cs_batch_done)
      return false;
    e.pending_batch = false;
    // Every readback recorded into it is in the host mirror now.
    if (e.readback_pending) {
      e.readback_pending = false;
      e.mirror_current = true;
    }
    return true;
  });
  b.submitted = false;
  b.log.clear();
  b.count = 0;
  // Imported ranges are written by the dispatches this batch executed,
  // straight into guest memory, and no writeback step will announce them, so
  // the batch itself is the visibility point for the staging caches.
  g_cs_writeback_gen++;
}

// Retire, in order, every submitted batch that has completed.
void CsBatchReap() {
  for (;;) {
    CsBatch* oldest = nullptr;
    for (auto& b : g_cs_batches)
      if (b.submitted && (!oldest || b.id < oldest->id))
        oldest = &b;
    if (!oldest || !Device().IsComplete(oldest->submission))
      return;
    CsBatchFinalize(*oldest);
  }
}

// DELTA_GPU_CSSYNC=1: how many submit+wait round trips a frame, and what asked
// for each. The wait itself is the single biggest term in a SotC frame, so the
// question that matters is which call site is forcing it.
enum CsSyncWhy {
  kSyncImported,
  kSyncWriteback,
  kSyncScratchGrow,
  kSyncRangeGrow,
  kSyncStageHazard,
  kSyncBatchCap,
  kSyncFrameEnd,
  kSyncChunk,
  kSyncRingFull,
  kSyncCount
};
const char* const kCsSyncName[kSyncCount] = {
    "imported",  "writeback", "scratch-grow", "range-grow", "stage-hazard",
    "batch-cap", "frame-end", "chunk",        "ring-full"};
// The frame's open chunk copies a texture out of a range the open batch
// writes: the batch has to be on the queue before the chunk is.
bool g_cs_chunk_needs_batch = false;
u64 g_cs_sync_n[kSyncCount] = {};
u64 g_cs_sync_ns[kSyncCount] = {};
// The guest reader a writeback wait is paid for (FlushCsWritesRange's `why`).
const char* g_cs_wait_reader = "other";
base::Map<base::String, base::Pair<u64, u64>> g_cs_reader_waits;
struct WaitReaderScope {
  explicit WaitReaderScope(const char* why) { g_cs_wait_reader = why; }
  ~WaitReaderScope() { g_cs_wait_reader = prev; }
  const char* prev = g_cs_wait_reader;
};

// Open the batch command buffer. Staging copies are recorded into the same
// buffer as the dispatches, so this has to be callable before the first one.
void CsBatchBeginImpl() {
  if (g_cs_batch_open || !CsBatchInit())
    return;
  CsBatchReap();
  CsBatch& b = g_cs_batches[g_cs_batch_cur];
  if (b.submitted) {
    // The ring has come round to a batch still in flight.
    const u64 t0 = NowNs();
    if (!Device().Wait(b.submission))
      CsBatchLogFault(b, "wait");
    CsBatchFinalize(b);
    g_ns_cs_gpu += NowNs() - t0;
    g_cs_sync_n[kSyncRingFull]++;
    g_cs_sync_ns[kSyncRingFull] += NowNs() - t0;
  }
  g_cs_list = b.list;
  g_cs_timestamps = b.timestamps;
  b.id = g_cs_batch_next_id++;
  g_cs_batch_id = b.id;
  g_cs_list->Begin();
  if (g_cs_timestamps)
    g_cs_list->ResetTimestamps(g_cs_timestamps, 0, 256);
  if (Device().caps().debug_labels) {
    char label[48];
    std::snprintf(label, sizeof(label), "cs batch (frame %llu)",
                  (unsigned long long)g_frame.num);
    g_cs_list->PushLabel(label);
  }
  if (kCsTexBridge) {
    // A frame command list submitted before this batch may still be copying
    // a bridged texture out of a range the batch overwrites.
    g_cs_list->Barrier(kAccessAll, kAccessAll);
  }
  g_cs_batch_open = true;
}

// Record the staging move into the batch, and mark the range as referenced by
// it so a later grow/rename does not pull the buffer out from under the copy.
void CsCopyStaging(CsRange& e, u64 bytes, bool to_device) {
  if (!e.device_local || !e.host_buf || !e.buf || !bytes)
    return;
  CsBatchBeginImpl();
  RecordStagingCopy(g_cs_list, e, bytes, to_device);
  g_cs_batch_access.erase(e.buf);
  MarkPending(e);
  if (to_device)
    e.mirror_current = true;  // both sides now hold what the CPU just wrote
  else
    e.readback_pending = true;
}

// Put the open batch on the queue and go on recording into the next slot.
bool CsBatchSubmit() {
  DELTA_ZONE("gpu.cs_batch_submit");
  if (!g_cs_batch_open)
    return !g_cs_failed;
  CsBatch& b = g_cs_batches[g_cs_batch_cur];
  base::String where;
  base::FormatTo(where, "batch n={} last-cs={:#x}", g_cs_batch_count,
                 g_cs_batch_log.empty() ? 0 : g_cs_batch_log.back().cs_addr);
  if (!QueueCheck(where.c_str())) {
    g_cs_failed = true;
    return false;
  }
  if (Device().caps().debug_labels)
    g_cs_list->PopLabel();  // close the "cs batch" scope
  b.count = g_cs_batch_count;
  b.log = base::move(g_cs_batch_log);
  g_cs_batch_log.clear();
  g_cs_list->End();
  // A failed submit returns an id that has already been handed out.
  const u64 before = Device().LastSubmission();
  b.submission = Device().Submit(g_cs_list);
  g_cs_batch_open = false;
  g_cs_chunk_needs_batch = false;
  g_cs_batch_count = 0;
  g_cs_batch_access.clear();
  g_cs_submit_n++;
  if (b.submission <= before) {
    CsBatchLogFault(b, "submit");
    return false;
  }
  b.submitted = true;
  g_cs_batch_cur = (g_cs_batch_cur + 1) % kCsBatchRing;
  return true;
}

// Wait until batch `id` (and so every earlier one) has completed.
bool CsBatchWaitId(u64 id, CsSyncWhy why) {
  DELTA_ZONE("gpu.cs_wait");
  if (g_cs_failed)
    return false;
  if (!id || id <= g_cs_batch_done)
    return true;
  if (g_cs_batch_open && id >= g_cs_batch_id && !CsBatchSubmit())
    return false;
  const u64 t0 = NowNs();
  for (;;) {
    CsBatch* oldest = nullptr;
    for (auto& b : g_cs_batches)
      if (b.submitted && b.id <= id && (!oldest || b.id < oldest->id))
        oldest = &b;
    if (!oldest)
      break;
    if (!Device().Wait(oldest->submission)) {
      CsBatchLogFault(*oldest, "wait");
      return false;
    }
    CsBatchFinalize(*oldest);
  }
  g_ns_cs_gpu += NowNs() - t0;
  g_cs_sync_n[why]++;
  g_cs_sync_ns[why] += NowNs() - t0;
  if (kCsSyncReport) {
    auto& waits = g_cs_reader_waits[g_cs_wait_reader];
    waits.first++;
    waits.second += NowNs() - t0;
  }
  return !g_cs_failed;
}

bool CsBatchWaitRange(const CsRange& e, CsSyncWhy why) {
  return !e.pending_batch || CsBatchWaitId(e.batch_id, why);
}

// Everything recorded so far, complete.
bool CsBatchFlush(CsSyncWhy why = kSyncWriteback) {
  const u64 last = g_cs_batch_open ? g_cs_batch_id : g_cs_batch_next_id - 1;
  return CsBatchWaitId(last, why);
}

// Record the VRAM->host readback for every dirty range in [first, last), so
// the single flush that follows covers all of them.
// A truth is what every GPU consumer reads from, so only a CPU reader ever
// needs its guest bytes, and those ask on demand; a live target aliasing it
// still wants the image refreshed at frame end.
bool CsLazyKept(u64 base, const CsRange& e) {
  return kCsImageLazyWb && e.truth && e.gpu_dirty && !e.imported &&
         !CsAliasedBase(base);
}

// `all`: every dirty range, or only those nothing keeps in VRAM on purpose.
void CsStageReadback(u64 base, CsRange& e, bool all) {
  if (e.gpu_dirty && e.device_local && !e.mirror_current &&
      !e.readback_pending && !e.imported &&
      (!CanGpuTile(e.res, e) || e.truth) && (all || !CsLazyKept(base, e)))
    CsCopyStaging(e, e.size ? e.size : e.cap, /*to_device=*/false);
}

void CsStageReadbacks(bool all = true) {
  // A copy: staging can open a batch, and finalizing an old one may unindex.
  base::Vector<u64> bases;
  bases.reserve(g_cs_dirty_bases.size());
  for (const auto& [base, end] : g_cs_dirty_bases)
    bases.push_back(base);
  for (u64 base : bases) {
    auto found = g_cs_ranges.find(base);
    if (found != g_cs_ranges.end())
      CsStageReadback(base, found->second, all);
  }
}

// Submit the frame's open chunk, behind the open batch when the chunk copies
// out of a range that batch writes.
bool CsSplitFrameChunk() {
  if (g_cs_chunk_needs_batch && !CsBatchSubmit())
    return false;
  if (!SubmitFrameChunk())
    return false;
  g_cs_chunk_splits++;
  return true;
}

// Write one dirty range back to guest memory (retile for images) and re-stamp
// its hash so the next validation sees guest == buffer.
bool CsRangeFlushOne(u64 base, CsRange& e) {
  if (kCsWbAudit) {
    // Counters, not a sample: the first N flushes are all startup, and the
    // question ("does a tiled-image range ever reach the retile?") is about
    // the steady state.
    static u64 n = 0, dirty_n = 0, img_n = 0, img_dirty_n = 0;
    n++;
    dirty_n += e.gpu_dirty;
    img_n += e.image_staging;
    img_dirty_n += e.gpu_dirty && e.image_staging;
    if ((n % 20000) == 0)
      BASE_LOGI("cswb",
                "flushes={} dirty={} image={} image+dirty={} "
                "staged_as_image={}",
                (unsigned long long)n, (unsigned long long)dirty_n,
                (unsigned long long)img_n, (unsigned long long)img_dirty_n,
                (unsigned long long)g_cs_image_staged);
  }
  // A clean range writes nothing, so texture answers memoized on the epoch
  // still hold. Callers sweep every overlapping sibling through here.
  if (!e.gpu_dirty)
    return true;
  DELTA_ZONE("gpu.cs_writeback");
  BumpTextureEpoch();
  if (e.imported) {  // the dispatch wrote straight into guest memory
    if (!CsBatchWaitRange(e, kSyncImported))
      return false;
    NoteGuestWrite(base, e.guest_bytes);
    e.gpu_dirty = false;
    e.cpu_writes.clear();
    return true;
  }
  if (CsTracksGuestWrites())
    CollectGuestWritesForSubmission();
  // Results live in VRAM: pull them into the host mirror on the same batch that
  // produced them, so the one fence wait covers the dispatch and the copy.
  // Already-recorded readbacks (CsStageReadbacks) skip this.
  if (e.device_local && !e.mirror_current && !e.readback_pending &&
      (!CanGpuTile(e.res, e) || e.truth))
    CsCopyStaging(e, e.size ? e.size : e.cap, /*to_device=*/false);
  if (e.pending_batch) {
    const u64 tw = NowNs();
    const bool ok = CsBatchWaitRange(e, kSyncWriteback);
    if (kCsSyncReport) {
      StageStat& st = g_stage_stats[base];
      st.wait++;
      st.wait_ns += NowNs() - tw;
    }
    if (!ok)
      return false;  // results must exist before readback
  }
  if (g_cs_failed)
    return false;
  g_cs_flush_n++;
  if (e.image_staging && kCsWbAudit) {
    // An image range always retiles, with no write-coverage test to report, so
    // "the writeback ran" says nothing about whether the dispatch produced
    // anything. Count what the staging actually holds: all-zero here means the
    // shader stored nothing, which is a different bug from a failed retile.
    const auto* p = static_cast<const u8*>(e.map);
    u64 nz = 0;
    for (u64 i = 0; i < e.res.size; i++)
      if (p[i])
        nz++;
    // Once per destination, not the first 24 overall: a title that uploads
    // hundreds of surfaces spends a flat cap entirely on the first few, and
    // the one surface that stays empty is never among them.
    static base::HashSet<u64> seen;
    if (seen.size() < 4096 && seen.insert(base).second)
      BASE_LOGI("cswb",
                "image base={:#x} {}x{} layers={} tiling={} staged {}/{} "
                "non-zero",
                (unsigned long long)base, e.res.width, e.res.height,
                e.res.layers, e.res.tiling_idx, (unsigned long long)nz,
                (unsigned long long)e.res.size);
  }
  const u64 t_wb = NowNs();
  // A range with a shadow merges the guest's writes word by word below. Any
  // other writeback covers whole pages, so with write tracking the pages the
  // CPU wrote since the dispatch are set aside here and put back after it.
  // Without it, a report can be of our own earlier writeback.
  const u64 linear_n =
      e.guest_bytes ? base::Min<u64>(e.size, e.guest_bytes) : e.size;
  const bool merges = (!e.image_staging || e.truth) && !kCsWbFull &&
                      !e.rt_sourced && e.shadow_valid &&
                      e.shadow.size() >= linear_n;
  struct KeptPages {
    base::Vector<base::Pair<u64, u64>> spans;
    base::Vector<u8> bytes;
    ~KeptPages() {
      u64 off = 0;
      for (const auto& [first, end] : spans) {
        std::memcpy(reinterpret_cast<void*>(first), bytes.data() + off,
                    end - first);
        off += end - first;
      }
    }
  } kept;
  if (!merges && CsTracksGuestWrites()) {
    const u64 footprint_end = base + (e.guest_bytes ? e.guest_bytes : e.size);
    for (const auto& [first, end] : e.cpu_writes) {
      const u64 lo = base::Max(first, base), hi = base::Min(end, footprint_end);
      if (lo >= hi)
        continue;
      kept.spans.emplace_back(lo, hi);
      const u64 at = kept.bytes.size();
      kept.bytes.resize(at + (hi - lo));
      std::memcpy(kept.bytes.data() + at, reinterpret_cast<const void*>(lo),
                  hi - lo);
    }
  }
  bool adopted = !kept.spans.empty();
  // DELTA_GPU_CS_SKIP_RETILE: an image the GPU refresh below can carry needs no
  // CPU retile into guest memory. Only readers that go through guest memory
  // (the texture cache's own upload, a CP DMA, the guest CPU) lose anything,
  // and for a live image the draw path samples the image instead.
  CsAliasedImage refreshed{};
  const bool gpu_carries =
      kCsSkipRetile && e.image_staging && !e.truth &&
      FindCsAliasedImage(base, refreshed, /*for_write=*/true,
                         /*prefer_depth=*/e.res.dfmt == 4);
  if (e.image_staging && !e.truth && !gpu_carries) {
    const bool converted = ConvertCsRangeImage(e.res, e, false);
    if (!converted && e.device_local && !e.mirror_current) {
      CsCopyStaging(e, e.size, /*to_device=*/false);
      if (!CsBatchWaitRange(e, kSyncWriteback))
        return false;
    }
    if (!converted && (g_cs_failed || !WritebackCsImage(e.res, e.map))) {
      // Count as well as sample: a flat cap of 8 lines cannot tell one
      // recurring bad descriptor apart from every upload in the title failing,
      // and those need opposite responses.
      static u64 failed = 0;
      if (failed++ < 8 || (failed % 4096) == 0)
        BASE_LOGI("gpuvk",
                  "cs image writeback #{} failed base={:#x} {}x{} mips={} "
                  "layers={} tiling={} elem={}/{} (range stays stale)",
                  (unsigned long long)failed, (unsigned long long)base,
                  e.res.width, e.res.height, e.res.mip_levels, e.res.layers,
                  e.res.tiling_idx, e.res.elem_bytes, e.res.stage_elem_bytes);
      return false;
    }
  } else if (!e.image_staging || e.truth) {
    // Never write back more than the guest footprint the range was staged
    // from, and never into memory that is not the guest's: the staged size is
    // the shader's view of the resource and a descriptor with a bogus size
    // would otherwise scribble compute results over whatever follows, which
    // reads as float data turning up in the title's allocator free lists.
    const u64 n =
        e.guest_bytes ? base::Min<u64>(e.size, e.guest_bytes) : e.size;
    if (!gpu::IsReadableRange(base, n)) {
      static int logged = 0;
      if (logged++ < 8)
        BASE_LOGI("gpuvk",
                  "cs writeback out of guest memory base={:#x} +{:#x} "
                  "(dropped)",
                  (unsigned long long)base, (unsigned long long)n);
      return false;
    }
    // Write back only what the DISPATCH changed. A staged range is the shader's
    // whole view of a resource, but a shader writes a part of it, and this
    // title puts its CPU heap in the same direct memory as the buffers it hands
    // to compute, so the bytes in between belong to the allocator. Copying the
    // full range back reverts every CPU write made since stage-in, and a word
    // the CPU was writing WHILE we snapshotted comes back half updated: the
    // 5-valid-bytes-plus-3-stale-bytes free-tree pointer SotC dies on at
    // eboot+0x48ac5 when New Game is confirmed. The shadow copy taken at
    // stage-in says which words the shader touched; untouched words are left
    // alone. The merge is symmetric, so staging, shadow and guest agree again
    // when it returns:
    //   the shader wrote the block  -> guest   <- staging  (publish the result)
    //   the shader left it alone    -> staging <- guest    (adopt the CPU's
    //   word)
    // Guest-sourced linear ranges only: a range staged from a live render
    // target exists precisely so the writeback can publish that image into
    // guest memory, so "the dispatch did not write it" must not stop it there.
    //
    // A block the CPU also changed since stage-in (guest != shadow) keeps the
    // CPU's bytes even where the shader wrote it: the dispatch ran before the
    // CPU write, and a title that reuses a compute buffer's memory for its
    // descriptor tables otherwise gets old dispatch output over them.
    if (merges) {
      auto* src = static_cast<u8*>(e.map);
      u8* shd = e.shadow.data();
      auto* dst = reinterpret_cast<u8*>(base);
      const u64 blocks = n / 64;
      u64 off = blocks * 64, wrote = 0;
      // 64-byte blocks: a block the shader left alone costs one compare, which
      // keeps "the shader wrote a little of a big range" (the common case)
      // no more expensive than the unconditional copy this replaces. Blocks are
      // independent, so the sweep is split across the detiler's pool; the
      // per-lane written counts are summed rather than shared.
      constexpr u64 kMergeBlocksPerChunk = 1024;  // 64 KiB a chunk
      const u32 chunks = static_cast<u32>((blocks + kMergeBlocksPerChunk - 1) /
                                          kMergeBlocksPerChunk);
      struct ChunkCounts {
        u64 wrote = 0, adopted = 0, conflicts = 0;
      };
      base::Vector<ChunkCounts> counts(chunks);
      gcn::DetileParallelWork(chunks, blocks * 64, [&](u32 c0, u32 c1) {
        for (u32 c = c0; c < c1; c++) {
          const u64 first = u64(c) * kMergeBlocksPerChunk;
          const u64 last = base::Min<u64>(first + kMergeBlocksPerChunk, blocks);
          ChunkCounts local;
          for (u64 b = first; b < last; b++) {
            const u64 o = b * 64;
            const bool gpu = std::memcmp(src + o, shd + o, 64) != 0;
            const bool cpu = std::memcmp(dst + o, shd + o, 64) != 0;
            if (cpu && gpu) {
              // Both changed the block: merge it word by word, the CPU's
              // word winning where both changed the same one.
              local.conflicts++;
              for (u64 w = o; w < o + 64; w += 4) {
                if (std::memcmp(dst + w, shd + w, 4) != 0) {
                  std::memcpy(src + w, dst + w, 4);
                  local.adopted += 4;
                } else if (std::memcmp(src + w, shd + w, 4) != 0) {
                  std::memcpy(dst + w, src + w, 4);
                  local.wrote += 4;
                }
                std::memcpy(shd + w, src + w, 4);
              }
            } else if (cpu) {
              std::memcpy(src + o, dst + o, 64);
              std::memcpy(shd + o, dst + o, 64);
              local.adopted += 64;
            } else if (gpu) {
              std::memcpy(dst + o, src + o, 64);
              std::memcpy(shd + o, src + o, 64);
              local.wrote += 64;
            }
          }
          counts[c] = local;
        }
      });
      u64 adopted_bytes = 0;
      for (const ChunkCounts& c : counts) {
        wrote += c.wrote;
        adopted_bytes += c.adopted;
        g_cs_wb_conflicts += c.conflicts;
      }
      for (; off < n; off++) {
        if (dst[off] != shd[off]) {
          src[off] = dst[off];
          shd[off] = dst[off];
          adopted_bytes++;
        } else if (src[off] != shd[off]) {
          dst[off] = src[off];
          shd[off] = src[off];
          wrote++;
        }
      }
      adopted |= adopted_bytes != 0;
      g_cs_wb_bytes_written += wrote;
      g_cs_wb_bytes_total += n;
      if (kCsWbAudit && wrote != n) {
        static int logged = 0;
        if (logged++ < 32)
          BASE_LOGI("cswb",
                    "base={:#x} +{:#x}: shader wrote {} of {} bytes; the other "
                    "{} would have been reverted",
                    (unsigned long long)base, (unsigned long long)n,
                    (unsigned long long)wrote, (unsigned long long)n,
                    (unsigned long long)(n - wrote));
      }
    } else {
      // Full-range writeback (the pre-merge behaviour, kept for A/B). Still
      // measure how much of it the dispatch actually wrote when a shadow is
      // available, so the two modes report the same number and the A/B is a
      // comparison rather than a guess.
      u64 wrote = n;
      if (e.shadow_valid && e.shadow.size() >= n) {
        wrote = 0;
        const auto* src = static_cast<const u8*>(e.map);
        const u8* shd = e.shadow.data();
        for (u64 off = 0; off < n; off += 64) {
          const u64 blk = base::Min<u64>(64, n - off);
          if (std::memcmp(src + off, shd + off, blk) != 0)
            wrote += blk;
        }
        if (kCsWbAudit && wrote != n) {
          static int logged = 0;
          if (logged++ < 32)
            BASE_LOGI("cswb",
                      "base={:#x} +{:#x}: shader wrote {} of {} bytes; the "
                      "other {} ARE being reverted",
                      (unsigned long long)base, (unsigned long long)n,
                      (unsigned long long)wrote, (unsigned long long)n,
                      (unsigned long long)(n - wrote));
        }
      }
      std::memcpy(reinterpret_cast<void*>(base), e.map, n);
      g_cs_wb_bytes_written += wrote;
      g_cs_wb_bytes_total += n;
    }
  }
  const u64 t_rt = NowNs();
  if (e.image_staging)
    g_out_retile_ns += t_rt - t_wb;
  else
    g_out_buffer_ns += t_rt - t_wb;
  g_out_retile_n += e.image_staging;
  UploadCsRangeToRt(base, e);  // refresh a live RT image aliasing the range
  const u64 t_inv = NowNs();
  g_out_rt_ns += t_inv - t_rt;
  InvalidateTexRange(base, e.guest_bytes);
  UnindexDirtyRange(base, e.guest_bytes);
  // Guest memory just changed under any staged copy of it: retire the draw
  // path's per-frame staging cache entries (they validate against this).
  g_cs_writeback_gen++;
  // Our own writes to the range are reported like the guest's; take them now
  // so the next dispatch does not read them as CPU writes after it.
  if (CsTracksGuestWrites())
    CollectGuestWrites();
  e.gpu_dirty = false;
  e.cpu_writes.clear();
  if (adopted) {
    // The buffer lacks the CPU's bytes now in guest memory: stage it again
    // before anything runs on it.
    e.hash = 0;
    e.last_validated_frame = -1;
  } else {
    e.hash = TexHash(base, e.guest_bytes);
    e.last_validated_frame = g_frame.num;
  }
  g_out_tail_ns += NowNs() - t_inv;
  return true;
}

// Ensure staging slot i can hold `size` bytes (grow-on-demand, kept mapped).
bool CsEnsureStage(u32 i, u64 size) {
  GPU_BUGCHECK(i < ComputeInfo::kMaxResources, "stage index %u out of bounds",
               i);
  CsStage& s = g_cs_stage[i];
  if (s.buf && s.cap >= size)
    return true;
  s.map = nullptr;
  // A batch in flight may still bind the old scratch: it outlives the frame.
  if (s.buf)
    g_cs_frame_retired.push_back(s.buf);
  s.buf = nullptr;
  const u64 cap = (size + 0xFFFFF) & ~u64(0xFFFFF);  // 1 MiB granularity
  // Scratch is zeroed by FillBuffer and only ever touched by shaders, so it
  // wants VRAM and no mapping at all.
  const bool vram = kCsVram && SplitVram();
  s.buf = CreateCsBuffer(
      cap, rhi::kBufferStorage | rhi::kBufferCopyDst,
      vram ? rhi::MemoryKind::kDevice : rhi::MemoryKind::kReadback);
  if (!s.buf)
    return false;
  if (!vram)
    s.map = s.buf->mapped();
  s.cap = cap;
  return true;
}

struct ScopeCs {
  u64 t0 = NowNs();
  ~ScopeCs() {
    g_ns_cs += NowNs() - t0;
    g_cs_count++;
  }
};

}  // namespace

bool ConvertGfx10Image(const gcn::TextureLayout32& tiled,
                       const gcn::TextureLayout32& linear,
                       const void* src,
                       void* dst,
                       bool detile) {
#ifdef DELTA_HAVE_SPIRV_BACKEND
  const u64 started = NowNs();
  const bool ok = ConvertGfx10ImageImpl(tiled, linear, src, dst, detile);
  if (ok && kCsSyncReport) {
    g_gpu_tiling_ns[detile] += NowNs() - started;
    g_gpu_tiling_n[detile]++;
  }
  return ok;
#else
  return false;
#endif
}

void CsSyncReport(double frames) {
  if (kCsSyncReport && frames > 0) {
    base::Vector<base::Pair<u64, CsGpuTime>> times;
    for (const auto& [base, time] : g_cs_gpu_times)
      times.push_back({base, time});
    base::Sort(times.begin(), times.end(), [](const auto& a, const auto& b) {
      return a.second.ns > b.second.ns;
    });
    for (size_t i = 0; i < base::Min<size_t>(8, times.size()); ++i)
      BASE_LOGI("cstime", "CS {:#x} {:.2f}ms/f x{:.1f}", times[i].first,
                times[i].second.ns / frames / 1e6,
                double(times[i].second.count) / frames);
    g_cs_gpu_times.clear();
  }
  u64 total = 0;
  for (int i = 0; i < kSyncCount; i++)
    total += g_cs_sync_n[i];
  if (!kCsSyncReport || !total) {
    for (int i = 0; i < kSyncCount; i++)
      g_cs_sync_n[i] = g_cs_sync_ns[i] = 0;
    return;
  }
  BASE_LOGI("csin",
            "hash={:.1f}ms x{:.1f} detile={:.1f}ms rt-bridge={:.1f}ms x{:.1f} "
            "(in-frame x{:.1f}) copy={:.1f}ms",
            g_in_hash_ns / frames / 1e6, g_in_hash_n / frames,
            g_in_detile_ns / frames / 1e6, g_in_rt_ns / frames / 1e6,
            g_in_rt_n / frames, g_in_frame_bridge_n / frames,
            g_in_copy_ns / frames / 1e6);
  g_in_hash_ns = g_in_detile_ns = g_in_rt_ns = g_in_copy_ns = 0;
  g_in_hash_n = g_in_rt_n = g_in_frame_bridge_n = 0;
  BASE_LOGI(
      "csin2",
      "overlap-wb={:.1f}ms x{:.1f} reshape-wb={:.1f}ms x{:.1f} "
      "alloc={:.1f}ms x{:.1f} recycled={:.1f} ranges={} {:.0f}MB "
      "membuf={:.1f}ms "
      "shadow={:.1f}ms lazy-kept={:.1f}x {:.1f}MB | truth stage={:.1f}MB "
      "import={:.1f} detile={:.1f} retile={:.1f} parent={:.1f} "
      "fold={:.1f} rt={:.1f} view-hits={:.1f} scratch={:.0f}MB",
      g_in_overlap_ns / frames / 1e6, g_in_overlap_n / frames,
      g_in_reshape_ns / frames / 1e6, g_in_reshape_n / frames,
      g_in_alloc_ns / frames / 1e6, g_in_alloc_n / frames,
      g_cs_recycled_n / frames, g_cs_ranges.size(), g_cs_range_bytes / 1e6,
      g_in_membuf_ns / frames / 1e6, g_in_shadow_ns / frames / 1e6,
      g_lazy_kept_n / frames, g_lazy_kept_bytes / frames / 1e6,
      g_truth_stage_bytes / frames / 1e6, g_truth_import_n / frames,
      g_truth_detile_n / frames, g_truth_retile_n / frames,
      g_truth_parent_n / frames, g_truth_fold_n / frames, g_truth_rt_n / frames,
      g_truth_view_hits / frames, g_cs_scratch_bytes / 1e6);
  g_truth_view_hits = 0;
  {
    base::String why;
    base::FormatTo(
        why, "writebacks/frame ({:.1f}MB):", g_wb_why_bytes / frames / 1e6);
    for (const auto& kv : g_wb_why)
      base::FormatTo(why, " {}={:.1f}", kv.first.c_str(), kv.second / frames);
    BASE_LOGI("cswb", "{}", why.c_str());
    g_wb_why.clear();
    g_wb_why_bytes = 0;
  }
  g_in_overlap_ns = g_in_reshape_ns = g_in_alloc_ns = 0;
  g_in_membuf_ns = g_in_shadow_ns = 0;
  g_in_overlap_n = g_in_reshape_n = g_in_alloc_n = g_cs_recycled_n = 0;
  g_truth_stage_bytes = g_truth_import_n = g_truth_detile_n = 0;
  g_truth_retile_n = g_truth_parent_n = g_truth_fold_n = g_truth_rt_n = 0;
  g_lazy_kept_n = g_lazy_kept_bytes = 0;
  BASE_LOGI("cstiling",
            "GPU conversion including copies: in={:.1f}ms x{:.1f} "
            "out={:.1f}ms x{:.1f}",
            g_gpu_tiling_ns[1] / frames / 1e6, g_gpu_tiling_n[1] / frames,
            g_gpu_tiling_ns[0] / frames / 1e6, g_gpu_tiling_n[0] / frames);
  // Where that conversion time goes: "rest" is the address-table build, the
  // descriptor writes and the command recording.
  const double tile_all =
      double(g_gpu_tiling_ns[0] + g_gpu_tiling_ns[1]) / frames / 1e6;
  BASE_LOGI("cstilesplit",
            "host-in={:.1f}ms {:.1f}MB gpu-wait={:.1f}ms host-out={:.1f}ms "
            "{:.1f}MB rest={:.1f}ms",
            g_tile_hin_ns / frames / 1e6, g_tile_hin_bytes / frames / 1e6,
            g_tile_sync_ns / frames / 1e6, g_tile_hout_ns / frames / 1e6,
            g_tile_hout_bytes / frames / 1e6,
            tile_all - (g_tile_hin_ns + g_tile_sync_ns + g_tile_hout_ns) /
                           frames / 1e6);
  g_tile_hin_ns = g_tile_sync_ns = g_tile_hout_ns = 0;
  g_tile_hin_bytes = g_tile_hout_bytes = 0;
  g_gpu_tiling_ns[0] = g_gpu_tiling_ns[1] = 0;
  g_gpu_tiling_n[0] = g_gpu_tiling_n[1] = 0;
  BASE_LOGI("csstage",
            "per frame ro={:.1f}MB rw={:.1f}MB img={:.1f}MB (guest-detile "
            "{:.1f}MB x{:.1f})",
            g_stage_ro_bytes / frames / 1e6, g_stage_rw_bytes / frames / 1e6,
            g_stage_img_bytes / frames / 1e6,
            g_stage_guest_detile_bytes / frames / 1e6,
            g_stage_guest_detile_n / frames);
  g_stage_ro_bytes = g_stage_rw_bytes = g_stage_img_bytes = 0;
  g_stage_guest_detile_bytes = g_stage_guest_detile_n = 0;
  // The ranges that actually cost the frame, most expensive first: what to fix
  // is whichever reason column is not "hash" (guest memory genuinely changed).
  base::Vector<base::Pair<u64, StageStat>> top;
  for (const auto& [base, stat] : g_stage_stats)
    top.push_back({base, stat});
  base::Sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
    return a.second.bytes + a.second.wait_ns / 8 >
           b.second.bytes + b.second.wait_ns / 8;
  });
  for (size_t i = 0; i < top.size() && i < 8; i++) {
    const StageStat& s = top[i].second;
    const auto found = g_cs_ranges.find(top[i].first);
    const ComputeInfo::Res* r =
        found == g_cs_ranges.end() ? nullptr : &found->second.res;
    BASE_LOGI("cstop",
              "{:#x} {:.1f}MB/f x{:.1f} (first={:.1f} shape={:.1f} hash={:.1f} "
              "rt={:.1f}) wait={:.1f}x {:.1f}ms | {}x{} l={} m={} e={} t={}",
              (unsigned long)top[i].first, s.bytes / frames / 1e6, s.n / frames,
              s.first / frames, s.shape / frames, s.hash / frames,
              s.rt / frames, s.wait / frames, s.wait_ns / frames / 1e6,
              r ? r->width : 0, r ? r->height : 0, r ? r->layers : 0,
              r ? r->mip_levels : 0, r ? r->elem_bytes : 0,
              r ? r->tiling_idx : 0);
  }
  g_stage_stats.clear();
  BASE_LOGI("csout",
            "retile={:.1f}ms x{:.1f} buffers={:.1f}ms "
            "rt-upload={:.1f}ms x{:.1f} tail={:.1f}ms "
            "texbridge={:.1f}x {:.1f}MB bufbridge={:.1f}x {:.1f}MB "
            "(declined {:.1f}) vtxbridge={:.1f}x chunks={:.1f}",
            g_out_retile_ns / frames / 1e6, g_out_retile_n / frames,
            g_out_buffer_ns / frames / 1e6, g_out_rt_ns / frames / 1e6,
            g_out_rt_submits / frames, g_out_tail_ns / frames / 1e6,
            g_tex_bridge_n / frames, g_tex_bridge_bytes / frames / 1e6,
            g_buf_bridge_n / frames, g_buf_bridge_bytes / frames / 1e6,
            g_buf_bridge_declined / frames, g_vtx_bridge_n / frames,
            g_cs_chunk_splits / frames);
  g_buf_bridge_n = g_buf_bridge_bytes = g_buf_bridge_declined = 0;
  g_vtx_bridge_n = 0;
  g_out_retile_ns = g_out_rt_ns = g_out_tail_ns = 0;
  g_out_buffer_ns = 0;
  g_tex_bridge_n = g_tex_bridge_bytes = g_cs_chunk_splits = 0;
  g_out_retile_n = g_out_rt_submits = 0;
  base::String syncs;
  base::FormatTo(syncs, "{:.1f} syncs/frame ({:.1f} submits):", total / frames,
                 g_cs_submit_n / frames);
  g_cs_submit_n = 0;
  for (int i = 0; i < kSyncCount; i++) {
    if (!g_cs_sync_n[i])
      continue;
    base::FormatTo(syncs, " {}={:.1f}({:.1f}ms)", kCsSyncName[i],
                   g_cs_sync_n[i] / frames, g_cs_sync_ns[i] / frames / 1e6);
    g_cs_sync_n[i] = 0;
    g_cs_sync_ns[i] = 0;
  }
  base::FormatTo(syncs, " | by reader:");
  for (const auto& [reader, w] : g_cs_reader_waits)
    base::FormatTo(syncs, " {}={:.1f}({:.1f}ms)", reader.c_str(),
                   w.first / frames, w.second / frames / 1e6);
  g_cs_reader_waits.clear();
  BASE_LOGI("cssync", "{}", syncs.c_str());
}

bool PreserveCsDepthBeforeClear(u64 base) {
  auto depth_it = g_depths.find(base);
  auto range_it = g_cs_ranges.find(base);
  if (!g_frame.recording || depth_it == g_depths.end() ||
      range_it == g_cs_ranges.end())
    return false;

  DepthTarget& depth = depth_it->second;
  CsRange& range = range_it->second;
  if (!depth.texture || depth.layout == rhi::TextureState::kUndefined ||
      !range.buf || range.pending_batch || range.gpu_dirty ||
      !range.image_staging)
    return false;

  const rhi::TextureState old_state = depth.layout;
  CsAliasedImage image{depth.texture,     depth.w,   depth.h, 4,
                       rhi::kAspectDepth, old_state, true};
  AliasedCopyPlan plan;
  if (!PlanAliasedCopy(image, range.res, "preserves", plan) || plan.unpack ||
      plan.depth16 || range.res.layers > 1 || depth.layers > 1 || range.truth)
    return false;

  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(range.res, tiled, linear))
    return false;
  const auto& level = linear.mips[0];
  const u64 copy_bytes =
      static_cast<u64>(level.pitch) * plan.h * range.res.stage_elem_bytes;
  if (level.offset + copy_bytes > range.cap)
    return false;

  rhi::CommandList* const list = g_frame.list;
  AliasedImageBarrier(list, image, old_state, rhi::TextureState::kCopySrc);
  rhi::BufferTextureCopy copy;
  copy.buffer_offset = level.offset;
  copy.row_length = level.pitch;
  copy.region.aspect = rhi::kAspectDepth;
  copy.region.width = plan.w;
  copy.region.height = plan.h;
  list->CopyTextureToBuffer(range.buf, depth.texture, &copy, 1);
  list->Barrier(rhi::kAccessCopyWrite,
                rhi::kAccessComputeRead | rhi::kAccessHostRead);
  AliasedImageBarrier(list, image, rhi::TextureState::kCopySrc, old_state);

  // Compute for frame N runs before graphics N. This graphics-side snapshot
  // is therefore the input for compute N+1, and must suppress that frame's
  // usual copy from the now-cleared depth image.
  range.rt_sourced = true;
  range.last_rt_frame = static_cast<int>(g_frame.num) + 1;
  // The clear about to be bound takes the next serial: that one is not news.
  range.rt_serial = g_render_serial + 1;
  if (kCsRtTrace) {
    static int logged = 0;
    if (logged++ < 16)
      BASE_LOGI("csrt", "f{} preserved depth {:#x} before clear", g_frame.num,
                (unsigned long)base);
  }
  return true;
}

// DELTA_GPU_QCHECK=1: before every CS batch is submitted, run an EMPTY
// command list through the same queue and wait for it. The device is lost
// by whatever ran before the first failing submit, so a checkpoint that
// fails names PRIOR queue work (graphics draws, an earlier bridge copy) as
// the fault, and one that succeeds while the batch's own wait fails
// isolates the batch content. The checkpoint shares the graphics queue, so
// it runs after everything already recorded there.
DELTA_OPTION(bool, kQueueCheck, "DELTA_GPU_QCHECK", false);

bool QueueCheckArmed() {
  return kQueueCheck;
}

bool QueueCheck(const char* where) {
  if (!kQueueCheck)
    return true;
  rhi::CommandList* list = BeginImmediate();
  if (!list)
    return true;
  const bool ok = EndImmediate(list);
  if (!ok)
    BASE_LOGI("gpuvk", "queue CHECKPOINT FAILED at {}", where);
  return ok;
}

bool CsSupplyTexture(u64 base,
                     const gcn::TextureLayout32& layout,
                     u32 w,
                     u32 h,
                     rhi::Texture* img,
                     rhi::TextureState state,
                     u64* seq) {
  if (!kCsTexBridge || !kCsVram || g_cs_failed)
    return false;
  auto it = g_cs_ranges.find(base);
  if (it == g_cs_ranges.end())
    return false;
  CsRange& e = it->second;
  const ComputeInfo::Res& r = e.res;
  if (!e.buf || !e.gpu_dirty || !e.image_staging || e.imported)
    return false;
  auto declined = [&](const char* why) {
    if (kCsTexBridgeTrace) {
      static base::HashSet<u64> seen;
      if (seen.size() < 256 && seen.insert(base).second)
        BASE_LOGI("texbridge",
                  "{:#x} {}: cs {}x{} l={} m={} e={}/{} d={} t={} | tex "
                  "{}x{} l={} m={} e={} t={}",
                  (unsigned long)base, why, r.width, r.height, r.layers,
                  r.mip_levels, r.elem_bytes, r.stage_elem_bytes, r.dfmt,
                  r.tiling_idx, w, h, layout.layers, layout.mip_levels,
                  layout.elem_bytes, layout.tiling_idx);
    }
    return false;
  };
  // Only a raw-staged surface is byte for byte what the image samples; the
  // unpacked (11/11/10), widened (8/16-bit) and decoded (BCn) stagings are
  // not. A live target's address belongs to the render-target path.
  if (r.width != w || r.height != h)
    return declined("extent");
  // A texture may name fewer layers than the dispatch's view of the same
  // memory (a cube read as its six faces of an eight-slice array): checked
  // against the tiled layouts below.
  if (r.layers < layout.layers || r.mip_levels != layout.mip_levels)
    return declined("layers/mips");
  // A truth's narrow texels come out packed by the tiling shader. An 11/11/10
  // surface is staged unpacked to RGBA32F: it goes through a float image and
  // a blit, which packs it. Every other converted staging (BCn decoded, widened
  // 8/16-bit) has no image the copy could land in.
  const bool narrow = e.truth && r.elem_bytes < 4 && r.stage_elem_bytes == 4;
  const bool unpacked = r.dfmt == 6 && !e.truth && r.stage_elem_bytes == 16 &&
                        r.mip_levels == 1 && r.layers == 1;
  if ((r.dfmt == 6 && !unpacked) || r.dfmt == 35 || r.dfmt == 40 ||
      (!narrow && !unpacked && r.elem_bytes != r.stage_elem_bytes))
    return declined("converted staging");
  if (r.elem_bytes != layout.elem_bytes)
    return declined("texel size");
  if (unpacked &&
      ((img && !BlitsBetween(rhi::Format::kRGBA32Float, img->desc().format)) ||
       !GetBridgeFloatImage(w, h)))
    return declined("no blit");
  // A live target answers for its own geometry only; a view of the same
  // memory at another size (a 2432x1368 upscale over a 1080p target's pages)
  // is this range's.
  if (LiveTargetOfSize(base, w, h))
    return declined("live target");
  // Another dirty range inside the footprint (a mip generator binding one
  // level at its own address) holds bytes this buffer does not.
  if (!DirtyRangesOverlapping(base, e.guest_bytes, base).empty())
    return declined("dirty sibling");
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(r, tiled, linear) || linear.layer_stride)
    return declined("layout");
  // Fewer layers than the view: only when every level of the texture starts
  // where the view's does (layers sit inside each level on Liverpool), so the
  // texture's layers are the view's first ones.
  if (layout.layers != r.layers)
    for (u32 mip = 0; mip < layout.mip_levels; mip++)
      if (layout.mips[mip].offset != tiled.mips[mip].offset)
        return declined("layer subset");
  if (narrow &&
      !gcn::BuildTextureLayout32(linear, r.width, r.height, r.pitch, r.layers,
                                 r.mip_levels, 8, r.pow2_pad, r.elem_bytes))
    return declined("packed layout");
  const TileTable* table =
      e.truth ? GetTileTable(tiled, linear, narrow) : nullptr;
  if (e.truth && !table)
    return declined("tile table");
  rhi::BufferTextureCopy copies[16];
  for (u32 mip = 0; mip < linear.mip_levels; mip++) {
    const auto& level = linear.mips[mip];
    if (level.offset % base::Max<u64>(4, r.elem_bytes) ||
        level.width != base::Max(w >> mip, 1u) ||
        level.height != base::Max(h >> mip, 1u))
      return declined("mip geometry");
    copies[mip].buffer_offset = level.offset;
    copies[mip].row_length = level.pitch;
    copies[mip].image_height = level.stored_height;
    copies[mip].region.mip = mip;
    copies[mip].region.layers = layout.layers;
    copies[mip].region.width = level.width;
    copies[mip].region.height = level.height;
  }
  if (!img)
    return true;
  // A volume's layers are its depth slices.
  const bool volume = img->desc().dim == rhi::TextureDim::k3D;
  if (volume) {
    if (linear.mip_levels != 1)
      return false;
    copies[0].region.depth = copies[0].region.layers;
    copies[0].region.layers = 1;
  }
  e.last_used_frame = g_frame.num;
  if (*seq == e.write_seq)
    return true;
  EndRegion();
  rhi::CommandList* const list = g_frame.list;
  list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                rhi::kAccessCopyRead);
  rhi::Buffer* src = e.buf;
  if (e.truth) {
    g_cs_scratch_owner++;
    bool fresh = true;
    src = AcquireView(linear.size, base, 0, table, e.write_seq, &fresh);
    if (!src)
      return false;
    if (fresh) {
      RecordImageTiling(list, *table, e.buf, 0, e.guest_bytes, src, linear.size,
                        /*detile=*/true);
      StampView(src, base, 0, table, e.write_seq);
    }
  }
  rhi::TextureBarrier b;
  b.texture = img;
  b.before = state;
  b.after = rhi::TextureState::kCopyDst;
  b.range.mips = linear.mip_levels;
  b.range.layers = volume ? 1 : layout.layers;
  list->Barrier(0, 0, &b, 1);
  if (unpacked) {
    // RGBA32F texels into the float image, then a blit packs them into the
    // texture's 11/11/10.
    rhi::Texture* float_image = GetBridgeFloatImage(w, h);
    rhi::TextureBarrier fb;
    fb.texture = float_image;
    fb.before = rhi::TextureState::kUndefined;
    fb.after = rhi::TextureState::kCopyDst;
    list->Barrier(0, 0, &fb, 1);
    list->CopyBufferToTexture(float_image, src, copies, 1);
    fb.before = rhi::TextureState::kCopyDst;
    fb.after = rhi::TextureState::kCopySrc;
    list->Barrier(0, 0, &fb, 1);
    rhi::TextureRegion region;
    region.width = w;
    region.height = h;
    list->BlitTexture(img, region, float_image, region, rhi::Filter::kNearest);
  } else {
    list->CopyBufferToTexture(img, src, copies, linear.mip_levels);
  }
  b.before = rhi::TextureState::kCopyDst;
  b.after = rhi::TextureState::kShaderRead;
  list->Barrier(0, 0, &b, 1);
  *seq = e.write_seq;
  e.frame_ref = g_frame.num;
  e.chunk_ref = g_frame.chunk_seq;
  if (e.pending_batch)
    g_cs_chunk_needs_batch = true;
  g_tex_bridge_n++;
  g_tex_bridge_bytes += e.size;
  return true;
}

DELTA_OPTION(bool, kCsBufBridge, "DELTA_GPU_CS_BUF_BRIDGE", true);
DELTA_OPTION(bool, kCsVtxBridge, "DELTA_GPU_CS_VTX_BRIDGE", true);
DELTA_OPTION(bool, kBufMemo, "DELTA_GPU_BUFMEMO", true);

namespace {
// The one dirty raw range holding all of [base, base+bytes), or null.
CsRange* BufferSourceUncached(u64 base, u64 bytes, u64* range_base);

// Asked once per raw buffer per draw, and the answer only moves when compute
// state does, which bumps the texture epoch.
CsRange* BufferSource(u64 base, u64 bytes, u64* range_base) {
  if (!kCsBufBridge || !kCsVram || g_cs_failed || !bytes || !g_frame.recording)
    return nullptr;
  struct Memo {
    u64 base = 0, bytes = 0, epoch = 0, range_base = 0;
    CsRange* range = nullptr;
  };
  static Memo memo[512];
  Memo& m = memo[((base >> 4) ^ (base >> 16) ^ bytes) & 511];
  const u64 epoch = TextureEpoch();
  if (!kBufMemo || m.epoch != epoch || m.base != base || m.bytes != bytes) {
    u64 found_base = 0;
    CsRange* found = BufferSourceUncached(base, bytes, &found_base);
    m = {base, bytes, epoch, found_base, found};
  }
  *range_base = m.range_base;
  return m.range;
}

CsRange* BufferSourceUncached(u64 base, u64 bytes, u64* range_base) {
  u64 only = 0;
  const bool several = AnyDirtyOverlapping(base, bytes, [&](u64 other) {
    if (only && other != only)
      return true;
    only = other;
    return false;
  });
  if (kCsOverlapTrace && g_frame.num >= kCsOverlapTrace && only) {
    static int shown = 0;
    if (shown++ < 30) {
      base::String line;
      for (u64 r : DirtyRangesOverlapping(base, bytes, base)) {
        const CsRange& o = g_cs_ranges.find(r)->second;
        base::FormatTo(line, " [{:#x}+{:#x} size={:#x} truth={} img={}]", r,
                       o.guest_bytes, o.size, (int)o.truth,
                       (int)o.image_staging);
      }
      BASE_LOGI("csvtx", "window {:#x}+{:#x}:{}", base, bytes, line.c_str());
    }
  }
  if (several || !only)
    return nullptr;
  CsRange& e = g_cs_ranges.find(only)->second;
  // Raw bytes only: an image staging or a truth holds another layout.
  const u64 valid = base::Min<u64>(e.size, e.guest_bytes);
  if (!e.buf || e.image_staging || e.truth || e.imported || base < only ||
      base + bytes > only + valid)
    return nullptr;
  *range_base = only;
  return &e;
}
}  // namespace

u64 CsBufferRevision(u64 base, u64 bytes) {
  u64 range_base;
  const CsRange* e = BufferSource(base, bytes, &range_base);
  return e ? e->write_seq : 0;
}

bool CsSupplyBuffer(u64 base, u64 bytes, rhi::Buffer* dst, u64 dst_off) {
  u64 range_base;
  CsRange* const found = BufferSource(base, bytes, &range_base);
  if (!found || !dst)
    return false;
  CsRange& e = *found;
  // The buffer holds the dispatch's output plus, around it, the guest bytes as
  // they were staged in; the writeback would merge in whatever the CPU wrote
  // there since. Only while the guest still holds what was staged are the two
  // the same bytes.
  const u64 off = base - range_base;
  if (!e.shadow_valid || e.shadow.size() < off + bytes ||
      std::memcmp(e.shadow.data() + off, reinterpret_cast<const void*>(base),
                  bytes)) {
    g_buf_bridge_declined++;
    return false;
  }
  e.last_used_frame = g_frame.num;
  EndRegion();
  rhi::CommandList* const list = g_frame.list;
  list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                rhi::kAccessCopyRead);
  list->CopyBuffer(dst, dst_off, e.buf, off, bytes);
  list->Barrier(rhi::kAccessCopyWrite, kAccessAll);
  e.frame_ref = g_frame.num;
  e.chunk_ref = g_frame.chunk_seq;
  if (e.pending_batch)
    g_cs_chunk_needs_batch = true;
  g_buf_bridge_n++;
  g_buf_bridge_bytes += bytes;
  return true;
}

// The range holding a draw's vertex window. Unlike a raw window, neighbours
// may touch its ends: a mesh's streams are written through V#s whose extents
// run a few bytes into the next mesh's vertices without writing them.
CsRange* VertexSource(u64 base, u64 bytes, u64* range_base) {
  if (!kCsBufBridge || !kCsVram || g_cs_failed || !bytes || !g_frame.recording)
    return nullptr;
  constexpr u64 kSlack = 64;
  CsRange* found = nullptr;
  bool bad = false;
  AnyDirtyOverlapping(base, bytes, [&](u64 other) {
    CsRange& e = g_cs_ranges.find(other)->second;
    const u64 valid = base::Min<u64>(e.size, e.guest_bytes);
    if (other <= base && base + bytes <= other + valid) {
      if (found || !e.buf || e.image_staging || e.truth || e.imported)
        bad = true;
      found = &e;
      *range_base = other;
    } else {
      const u64 lo = base::Max(base, other);
      const u64 hi = base::Min(base + bytes, other + e.guest_bytes);
      bad |= hi > lo && hi - lo > kSlack;
    }
    return bad;
  });
  return bad ? nullptr : found;
}

bool CsHoldsVertices(u64 base, u64 bytes) {
  u64 range_base;
  return kCsVtxBridge && VertexSource(base, bytes, &range_base);
}

bool CsVertexBuffer(u64 base, u64 bytes, rhi::Buffer** buffer, u64* offset) {
  if (!kCsVtxBridge)
    return false;
  u64 range_base;
  CsRange* const found = VertexSource(base, bytes, &range_base);
  if (!found)
    return false;
  CsRange& e = *found;
  // As for CsSupplyBuffer: only while the guest still holds what was staged
  // do the buffer's bytes around the output equal guest memory's.
  const u64 off = base - range_base;
  if (!e.shadow_valid || e.shadow.size() < off + bytes ||
      std::memcmp(e.shadow.data() + off, reinterpret_cast<const void*>(base),
                  bytes)) {
    g_buf_bridge_declined++;
    return false;
  }
  // One barrier outside the pass covers every batch recorded so far, so a pass
  // of skinned draws ends once, not per draw. A later dispatch into a range
  // this chunk reads splits the chunk (chunk_ref), so the draw still sees the
  // output it was recorded against.
  static u64 covered_chunk = ~0ull, covered_batch = 0;
  if (covered_chunk != g_frame.chunk_seq) {
    covered_chunk = g_frame.chunk_seq;
    covered_batch = 0;
  }
  if (e.batch_id > covered_batch) {
    EndRegion();
    g_frame.list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                          kAccessAll);
    covered_batch = g_cs_batch_open ? g_cs_batch_id : g_cs_batch_next_id - 1;
  }
  e.last_used_frame = g_frame.num;
  e.frame_ref = g_frame.num;
  e.chunk_ref = g_frame.chunk_seq;
  if (e.pending_batch)
    g_cs_chunk_needs_batch = true;
  g_vtx_bridge_n++;
  *buffer = e.buf;
  *offset = off;
  return true;
}

// The live colour target at `base` takes the truth's pixels on the frame
// command list: detile into a scratch, copy into the image, and leave the
// image in the layout the recording expects. No writeback, no wait.
bool CsRefreshRtInFrame(u64 base, CsRange& e) {
  if ((!e.truth && !e.image_staging) || !e.gpu_dirty || !e.buf ||
      e.rt_seq == e.write_seq || !g_frame.recording)
    return false;
  const ComputeInfo::Res& r = e.res;
  // A linear staging holds the texels as they are, except 11/11/10, which is
  // staged unpacked to RGBA32F and packed again by a blit.
  const bool unpacked = !e.truth && r.dfmt == 6 && r.stage_elem_bytes == 16;
  if (r.mip_levels != 1 || r.layers != 1 ||
      (!unpacked && r.elem_bytes != r.stage_elem_bytes))
    return false;
  ActivateWrittenRtVariant(base, r.width, r.height);
  auto rt_it = g_rts.find(base);
  if (rt_it == g_rts.end() || !rt_it->second.texture || rt_it->second.is_depth)
    return false;
  RTarget& rt = rt_it->second;
  if (rt.w + 8 < r.width || rt.h + 8 < r.height ||
      FormatBytes(rt.fmt) != r.elem_bytes)
    return false;
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(r, tiled, linear))
    return false;
  const u32 w = base::Min(rt.w, r.width), h = base::Min(rt.h, r.height);
  rhi::Texture* float_image = nullptr;
  if (unpacked && (!BlitsBetween(rhi::Format::kRGBA32Float, rt.fmt) ||
                   !(float_image = GetBridgeFloatImage(w, h))))
    return false;
  rhi::Buffer* src = e.buf;
  bool fresh = false;
  const TileTable* table = nullptr;
  if (e.truth) {
    if (!(table = GetTileTable(tiled, linear)))
      return false;
    g_cs_scratch_owner++;
    fresh = true;
    if (!(src = AcquireView(linear.size, base, 0, table, e.write_seq, &fresh)))
      return false;
  }
  EndRegion();
  rhi::CommandList* const list = g_frame.list;
  list->Barrier(rhi::kAccessComputeWrite | rhi::kAccessCopyWrite,
                rhi::kAccessCopyRead);
  // A partial conversion (mip 0, slice 0) is not a whole view: stamp nothing.
  if (fresh)
    RecordImageTiling(list, *table, e.buf, 0, e.guest_bytes, src, linear.size,
                      /*detile=*/true, 1, 1);
  const rhi::TextureState from = rt.layout;
  TransitionImage(list, rt.texture, rt.layout, rhi::TextureState::kCopyDst);
  const auto& level = linear.mips[0];
  rhi::BufferTextureCopy copy;
  copy.buffer_offset = level.offset;
  copy.row_length = level.pitch;
  copy.region.width = w;
  copy.region.height = h;
  if (unpacked) {
    rhi::TextureBarrier fb;
    fb.texture = float_image;
    fb.before = rhi::TextureState::kUndefined;
    fb.after = rhi::TextureState::kCopyDst;
    list->Barrier(0, 0, &fb, 1);
    list->CopyBufferToTexture(float_image, src, &copy, 1);
    fb.before = rhi::TextureState::kCopyDst;
    fb.after = rhi::TextureState::kCopySrc;
    list->Barrier(0, 0, &fb, 1);
    rhi::TextureRegion region;
    region.width = w;
    region.height = h;
    list->BlitTexture(rt.texture, region, float_image, region,
                      rhi::Filter::kNearest);
  } else {
    list->CopyBufferToTexture(rt.texture, src, &copy, 1);
  }
  // Back to the layout the recording had it in; a target nothing had laid
  // out yet stays a copy destination.
  if (from != rhi::TextureState::kUndefined)
    TransitionImage(list, rt.texture, rt.layout, from);
  rt.dirty_for_read = true;
  rt.ever_rendered = true;
  rt.last_frame = g_frame.num;
  e.rt_seq = e.write_seq;
  e.frame_ref = g_frame.num;
  e.chunk_ref = g_frame.chunk_seq;
  e.last_used_frame = g_frame.num;
  if (e.pending_batch)
    g_cs_chunk_needs_batch = true;
  g_truth_rt_n++;
  return true;
}

bool CsRefreshRtFromTruth(u64 base) {
  auto it = g_cs_ranges.find(base);
  if (it == g_cs_ranges.end())
    return false;
  CsRange& e = it->second;
  if ((!e.truth && !e.image_staging) || !e.gpu_dirty)
    return false;
  // Already holds this revision: nothing to flush either.
  return e.rt_seq == e.write_seq || CsRefreshRtInFrame(base, e);
}

void ReleaseRetiredCsBuffers() {
  static base::Vector<rhi::Buffer*> aged;
  for (rhi::Buffer* buf : aged)
    Device().Destroy(buf);
  aged = base::move(g_cs_frame_retired);
  g_cs_frame_retired.clear();
}

void CsForgetGuestRange(u64 base, u64 bytes) {
  BumpTextureEpoch();
  if (!bytes)
    return;
  const u64 end = RangeEnd(base, bytes);
  base::Vector<u64> bases;
  for (u64 b = base >> kCsRangeBlockShift; b <= (end - 1) >> kCsRangeBlockShift;
       b++)
    if (auto found = g_cs_range_blocks.find(b);
        found != g_cs_range_blocks.end())
      bases.insert(bases.end(), found->second.begin(), found->second.end());
  for (u64 range_base : bases) {
    auto found = g_cs_ranges.find(range_base);
    if (found == g_cs_ranges.end())
      continue;
    CsRange& e = found->second;
    if (range_base >= end || base >= range_base + e.guest_bytes || e.gpu_dirty)
      continue;
    e.last_validated_frame = -1;
    e.hash = 0;
    e.mirror_current = false;
    e.rt_sourced = false;
  }
}

bool CsTracksGuestWrites() {
  return kCsTrack && GuestWriteTracker().enabled();
}

void CsNoteGuestWrites(u64 first, u64 end) {
  // Untracked, the tracker also reports our own writebacks wherever a draw
  // cache armed the pages, and acting on those costs more than it saves.
  if (!CsTracksGuestWrites() || g_cs_range_blocks.empty() || first >= end)
    return;
  thread_local base::Vector<u64> bases;
  bases.clear();
  // A remapped reservation can span gigabytes: walk the smaller side.
  const u64 first_block = first >> kCsRangeBlockShift;
  const u64 last_block = (end - 1) >> kCsRangeBlockShift;
  if (last_block - first_block + 1 > g_cs_range_blocks.size()) {
    for (const auto& [block, list] : g_cs_range_blocks)
      if (block >= first_block && block <= last_block)
        bases.insert(bases.end(), list.begin(), list.end());
  } else {
    for (u64 b = first_block; b <= last_block; b++)
      if (auto found = g_cs_range_blocks.find(b);
          found != g_cs_range_blocks.end())
        bases.insert(bases.end(), found->second.begin(), found->second.end());
  }
  for (u64 range_base : bases) {
    auto found = g_cs_ranges.find(range_base);
    if (found == g_cs_ranges.end())
      continue;
    CsRange& e = found->second;
    const u64 range_end = RangeEnd(range_base, e.guest_bytes);
    if (range_base >= end || first >= range_end)
      continue;
    if (!e.gpu_dirty) {
      // Hash it again at the next use rather than trusting this frame's
      // check: the write may be our own writeback, which leaves it current.
      e.last_validated_frame = -1;
      continue;
    }
    const u64 lo = base::Max(first, range_base), hi = base::Min(end, range_end);
    if (!e.cpu_writes.empty() && e.cpu_writes.back().second >= lo &&
        e.cpu_writes.back().first <= hi) {
      auto& last = e.cpu_writes.back();
      last = {base::Min(last.first, lo), base::Max(last.second, hi)};
    } else {
      e.cpu_writes.emplace_back(lo, hi);
    }
  }
}

}  // namespace gpu::render

namespace gpu::render {
extern u64 g_tex_image_bytes;
}

namespace gpu::render {
// Declared in render/renderer.h for the kernel's crash handler.
bool DescribeCsRangeCovering(u64 addr, char* out, size_t out_size) {
  for (const auto& kv : g_cs_ranges) {
    const u64 base = kv.first;
    const CsRange& e = kv.second;
    const u64 n = e.guest_bytes ? e.guest_bytes : e.size;
    if (!n || addr < base || addr >= base + n)
      continue;
    std::snprintf(out, out_size,
                  "base=%#llx +%#llx (addr is base+%#llx) staged=%#llx "
                  "gpu_dirty=%d imported=%d rt_sourced=%d image=%d "
                  "shadow=%d last_used_frame=%d",
                  (unsigned long long)base, (unsigned long long)n,
                  (unsigned long long)(addr - base), (unsigned long long)e.size,
                  (int)e.gpu_dirty, (int)e.imported, (int)e.rt_sourced,
                  (int)e.image_staging, (int)e.shadow_valid, e.last_used_frame);
    return true;
  }
  return false;
}
}  // namespace gpu::render

namespace gpu::render {

static bool PrepareGdsTransfer(Renderer& renderer,
                               u32 offset,
                               u32 bytes,
                               bool read = false) {
  if (!renderer.available() || offset > GdsBuffer::kBytes ||
      bytes > GdsBuffer::kBytes - offset || !EnsureGdsBuffer())
    return false;
  if (read) {
    CsBatchBeginImpl();
    g_cs_list->Barrier(rhi::kAccessComputeWrite, rhi::kAccessHostRead);
  }
  return CsBatchFlush(kSyncWriteback);
}

bool ReadGds(Renderer& renderer, u32 offset, void* data, u32 bytes) {
  if (!PrepareGdsTransfer(renderer, offset, bytes, true))
    return false;
  std::memcpy(data, static_cast<const u8*>(g_gds.map) + offset, bytes);
  return true;
}

bool WriteGds(Renderer& renderer, u32 offset, const void* data, u32 bytes) {
  if (!PrepareGdsTransfer(renderer, offset, bytes))
    return false;
  std::memcpy(static_cast<u8*>(g_gds.map) + offset, data, bytes);
  return true;
}

bool FillGds(Renderer& renderer, u32 offset, u32 bytes, u32 value) {
  if (!PrepareGdsTransfer(renderer, offset, bytes))
    return false;
  auto* dst = static_cast<u8*>(g_gds.map) + offset;
  for (u32 i = 0; i < bytes; i++)
    dst[i] = static_cast<u8>(value >> ((i & 3u) * 8));
  return true;
}

// A dispatch the backend could not run, named once per shader and reason.
// The reason is the number of the `return false` it came from; the file's
// line is one grep away and a prose reason at each of them would be noise.
bool CsDeclined(const ComputeInfo& ci, const char* why) {
  static base::HashSet<base::String> reported;
  base::String key = base::ToString(ci.cs_addr) + why;
  if (reported.size() < 512 && reported.insert(key).second)
    BASE_LOGI("csgpu", "CS @{:#x} declined at exit {}", ci.cs_addr, why);
  return false;
}

// Two plain-buffer bindings on one guest base with different extents (a
// constant block read through a 0x1c-byte window and a 0x24-byte one) share
// one staging range, so they have to agree on its size: take the larger for
// both. Image bindings keep their own shapes and still conflict below.
// Plain bindings of one dispatch that overlap without sharing a base (a
// skinning job writing two interleaved vertex streams through V#s 8 bytes
// apart) are one piece of memory: stage them as one range covering both, each
// bound at its own offset into it. As two ranges, each held the other's words
// stale, so neither could be read back or fed to a draw without the other
// flushed.
bool MergeOverlapping(ComputeInfo& ci) {
  constexpr u64 kAlign = 16;
  const auto plain = [&](u32 i) {
    const auto& r = ci.res[i];
    return !r.zero_fill && !r.image_staging && r.binding < 48 &&
           !TruthEligible(r);
  };
  const auto end_of = [](const ComputeInfo::Res& r) {
    return r.base + base::Max(r.size, r.guest_size);
  };
  bool changed = false;
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!plain(i))
      continue;
    u64 lo = ci.res[i].base, hi = end_of(ci.res[i]);
    u32 members = 0;
    // Grow the group until no other plain binding overlaps it.
    for (bool grew = true; grew;) {
      grew = false;
      members = 0;
      for (u32 j = 0; j < ci.num_res; j++) {
        if (!plain(j) || ci.res[j].base >= hi || end_of(ci.res[j]) <= lo)
          continue;
        members++;
        if (ci.res[j].base < lo || end_of(ci.res[j]) > hi) {
          lo = base::Min(lo, ci.res[j].base);
          hi = base::Max(hi, end_of(ci.res[j]));
          grew = true;
        }
      }
    }
    bool distinct = false;
    for (u32 j = 0; j < ci.num_res; j++)
      distinct |= plain(j) && ci.res[j].base < hi && end_of(ci.res[j]) > lo &&
                  ci.res[j].base != ci.res[i].base;
    if (members < 2 || !distinct)
      continue;
    lo &= ~(kAlign - 1);
    hi = (hi + 3) & ~u64(3);
    for (u32 j = 0; j < ci.num_res; j++) {
      auto& r = ci.res[j];
      if (!plain(j) || r.base < lo || end_of(r) > hi)
        continue;
      r.view_offset = r.base - lo;
      r.base = lo;
      r.size = r.guest_size = hi - lo;
    }
    changed = true;
  }
  return changed;
}

const ComputeInfo& MergeSameBase(const ComputeInfo& ci, ComputeInfo& merged) {
  const ComputeInfo* src = &ci;
  if (kCsMergeOverlap) {
    bool overlap = false;
    for (u32 i = 0; i < ci.num_res && !overlap; i++)
      for (u32 j = 0; j < i && !overlap; j++) {
        const auto& a = ci.res[i];
        const auto& b = ci.res[j];
        overlap = !a.zero_fill && !b.zero_fill && !a.image_staging &&
                  !b.image_staging && a.base != b.base &&
                  a.base < b.base + base::Max(b.size, b.guest_size) &&
                  b.base < a.base + base::Max(a.size, a.guest_size);
      }
    if (overlap) {
      merged = ci;
      if (MergeOverlapping(merged))
        src = &merged;
    }
  }
  const ComputeInfo& in = *src;
  bool need = false;
  for (u32 i = 0; i < in.num_res && !need; i++)
    for (u32 j = 0; j < i && !need; j++)
      need = in.res[i].base == in.res[j].base && !in.res[i].zero_fill &&
             !in.res[j].zero_fill && !in.res[i].image_staging &&
             !in.res[j].image_staging &&
             (in.res[i].size != in.res[j].size ||
              in.res[i].guest_size != in.res[j].guest_size);
  if (!need)
    return in;
  if (src != &merged)
    merged = ci;
  for (u32 i = 0; i < merged.num_res; i++)
    for (u32 j = 0; j < i; j++) {
      auto& a = merged.res[i];
      auto& b = merged.res[j];
      if (a.base != b.base || a.zero_fill || b.zero_fill || a.image_staging ||
          b.image_staging)
        continue;
      a.size = b.size = base::Max(a.size, b.size);
      a.guest_size = b.guest_size = base::Max(a.guest_size, b.guest_size);
    }
  return merged;
}

void PrebuildComputePipeline(const base::Vector<u32>& spirv,
                             u32 num_res,
                             int guest_memory_binding) {
  if (!g_backend.device || spirv.empty())
    return;
  const auto slot = base::MakeShared<PrebuiltSlot>();
  {
    base::LockGuard<base::Mutex> lock(g_prebuilt_mutex);
    const u64 key = PrebuiltKey(spirv, num_res, guest_memory_binding);
    if (g_prebuilt.count(key))
      return;
    g_prebuilt.emplace(key, slot);
  }
  PrebuiltCs built;
  built.set_layout = CsSetLayout(num_res, -1, guest_memory_binding);
  if (built.set_layout)
    built.layout = CsPipelineLayout(built.set_layout);
  if (built.layout)
    built.pipe = CreateCsPipeline(built.layout, spirv);
  {
    base::LockGuard<base::Mutex> lock(slot->mutex);
    slot->value = built;
    slot->ready = true;
  }
  slot->built.NotifyAll();
}

bool Dispatch(Renderer& renderer, const ComputeInfo& ci_in) {
  DELTA_ZONE("gpu.cs_dispatch");
  BumpTextureEpoch();
  ComputeInfo ci_merged;
  const ComputeInfo& ci = MergeSameBase(ci_in, ci_merged);
  if (g_cs_failed) {
    renderer.state = nullptr;
    return CsDeclined(ci, "1");
  }
  if (!renderer.available() || !ci.recomp || !ci.recomp->ok || !ci.num_res)
    return CsDeclined(ci, "2");
  // Only a range still holding dispatch output needs the guest's latest
  // writes before this runs; the rest revalidate against guest memory.
  if (CsTracksGuestWrites()) {
    for (u32 i = 0; i < ci.num_res; i++) {
      const auto found = g_cs_ranges.find(ci.res[i].base);
      if (found != g_cs_ranges.end() && found->second.gpu_dirty) {
        CollectGuestWritesForSubmission();
        break;
      }
    }
  }
  const rhi::Caps& caps = Device().caps();
  const u32 max_resources =
      base::Min(gcn::kMaxCsResources, caps.max_compute_resources);
  if (ci.num_res + (ci.recomp->guest_memory_binding >= 0) +
          (ci.gds_binding >= 0) >
      max_resources)
    return CsDeclined(ci, "2");
  rhi::Buffer* guest_table = nullptr;
  u64 guest_table_bytes = 0;
  if (ci.recomp->guest_memory_binding >= 0) {
    // Runtime reads follow guest pointers across resources. Retire writes and
    // publish their guest mirrors before exposing those pages to this shader.
    // A read-only batch may have no dirty staged range for FlushCsWrites to
    // visit. Retire it explicitly before replacing imported memory or the map.
    // Only the pages this shader reaches need their writebacks; the wait is
    // for the previous pointer-chasing dispatch, whose imported memory
    // PrepareGuestMemory may replace.
    if (!CsBatchFlush(kSyncBatchCap))
      return CsDeclined(ci, "guest-address-map");
    for (const auto& r : ci.guest_memory)
      if (!FlushCsWritesRange(renderer, r.base, r.size, "guestmem"))
        return CsDeclined(ci, "guest-address-map");
    if (!PrepareGuestMemory(ci.guest_memory, guest_table, guest_table_bytes,
                            ci.recomp->guest_memory_written))
      return CsDeclined(ci, "guest-address-map");
  }
  ScopeCs cs;
  for (u32 i = 0; i < ci.num_res; i++)
    g_cs_bytes += ci.res[i].size;
  for (u32 i = 0; i < ci.num_res; i++)
    if (ci.res[i].size > caps.max_storage_buffer_range)
      return CsDeclined(ci, "3");
  for (u32 i = 0; i < ci.num_res; i++) {
    if (ci.res[i].zero_fill)
      continue;
    const u64 guest_bytes =
        ci.res[i].guest_size ? ci.res[i].guest_size : ci.res[i].size;
    const u64 size = ci.res[i].size ? ((ci.res[i].size + 3) & ~u64(3)) : 4;
    for (u32 j = 0; j < i; j++) {
      if (ci.res[j].zero_fill || ci.res[j].base != ci.res[i].base)
        continue;
      const u64 other_guest_bytes =
          ci.res[j].guest_size ? ci.res[j].guest_size : ci.res[j].size;
      const u64 other_size =
          ci.res[j].size ? ((ci.res[j].size + 3) & ~u64(3)) : 4;
      if (size != other_size || guest_bytes != other_guest_bytes ||
          !SameCsResourceShape(ci.res[i], ci.res[j]))
        return CsDeclined(ci, "4");
    }
  }
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!ci.res[i].written || ci.res[i].zero_fill)
      continue;
    const u64 bytes =
        ci.res[i].guest_size ? ci.res[i].guest_size : ci.res[i].size;
    NoteDccWrite(ci.res[i].base, bytes, nullptr);
    NoteSurfaceWrite(ci.res[i].base, bytes);
    if (!ci.res[i].image_staging)
      NoteRawWrite(ci.res[i].base, bytes);
  }
  CsPipe* cp = GetCsPipe(ci);
  if (!cp)
    return CsDeclined(ci, "5");

  if (!g_cs_batches[0].list) {
    if (!CsBatchInit())
      return CsDeclined(ci, "8");
    // What an EMPTY submit+wait costs here. SotC spends over half its frame in
    // waits while the GPU sits at 18% utilisation, so the question is
    // whether a round trip is inherently expensive on this system or whether
    // the GPU is really doing that work. Measured once, at init.
    if (kCsSyncReport) {
      double total = 0;
      rhi::CommandList* list = g_cs_batches[0].list;
      for (int i = 0; i < 100; i++) {
        list->Begin();
        list->End();
        const u64 t0 = NowNs();
        Device().Wait(Device().Submit(list));
        total += (NowNs() - t0) / 1e6;
      }
      BASE_LOGI("csbench", "empty submit+wait: {:.3f} ms", total / 100.0);
    }
  }

  // DELTA_GPU_CSLIST: per-dispatch resource staging list for the first 200
  // dispatches: shows what the chain actually round-trips per frame.
  // DELTA_GPU_CSHIST=<seconds>: every distinct guest range a dispatch WRITES,
  // for the whole run. The capped per-dispatch list above only ever showed the
  // first few frames, which cannot answer "does the title ever compute-copy
  // into its texture heap".
  if (kCsHist) {
    struct Cell {
      u64 size, n;
    };
    static base::Mutex m;
    static base::Map<u64, Cell> tbl;
    static auto last = base::TimeTicks::Now();
    base::LockGuard<base::Mutex> lk(m);
    for (u32 i = 0; i < ci.num_res; i++)
      if (ci.res[i].written && tbl.size() < 4096) {
        Cell& c = tbl[ci.res[i].base];
        c = {ci.res[i].size, c.n + 1};
      }
    const auto now = base::TimeTicks::Now();
    if ((now - last).InSeconds() >= kCsHist) {
      last = now;
      BASE_LOGI("cshist", "{} written ranges", tbl.size());
      for (const auto& kv : tbl)
        BASE_LOGI("cshist", "{:#x} size={:#x} x{}", (unsigned long)kv.first,
                  (unsigned long)kv.second.size, (unsigned long)kv.second.n);
      std::fflush(stderr);
    }
  }
  static u32 cs_listed = 0;
  if (kCsList && g_frame.num > 25 && cs_listed < 200) {
    cs_listed++;
    for (u32 i = 0; i < ci.num_res; i++)
      BASE_LOGI("cslist", "cs={:#x} bind={} base={:#x} size={:#x} {}{}",
                (unsigned long long)ci.cs_addr, ci.res[i].binding,
                (unsigned long)ci.res[i].base, (unsigned long)ci.res[i].size,
                ci.res[i].image_staging ? "img"
                : ci.res[i].zero_fill   ? "zero"
                                        : "buf",
                ci.res[i].written ? " written" : "");
  }

  // Bind each resource: zero-fill scratch per binding slot; everything else
  // uses the persistent range buffer for its guest base, staged only when the
  // buffer doesn't already hold current content.
  const u64 t_in0 = NowNs();
  rhi::Buffer* bind_buf[ComputeInfo::kMaxResources];
  u64 sz[ComputeInfo::kMaxResources];
  rhi::Buffer* truth_view[ComputeInfo::kMaxResources] = {};
  const TileTable* truth_table[ComputeInfo::kMaxResources] = {};
  // A binding served from a dirty truth that contains it: that range's base,
  // and the binding's byte offset into it.
  u64 parent_base[ComputeInfo::kMaxResources] = {};
  u64 parent_off[ComputeInfo::kMaxResources] = {};
  g_cs_scratch_owner++;
  // The shader's view of a truth: the bytes at `off` in `tiled_buf` detiled
  // into a scratch on the batch. Bindings of one dispatch that describe the
  // same surface the same way share the view (a shadow array bound through
  // several descriptors is one 320 MB conversion, not four).
  auto bind_truth_view = [&](u32 i, rhi::Buffer* tiled_buf, u64 off, u64 span,
                             u64 pbase) -> bool {
    gcn::TextureLayout32 tiled, linear;
    const TileTable* table = BuildCsImageLayouts(ci.res[i], tiled, linear)
                                 ? GetTileTable(tiled, linear)
                                 : nullptr;
    if (!table)
      return false;
    for (u32 j = 0; j < i; j++) {
      if (truth_view[j] && truth_table[j] == table &&
          ci.res[j].base == ci.res[i].base && parent_base[j] == pbase &&
          parent_off[j] == off && sz[j] == sz[i]) {
        truth_view[i] = truth_view[j];
        truth_table[i] = table;
        parent_base[i] = pbase;
        parent_off[i] = off;
        bind_buf[i] = truth_view[j];
        return true;
      }
    }
    auto owner_it = g_cs_ranges.find(pbase ? pbase : ci.res[i].base);
    const u64 seq =
        owner_it != g_cs_ranges.end() ? owner_it->second.write_seq : 0;
    bool fresh = true;
    rhi::Buffer* const view = AcquireView(sz[i], pbase ? pbase : ci.res[i].base,
                                          off, table, seq, &fresh);
    if (!view)
      return false;
    if (fresh) {
      CsBatchBeginImpl();
      RecordImageTiling(g_cs_list, *table, tiled_buf, off, span, view, sz[i],
                        /*detile=*/true);
      g_cs_batch_access.erase(view);
      g_truth_detile_n++;
      StampView(view, pbase ? pbase : ci.res[i].base, off, table, seq);
    }
    truth_view[i] = view;
    truth_table[i] = table;
    parent_base[i] = pbase;
    parent_off[i] = off;
    bind_buf[i] = view;
    return true;
  };
  for (u32 i = 0; i < ci.num_res; i++) {
    sz[i] = ci.res[i].size ? ((ci.res[i].size + 3) & ~u64(3)) : 4;
    if (ci.res[i].zero_fill) {
      if (!CsEnsureStage(i, sz[i]))
        return CsDeclined(ci, "10");
      bind_buf[i] = g_cs_stage[i].buf;
      continue;
    }
    const u64 base = ci.res[i].base;
    const u64 guest_bytes =
        ci.res[i].guest_size ? ci.res[i].guest_size : ci.res[i].size;
    const bool truth_i = TruthEligible(ci.res[i]);
    u64 folds[ComputeInfo::kMaxResources];
    u32 fold_count = 0;
    // A binding inside a dirty truth binds that truth's own bytes at an
    // offset: a surface's layers or mips bound one at a time are its bytes,
    // not a copy that has to be written back and restaged around it.
    if (kCsTruth && (truth_i || !ci.res[i].image_staging)) {
      CsRange* parent = nullptr;
      u64 pbase = 0;
      for (u64 cand : DirtyRangesOverlapping(base, guest_bytes, base)) {
        auto f = g_cs_ranges.find(cand);
        if (f == g_cs_ranges.end())
          continue;
        CsRange& r = f->second;
        const bool offset_ok =
            (base - cand) % caps.storage_offset_alignment == 0 ||
            ((base - cand) % 4 == 0 && ci.res[i].binding < 48 && !truth_i);
        if ((!r.truth && r.image_staging) || !r.gpu_dirty || !r.buf ||
            cand > base || base + guest_bytes > cand + r.guest_bytes ||
            !offset_ok)
          continue;
        parent = &r;
        pbase = cand;
        break;
      }
      auto own = g_cs_ranges.find(base);
      if (parent && own != g_cs_ranges.end() && own->second.gpu_dirty) {
        // Its own dirty bytes fold into the parent first. A linear-staged
        // image cannot: those bytes are not guest layout.
        CsRange& s = own->second;
        if (s.image_staging && !s.truth) {
          parent = nullptr;
        } else if (s.buf) {
          const u64 n = base::Min<u64>(s.size ? s.size : s.cap, guest_bytes);
          CsBatchBeginImpl();
          g_cs_list->Barrier(kAccessComputeRW | kAccessCopyRW, kAccessCopyRW);
          g_cs_list->CopyBuffer(parent->buf, base - pbase, s.buf, 0, n);
          g_cs_list->Barrier(rhi::kAccessCopyWrite,
                             kAccessComputeRW | kAccessCopyRW);
          g_cs_batch_access.erase(parent->buf);
          UnindexDirtyRange(base, s.guest_bytes);
          s.gpu_dirty = false;
          s.last_validated_frame = -1;
          s.hash = 0;
          s.mirror_current = false;
          MarkPending(s);
          parent->write_seq++;
          BumpTextureEpoch();
        }
      }
      if (parent) {
        const u64 off = base - pbase;
        if (parent->chunk_ref == g_frame.chunk_seq &&
            (ci.res[i].written || ci.res[i].shader_writes) &&
            !CsSplitFrameChunk()) {
          renderer.state = nullptr;
          return CsDeclined(ci, "22");
        }
        parent->last_used_frame = g_frame.num;
        MarkPending(*parent);
        parent_base[i] = pbase;
        if (truth_i) {
          if (!bind_truth_view(i, parent->buf, off, guest_bytes, pbase))
            return CsDeclined(ci, "23");
        } else {
          parent_off[i] = off;
          bind_buf[i] = parent->buf;
        }
        g_truth_parent_n++;
        continue;
      }
    }
    // A read overlapping some OTHER dirty range must see that data through
    // guest memory: flush those first.
    // A range whose writeback cannot be produced (an image the retiler does not
    // handle) stays stale whatever happens next; declining THIS dispatch over
    // it only loses a second result. Only a dead device stops the recording. A
    // range that is itself dirty keeps a VRAM copy without the sibling's bytes
    // after this, and one that is clean relies on the sampled hash to notice
    // them. Both are the pre-existing behaviour; the fix is one VRAM copy per
    // guest byte (see des-perf-profile).
    {
      const u64 to = NowNs();
      const auto siblings = DirtyRangesOverlapping(base, guest_bytes, base);
      for (u64 dirty : siblings) {
        auto found = g_cs_ranges.find(dirty);
        if (found == g_cs_ranges.end())
          continue;
        CsRange& s = found->second;
        // A dirty child inside this binding folds into its buffer once it is
        // staged, on the batch, instead of a trip through guest memory.
        if (kCsTruth && (truth_i || !ci.res[i].image_staging) && s.gpu_dirty &&
            s.buf && (s.truth || !s.image_staging) && dirty >= base &&
            dirty + s.guest_bytes <= base + guest_bytes &&
            fold_count < ComputeInfo::kMaxResources) {
          folds[fold_count++] = dirty;
          continue;
        }
        // A plain output this dispatch only writes needs none of the
        // sibling's bytes: each writeback publishes just the words its own
        // dispatch changed. A title's skinning jobs write interleaved vertex
        // streams whose windows overlap their neighbours by a few bytes, and
        // flushing each neighbour first cost UC2 ~440 GPU waits a frame.
        if (kCsKeepOverlap && s.gpu_dirty && !ci.res[i].read &&
            !ci.res[i].image_staging &&
            !truth_i && !s.image_staging && !s.truth) {
          g_wb_why["overlap-kept"]++;
          continue;
        }
        if (s.gpu_dirty) {
          g_wb_why["overlap"]++;
          g_wb_why_bytes += s.size;
          if (kCsOverlapTrace && g_frame.num >= kCsOverlapTrace) {
            static int shown = 0;
            if (shown++ < 1500)
              BASE_LOGI("csoverlap",
                        "cs={:#x} res{} {:#x}+{:#x} {} w={} {}x{}x{} mips={} "
                        "tile={} dfmt={} | dirty {:#x}+{:#x} truth={} img={} "
                        "{}x{}x{} mips={} tile={} dfmt={}",
                        ci.cs_addr, i, base, guest_bytes,
                        ci.res[i].image_staging ? "img" : "buf",
                        (int)ci.res[i].shader_writes, ci.res[i].width,
                        ci.res[i].height, ci.res[i].layers,
                        ci.res[i].mip_levels, ci.res[i].tiling_idx,
                        ci.res[i].dfmt, dirty, s.guest_bytes, (int)s.truth,
                        (int)s.image_staging, s.res.width, s.res.height,
                        s.res.layers, s.res.mip_levels, s.res.tiling_idx,
                        s.res.dfmt);
          }
        }
        if (!CsRangeFlushOne(dirty, s) && g_cs_failed)
          return CsDeclined(ci, "11");
      }
      g_in_overlap_n += !siblings.empty();
      g_in_overlap_ns += NowNs() - to;
    }
    CsRange& e = g_cs_ranges[base];
    // The guest CPU wrote the range while it held dispatch output: merge the
    // two before running on it again, or this dispatch reads around the
    // CPU's bytes and a later writeback puts old values over them.
    if (e.gpu_dirty && !e.cpu_writes.empty()) {
      g_wb_why["cpu-write"]++;
      g_wb_why_bytes += e.size;
      if (!CsRangeFlushOne(base, e) && g_cs_failed)
        return CsDeclined(ci, "24");
    }
    // A truth range is its guest footprint: the same footprint is the same
    // bytes whatever the descriptor makes of them, and a plain-buffer view
    // inside it binds those bytes directly.
    const bool truth_same =
        truth_i && e.buf && e.truth && e.guest_bytes == guest_bytes;
    const bool truth_sub =
        truth_i && e.buf && e.truth && guest_bytes < e.guest_bytes;
    const bool raw_view = !truth_i && !ci.res[i].image_staging && e.buf &&
                          e.truth && static_cast<u64>(sz[i]) <= e.cap &&
                          guest_bytes <= e.guest_bytes;
    const bool exact_shape =
        truth_same ||
        (!truth_i && !e.truth && e.buf && e.size == static_cast<u64>(sz[i]) &&
         e.guest_bytes == guest_bytes && SameCsResourceShape(e.res, ci.res[i]));
    // A plain buffer bound again through a SMALLER window is the same data:
    // the staged copy already covers it and the descriptor carries its own
    // range. Treating that as a reshape restages the whole thing: Astro Bot
    // has two dispatches a frame that size one 39 MB buffer differently, and
    // each flip cost a full copy and a fence wait.
    const bool subset =
        raw_view || truth_sub ||
        (kCsSubset && !exact_shape && e.buf && !e.imported && !e.truth &&
         !ci.res[i].image_staging && !e.image_staging &&
         e.size >= static_cast<u64>(sz[i]) && e.guest_bytes >= guest_bytes);
    const bool same_shape = exact_shape || subset;
    // What the buffer holds: the footprint for a truth, the linear view
    // otherwise. `stage_bytes` > 0: staged as raw guest bytes.
    const u64 foot = (raw_view || truth_sub) ? e.cap
                     : truth_i               ? guest_bytes
                                             : sz[i];
    const u64 stage_bytes = (raw_view || truth_sub) ? e.guest_bytes
                            : truth_i               ? guest_bytes
                                                    : 0;
    const u64 hash_bytes = stage_bytes ? stage_bytes : guest_bytes;
    if (!same_shape && e.gpu_dirty) {
      const u64 trs = NowNs();
      g_wb_why["reshape"]++;
      g_wb_why_bytes += e.size;
      const bool ok = CsRangeFlushOne(base, e);
      g_in_reshape_ns += NowNs() - trs;
      g_in_reshape_n++;
      if (!ok)
        return CsDeclined(ci, "12");  // reshaped: keep its data
    }

    // Guest-page import: valid only for plain (non-image) linear ranges, and
    // only while the range is not already staged some other way.
    if ((kCsImport || ci.res[i].prefer_import) && !e.buf &&
        !ci.res[i].image_staging && !ci.res[i].zero_fill)
      CsRangeImportGuest(e, base, sz[i]);
    const bool buffer_reused = e.buf && e.cap >= foot;
    const u64 ta = NowNs();
    const bool ensured = CsRangeEnsureBuffer(e, foot);
    g_in_alloc_ns += NowNs() - ta;
    g_in_alloc_n += !buffer_reused;
    if (!ensured)
      return CsDeclined(ci, "14");
    if (!buffer_reused && caps.debug_labels) {
      char name[32];
      std::snprintf(name, sizeof(name), "csbuf %#llx",
                    (unsigned long long)base);
      Device().SetName(e.buf, name);
    }
    // A buffer that was just rebuilt holds nothing, whatever the shape
    // bookkeeping says about it.
    bool valid = buffer_reused && same_shape &&
                 (e.gpu_dirty || e.last_validated_frame == g_frame.num);
    if (e.imported && same_shape)
      valid = true;  // the buffer IS the guest pages; nothing to copy
    // A read whose base is a live render target must be staged from the
    // image: the guest bytes under an RT are stale (draws never write them
    // back), and the image content changes every frame regardless of the
    // guest hash. Attempted at most once per frame per range; a CS-written
    // buffer (gpu_dirty) stays authoritative.
    const bool rt_backed = ci.res[i].image_staging && CsAliasedBase(base);
    // ... and at most once per frame per range, but a target NOTHING has
    // rendered into since the last bridge still holds the bytes already
    // staged. Astro Bot's shadow array (1536x1536 x16 layers = 151 MB) is
    // re-lit rarely and re-copied every frame without this; that one range is
    // 40% of a frame's image staging.
    const bool rt_stale = kCsRtCache && e.rt_sourced && e.last_rt_frame >= 0 &&
                          AliasedImageRenderSerial(base) <= e.rt_serial;
    const bool rt_attempt = rt_backed && !e.gpu_dirty && !rt_stale;
    if (rt_attempt)
      valid = false;
    else if (rt_backed && !e.gpu_dirty && e.rt_sourced)
      valid = same_shape;
    if (!valid && same_shape && !rt_attempt) {
      // The whole range, not sampled windows: a CPU write that missed every
      // window (a buffer streaming in at a scene cut) kept the stale staging
      // forever, and GTA:SA's lighting went psychedelic in 1 run of 4.
      const u64 th = NowNs();
      const u64 h = TexHash(base, hash_bytes);
      g_in_hash_ns += NowNs() - th;
      g_in_hash_n++;
      if (h == e.hash)
        valid = true;
      else
        e.hash = h;
      e.last_validated_frame = g_frame.num;
    }
    // A range the shader only WRITES needs no content from guest memory: it
    // supplies every byte it will write back. SotC's material fills are whole
    // 4 MiB arenas of exactly that shape, and staging them in is ~60 ms a frame
    // of pure waste (`in=` in the fps line). Opt-in because a shader that
    // writes only PART of such a range would write back whatever the buffer
    // happened to hold (the resource plan says "never read", not "writes all
    // of it").
    // Direct-memory arenas only. A compute range that aliases module .data
    // (modules at 0x2xxxxxxxxxxx, the dmem arena at 0x80xxxxxxxx) gets a
    // partial write + writeback of stale staging over live globals when the
    // plan's "written" bit is wrong or the write is partial; SotC's menu
    // transition died exactly that way (SIGSEGV in malloc).
    const bool dmem_arena = base >= 0x8000000000ull && base < 0x9000000000ull;
    if (kCsSkipUpload && !valid && dmem_arena && !ci.res[i].read &&
        ci.res[i].written && !ci.res[i].image_staging && !ci.res[i].zero_fill &&
        same_shape && !stage_bytes) {
      valid = true;
      g_cs_skip_n++;
    }
    // The frame's open chunk copies a texture out of this buffer and this
    // dispatch rewrites it: the chunk goes on the queue first, so the draw
    // gets the contents it sampled.
    if (e.buf && e.chunk_ref == g_frame.chunk_seq &&
        (!valid || fold_count || ci.res[i].written ||
         ci.res[i].shader_writes) &&
        !CsSplitFrameChunk()) {
      renderer.state = nullptr;
      return CsDeclined(ci, "22");
    }
    if (!valid) {
      // CPU write into a buffer a pending batched dispatch reads/writes.
      // Renaming avoids the stall when the old contents are read-only.
      if (kCsRename && e.pending_batch && !e.gpu_dirty && !e.res.written) {
        if (!CsRangeRename(e, sz[i])) {
          renderer.state = nullptr;
          return CsDeclined(ci, "15");
        }
        e.pending_batch = false;
      } else if (!CsBatchWaitRange(e, kSyncStageHazard)) {
        renderer.state = nullptr;
        return CsDeclined(ci, "16");
      }
      if (rt_attempt) {
        e.last_rt_frame = static_cast<int>(g_frame.num);
        e.rt_serial = AliasedImageRenderSerial(base);
        const u64 tr = NowNs();
        e.rt_sourced = StageCsRangeFromRt(ci.res[i], e);
        g_in_rt_ns += NowNs() - tr;
        g_in_rt_n++;
        // DELTA_GPU_CSRT: trace every RT-backed staging decision.
        static int rt_trace_logged = 0;
        if (kCsRtTrace && rt_trace_logged < 200) {
          rt_trace_logged++;
          BASE_LOGI("csrt", "f{} base={:#x} {}x{} -> {}", (int)g_frame.num,
                    (unsigned long)base, ci.res[i].width, ci.res[i].height,
                    e.rt_sourced ? "staged-from-RT" : "guest-fallback");
        }
        // A failed bridge is either "not a live target" (fall back) or a lost
        // device (stop recording altogether, like every other failed flush).
        if (g_cs_failed) {
          renderer.state = nullptr;
          return CsDeclined(ci, "17");
        }
      }
      if (!rt_attempt || !e.rt_sourced) {
        // Armed before the copy, so a write racing it is reported.
        if (ci.res[i].written && CsTracksGuestWrites())
          GuestWriteTracker().Arm(base, hash_bytes);
        e.cpu_writes.clear();
        bool gpu_staged = false;
        if (stage_bytes) {
          g_truth_stage_bytes += stage_bytes;
          gpu_staged = RecordTruthImport(e, base, stage_bytes);
          if (gpu_staged)
            g_cs_batch_access.erase(e.buf);
          else {
            const u64 tm = NowNs();
            ParallelCopy(e.map, reinterpret_cast<const void*>(base),
                         stage_bytes);
            g_in_membuf_ns += NowNs() - tm;
          }
        } else if (ci.res[i].image_staging) {
          g_stage_guest_detile_bytes += sz[i];
          g_stage_guest_detile_n++;
          const u64 td = NowNs();
          gpu_staged = ConvertCsRangeImage(ci.res[i], e, true);
          const bool ok =
              gpu_staged || (!g_cs_failed && StageCsImage(ci.res[i], e.map));
          g_in_detile_ns += NowNs() - td;
          if (!ok)
            return CsDeclined(ci, "18");
        } else {
          const u64 tm = NowNs();
          std::memcpy(e.map, reinterpret_cast<const void*>(base),
                      ci.res[i].size);
          if (sz[i] > ci.res[i].size)
            std::memset(static_cast<u8*>(e.map) + ci.res[i].size, 0,
                        sz[i] - ci.res[i].size);
          g_in_membuf_ns += NowNs() - tm;
        }
        e.rt_sourced = false;
        if (rt_attempt) {  // fell back: keep guest-hash bookkeeping coherent
          e.hash = TexHash(base, hash_bytes);
          e.last_validated_frame = g_frame.num;
        }
        // The CPU wrote the host mirror; the shaders bind the VRAM copy.
        const u64 tc = NowNs();
        if (!gpu_staged)
          CsCopyStaging(e, stage_bytes ? stage_bytes : sz[i],
                        /*to_device=*/true);
        g_in_copy_ns += NowNs() - tc;
      }
      if (!same_shape) {
        e.hash = TexHash(base, hash_bytes);
        e.last_validated_frame = g_frame.num;
      }
      // Baseline for the writeback's write-coverage merge: the staging buffer
      // as it stands BEFORE any dispatch runs. Comparing against it is the only
      // way the writeback can tell a shader's output from a byte it merely
      // staged in and would otherwise copy back over the CPU's newer value.
      // Imported ranges never write back, and images retile through
      // WritebackCsImage.
      if (!ci.res[i].image_staging && !e.imported && !stage_bytes) {
        const u64 ts = NowNs();
        e.shadow.resize(sz[i]);
        std::memcpy(e.shadow.data(), e.map, sz[i]);
        e.shadow_valid = true;
        g_in_shadow_ns += NowNs() - ts;
      } else {
        e.shadow_valid = false;
      }
      e.gpu_dirty = false;
      e.write_seq++;
      BumpTextureEpoch();
      g_cs_stage_n++;
      g_cs_stage_bytes += sz[i];
      if (kCsSyncReport) {
        StageStat& st = g_stage_stats[base];
        st.bytes += sz[i];
        st.n++;
        if (!buffer_reused)
          st.first++;
        else if (!same_shape) {
          st.shape++;
          // Which field flipped is the whole question: two dispatches that
          // describe the same surface slightly differently restage it in full.
          static base::HashSet<u64> shape_logged, big_logged;
          auto& logged = sz[i] >= (1u << 20) ? big_logged : shape_logged;
          if (logged.size() < 64 && logged.insert(base).second)
            BASE_LOGI(
                "csshape",
                "{:#x} was {}x{} p={} l={} m={} t={} e={}/{} d={} sz={:#x}"
                " now {}x{} p={} l={} m={} t={} e={}/{} d={} sz={:#x}",
                (unsigned long)base, e.res.width, e.res.height, e.res.pitch,
                e.res.layers, e.res.mip_levels, e.res.tiling_idx,
                e.res.elem_bytes, e.res.stage_elem_bytes, e.res.dfmt,
                (unsigned long)e.size, ci.res[i].width, ci.res[i].height,
                ci.res[i].pitch, ci.res[i].layers, ci.res[i].mip_levels,
                ci.res[i].tiling_idx, ci.res[i].elem_bytes,
                ci.res[i].stage_elem_bytes, ci.res[i].dfmt,
                (unsigned long)sz[i]);
        } else if (rt_attempt)
          st.rt++;
        else
          st.hash++;
      }
      // Which staged bytes the CPU only ever WRITES: those could live straight
      // in host-visible VRAM (ReBAR) with no mirror and no copy, because
      // nothing ever reads them back across the bus.
      if (ci.res[i].image_staging)
        g_stage_img_bytes += sz[i];
      else if (ci.res[i].shader_writes || ci.res[i].written)
        g_stage_rw_bytes += sz[i];
      else
        g_stage_ro_bytes += sz[i];
    }
    // A subset bind keeps the entry at the larger footprint it was staged and
    // hashed at: shrinking it here is what made the next full-size bind look
    // like a reshape.
    if (!subset) {
      e.size = truth_i ? guest_bytes : sz[i];
      if (e.guest_bytes != guest_bytes) {
        UnindexCsRange(base, e.guest_bytes);
        IndexCsRange(base, guest_bytes);
        // Still owed to guest memory: its dirty footprint grows with it, or a
        // reader of the new part would find nothing to flush.
        if (e.gpu_dirty)
          IndexDirtyRange(base, guest_bytes);
      }
      e.guest_bytes = guest_bytes;
      e.image_staging = ci.res[i].image_staging;
      e.truth = truth_i;
      e.res = ci.res[i];
    }
    if (ci.res[i].image_staging)
      g_cs_image_staged++;
    e.last_used_frame = g_frame.num;
    for (u32 k = 0; k < fold_count && e.buf; k++) {
      auto f = g_cs_ranges.find(folds[k]);
      if (f == g_cs_ranges.end())
        continue;
      CsRange& s = f->second;
      if (!s.gpu_dirty || !s.buf)
        continue;
      const u64 n = base::Min<u64>(s.size ? s.size : s.cap, s.guest_bytes);
      CsBatchBeginImpl();
      g_cs_list->Barrier(kAccessComputeRW | kAccessCopyRW, kAccessCopyRW);
      g_cs_list->CopyBuffer(e.buf, folds[k] - base, s.buf, 0, n);
      g_cs_list->Barrier(rhi::kAccessCopyWrite,
                         kAccessComputeRW | kAccessCopyRW);
      g_cs_batch_access.erase(e.buf);
      UnindexDirtyRange(folds[k], s.guest_bytes);
      s.gpu_dirty = false;
      s.last_validated_frame = -1;
      s.hash = 0;
      s.mirror_current = false;
      MarkPending(s);
      if (!e.gpu_dirty) {
        IndexDirtyRange(base, e.guest_bytes);
        e.gpu_dirty = true;
        e.dirty_frame = g_frame.num;
      }
      e.write_seq++;
      BumpTextureEpoch();
      e.mirror_current = false;
      MarkPending(e);
      g_truth_fold_n++;
    }
    bind_buf[i] = e.buf;
    if (truth_i) {
      if (!bind_truth_view(i, e.buf, 0, e.guest_bytes, 0))
        return CsDeclined(ci, "23");
      MarkPending(e);
    }
  }
  // Re-resolve handles: a later binding sharing an earlier binding's base may
  // have grown (destroyed + recreated) that range's buffer.
  for (u32 i = 0; i < ci.num_res; i++)
    if (!ci.res[i].zero_fill && !truth_view[i])
      bind_buf[i] =
          g_cs_ranges[parent_base[i] ? parent_base[i] : ci.res[i].base].buf;
  u64 bind_off[ComputeInfo::kMaxResources] = {};

  g_ns_cs_in += NowNs() - t_in0;

  // The storage buffers, pushed into the batch's command list.
  rhi::BindingWrite wr[ComputeInfo::kMaxResources + 2];
  for (u32 i = 0; i < ci.num_res; i++)
    bind_off[i] = ci.res[i].zero_fill ? 0
                  : truth_view[i]     ? 0
                  : parent_base[i]
                      ? parent_off[i]
                      : g_cs_ranges[ci.res[i].base].imported_offset;
  // A view that starts off the device's storage alignment binds from the
  // aligned offset below it; the shader adds the difference (see the bias
  // bits of the pushed bound in the recompiler).
  u32 bias_dwords[ComputeInfo::kMaxResources] = {};
  const u64 align = base::Max<u64>(caps.storage_offset_alignment, 4);
  for (u32 i = 0; i < ci.num_res; i++) {
    const u64 view = ci.res[i].zero_fill ? 0 : ci.res[i].view_offset;
    const u64 off = bind_off[i] + view;
    const u64 skew = off % align;
    if (skew && (off % 4 || ci.res[i].binding >= 48 ||
                 (sz[i] - view) / 4 >= (1u << 26)))
      return CsDeclined(ci, "unaligned-view");
    bias_dwords[i] = static_cast<u32>(skew / 4);
    wr[i].binding = ci.res[i].binding;
    wr[i].buffer = bind_buf[i];
    wr[i].offset = off - skew;
    wr[i].range = sz[i] - view + skew;
  }
  u32 nwrite = ci.num_res;
  if (ci.gds_binding >= 0 && EnsureGdsBuffer()) {
    wr[nwrite].binding = static_cast<u32>(ci.gds_binding);
    wr[nwrite].buffer = g_gds.buf;
    wr[nwrite].range = GdsBuffer::kBytes;
    nwrite++;
  }
  if (ci.recomp->guest_memory_binding >= 0) {
    wr[nwrite].binding = ci.recomp->guest_memory_binding;
    wr[nwrite].buffer = guest_table;
    wr[nwrite].range = guest_table_bytes;
    nwrite++;
  }

  // Record the dispatch into the open batch. Submission + the wait happen at
  // the next flush point, not here.
  CsBatchBeginImpl();
  u32 zero_count = 0;
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!ci.res[i].zero_fill)
      continue;
    zero_count++;
    g_cs_batch_access.erase(bind_buf[i]);
  }
  if (zero_count) {
    g_cs_list->Barrier(kAccessComputeRW, rhi::kAccessCopyWrite);
    for (u32 i = 0; i < ci.num_res; i++)
      if (ci.res[i].zero_fill)
        g_cs_list->FillBuffer(bind_buf[i], 0, sz[i], 0);
    g_cs_list->Barrier(rhi::kAccessCopyWrite, kAccessComputeRW);
  }
  if (ci.gds_binding >= 0)
    g_cs_list->Barrier(rhi::kAccessComputeWrite, kAccessComputeRW);
  g_cs_list->SetPipeline(cp->pipe);
  g_cs_list->PushBindGroup(0, cp->set_layout, wr, nwrite);
  // The block is 16 user-data dwords plus 48 bounds; its total is the driver's
  // maxPushConstantsSize (256 B == 64 dwords).
  constexpr u32 kCsPushDwords = 64;
  static_assert(16 + 48 == kCsPushDwords);
  u32 pc[kCsPushDwords];
  std::memcpy(pc, ci.user_data, sizeof(pc[0]) * 16);
  // Every SSBO access clamps to the bound the dispatch mapped for its
  // binding; 0 means unbounded (a binding the shader indexes nowhere).
  for (u32 i = 16; i < kCsPushDwords; i++)
    pc[i] = 0;
  // Bits [31:26] carry the bias, so a bound past them reads as unbounded.
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!bind_buf[i] || ci.res[i].binding >= kCsPushDwords - 16)
      continue;
    const u64 dwords = (sz[i] - ci.res[i].view_offset) / 4;
    pc[16 + ci.res[i].binding] =
        (dwords < (1u << 26) ? static_cast<u32>(dwords) : 0) |
        (bias_dwords[i] << 26);
  }
  g_cs_list->SetPushConstants(0, sizeof(pc), pc);
  // One barrier covers every hazard the dispatch has against the batch.
  u32 hazard_src = 0, hazard_dst = 0;
  rhi::Buffer* unique_buffers[ComputeInfo::kMaxResources];
  bool unique_writes[ComputeInfo::kMaxResources] = {};
  u32 unique_count = 0;
  for (u32 i = 0; i < ci.num_res; i++) {
    u32 j = 0;
    while (j < unique_count && unique_buffers[j] != bind_buf[i])
      j++;
    if (j == unique_count) {
      unique_buffers[unique_count] = bind_buf[i];
      unique_writes[unique_count] = ci.res[i].shader_writes;
      unique_count++;
    } else {
      unique_writes[j] |= ci.res[i].shader_writes;
    }
  }
  for (u32 i = 0; i < unique_count; i++) {
    ComputeBufferAccess& prior = g_cs_batch_access[unique_buffers[i]];
    const ComputeBufferAccess current{true, unique_writes[i]};
    const bool hazard = NeedsComputeBarrier(prior, current);
    if (hazard) {
      hazard_src |= (prior.read ? rhi::kAccessComputeRead : 0) |
                    (prior.write ? rhi::kAccessComputeWrite : 0);
      hazard_dst |= rhi::kAccessComputeRead |
                    (unique_writes[i] ? rhi::kAccessComputeWrite : 0);
      prior = {};
    }
    prior.read = true;
    prior.write |= unique_writes[i];
  }
  if (hazard_src)
    g_cs_list->Barrier(hazard_src, hazard_dst);
  if (caps.debug_labels) {
    char label[96];
    std::snprintf(label, sizeof(label), "dispatch cs=%#llx %ux%ux%u res=%u",
                  (unsigned long long)ci.cs_addr, ci.groups[0], ci.groups[1],
                  ci.groups[2], ci.num_res);
    g_cs_list->InsertLabel(label);
  }
  if (ci.recomp->guest_memory_binding >= 0)
    g_cs_list->Barrier(rhi::kAccessHostWrite, rhi::kAccessComputeRead);
  const bool traced = g_cs_timestamps && timeline::Active();
  u16 query = 0;
  if (traced) {
    char name[48];
    std::snprintf(name, sizeof(name), "cs %#llx",
                  (unsigned long long)ci.cs_addr);
    query = timeline::Begin(name);
  }
  const bool stamp = g_cs_timestamps && (kCsSyncReport || traced);
  if (stamp)
    g_cs_list->WriteTimestamp(g_cs_timestamps, g_cs_batch_count * 2,
                              /*start=*/true);
  DispatchCheckpoint(g_cs_list, ci.cs_addr, false);
  g_cs_list->DispatchBase(ci.group_base[0], ci.group_base[1], ci.group_base[2],
                          ci.groups[0], ci.groups[1], ci.groups[2]);
  DispatchCheckpoint(g_cs_list, ci.cs_addr, true);
  if (stamp)
    g_cs_list->WriteTimestamp(g_cs_timestamps, g_cs_batch_count * 2 + 1,
                              /*start=*/false);
  if (traced)
    timeline::End(query);
  // What the dispatch wrote through a detiled view goes back into the truth.
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!truth_view[i] || !ci.res[i].written)
      continue;
    bool done = false;
    for (u32 j = 0; j < i && !done; j++)
      done = truth_view[j] == truth_view[i] && ci.res[j].written;
    if (done)
      continue;
    auto it =
        g_cs_ranges.find(parent_base[i] ? parent_base[i] : ci.res[i].base);
    if (it == g_cs_ranges.end() || !it->second.buf)
      continue;
    const u64 span =
        parent_base[i]
            ? (ci.res[i].guest_size ? ci.res[i].guest_size : ci.res[i].size)
            : it->second.guest_bytes;
    RecordImageTiling(g_cs_list, *truth_table[i], it->second.buf, parent_off[i],
                      span, truth_view[i], sz[i],
                      /*detile=*/false);
    g_cs_batch_access.erase(it->second.buf);
    g_truth_retile_n++;
  }
  {
    BatchedDispatch bd{ci.cs_addr, {ci.groups[0], ci.groups[1], ci.groups[2]},
                       ci.num_res, {},
                       {},         0,
                       0};
    for (u32 i = 0; i < ci.num_res && i < 4; i++) {
      bd.res_base[i] = ci.res[i].base;
      bd.res_size[i] = ci.res[i].size;
      if (ci.res[i].image_staging)
        bd.res_img |= static_cast<u8>(1u << i);
      if (ci.res[i].written)
        bd.res_write |= static_cast<u8>(1u << i);
    }
    bd.traced = traced;
    bd.query = query;
    g_cs_batch_log.push_back(bd);
  }
  if (trace::Recording())
    trace::RecordDispatch(ci);
  for (u32 i = 0; i < ci.num_res; i++) {
    if (ci.res[i].zero_fill)
      continue;
    auto it =
        g_cs_ranges.find(parent_base[i] ? parent_base[i] : ci.res[i].base);
    if (it != g_cs_ranges.end())
      MarkPending(it->second);
  }
  ++g_cs_batch_count;

  // Mark written ranges GPU-dirty. Guest memory catches up lazily at the next
  // flush point (draw / DMA / frame end). Writing every dispatch's outputs
  // back immediately (the image retile especially) was ~100ms/frame.
  const u64 t_out0 = NowNs();
  bool kick = false;
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!ci.res[i].written || ci.res[i].zero_fill)
      continue;
    const u64 dirty_base = parent_base[i] ? parent_base[i] : ci.res[i].base;
    auto it = g_cs_ranges.find(dirty_base);
    if (it == g_cs_ranges.end())
      continue;
    if (!it->second.gpu_dirty) {
      IndexDirtyRange(dirty_base, it->second.guest_bytes);
      it->second.dirty_frame = g_frame.num;
    }
    it->second.gpu_dirty = true;
    it->second.write_seq++;
    BumpTextureEpoch();
    if (truth_view[i])
      StampView(truth_view[i], dirty_base, parent_off[i], truth_table[i],
                it->second.write_seq);
    it->second.mirror_current = false;    // the dispatch outran the host mirror
    it->second.readback_pending = false;  // and any readback still queued
    if (it->second.cpu_reader && kCsEagerReadback) {
      CsStageReadback(dirty_base, it->second, /*all=*/false);
      kick = true;
    }
    if (kGpuCsgpuVerbose) {
      const u8* b = static_cast<const u8*>(it->second.map);
      u64 nz = 0, step = ci.res[i].size > 65536 ? ci.res[i].size / 65536 : 1;
      for (u64 k = 0; k < ci.res[i].size; k += step)
        nz += b[k] != 0;
      BASE_LOGI("csgpu", "gpu wrote base={:#x} size={} nonzero={}/{}",
                (unsigned long)ci.res[i].base, (unsigned long)ci.res[i].size,
                (unsigned long)nz, (unsigned long)(ci.res[i].size / step));
    }
  }
  // A CPU reader is coming for what this batch wrote: start it now.
  if ((g_cs_batch_count >= base::Max<u32>(1, kCsBatchCap) || kick ||
       kGpuCsgpuVerbose) &&
      !CsBatchSubmit()) {
    renderer.state = nullptr;
    return CsDeclined(ci, "21");
  }
  if (ci.recomp->guest_memory_written) {
    // Publish physical writes before a CPU consumer or another dispatch can
    // use a stale staged copy. The dirty bitmap identifies actual stores,
    // rather than treating the shader's whole guest pool as overwritten.
    CsBatchBeginImpl();
    g_cs_list->Barrier(rhi::kAccessComputeWrite, rhi::kAccessHostRead);
    if (!CsBatchFlush(kSyncBatchCap))
      return CsDeclined(ci, "guest-writeback");
    // Staged copies of the pages it wrote go out first, so its direct writes
    // land on top of them, not under a later writeback.
    for (const auto& r : ci.guest_memory)
      if (r.writable &&
          !FlushCsWritesRange(renderer, r.base, r.size, "guestmem-w"))
        return CsDeclined(ci, "guest-writeback");
    const auto written = FinishGuestMemoryWrites();
    for (const auto& range : written) {
      NoteGuestWrite(range.base, range.size);
      NoteDccWrite(range.base, range.size, nullptr);
      NoteRawWrite(range.base, range.size);
      InvalidateTexRange(range.base, range.size);
    }
    for (auto it = g_cs_ranges.begin(); it != g_cs_ranges.end();) {
      const u64 base = it->first, end = base + it->second.guest_bytes;
      const bool touched =
          base::AnyOf(written.begin(), written.end(), [&](const auto& r) {
            return r.base < end && base < r.base + r.size;
          });
      if (touched) {
        it = EraseCsRange(it);
      } else {
        ++it;
      }
    }
    if (!written.empty())
      ++g_cs_writeback_gen;
  }
  g_ns_cs_out += NowNs() - t_out0;
  return true;
}

// Make guest memory current with every GPU-written compute range. Called
// before anything that consumes guest memory: draws (vertex/texture reads at
// record time), CP DMA copies, and the end of each frame (bounds staleness
// for direct guest CPU readers to one frame). Cheap no-op when nothing is
// dirty; also evicts cold entries so the working set stays bounded.
bool FlushCsWrites(Renderer& renderer) {
  if (g_cs_failed) {
    renderer.state = nullptr;
    return false;
  }
  const u64 t0 = NowNs();
  bool all_current = true;
  // Record every readback first: one fence wait then covers all of them,
  // instead of one submit+wait per dirty range.
  CsStageReadbacks();
  for (auto it = g_cs_ranges.begin(); it != g_cs_ranges.end();) {
    if (!CsRangeFlushOne(it->first, it->second)) {
      if (g_cs_failed) {
        renderer.state = nullptr;
        return false;
      }
      // Writeback of this one range failed; it stays dirty. Keep flushing the
      // rest so one bad range cannot hold every other range stale forever.
      all_current = false;
    }
    if (!it->second.gpu_dirty && !it->second.pending_batch &&
        !CsFrameReferenced(it->second) &&
        it->second.last_used_frame + 60 < g_frame.num) {
      it = EraseCsRange(it);
    } else {
      ++it;
    }
  }
  g_ns_cs_out += NowNs() - t0;
  return all_current;
}

// DELTA_GPU_MEMSTAT: what the renderer holds in device memory, every 300
// frames.
DELTA_OPTION(bool, kMemStat, "DELTA_GPU_MEMSTAT", false);

void ReportGpuMemory() {
  if (!kMemStat || g_frame.num % 300)
    return;
  u64 dirty = 0, truth = 0, cold = 0;
  for (const auto& [base, e] : g_cs_ranges) {
    (void)base;
    if (e.gpu_dirty)
      dirty += e.cap;
    if (e.truth)
      truth += e.cap;
    if (e.last_used_frame + 60 < g_frame.num)
      cold += e.cap;
  }
  u64 rt = 0, rt_parked = 0, depth = 0;
  for (const auto& [base, t] : g_rts) {
    (void)base;
    rt += RtByteSize(t);
  }
  for (const auto& [base, v] : g_rt_variants) {
    (void)base;
    for (const RTarget& t : v)
      rt_parked += RtByteSize(t);
  }
  for (const auto& [base, d] : g_depths) {
    (void)base;
    depth += u64(d.w) * d.h * 5 * d.layers;
  }
  for (const auto& [base, v] : g_depth_variants) {
    (void)base;
    for (const DepthTarget& d : v)
      depth += u64(d.w) * d.h * 5 * d.layers;
  }
  u64 scratch = 0;
  for (const CsScratch& c : g_cs_scratch)
    scratch += c.cap;
  const u64 tiling = g_tiling.buffers[0].cap + g_tiling.buffers[1].cap +
                     g_tiling.buffers[2].cap + g_truth_bridge_scratch.cap +
                     g_bridge.cap;
  u64 heap_used = 0, heap_budget = 0;
  Device().QueryMemoryBudget(&heap_used, &heap_budget);
  constexpr double kMb = 1024.0 * 1024.0;
  BASE_LOGI("memstat",
            "f{} device heaps {:.0f}/{:.0f}MB | tex {:.0f}MB "
            "scratch {:.0f}MB tiling {:.0f}MB",
            g_frame.num, heap_used / kMb, heap_budget / kMb,
            gpu::render::g_tex_image_bytes / kMb, scratch / kMb, tiling / kMb);
  BASE_LOGI("memstat",
            "f{} cs ranges={} {:.0f}MB (dirty {:.0f} truth {:.0f} cold {:.0f}) "
            "free {:.0f}MB | rt {} {:.0f}MB parked {:.0f}MB | depth {:.0f}MB",
            g_frame.num, g_cs_ranges.size(), g_cs_range_bytes / kMb,
            dirty / kMb, truth / kMb, cold / kMb, g_cs_free_bytes / kMb,
            g_rts.size(), rt / kMb, rt_parked / kMb, depth / kMb);
}

bool FlushCsWritesFrameEnd(Renderer& renderer, bool writeback) {
  DELTA_ZONE("gpu.cs_flush_frame_end");
  ReportGpuMemory();
  const WaitReaderScope reader("frame-end");
  if (g_cs_failed) {
    renderer.state = nullptr;
    return false;
  }
  const u64 t0 = NowNs();
  bool all_current = true;
  if (writeback) {
    // An image range no live target aliases has no CPU reader this frame: it
    // stays in VRAM, where the next dispatch or draw takes it from, and a
    // guest reader that does turn up gets its writeback then.
    // A truth that aliases a live colour target refreshes the image here,
    // on the frame's own command buffer, and then stays like any other.
    auto stays = [](u64 base, CsRange& e) {
      if (CsLazyKept(base, e))
        return true;
      return kCsImageLazyWb && e.truth && e.gpu_dirty && !e.imported &&
             (e.rt_seq == e.write_seq || CsRefreshRtInFrame(base, e));
    };
    CsStageReadbacks(/*all=*/false);
    for (auto it = g_cs_ranges.begin(); it != g_cs_ranges.end();) {
      if (stays(it->first, it->second)) {
        g_lazy_kept_n++;
        g_lazy_kept_bytes += it->second.size;
      } else if (it->second.gpu_dirty) {
        const CsRange& fe = it->second;
        g_wb_why[base::String("frame-end") + (fe.truth           ? "/img"
                                              : fe.image_staging ? "/lin"
                                                                 : "/buf")]++;
        g_wb_why_bytes += fe.size;
        if (!CsRangeFlushOne(it->first, it->second)) {
          if (g_cs_failed) {
            renderer.state = nullptr;
            return false;
          }
          all_current = false;
        }
      }
      if (!it->second.gpu_dirty && !it->second.pending_batch &&
          !CsFrameReferenced(it->second) &&
          it->second.last_used_frame + 60 < g_frame.num) {
        it = EraseCsRange(it);
      } else {
        ++it;
      }
    }
  }
  // DELTA_GPU_CS_AGED: ranges holding dispatch output no reader has asked for
  // in over two frames, i.e. what a blanket writeback would publish and the
  // lazy model does not.
  if (kCsAged && g_frame.num % 120 == 0) {
    u32 logged = 0;
    for (const auto& [base, e] : g_cs_ranges) {
      if (!e.gpu_dirty || e.dirty_frame >= g_frame.num - 2 || logged++ >= 24)
        continue;
      BASE_LOGI("csaged",
                "base={:#x} +{:#x} age={} truth={} img={} {}x{} layers={} "
                "dfmt={} tiling={} rt={} rtseq={} cpu_reader={} used={}",
                (unsigned long)base, (unsigned long)e.guest_bytes,
                g_frame.num - e.dirty_frame, (int)e.truth, (int)e.image_staging,
                e.res.width, e.res.height, e.res.layers, e.res.dfmt,
                e.res.tiling_idx, (int)CsAliasedBase(base),
                (int)(e.rt_seq == e.write_seq), (int)e.cpu_reader,
                g_frame.num - e.last_used_frame);
    }
  }
  if (!writeback) {
    // A range nothing has used for a while is evicted, and its buffer holds
    // the only copy of what a dispatch wrote: write that back first.
    for (auto it = g_cs_ranges.begin(); it != g_cs_ranges.end();) {
      CsRange& e = it->second;
      if (e.pending_batch || CsFrameReferenced(e) ||
          e.last_used_frame + 60 >= g_frame.num) {
        ++it;
        continue;
      }
      if (e.gpu_dirty) {
        g_wb_why["idle"]++;
        g_wb_why_bytes += e.size;
        if (!CsRangeFlushOne(it->first, e)) {
          if (g_cs_failed) {
            renderer.state = nullptr;
            return false;
          }
          ++it;
          continue;
        }
      }
      it = EraseCsRange(it);
    }
  }
  // The frame's command buffer may copy out of these ranges (CsSupplyTexture)
  // and is submitted next: whatever the batch still holds goes first.
  if (g_cs_batch_open && !CsBatchSubmit()) {
    renderer.state = nullptr;
    return false;
  }
  g_ns_cs_out += NowNs() - t0;
  return all_current;
}

// Targeted variant: flush only dirty ranges overlapping [base, base+bytes).
// The per-draw guest readers (texture upload, vertex copy, cbuffer ring) call
// this instead of the full flush: flushing every dirty range at every draw
// re-tiled the whole post chain ~19x/frame.
bool FlushCsWritesRange(Renderer& renderer,
                        u64 base,
                        u64 bytes,
                        const char* why) {
  // Nothing dirty anywhere: answer without touching the page index, which
  // otherwise allocates a vector, hashes a lookup per page, then sorts and
  // dedups it. That is called once per guest read, and SotC issues 1.2M
  // DRAW_INDEX_INDIRECT packets a run, each flushing before it reads its
  // argument struct, which measured 33 seconds of a 150 s run. The index is
  // maintained on both edges (indexed when a range goes dirty, unindexed when
  // it is flushed), so empty means "no writeback outstanding".
  if (g_cs_dirty_pages.empty())
    return true;
  if (g_cs_failed) {
    renderer.state = nullptr;
    return false;
  }
  // Most calls are per s_load of the descriptor replay and overlap nothing:
  // answer those before the timers and the candidate vector.
  if (!base || !bytes || g_cs_ranges.empty() ||
      (!CsRangeDirtyOverlapping(base, bytes) &&
       !(kCsFlushTrace && base == (u64)kCsFlushTrace)))
    return true;
  ScopeNs flush_timer(&g_ns_cs_flush);
  const u64 t0 = NowNs();
  bool all_current = true;
  // DELTA_GPU_CSFLUSHTRACE=<base>: what the dirty-range index finds for a
  // target a draw is about to sample. A compute result that is not found here
  // never reaches the target's image, and the draw samples black.
  if (kCsFlushTrace && base == (u64)kCsFlushTrace) {
    static int n = 0;
    if (n++ < 12) {
      const auto ranges = DirtyRangesOverlapping(base, bytes);
      base::String line;
      base::FormatTo(line, "base={:#x} bytes={:#x} dirty={}",
                     (unsigned long)base, (unsigned long)bytes, ranges.size());
      for (u64 r : ranges) {
        auto f = g_cs_ranges.find(r);
        base::FormatTo(
            line, " [{:#x} img={} gpu_dirty={} sz={:#x}]", (unsigned long)r,
            f != g_cs_ranges.end() ? (int)f->second.image_staging : -1,
            f != g_cs_ranges.end() ? (int)f->second.gpu_dirty : -1,
            f != g_cs_ranges.end() ? (unsigned long)f->second.size : 0);
      }
      // ...and every CS range that overlaps the target at all, found or not.
      for (auto& kv : g_cs_ranges) {
        const u64 e0 = kv.first, e1 = e0 + kv.second.guest_bytes;
        if (e0 < base + bytes && base < e1)
          base::FormatTo(
              line, " OVERLAP[{:#x}+{:#x} img={} dirty={}]", (unsigned long)e0,
              (unsigned long)kv.second.guest_bytes,
              (int)kv.second.image_staging, (int)kv.second.gpu_dirty);
      }
      BASE_LOGI("csflush", "{}", line.c_str());
    }
  }
  const auto overlapping = DirtyRangesOverlapping(base, bytes);
  const WaitReaderScope reader(why);
  // This read is about to cost a fence wait, so pull EVERY dirty range's
  // results across on it rather than only the ones it asked for. The waits are
  // the expensive part and one covers them all; the ranges this draw does not
  // want stay dirty, but their mirrors are now current, so the flush that
  // eventually wants them needs no wait at all. Draw-at-a-time flushing was
  // ~25 waits a frame where a frame needs 2 or 3.
  if (!overlapping.empty()) {
    CsStageReadbacks(/*all=*/false);
    for (u64 dirty : overlapping) {
      auto found = g_cs_ranges.find(dirty);
      if (found == g_cs_ranges.end() || !found->second.gpu_dirty)
        continue;
      CsRange& e = found->second;
      // The descriptor replay reads SRT tables, which never sit inside a
      // tiled image surface: a truth is not what it is after.
      if (e.truth && !std::strcmp(why, "srt"))
        continue;
      g_wb_why[base::String(why) + (e.truth           ? "/img"
                                    : e.image_staging ? "/lin"
                                                      : "/buf")]++;
      g_wb_why_bytes += e.size;
      if (e.device_local && !e.mirror_current && !e.imported &&
          (!CanGpuTile(e.res, e) || e.truth))
        CsCopyStaging(e, e.size ? e.size : e.cap, /*to_device=*/false);
    }
  }
  for (u64 dirty : overlapping) {
    auto found = g_cs_ranges.find(dirty);
    if (found != g_cs_ranges.end() && found->second.truth &&
        !std::strcmp(why, "srt"))
      continue;
    if (found != g_cs_ranges.end())
      found->second.cpu_reader = true;
    if (found != g_cs_ranges.end() && !CsRangeFlushOne(dirty, found->second)) {
      if (g_cs_failed) {
        renderer.state = nullptr;
        return false;
      }
      all_current = false;
    }
  }
  g_ns_cs_out += NowNs() - t0;
  return all_current;
}

void ApplyMemoryFill(Renderer& renderer, u64 base, u64 bytes, u32 value) {
  BumpTextureEpoch();
  if (!bytes)
    return;
  FlushCsWritesRange(renderer, base, bytes, "fill");
  u32* words = reinterpret_cast<u32*>(base);
  base::Fill(words, words + bytes / 4, value);
  CsForgetGuestRange(base, bytes);
  InvalidateTexRange(base, bytes);
  NoteMemoryFill(renderer, base, bytes, value);
  g_cs_writeback_gen++;
}

u64 CsWritebackGeneration() {
  return g_cs_writeback_gen;
}

bool CsRangeMaybeDirty(u64 base, u64 bytes) {
  return bytes && !DirtyFilterClear(base, RangeEnd(base, bytes));
}

bool CsRangeDirtyOverlapping(u64 base, u64 bytes) {
  return AnyDirtyOverlapping(base, bytes, [](u64) { return true; });
}

}  // namespace gpu::render
