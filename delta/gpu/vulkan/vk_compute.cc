/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// Compute dispatches. Each guest range a CS touches gets a persistent
// host-visible storage buffer keyed by its base address; dispatches are
// recorded into one batched command buffer, and their writes land back in guest
// memory lazily -- only when something needs guest memory to be current.

#include "gpu/vulkan/vk_guest_memory.h"
#include "gpu/rhi/renderer.h"
#include "base/arch.h"

#include "gpu/gpu_check.h"
#include "gpu/guest_memory.h"
#include "gpu/gcn/gcn_detile.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/vulkan/vk_capture.h"
#include "gpu/vulkan/vk_compute_hazard.h"
#include "gpu/vulkan/vk_debug.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_format.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_hash.h"
#include "gpu/gpu_perf.h"
#include "gpu/vulkan/vk_perf.h"
#include "gpu/vulkan/vk_render_target.h"
#include "gpu/vulkan/vk_texture_cache.h"
#include "gpu/vulkan/vk_trace.h"
#include "gpu/vulkan/vk_tiling.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <unordered_map>
#include <string>
#include <unordered_set>
#include <vector>

#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <utl/options.h>

#define BCDECDEF static inline
#define BCDEC_IMPLEMENTATION
#include <bcdec.h>

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
// part, where one buffer is already both (see FindDeviceMemoryType).
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
DELTA_OPTION(bool, kCsSyncReport, "DELTA_GPU_CSSYNC", false);
DELTA_OPTION(bool, kGpuCsgpuVerbose, "DELTA_GPU_CSGPU_VERBOSE", false);
u64 g_cs_image_staged = 0;
}  // namespace

namespace gpu::vk {
namespace {
// Writeback-half accounting (DELTA_GPU_CSSYNC); defined here because the
// aliased-image bridge below is the thing being counted.
u64 g_out_retile_ns = 0, g_out_rt_ns = 0, g_out_tail_ns = 0;
u64 g_out_buffer_ns = 0;
u64 g_tex_bridge_n = 0, g_tex_bridge_bytes = 0, g_cs_chunk_splits = 0;
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
std::map<std::string, u64> g_wb_why;
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
    std::memcpy(d + off, s + off, std::min<u64>(u64(c1 - c0) * kChunk,
                                                bytes - off));
  });
}

// Per-range staging accounting (DELTA_GPU_CSSYNC). A frame that stages half a
// gigabyte says nothing about what to fix until the bytes are attributed to a
// base and a REASON: guest memory really changed, the same base was rebound
// with a different shape, or a render target had to be bridged.
struct StageStat {
  u64 bytes = 0;
  u32 n = 0, first = 0, shape = 0, hash = 0, rt = 0, wait = 0;
  u64 wait_ns = 0;
};
std::unordered_map<u64, StageStat> g_stage_stats;

using rhi::ComputeInfo;

static_assert(ComputeInfo::kMaxResources == gcn::kMaxCsResources);

// A recompiled compute pipeline, cached by CS address: the SPIR-V + binding
// layout depend only on the code, so only the descriptor set + push constants +
// storage buffers are rebuilt per dispatch.
struct CsPipe {
  VkPipeline pipe = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  u32 num_res = 0;
  int gds_binding = -1;
};

// The GDS scratchpad: one small device-local buffer for the whole device, which
// is what the hardware's global data share is. Created on first use and left
// alone afterwards -- the counters in it are the title's to manage.
struct GdsBuffer {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void* map = nullptr;
  static constexpr VkDeviceSize kBytes = 64 * 1024;
};
GdsBuffer g_gds;

std::unordered_map<u64, CsPipe> g_cs_pipes;

// Memory for a buffer the GPU alone touches: VRAM, host visibility irrelevant.
// Only worth splitting off a host mirror when the device heap is NOT already
// cheap for the CPU to read -- on a UMA part the one buffer serves both sides
// and FindComputeMemoryType already returns it, so report none here.
u32 FindDeviceMemoryType(u32 type_bits) {
  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(g_dev.phys, &properties);
  constexpr VkMemoryPropertyFlags kUnified =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  for (u32 i = 0; i < properties.memoryTypeCount; i++)
    if ((type_bits & (1u << i)) &&
        (properties.memoryTypes[i].propertyFlags & kUnified) == kUnified)
      return UINT32_MAX;
  for (u32 i = 0; i < properties.memoryTypeCount; i++)
    if ((type_bits & (1u << i)) &&
        (properties.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
      return i;
  return UINT32_MAX;
}

u32 FindComputeMemoryType(u32 type_bits) {
  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(g_dev.phys, &properties);
  u32 best = UINT32_MAX;
  int best_score = -1;
  for (u32 i = 0; i < properties.memoryTypeCount; i++) {
    if (!(type_bits & (1u << i)))
      continue;
    const VkMemoryPropertyFlags flags = properties.memoryTypes[i].propertyFlags;
    constexpr VkMemoryPropertyFlags kRequired =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if ((flags & kRequired) != kRequired)
      continue;
    const int score = ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 4 : 0) +
                      ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? 2 : 0);
    if (score > best_score) {
      best = i;
      best_score = score;
    }
  }
  // Report the choice once: reading back from a write-combined heap is ~100
  // MB/s, which is invisible in the code and obvious in the numbers.
  static bool logged = false;
  if (!logged && best != UINT32_MAX) {
    logged = true;
    const VkMemoryPropertyFlags f = properties.memoryTypes[best].propertyFlags;
    BASE_LOGI("gpuvk", "cs staging memory type {}:{}{}{}{}", best,
              (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " DEVICE_LOCAL" : "",
              (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? " HOST_VISIBLE" : "",
              (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? " HOST_COHERENT" : "",
              (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? " HOST_CACHED" : "");
  }
  return best == UINT32_MAX
             ? FindMemoryType(type_bits,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
             : best;
}

CsPipe* GetCsPipe(const ComputeInfo& ci) {
  const u64 key = reinterpret_cast<uintptr_t>(ci.recomp);
  auto it = g_cs_pipes.find(key);
  if (it != g_cs_pipes.end())
    return it->second.num_res == ci.num_res ? &it->second : nullptr;
  CsPipe cp;
  cp.num_res = ci.num_res;
  VkDescriptorSetLayoutBinding binds[ComputeInfo::kMaxResources + 2];
  u32 nbind = 0;
  for (u32 i = 0; i < ci.num_res; i++)
    binds[nbind++] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  // The GDS scratchpad sits past the resources; it is ours, not a guest range.
  cp.gds_binding = ci.gds_binding;
  if (ci.gds_binding >= 0)
    binds[nbind++] = {static_cast<u32>(ci.gds_binding),
                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                      VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  if (ci.recomp->guest_memory_binding >= 0)
    binds[nbind++] = {static_cast<u32>(ci.recomp->guest_memory_binding),
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                     VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo sl{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  sl.bindingCount = nbind;
  sl.pBindings = binds;
  if (vkCreateDescriptorSetLayout(g_dev.device, &sl, nullptr, &cp.set_layout) !=
      VK_SUCCESS)
    return nullptr;
  // 16 user-data dwords, then one bound (in dwords) per SSBO binding. The
  // bound is what the emitted SSBO accesses clamp to; the SPIR-V block for a
  // compute stage declares the same 16 + 48 shape, 256 B total -- the driver
  // max this backend runs against.
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64 * 4};
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 1;
  li.pSetLayouts = &cp.set_layout;
  li.pushConstantRangeCount = 1;
  li.pPushConstantRanges = &pcr;
  if (vkCreatePipelineLayout(g_dev.device, &li, nullptr, &cp.layout) !=
      VK_SUCCESS) {
    vkDestroyDescriptorSetLayout(g_dev.device, cp.set_layout, nullptr);
    return nullptr;
  }
  VkShaderModule cs =
      MakeModule(ci.recomp->spirv.data(), ci.recomp->spirv.size() * 4);
  VkComputePipelineCreateInfo pi{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pi.flags = VK_PIPELINE_CREATE_DISPATCH_BASE_BIT;
  pi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pi.stage.module = cs;
  pi.stage.pName = "main";
  pi.layout = cp.layout;
  VkResult r;
  {
    if (kCsSyncReport)
      BASE_LOGI("cspipe", "compile cs={:#x} words={}", ci.cs_addr,
                ci.recomp->spirv.size());
    const u64 started = NowNs();
    ScopeNs t(&g_ns_pipe_build);
    r = vkCreateComputePipelines(g_dev.device, g_dev.pipeline_cache, 1, &pi,
                                 nullptr, &cp.pipe);
    if (kCsSyncReport)
      BASE_LOGI("cspipe", "compiled cs={:#x} ms={:.3f}", ci.cs_addr,
                double(NowNs() - started) / 1e6);
  }
  g_pipe_build_n++;
  SavePipelineCache();  // persist the driver's compiled pipeline
  vkDestroyShaderModule(g_dev.device, cs, nullptr);
  if (r != VK_SUCCESS) {
    BASE_LOGI("gpuvk", "compute pipeline failed: {}", (int)r);
    vkDestroyPipelineLayout(g_dev.device, cp.layout, nullptr);
    vkDestroyDescriptorSetLayout(g_dev.device, cp.set_layout, nullptr);
    return nullptr;
  }
  NameObject(VK_OBJECT_TYPE_PIPELINE, (u64)cp.pipe, "cs %#llx",
             (unsigned long long)ci.cs_addr);
  g_cs_pipes[key] = cp;
  return &g_cs_pipes[key];
}

// Persistent compute staging. Dispatches are serialized behind the fence, so
// one set of staging buffers (+ descriptor pool + command buffer) is reused
// across every dispatch: this avoids the per-dispatch
// vkCreateBuffer/vkAllocateMemory/pool/cmd- buffer churn (~3ms/frame). The
// buffers are HOST_CACHED so the copy-BACK read after the dispatch hits cache
// instead of stalling on write-combined memory (which was ~25ms/frame for
// Doom64's 8 MB atlas: the dominant compute cost).
struct CsStage {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void* map = nullptr;
  VkDeviceSize cap = 0;
};

CsStage g_cs_stage[ComputeInfo::kMaxResources];
VkDescriptorPool g_cs_desc_pool = VK_NULL_HANDLE;
VkCommandBuffer g_cs_cmd = VK_NULL_HANDLE;

bool BuildCsImageLayouts(const ComputeInfo::Res& res,
                         gcn::TextureLayout32& tiled,
                         gcn::TextureLayout32& linear) {
  const u32 stage_tiling = res.tiling_idx == 31 ? 31 : 8;
  const bool compressed = res.dfmt == 35 || res.dfmt == 40;
  return res.image_staging &&
         gcn::BuildTextureLayout32(tiled,
                                   compressed ? (res.width + 3) / 4 : res.width,
                                   compressed ? (res.height + 3) / 4 : res.height,
                                   compressed ? (res.pitch + 3) / 4 : res.pitch,
                                   res.layers, res.mip_levels, res.tiling_idx,
                                   res.pow2_pad, res.elem_bytes) &&
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
    return std::ldexp(static_cast<float>(mantissa), 1 - 15 - mantissa_bits);
  if (exponent == 0x1F)
    return mantissa ? std::numeric_limits<float>::quiet_NaN()
                    : std::numeric_limits<float>::infinity();
  return std::ldexp(1.f + static_cast<float>(mantissa) /
                              static_cast<float>(1u << mantissa_bits),
                    static_cast<int>(exponent) - 15);
}

u32 PackUnsignedFloat(float value, u32 mantissa_bits) {
  if (std::isnan(value))
    return (0x1Fu << mantissa_bits) | 1u;
  if (value <= 0.f)
    return 0;
  if (std::isinf(value))
    return 0x1Fu << mantissa_bits;
  int exponent;
  const float fraction = std::frexp(value, &exponent);
  int target_exponent = exponent - 1 + 15;
  if (target_exponent <= 0) {
    const long mantissa = std::lround(std::ldexp(value, 14 + mantissa_bits));
    return static_cast<u32>(
        std::clamp<long>(mantissa, 0, static_cast<long>(1u << mantissa_bits)));
  }
  if (target_exponent >= 0x1F)
    return 0x1Fu << mantissa_bits;
  long mantissa = std::lround((fraction * 2.f - 1.f) *
                              static_cast<float>(1u << mantissa_bits));
  if (mantissa == static_cast<long>(1u << mantissa_bits)) {
    mantissa = 0;
    if (++target_exponent >= 0x1F)
      return 0x1Fu << mantissa_bits;
  }
  return (static_cast<u32>(target_exponent) << mantissa_bits) |
         static_cast<u32>(mantissa);
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
  for (u32 mip = 0; fills_complete_layout && mip < linear.mip_levels;
       ++mip) {
    const auto& level = linear.mips[mip];
    const u64 logical_bytes = static_cast<u64>(level.width) *
                                   level.height * linear.layers *
                                   res.stage_elem_bytes;
    fills_complete_layout =
        level.offset == filled_bytes && level.pitch == level.width &&
        level.stored_height == level.height && level.size == logical_bytes;
    filled_bytes = level.offset + level.size;
  }
  fills_complete_layout = fills_complete_layout && filled_bytes == res.size;
  if (!fills_complete_layout)
    std::memset(dst, 0, res.size);
  std::vector<u8> tight;
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
              const u8* block = tight.data() +
                  (static_cast<size_t>(by) * src_level.width + bx) * 8;
              bcdec_bc1(block, rgba, 16);
              for (u32 y = 0; y < 4 && by * 4 + y < dst_level.height; ++y) {
                const size_t offset =
                    (static_cast<size_t>(by * 4 + y) * dst_level.pitch + bx * 4) * 4;
                std::memcpy(level_dst + offset, rgba[y],
                            std::min(4u, dst_level.width - bx * 4) * 4);
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
              const u8* block = tight.data() +
                  (static_cast<size_t>(by) * src_level.width + bx) * 16;
              bcdec_bc6h_float(block, rgb, 12, res.nfmt == 1);
              for (u32 y = 0; y < 4 && by * 4 + y < dst_level.height; ++y) {
                for (u32 x = 0; x < 4 && bx * 4 + x < dst_level.width; ++x) {
                  const float rgba[4] = {
                      rgb[y][x][0], rgb[y][x][1], rgb[y][x][2], 1.f};
                  const size_t offset =
                      (static_cast<size_t>(by * 4 + y) * dst_level.pitch +
                       bx * 4 + x) * 16;
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
          u8* dst_row = level_dst + static_cast<size_t>(y) *
                                             dst_level.pitch *
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
  // dispatch, which for a first upload is zeros -- indistinguishable from a
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
  std::vector<u8> tight;
  if (!direct)
    tight.resize(static_cast<size_t>(res.width) * res.height * res.elem_bytes);
  for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& dst_level = tiled.mips[mip];
    const auto& src_level = linear.mips[mip];
    for (u32 layer = 0; layer < tiled.layers; layer++) {
      const u8* level_src =
          static_cast<const u8*>(src) + src_level.offset +
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
              BASE_LOGI("cswb", "image retile REJECT base={:#x} mip={} "
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
                                                dst_level.width *
                                                res.elem_bytes;
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
// LAZILY — only when a draw / DMA / frame boundary needs guest memory to be
// current (FlushCsWrites), not after every dispatch.
struct CsRange {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void* map = nullptr;
  VkDeviceSize cap = 0;
  // DELTA_GPU_CSVRAM: `buf` is in VRAM (what the shaders bind) and `map` is a
  // separate host-cached mirror, moved across by DMA at the staging edges. The
  // shaders then read and write at VRAM bandwidth instead of over PCIe, which
  // is worth ~6x of the GPU time in a SotC frame; the CPU keeps a cached
  // pointer, which the mapped-VRAM alternative does not.
  VkBuffer host_buf = VK_NULL_HANDLE;
  VkDeviceMemory host_mem = VK_NULL_HANDLE;
  bool device_local = false;
  bool readback_pending = false;  // copy recorded, not yet waited on
  bool mirror_current = false;    // map holds the buffer's contents
  u64 size = 0;         // active staged (linear) byte size
  u64 guest_bytes = 0;  // guest footprint (hash + overlap checks)
  u64 hash = 0;         // TexHash of guest content when last in sync
  int last_validated_frame = -1;
  int last_used_frame = -1;
  bool gpu_dirty = false;      // buffer newer than guest memory
  bool pending_batch = false;  // referenced by a batch not yet seen complete
  u64 batch_id = 0;            // that batch (see CsBatchWaitId)
  bool image_staging = false;
  // Buffer memory IS the guest pages (VK_EXT_external_memory_host): no staging
  // copy in, no writeback out. See CsRangeImportGuest.
  bool imported = false;
  u64 imported_base = 0;
  VkDeviceSize imported_offset = 0;  // base - page-aligned import base
  // Staged from a live render-target image rather than guest memory; content
  // changes every frame regardless of the guest bytes, so validity is
  // per-frame (last_rt_frame), never the guest content hash.
  bool rt_sourced = false;
  int last_rt_frame = -1;
  ComputeInfo::Res res;  // writeback needs the full layout description
  // A copy of the guest bytes as they were staged IN, kept so the writeback can
  // tell "the shader wrote this word" from "the shader never touched it".
  // Without it the writeback has to assume the whole range is GPU output and
  // copies the stale snapshot back over anything the CPU changed meanwhile --
  // see CsRangeFlushOne.
  std::vector<u8> shadow;
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
};

bool CsSplitFrameChunk();

// The tiled-truth conversions (defined with the tiling shader, below).
struct TileTable;
// `packed`: narrow texels stay packed on the linear side (a texture upload)
// instead of being expanded to dwords (a shader view).
const TileTable* GetTileTable(const gcn::TextureLayout32& tiled,
                              const gcn::TextureLayout32& linear,
                              bool packed = false);
void RecordImageTiling(VkCommandBuffer c,
                       const TileTable& t,
                       VkBuffer tiled,
                       VkDeviceSize tiled_off,
                       VkDeviceSize tiled_bytes,
                       VkBuffer linear,
                       VkDeviceSize linear_bytes,
                       bool detile,
                       u32 mips = 0,
                       u32 layers = 0);
VkBuffer AcquireScratch(VkDeviceSize bytes);
// A scratch that already holds revision `seq` of the bytes at `base`+`off`
// converted with `table`, or a fresh one (`*fresh` says which).
VkBuffer AcquireView(VkDeviceSize bytes,
                     u64 base,
                     VkDeviceSize off,
                     const TileTable* table,
                     u64 seq,
                     bool* fresh);
// What a scratch holds once the conversion recorded into it has run.
void StampView(VkBuffer view,
               u64 base,
               VkDeviceSize off,
               const TileTable* table,
               u64 seq);
bool RecordTruthImport(CsRange& e, u64 base, VkDeviceSize bytes);
bool CsRangeEnsureBuffer(CsRange& e, VkDeviceSize size);
void CsBatchBeginImpl();
void MarkPending(CsRange& e);
// The linear side of a bridge copy for a truth range (its own buffer holds
// tiled bytes), used only inside the bridge's synchronous submit.
CsRange g_truth_bridge_scratch;
u64 g_cs_scratch_owner = 0;  // the dispatch or bridge being recorded
u64 g_cs_scratch_bytes = 0;

// Buffers a frame command buffer reads: destroyed two BeginFrames after
// retirement, like ReleaseRetiredTextures, once every command buffer that
// could reference them has been fence-waited.
std::vector<std::pair<VkBuffer, VkDeviceMemory>> g_cs_frame_retired;

bool CsFrameReferenced(const CsRange& e) {
  return e.frame_ref >= 0 && e.frame_ref + 2 > g_frame.num;
}

// Evicted range buffers kept by capacity for the next range of that size.
// The guest hands compute a fresh address for its per-frame data every
// frame, so without this every frame was ~50 buffer+memory pairs allocated
// and 20-60 ms of driver time.
struct CsFreeBuffer {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkBuffer host_buf = VK_NULL_HANDLE;
  VkDeviceMemory host_mem = VK_NULL_HANDLE;
  void* map = nullptr;
  VkDeviceSize cap = 0;
  bool device_local = false;
};
std::vector<CsFreeBuffer> g_cs_free;
u64 g_cs_free_bytes = 0, g_cs_recycled_n = 0;

// Move a split range between its host mirror and its VRAM buffer, bracketed by
// barriers against the dispatches on either side. Recorded rather than
// submitted, so the caller decides which command buffer carries it.
void RecordStagingCopy(VkCommandBuffer c,
                       CsRange& e,
                       VkDeviceSize bytes,
                       bool to_device) {
  if (!e.device_local || !e.host_buf || !e.buf || !bytes)
    return;
  if (bytes > e.cap)
    bytes = e.cap;
  VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.buffer = e.buf;
  b.offset = 0;
  b.size = VK_WHOLE_SIZE;
  // TRANSFER on both sides, not just compute and host. A batch stages the same
  // range more than once -- GTA:SA restages some buffers ~170 times in one cs
  // batch -- and a barrier that names only COMPUTE|HOST as its source leaves
  // copy-after-copy on the same buffer unordered. The later copy may then land
  // first and the dispatch reads the OLDER guest snapshot.
  b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                    VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                    VK_ACCESS_TRANSFER_READ_BIT;
  b.dstAccessMask =
      to_device ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(
      c,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT |
          VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &b, 0, nullptr);
  const VkBufferCopy region{0, 0, bytes};
  if (to_device)
    vkCmdCopyBuffer(c, e.host_buf, e.buf, 1, &region);
  else
    vkCmdCopyBuffer(c, e.buf, e.host_buf, 1, &region);
  b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                    VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                    VK_ACCESS_TRANSFER_READ_BIT;
  // The readback direction writes host_buf, and nothing above covers it: the
  // leading barrier is on e.buf, which is that copy's SOURCE.
  b.buffer = to_device ? e.buf : e.host_buf;
  vkCmdPipelineBarrier(
      c, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT |
          VK_PIPELINE_STAGE_TRANSFER_BIT,
      0, 0, nullptr, 1, &b, 0, nullptr);
}

bool SameCsResourceShape(const ComputeInfo::Res& a, const ComputeInfo::Res& b) {
  if (a.image_staging != b.image_staging)
    return false;
  if (!a.image_staging)
    return true;
  // RGBA8 sRGB conversion happens in the shader, so both views stage the
  // same bytes when a mip generator reads sRGB and writes through UNORM.
  const bool rgba8_views = a.dfmt == 10 && b.dfmt == 10 &&
      (a.nfmt == 0 || a.nfmt == 9) && (b.nfmt == 0 || b.nfmt == 9);
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
  VkImage image = VK_NULL_HANDLE;
  u32 w = 0, h = 0;
  u32 elem_bytes = 0;
  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  VkImageLayout submitted_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  bool is_depth = false;
  bool is_stencil = false;
};

// True when `base` names a live target the compute bridges apply to. The
// UNDEFINED-submitted-layout case (target created this frame, no submission
// yet) reports false: there is nothing real to copy either way yet.
// `for_write`: the dispatch produces the surface, so an image nothing has
// rendered into yet is still the one to fill. Its layout anchor may also be
// missing: a parked variant (see ActivateRtVariant) has no submitted-layout
// stamp, but one untouched this frame has nothing recorded against it either,
// so the layout its last frame left it in IS the one on the GPU.
// `prefer_depth`: the resource is a single 32-bit channel, which is what a
// depth surface looks like to a dispatch. One address can hold a colour
// target AND a depth target (Astro Bot's bloom pyramid and its half-res
// scene depth share 0x53a500000 in different passes), and the colour one
// answered first, so the depth downsample landed in the bloom image.
bool FindCsAliasedImage(u64 base,
                        CsAliasedImage& out,
                        bool for_write = false,
                        bool prefer_depth = false) {
  if (prefer_depth) {
    auto depth_it = g_depths.find(base);
    if (depth_it != g_depths.end() && depth_it->second.image) {
      DepthTarget& depth = depth_it->second;
      VkImageLayout anchor = depth.submitted_layout;
      if (anchor == VK_IMAGE_LAYOUT_UNDEFINED && for_write &&
          depth.last_frame != g_frame.num)
        anchor = depth.layout;
      out = {depth.image, depth.w, depth.h, 4, VK_IMAGE_ASPECT_DEPTH_BIT,
             anchor, true, false};
      return out.submitted_layout != VK_IMAGE_LAYOUT_UNDEFINED;
    }
  }
  auto rt_it = g_rts.find(base);
  if (rt_it != g_rts.end()) {
    RTarget& rt = rt_it->second;
    if (!rt.image || rt.is_depth || (!for_write && !rt.ever_rendered))
      return false;
    VkImageLayout anchor = rt.submitted_layout;
    if (anchor == VK_IMAGE_LAYOUT_UNDEFINED && for_write &&
        rt.last_frame != g_frame.num)
      anchor = rt.layout;
    out = {rt.image,
           rt.w,
           rt.h,
           FormatBytes(rt.fmt),
           VK_IMAGE_ASPECT_COLOR_BIT,
           anchor,
            false,
            false};
    return out.submitted_layout != VK_IMAGE_LAYOUT_UNDEFINED;
  }
  auto depth_it = g_depths.find(base);
  if (depth_it != g_depths.end()) {
    DepthTarget& depth = depth_it->second;
    if (!depth.image)
      return false;
    out = {depth.image,
           depth.w,
           depth.h,
           4,  // kDepthFormat == D32_SFLOAT
           VK_IMAGE_ASPECT_DEPTH_BIT,
           depth.submitted_layout,
            true,
            false};
    return out.submitted_layout != VK_IMAGE_LAYOUT_UNDEFINED;
  }
  for (auto& [depth_base, depth] : g_depths) {
    (void)depth_base;
    if (depth.stencil_base != base || !depth.image)
      continue;
    out = {depth.image,
           depth.w,
           depth.h,
           1,
           VK_IMAGE_ASPECT_STENCIL_BIT,
           depth.submitted_stencil_layout,
           false,
           true};
    return out.submitted_layout != VK_IMAGE_LAYOUT_UNDEFINED;
  }
  return false;
}

bool CsAliasedBase(u64 base) {
  if (g_rts.find(base) != g_rts.end() || g_depths.find(base) != g_depths.end())
    return true;
  return std::any_of(g_depths.begin(), g_depths.end(),
                     [base](const auto& entry) {
                       return entry.second.stencil_base == base;
                     });
}

// The frame a live image at this address was last rendered into. Unknown
// addresses report "now", so a caller asking "has it changed since?" re-reads
// rather than trusting a stale copy.
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

VkAccessFlags AliasedImageAccess(const CsAliasedImage& img, VkImageLayout l) {
  if (!img.is_depth && !img.is_stencil)
    return ColorImageAccess(l);
  switch (l) {
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL:
      return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
             VK_ACCESS_SHADER_READ_BIT;
    default:
      return 0;
  }
}

// Aspect-aware ImageBarrier for the bridge's one-shot transfer commands.
// ALL_COMMANDS stages: these command buffers are submitted alone and
// fence-waited, so precision buys nothing.
void AliasedImageBarrier(VkCommandBuffer c,
                         const CsAliasedImage& img,
                         VkImageLayout from,
                         VkImageLayout to,
                         VkAccessFlags src_a,
                         VkAccessFlags dst_a) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img.image;
  b.subresourceRange = {img.aspect, 0, 1, 0, 1};
  b.srcAccessMask = src_a;
  b.dstAccessMask = dst_a;
  vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &b);
}

// How a bridge copy moves texels between a live image and the CS staging
// buffer. The two descriptions of one surface disagree routinely and for
// reasons that are not defects: the IMAGE is sized from CB_COLOR's
// pitch/slice while the CS descriptor carries the surface's own width and
// height, and a target the shader reads as R11G11B10F is staged as four
// floats per texel because that is what StageCsImage produces on the
// guest-memory path. Demanding exact agreement rejected GTA:SA's scene
// target on both counts, so its whole compute post chain read the guest
// bytes under the target -- which draws never write, i.e. black.
struct AliasedCopyPlan {
  u32 w = 0;
  u32 h = 0;
  bool unpack = false;  // image holds packed 11/11/10, staging holds float4
  bool widen = false;   // image holds 8/16-bit texels, staging holds u32
};

bool PlanAliasedCopy(const CsAliasedImage& img,
                     const ComputeInfo::Res& res,
                     const char* dir,
                     AliasedCopyPlan& plan) {
  const u32 want_elem =
      img.is_stencil ? res.elem_bytes : res.stage_elem_bytes;
  const bool unpack = res.dfmt == 6 && !img.is_depth && !img.is_stencil &&
                      img.elem_bytes == 4 && want_elem == 16;
  const bool widen = !img.is_depth && !img.is_stencil &&
                     img.elem_bytes == res.elem_bytes &&
                     (img.elem_bytes == 1 || img.elem_bytes == 2) &&
                     want_elem == 4;
  // A tile row of slack in either direction: an image padded up to its tile
  // height and a descriptor rounded to the surface's own are the same
  // surface, and the copy takes the overlap.
  constexpr u32 kExtentSlack = 8;
  if (res.mip_levels == 1 && res.layers == 1 &&
      img.w + kExtentSlack >= res.width &&
      img.h + kExtentSlack >= res.height &&
      (img.elem_bytes == want_elem || unpack || widen)) {
    plan.w = std::min(img.w, res.width);
    plan.h = std::min(img.h, res.height);
    plan.unpack = unpack;
    plan.widen = widen;
    return true;
  }
  // One line per address and direction. A flat cap spends itself on the level
  // load and then says nothing at all about the steady state.
  static std::unordered_set<u64> warned;
  const u64 key = (res.base << 1) | (dir[0] == 'r' ? 1u : 0u);
  if (warned.size() < 256 && warned.insert(key).second)
    BASE_LOGI("gpuvk",
              "cs {} live {} target {:#x} shape mismatch: image {}x{} {}B vs "
              "cs {}x{} pitch={} mips={} layers={} dfmt={} elem={}/{}B tiling={} -> "
              "falling back to guest memory",
              dir, img.is_depth ? "depth" : img.is_stencil ? "stencil"
                                                          : "color",
              (unsigned long long)res.base, img.w, img.h, img.elem_bytes,
              res.width, res.height, res.pitch, res.mip_levels, res.layers, res.dfmt,
              res.elem_bytes, res.stage_elem_bytes, res.tiling_idx);
  return false;
}

// A host-visible scratch the format-converting bridge copies through: the
// image side is packed at the image's own element size, the staging side
// holds the unpacked floats, and no image<->buffer copy converts between the
// two in one step.
struct CsBridgeScratch {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void* map = nullptr;
  VkDeviceSize cap = 0;
};
CsBridgeScratch g_bridge;

bool EnsureBridgeScratch(VkDeviceSize bytes) {
  if (g_bridge.cap >= bytes)
    return true;
  if (g_bridge.map)
    vkUnmapMemory(g_dev.device, g_bridge.mem);
  if (g_bridge.buf)
    vkDestroyBuffer(g_dev.device, g_bridge.buf, nullptr);
  if (g_bridge.mem)
    vkFreeMemory(g_dev.device, g_bridge.mem, nullptr);
  g_bridge = CsBridgeScratch{};
  const VkDeviceSize cap = (bytes + 0xFFFF) & ~VkDeviceSize(0xFFFF);
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = cap;
  bi.usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &g_bridge.buf) != VK_SUCCESS) {
    g_bridge.buf = VK_NULL_HANDLE;
    return false;
  }
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev.device, g_bridge.buf, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = FindComputeMemoryType(mr.memoryTypeBits);
  if (ai.memoryTypeIndex == UINT32_MAX ||
      vkAllocateMemory(g_dev.device, &ai, nullptr, &g_bridge.mem) !=
          VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, g_bridge.buf, nullptr);
    g_bridge = CsBridgeScratch{};
    return false;
  }
  vkBindBufferMemory(g_dev.device, g_bridge.buf, g_bridge.mem, 0);
  vkMapMemory(g_dev.device, g_bridge.mem, 0, cap, 0, &g_bridge.map);
  g_bridge.cap = cap;
  return true;
}

void CsCopyStaging(CsRange& e, VkDeviceSize bytes, bool to_device);

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
  if (truth && (convert || img.is_stencil))
    return false;
  gcn::TextureLayout32 t_tiled, t_linear;
  const TileTable* table = nullptr;
  if (truth && (!BuildCsImageLayouts(res, t_tiled, t_linear) ||
                !(table = GetTileTable(t_tiled, t_linear)) ||
                !CsRangeEnsureBuffer(g_truth_bridge_scratch, res.size)))
    return false;
  const VkBuffer copy_buf = convert   ? g_bridge.buf
                            : truth   ? g_truth_bridge_scratch.buf
                                      : e.buf;
  auto* scratch = static_cast<u32*>(g_bridge.map);
  if (convert && to_image) {
    for (u32 y = 0; y < plan.h; y++) {
      const u8* row = static_cast<const u8*>(e.map) + level.offset +
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
  }
  // Host-zero any padding an image->buffer copy does not cover (host writes
  // are made available by the submission).
  if (!to_image &&
      (convert || level.offset != 0 || copy_bytes < res.size))
    std::memset(e.map, 0, res.size);
  if (img.is_stencil) {
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

  VkCommandBufferAllocateInfo ca{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = g_dev.pool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  VkCommandBuffer c = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(g_dev.device, &ca, &c) != VK_SUCCESS)
    return false;
  VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(c, &cbi) != VK_SUCCESS) {
    vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &c);
    return false;
  }
  {
    // A frame chunk submitted just before this may still read the buffer.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
  }
  // The dispatch result already lives in VRAM. An ordinary buffer->image copy
  // must read it there, without uploading the identical host readback again.
  // Stencil packing changes the host bytes, and image->buffer staging may have
  // zeroed padding, so those directions still need the host preparation copy.
  // Format conversion uses the separate scratch buffer.
  if (!truth && !convert && (!to_image || img.is_stencil))
    RecordStagingCopy(c, e, e.cap, /*to_device=*/true);
  if (truth && to_image)
    RecordImageTiling(c, *table, e.buf, 0, e.guest_bytes, copy_buf, res.size,
                      /*detile=*/true, 1, 1);
  if (truth && !to_image && (plan.w < res.width || plan.h < res.height)) {
    // The retile below covers the whole mip; texels the copy leaves alone
    // must not carry a previous conversion's bytes into the truth.
    vkCmdFillBuffer(c, copy_buf, level.offset,
                    VkDeviceSize(level.pitch) * level.stored_height *
                        res.stage_elem_bytes,
                    0);
    VkMemoryBarrier cleared{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    cleared.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    cleared.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &cleared, 0,
                         nullptr, 0, nullptr);
  }
  // Chain from -- and restore -- the SUBMITTED layout: this copy executes
  // before the current frame's still-recording barriers, whose oldLayout
  // chain must stay intact.
  const VkImageLayout transfer_layout =
      to_image ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
               : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  const VkAccessFlags transfer_access =
      to_image ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
  if (to_image) {
    // The buffer was last written by the dispatch (already fence-waited) or
    // the host; make those writes available to the transfer.
    VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = copy_buf;
    bb.offset = 0;
    bb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(
        c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &bb, 0, nullptr);
  }
  AliasedImageBarrier(c, img, img.submitted_layout, transfer_layout,
                      AliasedImageAccess(img, img.submitted_layout),
                      transfer_access);
  VkBufferImageCopy copy{};
  copy.bufferOffset = (img.is_stencil || convert) ? 0 : level.offset;
  copy.bufferRowLength = level.pitch;
  copy.imageSubresource = {img.aspect, 0, 0, 1};
  copy.imageExtent = {plan.w, plan.h, 1};
  if (to_image)
    vkCmdCopyBufferToImage(c, copy_buf, img.image, transfer_layout, 1, &copy);
  else
    vkCmdCopyImageToBuffer(c, img.image, transfer_layout, copy_buf, 1, &copy);
  if (truth && !to_image) {
    RecordImageTiling(c, *table, e.buf, 0, e.guest_bytes, copy_buf, res.size,
                      /*detile=*/false, 1, 1);
    e.mirror_current = false;
  }
  AliasedImageBarrier(c, img, transfer_layout, img.submitted_layout,
                      transfer_access,
                      AliasedImageAccess(img, img.submitted_layout));
  if (!to_image) {
    VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = copy_buf;
    bb.offset = 0;
    bb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(
        c, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
        nullptr, 1, &bb, 0, nullptr);
  }
  const VkResult end_result = vkEndCommandBuffer(c);
  base::String where;
  base::FormatTo(where, "bridge base={:#x}", (unsigned long long)res.base);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &c;
  VkResult r = end_result;
  if (r == VK_SUCCESS && !QueueCheck(where.c_str()))
    r = VK_ERROR_DEVICE_LOST;
  if (r == VK_SUCCESS)
    r = vkResetFences(g_dev.device, 1, &g_dev.fence);
  if (r == VK_SUCCESS)
    r = vkQueueSubmit(g_dev.queue, 1, &si, g_dev.fence);
  if (r == VK_SUCCESS)
    r = vkWaitForFences(g_dev.device, 1, &g_dev.fence, VK_TRUE, UINT64_MAX);
  g_out_rt_submits++;
  vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &c);
  if (r != VK_SUCCESS) {
    // A device loss is not a "shape mismatch": the queue is dead and every
    // fallback the caller would keep recording with is already doomed. Latch
    // the failure so the caller stops instead of building more work on a
    // lost device.
    if (r == VK_ERROR_DEVICE_LOST)
      g_cs_failed = true;
    BASE_LOGI("gpuvk", "cs {} bridge copy failed: {} (base={:#x})",
              to_image ? "upload" : "staging", (int)r,
              (unsigned long long)res.base);
    return false;
  }
  if (trace::Recording())
    trace::RecordBridge(to_image ? "upload" : "stage", res.base, img.image,
                        plan.w, plan.h);
  if (img.is_stencil) {
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
            std::memcpy(&expanded,
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

// Stage a CS input whose descriptor points at a live render/depth target from
// the VkImage instead of guest memory. Draws only ever render into the image
// -- the guest bytes under a target stay stale (usually zero), so the
// guest-memory path feeds a compute post chain black (SotC reads its HDR
// scene target AND its 1080p depth buffer this way for the whole
// downsample/tonemap/pyramid cascade). The copy is submitted on the queue and
// waited: it executes after the last submitted frame and before the current
// recording, so it sees the previous frame's completed content -- one frame
// of latency in a post input, not black.
// Returns false (caller falls back to guest staging) when the base is not a
// live target or the shapes disagree.
bool StageCsRangeFromRt(const ComputeInfo::Res& res, CsRange& e) {
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
// aliases must also land in the VkImage, because draws sample the image,
// never guest memory (SotC's exposure/bloom compute writes the adapted scene
// into an RT the tonemap then samples; its depth downsample writes the
// half-res depth pyramid). e.buf already holds the linear pixel data the
// dispatch produced, so upload straight from it. Guest memory was refreshed
// by the caller either way; a shape mismatch just leaves the image stale.
bool UploadCsRangeToRt(u64 base, CsRange& e) {
  CsAliasedImage img;
  if (!e.image_staging)
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
    // compute result is current in guest memory, while the old VkImage can no
    // longer represent it; stop resolving subsequent samples to that image.
    if (!img.is_depth) {
      // Losing this flag makes every later sample of that address fall back to
      // guest memory, which draws never write -- i.e. the target silently reads
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
    if (rt.submitted_layout == VK_IMAGE_LAYOUT_UNDEFINED)
      rt.submitted_layout = img.submitted_layout;
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

std::unordered_map<u64, CsRange> g_cs_ranges;
u64 g_cs_range_bytes = 0;
constexpr u32 kCsDirtyPageShift = 16;
std::unordered_map<u64, std::vector<u64>> g_cs_dirty_pages;
// Bumped whenever compute results land in guest memory (writeback or an
// executed batch of importing dispatches). The draw path's per-frame staging
// caches stamp entries with this and re-copy when it has moved -- a cached
// window of guest bytes is only as fresh as the last compute visibility point.
u64 g_cs_writeback_gen = 1;

u64 RangeEnd(u64 base, u64 bytes) {
  return bytes > UINT64_MAX - base ? UINT64_MAX : base + bytes;
}

void IndexDirtyRange(u64 base, u64 bytes) {
  if (!bytes)
    return;
  const u64 end = RangeEnd(base, bytes);
  for (u64 page = base >> kCsDirtyPageShift;
       page <= (end - 1) >> kCsDirtyPageShift; page++)
    g_cs_dirty_pages[page].push_back(base);
}

void UnindexDirtyRange(u64 base, u64 bytes) {
  if (!bytes)
    return;
  const u64 end = RangeEnd(base, bytes);
  for (u64 page = base >> kCsDirtyPageShift;
       page <= (end - 1) >> kCsDirtyPageShift; page++) {
    auto found = g_cs_dirty_pages.find(page);
    if (found == g_cs_dirty_pages.end())
      continue;
    auto& bases = found->second;
    bases.erase(std::remove(bases.begin(), bases.end(), base), bases.end());
    if (bases.empty())
      g_cs_dirty_pages.erase(found);
  }
}

std::vector<u64> DirtyRangesOverlapping(u64 base,
                                             u64 bytes,
                                             u64 exclude = UINT64_MAX) {
  std::vector<u64> candidates;
  if (!bytes)
    return candidates;
  const u64 end = RangeEnd(base, bytes);
  // Walking the QUERY's pages costs a lookup per 64 KB of it, and a texture is
  // hundreds of pages while the whole dirty index is a handful of entries.
  // Whichever side is smaller gives the same answer, because the filter below
  // rechecks every candidate's overlap exactly -- a superset is safe here.
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
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
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

// Sampled content hash for CS range validation: length + 256 evenly spaced
// 64-byte windows. Reading a whole 16MB image per validation was the point of
// the exercise; a CPU write that dodges every window for a whole frame is a
// risk we accept for the ~50x cheaper check (full TexHash still guards the
// sampled-texture cache).
u64 RangeHash(u64 base, u64 bytes) {
  if (bytes <= 16384)
    return TexHash(base, bytes);
  constexpr u64 kPrime = 1099511628211ull;
  u64 h = 1469598103934665603ull ^ (bytes * kPrime);
  const u64 step = (bytes - 64) / 255;
  for (u32 i = 0; i < 256; i++) {
    u64 w[8];
    std::memcpy(w, reinterpret_cast<const void*>(base + i * step), 64);
    for (int j = 0; j < 8; j++)
      h = (h ^ w[j]) * kPrime;
  }
  return h;
}

// DELTA_GPU_CSIMPORT: back the range's buffer with the GUEST PAGES themselves
// instead of a staging copy. Everything a compute dispatch reads then costs
// nothing to get in, and anything it writes is already in guest memory when the
// dispatch retires -- SotC spends ~550 ms of a 2 fps frame on that copy in and
// back out. Needs the guest range page-aligned to the driver's import
// granularity; callers fall back to staging when this returns false.
bool CsRangeImportGuest(CsRange& e, u64 base, VkDeviceSize size) {
  const size_t align = g_dev.host_import_align;
  if (!g_dev.host_import_available || !align)
    return false;
  const u64 lo = base & ~(u64)(align - 1);
  const u64 hi = (base + size + align - 1) & ~(u64)(align - 1);
  // The shader indexes from the binding, so an unaligned base is fine as long
  // as the descriptor can carry the difference.
  const VkDeviceSize off = base - lo;
  if (off % g_dev.storage_buffer_offset_align)
    return false;
  VkMemoryHostPointerPropertiesEXT hpp{
      VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
  auto fn = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(
      g_dev.device, "vkGetMemoryHostPointerPropertiesEXT");
  if (!fn ||
      fn(g_dev.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
         reinterpret_cast<void*>(lo), &hpp) != VK_SUCCESS ||
      !hpp.memoryTypeBits)
    return false;
  VkExternalMemoryBufferCreateInfo ebi{
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
  ebi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.pNext = &ebi;
  bi.size = hi - lo;
  bi.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkBuffer buf = VK_NULL_HANDLE;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &buf) != VK_SUCCESS)
    return false;
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev.device, buf, &mr);
  const u32 bits = mr.memoryTypeBits & hpp.memoryTypeBits;
  if (!bits) {
    vkDestroyBuffer(g_dev.device, buf, nullptr);
    return false;
  }
  VkImportMemoryHostPointerInfoEXT ihp{
      VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
  ihp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  ihp.pHostPointer = reinterpret_cast<void*>(lo);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.pNext = &ihp;
  ai.allocationSize = hi - lo;
  ai.memoryTypeIndex = (u32)__builtin_ctz(bits);
  VkDeviceMemory mem = VK_NULL_HANDLE;
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &mem) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, buf, nullptr);
    return false;
  }
  if (vkBindBufferMemory(g_dev.device, buf, mem, 0) != VK_SUCCESS) {
    vkFreeMemory(g_dev.device, mem, nullptr);
    vkDestroyBuffer(g_dev.device, buf, nullptr);
    return false;
  }
  e.buf = buf;
  e.mem = mem;
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
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = GdsBuffer::kBytes;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &g_gds.buf) != VK_SUCCESS) {
    g_gds.buf = VK_NULL_HANDLE;
    return false;
  }
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev.device, g_gds.buf, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = FindComputeMemoryType(mr.memoryTypeBits);
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &g_gds.mem) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, g_gds.buf, nullptr);
    g_gds.buf = VK_NULL_HANDLE;
    g_gds.mem = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(g_dev.device, g_gds.buf, g_gds.mem, 0);
  if (vkMapMemory(g_dev.device, g_gds.mem, 0, GdsBuffer::kBytes, 0,
                   &g_gds.map) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, g_gds.buf, nullptr);
    vkFreeMemory(g_dev.device, g_gds.mem, nullptr);
    g_gds = {};
    return false;
  }
  std::memset(g_gds.map, 0, GdsBuffer::kBytes);
  return true;
}

bool CsRangeEnsureBuffer(CsRange& e, VkDeviceSize size) {
  if (e.buf && e.cap >= size)
    return true;
  if (e.map && !e.imported) {
    vkUnmapMemory(g_dev.device, e.device_local ? e.host_mem : e.mem);
  }
  e.map = nullptr;
  if (CsFrameReferenced(e) || e.pending_batch) {
    g_cs_frame_retired.emplace_back(e.buf, e.mem);
    g_cs_frame_retired.emplace_back(e.host_buf, e.host_mem);
    e.frame_ref = -1;
    e.pending_batch = false;
  } else {
    if (e.buf)
      vkDestroyBuffer(g_dev.device, e.buf, nullptr);
    if (e.mem)
      vkFreeMemory(g_dev.device, e.mem, nullptr);
    if (e.host_buf)
      vkDestroyBuffer(g_dev.device, e.host_buf, nullptr);
    if (e.host_mem)
      vkFreeMemory(g_dev.device, e.host_mem, nullptr);
  }
  e.buf = VK_NULL_HANDLE;
  e.mem = VK_NULL_HANDLE;
  e.host_buf = VK_NULL_HANDLE;
  e.host_mem = VK_NULL_HANDLE;
  e.device_local = false;
  e.imported = false;
  if (!e.imported)
    g_cs_range_bytes -= e.cap;
  e.cap = 0;
  VkDeviceSize cap = (size + 0xFFFF) & ~VkDeviceSize(0xFFFF);
  for (size_t i = g_cs_free.size(); i-- > 0;) {
    const CsFreeBuffer& f = g_cs_free[i];
    if (f.cap != cap || f.device_local != static_cast<bool>(kCsVram))
      continue;
    e.buf = f.buf;
    e.mem = f.mem;
    e.host_buf = f.host_buf;
    e.host_mem = f.host_mem;
    e.map = f.map;
    e.device_local = f.device_local;
    e.cap = cap;
    g_cs_free_bytes -= cap;
    g_cs_free.erase(g_cs_free.begin() + i);
    g_cs_range_bytes += cap;
    g_cs_recycled_n++;
    return true;
  }
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = cap;
  // TRANSFER_DST: RT-backed inputs are staged by an image->buffer copy on the
  // queue (StageCsRangeFromRt) instead of a CPU memcpy from guest memory.
  bi.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (kCsVram)
    bi.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;  // readback of results
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &e.buf) != VK_SUCCESS) {
    e.buf = VK_NULL_HANDLE;
    return false;
  }
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev.device, e.buf, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  const u32 device_type =
      kCsVram ? FindDeviceMemoryType(mr.memoryTypeBits) : UINT32_MAX;
  ai.memoryTypeIndex = device_type != UINT32_MAX
                           ? device_type
                           : FindComputeMemoryType(mr.memoryTypeBits);
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &e.mem) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, e.buf, nullptr);
    e.buf = VK_NULL_HANDLE;
    e.mem = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(g_dev.device, e.buf, e.mem, 0);
  // VRAM cannot be mapped usefully (uncached reads are ~100 MB/s), so the CPU
  // side gets its own host-cached buffer and the two are joined by DMA.
  if (device_type != UINT32_MAX) {
    VkBufferCreateInfo hi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    hi.size = cap;
    hi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryRequirements hmr;
    if (vkCreateBuffer(g_dev.device, &hi, nullptr, &e.host_buf) != VK_SUCCESS) {
      e.host_buf = VK_NULL_HANDLE;
      return false;
    }
    vkGetBufferMemoryRequirements(g_dev.device, e.host_buf, &hmr);
    VkMemoryAllocateInfo hai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    hai.allocationSize = hmr.size;
    hai.memoryTypeIndex = FindComputeMemoryType(hmr.memoryTypeBits);
    if (vkAllocateMemory(g_dev.device, &hai, nullptr, &e.host_mem) !=
        VK_SUCCESS) {
      vkDestroyBuffer(g_dev.device, e.host_buf, nullptr);
      e.host_buf = VK_NULL_HANDLE;
      e.host_mem = VK_NULL_HANDLE;
      return false;
    }
    vkBindBufferMemory(g_dev.device, e.host_buf, e.host_mem, 0);
    vkMapMemory(g_dev.device, e.host_mem, 0, cap, 0, &e.map);
    e.device_local = true;
  } else {
    vkMapMemory(g_dev.device, e.mem, 0, cap, 0, &e.map);
  }
  e.cap = cap;
  g_cs_range_bytes += cap;
  return true;
}

// Buffers still referenced by the open dispatch batch, freed once its fence
// signals. See CsRangeRename.
std::vector<std::pair<VkBuffer, VkDeviceMemory>> g_cs_retired;

// DELTA_GPU_CSRENAME: a CPU write into a range the open batch already reads has
// to flush that batch and wait on the GPU -- SotC's streaming does that 37 times
// a frame, and staging then costs 272 ms of a 2 fps frame. Giving the range a
// FRESH buffer and retiring the old one until the batch completes takes that to
// 24 ms: the recorded descriptors keep reading the contents they were built
// against, and this dispatch binds the new one. Only safe while nothing is
// waiting to read results back OUT of the old buffer, hence the caller's
// gpu_dirty/written check. Default OFF: it moves cost into the writeback rather
// than removing it (SotC gains ~7% overall), and no title that currently
// renders through the compute path could be used to gate the change.
bool CsRangeRename(CsRange& e, VkDeviceSize size) {
  if (e.map) {
    vkUnmapMemory(g_dev.device, e.device_local ? e.host_mem : e.mem);
    e.map = nullptr;
  }
  if (e.buf || e.mem)
    g_cs_frame_retired.emplace_back(e.buf, e.mem);
  if (e.host_buf || e.host_mem)
    g_cs_frame_retired.emplace_back(e.host_buf, e.host_mem);
  e.frame_ref = -1;
  e.pending_batch = false;
  g_cs_range_bytes -= e.cap;
  e.buf = VK_NULL_HANDLE;
  e.mem = VK_NULL_HANDLE;
  e.host_buf = VK_NULL_HANDLE;
  e.host_mem = VK_NULL_HANDLE;
  e.device_local = false;
  e.cap = 0;
  return CsRangeEnsureBuffer(e, size);
}

void DropTiledImport(u64 base);

void CsRangeDestroy(CsRange& e) {
  if (e.res.base)
    DropTiledImport(e.res.base);
  if (!e.imported && !CsFrameReferenced(e) && !e.pending_batch && e.buf &&
      e.cap && g_cs_free_bytes + e.cap <= (256ull << 20)) {
    g_cs_free.push_back({e.buf, e.mem, e.host_buf, e.host_mem, e.map, e.cap,
                         e.device_local});
    g_cs_free_bytes += e.cap;
    g_cs_range_bytes -= e.cap;
    e = CsRange{};
    return;
  }
  if (e.map && !e.imported)
    vkUnmapMemory(g_dev.device, e.device_local ? e.host_mem : e.mem);
  if (CsFrameReferenced(e) || e.pending_batch) {
    g_cs_frame_retired.emplace_back(e.buf, e.mem);
    g_cs_frame_retired.emplace_back(e.host_buf, e.host_mem);
  } else {
    if (e.buf)
      vkDestroyBuffer(g_dev.device, e.buf, nullptr);
    if (e.mem)
      vkFreeMemory(g_dev.device, e.mem, nullptr);
    if (e.host_buf)
      vkDestroyBuffer(g_dev.device, e.host_buf, nullptr);
    if (e.host_mem)
      vkFreeMemory(g_dev.device, e.host_mem, nullptr);
  }
  g_cs_range_bytes -= e.cap;
  e = CsRange{};
}

#ifdef DELTA_HAVE_SPIRV_BACKEND
// A guest range mapped into the device so the GPU addresses it in place.
struct ImportedRange {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;  // base - the alignment floor the import starts at
  u64 bytes = 0;            // usable bytes from `offset`
};
std::unordered_map<u64, ImportedRange> g_tiled_imports;

// The tiled side of a layout conversion is guest memory the CPU otherwise
// copies twice a frame: in on the way to the shader, out again on the way back.
// Importing those pages lets the tiling shader read and write them where they
// already live, which is the whole of `host-in` and `host-out` in the
// DELTA_GPU_CSSYNC report. Cached: the allocate/create pair is far too
// expensive to repeat per dispatch.
const ImportedRange* ImportTiledRange(u64 base, u64 bytes) {
  const size_t align = g_dev.host_import_align;
  if (!g_dev.host_import_available || !align || !base || !bytes)
    return nullptr;
  auto it = g_tiled_imports.find(base);
  if (it != g_tiled_imports.end()) {
    // The import pins host pages by address. If the title has since unmapped
    // that surface, the mapping still points at pages that now belong to
    // something else, and a conversion would write through it -- so confirm
    // the range before every reuse, not only when it is created.
    if (it->second.bytes >= bytes && gpu::IsReadableRangeCached(base, bytes))
      return &it->second;
    // A bigger view of the same base needs a bigger mapping.
    if (it->second.buf)
      vkDestroyBuffer(g_dev.device, it->second.buf, nullptr);
    if (it->second.mem)
      vkFreeMemory(g_dev.device, it->second.mem, nullptr);
    g_tiled_imports.erase(it);
  }
  // Unbounded caching would pin every surface a title ever converts; these are
  // a handful of large render targets, so a small cap is plenty.
  if (g_tiled_imports.size() >= 64)
    return nullptr;
  if (!gpu::IsReadableRange(base, bytes))
    return nullptr;
  const u64 lo = base & ~u64(align - 1);
  const u64 hi = (base + bytes + align - 1) & ~u64(align - 1);
  const VkDeviceSize off = base - lo;
  if (off % g_dev.storage_buffer_offset_align)
    return nullptr;
  auto fn = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(
      g_dev.device, "vkGetMemoryHostPointerPropertiesEXT");
  VkMemoryHostPointerPropertiesEXT hpp{
      VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
  if (!fn ||
      fn(g_dev.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
         reinterpret_cast<void*>(lo), &hpp) != VK_SUCCESS ||
      !hpp.memoryTypeBits)
    return nullptr;
  VkExternalMemoryBufferCreateInfo ebi{
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
  ebi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.pNext = &ebi;
  bi.size = hi - lo;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkBuffer buf = VK_NULL_HANDLE;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &buf) != VK_SUCCESS)
    return nullptr;
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev.device, buf, &mr);
  const u32 bits = mr.memoryTypeBits & hpp.memoryTypeBits;
  if (!bits) {
    vkDestroyBuffer(g_dev.device, buf, nullptr);
    return nullptr;
  }
  // The guest CPU and our own readers touch these pages without any map/unmap
  // of their own, so nothing here ever issues a vkInvalidateMappedMemoryRanges:
  // a non-coherent type would hand them stale bytes after every conversion.
  // Take the first coherent type and decline the import if there is none.
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(g_dev.phys, &mem_props);
  u32 type = UINT32_MAX;
  for (u32 i = 0; i < mem_props.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags &
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      type = i;
      break;
    }
  if (type == UINT32_MAX) {
    vkDestroyBuffer(g_dev.device, buf, nullptr);
    return nullptr;
  }
  VkImportMemoryHostPointerInfoEXT ihp{
      VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
  ihp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  ihp.pHostPointer = reinterpret_cast<void*>(lo);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.pNext = &ihp;
  ai.allocationSize = hi - lo;
  ai.memoryTypeIndex = type;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &mem) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, buf, nullptr);
    return nullptr;
  }
  if (vkBindBufferMemory(g_dev.device, buf, mem, 0) != VK_SUCCESS) {
    vkFreeMemory(g_dev.device, mem, nullptr);
    vkDestroyBuffer(g_dev.device, buf, nullptr);
    return nullptr;
  }
  ImportedRange& out = g_tiled_imports[base];
  out.buf = buf;
  out.mem = mem;
  out.offset = off;
  out.bytes = hi - lo - off;
  return &out;
}


struct TileTable {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkDeviceSize bytes = 0;
  std::vector<gcn::ImageTilingParams> params;  // per mip; `detile` patched
  u32 layers = 0;
  bool expanded = false;
};
std::unordered_map<u64, TileTable> g_tile_tables;

struct TilingState {
  CsPipe pipeline;
  CsPipe push;  // the same shader, descriptors pushed into the command buffer
  CsRange buffers[3];
  VkDescriptorPool pool = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
};
TilingState g_tiling;

bool InitTiling() {
  auto& s = g_tiling;
  if (s.fence)
    return true;
  static bool attempted = false;
  if (attempted)
    return false;
  attempted = true;
  const auto code = gcn::BuildImageTilingShader();
  if (code.empty())
    return false;
  VkDescriptorSetLayoutBinding bindings[3];
  for (u32 i = 0; i < 3; i++)
    bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo sl{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  sl.bindingCount = 3;
  sl.pBindings = bindings;
  VKOK(vkCreateDescriptorSetLayout(g_dev.device, &sl, nullptr,
                                   &s.pipeline.set_layout));
  VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(gcn::ImageTilingParams)};
  VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pl.setLayoutCount = 1;
  pl.pSetLayouts = &s.pipeline.set_layout;
  pl.pushConstantRangeCount = 1;
  pl.pPushConstantRanges = &push;
  VKOK(vkCreatePipelineLayout(g_dev.device, &pl, nullptr, &s.pipeline.layout));
  const VkShaderModule shader = MakeModuleVec(code);
  VkComputePipelineCreateInfo pi{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pi.stage.module = shader;
  pi.stage.pName = "main";
  pi.layout = s.pipeline.layout;
  const VkResult result = vkCreateComputePipelines(
      g_dev.device, g_dev.pipeline_cache, 1, &pi, nullptr, &s.pipeline.pipe);
  if (result == VK_SUCCESS && g_dev.push_descriptor) {
    VkDescriptorSetLayoutCreateInfo psl = sl;
    psl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    VkPipelineLayoutCreateInfo ppl = pl;
    ppl.pSetLayouts = &s.push.set_layout;
    VkComputePipelineCreateInfo ppi = pi;
    if (vkCreateDescriptorSetLayout(g_dev.device, &psl, nullptr,
                                    &s.push.set_layout) != VK_SUCCESS ||
        vkCreatePipelineLayout(g_dev.device, &ppl, nullptr, &s.push.layout) !=
            VK_SUCCESS)
      s.push.pipe = VK_NULL_HANDLE;
    else {
      ppi.layout = s.push.layout;
      if (vkCreateComputePipelines(g_dev.device, g_dev.pipeline_cache, 1, &ppi,
                                   nullptr, &s.push.pipe) != VK_SUCCESS)
        s.push.pipe = VK_NULL_HANDLE;
    }
  }
  vkDestroyShaderModule(g_dev.device, shader, nullptr);
  VKOK(result);
  VkDescriptorPoolSize count{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo pool{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  pool.maxSets = 1;
  pool.poolSizeCount = 1;
  pool.pPoolSizes = &count;
  VKOK(vkCreateDescriptorPool(g_dev.device, &pool, nullptr, &s.pool));
  VkDescriptorSetAllocateInfo ds{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ds.descriptorPool = s.pool;
  ds.descriptorSetCount = 1;
  ds.pSetLayouts = &s.pipeline.set_layout;
  VKOK(vkAllocateDescriptorSets(g_dev.device, &ds, &s.set));
  VkCommandBufferAllocateInfo cb{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cb.commandPool = g_dev.pool;
  cb.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cb.commandBufferCount = 1;
  VKOK(vkAllocateCommandBuffers(g_dev.device, &cb, &s.cmd));
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VKOK(vkCreateFence(g_dev.device, &fi, nullptr, &s.fence));
  return true;
}

void DropTiledImportImpl(u64 base) {
  auto it = g_tiled_imports.find(base);
  if (it == g_tiled_imports.end())
    return;
  if (it->second.buf)
    vkDestroyBuffer(g_dev.device, it->second.buf, nullptr);
  if (it->second.mem)
    vkFreeMemory(g_dev.device, it->second.mem, nullptr);
  g_tiled_imports.erase(it);
}

bool ConvertGfx10ImageImpl(const gcn::TextureLayout32& tiled,
                           const gcn::TextureLayout32& linear,
                           const void* src,
                           void* dst,
                           bool detile,
                           VkBuffer linear_buffer = VK_NULL_HANDLE) {
  const bool expanded =
      (tiled.elem_bytes == 1 || tiled.elem_bytes == 2) && linear.elem_bytes == 4;
  if (!g_dev.ready || g_cs_failed ||
      (!expanded && (tiled.elem_bytes < 4 ||
                     tiled.elem_bytes != linear.elem_bytes)) ||
      !tiled.layer_stride ||
      linear.tiling_idx != 8 || tiled.layers != linear.layers ||
      tiled.mip_levels != linear.mip_levels ||
      tiled.size > g_dev.max_storage_buffer_range ||
      linear.size > g_dev.max_storage_buffer_range || tiled.size > UINT32_MAX ||
      linear.size > UINT32_MAX)
    return false;
  std::vector<u32> table, terms;
  std::vector<gcn::ImageTilingParams> params;
  for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& t = tiled.mips[mip];
    const auto& l = linear.mips[mip];
    u32 mask;
    if (t.width != l.width || t.height != l.height ||
        !gcn::BuildGfx10AddressTable(tiled, mip, terms, mask))
      return false;
    params.push_back({t.width, t.height, linear.elem_bytes / 4,
                      static_cast<u32>(table.size()), mask,
                      static_cast<u32>(t.offset),
                      static_cast<u32>(tiled.layer_stride),
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
  VkDescriptorBufferInfo infos[3];
  VkWriteDescriptorSet writes[3];
  for (u32 i = 0; i < 3; i++) {
    const bool bound = i == 1 && linear_buffer;
    if (!bound &&
        (!CsRangeEnsureBuffer(s.buffers[i], sizes[i]) || !s.buffers[i].map))
      return false;
    infos[i] = {bound ? linear_buffer : s.buffers[i].buf, 0, sizes[i]};
    writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[i].dstSet = s.set;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(g_dev.device, 3, writes, 0, nullptr);
  const u32 input = detile ? 0 : 1, output = detile ? 1 : 0;
  // Whichever half is bound to memory the shader reaches directly needs no
  // copy: the linear side is the caller's own buffer, the tiled side is guest
  // memory.
  const bool input_bound = (input == 1 && linear_buffer) || (input == 0 && dma_tiled);
  const bool output_bound =
      (output == 1 && linear_buffer) || (output == 0 && dma_tiled);
  const u64 _t_hin = NowNs();
  if (!input_bound) {
    ParallelCopy(s.buffers[input].map, src, sizes[input]);
    g_tile_hin_bytes += sizes[input];
  }
  if (!output_bound && !s.buffers[output].device_local)
    std::memset(s.buffers[output].map, 0, sizes[output]);
  std::memcpy(s.buffers[2].map, table.data(), sizes[2]);
  g_tile_hin_ns += NowNs() - _t_hin;
  VKOK(vkResetCommandBuffer(s.cmd, 0));
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VKOK(vkBeginCommandBuffer(s.cmd, &begin));
  VkMemoryBarrier reuse{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  reuse.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  reuse.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &reuse, 0, nullptr,
                       0, nullptr);
  if (dma_tiled) {
    const VkBufferCopy in_copy{imported->offset, 0, tiled.size};
    vkCmdCopyBuffer(s.cmd, imported->buf, s.buffers[0].buf, 1, &in_copy);
  } else if (!input_bound) {
    RecordStagingCopy(s.cmd, s.buffers[input], sizes[input], true);
  }
  RecordStagingCopy(s.cmd, s.buffers[2], sizes[2], true);
  // A tiled destination seeded from guest memory must not be cleared after it.
  if (!(output == 0 && dma_tiled) &&
      (output_bound || s.buffers[output].device_local))
    vkCmdFillBuffer(s.cmd, infos[output].buffer, infos[output].offset,
                    sizes[output], 0);
  VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(
      s.cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0,
      nullptr);
  vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeline.pipe);
  vkCmdBindDescriptorSets(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          s.pipeline.layout, 0, 1, &s.set, 0, nullptr);
  for (const auto& p : params) {
    vkCmdPushConstants(s.cmd, s.pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(p), &p);
    vkCmdDispatch(s.cmd, (p.width * p.words + 63) / 64, p.height, tiled.layers);
  }
  if (dma_tiled && !detile) {
    VkMemoryBarrier to_dma{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    to_dma.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    to_dma.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &to_dma, 0,
                         nullptr, 0, nullptr);
    const VkBufferCopy out_copy{0, imported->offset, tiled.size};
    vkCmdCopyBuffer(s.cmd, s.buffers[0].buf, imported->buf, 1, &out_copy);
    g_tile_hout_bytes += tiled.size;
  } else if (!output_bound) {
    RecordStagingCopy(s.cmd, s.buffers[output], sizes[output], false);
  }
  VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  after.srcAccessMask =
      VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT |
                        VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(
      s.cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
      &after, 0, nullptr, 0, nullptr);
  VKOK(vkEndCommandBuffer(s.cmd));
  VKOK(vkResetFences(g_dev.device, 1, &s.fence));
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &s.cmd;
  const u64 _t_sync = NowNs();
  VkResult result = vkQueueSubmit(g_dev.queue, 1, &submit, s.fence);
  if (result == VK_SUCCESS)
    result = vkWaitForFences(g_dev.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
  g_tile_sync_ns += NowNs() - _t_sync;
  if (result != VK_SUCCESS) {
    BASE_LOGI("gpuvk", "image conversion failed: {}", static_cast<int>(result));
    g_cs_failed = true;
    return false;
  }
  const u64 _t_hout = NowNs();
  if (detile && !linear_buffer) {
    ParallelCopy(dst, s.buffers[output].map, sizes[output]);
    g_tile_hout_bytes += sizes[output];
  } else if (!detile && !dma_tiled) {
    gcn::CopyGfx10ImageContents(tiled, s.buffers[output].map, dst);
    g_tile_hout_bytes += tiled.size;
  }
  g_tile_hout_ns += NowNs() - _t_hout;
  return true;
}

// ---- Tiled-truth ranges (DELTA_GPU_CS_TRUTH) ----

u32 MemoryTypeWith(u32 type_bits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g_dev.phys, &mp);
  for (u32 i = 0; i < mp.memoryTypeCount; i++)
    if ((type_bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  return UINT32_MAX;
}

u64 TileTableKey(const gcn::TextureLayout32& t,
                 const gcn::TextureLayout32& l,
                 bool packed) {
  u64 h = 1469598103934665603ull;
  const u64 words[] = {t.tiling_idx,   t.mips[0].width, t.mips[0].height,
                       t.mips[0].pitch, t.layers,        t.mip_levels,
                       t.elem_bytes,   t.size,          l.elem_bytes,
                       l.size,         l.mips[0].pitch, packed ? 1u : 0u};
  for (u64 w : words)
    h = (h ^ w) * 1099511628211ull;
  return h;
}

// The address terms of one surface layout, built once. The shader reads
// three of them per texel, so they live where it can reach them quickly.
const TileTable* GetTileTable(const gcn::TextureLayout32& tiled,
                              const gcn::TextureLayout32& linear,
                              bool packed) {
  const bool expanded = !packed && (tiled.elem_bytes == 1 ||
                                    tiled.elem_bytes == 2) &&
                        linear.elem_bytes == 4;
  if (packed && (tiled.elem_bytes >= 4 || tiled.elem_bytes != linear.elem_bytes))
    return nullptr;
  if (!InitTiling() || !g_tiling.push.pipe ||
      (!expanded && !packed &&
       (tiled.elem_bytes < 4 || tiled.elem_bytes != linear.elem_bytes)) ||
      !tiled.layer_stride || linear.tiling_idx != 8 ||
      tiled.layers != linear.layers ||
      tiled.mip_levels != linear.mip_levels ||
      tiled.size > g_dev.max_storage_buffer_range ||
      linear.size > g_dev.max_storage_buffer_range || tiled.size > UINT32_MAX ||
      linear.size > UINT32_MAX)
    return nullptr;
  const u64 key = TileTableKey(tiled, linear, packed);
  auto it = g_tile_tables.find(key);
  if (it != g_tile_tables.end())
    return it->second.buf ? &it->second : nullptr;
  if (g_tile_tables.size() >= 1024)
    return nullptr;
  TileTable& t = g_tile_tables[key];  // a failed build stays empty: no retry
  std::vector<u32> table, terms;
  for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& tm = tiled.mips[mip];
    const auto& lm = linear.mips[mip];
    u32 mask;
    if (tm.width != lm.width || tm.height != lm.height ||
        !gcn::BuildGfx10AddressTable(tiled, mip, terms, mask))
      return nullptr;
    const u32 words = packed ? 1u : linear.elem_bytes / 4;
    t.params.push_back(
        {tm.width, tm.height, words, static_cast<u32>(table.size()), mask,
         static_cast<u32>(tm.offset), static_cast<u32>(tiled.layer_stride),
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
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = t.bytes;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &t.buf) != VK_SUCCESS) {
    t.buf = VK_NULL_HANDLE;
    return nullptr;
  }
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev.device, t.buf, &mr);
  // Written once by the host, read by every conversion: device-local
  // host-visible memory where the heap offers it, plain host memory otherwise.
  u32 type = MemoryTypeWith(mr.memoryTypeBits,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (type == UINT32_MAX)
    type = MemoryTypeWith(mr.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = type;
  void* map = nullptr;
  if (type == UINT32_MAX ||
      vkAllocateMemory(g_dev.device, &ai, nullptr, &t.mem) != VK_SUCCESS ||
      vkBindBufferMemory(g_dev.device, t.buf, t.mem, 0) != VK_SUCCESS ||
      vkMapMemory(g_dev.device, t.mem, 0, t.bytes, 0, &map) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, t.buf, nullptr);
    t.buf = VK_NULL_HANDLE;
    if (t.mem) {
      vkFreeMemory(g_dev.device, t.mem, nullptr);
      t.mem = VK_NULL_HANDLE;
    }
    return nullptr;
  }
  std::memcpy(map, table.data(), t.bytes);
  vkUnmapMemory(g_dev.device, t.mem);
  return &t;
}

// Record one layout conversion between two device buffers: `tiled` holds the
// guest bytes (a range's truth), `linear` the shader's view of them. `mips`
// and `layers` limit the pass to the leading levels and slices (a bridge
// copies mip 0 of layer 0 only); 0 means all of them.
void RecordImageTiling(VkCommandBuffer c,
                       const TileTable& t,
                       VkBuffer tiled,
                       VkDeviceSize tiled_off,
                       VkDeviceSize tiled_bytes,
                       VkBuffer linear,
                       VkDeviceSize linear_bytes,
                       bool detile,
                       u32 mips,
                       u32 layers) {
  auto& s = g_tiling;
  const u32 mip_count =
      mips ? std::min<u32>(mips, t.params.size()) : t.params.size();
  const u32 layer_count = layers ? std::min(layers, t.layers) : t.layers;
  VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  before.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                         VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
  before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                         VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(c,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT |
                           VK_PIPELINE_STAGE_HOST_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 1, &before, 0, nullptr, 0, nullptr);
  vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, s.push.pipe);
  const VkDescriptorBufferInfo infos[3] = {{tiled, tiled_off, tiled_bytes},
                                           {linear, 0, linear_bytes},
                                           {t.buf, 0, t.bytes}};
  VkWriteDescriptorSet writes[3];
  for (u32 i = 0; i < 3; i++) {
    writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  g_dev.push_descriptor_set(c, VK_PIPELINE_BIND_POINT_COMPUTE, s.push.layout,
                            0, 3, writes);
  for (u32 m = 0; m < mip_count; m++) {
    gcn::ImageTilingParams p = t.params[m];
    p.detile = detile ? 1u : 0u;
    vkCmdPushConstants(c, s.push.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(p), &p);
    vkCmdDispatch(c, (p.width * p.words + 63) / 64, p.height, layer_count);
  }
  VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                        VK_ACCESS_TRANSFER_READ_BIT |
                        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT |
                           VK_PIPELINE_STAGE_HOST_BIT,
                       0, 1, &after, 0, nullptr, 0, nullptr);
}

// Transient detiled views. A scratch is reused within the frame as soon as
// the pass that consumed it has been recorded -- the conversions' own
// barriers order the accesses -- and never by the dispatch being recorded.
// A scratch that goes cold is retired through the frame ring.
struct CsScratch {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkDeviceSize cap = 0;
  u64 owner = 0;
  int last_frame = -1;
  // The view it holds: which bytes, converted how, at which revision. A
  // later binding of the same unchanged truth reuses it as it is.
  u64 view_base = 0;
  VkDeviceSize view_off = 0;
  const TileTable* view_table = nullptr;
  u64 view_seq = 0;
};
std::vector<CsScratch> g_cs_scratch;
u64 g_truth_view_hits = 0;

VkBuffer AcquireView(VkDeviceSize bytes,
                     u64 base,
                     VkDeviceSize off,
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

void StampView(VkBuffer view,
               u64 base,
               VkDeviceSize off,
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

VkBuffer AcquireScratch(VkDeviceSize bytes) {
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
  s.cap = (bytes + 0x3FFFFF) & ~VkDeviceSize(0x3FFFFF);
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
    g_cs_frame_retired.emplace_back(g_cs_scratch[coldest].buf,
                                    g_cs_scratch[coldest].mem);
    g_cs_scratch_bytes -= g_cs_scratch[coldest].cap;
    g_cs_scratch.erase(g_cs_scratch.begin() + coldest);
  }
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = s.cap;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &s.buf) != VK_SUCCESS)
    return VK_NULL_HANDLE;
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev.device, s.buf, &mr);
  u32 type = FindDeviceMemoryType(mr.memoryTypeBits);
  if (type == UINT32_MAX)
    type = FindComputeMemoryType(mr.memoryTypeBits);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = type;
  if (type == UINT32_MAX ||
      vkAllocateMemory(g_dev.device, &ai, nullptr, &s.mem) != VK_SUCCESS ||
      vkBindBufferMemory(g_dev.device, s.buf, s.mem, 0) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, s.buf, nullptr);
    if (s.mem)
      vkFreeMemory(g_dev.device, s.mem, nullptr);
    return VK_NULL_HANDLE;
  }
  s.owner = g_cs_scratch_owner;
  s.last_frame = g_frame.num;
  NameObject(VK_OBJECT_TYPE_BUFFER, (u64)s.buf, "cs scratch %llu KB",
             (unsigned long long)(s.cap / 1024));
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
bool RecordTruthImport(CsRange& e, u64 base, VkDeviceSize bytes) {
  const ImportedRange* imp = ImportTiledRange(base, bytes);
  if (!imp || !e.buf)
    return false;
  CsBatchBeginImpl();
  VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.buffer = e.buf;
  b.offset = 0;
  b.size = VK_WHOLE_SIZE;
  b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                    VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(g_cs_cmd,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &b, 0,
                       nullptr);
  const VkBufferCopy region{imp->offset, 0, bytes};
  vkCmdCopyBuffer(g_cs_cmd, imp->buf, e.buf, 1, &region);
  b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                    VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(g_cs_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 0, nullptr, 1, &b, 0, nullptr);
  MarkPending(e);
  e.mirror_current = false;
  g_truth_import_n++;
  return true;
}

bool TruthAvailable() {
  return InitTiling() && g_tiling.push.pipe != VK_NULL_HANDLE;
}
#else
const TileTable* GetTileTable(const gcn::TextureLayout32&,
                              const gcn::TextureLayout32&,
                              bool) {
  return nullptr;
}
VkBuffer AcquireView(VkDeviceSize, u64, VkDeviceSize, const TileTable*, u64,
                     bool* fresh) {
  *fresh = true;
  return VK_NULL_HANDLE;
}
void StampView(VkBuffer, u64, VkDeviceSize, const TileTable*, u64) {}
void RecordImageTiling(VkCommandBuffer, const TileTable&, VkBuffer,
                       VkDeviceSize, VkDeviceSize, VkBuffer, VkDeviceSize,
                       bool, u32, u32) {}
VkBuffer AcquireScratch(VkDeviceSize) {
  return VK_NULL_HANDLE;
}
bool RecordTruthImport(CsRange&, u64, VkDeviceSize) {
  return false;
}
bool TruthAvailable() {
  return false;
}
#endif

// A resource whose range can be a tiled truth: a gfx10 surface the tiling
// shader handles, with the shader reading its texels directly or expanded.
bool TruthEligible(const ComputeInfo::Res& r) {
  return kCsTruth && kCsVram && g_dev.push_descriptor && r.image_staging &&
         !r.zero_fill && r.size && r.guest_size &&
         ((r.elem_bytes >= 4 && r.elem_bytes == r.stage_elem_bytes) ||
          ((r.elem_bytes == 1 || r.elem_bytes == 2) &&
           r.stage_elem_bytes == 4)) &&
         r.dfmt != 6 && r.dfmt != 35 && r.dfmt != 40 &&
         r.tiling_idx >= gcn::kGfx10TilingBase &&
         !gcn::TilingIsLinear(r.tiling_idx) &&
         r.guest_size <= g_dev.max_storage_buffer_range &&
         r.size <= g_dev.max_storage_buffer_range &&
         r.guest_size <= (768ull << 20) && r.size <= (768ull << 20) &&
         TruthAvailable();
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
           res.stage_elem_bytes == 4)) && res.dfmt != 35 &&
         res.dfmt != 40 && res.tiling_idx >= gcn::kGfx10TilingBase &&
         !gcn::TilingIsLinear(res.tiling_idx);
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
// Batches are submitted without waiting and retired in order as their fences
// signal; a reader waits for the one batch its range was last recorded into
// (CsBatchWaitId), and only then. The GPU then executes batch N while the
// walk records N+1, where a synchronous flush idled both in turn.
constexpr u32 kCsBatchRing = 6;
u64 g_cs_batch_id = 0;       // the open batch
u64 g_cs_batch_next_id = 1;
u64 g_cs_batch_done = 0;     // every batch with an id <= this has completed
u32 g_cs_batch_cur = 0;      // ring slot the open batch records into
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
};
std::vector<BatchedDispatch> g_cs_batch_log;
VkQueryPool g_cs_timestamps = VK_NULL_HANDLE;
struct CsGpuTime {
  double ns = 0;
  u64 count = 0;
};
std::unordered_map<u64, CsGpuTime> g_cs_gpu_times;
VkFence g_cs_batch_fence = VK_NULL_HANDLE;
std::unordered_map<VkBuffer, ComputeBufferAccess> g_cs_batch_access;

struct CsBatch {
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
  VkQueryPool timestamps = VK_NULL_HANDLE;
  u64 id = 0;
  bool submitted = false;
  u32 count = 0;
  std::vector<BatchedDispatch> log;
};
CsBatch g_cs_batches[kCsBatchRing];

// The batch a range referenced now belongs to: the open one, or the one that
// opens next when nothing is recording yet.
void MarkPending(CsRange& e) {
  e.pending_batch = true;
  e.batch_id = g_cs_batch_open ? g_cs_batch_id : g_cs_batch_next_id;
}

bool CsBatchInit() {
  if (g_cs_batches[0].fence)
    return true;
  for (auto& b : g_cs_batches) {
    VkCommandBufferAllocateInfo ca{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = g_dev.pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_dev.device, &ca, &b.cmd) != VK_SUCCESS)
      return false;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(g_dev.device, &fci, nullptr, &b.fence) != VK_SUCCESS)
      return false;
    // Sized for a whole batch of dispatches between submits.
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            256 * ComputeInfo::kMaxResources};
    VkDescriptorPoolCreateInfo pci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 256;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(g_dev.device, &pci, nullptr, &b.pool) !=
        VK_SUCCESS)
      return false;
    if (kCsSyncReport && g_dev.timestamp_valid_bits) {
      VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
      qi.queryCount = 256;  // two timestamps for each of at most 128 dispatches
      if (vkCreateQueryPool(g_dev.device, &qi, nullptr, &b.timestamps) !=
          VK_SUCCESS)
        b.timestamps = VK_NULL_HANDLE;
    }
  }
  g_cs_cmd = g_cs_batches[0].cmd;
  g_cs_desc_pool = g_cs_batches[0].pool;
  g_cs_batch_fence = g_cs_batches[0].fence;
  return true;
}

void CsBatchLogFault(const CsBatch& b, const char* what, VkResult r) {
  BASE_LOGI("gpuvk", "cs batch DEVICE FAULT: {}={} id={} n={}", what, (int)r,
            (unsigned long long)b.id, b.count);
  for (const auto& d : b.log) {
    base::String res;
    for (u32 i = 0; i < d.num_res && i < 4; i++)
      base::FormatTo(res, " [{}]{:#x}+{:#x}{}{}", i,
                     (unsigned long)d.res_base[i],
                     (unsigned long)d.res_size[i],
                     (d.res_img >> i) & 1 ? " img" : " buf",
                     (d.res_write >> i) & 1 ? " w" : "");
    BASE_LOGI("gpuvk", "  batched cs={:#x} groups=[{} {} {}] res={}{}",
              (unsigned long)d.cs_addr, d.groups[0], d.groups[1], d.groups[2],
              d.num_res, res.c_str());
  }
  ReportDeviceFault(g_dev);
  g_cs_failed = true;
}

// The batch's fence has signaled: what it produced is real now.
void CsBatchFinalize(CsBatch& b) {
  if (b.timestamps && !b.log.empty()) {
    std::array<u64, 256> stamps{};
    const u32 count = static_cast<u32>(b.log.size());
    if (vkGetQueryPoolResults(g_dev.device, b.timestamps, 0, count * 2,
                             count * 2 * sizeof(u64), stamps.data(), sizeof(u64),
                             VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      const u64 mask = UINT64_MAX >> (64 - g_dev.timestamp_valid_bits);
      for (u32 i = 0; i < count; ++i) {
        auto& time = g_cs_gpu_times[b.log[i].cs_addr];
        time.ns += ((stamps[i * 2 + 1] - stamps[i * 2]) & mask) *
                   double(g_dev.timestamp_period);
        ++time.count;
      }
    }
  }
  g_cs_batch_done = std::max(g_cs_batch_done, b.id);
  for (auto& kv : g_cs_ranges) {
    CsRange& e = kv.second;
    if (!e.pending_batch || e.batch_id > g_cs_batch_done)
      continue;
    e.pending_batch = false;
    // Every readback recorded into it is in the host mirror now.
    if (e.readback_pending) {
      e.readback_pending = false;
      e.mirror_current = true;
    }
  }
  b.submitted = false;
  b.log.clear();
  b.count = 0;
  // Imported ranges are written by the dispatches this batch executed,
  // straight into guest memory -- no writeback step will announce them, so the
  // batch itself is the visibility point for the staging caches.
  g_cs_writeback_gen++;
}

// Retire, in order, every submitted batch whose fence has signaled.
void CsBatchReap() {
  for (;;) {
    CsBatch* oldest = nullptr;
    for (auto& b : g_cs_batches)
      if (b.submitted && (!oldest || b.id < oldest->id))
        oldest = &b;
    if (!oldest || vkGetFenceStatus(g_dev.device, oldest->fence) != VK_SUCCESS)
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
  kSyncDescPool,
  kSyncBatchCap,
  kSyncFrameEnd,
  kSyncChunk,
  kSyncRingFull,
  kSyncCount
};
const char* const kCsSyncName[kSyncCount] = {
    "imported", "writeback", "scratch-grow", "range-grow",
    "stage-hazard", "desc-pool", "batch-cap", "frame-end", "chunk",
    "ring-full"};
// The frame's open chunk copies a texture out of a range the open batch
// writes: the batch has to be on the queue before the chunk is.
bool g_cs_chunk_needs_batch = false;
u64 g_cs_sync_n[kSyncCount] = {};
u64 g_cs_sync_ns[kSyncCount] = {};

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
    const VkResult r =
        vkWaitForFences(g_dev.device, 1, &b.fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS)
      CsBatchLogFault(b, "wait", r);
    CsBatchFinalize(b);
    g_ns_cs_gpu += NowNs() - t0;
    g_cs_sync_n[kSyncRingFull]++;
    g_cs_sync_ns[kSyncRingFull] += NowNs() - t0;
  }
  g_cs_cmd = b.cmd;
  g_cs_desc_pool = b.pool;
  g_cs_timestamps = b.timestamps;
  g_cs_batch_fence = b.fence;
  b.id = g_cs_batch_next_id++;
  g_cs_batch_id = b.id;
  vkResetCommandBuffer(g_cs_cmd, 0);
  vkResetDescriptorPool(g_dev.device, g_cs_desc_pool, 0);
  VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(g_cs_cmd, &cbi);
  if (g_cs_timestamps)
    vkCmdResetQueryPool(g_cs_cmd, g_cs_timestamps, 0, 256);
  CmdBeginLabel(g_cs_cmd, "cs batch (frame %llu)",
                (unsigned long long)g_frame.num);
  if (kCsTexBridge) {
    // A frame command buffer submitted before this batch may still be copying
    // a bridged texture out of a range the batch overwrites.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(g_cs_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
  }
  g_cs_batch_open = true;
}


// Record the staging move into the batch, and mark the range as referenced by
// it so a later grow/rename does not pull the buffer out from under the copy.
void CsCopyStaging(CsRange& e, VkDeviceSize bytes, bool to_device) {
  if (!e.device_local || !e.host_buf || !e.buf || !bytes)
    return;
  CsBatchBeginImpl();
  RecordStagingCopy(g_cs_cmd, e, bytes, to_device);
  g_cs_batch_access.erase(e.buf);
  MarkPending(e);
  if (to_device)
    e.mirror_current = true;  // both sides now hold what the CPU just wrote
  else
    e.readback_pending = true;
}

// Put the open batch on the queue and go on recording into the next slot.
bool CsBatchSubmit() {
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
  CmdEndLabel(g_cs_cmd);  // close the "cs batch" scope
  b.count = g_cs_batch_count;
  b.log = std::move(g_cs_batch_log);
  g_cs_batch_log.clear();
  VkResult r = vkEndCommandBuffer(g_cs_cmd);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &g_cs_cmd;
  if (r == VK_SUCCESS)
    r = vkResetFences(g_dev.device, 1, &b.fence);
  if (r == VK_SUCCESS)
    r = vkQueueSubmit(g_dev.queue, 1, &si, b.fence);
  g_cs_batch_open = false;
  g_cs_chunk_needs_batch = false;
  g_cs_batch_count = 0;
  g_cs_batch_access.clear();
  g_cs_submit_n++;
  if (r != VK_SUCCESS) {
    CsBatchLogFault(b, "submit", r);
    return false;
  }
  b.submitted = true;
  g_cs_batch_cur = (g_cs_batch_cur + 1) % kCsBatchRing;
  return true;
}

// Wait until batch `id` (and so every earlier one) has completed.
bool CsBatchWaitId(u64 id, CsSyncWhy why) {
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
    const VkResult r =
        vkWaitForFences(g_dev.device, 1, &oldest->fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
      CsBatchLogFault(*oldest, "wait", r);
      return false;
    }
    CsBatchFinalize(*oldest);
  }
  g_ns_cs_gpu += NowNs() - t0;
  g_cs_sync_n[why]++;
  g_cs_sync_ns[why] += NowNs() - t0;
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
template <typename It>
void CsStageReadbacks(It first, It last, bool all = true) {
  for (It it = first; it != last; ++it) {
    CsRange& e = it->second;
    if (e.gpu_dirty && e.device_local && !e.mirror_current && !e.imported &&
        (!CanGpuTile(e.res, e) || e.truth) &&
        (all || !CsLazyKept(it->first, e)))
      CsCopyStaging(e, e.size ? e.size : e.cap, /*to_device=*/false);
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
  if (!e.gpu_dirty)
    return true;
  if (e.imported) {  // the dispatch wrote straight into guest memory
    if (!CsBatchWaitRange(e, kSyncImported))
      return false;
    e.gpu_dirty = false;
    return true;
  }
  // Results live in VRAM: pull them into the host mirror on the same batch that
  // produced them, so the one fence wait covers the dispatch and the copy.
  // Already-recorded readbacks (CsStageReadbacks) skip this.
  if (e.device_local && !e.mirror_current &&
      (!CanGpuTile(e.res, e) || e.truth))
    CsCopyStaging(e, e.size ? e.size : e.cap, /*to_device=*/false);
  if (e.pending_batch) {
    const u64 _tw = NowNs();
    const bool ok = CsBatchWaitRange(e, kSyncWriteback);
    if (kCsSyncReport) {
      StageStat& st = g_stage_stats[base];
      st.wait++;
      st.wait_ns += NowNs() - _tw;
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
    static std::unordered_set<u64> seen;
    if (seen.size() < 4096 && seen.insert(base).second)
      BASE_LOGI("cswb",
                "image base={:#x} {}x{} layers={} tiling={} staged {}/{} "
                "non-zero",
                (unsigned long long)base, e.res.width, e.res.height,
                e.res.layers, e.res.tiling_idx, (unsigned long long)nz,
                (unsigned long long)e.res.size);
  }
  const u64 _t_wb = NowNs();
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
                  e.res.tiling_idx, e.res.elem_bytes,
                  e.res.stage_elem_bytes);
      return false;
    }
  } else if (!e.image_staging || e.truth) {
    // Never write back more than the guest footprint the range was staged
    // from, and never into memory that is not the guest's: the staged size is
    // the shader's view of the resource and a descriptor with a bogus size
    // would otherwise scribble compute results over whatever follows -- which
    // reads as float data turning up in the title's allocator free lists.
    const u64 n =
        e.guest_bytes ? std::min<u64>(e.size, e.guest_bytes) : e.size;
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
    // whole view of a resource, but a shader writes a part of it -- and this
    // title puts its CPU heap in the same direct memory as the buffers it hands
    // to compute, so the bytes in between belong to the allocator. Copying the
    // full range back therefore reverts every CPU write made since stage-in,
    // and a word the CPU was writing WHILE we snapshotted comes back half
    // updated: that is the 5-valid-bytes-plus-3-stale-bytes free-tree pointer
    // SotC dies on at eboot+0x48ac5 when New Game is confirmed.
    // The shadow copy taken at stage-in says which words the shader touched;
    // untouched words are left alone. (A shader that rewrites a word with the
    // value it already had is skipped too, which is a no-op by definition.)
    // The merge is symmetric, so that staging, shadow and guest all agree again
    // when it returns -- the bookkeeping below this point promises exactly that:
    //   the shader wrote the block  -> guest   <- staging  (publish the result)
    //   the shader left it alone    -> staging <- guest    (adopt the CPU's word)
    // Skipping the second half would leave a dispatch reading stale values for
    // whatever the CPU changed, which is the same defect pointed the other way.
    // Guest-sourced linear ranges only. A range staged from a live render
    // target exists precisely so the writeback can publish that image into
    // guest memory, so "the dispatch did not write it" must not stop it there.
    if (!kCsWbFull && !e.rt_sourced && e.shadow_valid && e.shadow.size() >= n) {
      auto* src = static_cast<u8*>(e.map);
      u8* shd = e.shadow.data();
      auto* dst = reinterpret_cast<u8*>(base);
      const u64 blocks = n / 64;
      u64 off = blocks * 64, wrote = 0;
      // 64-byte blocks: a block the shader left alone costs one compare, which
      // keeps "the shader wrote a little of a big range" -- the common case --
      // no more expensive than the unconditional copy this replaces. Blocks are
      // independent, so the sweep is split across the detiler's pool; the
      // per-lane written counts are summed rather than shared.
      constexpr u64 kMergeBlocksPerChunk = 1024;  // 64 KiB a chunk
      const u32 chunks =
          static_cast<u32>((blocks + kMergeBlocksPerChunk - 1) /
                           kMergeBlocksPerChunk);
      std::vector<u64> chunk_wrote(chunks, 0);
      gcn::DetileParallelWork(chunks, blocks * 64, [&](u32 c0, u32 c1) {
        for (u32 c = c0; c < c1; c++) {
          const u64 first = u64(c) * kMergeBlocksPerChunk;
          const u64 last = std::min<u64>(first + kMergeBlocksPerChunk, blocks);
          u64 local = 0;
          for (u64 b = first; b < last; b++) {
            const u64 o = b * 64;
            if (std::memcmp(src + o, shd + o, 64) != 0) {
              std::memcpy(dst + o, src + o, 64);
              std::memcpy(shd + o, src + o, 64);
              local += 64;
            } else {
              std::memcpy(src + o, dst + o, 64);
              std::memcpy(shd + o, dst + o, 64);
            }
          }
          chunk_wrote[c] = local;
        }
      });
      for (const u64 w : chunk_wrote)
        wrote += w;
      for (; off < n; off++) {
        if (src[off] != shd[off]) {
          dst[off] = src[off];
          shd[off] = src[off];
          wrote++;
        } else {
          src[off] = dst[off];
          shd[off] = dst[off];
        }
      }
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
          const u64 blk = std::min<u64>(64, n - off);
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
  const u64 _t_rt = NowNs();
  if (e.image_staging)
    g_out_retile_ns += _t_rt - _t_wb;
  else
    g_out_buffer_ns += _t_rt - _t_wb;
  g_out_retile_n += e.image_staging;
  UploadCsRangeToRt(base, e);  // refresh a live RT image aliasing the range
  const u64 _t_inv = NowNs();
  g_out_rt_ns += _t_inv - _t_rt;
  InvalidateTexRange(base, e.guest_bytes);
  UnindexDirtyRange(base, e.guest_bytes);
  // Guest memory just changed under any staged copy of it: retire the draw
  // path's per-frame staging cache entries (they validate against this).
  g_cs_writeback_gen++;
  e.gpu_dirty = false;
  e.hash = RangeHash(base, e.guest_bytes);
  e.last_validated_frame = g_frame.num;
  g_out_tail_ns += NowNs() - _t_inv;
  return true;
}

// Ensure staging slot i can hold `size` bytes (grow-on-demand, kept mapped).
bool CsEnsureStage(u32 i, VkDeviceSize size) {
  GPU_BUGCHECK(i < ComputeInfo::kMaxResources, "stage index %u out of bounds",
               i);
  CsStage& s = g_cs_stage[i];
  if (s.buf && s.cap >= size)
    return true;
  if (s.map) {
    vkUnmapMemory(g_dev.device, s.mem);
    s.map = nullptr;
  }
  // A batch in flight may still bind the old scratch: it outlives the frame.
  if (s.buf || s.mem)
    g_cs_frame_retired.emplace_back(s.buf, s.mem);
  s.buf = VK_NULL_HANDLE;
  s.mem = VK_NULL_HANDLE;
  VkDeviceSize cap =
      (size + 0xFFFFF) & ~VkDeviceSize(0xFFFFF);  // 1 MiB granularity
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = cap;
  bi.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &s.buf) != VK_SUCCESS) {
    s.buf = VK_NULL_HANDLE;
    return false;
  }
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev.device, s.buf, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  // Scratch is zeroed by vkCmdFillBuffer and only ever touched by shaders, so
  // it wants VRAM and no mapping at all.
  const u32 scratch_device =
      kCsVram ? FindDeviceMemoryType(mr.memoryTypeBits) : UINT32_MAX;
  ai.memoryTypeIndex = scratch_device != UINT32_MAX
                           ? scratch_device
                           : FindComputeMemoryType(mr.memoryTypeBits);
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &s.mem) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, s.buf, nullptr);
    s.buf = VK_NULL_HANDLE;
    s.mem = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(g_dev.device, s.buf, s.mem, 0);
  if (scratch_device == UINT32_MAX)
    vkMapMemory(g_dev.device, s.mem, 0, cap, 0, &s.map);
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
                       const void* src, void* dst, bool detile) {
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
    std::vector<std::pair<u64, CsGpuTime>> times(g_cs_gpu_times.begin(),
                                               g_cs_gpu_times.end());
    std::sort(times.begin(), times.end(), [](const auto& a, const auto& b) {
      return a.second.ns > b.second.ns;
    });
    for (size_t i = 0; i < std::min<size_t>(8, times.size()); ++i)
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
            "copy={:.1f}ms",
            g_in_hash_ns / frames / 1e6, g_in_hash_n / frames,
            g_in_detile_ns / frames / 1e6, g_in_rt_ns / frames / 1e6,
            g_in_rt_n / frames, g_in_copy_ns / frames / 1e6);
  g_in_hash_ns = g_in_detile_ns = g_in_rt_ns = g_in_copy_ns = 0;
  g_in_hash_n = g_in_rt_n = 0;
  BASE_LOGI("csin2",
            "overlap-wb={:.1f}ms x{:.1f} reshape-wb={:.1f}ms x{:.1f} "
            "alloc={:.1f}ms x{:.1f} recycled={:.1f} ranges={} {:.0f}MB "
            "membuf={:.1f}ms "
            "shadow={:.1f}ms lazy-kept={:.1f}x {:.1f}MB | truth stage={:.1f}MB "
            "import={:.1f} detile={:.1f} retile={:.1f} parent={:.1f} "
            "fold={:.1f} rt={:.1f} view-hits={:.1f} scratch={:.0f}MB",
            g_in_overlap_ns / frames / 1e6, g_in_overlap_n / frames,
            g_in_reshape_ns / frames / 1e6, g_in_reshape_n / frames,
            g_in_alloc_ns / frames / 1e6, g_in_alloc_n / frames,
            g_cs_recycled_n / frames, g_cs_ranges.size(),
            g_cs_range_bytes / 1e6,
            g_in_membuf_ns / frames / 1e6, g_in_shadow_ns / frames / 1e6,
            g_lazy_kept_n / frames, g_lazy_kept_bytes / frames / 1e6,
            g_truth_stage_bytes / frames / 1e6, g_truth_import_n / frames,
            g_truth_detile_n / frames, g_truth_retile_n / frames,
            g_truth_parent_n / frames, g_truth_fold_n / frames,
            g_truth_rt_n / frames, g_truth_view_hits / frames,
            g_cs_scratch_bytes / 1e6);
  g_truth_view_hits = 0;
  {
    base::String why;
    base::FormatTo(why, "writebacks/frame ({:.1f}MB):",
                   g_wb_why_bytes / frames / 1e6);
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
  BASE_LOGI("cstiling", "GPU conversion including copies: in={:.1f}ms x{:.1f} "
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
  std::vector<std::pair<u64, StageStat>> top(g_stage_stats.begin(),
                                             g_stage_stats.end());
  std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
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
            "texbridge={:.1f}x {:.1f}MB chunks={:.1f}",
            g_out_retile_ns / frames / 1e6, g_out_retile_n / frames,
            g_out_buffer_ns / frames / 1e6,
            g_out_rt_ns / frames / 1e6, g_out_rt_submits / frames,
            g_out_tail_ns / frames / 1e6, g_tex_bridge_n / frames,
            g_tex_bridge_bytes / frames / 1e6, g_cs_chunk_splits / frames);
  g_out_retile_ns = g_out_rt_ns = g_out_tail_ns = 0;
  g_out_buffer_ns = 0;
  g_tex_bridge_n = g_tex_bridge_bytes = g_cs_chunk_splits = 0;
  g_out_retile_n = g_out_rt_submits = 0;
  base::String syncs;
  base::FormatTo(syncs, "{:.1f} syncs/frame ({:.1f} submits):",
                 total / frames, g_cs_submit_n / frames);
  g_cs_submit_n = 0;
  for (int i = 0; i < kSyncCount; i++) {
    if (!g_cs_sync_n[i])
      continue;
    base::FormatTo(syncs, " {}={:.1f}({:.1f}ms)", kCsSyncName[i],
                   g_cs_sync_n[i] / frames, g_cs_sync_ns[i] / frames / 1e6);
    g_cs_sync_n[i] = 0;
    g_cs_sync_ns[i] = 0;
  }
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
  if (!depth.image || depth.layout == VK_IMAGE_LAYOUT_UNDEFINED || !range.buf ||
      range.pending_batch || range.gpu_dirty || !range.image_staging)
    return false;

  CsAliasedImage image{depth.image,
                       depth.w,
                       depth.h,
                       4,
                       VK_IMAGE_ASPECT_DEPTH_BIT,
                       depth.layout,
                       true};
  AliasedCopyPlan plan;
  if (!PlanAliasedCopy(image, range.res, "preserves", plan) || plan.unpack)
    return false;

  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(range.res, tiled, linear))
    return false;
  const auto& level = linear.mips[0];
  const u64 copy_bytes = static_cast<u64>(level.pitch) * plan.h *
                              range.res.stage_elem_bytes;
  if (level.offset + copy_bytes > range.cap)
    return false;

  const VkImageLayout old_layout = depth.layout;
  AliasedImageBarrier(g_frame.cmd, image, old_layout,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      AliasedImageAccess(image, old_layout),
                      VK_ACCESS_TRANSFER_READ_BIT);
  VkBufferImageCopy copy{};
  copy.bufferOffset = level.offset;
  copy.bufferRowLength = level.pitch;
  copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
  copy.imageExtent = {plan.w, plan.h, 1};
  vkCmdCopyImageToBuffer(g_frame.cmd, depth.image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, range.buf, 1,
                         &copy);

  VkBufferMemoryBarrier buffer_barrier{
      VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  buffer_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  buffer_barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
  buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  buffer_barrier.buffer = range.buf;
  buffer_barrier.offset = 0;
  buffer_barrier.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(
      g_frame.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
      nullptr, 1, &buffer_barrier, 0, nullptr);
  AliasedImageBarrier(g_frame.cmd, image,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, old_layout,
                      VK_ACCESS_TRANSFER_READ_BIT,
                      AliasedImageAccess(image, old_layout));

  // Compute for frame N runs before graphics N. This graphics-side snapshot
  // is therefore the input for compute N+1, and must suppress that frame's
  // usual copy from the now-cleared depth image.
  range.rt_sourced = true;
  range.last_rt_frame = static_cast<int>(g_frame.num) + 1;
  if (kCsRtTrace) {
    static int logged = 0;
    if (logged++ < 16)
      BASE_LOGI("csrt", "f{} preserved depth {:#x} before clear", g_frame.num,
                (unsigned long)base);
  }
  return true;
}

// DELTA_GPU_QCHECK=1: before every CS batch is submitted, run an EMPTY
// command buffer through the same queue and wait for it. The device is lost
// by whatever ran before the first failing submit, so a checkpoint that
// fails names PRIOR queue work (graphics draws, an earlier bridge copy) as
// the fault, and one that succeeds while the batch's own wait fails
// isolates the batch content. The checkpoint shares the graphics queue, so
// it runs after everything already recorded there.
struct QueueCheckpoint {
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  bool ready = false;
};
QueueCheckpoint g_qcheck;
DELTA_OPTION(bool, kQueueCheck, "DELTA_GPU_QCHECK", false);

bool QueueCheckArmed() {
  return kQueueCheck;
}

bool QueueCheck(const char* where) {
  if (!kQueueCheck)
    return true;
  if (!g_qcheck.ready) {
    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_dev.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_dev.device, &ai, &g_qcheck.cmd) !=
        VK_SUCCESS)
      return true;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(g_dev.device, &fi, nullptr, &g_qcheck.fence) !=
        VK_SUCCESS)
      return true;
    g_qcheck.ready = true;
  }
  VkResult r = vkResetCommandBuffer(g_qcheck.cmd, 0);
  if (r == VK_SUCCESS) {
    VkCommandBufferBeginInfo bi{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    r = vkBeginCommandBuffer(g_qcheck.cmd, &bi);
    if (r == VK_SUCCESS)
      r = vkEndCommandBuffer(g_qcheck.cmd);
  }
  if (r == VK_SUCCESS)
    r = vkResetFences(g_dev.device, 1, &g_qcheck.fence);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &g_qcheck.cmd;
  if (r == VK_SUCCESS)
    r = vkQueueSubmit(g_dev.queue, 1, &si, g_qcheck.fence);
  if (r == VK_SUCCESS)
    r = vkWaitForFences(g_dev.device, 1, &g_qcheck.fence, VK_TRUE, UINT64_MAX);
  if (r != VK_SUCCESS)
    BASE_LOGI("gpuvk", "queue CHECKPOINT FAILED at {}: {}", where, (int)r);
  return r == VK_SUCCESS;
}

bool CsSupplyTexture(u64 base,
                     const gcn::TextureLayout32& layout,
                     u32 w,
                     u32 h,
                     VkImage img,
                     VkImageLayout old_layout,
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
      static std::unordered_set<u64> seen;
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
  if (r.layers != layout.layers || r.mip_levels != layout.mip_levels)
    return declined("layers/mips");
  // A truth's narrow texels come out packed by the tiling shader; every
  // other converted staging (11/11/10 unpacked, BCn decoded) has no image
  // the copy could land in.
  const bool narrow = e.truth && r.elem_bytes < 4 && r.stage_elem_bytes == 4;
  if (r.dfmt == 6 || r.dfmt == 35 || r.dfmt == 40 ||
      (!narrow && r.elem_bytes != r.stage_elem_bytes))
    return declined("converted staging");
  if (r.elem_bytes != layout.elem_bytes)
    return declined("texel size");
  if (CsAliasedBase(base))
    return declined("live target");
  // Another dirty range inside the footprint (a mip generator binding one
  // level at its own address) holds bytes this buffer does not.
  if (!DirtyRangesOverlapping(base, e.guest_bytes, base).empty())
    return declined("dirty sibling");
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(r, tiled, linear) || linear.layer_stride)
    return declined("layout");
  if (narrow && !gcn::BuildTextureLayout32(linear, r.width, r.height, r.pitch,
                                           r.layers, r.mip_levels, 8,
                                           r.pow2_pad, r.elem_bytes))
    return declined("packed layout");
  const TileTable* table =
      e.truth ? GetTileTable(tiled, linear, narrow) : nullptr;
  if (e.truth && !table)
    return declined("tile table");
  VkBufferImageCopy copies[16]{};
  for (u32 mip = 0; mip < linear.mip_levels; mip++) {
    const auto& level = linear.mips[mip];
    if (level.offset % std::max<u64>(4, r.elem_bytes) ||
        level.width != std::max(w >> mip, 1u) ||
        level.height != std::max(h >> mip, 1u))
      return declined("mip geometry");
    copies[mip].bufferOffset = level.offset;
    copies[mip].bufferRowLength = level.pitch;
    copies[mip].bufferImageHeight = level.stored_height;
    copies[mip].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0,
                                    linear.layers};
    copies[mip].imageExtent = {level.width, level.height, 1};
  }
  if (!img)
    return true;
  e.last_used_frame = g_frame.num;
  if (*seq == e.write_seq)
    return true;
  EndRegion();
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(g_frame.cmd,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr,
                       0, nullptr);
  VkBuffer src = e.buf;
  if (e.truth) {
    g_cs_scratch_owner++;
    bool fresh = true;
    src = AcquireView(linear.size, base, 0, table, e.write_seq, &fresh);
    if (!src)
      return false;
    if (fresh) {
      RecordImageTiling(g_frame.cmd, *table, e.buf, 0, e.guest_bytes, src,
                        linear.size, /*detile=*/true);
      StampView(src, base, 0, table, e.write_seq);
    }
  }
  ImageBarrier(g_frame.cmd, img, old_layout,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                   ? VK_ACCESS_SHADER_READ_BIT
                   : 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, linear.layers, linear.mip_levels);
  vkCmdCopyBufferToImage(g_frame.cmd, src, img,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         linear.mip_levels, copies);
  ImageBarrier(g_frame.cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               linear.layers, linear.mip_levels);
  *seq = e.write_seq;
  e.frame_ref = g_frame.num;
  e.chunk_ref = g_frame.chunk_seq;
  if (e.pending_batch)
    g_cs_chunk_needs_batch = true;
  g_tex_bridge_n++;
  g_tex_bridge_bytes += e.size;
  return true;
}

// The live colour target at `base` takes the truth's pixels on the frame
// command buffer: detile into a scratch, copy into the image, and leave the
// image in the layout the recording expects. No writeback, no wait.
bool CsRefreshRtInFrame(u64 base, CsRange& e) {
  if (!e.truth || !e.gpu_dirty || !e.buf || e.rt_seq == e.write_seq ||
      !g_frame.recording)
    return false;
  const ComputeInfo::Res& r = e.res;
  if (r.mip_levels != 1 || r.layers != 1 ||
      r.elem_bytes != r.stage_elem_bytes)
    return false;
  ActivateWrittenRtVariant(base, r.width, r.height);
  auto rt_it = g_rts.find(base);
  if (rt_it == g_rts.end() || !rt_it->second.image || rt_it->second.is_depth)
    return false;
  RTarget& rt = rt_it->second;
  if (rt.w + 8 < r.width || rt.h + 8 < r.height ||
      FormatBytes(rt.fmt) != r.elem_bytes)
    return false;
  gcn::TextureLayout32 tiled, linear;
  const TileTable* table = BuildCsImageLayouts(r, tiled, linear)
                               ? GetTileTable(tiled, linear)
                               : nullptr;
  if (!table)
    return false;
  g_cs_scratch_owner++;
  bool fresh = true;
  const VkBuffer scratch =
      AcquireView(linear.size, base, 0, table, e.write_seq, &fresh);
  if (!scratch)
    return false;
  EndRegion();
  // A partial conversion (mip 0, slice 0) is not a whole view: stamp nothing.
  if (fresh)
    RecordImageTiling(g_frame.cmd, *table, e.buf, 0, e.guest_bytes, scratch,
                      linear.size, /*detile=*/true, 1, 1);
  const VkImageLayout from = rt.layout;
  ImageBarrier(g_frame.cmd, rt.image, from,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ColorImageAccess(from),
               VK_ACCESS_TRANSFER_WRITE_BIT);
  const auto& level = linear.mips[0];
  VkBufferImageCopy copy{};
  copy.bufferOffset = level.offset;
  copy.bufferRowLength = level.pitch;
  copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.imageExtent = {std::min(rt.w, r.width), std::min(rt.h, r.height), 1};
  vkCmdCopyBufferToImage(g_frame.cmd, scratch, rt.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  const VkImageLayout to = from == VK_IMAGE_LAYOUT_UNDEFINED
                               ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                               : from;
  if (to != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    ImageBarrier(g_frame.cmd, rt.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 to, VK_ACCESS_TRANSFER_WRITE_BIT, ColorImageAccess(to));
  rt.layout = to;
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
  if (!e.truth || !e.gpu_dirty)
    return false;
  // Already holds this revision: nothing to flush either.
  return e.rt_seq == e.write_seq || CsRefreshRtInFrame(base, e);
}

void ReleaseRetiredCsBuffers() {
  static std::vector<std::pair<VkBuffer, VkDeviceMemory>> aged;
  for (auto& [buf, mem] : aged) {
    if (buf)
      vkDestroyBuffer(g_dev.device, buf, nullptr);
    if (mem)
      vkFreeMemory(g_dev.device, mem, nullptr);
  }
  aged = std::move(g_cs_frame_retired);
  g_cs_frame_retired.clear();
}

}  // namespace gpu::vk

namespace gpu::rhi {
// Declared in rhi/renderer.h for the kernel's crash handler.
bool DescribeCsRangeCovering(u64 addr, char* out, size_t out_size) {
  using namespace gpu::vk;
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
                  (unsigned long long)(addr - base),
                  (unsigned long long)e.size, (int)e.gpu_dirty,
                  (int)e.imported, (int)e.rt_sourced, (int)e.image_staging,
                  (int)e.shadow_valid, e.last_used_frame);
    return true;
  }
  return false;
}
}  // namespace gpu::rhi

namespace gpu::rhi {
using namespace gpu::vk;

static bool PrepareGdsTransfer(Renderer& renderer, u32 offset, u32 bytes,
                                bool read = false) {
  if (!renderer.available() || offset > GdsBuffer::kBytes ||
      bytes > GdsBuffer::kBytes - offset || !EnsureGdsBuffer())
    return false;
  if (read) {
    CsBatchBeginImpl();
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g_cs_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
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
  static std::unordered_set<std::string> reported;
  std::string key = std::to_string(ci.cs_addr) + why;
  if (reported.size() < 512 && reported.insert(key).second)
    BASE_LOGI("csgpu", "CS @{:#x} declined at exit {}", ci.cs_addr, why);
  return false;
}

// Two plain-buffer bindings on one guest base with different extents (a
// constant block read through a 0x1c-byte window and a 0x24-byte one) share
// one staging range, so they have to agree on its size: take the larger for
// both. Image bindings keep their own shapes and still conflict below.
const ComputeInfo& MergeSameBase(const ComputeInfo& ci, ComputeInfo& merged) {
  bool need = false;
  for (u32 i = 0; i < ci.num_res && !need; i++)
    for (u32 j = 0; j < i && !need; j++)
      need = ci.res[i].base == ci.res[j].base && !ci.res[i].zero_fill &&
             !ci.res[j].zero_fill && !ci.res[i].image_staging &&
             !ci.res[j].image_staging &&
             (ci.res[i].size != ci.res[j].size ||
              ci.res[i].guest_size != ci.res[j].guest_size);
  if (!need)
    return ci;
  merged = ci;
  for (u32 i = 0; i < merged.num_res; i++)
    for (u32 j = 0; j < i; j++) {
      auto& a = merged.res[i];
      auto& b = merged.res[j];
      if (a.base != b.base || a.zero_fill || b.zero_fill || a.image_staging ||
          b.image_staging)
        continue;
      a.size = b.size = std::max(a.size, b.size);
      a.guest_size = b.guest_size = std::max(a.guest_size, b.guest_size);
    }
  return merged;
}

bool Dispatch(Renderer& renderer, const ComputeInfo& ci_in) {
  ComputeInfo ci_merged;
  const ComputeInfo& ci = MergeSameBase(ci_in, ci_merged);
  if (g_cs_failed) {
    renderer.state = nullptr;
    return CsDeclined(ci, "1");
  }
  if (!renderer.available() || !ci.recomp || !ci.recomp->ok || !ci.num_res ||
      ci.num_res + (ci.recomp->guest_memory_binding >= 0) + (ci.gds_binding >= 0) > g_dev.max_cs_resources)
    return CsDeclined(ci, "2");
  VkDescriptorBufferInfo guest_table{};
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
    if (!PrepareGuestMemory(ci.guest_memory, guest_table, ci.recomp->guest_memory_written))
      return CsDeclined(ci, "guest-address-map");
  }
  ScopeCs _cs;
  for (u32 i = 0; i < ci.num_res; i++)
    g_cs_bytes += ci.res[i].size;
  for (u32 i = 0; i < ci.num_res; i++)
    if (ci.res[i].size > g_dev.max_storage_buffer_range)
      return CsDeclined(ci, "3");
  for (u32 i = 0; i < ci.num_res; i++) {
    if (ci.res[i].zero_fill)
      continue;
    const u64 guest_bytes =
        ci.res[i].guest_size ? ci.res[i].guest_size : ci.res[i].size;
    const VkDeviceSize size =
        ci.res[i].size ? ((ci.res[i].size + 3) & ~VkDeviceSize(3)) : 4;
    for (u32 j = 0; j < i; j++) {
      if (ci.res[j].zero_fill || ci.res[j].base != ci.res[i].base)
        continue;
      const u64 other_guest_bytes =
          ci.res[j].guest_size ? ci.res[j].guest_size : ci.res[j].size;
      const VkDeviceSize other_size =
          ci.res[j].size ? ((ci.res[j].size + 3) & ~VkDeviceSize(3)) : 4;
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
    if (!ci.res[i].image_staging)
      NoteRawWrite(ci.res[i].base, bytes);
  }
  CsPipe* cp = GetCsPipe(ci);
  if (!cp)
    return CsDeclined(ci, "5");

  // Persistent command buffer + descriptor pool (created once, reused).
  if (g_cs_cmd == VK_NULL_HANDLE) {
    VkCommandBufferAllocateInfo ca{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = g_dev.pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_dev.device, &ca, &g_cs_cmd) != VK_SUCCESS) {
      g_cs_cmd = VK_NULL_HANDLE;
      return CsDeclined(ci, "6");
    }
  }
  if (g_cs_batch_fence == VK_NULL_HANDLE) {
    if (!CsBatchInit())
      return CsDeclined(ci, "8");
    // What an EMPTY submit+wait costs here. SotC spends over half its frame in
    // fence waits while the GPU sits at 18% utilisation, so the question is
    // whether a round trip is inherently expensive on this system or whether
    // the GPU is really doing that work. Measured once, at init.
    if (kCsSyncReport) {
      double total = 0;
      for (int i = 0; i < 100; i++) {
        vkResetCommandBuffer(g_cs_cmd, 0);
        VkCommandBufferBeginInfo bi{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(g_cs_cmd, &bi);
        vkEndCommandBuffer(g_cs_cmd);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &g_cs_cmd;
        const u64 t0 = NowNs();
        vkResetFences(g_dev.device, 1, &g_cs_batch_fence);
        vkQueueSubmit(g_dev.queue, 1, &si, g_cs_batch_fence);
        vkWaitForFences(g_dev.device, 1, &g_cs_batch_fence, VK_TRUE,
                        UINT64_MAX);
        total += (NowNs() - t0) / 1e6;
      }
      BASE_LOGI("csbench", "empty submit+wait: {:.3f} ms", total / 100.0);
    }
  }

  // DELTA_GPU_CSLIST: per-dispatch resource staging list for the first 200
  // dispatches — shows what the chain actually round-trips per frame.
  // DELTA_GPU_CSHIST=<seconds>: every distinct guest range a dispatch WRITES,
  // for the whole run. The capped per-dispatch list above only ever showed the
  // first few frames, which cannot answer "does the title ever compute-copy
  // into its texture heap".
  if (kCsHist) {
    struct Cell { u64 size, n; };
    static std::mutex m;
    static std::map<u64, Cell> tbl;
    static auto last = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(m);
    for (u32 i = 0; i < ci.num_res; i++)
      if (ci.res[i].written && tbl.size() < 4096) {
        Cell& c = tbl[ci.res[i].base];
        c = {ci.res[i].size, c.n + 1};
      }
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last).count() >=
        kCsHist) {
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
  const u64 _t_in0 = NowNs();
  VkBuffer bind_buf[ComputeInfo::kMaxResources];
  VkDeviceSize sz[ComputeInfo::kMaxResources];
  VkBuffer truth_view[ComputeInfo::kMaxResources] = {};
  const TileTable* truth_table[ComputeInfo::kMaxResources] = {};
  // A binding served from a dirty truth that contains it: that range's base,
  // and the binding's byte offset into it.
  u64 parent_base[ComputeInfo::kMaxResources] = {};
  VkDeviceSize parent_off[ComputeInfo::kMaxResources] = {};
  g_cs_scratch_owner++;
  // The shader's view of a truth: the bytes at `off` in `tiled_buf` detiled
  // into a scratch on the batch. Bindings of one dispatch that describe the
  // same surface the same way share the view (a shadow array bound through
  // several descriptors is one 320 MB conversion, not four).
  auto bind_truth_view = [&](u32 i, VkBuffer tiled_buf, VkDeviceSize off,
                             u64 span, u64 pbase) -> bool {
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
    const VkBuffer view =
        AcquireView(sz[i], pbase ? pbase : ci.res[i].base, off, table, seq,
                    &fresh);
    if (!view)
      return false;
    if (fresh) {
      CsBatchBeginImpl();
      RecordImageTiling(g_cs_cmd, *table, tiled_buf, off, span, view, sz[i],
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
    sz[i] = ci.res[i].size ? ((ci.res[i].size + 3) & ~VkDeviceSize(3)) : 4;
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
        if ((!r.truth && r.image_staging) || !r.gpu_dirty || !r.buf ||
            cand > base || base + guest_bytes > cand + r.guest_bytes ||
            (base - cand) % g_dev.storage_buffer_offset_align)
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
          const VkDeviceSize n = std::min<VkDeviceSize>(
              s.size ? s.size : s.cap, static_cast<VkDeviceSize>(guest_bytes));
          CsBatchBeginImpl();
          VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
          mb.srcAccessMask =
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
              VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
          mb.dstAccessMask =
              VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
          vkCmdPipelineBarrier(g_cs_cmd,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                               nullptr, 0, nullptr);
          const VkBufferCopy fold{0, base - pbase, n};
          vkCmdCopyBuffer(g_cs_cmd, s.buf, parent->buf, 1, &fold);
          mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
          mb.dstAccessMask =
              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
              VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
          vkCmdPipelineBarrier(g_cs_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, 1, &mb, 0, nullptr, 0, nullptr);
          g_cs_batch_access.erase(parent->buf);
          UnindexDirtyRange(base, s.guest_bytes);
          s.gpu_dirty = false;
          s.last_validated_frame = -1;
          s.hash = 0;
          s.mirror_current = false;
          MarkPending(s);
          parent->write_seq++;
        }
      }
      if (parent) {
        const VkDeviceSize off = base - pbase;
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
    // A range whose writeback cannot be produced (an image the retiler
    // does not handle) stays stale whatever happens next; declining THIS
    // dispatch over it only loses a second result. Only a dead device stops
    // the recording.
    // NOTE: a range that is itself dirty keeps a VRAM copy without the
    // sibling's bytes after this, and one that is clean relies on the
    // sampled hash to notice them. Both are the pre-existing behaviour; the
    // fix is one VRAM copy per guest byte (see des-perf-profile).
    {
      const u64 _to = NowNs();
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
        if (s.gpu_dirty) {
          g_wb_why["overlap"]++;
          g_wb_why_bytes += s.size;
        }
        if (!CsRangeFlushOne(dirty, s) && g_cs_failed)
          return CsDeclined(ci, "11");
      }
      g_in_overlap_n += !siblings.empty();
      g_in_overlap_ns += NowNs() - _to;
    }
    CsRange& e = g_cs_ranges[base];
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
         e.guest_bytes == guest_bytes &&
         SameCsResourceShape(e.res, ci.res[i]));
    // A plain buffer bound again through a SMALLER window is the same data:
    // the staged copy already covers it and the descriptor carries its own
    // range. Treating that as a reshape restages the whole thing -- Astro Bot
    // has two dispatches a frame that size one 39 MB buffer differently, and
    // each flip cost a full copy and a fence wait.
    const bool subset =
        raw_view || truth_sub ||
        (kCsSubset && !exact_shape && e.buf && !e.imported &&
                     !e.truth && !ci.res[i].image_staging &&
                     !e.image_staging && e.size >= static_cast<u64>(sz[i]) &&
                     e.guest_bytes >= guest_bytes);
    const bool same_shape = exact_shape || subset;
    // What the buffer holds: the footprint for a truth, the linear view
    // otherwise. `stage_bytes` > 0: staged as raw guest bytes.
    const VkDeviceSize foot =
        (raw_view || truth_sub) ? e.cap : truth_i ? guest_bytes : sz[i];
    const u64 stage_bytes = (raw_view || truth_sub) ? e.guest_bytes
                            : truth_i               ? guest_bytes
                                                    : 0;
    const u64 hash_bytes = stage_bytes ? stage_bytes : guest_bytes;
    if (!same_shape && e.gpu_dirty) {
      const u64 _trs = NowNs();
      g_wb_why["reshape"]++;
      g_wb_why_bytes += e.size;
      const bool ok = CsRangeFlushOne(base, e);
      g_in_reshape_ns += NowNs() - _trs;
      g_in_reshape_n++;
      if (!ok)
        return CsDeclined(ci, "12");  // reshaped: keep its data
    }

    // Guest-page import: valid only for plain (non-image) linear ranges, and
    // only while the range is not already staged some other way.
    if ((kCsImport || ci.res[i].prefer_import) && !e.buf &&
        !ci.res[i].image_staging && !ci.res[i].zero_fill)
      CsRangeImportGuest(e, base, static_cast<VkDeviceSize>(sz[i]));
    const bool buffer_reused = e.buf && e.cap >= foot;
    const u64 _ta = NowNs();
    const bool ensured = CsRangeEnsureBuffer(e, foot);
    g_in_alloc_ns += NowNs() - _ta;
    g_in_alloc_n += !buffer_reused;
    if (!ensured)
      return CsDeclined(ci, "14");
    if (!buffer_reused)
      NameObject(VK_OBJECT_TYPE_BUFFER, (u64)e.buf, "csbuf %#llx",
                 (unsigned long long)base);
    // A buffer that was just rebuilt holds nothing, whatever the shape
    // bookkeeping says about it.
    bool valid = buffer_reused && same_shape &&
                 (e.gpu_dirty || e.last_validated_frame == g_frame.num);
    if (e.imported && same_shape)
      valid = true;  // the buffer IS the guest pages; nothing to copy
    // A read whose base is a live render target must be staged from the
    // VkImage: the guest bytes under an RT are stale (draws never write them
    // back), and the image content changes every frame regardless of the
    // guest hash. Attempted at most once per frame per range; a CS-written
    // buffer (gpu_dirty) stays authoritative.
    const bool rt_backed = ci.res[i].image_staging && CsAliasedBase(base);
    // ... and at most once per frame per range, but a target NOTHING has
    // rendered into since the last bridge still holds the bytes already
    // staged. Astro Bot's shadow array (1536x1536 x16 layers = 151 MB) is
    // re-lit rarely and re-copied every frame without this; that one range is
    // 40% of a frame's image staging.
    const bool rt_stale =
        kCsRtCache && e.rt_sourced && e.last_rt_frame >= 0 &&
        AliasedImageLastRender(base) <= e.last_rt_frame;
    const bool rt_attempt = rt_backed && !e.gpu_dirty && !rt_stale &&
                            e.last_rt_frame != static_cast<int>(g_frame.num);
    if (rt_attempt)
      valid = false;
    else if (rt_backed && !e.gpu_dirty && e.rt_sourced)
      valid = same_shape;
    if (!valid && same_shape && !rt_attempt) {
      const u64 _th = NowNs();
      const u64 h = RangeHash(base, hash_bytes);
      g_in_hash_ns += NowNs() - _th;
      g_in_hash_n++;
      if (h == e.hash)
        valid = true;
      else
        e.hash = h;
      e.last_validated_frame = g_frame.num;
    }
    // A range the shader only WRITES needs no content from guest memory: it
    // supplies every byte it will write back. SotC's material fills are whole
    // 4 MiB arenas of exactly that shape, and staging them in is ~60 ms a
    // frame of pure waste (`in=` in the fps line). Opt-in because a shader
    // that writes only PART of such a range would then write back whatever the
    // buffer happened to hold -- the resource plan says "never read", not
    // "writes all of it".
    //
    // Direct-memory arenas only. A compute range that aliases module .data
    // (modules sit at 0x2xxxxxxxxxxx, the dmem arena at 0x80xxxxxxxx) gets a
    // partial write + writeback of stale staging over live globals when the
    // plan's "written" bit is wrong or the write is partial -- SotC's menu
    // transition died exactly that way, float garbage across the allocator's
    // static bin array (SIGSEGV in malloc, Shadow_Shipping+0xfacff0).
    const bool dmem_arena =
        base >= 0x8000000000ull && base < 0x9000000000ull;
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
        const u64 _tr = NowNs();
        e.rt_sourced = StageCsRangeFromRt(ci.res[i], e);
        g_in_rt_ns += NowNs() - _tr;
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
        bool gpu_staged = false;
        if (stage_bytes) {
          g_truth_stage_bytes += stage_bytes;
          gpu_staged = RecordTruthImport(e, base, stage_bytes);
          if (gpu_staged)
            g_cs_batch_access.erase(e.buf);
          else {
            const u64 _tm = NowNs();
            ParallelCopy(e.map, reinterpret_cast<const void*>(base),
                         stage_bytes);
            g_in_membuf_ns += NowNs() - _tm;
          }
        } else if (ci.res[i].image_staging) {
          g_stage_guest_detile_bytes += sz[i];
          g_stage_guest_detile_n++;
          const u64 _td = NowNs();
          gpu_staged = ConvertCsRangeImage(ci.res[i], e, true);
          const bool ok = gpu_staged ||
              (!g_cs_failed && StageCsImage(ci.res[i], e.map));
          g_in_detile_ns += NowNs() - _td;
          if (!ok)
            return CsDeclined(ci, "18");
        } else {
          const u64 _tm = NowNs();
          std::memcpy(e.map, reinterpret_cast<const void*>(base),
                      ci.res[i].size);
          if (sz[i] > ci.res[i].size)
            std::memset(static_cast<u8*>(e.map) + ci.res[i].size, 0,
                        sz[i] - ci.res[i].size);
          g_in_membuf_ns += NowNs() - _tm;
        }
        e.rt_sourced = false;
        if (rt_attempt) {  // fell back: keep guest-hash bookkeeping coherent
          e.hash = RangeHash(base, hash_bytes);
          e.last_validated_frame = g_frame.num;
        }
        // The CPU wrote the host mirror; the shaders bind the VRAM copy.
        const u64 _tc = NowNs();
        if (!gpu_staged)
          CsCopyStaging(e, stage_bytes ? stage_bytes : sz[i],
                        /*to_device=*/true);
        g_in_copy_ns += NowNs() - _tc;
      }
      if (!same_shape) {
        e.hash = RangeHash(base, hash_bytes);
        e.last_validated_frame = g_frame.num;
      }
      // Baseline for the writeback's write-coverage merge: the staging buffer as
      // it stands BEFORE any dispatch runs. Comparing against it is the only way
      // the writeback can tell a shader's output from a byte it merely staged in
      // and would otherwise copy back over the CPU's newer value. Imported
      // ranges never write back, and images retile through WritebackCsImage.
      if (!ci.res[i].image_staging && !e.imported && !stage_bytes) {
        const u64 _ts = NowNs();
        e.shadow.resize(sz[i]);
        std::memcpy(e.shadow.data(), e.map, sz[i]);
        e.shadow_valid = true;
        g_in_shadow_ns += NowNs() - _ts;
      } else {
        e.shadow_valid = false;
      }
      e.gpu_dirty = false;
      e.write_seq++;
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
          static std::unordered_set<u64> shape_logged, big_logged;
          auto& logged = sz[i] >= (1u << 20) ? big_logged : shape_logged;
          if (logged.size() < 64 && logged.insert(base).second)
            BASE_LOGI("csshape",
                      "{:#x} was {}x{} p={} l={} m={} t={} e={}/{} d={} sz={:#x}"
                      " now {}x{} p={} l={} m={} t={} e={}/{} d={} sz={:#x}",
                      (unsigned long)base, e.res.width, e.res.height,
                      e.res.pitch, e.res.layers, e.res.mip_levels,
                      e.res.tiling_idx, e.res.elem_bytes,
                      e.res.stage_elem_bytes, e.res.dfmt,
                      (unsigned long)e.size, ci.res[i].width, ci.res[i].height,
                      ci.res[i].pitch, ci.res[i].layers, ci.res[i].mip_levels,
                      ci.res[i].tiling_idx, ci.res[i].elem_bytes,
                      ci.res[i].stage_elem_bytes, ci.res[i].dfmt,
                      (unsigned long)sz[i]);
        }
        else if (rt_attempt)
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
      const VkDeviceSize n = std::min<VkDeviceSize>(
          s.size ? s.size : s.cap, static_cast<VkDeviceSize>(s.guest_bytes));
      CsBatchBeginImpl();
      VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                         VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_TRANSFER_WRITE_BIT;
      mb.dstAccessMask =
          VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(g_cs_cmd,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                           nullptr, 0, nullptr);
      const VkBufferCopy fold{0, folds[k] - base, n};
      vkCmdCopyBuffer(g_cs_cmd, s.buf, e.buf, 1, &fold);
      mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                         VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(g_cs_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, 1, &mb, 0, nullptr, 0, nullptr);
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
      }
      e.write_seq++;
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
  VkDeviceSize bind_off[ComputeInfo::kMaxResources] = {};

  g_ns_cs_in += NowNs() - _t_in0;

  // Descriptor set binding the storage buffers (pool lives for a whole batch;
  // reset happens at batch flush).
  VkDescriptorSet set;
  VkDescriptorSetAllocateInfo da{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  da.descriptorPool = g_cs_desc_pool;
  da.descriptorSetCount = 1;
  da.pSetLayouts = &cp->set_layout;
  if (vkAllocateDescriptorSets(g_dev.device, &da, &set) != VK_SUCCESS) {
    if (!CsBatchSubmit()) {
      renderer.state = nullptr;
      return CsDeclined(ci, "19");
    }
    CsBatchBeginImpl();
    da.descriptorPool = g_cs_desc_pool;
    if (vkAllocateDescriptorSets(g_dev.device, &da, &set) != VK_SUCCESS)
      return CsDeclined(ci, "20");
  }
  VkDescriptorBufferInfo dbi[ComputeInfo::kMaxResources + 2];
  VkWriteDescriptorSet wr[ComputeInfo::kMaxResources + 2];
  for (u32 i = 0; i < ci.num_res; i++)
    bind_off[i] = ci.res[i].zero_fill ? 0
                  : truth_view[i]     ? 0
                  : parent_base[i]    ? parent_off[i]
                                      : g_cs_ranges[ci.res[i].base].imported_offset;
  for (u32 i = 0; i < ci.num_res; i++) {
    dbi[i] = {bind_buf[i], bind_off[i], sz[i]};
    wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr[i].dstSet = set;
    wr[i].dstBinding = ci.res[i].binding;
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo = &dbi[i];
  }
  u32 nwrite = ci.num_res;
  if (ci.gds_binding >= 0 && EnsureGdsBuffer()) {
    dbi[nwrite] = {g_gds.buf, 0, GdsBuffer::kBytes};
    wr[nwrite] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr[nwrite].dstSet = set;
    wr[nwrite].dstBinding = static_cast<u32>(ci.gds_binding);
    wr[nwrite].descriptorCount = 1;
    wr[nwrite].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[nwrite].pBufferInfo = &dbi[nwrite];
    nwrite++;
  }
  if (ci.recomp->guest_memory_binding >= 0) {
    dbi[nwrite] = guest_table;
    wr[nwrite] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr[nwrite].dstSet = set;
    wr[nwrite].dstBinding = ci.recomp->guest_memory_binding;
    wr[nwrite].descriptorCount = 1;
    wr[nwrite].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[nwrite].pBufferInfo = &dbi[nwrite];
    nwrite++;
  }
  vkUpdateDescriptorSets(g_dev.device, nwrite, wr, 0, nullptr);

  // Record the dispatch into the open batch. Submission + the fence wait
  // happen at the next flush point, not here.
  CsBatchBeginImpl();
  VkBufferMemoryBarrier zero_before[ComputeInfo::kMaxResources];
  VkBufferMemoryBarrier zero_after[ComputeInfo::kMaxResources];
  u32 zero_count = 0;
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!ci.res[i].zero_fill)
      continue;
    VkBufferMemoryBarrier& before = zero_before[zero_count];
    before = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    before.srcAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.buffer = bind_buf[i];
    before.offset = 0;
    before.size = sz[i];
    VkBufferMemoryBarrier& after = zero_after[zero_count++];
    after = before;
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    g_cs_batch_access.erase(bind_buf[i]);
  }
  if (zero_count) {
    vkCmdPipelineBarrier(g_cs_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                         zero_count, zero_before, 0, nullptr);
    for (u32 i = 0; i < ci.num_res; i++)
      if (ci.res[i].zero_fill)
        vkCmdFillBuffer(g_cs_cmd, bind_buf[i], 0, sz[i], 0);
    vkCmdPipelineBarrier(g_cs_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         zero_count, zero_after, 0, nullptr);
  }
  if (ci.gds_binding >= 0) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(g_cs_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
  }
  vkCmdBindPipeline(g_cs_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp->pipe);
  vkCmdBindDescriptorSets(g_cs_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp->layout,
                          0, 1, &set, 0, nullptr);
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
  for (u32 i = 0; i < ci.num_res; i++)
    if (bind_buf[i] && ci.res[i].binding < kCsPushDwords - 16)
      pc[16 + ci.res[i].binding] = static_cast<u32>(sz[i] / 4);
  vkCmdPushConstants(g_cs_cmd, cp->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(pc), pc);
  VkBufferMemoryBarrier barriers[ComputeInfo::kMaxResources];
  u32 barrier_count = 0;
  VkBuffer unique_buffers[ComputeInfo::kMaxResources];
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
      VkBufferMemoryBarrier& barrier = barriers[barrier_count++];
      barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      barrier.srcAccessMask = (prior.read ? VK_ACCESS_SHADER_READ_BIT : 0) |
                              (prior.write ? VK_ACCESS_SHADER_WRITE_BIT : 0);
      barrier.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT |
          (unique_writes[i] ? VK_ACCESS_SHADER_WRITE_BIT : 0);
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.buffer = unique_buffers[i];
      barrier.offset = 0;
      barrier.size = VK_WHOLE_SIZE;
      prior = {};
    }
    prior.read = true;
    prior.write |= unique_writes[i];
  }
  if (barrier_count)
    vkCmdPipelineBarrier(g_cs_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         barrier_count, barriers, 0, nullptr);
  CmdInsertLabel(g_cs_cmd, "dispatch cs=%#llx %ux%ux%u res=%u",
                 (unsigned long long)ci.cs_addr, ci.groups[0], ci.groups[1],
                 ci.groups[2], ci.num_res);
  if (ci.recomp->guest_memory_binding >= 0) {
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(g_cs_cmd, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &host, 0, nullptr, 0, nullptr);
  }
  if (g_cs_timestamps)
    vkCmdWriteTimestamp(g_cs_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         g_cs_timestamps, g_cs_batch_count * 2);
  vkCmdDispatchBase(g_cs_cmd, ci.group_base[0], ci.group_base[1],
                    ci.group_base[2], ci.groups[0], ci.groups[1], ci.groups[2]);
  if (g_cs_timestamps)
    vkCmdWriteTimestamp(g_cs_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         g_cs_timestamps, g_cs_batch_count * 2 + 1);
  // What the dispatch wrote through a detiled view goes back into the truth.
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!truth_view[i] || !ci.res[i].written)
      continue;
    bool done = false;
    for (u32 j = 0; j < i && !done; j++)
      done = truth_view[j] == truth_view[i] && ci.res[j].written;
    if (done)
      continue;
    auto it = g_cs_ranges.find(parent_base[i] ? parent_base[i]
                                              : ci.res[i].base);
    if (it == g_cs_ranges.end() || !it->second.buf)
      continue;
    const u64 span = parent_base[i]
                         ? (ci.res[i].guest_size ? ci.res[i].guest_size
                                                 : ci.res[i].size)
                         : it->second.guest_bytes;
    RecordImageTiling(g_cs_cmd, *truth_table[i], it->second.buf,
                      parent_off[i], span, truth_view[i], sz[i],
                      /*detile=*/false);
    g_cs_batch_access.erase(it->second.buf);
    g_truth_retile_n++;
  }
  {
    BatchedDispatch bd{ci.cs_addr,
                       {ci.groups[0], ci.groups[1], ci.groups[2]},
                       ci.num_res,
                       {},
                       {},
                       0,
                       0};
    for (u32 i = 0; i < ci.num_res && i < 4; i++) {
      bd.res_base[i] = ci.res[i].base;
      bd.res_size[i] = ci.res[i].size;
      if (ci.res[i].image_staging)
        bd.res_img |= static_cast<u8>(1u << i);
      if (ci.res[i].written)
        bd.res_write |= static_cast<u8>(1u << i);
    }
    g_cs_batch_log.push_back(bd);
  }
  if (trace::Recording())
    trace::RecordDispatch(ci);
  for (u32 i = 0; i < ci.num_res; i++) {
    if (ci.res[i].zero_fill)
      continue;
    auto it = g_cs_ranges.find(parent_base[i] ? parent_base[i]
                                              : ci.res[i].base);
    if (it != g_cs_ranges.end())
      MarkPending(it->second);
  }
  if ((++g_cs_batch_count >= std::max<u32>(1, kCsBatchCap) ||
       kGpuCsgpuVerbose) &&
      !CsBatchSubmit()) {
    renderer.state = nullptr;
    return CsDeclined(ci, "21");
  }

  // Mark written ranges GPU-dirty. Guest memory catches up lazily at the next
  // flush point (draw / DMA / frame end) — writing every dispatch's outputs
  // back immediately (the image retile especially) was ~100ms/frame.
  const u64 _t_out0 = NowNs();
  for (u32 i = 0; i < ci.num_res; i++) {
    if (!ci.res[i].written || ci.res[i].zero_fill)
      continue;
    const u64 dirty_base = parent_base[i] ? parent_base[i] : ci.res[i].base;
    auto it = g_cs_ranges.find(dirty_base);
    if (it == g_cs_ranges.end())
      continue;
    if (!it->second.gpu_dirty)
      IndexDirtyRange(dirty_base, it->second.guest_bytes);
    it->second.gpu_dirty = true;
    it->second.write_seq++;
    if (truth_view[i])
      StampView(truth_view[i], dirty_base, parent_off[i], truth_table[i],
                it->second.write_seq);
    it->second.mirror_current = false;  // the dispatch outran the host mirror
    it->second.readback_pending = false;  // and any readback still queued
    if (kGpuCsgpuVerbose) {
      const u8* b = static_cast<const u8*>(it->second.map);
      u64 nz = 0,
               step = ci.res[i].size > 65536 ? ci.res[i].size / 65536 : 1;
      for (u64 k = 0; k < ci.res[i].size; k += step)
        nz += b[k] != 0;
      BASE_LOGI("csgpu", "gpu wrote base={:#x} size={} nonzero={}/{}",
                (unsigned long)ci.res[i].base, (unsigned long)ci.res[i].size,
                (unsigned long)nz, (unsigned long)(ci.res[i].size / step));
    }
  }
  if (ci.recomp->guest_memory_written) {
    // Publish physical writes before a CPU consumer or another dispatch can
    // use a stale staged copy. The dirty bitmap identifies actual stores,
    // rather than treating the shader's whole guest pool as overwritten.
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    CsBatchBeginImpl();
    vkCmdPipelineBarrier(g_cs_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
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
      NoteDccWrite(range.base, range.size, nullptr);
      NoteRawWrite(range.base, range.size);
      InvalidateTexRange(range.base, range.size);
    }
    for (auto it = g_cs_ranges.begin(); it != g_cs_ranges.end();) {
      const u64 base = it->first, end = base + it->second.guest_bytes;
      const bool touched = std::any_of(written.begin(), written.end(),
          [&](const auto& r) { return r.base < end && base < r.base + r.size; });
      if (touched) {
        CsRangeDestroy(it->second);
        it = g_cs_ranges.erase(it);
      } else {
        ++it;
      }
    }
    if (!written.empty())
      ++g_cs_writeback_gen;
  }
  g_ns_cs_out += NowNs() - _t_out0;
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
  const u64 _t0 = NowNs();
  bool all_current = true;
  // Record every readback first: one fence wait then covers all of them,
  // instead of one submit+wait per dirty range.
  CsStageReadbacks(g_cs_ranges.begin(), g_cs_ranges.end());
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
      CsRangeDestroy(it->second);
      it = g_cs_ranges.erase(it);
    } else {
      ++it;
    }
  }
  g_ns_cs_out += NowNs() - _t0;
  return all_current;
}

bool FlushCsWritesFrameEnd(Renderer& renderer, bool writeback) {
  if (g_cs_failed) {
    renderer.state = nullptr;
    return false;
  }
  const u64 _t0 = NowNs();
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
    CsStageReadbacks(g_cs_ranges.begin(), g_cs_ranges.end(), /*all=*/false);
    for (auto it = g_cs_ranges.begin(); it != g_cs_ranges.end();) {
      if (stays(it->first, it->second)) {
        g_lazy_kept_n++;
        g_lazy_kept_bytes += it->second.size;
      } else if (it->second.gpu_dirty) {
        const CsRange& fe = it->second;
        g_wb_why[std::string("frame-end") +
                 (fe.truth ? "/img" : fe.image_staging ? "/lin" : "/buf")]++;
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
        CsRangeDestroy(it->second);
        it = g_cs_ranges.erase(it);
      } else {
        ++it;
      }
    }
  }
  // The frame's command buffer may copy out of these ranges (CsSupplyTexture)
  // and is submitted next: whatever the batch still holds goes first.
  if (g_cs_batch_open && !CsBatchSubmit()) {
    renderer.state = nullptr;
    return false;
  }
  g_ns_cs_out += NowNs() - _t0;
  return all_current;
}

// Targeted variant: flush only dirty ranges overlapping [base, base+bytes).
// The per-draw guest readers (texture upload, vertex copy, cbuffer ring) call
// this instead of the full flush — flushing every dirty range at every draw
// re-tiled the whole post chain ~19x/frame.
bool FlushCsWritesRange(Renderer& renderer,
                        u64 base,
                        u64 bytes,
                        const char* why) {
  ScopeNs _flush_timer(&g_ns_cs_flush);
  // Nothing dirty anywhere: answer without touching the page index, which
  // otherwise allocates a vector, hashes a lookup per page, then sorts and
  // dedups it. That is called once per guest read -- and SotC issues 1.2M
  // DRAW_INDEX_INDIRECT packets a run, each flushing before it reads its
  // argument struct, which measured 33 seconds of a 150 s run. The index is
  // maintained on both edges (indexed when a range goes dirty, unindexed when
  // it is flushed), so empty really does mean "no writeback outstanding".
  if (g_cs_dirty_pages.empty())
    return true;
  if (g_cs_failed) {
    renderer.state = nullptr;
    return false;
  }
  if (!base || !bytes || g_cs_ranges.empty())
    return true;
  const u64 _t0 = NowNs();
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
        base::FormatTo(line, " [{:#x} img={} gpu_dirty={} sz={:#x}]",
                       (unsigned long)r,
                       f != g_cs_ranges.end() ? (int)f->second.image_staging
                                              : -1,
                       f != g_cs_ranges.end() ? (int)f->second.gpu_dirty : -1,
                       f != g_cs_ranges.end() ? (unsigned long)f->second.size
                                              : 0);
      }
      // ...and every CS range that overlaps the target at all, found or not.
      for (auto& kv : g_cs_ranges) {
        const u64 e0 = kv.first, e1 = e0 + kv.second.guest_bytes;
        if (e0 < base + bytes && base < e1)
          base::FormatTo(line, " OVERLAP[{:#x}+{:#x} img={} dirty={}]",
                         (unsigned long)e0, (unsigned long)kv.second.guest_bytes,
                         (int)kv.second.image_staging, (int)kv.second.gpu_dirty);
      }
      BASE_LOGI("csflush", "{}", line.c_str());
    }
  }
  const auto overlapping = DirtyRangesOverlapping(base, bytes);
  // This read is about to cost a fence wait, so pull EVERY dirty range's
  // results across on it rather than only the ones it asked for. The waits are
  // the expensive part and one covers them all; the ranges this draw does not
  // want stay dirty, but their mirrors are now current, so the flush that
  // eventually wants them needs no wait at all. Draw-at-a-time flushing was
  // ~25 waits a frame where a frame needs 2 or 3.
  if (!overlapping.empty()) {
    CsStageReadbacks(g_cs_ranges.begin(), g_cs_ranges.end(), /*all=*/false);
    for (u64 dirty : overlapping) {
      auto found = g_cs_ranges.find(dirty);
      if (found == g_cs_ranges.end() || !found->second.gpu_dirty)
        continue;
      CsRange& e = found->second;
      // The descriptor replay reads SRT tables, which never sit inside a
      // tiled image surface: a truth is not what it is after.
      if (e.truth && !std::strcmp(why, "srt"))
        continue;
      g_wb_why[std::string(why) + (e.truth ? "/img" : e.image_staging ? "/lin" : "/buf")]++;
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
    if (found != g_cs_ranges.end() && !CsRangeFlushOne(dirty, found->second)) {
      if (g_cs_failed) {
        renderer.state = nullptr;
        return false;
      }
      all_current = false;
    }
  }
  g_ns_cs_out += NowNs() - _t0;
  return all_current;
}

u64 CsWritebackGeneration() {
  return g_cs_writeback_gen;
}

bool CsRangeDirtyOverlapping(u64 base, u64 bytes) {
  if (g_cs_dirty_pages.empty() || !bytes)
    return false;
  // Boolean early-out, not DirtyRangesOverlapping: this runs per staging-cache
  // lookup on the draw path, and building/sorting the candidate vector there
  // costs more than the memcpy the cache hit saves for small windows.
  const u64 end = RangeEnd(base, bytes);
  const auto hits = [&](u64 other) {
    auto range = g_cs_ranges.find(other);
    return range != g_cs_ranges.end() && range->second.gpu_dirty &&
           other < end && base < RangeEnd(other, range->second.guest_bytes);
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

}  // namespace gpu::rhi
