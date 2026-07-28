/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// Compute dispatches. Each guest range a CS touches gets a persistent
// host-visible storage buffer keyed by its base address; dispatches are
// recorded into one batched command buffer, and their writes land back in guest
// memory lazily -- only when something needs guest memory to be current.

#include "gpu/rhi/renderer.h"

#include "gpu/gpu_check.h"
#include "gpu/ps4/gcn/gcn_detile.h"
#include "gpu/ps4/gcn/gcn_translate.h"
#include "gpu/vulkan/vk_capture.h"
#include "gpu/vulkan/vk_compute_hazard.h"
#include "gpu/vulkan/vk_debug.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_format.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_hash.h"
#include "gpu/vulkan/vk_perf.h"
#include "gpu/vulkan/vk_render_target.h"
#include "gpu/vulkan/vk_texture_cache.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

namespace gpu::vk {
namespace {

using rhi::ComputeInfo;

static_assert(ComputeInfo::kMaxResources == gcn::kMaxCsResources);

// A recompiled compute pipeline, cached by CS address: the SPIR-V + binding
// layout depend only on the code, so only the descriptor set + push constants +
// storage buffers are rebuilt per dispatch.
struct CsPipe {
  VkPipeline pipe = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
  uint32_t num_res = 0;
};

std::unordered_map<uint64_t, CsPipe> g_cs_pipes;

uint32_t FindComputeMemoryType(uint32_t type_bits) {
  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(g_dev.phys, &properties);
  uint32_t best = UINT32_MAX;
  int best_score = -1;
  for (uint32_t i = 0; i < properties.memoryTypeCount; i++) {
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
  return best == UINT32_MAX
             ? FindMemoryType(type_bits,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
             : best;
}

CsPipe* GetCsPipe(const ComputeInfo& ci) {
  auto it = g_cs_pipes.find(ci.cs_addr);
  if (it != g_cs_pipes.end())
    return it->second.num_res == ci.num_res ? &it->second : nullptr;
  CsPipe cp;
  cp.num_res = ci.num_res;
  VkDescriptorSetLayoutBinding binds[ComputeInfo::kMaxResources];
  for (uint32_t i = 0; i < ci.num_res; i++)
    binds[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo sl{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  sl.bindingCount = ci.num_res;
  sl.pBindings = binds;
  if (vkCreateDescriptorSetLayout(g_dev.device, &sl, nullptr, &cp.set_layout) !=
      VK_SUCCESS)
    return nullptr;
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                          64};  // 16 user-data dwords
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
  pi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pi.stage.module = cs;
  pi.stage.pName = "main";
  pi.layout = cp.layout;
  VkResult r = vkCreateComputePipelines(g_dev.device, g_dev.pipeline_cache, 1,
                                        &pi, nullptr, &cp.pipe);
  vkDestroyShaderModule(g_dev.device, cs, nullptr);
  if (r != VK_SUCCESS) {
    std::fprintf(stderr, "[gpuvk] compute pipeline failed: %d\n", (int)r);
    vkDestroyPipelineLayout(g_dev.device, cp.layout, nullptr);
    vkDestroyDescriptorSetLayout(g_dev.device, cp.set_layout, nullptr);
    return nullptr;
  }
  NameObject(VK_OBJECT_TYPE_PIPELINE, (uint64_t)cp.pipe, "cs %#llx",
             (unsigned long long)ci.cs_addr);
  g_cs_pipes[ci.cs_addr] = cp;
  return &g_cs_pipes[ci.cs_addr];
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
  const uint32_t stage_tiling = res.tiling_idx == 31 ? 31 : 8;
  return res.image_staging &&
         gcn::BuildTextureLayout32(tiled, res.width, res.height, res.pitch,
                                   res.layers, res.mip_levels, res.tiling_idx,
                                   res.pow2_pad, res.elem_bytes) &&
         gcn::BuildTextureLayout32(linear, res.width, res.height, res.pitch,
                                   res.layers, res.mip_levels, stage_tiling,
                                   res.pow2_pad, res.stage_elem_bytes) &&
         tiled.size == res.guest_size && linear.size == res.size;
}

float UnpackUnsignedFloat(uint32_t value, uint32_t mantissa_bits) {
  const uint32_t mantissa_mask = (1u << mantissa_bits) - 1;
  const uint32_t mantissa = value & mantissa_mask;
  const uint32_t exponent = (value >> mantissa_bits) & 0x1F;
  if (!exponent)
    return std::ldexp(static_cast<float>(mantissa), 1 - 15 - mantissa_bits);
  if (exponent == 0x1F)
    return mantissa ? std::numeric_limits<float>::quiet_NaN()
                    : std::numeric_limits<float>::infinity();
  return std::ldexp(1.f + static_cast<float>(mantissa) /
                              static_cast<float>(1u << mantissa_bits),
                    static_cast<int>(exponent) - 15);
}

uint32_t PackUnsignedFloat(float value, uint32_t mantissa_bits) {
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
    return static_cast<uint32_t>(
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
  return (static_cast<uint32_t>(target_exponent) << mantissa_bits) |
         static_cast<uint32_t>(mantissa);
}

void UnpackR11G11B10(uint32_t packed, uint8_t* dst) {
  const float value[4] = {
      UnpackUnsignedFloat(packed, 6),
      UnpackUnsignedFloat(packed >> 11, 6),
      UnpackUnsignedFloat(packed >> 22, 5),
      1.f,
  };
  std::memcpy(dst, value, sizeof(value));
}

uint32_t PackR11G11B10(const uint8_t* src) {
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
  const bool direct = res.elem_bytes == res.stage_elem_bytes;
  bool fills_complete_layout = direct;
  uint64_t filled_bytes = 0;
  for (uint32_t mip = 0; fills_complete_layout && mip < linear.mip_levels;
       ++mip) {
    const auto& level = linear.mips[mip];
    const uint64_t logical_bytes = static_cast<uint64_t>(level.width) *
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
  std::vector<uint8_t> tight;
  if (!direct)
    tight.resize(static_cast<size_t>(res.width) * res.height * res.elem_bytes);
  // DELTA_GPU_DETILEDUMP=<base>: write the de-tiled level-0 bytes of that guest
  // surface to <dumpdir>/detiled.bin once, so the swizzle can be checked
  // against an offline decode of the same texture.
  static const uint64_t kDetileDump = [] {
    const char* e = std::getenv("DELTA_GPU_DETILEDUMP");
    return e ? std::strtoull(e, nullptr, 0) : 0ull;
  }();
  static bool detile_dumped = false;
  for (uint32_t mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& src_level = tiled.mips[mip];
    const auto& dst_level = linear.mips[mip];
    for (uint32_t layer = 0; layer < tiled.layers; layer++) {
      uint8_t* level_dst = static_cast<uint8_t*>(dst) + dst_level.offset +
                           static_cast<uint64_t>(layer) * dst_level.pitch *
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
            std::fprintf(stderr, "[detiledump] %#lx pitch=%u h=%u elem=%u\n",
                         (unsigned long)res.base, dst_level.pitch,
                         dst_level.stored_height, res.stage_elem_bytes);
          }
        }
        continue;
      }
      if (!gcn::DetileTextureMip32(reinterpret_cast<const void*>(res.base),
                                   tight.data(), tiled, mip, layer))
        return false;
      gcn::DetileParallelRows(src_level.height, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; y++) {
          uint8_t* dst_row = level_dst + static_cast<size_t>(y) *
                                             dst_level.pitch *
                                             res.stage_elem_bytes;
          const uint8_t* src_row = tight.data() + static_cast<size_t>(y) *
                                                      src_level.width *
                                                      res.elem_bytes;
          if (res.dfmt == 6) {
            for (uint32_t x = 0; x < src_level.width; x++) {
              uint32_t packed;
              std::memcpy(&packed, src_row + static_cast<size_t>(x) * 4, 4);
              UnpackR11G11B10(packed, dst_row + static_cast<size_t>(x) * 16);
            }
          } else {
            for (uint32_t x = 0; x < src_level.width; x++) {
              uint16_t value;
              std::memcpy(&value, src_row + static_cast<size_t>(x) * 2, 2);
              const uint32_t expanded = value;
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
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(res, tiled, linear))
    return false;
  const bool direct = res.elem_bytes == res.stage_elem_bytes;
  std::vector<uint8_t> tight;
  if (!direct)
    tight.resize(static_cast<size_t>(res.width) * res.height * res.elem_bytes);
  for (uint32_t mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& dst_level = tiled.mips[mip];
    const auto& src_level = linear.mips[mip];
    for (uint32_t layer = 0; layer < tiled.layers; layer++) {
      const uint8_t* level_src =
          static_cast<const uint8_t*>(src) + src_level.offset +
          static_cast<uint64_t>(layer) * src_level.pitch *
              src_level.stored_height * res.stage_elem_bytes;
      if (direct) {
        if (!gcn::RetileTextureMip32Pitched(
                level_src,
                static_cast<size_t>(src_level.pitch) * res.stage_elem_bytes,
                reinterpret_cast<void*>(res.base), tiled, mip, layer))
          return false;
        continue;
      }
      gcn::DetileParallelRows(dst_level.height, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; y++) {
          uint8_t* dst_row = tight.data() + static_cast<size_t>(y) *
                                                dst_level.width *
                                                res.elem_bytes;
          const uint8_t* src_row = level_src + static_cast<size_t>(y) *
                                                   src_level.pitch *
                                                   res.stage_elem_bytes;
          if (res.dfmt == 6) {
            for (uint32_t x = 0; x < dst_level.width; x++) {
              const uint32_t packed =
                  PackR11G11B10(src_row + static_cast<size_t>(x) * 16);
              std::memcpy(dst_row + static_cast<size_t>(x) * 4, &packed, 4);
            }
          } else {
            for (uint32_t x = 0; x < dst_level.width; x++) {
              uint32_t expanded;
              std::memcpy(&expanded, src_row + static_cast<size_t>(x) * 4, 4);
              const uint16_t value = static_cast<uint16_t>(expanded);
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
  uint64_t size = 0;         // active staged (linear) byte size
  uint64_t guest_bytes = 0;  // guest footprint (hash + overlap checks)
  uint64_t hash = 0;         // TexHash of guest content when last in sync
  int last_validated_frame = -1;
  int last_used_frame = -1;
  bool gpu_dirty = false;      // buffer newer than guest memory
  bool pending_batch = false;  // referenced by the open dispatch batch
  bool image_staging = false;
  // Staged from a live render-target image rather than guest memory; content
  // changes every frame regardless of the guest bytes, so validity is
  // per-frame (last_rt_frame), never the guest content hash.
  bool rt_sourced = false;
  int last_rt_frame = -1;
  ComputeInfo::Res res;  // writeback needs the full layout description
};

bool SameCsResourceShape(const ComputeInfo::Res& a, const ComputeInfo::Res& b) {
  if (a.image_staging != b.image_staging)
    return false;
  if (!a.image_staging)
    return true;
  return a.width == b.width && a.height == b.height && a.pitch == b.pitch &&
         a.layers == b.layers && a.mip_levels == b.mip_levels &&
         a.tiling_idx == b.tiling_idx && a.elem_bytes == b.elem_bytes &&
         a.stage_elem_bytes == b.stage_elem_bytes && a.dfmt == b.dfmt &&
         a.pow2_pad == b.pow2_pad;
}

// A live image (color RT or depth target) aliasing a CS resource's guest
// range. Both bridge directions (StageCsRangeFromRt / UploadCsRangeToRt) use
// the same lookup, shape checks and barrier recipe.
struct CsAliasedImage {
  VkImage image = VK_NULL_HANDLE;
  uint32_t w = 0, h = 0;
  uint32_t elem_bytes = 0;
  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  VkImageLayout submitted_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  bool is_depth = false;
};

// True when `base` names a live target the compute bridges apply to. The
// UNDEFINED-submitted-layout case (target created this frame, no submission
// yet) reports false: there is nothing real to copy either way yet.
bool FindCsAliasedImage(uint64_t base, CsAliasedImage& out) {
  auto rt_it = g_rts.find(base);
  if (rt_it != g_rts.end()) {
    RTarget& rt = rt_it->second;
    if (!rt.image || rt.is_depth || !rt.ever_rendered)
      return false;
    out = {rt.image,
           rt.w,
           rt.h,
           FormatBytes(rt.fmt),
           VK_IMAGE_ASPECT_COLOR_BIT,
           rt.submitted_layout,
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
           true};
    return out.submitted_layout != VK_IMAGE_LAYOUT_UNDEFINED;
  }
  return false;
}

bool CsAliasedBase(uint64_t base) {
  return g_rts.find(base) != g_rts.end() ||
         g_depths.find(base) != g_depths.end();
}

VkAccessFlags AliasedImageAccess(const CsAliasedImage& img, VkImageLayout l) {
  if (!img.is_depth)
    return ColorImageAccess(l);
  switch (l) {
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
      return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
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

// The CS side of an image<->buffer bridge copy expects the linear staged
// layout; reject targets whose shape disagrees with the descriptor.
bool AliasedShapeMatches(const CsAliasedImage& img,
                         const ComputeInfo::Res& res,
                         const char* dir) {
  if (res.mip_levels == 1 && res.layers == 1 && img.w == res.width &&
      img.h == res.height && img.elem_bytes == res.stage_elem_bytes)
    return true;
  static int warned = 0;
  if (warned < 8) {
    warned++;
    std::fprintf(stderr,
                 "[gpuvk] cs %s live %s target %#llx shape mismatch: image "
                 "%ux%u %uB vs cs %ux%u mips=%u %uB -> falling back to guest "
                 "memory\n",
                 dir, img.is_depth ? "depth" : "color",
                 (unsigned long long)res.base, img.w, img.h, img.elem_bytes,
                 res.width, res.height, res.mip_levels, res.stage_elem_bytes);
  }
  return false;
}

// Record one bridge copy (image->buffer or buffer->image), submit and wait.
bool RunAliasedCopy(const CsAliasedImage& img,
                    const ComputeInfo::Res& res,
                    CsRange& e,
                    bool to_image) {
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(res, tiled, linear))
    return false;
  const auto& level = linear.mips[0];
  const uint64_t copy_bytes =
      static_cast<uint64_t>(level.pitch) * img.h * res.stage_elem_bytes;
  if (level.offset + copy_bytes > e.cap)
    return false;
  // Host-zero any padding an image->buffer copy does not cover (host writes
  // are made available by the submission).
  if (!to_image && (level.offset != 0 || copy_bytes < res.size))
    std::memset(e.map, 0, res.size);

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
    bb.buffer = e.buf;
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
  copy.bufferOffset = level.offset;
  copy.bufferRowLength = level.pitch;
  copy.imageSubresource = {img.aspect, 0, 0, 1};
  copy.imageExtent = {img.w, img.h, 1};
  if (to_image)
    vkCmdCopyBufferToImage(c, e.buf, img.image, transfer_layout, 1, &copy);
  else
    vkCmdCopyImageToBuffer(c, img.image, transfer_layout, e.buf, 1, &copy);
  AliasedImageBarrier(c, img, transfer_layout, img.submitted_layout,
                      transfer_access,
                      AliasedImageAccess(img, img.submitted_layout));
  if (!to_image) {
    VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = e.buf;
    bb.offset = 0;
    bb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(
        c, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
        nullptr, 1, &bb, 0, nullptr);
  }
  const VkResult end_result = vkEndCommandBuffer(c);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &c;
  VkResult r = end_result;
  if (r == VK_SUCCESS)
    r = vkResetFences(g_dev.device, 1, &g_dev.fence);
  if (r == VK_SUCCESS)
    r = vkQueueSubmit(g_dev.queue, 1, &si, g_dev.fence);
  if (r == VK_SUCCESS)
    r = vkWaitForFences(g_dev.device, 1, &g_dev.fence, VK_TRUE, UINT64_MAX);
  vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &c);
  if (r != VK_SUCCESS) {
    std::fprintf(stderr, "[gpuvk] cs %s bridge copy failed: %d (base=%#llx)\n",
                 to_image ? "upload" : "staging", (int)r,
                 (unsigned long long)res.base);
    return false;
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
  CsAliasedImage img;
  if (!FindCsAliasedImage(res.base, img))
    return false;
  if (!AliasedShapeMatches(img, res, "reads"))
    return false;
  return RunAliasedCopy(img, res, e, /*to_image=*/false);
}

// The reverse: a CS result written to a range that a live render/depth target
// aliases must also land in the VkImage, because draws sample the image,
// never guest memory (SotC's exposure/bloom compute writes the adapted scene
// into an RT the tonemap then samples; its depth downsample writes the
// half-res depth pyramid). e.buf already holds the linear pixel data the
// dispatch produced, so upload straight from it. Guest memory was refreshed
// by the caller either way; a shape mismatch just leaves the image stale.
bool UploadCsRangeToRt(uint64_t base, CsRange& e) {
  CsAliasedImage img;
  if (!e.image_staging || !FindCsAliasedImage(base, img))
    return true;  // nothing to refresh
  if (!AliasedShapeMatches(img, e.res, "writes"))
    return true;
  if (!RunAliasedCopy(img, e.res, e, /*to_image=*/true))
    return false;
  if (!img.is_depth)
    g_rts[base].ever_rendered = true;  // CS content is real content
  return true;
}

std::unordered_map<uint64_t, CsRange> g_cs_ranges;
uint64_t g_cs_range_bytes = 0;
constexpr uint32_t kCsDirtyPageShift = 16;
std::unordered_map<uint64_t, std::vector<uint64_t>> g_cs_dirty_pages;

uint64_t RangeEnd(uint64_t base, uint64_t bytes) {
  return bytes > UINT64_MAX - base ? UINT64_MAX : base + bytes;
}

void IndexDirtyRange(uint64_t base, uint64_t bytes) {
  if (!bytes)
    return;
  const uint64_t end = RangeEnd(base, bytes);
  for (uint64_t page = base >> kCsDirtyPageShift;
       page <= (end - 1) >> kCsDirtyPageShift; page++)
    g_cs_dirty_pages[page].push_back(base);
}

void UnindexDirtyRange(uint64_t base, uint64_t bytes) {
  if (!bytes)
    return;
  const uint64_t end = RangeEnd(base, bytes);
  for (uint64_t page = base >> kCsDirtyPageShift;
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

std::vector<uint64_t> DirtyRangesOverlapping(uint64_t base,
                                             uint64_t bytes,
                                             uint64_t exclude = UINT64_MAX) {
  std::vector<uint64_t> candidates;
  if (!bytes)
    return candidates;
  const uint64_t end = RangeEnd(base, bytes);
  for (uint64_t page = base >> kCsDirtyPageShift;
       page <= (end - 1) >> kCsDirtyPageShift; page++) {
    auto found = g_cs_dirty_pages.find(page);
    if (found != g_cs_dirty_pages.end())
      candidates.insert(candidates.end(), found->second.begin(),
                        found->second.end());
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [&](uint64_t other) {
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
uint64_t RangeHash(uint64_t base, uint64_t bytes) {
  if (bytes <= 16384)
    return TexHash(base, bytes);
  constexpr uint64_t kPrime = 1099511628211ull;
  uint64_t h = 1469598103934665603ull ^ (bytes * kPrime);
  const uint64_t step = (bytes - 64) / 255;
  for (uint32_t i = 0; i < 256; i++) {
    uint64_t w[8];
    std::memcpy(w, reinterpret_cast<const void*>(base + i * step), 64);
    for (int j = 0; j < 8; j++)
      h = (h ^ w[j]) * kPrime;
  }
  return h;
}

bool CsRangeEnsureBuffer(CsRange& e, VkDeviceSize size) {
  if (e.buf && e.cap >= size)
    return true;
  if (e.map) {
    vkUnmapMemory(g_dev.device, e.mem);
    e.map = nullptr;
  }
  if (e.buf) {
    vkDestroyBuffer(g_dev.device, e.buf, nullptr);
    e.buf = VK_NULL_HANDLE;
  }
  if (e.mem) {
    vkFreeMemory(g_dev.device, e.mem, nullptr);
    e.mem = VK_NULL_HANDLE;
  }
  g_cs_range_bytes -= e.cap;
  e.cap = 0;
  VkDeviceSize cap = (size + 0xFFFF) & ~VkDeviceSize(0xFFFF);
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = cap;
  // TRANSFER_DST: RT-backed inputs are staged by an image->buffer copy on the
  // queue (StageCsRangeFromRt) instead of a CPU memcpy from guest memory.
  bi.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &e.buf) != VK_SUCCESS) {
    e.buf = VK_NULL_HANDLE;
    return false;
  }
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_dev.device, e.buf, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = FindComputeMemoryType(mr.memoryTypeBits);
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &e.mem) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, e.buf, nullptr);
    e.buf = VK_NULL_HANDLE;
    e.mem = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(g_dev.device, e.buf, e.mem, 0);
  vkMapMemory(g_dev.device, e.mem, 0, cap, 0, &e.map);
  e.cap = cap;
  g_cs_range_bytes += cap;
  return true;
}

void CsRangeDestroy(CsRange& e) {
  if (e.map)
    vkUnmapMemory(g_dev.device, e.mem);
  if (e.buf)
    vkDestroyBuffer(g_dev.device, e.buf, nullptr);
  if (e.mem)
    vkFreeMemory(g_dev.device, e.mem, nullptr);
  g_cs_range_bytes -= e.cap;
  e = CsRange{};
}

// Dispatch batching: dispatches are recorded into one command buffer and
// submitted/waited only when something needs their results (a flush point,
// a staging hazard, or the batch cap). 228 individual submit+fence round
// trips per frame were ~40% of the whole compute cost.
bool g_cs_batch_open = false;
bool g_cs_failed = false;
uint32_t g_cs_batch_count = 0;
VkFence g_cs_batch_fence = VK_NULL_HANDLE;
bool g_cs_stage_pending[ComputeInfo::kMaxResources] = {};
std::unordered_map<VkBuffer, ComputeBufferAccess> g_cs_batch_access;

bool CsBatchFlush() {
  if (!g_cs_batch_open)
    return !g_cs_failed;
  const uint64_t t0 = NowNs();
  CmdEndLabel(g_cs_cmd);  // close the "cs batch" scope
  const VkResult end_result = vkEndCommandBuffer(g_cs_cmd);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &g_cs_cmd;
  VkResult submit_result = end_result;
  if (submit_result == VK_SUCCESS)
    submit_result = vkResetFences(g_dev.device, 1, &g_cs_batch_fence);
  if (submit_result == VK_SUCCESS)
    submit_result = vkQueueSubmit(g_dev.queue, 1, &si, g_cs_batch_fence);
  const VkResult wait_result =
      submit_result == VK_SUCCESS
          ? vkWaitForFences(g_dev.device, 1, &g_cs_batch_fence, VK_TRUE,
                            UINT64_MAX)
          : submit_result;
  if (wait_result != VK_SUCCESS) {
    std::fprintf(stderr,
                 "[gpuvk] cs batch DEVICE FAULT: end=%d submit=%d wait=%d "
                 "n=%u\n",
                 (int)end_result, (int)submit_result, (int)wait_result,
                 g_cs_batch_count);
    ReportDeviceFault(g_dev);
    g_cs_failed = true;
    g_ns_cs_gpu += NowNs() - t0;
    return false;
  }
  if (vkResetDescriptorPool(g_dev.device, g_cs_desc_pool, 0) != VK_SUCCESS) {
    g_cs_failed = true;
    g_ns_cs_gpu += NowNs() - t0;
    return false;
  }
  g_cs_batch_open = false;
  g_cs_batch_count = 0;
  for (auto& kv : g_cs_ranges)
    kv.second.pending_batch = false;
  std::memset(g_cs_stage_pending, 0, sizeof g_cs_stage_pending);
  g_cs_batch_access.clear();
  g_ns_cs_gpu += NowNs() - t0;
  return true;
}

// Write one dirty range back to guest memory (retile for images) and re-stamp
// its hash so the next validation sees guest == buffer.
bool CsRangeFlushOne(uint64_t base, CsRange& e) {
  if (!e.gpu_dirty)
    return true;
  if (e.pending_batch && !CsBatchFlush())
    return false;  // results must exist before readback
  if (g_cs_failed)
    return false;
  g_cs_flush_n++;
  if (e.image_staging) {
    if (!WritebackCsImage(e.res, e.map)) {
      static int logged = 0;
      if (logged++ < 8)
        std::fprintf(stderr,
                     "[gpuvk] cs image writeback failed base=%#llx "
                     "(range stays stale)\n",
                     (unsigned long long)base);
      return false;
    }
  } else {
    std::memcpy(reinterpret_cast<void*>(base), e.map, e.size);
  }
  UploadCsRangeToRt(base, e);  // refresh a live RT image aliasing the range
  InvalidateTexRange(base, e.guest_bytes);
  UnindexDirtyRange(base, e.guest_bytes);
  e.gpu_dirty = false;
  e.hash = RangeHash(base, e.guest_bytes);
  e.last_validated_frame = g_frame.num;
  return true;
}

// Ensure staging slot i can hold `size` bytes (grow-on-demand, kept mapped).
bool CsEnsureStage(uint32_t i, VkDeviceSize size) {
  GPU_BUGCHECK(i < ComputeInfo::kMaxResources, "stage index %u out of bounds",
               i);
  CsStage& s = g_cs_stage[i];
  if (s.buf && s.cap >= size)
    return true;
  if (s.map) {
    vkUnmapMemory(g_dev.device, s.mem);
    s.map = nullptr;
  }
  if (s.buf) {
    vkDestroyBuffer(g_dev.device, s.buf, nullptr);
    s.buf = VK_NULL_HANDLE;
  }
  if (s.mem) {
    vkFreeMemory(g_dev.device, s.mem, nullptr);
    s.mem = VK_NULL_HANDLE;
  }
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
  ai.memoryTypeIndex = FindComputeMemoryType(mr.memoryTypeBits);
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &s.mem) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, s.buf, nullptr);
    s.buf = VK_NULL_HANDLE;
    s.mem = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(g_dev.device, s.buf, s.mem, 0);
  vkMapMemory(g_dev.device, s.mem, 0, cap, 0, &s.map);
  s.cap = cap;
  return true;
}

struct ScopeCs {
  uint64_t t0 = NowNs();
  ~ScopeCs() {
    g_ns_cs += NowNs() - t0;
    g_cs_count++;
  }
};

}  // namespace
}  // namespace gpu::vk

namespace gpu::rhi {
using namespace gpu::vk;

bool Dispatch(Renderer& renderer, const ComputeInfo& ci) {
  if (g_cs_failed) {
    renderer.state = nullptr;
    return false;
  }
  if (!renderer.available() || !ci.recomp || !ci.recomp->ok || !ci.num_res ||
      ci.num_res > g_dev.max_cs_resources)
    return false;
  ScopeCs _cs;
  for (uint32_t i = 0; i < ci.num_res; i++)
    g_cs_bytes += ci.res[i].size;
  for (uint32_t i = 0; i < ci.num_res; i++)
    if (ci.res[i].size > g_dev.max_storage_buffer_range)
      return false;
  for (uint32_t i = 0; i < ci.num_res; i++) {
    if (ci.res[i].zero_fill)
      continue;
    const uint64_t guest_bytes =
        ci.res[i].guest_size ? ci.res[i].guest_size : ci.res[i].size;
    const VkDeviceSize size =
        ci.res[i].size ? ((ci.res[i].size + 3) & ~VkDeviceSize(3)) : 4;
    for (uint32_t j = 0; j < i; j++) {
      if (ci.res[j].zero_fill || ci.res[j].base != ci.res[i].base)
        continue;
      const uint64_t other_guest_bytes =
          ci.res[j].guest_size ? ci.res[j].guest_size : ci.res[j].size;
      const VkDeviceSize other_size =
          ci.res[j].size ? ((ci.res[j].size + 3) & ~VkDeviceSize(3)) : 4;
      if (size != other_size || guest_bytes != other_guest_bytes ||
          !SameCsResourceShape(ci.res[i], ci.res[j]))
        return false;
    }
  }
  CsPipe* cp = GetCsPipe(ci);
  if (!cp)
    return false;
  static const bool verbose = std::getenv("DELTA_GPU_CSGPU_VERBOSE") != nullptr;

  // Persistent command buffer + descriptor pool (created once, reused).
  if (g_cs_cmd == VK_NULL_HANDLE) {
    VkCommandBufferAllocateInfo ca{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = g_dev.pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_dev.device, &ca, &g_cs_cmd) != VK_SUCCESS) {
      g_cs_cmd = VK_NULL_HANDLE;
      return false;
    }
  }
  if (g_cs_desc_pool == VK_NULL_HANDLE) {
    // Sized for a whole batch of dispatches between flushes.
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            256 * ComputeInfo::kMaxResources};
    VkDescriptorPoolCreateInfo pci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 256;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(g_dev.device, &pci, nullptr, &g_cs_desc_pool) !=
        VK_SUCCESS) {
      g_cs_desc_pool = VK_NULL_HANDLE;
      return false;
    }
  }
  if (g_cs_batch_fence == VK_NULL_HANDLE) {
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(g_dev.device, &fci, nullptr, &g_cs_batch_fence) !=
        VK_SUCCESS) {
      g_cs_batch_fence = VK_NULL_HANDLE;
      return false;
    }
  }

  // DELTA_GPU_CSLIST: per-dispatch resource staging list for the first 200
  // dispatches — shows what the chain actually round-trips per frame.
  static const bool kCsList = std::getenv("DELTA_GPU_CSLIST") != nullptr;
  static uint32_t cs_listed = 0;
  if (kCsList && g_frame.num > 25 && cs_listed < 200) {
    cs_listed++;
    for (uint32_t i = 0; i < ci.num_res; i++)
      std::fprintf(stderr,
                   "[cslist] cs=%#llx bind=%u base=%#lx size=%#lx %s%s\n",
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
  const uint64_t _t_in0 = NowNs();
  VkBuffer bind_buf[ComputeInfo::kMaxResources];
  VkDeviceSize sz[ComputeInfo::kMaxResources];
  for (uint32_t i = 0; i < ci.num_res; i++) {
    sz[i] = ci.res[i].size ? ((ci.res[i].size + 3) & ~VkDeviceSize(3)) : 4;
    if (ci.res[i].zero_fill) {
      // Growing the scratch slot recreates its buffer; a pending batched
      // dispatch still references the old handle.
      if (g_cs_stage_pending[i] && g_cs_stage[i].cap < sz[i] &&
          !CsBatchFlush()) {
        renderer.state = nullptr;
        return false;
      }
      if (!CsEnsureStage(i, sz[i]))
        return false;
      bind_buf[i] = g_cs_stage[i].buf;
      continue;
    }
    const uint64_t base = ci.res[i].base;
    const uint64_t guest_bytes =
        ci.res[i].guest_size ? ci.res[i].guest_size : ci.res[i].size;
    // A read overlapping some OTHER dirty range must see that data through
    // guest memory: flush those first.
    for (uint64_t dirty : DirtyRangesOverlapping(base, guest_bytes, base)) {
      auto found = g_cs_ranges.find(dirty);
      if (found != g_cs_ranges.end() && !CsRangeFlushOne(dirty, found->second))
        return false;
    }
    CsRange& e = g_cs_ranges[base];
    const bool same_shape = e.buf && e.size == static_cast<uint64_t>(sz[i]) &&
                            e.guest_bytes == guest_bytes &&
                            SameCsResourceShape(e.res, ci.res[i]);
    if (!same_shape && e.gpu_dirty)
      if (!CsRangeFlushOne(base, e))
        return false;  // reshaped: keep its data
    if (e.pending_batch && (!e.buf || e.cap < sz[i]) && !CsBatchFlush()) {
      renderer.state = nullptr;
      return false;  // growth would destroy a buffer the batch references
    }
    const bool buffer_reused =
        e.buf && e.cap >= static_cast<VkDeviceSize>(sz[i]);
    if (!CsRangeEnsureBuffer(e, sz[i]))
      return false;
    if (!buffer_reused)
      NameObject(VK_OBJECT_TYPE_BUFFER, (uint64_t)e.buf, "csbuf %#llx",
                 (unsigned long long)base);
    bool valid =
        same_shape && (e.gpu_dirty || e.last_validated_frame == g_frame.num);
    // A read whose base is a live render target must be staged from the
    // VkImage: the guest bytes under an RT are stale (draws never write them
    // back), and the image content changes every frame regardless of the
    // guest hash. Attempted at most once per frame per range; a CS-written
    // buffer (gpu_dirty) stays authoritative.
    const bool rt_backed = ci.res[i].image_staging && CsAliasedBase(base);
    const bool rt_attempt = rt_backed && !e.gpu_dirty &&
                            e.last_rt_frame != static_cast<int>(g_frame.num);
    if (rt_attempt)
      valid = false;
    else if (rt_backed && !e.gpu_dirty && e.rt_sourced)
      valid = same_shape;
    if (!valid && same_shape && !rt_attempt) {
      const uint64_t h = RangeHash(base, guest_bytes);
      if (h == e.hash)
        valid = true;
      else
        e.hash = h;
      e.last_validated_frame = g_frame.num;
    }
    if (!valid) {
      // CPU write into a buffer a pending batched dispatch reads/writes.
      if (e.pending_batch && !CsBatchFlush()) {
        renderer.state = nullptr;
        return false;
      }
      if (rt_attempt) {
        e.last_rt_frame = static_cast<int>(g_frame.num);
        e.rt_sourced = StageCsRangeFromRt(ci.res[i], e);
        // DELTA_GPU_CSRT: trace every RT-backed staging decision.
        static const bool kCsRtTrace = std::getenv("DELTA_GPU_CSRT") != nullptr;
        static int rt_trace_logged = 0;
        if (kCsRtTrace && rt_trace_logged < 200) {
          rt_trace_logged++;
          std::fprintf(stderr, "[csrt] f%d base=%#lx %ux%u -> %s\n",
                       (int)g_frame.num, (unsigned long)base, ci.res[i].width,
                       ci.res[i].height,
                       e.rt_sourced ? "staged-from-RT" : "guest-fallback");
        }
      }
      if (!rt_attempt || !e.rt_sourced) {
        if (ci.res[i].image_staging) {
          if (!StageCsImage(ci.res[i], e.map))
            return false;
        } else {
          std::memcpy(e.map, reinterpret_cast<const void*>(base),
                      ci.res[i].size);
          if (sz[i] > ci.res[i].size)
            std::memset(static_cast<uint8_t*>(e.map) + ci.res[i].size, 0,
                        sz[i] - ci.res[i].size);
        }
        e.rt_sourced = false;
        if (rt_attempt) {  // fell back: keep guest-hash bookkeeping coherent
          e.hash = RangeHash(base, guest_bytes);
          e.last_validated_frame = g_frame.num;
        }
      }
      if (!same_shape) {
        e.hash = RangeHash(base, guest_bytes);
        e.last_validated_frame = g_frame.num;
      }
      e.gpu_dirty = false;
      g_cs_stage_n++;
      g_cs_stage_bytes += sz[i];
    }
    e.size = sz[i];
    e.guest_bytes = guest_bytes;
    e.image_staging = ci.res[i].image_staging;
    e.res = ci.res[i];
    e.last_used_frame = g_frame.num;
    bind_buf[i] = e.buf;
  }
  // Re-resolve handles: a later binding sharing an earlier binding's base may
  // have grown (destroyed + recreated) that range's buffer.
  for (uint32_t i = 0; i < ci.num_res; i++)
    if (!ci.res[i].zero_fill)
      bind_buf[i] = g_cs_ranges[ci.res[i].base].buf;

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
    if (!CsBatchFlush()) {
      renderer.state = nullptr;
      return false;
    }
    if (vkAllocateDescriptorSets(g_dev.device, &da, &set) != VK_SUCCESS)
      return false;
  }
  VkDescriptorBufferInfo dbi[ComputeInfo::kMaxResources];
  VkWriteDescriptorSet wr[ComputeInfo::kMaxResources];
  for (uint32_t i = 0; i < ci.num_res; i++) {
    dbi[i] = {bind_buf[i], 0, sz[i]};
    wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr[i].dstSet = set;
    wr[i].dstBinding = ci.res[i].binding;
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo = &dbi[i];
  }
  vkUpdateDescriptorSets(g_dev.device, ci.num_res, wr, 0, nullptr);

  // Record the dispatch into the open batch. Submission + the fence wait
  // happen at the next flush point, not here.
  if (!g_cs_batch_open) {
    vkResetCommandBuffer(g_cs_cmd, 0);
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(g_cs_cmd, &cbi);
    CmdBeginLabel(g_cs_cmd, "cs batch (frame %llu)",
                  (unsigned long long)g_frame.num);
    g_cs_batch_open = true;
  }
  VkBufferMemoryBarrier zero_before[ComputeInfo::kMaxResources];
  VkBufferMemoryBarrier zero_after[ComputeInfo::kMaxResources];
  uint32_t zero_count = 0;
  for (uint32_t i = 0; i < ci.num_res; i++) {
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
    for (uint32_t i = 0; i < ci.num_res; i++)
      if (ci.res[i].zero_fill)
        vkCmdFillBuffer(g_cs_cmd, bind_buf[i], 0, sz[i], 0);
    vkCmdPipelineBarrier(g_cs_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         zero_count, zero_after, 0, nullptr);
  }
  vkCmdBindPipeline(g_cs_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp->pipe);
  vkCmdBindDescriptorSets(g_cs_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp->layout,
                          0, 1, &set, 0, nullptr);
  vkCmdPushConstants(g_cs_cmd, cp->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 64,
                     ci.user_data);
  VkBufferMemoryBarrier barriers[ComputeInfo::kMaxResources];
  uint32_t barrier_count = 0;
  VkBuffer unique_buffers[ComputeInfo::kMaxResources];
  bool unique_writes[ComputeInfo::kMaxResources] = {};
  uint32_t unique_count = 0;
  for (uint32_t i = 0; i < ci.num_res; i++) {
    uint32_t j = 0;
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
  for (uint32_t i = 0; i < unique_count; i++) {
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
  vkCmdDispatch(g_cs_cmd, ci.groups[0], ci.groups[1], ci.groups[2]);
  for (uint32_t i = 0; i < ci.num_res; i++) {
    if (ci.res[i].zero_fill) {
      g_cs_stage_pending[i] = true;
    } else {
      auto it = g_cs_ranges.find(ci.res[i].base);
      if (it != g_cs_ranges.end())
        it->second.pending_batch = true;
    }
  }
  if ((++g_cs_batch_count >= 128 || verbose) && !CsBatchFlush()) {
    renderer.state = nullptr;
    return false;
  }

  // Mark written ranges GPU-dirty. Guest memory catches up lazily at the next
  // flush point (draw / DMA / frame end) — writing every dispatch's outputs
  // back immediately (the image retile especially) was ~100ms/frame.
  const uint64_t _t_out0 = NowNs();
  for (uint32_t i = 0; i < ci.num_res; i++) {
    if (!ci.res[i].written || ci.res[i].zero_fill)
      continue;
    auto it = g_cs_ranges.find(ci.res[i].base);
    if (it == g_cs_ranges.end())
      continue;
    if (!it->second.gpu_dirty)
      IndexDirtyRange(ci.res[i].base, it->second.guest_bytes);
    it->second.gpu_dirty = true;
    if (verbose) {
      const uint8_t* b = static_cast<const uint8_t*>(it->second.map);
      uint64_t nz = 0,
               step = ci.res[i].size > 65536 ? ci.res[i].size / 65536 : 1;
      for (uint64_t k = 0; k < ci.res[i].size; k += step)
        nz += b[k] != 0;
      std::fprintf(stderr,
                   "[csgpu] gpu wrote base=%#lx size=%lu nonzero=%lu/%lu\n",
                   (unsigned long)ci.res[i].base, (unsigned long)ci.res[i].size,
                   (unsigned long)nz, (unsigned long)(ci.res[i].size / step));
    }
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
  const uint64_t _t0 = NowNs();
  bool all_current = true;
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
    if (g_cs_range_bytes > (1ull << 30) && !it->second.gpu_dirty &&
        !it->second.pending_batch &&
        it->second.last_used_frame + 300 < g_frame.num) {
      CsRangeDestroy(it->second);
      it = g_cs_ranges.erase(it);
    } else {
      ++it;
    }
  }
  g_ns_cs_out += NowNs() - _t0;
  return all_current;
}

// Targeted variant: flush only dirty ranges overlapping [base, base+bytes).
// The per-draw guest readers (texture upload, vertex copy, cbuffer ring) call
// this instead of the full flush — flushing every dirty range at every draw
// re-tiled the whole post chain ~19x/frame.
bool FlushCsWritesRange(Renderer& renderer, uint64_t base, uint64_t bytes) {
  if (g_cs_failed) {
    renderer.state = nullptr;
    return false;
  }
  if (!base || !bytes || g_cs_ranges.empty())
    return true;
  const uint64_t _t0 = NowNs();
  bool all_current = true;
  for (uint64_t dirty : DirtyRangesOverlapping(base, bytes)) {
    auto found = g_cs_ranges.find(dirty);
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

}  // namespace gpu::rhi
