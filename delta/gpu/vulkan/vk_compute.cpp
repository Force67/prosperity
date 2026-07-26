/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// Compute dispatches. Each guest range a CS touches gets a persistent
// host-visible storage buffer keyed by its base address; dispatches are
// recorded into one batched command buffer, and their writes land back in guest
// memory lazily -- only when something needs guest memory to be current.

#include "rhi/renderer.h"

#include "gcn/gcn_detile.h"
#include "gcn/gcn_translate.h"
#include "vulkan/vk_capture.h"
#include "vulkan/vk_device.h"
#include "vulkan/vk_frame.h"
#include "vulkan/vk_hash.h"
#include "vulkan/vk_perf.h"
#include "vulkan/vk_texture_cache.h"

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

// A recompiled compute pipeline, cached by CS address: the SPIR-V + binding layout
// depend only on the code, so only the descriptor set + push constants + storage
// buffers are rebuilt per dispatch.
struct CsPipe {
  VkPipeline pipe = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  uint32_t nres = 0;
};

std::unordered_map<uint64_t, CsPipe> g_csPipes;

CsPipe *getCsPipe(const ComputeInfo &ci) {
  auto it = g_csPipes.find(ci.csAddr);
  if (it != g_csPipes.end())
    return it->second.nres == ci.nres ? &it->second : nullptr;
  CsPipe cp; cp.nres = ci.nres;
  VkDescriptorSetLayoutBinding binds[ComputeInfo::kMaxResources];
  for (uint32_t i = 0; i < ci.nres; i++)
    binds[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo sl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  sl.bindingCount = ci.nres; sl.pBindings = binds;
  if (vkCreateDescriptorSetLayout(g_dev.device, &sl, nullptr, &cp.setLayout) != VK_SUCCESS)
    return nullptr;
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};  // 16 user-data dwords
  VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  li.setLayoutCount = 1; li.pSetLayouts = &cp.setLayout;
  li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pcr;
  if (vkCreatePipelineLayout(g_dev.device, &li, nullptr, &cp.layout) != VK_SUCCESS) {
    vkDestroyDescriptorSetLayout(g_dev.device, cp.setLayout, nullptr); return nullptr; }
  VkShaderModule cs = makeModule(ci.recomp->spirv.data(), ci.recomp->spirv.size() * 4);
  VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pi.stage.module = cs; pi.stage.pName = "main";
  pi.layout = cp.layout;
  VkResult r = vkCreateComputePipelines(g_dev.device, VK_NULL_HANDLE, 1, &pi, nullptr, &cp.pipe);
  vkDestroyShaderModule(g_dev.device, cs, nullptr);
  if (r != VK_SUCCESS) {
    std::fprintf(stderr, "[gpuvk] compute pipeline failed: %d\n", (int)r);
    vkDestroyPipelineLayout(g_dev.device, cp.layout, nullptr);
    vkDestroyDescriptorSetLayout(g_dev.device, cp.setLayout, nullptr);
    return nullptr;
  }
  g_csPipes[ci.csAddr] = cp;
  return &g_csPipes[ci.csAddr];
}

// Persistent compute staging. Dispatches are serialized behind the fence, so one set
// of staging buffers (+ descriptor pool + command buffer) is reused across every
// dispatch: this avoids the per-dispatch vkCreateBuffer/vkAllocateMemory/pool/cmd-
// buffer churn (~3ms/frame). The buffers are HOST_CACHED so the copy-BACK read after
// the dispatch hits cache instead of stalling on write-combined memory (which was
// ~25ms/frame for Doom64's 8 MB atlas: the dominant compute cost).
struct CsStage {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void *map = nullptr;
  VkDeviceSize cap = 0;
};

CsStage g_csStage[ComputeInfo::kMaxResources];
VkDescriptorPool g_csDescPool = VK_NULL_HANDLE;
VkCommandBuffer g_csCmd = VK_NULL_HANDLE;

bool buildCsImageLayouts(const ComputeInfo::Res &res,
                         gcn::TextureLayout32 &tiled,
                         gcn::TextureLayout32 &linear) {
  const uint32_t stageTiling = res.tilingIdx == 31 ? 31 : 8;
  return res.imageStaging &&
         gcn::BuildTextureLayout32(tiled, res.width, res.height, res.pitch,
                                   res.layers, res.mipLevels, res.tilingIdx,
                                   res.pow2Pad, res.elemBytes) &&
         gcn::BuildTextureLayout32(linear, res.width, res.height, res.pitch,
                                   res.layers, res.mipLevels, stageTiling,
                                   res.pow2Pad, res.stageElemBytes) &&
         tiled.size == res.guestSize && linear.size == res.size;
}

float unpackUnsignedFloat(uint32_t value, uint32_t mantissaBits) {
  const uint32_t mantissaMask = (1u << mantissaBits) - 1;
  const uint32_t mantissa = value & mantissaMask;
  const uint32_t exponent = (value >> mantissaBits) & 0x1F;
  if (!exponent)
    return std::ldexp(static_cast<float>(mantissa), 1 - 15 - mantissaBits);
  if (exponent == 0x1F)
    return mantissa ? std::numeric_limits<float>::quiet_NaN()
                    : std::numeric_limits<float>::infinity();
  return std::ldexp(1.f + static_cast<float>(mantissa) /
                              static_cast<float>(1u << mantissaBits),
                    static_cast<int>(exponent) - 15);
}

uint32_t packUnsignedFloat(float value, uint32_t mantissaBits) {
  if (std::isnan(value)) return (0x1Fu << mantissaBits) | 1u;
  if (value <= 0.f) return 0;
  if (std::isinf(value)) return 0x1Fu << mantissaBits;
  int exponent;
  const float fraction = std::frexp(value, &exponent);
  int targetExponent = exponent - 1 + 15;
  if (targetExponent <= 0) {
    const long mantissa =
        std::lround(std::ldexp(value, 14 + mantissaBits));
    return static_cast<uint32_t>(std::clamp<long>(
        mantissa, 0, static_cast<long>(1u << mantissaBits)));
  }
  if (targetExponent >= 0x1F) return 0x1Fu << mantissaBits;
  long mantissa = std::lround(
      (fraction * 2.f - 1.f) * static_cast<float>(1u << mantissaBits));
  if (mantissa == static_cast<long>(1u << mantissaBits)) {
    mantissa = 0;
    if (++targetExponent >= 0x1F) return 0x1Fu << mantissaBits;
  }
  return (static_cast<uint32_t>(targetExponent) << mantissaBits) |
         static_cast<uint32_t>(mantissa);
}

void unpackR11G11B10(uint32_t packed, uint8_t *dst) {
  const float value[4] = {
      unpackUnsignedFloat(packed, 6),
      unpackUnsignedFloat(packed >> 11, 6),
      unpackUnsignedFloat(packed >> 22, 5),
      1.f,
  };
  std::memcpy(dst, value, sizeof(value));
}

uint32_t packR11G11B10(const uint8_t *src) {
  float value[4];
  std::memcpy(value, src, sizeof(value));
  return packUnsignedFloat(value[0], 6) |
         (packUnsignedFloat(value[1], 6) << 11) |
         (packUnsignedFloat(value[2], 5) << 22);
}

bool stageCsImage(const ComputeInfo::Res &res, void *dst) {
  gcn::TextureLayout32 tiled, linear;
  if (!buildCsImageLayouts(res, tiled, linear)) return false;
  const bool direct = res.elemBytes == res.stageElemBytes;
  bool fillsCompleteLayout = direct;
  uint64_t filledBytes = 0;
  for (uint32_t mip = 0; fillsCompleteLayout && mip < linear.mip_levels; ++mip) {
    const auto &level = linear.mips[mip];
    const uint64_t logicalBytes = static_cast<uint64_t>(level.width) *
                                  level.height * linear.layers *
                                  res.stageElemBytes;
    fillsCompleteLayout = level.offset == filledBytes &&
                          level.pitch == level.width &&
                          level.stored_height == level.height &&
                          level.size == logicalBytes;
    filledBytes = level.offset + level.size;
  }
  fillsCompleteLayout = fillsCompleteLayout && filledBytes == res.size;
  if (!fillsCompleteLayout) std::memset(dst, 0, res.size);
  std::vector<uint8_t> tight;
  if (!direct)
    tight.resize(static_cast<size_t>(res.width) * res.height * res.elemBytes);
  // DELTA_GPU_DETILEDUMP=<base>: write the de-tiled level-0 bytes of that guest
  // surface to <dumpdir>/detiled.bin once, so the swizzle can be checked against
  // an offline decode of the same texture.
  static const uint64_t detileDump = [] {
    const char *e = std::getenv("DELTA_GPU_DETILEDUMP");
    return e ? std::strtoull(e, nullptr, 0) : 0ull;
  }();
  static bool detileDumped = false;
  for (uint32_t mip = 0; mip < tiled.mip_levels; mip++) {
    const auto &srcLevel = tiled.mips[mip];
    const auto &dstLevel = linear.mips[mip];
    for (uint32_t layer = 0; layer < tiled.layers; layer++) {
      uint8_t *levelDst = static_cast<uint8_t *>(dst) + dstLevel.offset +
          static_cast<uint64_t>(layer) * dstLevel.pitch *
              dstLevel.stored_height * res.stageElemBytes;
      if (direct) {
        if (!gcn::DetileTextureMip32Pitched(
                reinterpret_cast<const void *>(res.base), levelDst,
                static_cast<size_t>(dstLevel.pitch) * res.stageElemBytes, tiled,
                mip, layer))
          return false;
        if (detileDump && res.base == detileDump && !mip && !layer &&
            !detileDumped) {
          detileDumped = true;
          char p[256];
          std::snprintf(p, sizeof p, "%s/detiled.bin", dumpDir());
          if (FILE *f = std::fopen(p, "wb")) {
            std::fwrite(levelDst, 1,
                        static_cast<size_t>(dstLevel.pitch) *
                            dstLevel.stored_height * res.stageElemBytes, f);
            std::fclose(f);
            std::fprintf(stderr, "[detiledump] %#lx pitch=%u h=%u elem=%u\n",
                         (unsigned long)res.base, dstLevel.pitch,
                         dstLevel.stored_height, res.stageElemBytes);
          }
        }
        continue;
      }
      if (!gcn::DetileTextureMip32(reinterpret_cast<const void *>(res.base),
                                   tight.data(), tiled, mip, layer))
        return false;
      gcn::DetileParallelRows(srcLevel.height, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; y++) {
          uint8_t *dstRow = levelDst + static_cast<size_t>(y) *
                                           dstLevel.pitch * res.stageElemBytes;
          const uint8_t *srcRow = tight.data() + static_cast<size_t>(y) *
                                                     srcLevel.width *
                                                     res.elemBytes;
          if (res.dfmt == 6) {
            for (uint32_t x = 0; x < srcLevel.width; x++) {
              uint32_t packed;
              std::memcpy(&packed, srcRow + static_cast<size_t>(x) * 4, 4);
              unpackR11G11B10(packed,
                              dstRow + static_cast<size_t>(x) * 16);
            }
          } else {
            for (uint32_t x = 0; x < srcLevel.width; x++) {
              uint16_t value;
              std::memcpy(&value, srcRow + static_cast<size_t>(x) * 2, 2);
              const uint32_t expanded = value;
              std::memcpy(dstRow + static_cast<size_t>(x) * 4, &expanded, 4);
            }
          }
        }
      });
    }
  }
  return true;
}

bool writebackCsImage(const ComputeInfo::Res &res, const void *src) {
  gcn::TextureLayout32 tiled, linear;
  if (!buildCsImageLayouts(res, tiled, linear)) return false;
  const bool direct = res.elemBytes == res.stageElemBytes;
  std::vector<uint8_t> tight;
  if (!direct)
    tight.resize(static_cast<size_t>(res.width) * res.height * res.elemBytes);
  for (uint32_t mip = 0; mip < tiled.mip_levels; mip++) {
    const auto &dstLevel = tiled.mips[mip];
    const auto &srcLevel = linear.mips[mip];
    for (uint32_t layer = 0; layer < tiled.layers; layer++) {
      const uint8_t *levelSrc = static_cast<const uint8_t *>(src) +
          srcLevel.offset + static_cast<uint64_t>(layer) * srcLevel.pitch *
                                srcLevel.stored_height * res.stageElemBytes;
      if (direct) {
        if (!gcn::RetileTextureMip32Pitched(
                levelSrc,
                static_cast<size_t>(srcLevel.pitch) * res.stageElemBytes,
                reinterpret_cast<void *>(res.base), tiled, mip, layer))
          return false;
        continue;
      }
      gcn::DetileParallelRows(dstLevel.height, [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; y++) {
          uint8_t *dstRow = tight.data() + static_cast<size_t>(y) *
                                               dstLevel.width * res.elemBytes;
          const uint8_t *srcRow = levelSrc + static_cast<size_t>(y) *
                                                 srcLevel.pitch *
                                                 res.stageElemBytes;
          if (res.dfmt == 6) {
            for (uint32_t x = 0; x < dstLevel.width; x++) {
              const uint32_t packed =
                  packR11G11B10(srcRow + static_cast<size_t>(x) * 16);
              std::memcpy(dstRow + static_cast<size_t>(x) * 4, &packed, 4);
            }
          } else {
            for (uint32_t x = 0; x < dstLevel.width; x++) {
              uint32_t expanded;
              std::memcpy(&expanded, srcRow + static_cast<size_t>(x) * 4, 4);
              const uint16_t value = static_cast<uint16_t>(expanded);
              std::memcpy(dstRow + static_cast<size_t>(x) * 2, &value, 2);
            }
          }
        }
      });
      if (!gcn::RetileTextureMip32(tight.data(),
                                   reinterpret_cast<void *>(res.base), tiled,
                                   mip, layer))
        return false;
    }
  }
  return true;
}

// GPU-resident compute working set. Each guest range a CS touches gets a
// persistent host-visible storage buffer keyed by its base address. Staged
// content persists across dispatches and frames: a range the GPU wrote
// (gpuDirty) is the newest copy and is bound directly with no re-staging;

// guest-sourced ranges revalidate against a content hash at most once per
// frame. Writebacks to guest memory (the expensive image retile) happen
// LAZILY — only when a draw / DMA / frame boundary needs guest memory to be
// current (flushCsWrites), not after every dispatch.
struct CsRange {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void *map = nullptr;
  VkDeviceSize cap = 0;
  uint64_t size = 0;        // active staged (linear) byte size
  uint64_t guestBytes = 0;  // guest footprint (hash + overlap checks)
  uint64_t hash = 0;        // texHash of guest content when last in sync
  int lastValidatedFrame = -1;
  int lastUsedFrame = -1;
  bool gpuDirty = false;    // buffer newer than guest memory
  bool pendingBatch = false;  // referenced by the open dispatch batch
  bool imageStaging = false;
  ComputeInfo::Res res;     // writeback needs the full layout description
};

std::unordered_map<uint64_t, CsRange> g_csRanges;
uint64_t g_csRangeBytes = 0;

// Sampled content hash for CS range validation: length + 256 evenly spaced
// 64-byte windows. Reading a whole 16MB image per validation was the point of
// the exercise; a CPU write that dodges every window for a whole frame is a
// risk we accept for the ~50x cheaper check (full texHash still guards the
// sampled-texture cache).
uint64_t rangeHash(uint64_t base, uint64_t bytes) {
  if (bytes <= 16384) return texHash(base, bytes);
  constexpr uint64_t kPrime = 1099511628211ull;
  uint64_t h = 1469598103934665603ull ^ (bytes * kPrime);
  const uint64_t step = (bytes - 64) / 255;
  for (uint32_t i = 0; i < 256; i++) {
    uint64_t w[8];
    std::memcpy(w, reinterpret_cast<const void *>(base + i * step), 64);
    for (int j = 0; j < 8; j++) h = (h ^ w[j]) * kPrime;
  }
  return h;
}

bool csRangeEnsureBuffer(CsRange &e, VkDeviceSize size) {
  if (e.buf && e.cap >= size) return true;
  if (e.map) { vkUnmapMemory(g_dev.device, e.mem); e.map = nullptr; }
  if (e.buf) { vkDestroyBuffer(g_dev.device, e.buf, nullptr); e.buf = VK_NULL_HANDLE; }
  if (e.mem) { vkFreeMemory(g_dev.device, e.mem, nullptr); e.mem = VK_NULL_HANDLE; }
  g_csRangeBytes -= e.cap;
  e.cap = 0;
  VkDeviceSize cap = (size + 0xFFFF) & ~VkDeviceSize(0xFFFF);
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = cap; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &e.buf) != VK_SUCCESS) {
    e.buf = VK_NULL_HANDLE;
    return false;
  }
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev.device, e.buf, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMemoryTypePref(mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &e.mem) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, e.buf, nullptr);
    e.buf = VK_NULL_HANDLE; e.mem = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(g_dev.device, e.buf, e.mem, 0);
  vkMapMemory(g_dev.device, e.mem, 0, cap, 0, &e.map);
  e.cap = cap;
  g_csRangeBytes += cap;
  return true;
}

void csRangeDestroy(CsRange &e) {
  if (e.map) vkUnmapMemory(g_dev.device, e.mem);
  if (e.buf) vkDestroyBuffer(g_dev.device, e.buf, nullptr);
  if (e.mem) vkFreeMemory(g_dev.device, e.mem, nullptr);
  g_csRangeBytes -= e.cap;
  e = CsRange{};
}

// Dispatch batching: dispatches are recorded into one command buffer and
// submitted/waited only when something needs their results (a flush point,
// a staging hazard, or the batch cap). 228 individual submit+fence round
// trips per frame were ~40% of the whole compute cost.
bool g_csBatchOpen = false;
uint32_t g_csBatchCount = 0;
VkFence g_csBatchFence = VK_NULL_HANDLE;
bool g_csStagePending[ComputeInfo::kMaxResources] = {};

void csBatchFlush() {
  if (!g_csBatchOpen) return;
  const uint64_t t0 = nowNs();
  vkEndCommandBuffer(g_csCmd);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &g_csCmd;
  vkResetFences(g_dev.device, 1, &g_csBatchFence);
  const VkResult sr = vkQueueSubmit(g_dev.queue, 1, &si, g_csBatchFence);
  const VkResult wr =
      sr == VK_SUCCESS
          ? vkWaitForFences(g_dev.device, 1, &g_csBatchFence, VK_TRUE, UINT64_MAX)
          : sr;
  if (sr != VK_SUCCESS || wr != VK_SUCCESS) {
    std::fprintf(stderr,
                 "[gpuvk] cs batch DEVICE FAULT: submit=%d wait=%d n=%u\n",
                 (int)sr, (int)wr, g_csBatchCount);
    reportDeviceFault(g_dev.device);
  }
  vkResetDescriptorPool(g_dev.device, g_csDescPool, 0);
  g_csBatchOpen = false;
  g_csBatchCount = 0;
  for (auto &kv : g_csRanges) kv.second.pendingBatch = false;
  std::memset(g_csStagePending, 0, sizeof g_csStagePending);
  g_nsCsGpu += nowNs() - t0;
}

// Write one dirty range back to guest memory (retile for images) and re-stamp
// its hash so the next validation sees guest == buffer.
bool csRangeFlushOne(uint64_t base, CsRange &e) {
  if (!e.gpuDirty) return true;
  if (e.pendingBatch) csBatchFlush();  // results must exist before readback
  g_csFlushN++;
  if (e.imageStaging) {
    if (!writebackCsImage(e.res, e.map)) return false;
  } else {
    std::memcpy(reinterpret_cast<void *>(base), e.map, e.size);
  }
  invalidateTexRange(base, e.guestBytes);
  e.gpuDirty = false;
  e.hash = rangeHash(base, e.guestBytes);
  e.lastValidatedFrame = g_frame.num;
  return true;
}

// Ensure staging slot i can hold `size` bytes (grow-on-demand, kept mapped).
bool csEnsureStage(uint32_t i, VkDeviceSize size) {
  CsStage &s = g_csStage[i];
  if (s.buf && s.cap >= size) return true;
  if (s.map) { vkUnmapMemory(g_dev.device, s.mem); s.map = nullptr; }
  if (s.buf) { vkDestroyBuffer(g_dev.device, s.buf, nullptr); s.buf = VK_NULL_HANDLE; }
  if (s.mem) { vkFreeMemory(g_dev.device, s.mem, nullptr); s.mem = VK_NULL_HANDLE; }
  VkDeviceSize cap = (size + 0xFFFFF) & ~VkDeviceSize(0xFFFFF);  // 1 MiB granularity
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = cap; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &s.buf) != VK_SUCCESS) { s.buf = VK_NULL_HANDLE; return false; }
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev.device, s.buf, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMemoryTypePref(mr.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &s.mem) != VK_SUCCESS) {
    vkDestroyBuffer(g_dev.device, s.buf, nullptr); s.buf = VK_NULL_HANDLE; s.mem = VK_NULL_HANDLE;
    return false;
  }
  vkBindBufferMemory(g_dev.device, s.buf, s.mem, 0);
  vkMapMemory(g_dev.device, s.mem, 0, cap, 0, &s.map);
  s.cap = cap;
  return true;
}

struct ScopeCs {
  uint64_t t0 = nowNs();
  ~ScopeCs() { g_nsCs += nowNs() - t0; g_csCount++; }
};

}  // namespace
}  // namespace gpu::vk

namespace gpu::rhi {
using namespace gpu::vk;

bool dispatch(const ComputeInfo &ci) {
  if (!g_dev.ready || !ci.recomp || !ci.recomp->ok || !ci.nres ||
      ci.nres > g_dev.maxCsResources)
    return false;
  ScopeCs _cs;
  for (uint32_t i = 0; i < ci.nres; i++) g_csBytes += ci.res[i].size;
  for (uint32_t i = 0; i < ci.nres; i++)
    if (ci.res[i].size > g_dev.maxStorageBufferRange) return false;
  CsPipe *cp = getCsPipe(ci);
  if (!cp) return false;
  static const bool verbose = std::getenv("DELTA_GPU_CSGPU_VERBOSE") != nullptr;

  // Persistent command buffer + descriptor pool (created once, reused).
  if (g_csCmd == VK_NULL_HANDLE) {
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = g_dev.pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_dev.device, &ca, &g_csCmd) != VK_SUCCESS) { g_csCmd = VK_NULL_HANDLE; return false; }
  }
  if (g_csDescPool == VK_NULL_HANDLE) {
    // Sized for a whole batch of dispatches between flushes.
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            256 * ComputeInfo::kMaxResources};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 256; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(g_dev.device, &pci, nullptr, &g_csDescPool) != VK_SUCCESS) { g_csDescPool = VK_NULL_HANDLE; return false; }
  }
  if (g_csBatchFence == VK_NULL_HANDLE) {
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(g_dev.device, &fci, nullptr, &g_csBatchFence) != VK_SUCCESS) {
      g_csBatchFence = VK_NULL_HANDLE;
      return false;
    }
  }

  // DELTA_GPU_CSLIST: per-dispatch resource staging list for the first 200
  // dispatches — shows what the chain actually round-trips per frame.
  static const bool csList = std::getenv("DELTA_GPU_CSLIST") != nullptr;
  static uint32_t csListed = 0;
  if (csList && g_frame.num > 25 && csListed < 200) {
    csListed++;
    for (uint32_t i = 0; i < ci.nres; i++)
      std::fprintf(stderr,
                   "[cslist] cs=%#llx bind=%u base=%#lx size=%#lx %s%s\n",
                   (unsigned long long)ci.csAddr, ci.res[i].binding,
                   (unsigned long)ci.res[i].base,
                   (unsigned long)ci.res[i].size,
                   ci.res[i].imageStaging ? "img"
                   : ci.res[i].zeroFill   ? "zero"
                                          : "buf",
                   ci.res[i].written ? " written" : "");
  }

  // Bind each resource: zero-fill scratch per binding slot; everything else
  // uses the persistent range buffer for its guest base, staged only when the
  // buffer doesn't already hold current content.
  const uint64_t _tIn0 = nowNs();
  VkBuffer bindBuf[ComputeInfo::kMaxResources];
  VkDeviceSize sz[ComputeInfo::kMaxResources];
  for (uint32_t i = 0; i < ci.nres; i++) {
    sz[i] = ci.res[i].size ? ((ci.res[i].size + 3) & ~VkDeviceSize(3)) : 4;
    if (ci.res[i].zeroFill) {
      // Growing the scratch slot recreates its buffer; a pending batched
      // dispatch still references the old handle.
      if (g_csStagePending[i] && g_csStage[i].cap < sz[i]) csBatchFlush();
      if (!csEnsureStage(i, sz[i])) return false;
      std::memset(g_csStage[i].map, 0, sz[i]);
      bindBuf[i] = g_csStage[i].buf;
      continue;
    }
    const uint64_t base = ci.res[i].base;
    const uint64_t guestBytes =
        ci.res[i].guestSize ? ci.res[i].guestSize : ci.res[i].size;
    // A read overlapping some OTHER dirty range must see that data through
    // guest memory: flush those first.
    for (auto &kv : g_csRanges) {
      if (kv.first == base) continue;
      CsRange &o = kv.second;
      if (o.gpuDirty && kv.first < base + guestBytes &&
          base < kv.first + o.guestBytes)
        if (!csRangeFlushOne(kv.first, o)) return false;
    }
    CsRange &e = g_csRanges[base];
    const bool sameShape = e.buf && e.size == static_cast<uint64_t>(sz[i]) &&
                           e.imageStaging == ci.res[i].imageStaging;
    if (!sameShape && e.gpuDirty)
      if (!csRangeFlushOne(base, e)) return false;  // reshaped: keep its data
    if (e.pendingBatch && (!e.buf || e.cap < sz[i]))
      csBatchFlush();  // growth would destroy a buffer the batch references
    if (!csRangeEnsureBuffer(e, sz[i])) return false;
    bool valid = sameShape && (e.gpuDirty || e.lastValidatedFrame == g_frame.num);
    if (!valid && sameShape) {
      const uint64_t h = rangeHash(base, guestBytes);
      if (h == e.hash) valid = true; else e.hash = h;
      e.lastValidatedFrame = g_frame.num;
    }
    if (!valid) {
      // CPU write into a buffer a pending batched dispatch reads/writes.
      if (e.pendingBatch) csBatchFlush();
      if (ci.res[i].imageStaging) {
        if (!stageCsImage(ci.res[i], e.map)) return false;
      } else {
        std::memcpy(e.map, reinterpret_cast<const void *>(base),
                    ci.res[i].size);
        if (sz[i] > ci.res[i].size)
          std::memset(static_cast<uint8_t *>(e.map) + ci.res[i].size, 0,
                      sz[i] - ci.res[i].size);
      }
      if (!sameShape) {
        e.hash = rangeHash(base, guestBytes);
        e.lastValidatedFrame = g_frame.num;
      }
      e.gpuDirty = false;
      g_csStageN++;
      g_csStageBytes += sz[i];
    }
    e.size = sz[i];
    e.guestBytes = guestBytes;
    e.imageStaging = ci.res[i].imageStaging;
    e.res = ci.res[i];
    e.lastUsedFrame = g_frame.num;
    bindBuf[i] = e.buf;
  }
  // Re-resolve handles: a later binding sharing an earlier binding's base may
  // have grown (destroyed + recreated) that range's buffer.
  for (uint32_t i = 0; i < ci.nres; i++)
    if (!ci.res[i].zeroFill) bindBuf[i] = g_csRanges[ci.res[i].base].buf;

  g_nsCsIn += nowNs() - _tIn0;

  // Descriptor set binding the storage buffers (pool lives for a whole batch;
  // reset happens at batch flush).
  VkDescriptorSet set;
  VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  da.descriptorPool = g_csDescPool; da.descriptorSetCount = 1; da.pSetLayouts = &cp->setLayout;
  if (vkAllocateDescriptorSets(g_dev.device, &da, &set) != VK_SUCCESS) {
    csBatchFlush();  // pool exhausted: flush resets it, then retry once
    if (vkAllocateDescriptorSets(g_dev.device, &da, &set) != VK_SUCCESS)
      return false;
  }
  VkDescriptorBufferInfo dbi[ComputeInfo::kMaxResources];
  VkWriteDescriptorSet wr[ComputeInfo::kMaxResources];
  for (uint32_t i = 0; i < ci.nres; i++) {
    dbi[i] = {bindBuf[i], 0, sz[i]};
    wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr[i].dstSet = set; wr[i].dstBinding = ci.res[i].binding; wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i];
  }
  vkUpdateDescriptorSets(g_dev.device, ci.nres, wr, 0, nullptr);

  // Record the dispatch into the open batch. Submission + the fence wait
  // happen at the next flush point, not here.
  if (!g_csBatchOpen) {
    vkResetCommandBuffer(g_csCmd, 0);
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(g_csCmd, &cbi);
    g_csBatchOpen = true;
  }
  vkCmdBindPipeline(g_csCmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp->pipe);
  vkCmdBindDescriptorSets(g_csCmd, VK_PIPELINE_BIND_POINT_COMPUTE, cp->layout, 0, 1, &set, 0, nullptr);
  vkCmdPushConstants(g_csCmd, cp->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 64, ci.userData);
  vkCmdDispatch(g_csCmd, ci.groups[0], ci.groups[1], ci.groups[2]);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(g_csCmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                       nullptr, 0, nullptr);
  for (uint32_t i = 0; i < ci.nres; i++) {
    if (ci.res[i].zeroFill) {
      g_csStagePending[i] = true;
    } else {
      auto it = g_csRanges.find(ci.res[i].base);
      if (it != g_csRanges.end()) it->second.pendingBatch = true;
    }
  }
  if (++g_csBatchCount >= 128 || verbose) csBatchFlush();

  // Mark written ranges GPU-dirty. Guest memory catches up lazily at the next
  // flush point (draw / DMA / frame end) — writing every dispatch's outputs
  // back immediately (the image retile especially) was ~100ms/frame.
  const uint64_t _tOut0 = nowNs();
  for (uint32_t i = 0; i < ci.nres; i++) {
    if (!ci.res[i].written || ci.res[i].zeroFill) continue;
    auto it = g_csRanges.find(ci.res[i].base);
    if (it == g_csRanges.end()) continue;
    it->second.gpuDirty = true;
    if (verbose) {
      const uint8_t *b = static_cast<const uint8_t *>(it->second.map);
      uint64_t nz = 0, step = ci.res[i].size > 65536 ? ci.res[i].size / 65536 : 1;
      for (uint64_t k = 0; k < ci.res[i].size; k += step) nz += b[k] != 0;
      std::fprintf(stderr, "[csgpu] gpu wrote base=%#lx size=%lu nonzero=%lu/%lu\n",
                   (unsigned long)ci.res[i].base, (unsigned long)ci.res[i].size,
                   (unsigned long)nz, (unsigned long)(ci.res[i].size / step));
    }
  }
  g_nsCsOut += nowNs() - _tOut0;
  return true;
}

// Make guest memory current with every GPU-written compute range. Called
// before anything that consumes guest memory: draws (vertex/texture reads at
// record time), CP DMA copies, and the end of each frame (bounds staleness
// for direct guest CPU readers to one frame). Cheap no-op when nothing is
// dirty; also evicts cold entries so the working set stays bounded.
void flushCsWrites() {
  const uint64_t _t0 = nowNs();
  for (auto it = g_csRanges.begin(); it != g_csRanges.end();) {
    csRangeFlushOne(it->first, it->second);
    if (g_csRangeBytes > (1ull << 30) && !it->second.gpuDirty &&
        !it->second.pendingBatch &&
        it->second.lastUsedFrame + 300 < g_frame.num) {
      csRangeDestroy(it->second);
      it = g_csRanges.erase(it);
    } else {
      ++it;
    }
  }
  g_nsCsOut += nowNs() - _t0;
}

// Targeted variant: flush only dirty ranges overlapping [base, base+bytes).
// The per-draw guest readers (texture upload, vertex copy, cbuffer ring) call
// this instead of the full flush — flushing every dirty range at every draw
// re-tiled the whole post chain ~19x/frame.
void flushCsWritesRange(uint64_t base, uint64_t bytes) {
  if (!base || !bytes || g_csRanges.empty()) return;
  const uint64_t _t0 = nowNs();
  for (auto &kv : g_csRanges) {
    CsRange &e = kv.second;
    if (e.gpuDirty && kv.first < base + bytes && base < kv.first + e.guestBytes)
      csRangeFlushOne(kv.first, e);
  }
  g_nsCsOut += nowNs() - _t0;
}

}  // namespace gpu::rhi
