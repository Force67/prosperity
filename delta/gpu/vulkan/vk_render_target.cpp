/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "vulkan/vk_render_target.h"

#include "rhi/renderer.h"
#include "vulkan/vk_device.h"
#include "vulkan/vk_format.h"
#include "vulkan/vk_frame.h"
#include "vulkan/vk_texture_cache.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gpu::vk {

using rhi::DrawInfo;

std::unordered_map<uint64_t, RTarget> g_rts;
std::unordered_map<uint64_t, DepthTarget> g_depths;
std::unordered_map<uint64_t, std::vector<uint64_t>> g_rtPages;
RenderRegion g_region;

VkImageView sampledImageView(VkImage image, VkImageView identity,
                             VkFormat format, VkImageAspectFlags aspect,
                             uint32_t swizzle,
                             std::unordered_map<uint32_t, VkImageView> &views) {
  const VkComponentMapping components = textureComponents(swizzle);
  if (components.r == VK_COMPONENT_SWIZZLE_IDENTITY &&
      components.g == VK_COMPONENT_SWIZZLE_IDENTITY &&
      components.b == VK_COMPONENT_SWIZZLE_IDENTITY &&
      components.a == VK_COMPONENT_SWIZZLE_IDENTITY)
    return identity;
  const auto it = views.find(swizzle);
  if (it != views.end())
    return it->second;
  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = image;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = format;
  vi.components = components;
  vi.subresourceRange = {aspect, 0, 1, 0, 1};
  VkImageView view = VK_NULL_HANDLE;
  if (vkCreateImageView(g_dev.device, &vi, nullptr, &view) != VK_SUCCESS)
    return VK_NULL_HANDLE;
  views.emplace(swizzle, view);
  return view;
}

VkImageView sampledView(RTarget &rt, uint32_t swizzle, bool feedback) {
  return feedback ? sampledImageView(rt.feedbackImage, rt.feedbackView, rt.fmt,
                                     VK_IMAGE_ASPECT_COLOR_BIT, swizzle,
                                     rt.feedbackSampledViews)
                  : sampledImageView(rt.image, rt.view, rt.fmt,
                                     VK_IMAGE_ASPECT_COLOR_BIT, swizzle,
                                     rt.sampledViews);
}

VkImageView sampledView(DepthTarget &depth, uint32_t swizzle) {
  return sampledImageView(depth.image, depth.view, kDepthFormat,
                          VK_IMAGE_ASPECT_DEPTH_BIT, swizzle,
                          depth.sampledViews);
}

uint64_t rtByteSizeWH(uint32_t w, uint32_t h, VkFormat fmt) {
  return (uint64_t)w * h * formatBytes(fmt);
}

// Register an RT's footprint pages so the page table can find it by overlap.
void registerRtPages(uint64_t base, uint32_t w, uint32_t h, VkFormat fmt) {
  uint64_t lo = base >> kRtPageShift;
  uint64_t hi = (base + rtByteSizeWH(w, h, fmt) - 1) >> kRtPageShift;
  for (uint64_t p = lo; p <= hi; p++) {
    auto &v = g_rtPages[p];
    bool seen = false;
    for (uint64_t b : v) if (b == base) { seen = true; break; }
    if (!seen) v.push_back(base);
  }
}

// Find or create the render target at guest address `base` (dimensions w x h).
RTarget *getRT(uint64_t base, uint32_t w, uint32_t h, VkFormat fmt) {
  auto it = g_rts.find(base);
  if (it != g_rts.end()) {
    static std::unordered_map<uint64_t, bool> aliasWarned;
    if ((it->second.w != w || it->second.h != h || it->second.fmt != fmt) &&
        !aliasWarned[base]) {
      aliasWarned[base] = true;
      std::fprintf(stderr,
                   "[gpuvk] RT alias mismatch %#lx: have %ux%u fmt=%d, requested %ux%u fmt=%d\n",
                   (unsigned long)base, it->second.w, it->second.h, (int)it->second.fmt,
                   w, h, (int)fmt);
    }
    return &it->second;
  }
  if (g_rts.size() > 64 || !w || !h)
    return nullptr;
  // Robustness: reject render targets with implausible dimensions or an undefined
  // format. A garbage CB_COLOR base/scissor (e.g. a stray shader-pool RT address on
  // the PS5 path) would otherwise feed the driver an invalid image and hard-crash it.
  if (w > 8192 || h > 8192 || fmt == VK_FORMAT_UNDEFINED) {
    std::fprintf(stderr, "[gpuvk] skip bad RT %#lx %ux%u fmt=%d\n",
                 (unsigned long)base, w, h, (int)fmt);
    return nullptr;
  }
  RTarget t; t.w = w; t.h = h; t.fmt = fmt;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = fmt;
  ii.extent = {w, h, 1};
  ii.mipLevels = 1; ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
             VK_IMAGE_USAGE_STORAGE_BIT |
             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (vkCreateImage(g_dev.device, &ii, nullptr, &t.image) != VK_SUCCESS) return nullptr;
  VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev.device, t.image, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &t.mem) != VK_SUCCESS) {
    vkDestroyImage(g_dev.device, t.image, nullptr);
    return nullptr;  // GPU OOM -> don't bind/view a memory-less image (driver crash)
  }
  vkBindImageMemory(g_dev.device, t.image, t.mem, 0);
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = t.image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = fmt;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCreateImageView(g_dev.device, &vci, nullptr, &t.view);
  // descriptor set so this RT can be sampled (render-to-texture).
  if (g_tex.dsPool) {
    VkDescriptorPool owner;
    t.set = allocateSamplerSet(g_tex.dsLayout, false, owner);
    if (t.set) {
      VkDescriptorImageInfo dii{g_tex.sampler, t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      wr.dstSet = t.set; wr.descriptorCount = 1;
      wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      wr.pImageInfo = &dii;
      vkUpdateDescriptorSets(g_dev.device, 1, &wr, 0, nullptr);
    }
  }
  std::fprintf(stderr, "[gpuvk] new RT %#lx %ux%u fmt=%d\n",
               (unsigned long)base, w, h, (int)fmt);
  g_rts[base] = t;
  if (!g_region.firstRt) g_region.firstRt = base;
  registerRtPages(base, w, h, fmt);  // resource-model page table
  return &g_rts[base];
}

VkDescriptorSet snapshotRT(RTarget &rt) {
  if (!rt.feedbackImage) {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = rt.fmt;
    ii.extent = {rt.w, rt.h, 1};
    ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (vkCreateImage(g_dev.device, &ii, nullptr, &rt.feedbackImage) != VK_SUCCESS)
      return VK_NULL_HANDLE;
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(g_dev.device, rt.feedbackImage, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(g_dev.device, &ai, nullptr, &rt.feedbackMem) != VK_SUCCESS) {
      vkDestroyImage(g_dev.device, rt.feedbackImage, nullptr);
      rt.feedbackImage = VK_NULL_HANDLE;
      return VK_NULL_HANDLE;
    }
    if (vkBindImageMemory(g_dev.device, rt.feedbackImage, rt.feedbackMem, 0) != VK_SUCCESS) {
      vkFreeMemory(g_dev.device, rt.feedbackMem, nullptr);
      vkDestroyImage(g_dev.device, rt.feedbackImage, nullptr);
      rt.feedbackMem = VK_NULL_HANDLE;
      rt.feedbackImage = VK_NULL_HANDLE;
      return VK_NULL_HANDLE;
    }
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = rt.feedbackImage;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = rt.fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(g_dev.device, &vi, nullptr, &rt.feedbackView) != VK_SUCCESS)
      return VK_NULL_HANDLE;
    VkDescriptorPool owner;
    rt.feedbackSet = allocateSamplerSet(g_tex.dsLayout, false, owner);
    if (!rt.feedbackSet) return VK_NULL_HANDLE;
    VkDescriptorImageInfo dii{g_tex.sampler, rt.feedbackView,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr.dstSet = rt.feedbackSet;
    wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr.pImageInfo = &dii;
    vkUpdateDescriptorSets(g_dev.device, 1, &wr, 0, nullptr);
  }

  VkAccessFlags srcAccess = rt.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                            : rt.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                ? VK_ACCESS_SHADER_READ_BIT
                            : rt.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                ? VK_ACCESS_TRANSFER_READ_BIT
                                : 0;
  imageBarrier(g_frame.cmd, rt.image, rt.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               srcAccess, VK_ACCESS_TRANSFER_READ_BIT);
  rt.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  VkAccessFlags feedbackAccess = rt.feedbackLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                     ? VK_ACCESS_SHADER_READ_BIT
                                     : 0;
  imageBarrier(g_frame.cmd, rt.feedbackImage, rt.feedbackLayout,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, feedbackAccess,
               VK_ACCESS_TRANSFER_WRITE_BIT);
  rt.feedbackLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  VkImageCopy copy{};
  copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.extent = {rt.w, rt.h, 1};
  vkCmdCopyImage(g_frame.cmd, rt.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 rt.feedbackImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  imageBarrier(g_frame.cmd, rt.feedbackImage, rt.feedbackLayout,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
  rt.feedbackLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  return rt.feedbackSet;
}

uint64_t rtByteSize(const RTarget &rt) { return rtByteSizeWH(rt.w, rt.h, rt.fmt); }
// Find or create the depth target at guest address `base` (dimensions w x h).
DepthTarget *getDepthRT(uint64_t base, uint32_t w, uint32_t h) {
  auto it = g_depths.find(base);
  if (it != g_depths.end()) return &it->second;
  if (g_depths.size() > 32 || !w || !h) return nullptr;
  DepthTarget t; t.w = w; t.h = h;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = kDepthFormat;
  ii.extent = {w, h, 1};
  ii.mipLevels = 1; ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  if (vkCreateImage(g_dev.device, &ii, nullptr, &t.image) != VK_SUCCESS) return nullptr;
  VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev.device, t.image, &mr);
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkAllocateMemory(g_dev.device, &ai, nullptr, &t.mem);
  vkBindImageMemory(g_dev.device, t.image, t.mem, 0);
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = t.image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = kDepthFormat;
  vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
  vkCreateImageView(g_dev.device, &vci, nullptr, &t.view);
  if (g_tex.dsPool) {
    VkDescriptorPool owner;
    t.set = allocateSamplerSet(g_tex.dsLayout, false, owner);
    if (t.set) {
      VkDescriptorImageInfo dii{g_tex.sampler, t.view,
                                VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      wr.dstSet = t.set; wr.descriptorCount = 1;
      wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      wr.pImageInfo = &dii;
      vkUpdateDescriptorSets(g_dev.device, 1, &wr, 0, nullptr);
    }
  }
  std::fprintf(stderr, "[gpuvk] new depth %#lx %ux%u\n", (unsigned long)base, w, h);
  g_depths[base] = t;
  return &g_depths[base];
}

// Resolve a sampled texture address to the live RT image that backs it (the
// resource model's "the page table collects all overlappers" lookup). The game
// cycles/aliases RT addresses (double-buffered room layers, a pool of scene
// buffers), so a composite often samples an address that does not exactly match
// the RT base it was rendered into. We gather every RT whose footprint touches
// the sampled region's pages and resolve by IDENTITY, not recency: an exact base
// whose size matches wins outright (the guest bound that buffer, so a more recently
// rendered overlapping buffer must never override it); only when no exact match
// exists do we fall back to the best-fitting overlapper (dimension match, then the
// freshest). Returns the g_rts key, or 0.
uint64_t resolveSampledRT(uint64_t addr, uint32_t w, uint32_t h) {
  if (!addr) return 0;
  uint64_t reqSize = w && h ? (uint64_t)w * h * 4 : 4;
  uint64_t a0 = addr, a1 = addr + reqSize;
  // Exact-identity hit: the guest sampled this exact base and it is a live RT of the
  // requested size. That is unambiguously the right image -- return it before any
  // freshness comparison can pick an overlapping cycled buffer instead.
  auto ex = g_rts.find(addr);
  if (ex != g_rts.end() && ex->second.everRendered &&
      ((!w || !h) || (ex->second.w == w && ex->second.h == h)))
    return addr;
  uint64_t best = 0;
  long bestScore = -1;
  auto consider = [&](uint64_t b0) {
    auto it = g_rts.find(b0);
    if (it == g_rts.end()) return;
    const RTarget &rt = it->second;
    if (!rt.everRendered) return;  // never sample an RT with no content
    uint64_t b1 = b0 + rtByteSize(rt);
    if (!(a0 < b1 && b0 < a1)) return;  // no interval overlap
    bool dimMatch = (!w || !h) || (rt.w == w && rt.h == h);
    long score = (long)rt.lastFrame;
    if (dimMatch) score += 1L << 30;
    if (b0 == addr) score += 1L << 31;
    if (score > bestScore) { bestScore = score; best = b0; }
  };
  for (uint64_t p = a0 >> kRtPageShift; p <= (a1 - 1) >> kRtPageShift; p++) {
    auto it = g_rtPages.find(p);
    if (it != g_rtPages.end())
      for (uint64_t b0 : it->second) consider(b0);
  }
  return best;
}

// Depth images use a separate registry from color RTs. A GFX7 depth surface may be
// sampled through an R32_FLOAT descriptor whose base denotes an overlapping view
// rather than DB_Z_WRITE_BASE, so resolve typed depth aliases by footprint too.
uint64_t resolveSampledDepth(uint64_t addr, uint32_t w, uint32_t h) {
  if (!addr) return 0;
  uint64_t reqSize = w && h ? (uint64_t)w * h * 4 : 4;
  uint64_t a1 = addr + reqSize;
  uint64_t best = 0;
  long bestScore = -1;
  for (const auto &[base, depth] : g_depths) {
    if (depth.lastFrame <= -1000) continue;
    uint64_t b1 = base + rtByteSizeWH(depth.w, depth.h, kDepthFormat);
    if (!(addr < b1 && base < a1)) continue;
    bool dimMatch = (!w || !h) || (depth.w == w && depth.h == h);
    long score = depth.lastFrame;
    if (dimMatch) score += 1L << 30;
    if (base == addr) score += 1L << 31;
    if (score > bestScore) {
      bestScore = score;
      best = base;
    }
  }
  return best;
}

// End the current dynamic-rendering region (if any), leaving its RTs readable.
void endRegion() {
  if (!g_region.open) return;
  p_vkCmdEndRendering(g_frame.cmd);
  g_region.open = false;
  for (uint32_t i = 0; i < g_region.curMrtCount; i++) {
    auto it = g_rts.find(g_region.curMrt[i]);
    if (it == g_rts.end()) continue;
    auto &rt = it->second;
    imageBarrier(g_frame.cmd, rt.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    rt.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  }
  if (g_region.curDepth) {
    auto it = g_depths.find(g_region.curDepth);
    if (it != g_depths.end()) {
      auto &depth = it->second;
      depthBarrier(g_frame.cmd, depth.image, depth.layout,
                   VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT);
      depth.layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    }
  }
  g_region.curRt = 0;
  g_region.curMrtCount = 0;
  g_region.curDepth = 0;
}

void setGuestViewport(const DrawInfo &d) {
  if (!std::isfinite(d.viewportXScale) || !std::isfinite(d.viewportXOffset) ||
      !std::isfinite(d.viewportYScale) || !std::isfinite(d.viewportYOffset) ||
      d.viewportXScale <= 0.0f || d.viewportYScale == 0.0f)
    return;
  VkViewport vp{
      d.viewportXOffset - d.viewportXScale,
      d.viewportYOffset - d.viewportYScale,
      d.viewportXScale * 2.0f,
      d.viewportYScale * 2.0f,
      0.0f,
      1.0f,
  };
  vkCmdSetViewport(g_frame.cmd, 0, 1, &vp);
}

// Begin a dynamic-rendering region binding mrtCount color targets (mrtBase[0] is the
// primary). The common single-RT case (mrtCount == 1) binds exactly one attachment.
// depthBase != 0 additionally binds a depth attachment (cleared to depthClear on its
// first use each frame, loaded thereafter); depthBase == 0 leaves depth unbound (the
// 2D path).
void beginRegion(const uint64_t *mrtBase, const uint32_t *mrtInfo,
                 uint32_t mrtCount, uint32_t w, uint32_t h,
                 uint64_t depthBase, float depthClear) {
  static const bool lazyClear = [] { const char *e = std::getenv("DELTA_GPU_LAZYCLEAR");
    return !e || std::strcmp(e, "0") != 0; }();
  VkRenderingAttachmentInfo colors[8]{};
  g_region.curMrtCount = 0;
  for (uint32_t i = 0; i < mrtCount && i < 8; i++) {
    RTarget *rtp = getRT(mrtBase[i], w, h, colorTargetFormat(mrtInfo[i]));
    if (!rtp) continue;
    RTarget &rt = *rtp;
    VkAccessFlags srcAccess = rt.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                  ? VK_ACCESS_SHADER_READ_BIT
                              : rt.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                  ? VK_ACCESS_TRANSFER_READ_BIT
                               : rt.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                   ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                               : rt.layout == VK_IMAGE_LAYOUT_GENERAL
                                   ? VK_ACCESS_SHADER_WRITE_BIT
                                   : 0;
    imageBarrier(g_frame.cmd, rt.image, rt.layout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 srcAccess, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    rt.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    auto &color = colors[i];
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView = rt.view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    // Lazy clear (DELTA_GPU_LAZYCLEAR, default on): persist RT content across frames
    // (LOAD), clearing only when the game explicitly requested a clear (clearPending) or
    // the RT was never rendered. The old per-frame auto-clear wiped baked-once content
    // (room floor) whose redraw lands on a different frame than its clear.
    if (lazyClear) {
      // DELTA_GPU_CLEARTRACE: which draw opens a region with a clear, and to what.
      if (std::getenv("DELTA_GPU_CLEARTRACE") && (rt.clearPending || !rt.everRendered)) {
        static int n = 0;
        if (n++ < 24)
          std::fprintf(stderr,
                       "[clear] draw#%u RT %ux%u loadOp=CLEAR value=(%g %g %g %g) "
                       "pending=%d ever=%d\n",
                       g_frame.draws, rt.w, rt.h, rt.clearValue.float32[0],
                       rt.clearValue.float32[1], rt.clearValue.float32[2],
                       rt.clearValue.float32[3], (int)rt.clearPending,
                       (int)rt.everRendered);
      }
      color.loadOp = (rt.clearPending || !rt.everRendered) ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                           : VK_ATTACHMENT_LOAD_OP_LOAD;
    } else
      color.loadOp = rt.usedThisFrame ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    rt.clearPending = false;
    rt.everRendered = true;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = rt.clearValue;
    // DELTA_GPU_CLEARCOLOR / DELTA_GPU_CLEARRED: diagnostic knobs that force every
    // bound RT to clear to a solid colour this frame, to verify which RTs are bound.
    static const bool forceClear = std::getenv("DELTA_GPU_CLEARCOLOR") != nullptr;
    static const bool clearRed = std::getenv("DELTA_GPU_CLEARRED") != nullptr;
    if (forceClear) {
      color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      color.clearValue.color = {{0.f, 1.f, 0.f, 1.f}};
    }
    if (clearRed) {
      color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      color.clearValue.color = {{1.0f, 0.0f, 0.0f, 1.0f}};
    }
    rt.usedThisFrame = true;
    rt.lastFrame = g_frame.num;
    g_region.curMrt[g_region.curMrtCount++] = mrtBase[i];
  }
  RTarget *primary = mrtCount
                         ? getRT(mrtBase[0], w, h, colorTargetFormat(mrtInfo[0]))
                         : nullptr;
  uint64_t base = primary ? mrtBase[0] : 0;
  // Depth attachment (3D). Cleared to the guest DB_DEPTH_CLEAR value on its first use
  // this frame, then loaded so multiple regions in a frame share one Z buffer.
  VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  DepthTarget *dt = depthBase ? getDepthRT(depthBase, w, h) : nullptr;
  if (dt) {
    depthBarrier(g_frame.cmd, dt->image, dt->layout, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
    dt->layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAtt.imageView = dt->view;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    bool clearDepth = dt->clearPending || !dt->usedThisFrame;
    depthAtt.loadOp = clearDepth
                          ? VK_ATTACHMENT_LOAD_OP_CLEAR
                          : VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {dt->clearPending ? dt->clearValue : depthClear, 0};
    dt->clearPending = false;
    dt->usedThisFrame = true;
    dt->lastFrame = g_frame.num;
    g_region.curDepth = depthBase;
  }
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {w, h}};
  ri.layerCount = 1; ri.colorAttachmentCount = g_region.curMrtCount; ri.pColorAttachments = colors;
  if (dt) ri.pDepthAttachment = &depthAtt;
  p_vkCmdBeginRendering(g_frame.cmd, &ri);
  g_region.open = true;
  // Negative-height (y-up) viewport: GCN/PS4 rasterises y-up, so we do too. This
  // stores render-target content upright, so render-to-texture composites (the
  // scene->scanout copy, effect overlays) sample it with aligned UVs when run through
  // the game's real recompiled shader, and the presented scanout is already upright
  // (no readback flip needed; DELTA_GPU_FLIP defaults to 0).
  VkViewport vpt{0, (float)h, (float)w, -(float)h, 0, 1};
  vkCmdSetViewport(g_frame.cmd, 0, 1, &vpt);
  VkRect2D sc{{0, 0}, {w, h}};
  vkCmdSetScissor(g_frame.cmd, 0, 1, &sc);
  if (primary) {
    primary->usedThisFrame = true;
    primary->lastFrame = g_frame.num;
    if (primary->w >= 700 && primary->w <= 900) g_frame.roomBake = true;
    g_region.lastRt = base;
  }
  g_region.curRt = base;
  static const bool regTrace = std::getenv("DELTA_GPU_REGTRACE") != nullptr;
  if (regTrace && w < 1280)
    std::fprintf(stderr, "[reg] f%d begin RT %#lx %ux%u mrt=%u clear=%d\n", g_frame.num,
                  (unsigned long)base, w, h, g_region.curMrtCount,
                  g_region.curMrtCount && colors[0].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
}

}  // namespace gpu::vk

namespace gpu::rhi {
using namespace gpu::vk;

void noteMemoryFill(uint64_t base, uint64_t bytes, uint32_t value) {
  if (!g_dev.ready || !bytes) return;
  const uint64_t end = base + bytes;
  for (auto &kv : g_rts) {
    RTarget &rt = kv.second;
    const uint64_t rtEnd = kv.first + rtByteSize(rt);
    // Only a fill that covers the whole surface is a clear; a partial one is a
    // buffer update that happens to overlap.
    if (base > kv.first || end < rtEnd) continue;
    rt.clearPending = true;
    // The fill value is one dword of the target's own format. Unpacking every
    // format is not worth it: a clear is almost always zero (black), and a
    // non-zero fill lands as its 8-bit-per-channel reading.
    const float inv = 1.0f / 255.0f;
    rt.clearValue.float32[0] = ((value >> 0) & 0xFF) * inv;
    rt.clearValue.float32[1] = ((value >> 8) & 0xFF) * inv;
    rt.clearValue.float32[2] = ((value >> 16) & 0xFF) * inv;
    rt.clearValue.float32[3] = ((value >> 24) & 0xFF) * inv;
    static const bool trace = std::getenv("DELTA_GPU_FILLTRACE") != nullptr;
    static int n = 0;
    if (trace && n++ < 20)
      std::fprintf(stderr, "[fill] RT %#lx cleared by CP DMA fill %08x (%lu bytes)\n",
                   (unsigned long)kv.first, value, (unsigned long)bytes);
  }
}

}  // namespace gpu::rhi
