/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_render_target.h"

#include "gpu/rhi/renderer.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_format.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_texture_cache.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gpu::vk {

using rhi::DrawInfo;

VkImageView SampledImageView(VkImage image,
                             VkImageView identity,
                             VkFormat format,
                             VkImageAspectFlags aspect,
                             uint32_t swizzle,
                             std::unordered_map<uint32_t, VkImageView>& views) {
  const VkComponentMapping components = TextureComponents(swizzle);
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

VkImageView SampledView(RTarget& rt, uint32_t swizzle, bool feedback) {
  return feedback ? SampledImageView(rt.feedback_image, rt.feedback_view,
                                     rt.fmt, VK_IMAGE_ASPECT_COLOR_BIT, swizzle,
                                     rt.feedback_sampled_views)
                  : SampledImageView(rt.image, rt.view, rt.fmt,
                                     VK_IMAGE_ASPECT_COLOR_BIT, swizzle,
                                     rt.sampled_views);
}

VkImageView SampledView(DepthTarget& depth, uint32_t swizzle) {
  return SampledImageView(depth.image, depth.view, kDepthFormat,
                          VK_IMAGE_ASPECT_DEPTH_BIT, swizzle,
                          depth.sampled_views);
}

uint64_t RtByteSizeWH(uint32_t w, uint32_t h, VkFormat fmt) {
  return (uint64_t)w * h * FormatBytes(fmt);
}

// Register an RT's footprint pages so the page table can find it by overlap.
void RegisterRtPages(uint64_t base, uint32_t w, uint32_t h, VkFormat fmt) {
  uint64_t lo = base >> kRtPageShift;
  uint64_t hi = (base + RtByteSizeWH(w, h, fmt) - 1) >> kRtPageShift;
  for (uint64_t p = lo; p <= hi; p++) {
    auto& v = g_rt_pages[p];
    bool seen = false;
    for (uint64_t b : v)
      if (b == base) {
        seen = true;
        break;
      }
    if (!seen)
      v.push_back(base);
  }
}

// Find or create the render target at guest address `base` (dimensions w x h).
RTarget* GetRT(uint64_t base, uint32_t w, uint32_t h, VkFormat fmt) {
  auto it = g_rts.find(base);
  if (it != g_rts.end()) {
    static std::unordered_map<uint64_t, bool> alias_warned;
    if ((it->second.w != w || it->second.h != h || it->second.fmt != fmt) &&
        !alias_warned[base]) {
      alias_warned[base] = true;
      std::fprintf(stderr,
                   "[gpuvk] RT alias mismatch %#lx: have %ux%u fmt=%d, "
                   "requested %ux%u fmt=%d\n",
                   (unsigned long)base, it->second.w, it->second.h,
                   (int)it->second.fmt, w, h, (int)fmt);
    }
    return &it->second;
  }
  if (g_rts.size() >= 64 || !w || !h)
    return nullptr;
  // Robustness: reject render targets with implausible dimensions or an
  // undefined format. A garbage CB_COLOR base/scissor (e.g. a stray shader-pool
  // RT address on the PS5 path) would otherwise feed the driver an invalid
  // image and hard-crash it.
  if (w > 8192 || h > 8192 || fmt == VK_FORMAT_UNDEFINED) {
    std::fprintf(stderr, "[gpuvk] skip bad RT %#lx %ux%u fmt=%d\n",
                 (unsigned long)base, w, h, (int)fmt);
    return nullptr;
  }
  RTarget t;
  t.w = w;
  t.h = h;
  t.fmt = fmt;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = fmt;
  ii.extent = {w, h, 1};
  ii.mipLevels = 1;
  ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
             VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (vkCreateImage(g_dev.device, &ii, nullptr, &t.image) != VK_SUCCESS)
    return nullptr;
  if (!AllocateImageMemory(t.image, t.allocation)) {
    vkDestroyImage(g_dev.device, t.image, nullptr);
    return nullptr;  // GPU OOM -> don't bind/view a memory-less image (driver
                     // crash)
  }
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = t.image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = fmt;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  if (vkCreateImageView(g_dev.device, &vci, nullptr, &t.view) != VK_SUCCESS) {
    vkDestroyImage(g_dev.device, t.image, nullptr);
    FreeImageMemory(t.allocation);
    return nullptr;
  }
  // descriptor set so this RT can be sampled (render-to-texture).
  if (g_tex.ds_pool) {
    VkDescriptorPool owner;
    t.set = AllocateSamplerSet(g_tex.ds_layout, false, owner);
    if (t.set) {
      VkDescriptorImageInfo dii{g_tex.sampler, t.view,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      wr.dstSet = t.set;
      wr.descriptorCount = 1;
      wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      wr.pImageInfo = &dii;
      vkUpdateDescriptorSets(g_dev.device, 1, &wr, 0, nullptr);
    }
  }
  std::fprintf(stderr, "[gpuvk] new RT %#lx %ux%u fmt=%d\n",
               (unsigned long)base, w, h, (int)fmt);
  g_rts[base] = t;
  if (!g_region.first_rt)
    g_region.first_rt = base;
  RegisterRtPages(base, w, h, fmt);  // resource-model page table
  return &g_rts[base];
}

VkDescriptorSet SnapshotRT(RTarget& rt) {
  if (!rt.feedback_image) {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = rt.fmt;
    ii.extent = {rt.w, rt.h, 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (vkCreateImage(g_dev.device, &ii, nullptr, &rt.feedback_image) !=
        VK_SUCCESS)
      return VK_NULL_HANDLE;
    if (!AllocateImageMemory(rt.feedback_image, rt.feedback_allocation)) {
      vkDestroyImage(g_dev.device, rt.feedback_image, nullptr);
      rt.feedback_image = VK_NULL_HANDLE;
      return VK_NULL_HANDLE;
    }
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = rt.feedback_image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = rt.fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(g_dev.device, &vi, nullptr, &rt.feedback_view) !=
        VK_SUCCESS) {
      vkDestroyImage(g_dev.device, rt.feedback_image, nullptr);
      FreeImageMemory(rt.feedback_allocation);
      rt.feedback_image = VK_NULL_HANDLE;
      return VK_NULL_HANDLE;
    }
    VkDescriptorPool owner;
    rt.feedback_set = AllocateSamplerSet(g_tex.ds_layout, false, owner);
    if (!rt.feedback_set) {
      vkDestroyImageView(g_dev.device, rt.feedback_view, nullptr);
      vkDestroyImage(g_dev.device, rt.feedback_image, nullptr);
      FreeImageMemory(rt.feedback_allocation);
      rt.feedback_view = VK_NULL_HANDLE;
      rt.feedback_image = VK_NULL_HANDLE;
      return VK_NULL_HANDLE;
    }
    VkDescriptorImageInfo dii{g_tex.sampler, rt.feedback_view,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr.dstSet = rt.feedback_set;
    wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr.pImageInfo = &dii;
    vkUpdateDescriptorSets(g_dev.device, 1, &wr, 0, nullptr);
  }

  ImageBarrier(g_frame.cmd, rt.image, rt.layout,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               ColorImageAccess(rt.layout), VK_ACCESS_TRANSFER_READ_BIT);
  rt.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  VkAccessFlags feedback_access =
      rt.feedback_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
          ? VK_ACCESS_SHADER_READ_BIT
          : 0;
  ImageBarrier(g_frame.cmd, rt.feedback_image, rt.feedback_layout,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, feedback_access,
               VK_ACCESS_TRANSFER_WRITE_BIT);
  rt.feedback_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  VkImageCopy copy{};
  copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copy.extent = {rt.w, rt.h, 1};
  vkCmdCopyImage(g_frame.cmd, rt.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 rt.feedback_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                 &copy);
  ImageBarrier(g_frame.cmd, rt.feedback_image, rt.feedback_layout,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
  rt.feedback_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  return rt.feedback_set;
}

uint64_t RtByteSize(const RTarget& rt) {
  return RtByteSizeWH(rt.w, rt.h, rt.fmt);
}
// Find or create the depth target at guest address `base` (dimensions w x h).
DepthTarget* GetDepthRT(uint64_t base, uint32_t w, uint32_t h) {
  auto it = g_depths.find(base);
  if (it != g_depths.end())
    return &it->second;
  if (g_depths.size() >= 32 || !w || !h)
    return nullptr;
  DepthTarget t;
  t.w = w;
  t.h = h;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = kDepthFormat;
  ii.extent = {w, h, 1};
  ii.mipLevels = 1;
  ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage =
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  if (vkCreateImage(g_dev.device, &ii, nullptr, &t.image) != VK_SUCCESS)
    return nullptr;
  if (!AllocateImageMemory(t.image, t.allocation)) {
    vkDestroyImage(g_dev.device, t.image, nullptr);
    return nullptr;
  }
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = t.image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = kDepthFormat;
  vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
  if (vkCreateImageView(g_dev.device, &vci, nullptr, &t.view) != VK_SUCCESS) {
    vkDestroyImage(g_dev.device, t.image, nullptr);
    FreeImageMemory(t.allocation);
    return nullptr;
  }
  if (g_tex.ds_pool) {
    VkDescriptorPool owner;
    t.set = AllocateSamplerSet(g_tex.ds_layout, false, owner);
    if (t.set) {
      VkDescriptorImageInfo dii{g_tex.sampler, t.view,
                                VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      wr.dstSet = t.set;
      wr.descriptorCount = 1;
      wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      wr.pImageInfo = &dii;
      vkUpdateDescriptorSets(g_dev.device, 1, &wr, 0, nullptr);
    }
  }
  std::fprintf(stderr, "[gpuvk] new depth %#lx %ux%u\n", (unsigned long)base, w,
               h);
  g_depths[base] = t;
  return &g_depths[base];
}

// Resolve a sampled texture address to the live RT image that backs it (the
// resource model's "the page table collects all overlappers" lookup). The game
// cycles/aliases RT addresses (double-buffered room layers, a pool of scene
// buffers), so a composite often samples an address that does not exactly match
// the RT base it was rendered into. We gather every RT whose footprint touches
// the sampled region's pages and resolve by IDENTITY, not recency: an exact
// base whose size matches wins outright (the guest bound that buffer, so a more
// recently rendered overlapping buffer must never override it); only when no
// exact match exists do we fall back to the best-fitting overlapper (dimension
// match, then the freshest). Returns the g_rts key, or 0.
uint64_t ResolveSampledRT(uint64_t addr, uint32_t w, uint32_t h) {
  if (!addr)
    return 0;
  uint64_t req_size = w && h ? (uint64_t)w * h * 4 : 4;
  uint64_t a0 = addr, a1 = addr + req_size;
  // Exact-identity hit: the guest sampled this exact base and it is a live RT
  // of the requested size. That is unambiguously the right image -- return it
  // before any freshness comparison can pick an overlapping cycled buffer
  // instead.
  auto ex = g_rts.find(addr);
  if (ex != g_rts.end() && ex->second.ever_rendered &&
      ((!w || !h) || (ex->second.w == w && ex->second.h == h)))
    return addr;
  uint64_t best = 0;
  long best_score = -1;
  auto consider = [&](uint64_t b0) {
    auto it = g_rts.find(b0);
    if (it == g_rts.end())
      return;
    const RTarget& rt = it->second;
    if (!rt.ever_rendered)
      return;  // never sample an RT with no content
    uint64_t b1 = b0 + RtByteSize(rt);
    if (!(a0 < b1 && b0 < a1))
      return;  // no interval overlap
    bool dim_match = (!w || !h) || (rt.w == w && rt.h == h);
    long score = (long)rt.last_frame;
    if (dim_match)
      score += 1L << 30;
    if (b0 == addr)
      score += 1L << 31;
    if (score > best_score) {
      best_score = score;
      best = b0;
    }
  };
  for (uint64_t p = a0 >> kRtPageShift; p <= (a1 - 1) >> kRtPageShift; p++) {
    auto it = g_rt_pages.find(p);
    if (it != g_rt_pages.end())
      for (uint64_t b0 : it->second)
        consider(b0);
  }
  return best;
}

// Depth images use a separate registry from color RTs. A GFX7 depth surface may
// be sampled through an R32_FLOAT descriptor whose base denotes an overlapping
// view rather than DB_Z_WRITE_BASE, so resolve typed depth aliases by footprint
// too.
uint64_t ResolveSampledDepth(uint64_t addr, uint32_t w, uint32_t h) {
  if (!addr)
    return 0;
  uint64_t req_size = w && h ? (uint64_t)w * h * 4 : 4;
  uint64_t a1 = addr + req_size;
  uint64_t best = 0;
  long best_score = -1;
  for (const auto& [base, depth] : g_depths) {
    if (depth.last_frame <= -1000)
      continue;
    uint64_t b1 = base + RtByteSizeWH(depth.w, depth.h, kDepthFormat);
    if (!(addr < b1 && base < a1))
      continue;
    bool dim_match = (!w || !h) || (depth.w == w && depth.h == h);
    long score = depth.last_frame;
    if (dim_match)
      score += 1L << 30;
    if (base == addr)
      score += 1L << 31;
    if (score > best_score) {
      best_score = score;
      best = base;
    }
  }
  return best;
}

// End the current dynamic-rendering region. Attachments remain in attachment
// layouts until an actual sampled/transfer consumer requests a transition.
void EndRegion() {
  if (!g_region.open)
    return;
  g_cmd_end_rendering(g_frame.cmd);
  g_region.open = false;
  g_region.cur_rt = 0;
  g_region.cur_mrt_count = 0;
  g_region.cur_depth = 0;
}

void SetGuestViewport(const DrawInfo& d) {
  if (!std::isfinite(d.viewport_x_scale) ||
      !std::isfinite(d.viewport_x_offset) ||
      !std::isfinite(d.viewport_y_scale) ||
      !std::isfinite(d.viewport_y_offset) || d.viewport_x_scale <= 0.0f ||
      d.viewport_y_scale == 0.0f)
    return;
  VkViewport vp{
      d.viewport_x_offset - d.viewport_x_scale,
      d.viewport_y_offset - d.viewport_y_scale,
      d.viewport_x_scale * 2.0f,
      d.viewport_y_scale * 2.0f,
      0.0f,
      1.0f,
  };
  vkCmdSetViewport(g_frame.cmd, 0, 1, &vp);
}

// Begin a dynamic-rendering region binding mrt_count color targets (mrt_base[0]
// is the primary). The common single-RT case (mrt_count == 1) binds exactly one
// attachment. depth_base != 0 additionally binds a depth attachment (cleared to
// depth_clear on its first use each frame, loaded thereafter); depth_base == 0
// leaves depth unbound (the 2D path).
bool BeginRegion(const uint64_t* mrt_base,
                 const uint32_t* mrt_info,
                 uint32_t mrt_count,
                 uint32_t w,
                 uint32_t h,
                 uint64_t depth_base,
                 float depth_clear) {
  static const bool kLazyClear = [] {
    const char* e = std::getenv("DELTA_GPU_LAZYCLEAR");
    return !e || std::strcmp(e, "0") != 0;
  }();
  VkRenderingAttachmentInfo colors[8]{};
  RTarget* targets[8]{};
  mrt_count = std::min(mrt_count, 8u);
  for (uint32_t i = 0; i < mrt_count; i++) {
    targets[i] = GetRT(mrt_base[i], w, h, ColorTargetFormat(mrt_info[i]));
    if (!targets[i])
      return false;
  }
  DepthTarget* dt = depth_base ? GetDepthRT(depth_base, w, h) : nullptr;
  if (depth_base && !dt)
    return false;
  g_region.cur_mrt_count = 0;
  for (uint32_t i = 0; i < mrt_count; i++) {
    RTarget& rt = *targets[i];
    ImageBarrier(g_frame.cmd, rt.image, rt.layout,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 ColorImageAccess(rt.layout),
                 VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    rt.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    auto& color = colors[i];
    color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color.imageView = rt.view;
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    // Lazy clear (DELTA_GPU_LAZYCLEAR, default on): persist RT content across
    // frames (LOAD), clearing only when the game explicitly requested a clear
    // (clear_pending) or the RT was never rendered. The old per-frame
    // auto-clear wiped baked-once content (room floor) whose redraw lands on a
    // different frame than its clear.
    if (kLazyClear) {
      // DELTA_GPU_CLEARTRACE: which draw opens a region with a clear, and to
      // what.
      if (std::getenv("DELTA_GPU_CLEARTRACE") &&
          (rt.clear_pending || !rt.ever_rendered)) {
        static int n = 0;
        if (n++ < 24)
          std::fprintf(
              stderr,
              "[clear] draw#%u RT %ux%u loadOp=CLEAR value=(%g %g %g %g) "
              "pending=%d ever=%d\n",
              g_frame.draws, rt.w, rt.h, rt.clear_value.float32[0],
              rt.clear_value.float32[1], rt.clear_value.float32[2],
              rt.clear_value.float32[3], (int)rt.clear_pending,
              (int)rt.ever_rendered);
      }
      color.loadOp = (rt.clear_pending || !rt.ever_rendered)
                         ? VK_ATTACHMENT_LOAD_OP_CLEAR
                         : VK_ATTACHMENT_LOAD_OP_LOAD;
    } else
      color.loadOp = rt.used_this_frame ? VK_ATTACHMENT_LOAD_OP_LOAD
                                        : VK_ATTACHMENT_LOAD_OP_CLEAR;
    rt.clear_pending = false;
    rt.ever_rendered = true;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = rt.clear_value;
    // DELTA_GPU_CLEARCOLOR / DELTA_GPU_CLEARRED: diagnostic knobs that force
    // every bound RT to clear to a solid colour this frame, to verify which RTs
    // are bound.
    static const bool kForceClear =
        std::getenv("DELTA_GPU_CLEARCOLOR") != nullptr;
    static const bool kClearRed = std::getenv("DELTA_GPU_CLEARRED") != nullptr;
    if (kForceClear) {
      color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      color.clearValue.color = {{0.f, 1.f, 0.f, 1.f}};
    }
    if (kClearRed) {
      color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      color.clearValue.color = {{1.0f, 0.0f, 0.0f, 1.0f}};
    }
    rt.used_this_frame = true;
    rt.last_frame = g_frame.num;
    g_region.cur_mrt[i] = mrt_base[i];
  }
  g_region.cur_mrt_count = mrt_count;
  RTarget* primary = mrt_count ? targets[0] : nullptr;
  uint64_t base = primary ? mrt_base[0] : 0;
  // Depth attachment (3D). Cleared to the guest DB_DEPTH_CLEAR value on its
  // first use this frame, then loaded so multiple regions in a frame share one
  // Z buffer.
  VkRenderingAttachmentInfo depth_att{
      VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  if (dt) {
    const VkAccessFlags depth_source =
        dt->layout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
            ? VK_ACCESS_SHADER_READ_BIT
        : dt->layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
            ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            : 0;
    DepthBarrier(g_frame.cmd, dt->image, dt->layout,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, depth_source,
                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
    dt->layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_att.imageView = dt->view;
    depth_att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    bool clear_depth = dt->clear_pending || !dt->used_this_frame;
    depth_att.loadOp =
        clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_att.clearValue.depthStencil = {
        dt->clear_pending ? dt->clear_value : depth_clear, 0};
    dt->clear_pending = false;
    dt->used_this_frame = true;
    dt->last_frame = g_frame.num;
    g_region.cur_depth = depth_base;
  }
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {w, h}};
  ri.layerCount = 1;
  ri.colorAttachmentCount = g_region.cur_mrt_count;
  ri.pColorAttachments = colors;
  if (dt)
    ri.pDepthAttachment = &depth_att;
  g_cmd_begin_rendering(g_frame.cmd, &ri);
  g_region.open = true;
  // Negative-height (y-up) viewport: GCN/PS4 rasterises y-up, so we do too.
  // This stores render-target content upright, so render-to-texture composites
  // (the scene->scanout copy, effect overlays) sample it with aligned UVs when
  // run through the game's real recompiled shader, and the presented scanout is
  // already upright (no readback flip needed; DELTA_GPU_FLIP defaults to 0).
  VkViewport vpt{0, (float)h, (float)w, -(float)h, 0, 1};
  vkCmdSetViewport(g_frame.cmd, 0, 1, &vpt);
  VkRect2D sc{{0, 0}, {w, h}};
  vkCmdSetScissor(g_frame.cmd, 0, 1, &sc);
  if (primary) {
    primary->used_this_frame = true;
    primary->last_frame = g_frame.num;
    if (primary->w >= 700 && primary->w <= 900)
      g_frame.room_bake = true;
    g_region.last_rt = base;
  }
  g_region.cur_rt = base;
  static const bool kRegTrace = std::getenv("DELTA_GPU_REGTRACE") != nullptr;
  if (kRegTrace && w < 1280)
    std::fprintf(stderr, "[reg] f%d begin RT %#lx %ux%u mrt=%u clear=%d\n",
                 g_frame.num, (unsigned long)base, w, h, g_region.cur_mrt_count,
                 g_region.cur_mrt_count &&
                     colors[0].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
  return true;
}

}  // namespace gpu::vk

namespace gpu::rhi {
using namespace gpu::vk;

void NoteMemoryFill(Renderer& renderer,
                    uint64_t base,
                    uint64_t bytes,
                    uint32_t value) {
  if (!renderer.available() || !bytes)
    return;
  const uint64_t end = base + bytes;
  for (auto& kv : g_rts) {
    RTarget& rt = kv.second;
    const uint64_t rt_end = kv.first + RtByteSize(rt);
    // Only a fill that covers the whole surface is a clear; a partial one is a
    // buffer update that happens to overlap.
    if (base > kv.first || end < rt_end)
      continue;
    rt.clear_pending = true;
    // The fill value is one dword of the target's own format. Unpacking every
    // format is not worth it: a clear is almost always zero (black), and a
    // non-zero fill lands as its 8-bit-per-channel reading.
    const float inv = 1.0f / 255.0f;
    rt.clear_value.float32[0] = ((value >> 0) & 0xFF) * inv;
    rt.clear_value.float32[1] = ((value >> 8) & 0xFF) * inv;
    rt.clear_value.float32[2] = ((value >> 16) & 0xFF) * inv;
    rt.clear_value.float32[3] = ((value >> 24) & 0xFF) * inv;
    static const bool trace = std::getenv("DELTA_GPU_FILLTRACE") != nullptr;
    static int n = 0;
    if (trace && n++ < 20)
      std::fprintf(stderr,
                   "[fill] RT %#lx cleared by CP DMA fill %08x (%lu bytes)\n",
                   (unsigned long)kv.first, value, (unsigned long)bytes);
  }
}

}  // namespace gpu::rhi
