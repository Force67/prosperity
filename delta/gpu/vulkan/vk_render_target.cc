/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_render_target.h"

#include "gpu/rhi/renderer.h"
#include "gpu/vulkan/vk_debug.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_format.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_texture_cache.h"
#include "gpu/vulkan/vk_trace.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kClearTrace, "DELTA_GPU_CLEARTRACE", false);
DELTA_OPTION(bool, kLazyClear, "DELTA_GPU_LAZYCLEAR", true);
DELTA_OPTION(bool, kClearRed, "DELTA_GPU_CLEARRED", false);
DELTA_OPTION(bool, kForceClear, "DELTA_GPU_CLEARCOLOR", false);
DELTA_OPTION(bool, kGpuFilltrace, "DELTA_GPU_FILLTRACE", false);
DELTA_OPTION(bool, kRegTrace, "DELTA_GPU_REGTRACE", false);
}  // namespace

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

// Sampled view of a colour target in `want` rather than the target's own
// format, when the two are the same size (so the reinterpretation is legal and
// the bytes line up). Falls back to the target's format when they are not.
VkImageView SampledViewAs(RTarget& rt, uint32_t swizzle, VkFormat want) {
  if (want == VK_FORMAT_UNDEFINED || want == rt.fmt ||
      FormatBytes(want) != FormatBytes(rt.fmt))
    return SampledView(rt, swizzle);
  const uint32_t key = swizzle | (static_cast<uint32_t>(want) << 16);
  const auto it = rt.alias_views.find(key);
  if (it != rt.alias_views.end())
    return it->second;
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = rt.image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = want;
  vci.components = TextureComponents(swizzle);
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView v = VK_NULL_HANDLE;
  if (vkCreateImageView(g_dev.device, &vci, nullptr, &v) != VK_SUCCESS)
    return SampledView(rt, swizzle);
  rt.alias_views[key] = v;
  return v;
}

VkImageView SampledView(RTarget& rt, uint32_t swizzle, bool feedback) {
  // A target that became live at this address after the draw took its snapshot
  // (an alias switch, see ActivateRtVariant) has no copy of its own yet.
  // Sampling the attachment instead would be the feedback loop the copy exists
  // to avoid, so leave the caller its default texture.
  if (feedback && !rt.feedback_image)
    return VK_NULL_HANDLE;
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

// Allocate the image, view and sampler descriptor backing one target. Split out
// of GetRT because an address the guest renders to at two geometries needs a
// second image (see ActivateRtVariant).
bool CreateRtImage(RTarget& t,
                   uint64_t base,
                   uint32_t w,
                   uint32_t h,
                   VkFormat fmt) {
  if (!w || !h)
    return false;
  // Robustness: reject render targets with implausible dimensions or an
  // undefined format. A garbage CB_COLOR base/scissor (e.g. a stray shader-pool
  // RT address on the PS5 path) would otherwise feed the driver an invalid
  // image and hard-crash it.
  if (w > 8192 || h > 8192 || fmt == VK_FORMAT_UNDEFINED) {
    std::fprintf(stderr, "[gpuvk] skip bad RT %#lx %ux%u fmt=%d\n",
                 (unsigned long)base, w, h, (int)fmt);
    return false;
  }
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
  // The colour format a pass RENDERS with and the format a later shader SAMPLES
  // the same memory with are independent on PS4 -- a G-buffer plane written as
  // UINT is read back through a T# that may name a different numeric type, and
  // Vulkan requires the view's numeric type to match the shader's sampled type.
  // Mutable format lets SampledViewAs() hand out a view in the format the
  // descriptor asked for instead of the one the attachment was created with.
  ii.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
  if (vkCreateImage(g_dev.device, &ii, nullptr, &t.image) != VK_SUCCESS)
    return false;
  if (!g_image_memory.Allocate(g_dev, t.image, t.allocation)) {
    vkDestroyImage(g_dev.device, t.image, nullptr);
    t.image = VK_NULL_HANDLE;
    return false;  // GPU OOM -> don't bind/view a memory-less image (driver
                   // crash)
  }
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = t.image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = fmt;
  vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  if (vkCreateImageView(g_dev.device, &vci, nullptr, &t.view) != VK_SUCCESS) {
    vkDestroyImage(g_dev.device, t.image, nullptr);
    g_image_memory.Free(g_dev, t.allocation);
    t.image = VK_NULL_HANDLE;
    return false;
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
  NameObject(VK_OBJECT_TYPE_IMAGE, (uint64_t)t.image, "rt %#lx %ux%u fmt=%d",
             (unsigned long)base, w, h, (int)fmt);
  return true;
}

// The images of a base the guest renders to at more than one geometry, minus
// the one that is live in g_rts. P.T. renders a fullscreen pass and then a
// 512x512 pass to the same address inside one frame; sharing one image lands
// the small pass in a corner of the big one and every later sample of that
// address reads mostly stale pixels. Every lookup in the backend names a target
// by address alone, so g_rts keeps holding the live target and the other
// geometries wait here until a draw asks for them again.
std::unordered_map<uint64_t, std::vector<RTarget>> g_rt_variants;
constexpr size_t kMaxRtVariants = 3;

// Make the image of geometry (w, h, fmt) the live target at `base`, creating it
// on first use.
RTarget* ActivateRtVariant(RTarget& live,
                           uint64_t base,
                           uint32_t w,
                           uint32_t h,
                           VkFormat fmt) {
  auto& parked = g_rt_variants[base];
  RTarget* alt = nullptr;
  for (RTarget& v : parked)
    if (v.w == w && v.h == h && v.fmt == fmt) {
      alt = &v;
      break;
    }
  if (!alt) {
    if (parked.size() >= kMaxRtVariants)
      return &live;
    RTarget t;
    if (!CreateRtImage(t, base, w, h, fmt))
      return nullptr;
    std::fprintf(stderr,
                 "[gpuvk] RT alias %#lx: have %ux%u fmt=%d, requested %ux%u "
                 "fmt=%d -> own image\n",
                 (unsigned long)base, live.w, live.h, (int)live.fmt, w, h,
                 (int)fmt);
    parked.push_back(t);
    alt = &parked.back();
    RegisterRtPages(base, w, h, fmt);
  }
  std::swap(live, *alt);
  // BeginFrame's per-frame reset only walks the live targets, so one that slept
  // through a frame boundary catches up here.
  if (live.last_frame != g_frame.num) {
    live.used_this_frame = false;
    live.draws = 0;
  }
  // EndFrame's submitted-layout stamp skips a parked target too. Leaving a
  // value that predates its last recorded transition would have the compute
  // bridge barrier from the wrong layout; UNDEFINED means "nothing submitted
  // yet", which that path already handles.
  alt->submitted_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  return &live;
}

// Find or create the render target at guest address `base` (dimensions w x h).
RTarget* GetRT(uint64_t base, uint32_t w, uint32_t h, VkFormat fmt) {
  auto it = g_rts.find(base);
  if (it != g_rts.end()) {
    RTarget& live = it->second;
    if (live.w != w || live.h != h || live.fmt != fmt)
      return ActivateRtVariant(live, base, w, h, fmt);
    return &live;
  }
  if (g_rts.size() >= 64) {
    static int n = 0;
    if (n++ < 4)
      std::fprintf(stderr,
                   "[gpuvk] RT table full (64) -- dropping %#lx %ux%u fmt=%d "
                   "and every draw that targets it\n",
                   (unsigned long)base, w, h, (int)fmt);
    return nullptr;
  }
  RTarget t;
  if (!CreateRtImage(t, base, w, h, fmt)) {
    static int n = 0;
    if (n++ < 8)
      std::fprintf(stderr, "[gpuvk] RT image create FAILED %#lx %ux%u fmt=%d\n",
                   (unsigned long)base, w, h, (int)fmt);
    return nullptr;
  }
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
    if (!g_image_memory.Allocate(g_dev, rt.feedback_image,
                                 rt.feedback_allocation)) {
      vkDestroyImage(g_dev.device, rt.feedback_image, nullptr);
      rt.feedback_image = VK_NULL_HANDLE;
      return VK_NULL_HANDLE;
    }
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = rt.feedback_image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = rt.fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    NameObject(VK_OBJECT_TYPE_IMAGE, (uint64_t)rt.feedback_image,
               "rt feedback %ux%u", rt.w, rt.h);
    if (vkCreateImageView(g_dev.device, &vi, nullptr, &rt.feedback_view) !=
        VK_SUCCESS) {
      vkDestroyImage(g_dev.device, rt.feedback_image, nullptr);
      g_image_memory.Free(g_dev, rt.feedback_allocation);
      rt.feedback_image = VK_NULL_HANDLE;
      return VK_NULL_HANDLE;
    }
    VkDescriptorPool owner;
    rt.feedback_set = AllocateSamplerSet(g_tex.ds_layout, false, owner);
    if (!rt.feedback_set) {
      vkDestroyImageView(g_dev.device, rt.feedback_view, nullptr);
      vkDestroyImage(g_dev.device, rt.feedback_image, nullptr);
      g_image_memory.Free(g_dev, rt.feedback_allocation);
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
DepthTarget* GetDepthRT(uint64_t base,
                        uint32_t w,
                        uint32_t h,
                        uint64_t stencil_base) {
  auto it = g_depths.find(base);
  if (it != g_depths.end()) {
    if (stencil_base)
      it->second.stencil_base = stencil_base;
    return &it->second;
  }
  if (g_depths.size() >= 32 || !w || !h)
    return nullptr;
  DepthTarget t;
  t.w = w;
  t.h = h;
  t.stencil_base = stencil_base;
  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = kDepthFormat;
  ii.extent = {w, h, 1};
  ii.mipLevels = 1;
  ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  // TRANSFER src/dst: the compute path bridges CS reads/writes of a live
  // depth target through image<->buffer copies (see vk_compute.cc).
  ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
             VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (vkCreateImage(g_dev.device, &ii, nullptr, &t.image) != VK_SUCCESS)
    return nullptr;
  if (!g_image_memory.Allocate(g_dev, t.image, t.allocation)) {
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
    g_image_memory.Free(g_dev, t.allocation);
    return nullptr;
  }
  vci.subresourceRange.aspectMask =
      VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  if (vkCreateImageView(g_dev.device, &vci, nullptr, &t.attachment_view) !=
      VK_SUCCESS) {
    vkDestroyImageView(g_dev.device, t.view, nullptr);
    vkDestroyImage(g_dev.device, t.image, nullptr);
    g_image_memory.Free(g_dev, t.allocation);
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
  NameObject(VK_OBJECT_TYPE_IMAGE, (uint64_t)t.image, "depth %#lx %ux%u",
             (unsigned long)base, w, h);
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
uint64_t ResolveSampledStencil(uint64_t addr) {
  if (!addr)
    return 0;
  for (const auto& [base, depth] : g_depths)
    if (depth.stencil_base == addr && depth.last_frame > -1000)
      return base;
  return 0;
}

VkImageView StencilSampledView(DepthTarget& depth) {
  if (depth.stencil_view)
    return depth.stencil_view;
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = depth.image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = kDepthFormat;
  vci.subresourceRange = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
  if (vkCreateImageView(g_dev.device, &vci, nullptr, &depth.stencil_view) !=
      VK_SUCCESS)
    depth.stencil_view = VK_NULL_HANDLE;
  return depth.stencil_view;
}

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
  CmdEndLabel(g_frame.cmd);
  if (trace::Recording())
    trace::RegionEnd();
  g_region.open = false;
  g_region.cur_rt = 0;
  g_region.cur_mrt_count = 0;
  g_region.cur_depth = 0;
  g_region.cur_stencil = 0;
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
                  float depth_clear,
                  uint64_t stencil_base,
                  uint8_t stencil_clear,
                  bool depth_read_only) {
  VkRenderingAttachmentInfo colors[8]{};
  RTarget* targets[8]{};
  mrt_count = std::min(mrt_count, 8u);
  for (uint32_t i = 0; i < mrt_count; i++) {
    targets[i] = GetRT(mrt_base[i], w, h, ColorTargetFormat(mrt_info[i]));
    if (!targets[i])
      return false;
  }
  DepthTarget* dt =
      depth_base ? GetDepthRT(depth_base, w, h, stencil_base) : nullptr;
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
    rt.dirty_for_read = true;
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
      if (kClearTrace &&
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
  VkRenderingAttachmentInfo stencil_att{
      VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  if (dt) {
    const bool clear_depth = dt->clear_pending || !dt->used_this_frame;
    // Async compute is currently serialized ahead of the next graphics frame.
    // Keep this frame's scene depth in its persistent CS range before a later
    // pass clears the shared depth image, or next frame's compute sees zero.
    if (clear_depth && dt->used_this_frame)
      PreserveCsDepthBeforeClear(depth_base);
    const VkAccessFlags depth_source =
        dt->layout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
            ? VK_ACCESS_SHADER_READ_BIT
        : dt->layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
            ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            : 0;
    // A pass that samples the depth it tests against keeps the image in the
    // read-only layout, which is what makes attachment and sampled view legal
    // at the same time. A clear still needs write access, so never both.
    const bool read_only = depth_read_only && !clear_depth;
    const VkImageLayout depth_layout =
        read_only ? VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
                  : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    DepthBarrier(g_frame.cmd, dt->image, dt->layout, depth_layout, depth_source,
                 read_only ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_SHADER_READ_BIT
                           : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
    dt->layout = depth_layout;
    depth_att.imageView = dt->attachment_view;
    depth_att.imageLayout = depth_layout;
    depth_att.loadOp =
        clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_att.storeOp = read_only ? VK_ATTACHMENT_STORE_OP_NONE
                                  : VK_ATTACHMENT_STORE_OP_STORE;
    depth_att.clearValue.depthStencil = {
        dt->clear_pending ? dt->clear_value : depth_clear, 0};
    dt->clear_pending = false;
    dt->used_this_frame = true;
    dt->last_frame = g_frame.num;
    g_region.cur_depth = depth_base;
    if (stencil_base) {
      const bool clear_stencil = !dt->stencil_used_this_frame;
      const VkAccessFlags stencil_source =
          dt->stencil_layout == VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL
              ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
          : dt->stencil_layout == VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL
              ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                    VK_ACCESS_SHADER_READ_BIT
              : 0;
      StencilBarrier(g_frame.cmd, dt->image, dt->stencil_layout,
                     VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL,
                     stencil_source,
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
      dt->stencil_layout = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL;
      stencil_att.imageView = dt->attachment_view;
      stencil_att.imageLayout = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL;
      stencil_att.loadOp = clear_stencil ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                         : VK_ATTACHMENT_LOAD_OP_LOAD;
      stencil_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      stencil_att.clearValue.depthStencil = {depth_clear, stencil_clear};
      dt->stencil_used_this_frame = true;
      g_region.cur_stencil = stencil_base;
    }
  }
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {w, h}};
  ri.layerCount = 1;
  ri.colorAttachmentCount = g_region.cur_mrt_count;
  ri.pColorAttachments = colors;
  if (dt)
    ri.pDepthAttachment = &depth_att;
  if (dt && stencil_base)
    ri.pStencilAttachment = &stencil_att;
  if (trace::Recording()) {
    trace::RegionInfo info;
    info.mrt_base = mrt_base;
    info.mrt_info = mrt_info;
    info.mrt_count = g_region.cur_mrt_count;
    info.width = w;
    info.height = h;
    info.depth_base = depth_base;
    info.stencil_base = stencil_base;
    for (uint32_t i = 0; i < g_region.cur_mrt_count; i++)
      if (colors[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
        info.color_clear_mask |= 1u << i;
    info.depth_clear = dt && depth_att.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
    info.depth_clear_value = depth_att.clearValue.depthStencil.depth;
    trace::RegionBegin(info);
  }
  CmdBeginLabel(g_frame.cmd, "region rt=%#llx %ux%u mrt=%u depth=%#llx",
                (unsigned long long)base, w, h, g_region.cur_mrt_count,
                (unsigned long long)depth_base);
  g_cmd_begin_rendering(g_frame.cmd, &ri);
  g_region.open = true;
  g_region.depth_read_only = depth_base && depth_read_only;
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
  if (trace::Recording())
    trace::RecordMemoryFill(base, bytes, value);
  const uint64_t end = base + bytes;
  const auto note = [&](RTarget& rt, uint64_t rt_base) {
    const uint64_t rt_end = rt_base + RtByteSize(rt);
    // Only a fill that covers the whole surface is a clear; a partial one is a
    // buffer update that happens to overlap.
    if (base > rt_base || end < rt_end)
      return;
    rt.clear_pending = true;
    // The fill value is one dword of the target's own format. Unpacking every
    // format is not worth it: a clear is almost always zero (black), and a
    // non-zero fill lands as its 8-bit-per-channel reading.
    const float inv = 1.0f / 255.0f;
    rt.clear_value.float32[0] = ((value >> 0) & 0xFF) * inv;
    rt.clear_value.float32[1] = ((value >> 8) & 0xFF) * inv;
    rt.clear_value.float32[2] = ((value >> 16) & 0xFF) * inv;
    rt.clear_value.float32[3] = ((value >> 24) & 0xFF) * inv;
    static int n = 0;
    if (kGpuFilltrace && n++ < 20)
      std::fprintf(stderr,
                   "[fill] RT %#lx cleared by CP DMA fill %08x (%lu bytes)\n",
                   (unsigned long)rt_base, value, (unsigned long)bytes);
  };
  for (auto& kv : g_rts) {
    note(kv.second, kv.first);
    // A parked alias variant occupies the same address, so a fill that covers
    // it is its clear too.
    auto v = g_rt_variants.find(kv.first);
    if (v == g_rt_variants.end())
      continue;
    for (RTarget& alt : v->second)
      note(alt, kv.first);
  }
}

}  // namespace gpu::rhi
