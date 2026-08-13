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
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <utl/options.h>

namespace {
// DELTA_GPU_CLEARTRACE=<n>: print up to n clear-opening lines. A small cap only
// ever shows the loading screens; a lazy clear that lands on the wrong draw in
// a steady-state frame needs a cap large enough to reach that frame.
DELTA_OPTION(int, kClearTrace, "DELTA_GPU_CLEARTRACE", 0);
DELTA_OPTION(bool, kLazyClear, "DELTA_GPU_LAZYCLEAR", true);
DELTA_OPTION(bool, kClearRed, "DELTA_GPU_CLEARRED", false);
DELTA_OPTION(bool, kForceClear, "DELTA_GPU_CLEARCOLOR", false);
// DELTA_GPU_FILLTRACE=<n>: print up to n fill-clears. Fills happen every
// frame, so a fixed cap only ever shows the first seconds.
DELTA_OPTION(int, kGpuFilltrace, "DELTA_GPU_FILLTRACE", 0);
DELTA_OPTION(bool, kRegTrace, "DELTA_GPU_REGTRACE", false);
// DELTA_GPU_RESOLVETRACE=1: report a sampled address that more than one live
// render target can answer for. The winner is scored by freshness, and two
// targets rendered in the same frame TIE -- broken by page-table insertion
// order, i.e. by the guest addresses of the run. That is a per-run coin flip
// deciding which image a pass reads.
DELTA_OPTION(bool, kResolveTrace, "DELTA_GPU_RESOLVETRACE", false);
// DELTA_GPU_NOVARIANT=1: never swap in a parked geometry variant. One base
// rendered at two geometries owns two images, and which one answers to the
// address depends on draw order -- so an artefact that alternates frame to
// frame has to be tested against variant switching before anything else.
DELTA_OPTION(bool, kNoVariant, "DELTA_GPU_NOVARIANT", false);
DELTA_OPTION(uint64_t, kDepthResolveTrace, "DELTA_GPU_DEPTHRESOLVE", 0);
DELTA_OPTION(uint64_t, kFrameClearRt, "DELTA_GPU_FRAMECLEAR", 0);
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
  // A view inherits the image's usage unless told otherwise, and an SRGB view
  // of a storage-capable image is then invalid -- SRGB cannot be a storage
  // format (VUID-VkImageViewCreateInfo-usage-02275). These views are only ever
  // sampled, so say so.
  VkImageViewUsageCreateInfo vu{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
  vu.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
  vi.pNext = &vu;
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
  VkImageViewUsageCreateInfo vu{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
  vu.usage = VK_IMAGE_USAGE_SAMPLED_BIT;  // see SampledImageView
  vci.pNext = &vu;
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
// A freshly created VkImage's contents are UNDEFINED. That is invisible for a
// target whose first use is a clear or a full-screen draw, but it is not
// invisible for one a pass READS before it writes: P.T.'s light buffer is
// consumed by a feedback pass (ps=0x80b54b4000) that un-premultiplies it,
// out.rgb = rgb / (1 - alpha), which can only ever INCREASE a texel. Fed
// undefined memory on its first frame, whatever garbage the allocation happened
// to hold is amplified every frame until it saturates fp16, and because that
// pass never decreases a pixel it can never recover -- the frame ends up with a
// few dozen texels pinned at 65504 for the rest of the run, which then bloom
// into blown-white blobs. Guest memory a title renders into holds defined
// values; an undefined image is our artefact, so define it.
void ClearNewRt(RTarget& t) {
  VkCommandBufferAllocateInfo ca{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = g_dev.pool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  VkCommandBuffer c = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(g_dev.device, &ca, &c) != VK_SUCCESS)
    return;
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(c, &bi) != VK_SUCCESS) {
    vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &c);
    return;
  }
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  b.image = t.image;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                       1, &b);
  const VkClearColorValue zero{{0.f, 0.f, 0.f, 0.f}};
  const VkImageSubresourceRange sr{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(c, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero,
                       1, &sr);
  VkImageMemoryBarrier b1 = b;
  b1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  b1.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  b1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &b1);
  if (vkEndCommandBuffer(c) == VK_SUCCESS) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &c;
    if (vkResetFences(g_dev.device, 1, &g_dev.fence) == VK_SUCCESS &&
        vkQueueSubmit(g_dev.queue, 1, &si, g_dev.fence) == VK_SUCCESS) {
      vkWaitForFences(g_dev.device, 1, &g_dev.fence, VK_TRUE, UINT64_MAX);
      // Submitted and waited, so this layout is the real one on both timelines.
      t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      t.submitted_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
  }
  vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &c);
}

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
             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  // An SRGB format cannot be a storage image, and asking for it makes every
  // view of this image invalid (VUID-VkImageViewCreateInfo-usage-02275). A
  // compute shader that wants to write this surface reaches it through the
  // linear alias anyway, which is what MUTABLE_FORMAT is here for.
  if (fmt != VK_FORMAT_R8G8B8A8_SRGB && fmt != VK_FORMAT_B8G8R8A8_SRGB)
    ii.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
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
  ClearNewRt(t);
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
  if (kNoVariant)
    return &live;
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
    // ...including the orphaned lazy clear BeginFrame drops. Without this a
    // parked variant freezes whatever clear was pending when it was swapped
    // out, and re-applies it on EVERY later activation: P.T. aliases its
    // 1920x1080 composite with a 64x64 target, so one clear the guest asked
    // for once came back five times a frame and wiped the scene right before
    // the tonemap sampled it.
    live.clear_pending = false;
    live.clear_src = "none";
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
// The images of a depth base the guest renders to at more than one geometry,
// minus the one live in g_depths -- the same arrangement the colour targets
// use (see g_rt_variants). Without it, the first geometry to touch a base
// fixed it forever and every later pass at another size rendered into a corner
// of that image: P.T.'s 1920x1080 depth buffer had a 960x540 clear in its
// top-left quadrant and nothing anywhere else, so every pass that sampled the
// depth -- SSAO first, then the whole post chain -- read zero.
std::unordered_map<uint64_t, std::vector<DepthTarget>> g_depth_variants;
constexpr size_t kMaxDepthVariants = 3;

bool CreateDepthImage(DepthTarget& t,
                      uint64_t base,
                      uint32_t w,
                      uint32_t h,
                      uint64_t stencil_base) {
  if (!w || !h)
    return false;
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
    return false;
  if (!g_image_memory.Allocate(g_dev, t.image, t.allocation)) {
    vkDestroyImage(g_dev.device, t.image, nullptr);
    return false;
  }
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = t.image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = kDepthFormat;
  vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
  if (vkCreateImageView(g_dev.device, &vci, nullptr, &t.view) != VK_SUCCESS) {
    vkDestroyImage(g_dev.device, t.image, nullptr);
    g_image_memory.Free(g_dev, t.allocation);
    return false;
  }
  vci.subresourceRange.aspectMask =
      VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  if (vkCreateImageView(g_dev.device, &vci, nullptr, &t.attachment_view) !=
      VK_SUCCESS) {
    vkDestroyImageView(g_dev.device, t.view, nullptr);
    vkDestroyImage(g_dev.device, t.image, nullptr);
    g_image_memory.Free(g_dev, t.allocation);
    return false;
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
  return true;
}

// Make the image of geometry (w, h) the live depth target at `base`, creating
// it on first use. Mirrors ActivateRtVariant.
DELTA_OPTION(bool, kDepthVariants, "DELTA_GPU_DEPTHVARIANTS", true);

DepthTarget* ActivateDepthVariant(DepthTarget& live,
                                  uint64_t base,
                                  uint32_t w,
                                  uint32_t h,
                                  uint64_t stencil_base) {
  // A depth attachment must COVER the render area: Vulkan lets it be larger,
  // never smaller (VUID-VkRenderingInfo-pNext-06079/06080). Every fallback
  // below used to hand the live target back whatever its geometry, and
  // BeginRegion then bound it -- which is how P.T. began a 512x512 region with
  // a 64x64 depth view and LOST THE DEVICE. Its environment-probe bake requests
  // one depth base at five geometries (960x540, 512x512, 256x256, 128x128,
  // 64x64), so the three-variant cap is reached and the cap path fires. The
  // guest walks away with the device gone: no EndFrame, no present, black
  // window, while its own threads keep running. Declining the draw (nullptr ->
  // BeginRegion false) is the only safe fallback -- a dropped draw costs one
  // pass, a lost device costs the run.
  const auto covers = [&](DepthTarget* t) -> DepthTarget* {
    return (t && t->w >= w && t->h >= h) ? t : nullptr;
  };
  if (!kDepthVariants)
    return covers(&live);
  auto& parked = g_depth_variants[base];
  DepthTarget* alt = nullptr;
  for (DepthTarget& v : parked)
    if (v.w == w && v.h == h) {
      alt = &v;
      break;
    }
  if (!alt) {
    if (parked.size() >= kMaxDepthVariants)
      return covers(&live);
    DepthTarget t;
    if (!CreateDepthImage(t, base, w, h, stencil_base))
      return covers(&live);
    std::fprintf(stderr,
                 "[gpuvk] depth alias %#lx: have %ux%u, requested %ux%u -> "
                 "own image\n",
                 (unsigned long)base, live.w, live.h, w, h);
    parked.push_back(t);
    alt = &parked.back();
  }
  std::swap(live, *alt);
  if (live.last_frame != g_frame.num) {
    live.used_this_frame = false;
    live.stencil_used_this_frame = false;
  }
  // See ActivateRtVariant: a parked target missed EndFrame's layout stamp.
  alt->submitted_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  alt->submitted_stencil_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  return &live;
}

// The depth counterpart of ActivateSampledRtVariant: a sample names a depth
// target by address, and the live geometry at that address is whichever pass
// ran last -- routinely the small half-resolution one. P.T.'s SSAO samples the
// full-resolution scene depth and was handed the 960x540 variant instead.
bool ActivateSampledDepthVariant(uint64_t base, uint32_t w, uint32_t h) {
  if (!base || !w || !h)
    return false;
  auto it = g_depths.find(base);
  if (it == g_depths.end())
    return false;
  DepthTarget& live = it->second;
  if (live.w == w && live.h == h)
    return true;
  auto parked = g_depth_variants.find(base);
  if (parked == g_depth_variants.end())
    return false;
  for (const DepthTarget& v : parked->second)
    if (v.w == w && v.h == h)
      return ActivateDepthVariant(live, base, w, h, live.stencil_base) !=
             nullptr;
  return false;
}

DepthTarget* GetDepthRT(uint64_t base,
                        uint32_t w,
                        uint32_t h,
                        uint64_t stencil_base) {
  auto it = g_depths.find(base);
  if (it != g_depths.end()) {
    DepthTarget& live = it->second;
    if (stencil_base)
      live.stencil_base = stencil_base;
    if (!w || !h || (live.w == w && live.h == h))
      return &live;
    return ActivateDepthVariant(live, base, w, h, stencil_base);
  }
  if (g_depths.size() >= 32 || !w || !h)
    return nullptr;
  DepthTarget t;
  if (!CreateDepthImage(t, base, w, h, stencil_base))
    return nullptr;
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
// Only the RT-bind path used to activate a variant, so a base whose live
// target was some other geometry stayed that way for samples. P.T. renders its
// fullscreen composite and a 64x64 pass to the same address; when the small one
// was live, the present blit sampled it, found `ever_rendered` false, and
// resolved to nothing -- a black screen with the whole scene one variant away.
bool ActivateSampledRtVariant(uint64_t base, uint32_t w, uint32_t h) {
  if (!base || !w || !h)
    return false;
  auto it = g_rts.find(base);
  if (it == g_rts.end())
    return false;
  RTarget& live = it->second;
  if (live.w == w && live.h == h && live.ever_rendered)
    return true;
  auto parked = g_rt_variants.find(base);
  if (parked == g_rt_variants.end())
    return false;
  for (const RTarget& v : parked->second) {
    // Only swap in something the guest has actually drawn into: an empty
    // variant of the right size is no better than the wrong-size one, and
    // swapping would lose the live target's content for later samples.
    if (v.w != w || v.h != h || !v.ever_rendered)
      continue;
    return ActivateRtVariant(live, base, w, h, v.fmt) != nullptr;
  }
  return false;
}

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
  uint32_t candidates = 0, ties = 0;
  uint64_t tie_with = 0;
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
    // A target may answer for an address only if it is the geometry being
    // sampled, or if it at least CONTAINS the sampled footprint. Bare interval
    // overlap is not enough: P.T. samples a 2048x1024 compute-written surface
    // whose 8 MB footprint runs across a pool of 480x270 targets, and seven of
    // them tied on freshness -- so an arbitrary half-megabyte target answered
    // for it, and which one depended on the guest addresses of that run. A
    // surface that is not a render target belongs to the texture cache.
    if (!dim_match && !(b0 <= a0 && b1 >= a1))
      return;
    long score = (long)rt.last_frame;
    if (dim_match)
      score += 1L << 30;
    if (b0 == addr)
      score += 1L << 31;
    candidates++;
    if (score == best_score) {
      ties++;
      tie_with = b0;
    }
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
  if (kResolveTrace && ties) {
    static std::atomic<uint64_t> n{0};
    if (n.fetch_add(1) < 40)
      std::fprintf(stderr,
                   "[resolve] AMBIGUOUS %#lx %ux%u: %u candidates, %u tied at "
                   "score %ld -- chose %#lx over %#lx\n",
                   (unsigned long)addr, w, h, candidates, ties, best_score,
                   (unsigned long)best, (unsigned long)tie_with);
    else if ((n.load() % 20000) == 0)
      std::fprintf(stderr, "[resolve] %llu ambiguous resolves so far\n",
                   (unsigned long long)n.load());
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
    // A base can hold several geometries and only one of them is live, but a
    // sample names the address, not the geometry. Score against every variant:
    // otherwise a base whose small variant happens to be live stops covering
    // an address its full-resolution one does, and the sample falls through to
    // guest memory -- which for a depth buffer is empty.
    uint64_t span = depth.guest_w && depth.guest_h
                        ? (uint64_t)depth.guest_w * depth.guest_h * 4
                        : RtByteSizeWH(depth.w, depth.h, kDepthFormat);
    bool dim_match = (!w || !h) || (depth.w == w && depth.h == h);
    const auto parked = g_depth_variants.find(base);
    if (parked != g_depth_variants.end())
      for (const DepthTarget& v : parked->second) {
        span = std::max(span, v.guest_w && v.guest_h
                                  ? (uint64_t)v.guest_w * v.guest_h * 4
                                  : RtByteSizeWH(v.w, v.h, kDepthFormat));
        if (w && h && v.w == w && v.h == h)
          dim_match = true;
      }
    uint64_t b1 = base + span;
    if (!(addr < b1 && base < a1))
      continue;
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
  // DELTA_GPU_DEPTHRESOLVE=<addr>: why a sampled address did or did not land on
  // a depth target. An address INSIDE a depth allocation that misses resolves
  // to a guest upload of undefined bytes, which reads as depth 0 -- i.e. the
  // far plane -- and nothing downstream looks wrong.
  if (kDepthResolveTrace && addr == (uint64_t)kDepthResolveTrace) {
    static int n = 0;
    if (n++ < 6) {
      std::fprintf(stderr, "[dres] addr=%#lx %ux%u req=%lu -> %#lx; depths:",
                   (unsigned long)addr, w, h, (unsigned long)req_size,
                   (unsigned long)best);
      for (const auto& [base, depth] : g_depths)
        std::fprintf(stderr, " [%#lx %ux%u guest=%ux%u lf=%d]",
                     (unsigned long)base, depth.w, depth.h, depth.guest_w,
                     depth.guest_h, depth.last_frame);
      std::fprintf(stderr, "\n");
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
  // Depth range: the hardware computes window_z = ndc_z * ZSCALE + ZOFFSET,
  // which is exactly Vulkan's minDepth + ndc_z * (maxDepth - minDepth). Both
  // ends must land in [0,1]; a descriptor that does not is left at the full
  // range rather than clamped into a different transform.
  // NOTE: PA_CL_VPORT_ZSCALE/ZOFFSET are deliberately NOT applied here. Doing
  // so collapses P.T.'s depth range -- every fragment lands on one value and
  // the scene stops rendering -- so whatever this title puts in those
  // registers is not the plain window_z = ndc_z * scale + offset this would
  // assume. Kept as a note rather than a knob: the registers are read into
  // DrawInfo, and the next person to try this needs to explain that first.
  const float min_depth = 0.0f, max_depth = 1.0f;
  VkViewport vp{
      d.viewport_x_offset - d.viewport_x_scale,
      d.viewport_y_offset - d.viewport_y_scale,
      d.viewport_x_scale * 2.0f,
      d.viewport_y_scale * 2.0f,
      min_depth,
      max_depth,
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
                  bool depth_read_only,
                 uint32_t depth_w,
                 uint32_t depth_h) {
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
    // DELTA_GPU_FRAMECLEAR=<base>: force this target to load CLEAR on the FIRST
    // region of each frame (not every region, which would wipe a pass before a
    // later one in the same frame reads it). Setting clear_pending from EndFrame
    // does NOT work -- it never reaches this gate -- so the earlier probe that
    // did so tested nothing.
    if (kFrameClearRt && mrt_base[i] == (uint64_t)kFrameClearRt) {
      static int cleared_frame = -1;
      if (cleared_frame != g_frame.num) {
        cleared_frame = g_frame.num;
        rt.clear_pending = true;
        rt.clear_value = VkClearColorValue{{0.f, 0.f, 0.f, 0.f}};
        rt.clear_src = "frameclear-probe";
      }
    }
    if (kLazyClear) {
      // DELTA_GPU_CLEARTRACE: which draw opens a region with a clear, and to
      // what.
      if (kClearTrace &&
          (rt.clear_pending || !rt.ever_rendered)) {
        static int n = 0;
        if (n++ < kClearTrace)
          std::fprintf(
              stderr,
              "[clear] f%d draw#%u RT %#lx %ux%u loadOp=CLEAR "
              "value=(%g %g %g %g) pending=%d(%s) ever=%d\n",
              g_frame.num, g_frame.draws, (unsigned long)mrt_base[i], rt.w,
              rt.h,
              rt.clear_value.float32[0],
              rt.clear_value.float32[1], rt.clear_value.float32[2],
              rt.clear_value.float32[3], (int)rt.clear_pending, rt.clear_src,
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
  if (dt && depth_w && depth_h) {
    // The IMAGE has to cover the render area, but the guest FOOTPRINT is the
    // Z surface's own padded geometry. Keeping them apart is what stops a
    // half-resolution depth buffer bound to a full-resolution pass from
    // claiming several megabytes it does not own -- and swallowing a
    // neighbouring surface in the sampled-address lookup.
    dt->guest_w = depth_w;
    dt->guest_h = depth_h;
  }
  if (!dt && depth_base && kRegTrace) {
    static int n = 0;
    if (n++ < 8)
      std::fprintf(stderr,
                   "[region] depth %#lx REQUESTED BUT NOT BOUND -- every depth "
                   "write in this region is dropped\n",
                   (unsigned long)depth_base);
  }
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
    if (kRegTrace) {
      static int n = 0;
      if (n++ < 200)
        std::fprintf(stderr,
                     "[region] depth %#lx %ux%u load=%s store=%s clear=%g "
                     "read_only=%d layout=%d rt=%#lx\n",
                     (unsigned long)depth_base, dt->w, dt->h,
                     clear_depth ? "CLEAR" : "LOAD",
                     read_only ? "NONE" : "STORE",
                     depth_att.clearValue.depthStencil.depth, (int)read_only,
                     (int)depth_layout, (unsigned long)base);
    }
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
    rt.clear_src = "memory-fill";
    // The fill value is one dword of the target's own format. Unpacking every
    // format is not worth it: a clear is almost always zero (black), and a
    // non-zero fill lands as its 8-bit-per-channel reading.
    const float inv = 1.0f / 255.0f;
    rt.clear_value.float32[0] = ((value >> 0) & 0xFF) * inv;
    rt.clear_value.float32[1] = ((value >> 8) & 0xFF) * inv;
    rt.clear_value.float32[2] = ((value >> 16) & 0xFF) * inv;
    rt.clear_value.float32[3] = ((value >> 24) & 0xFF) * inv;
    static int n = 0;
    if (kGpuFilltrace && n++ < kGpuFilltrace)
      std::fprintf(stderr,
                   "[fill] f%d draw#%u RT %#lx cleared by fill %08x "
                   "(base=%#lx %lu bytes)\n",
                   g_frame.num, g_frame.draws, (unsigned long)rt_base,
                   value, (unsigned long)base, (unsigned long)bytes);
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
