/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Render targets keyed by their guest address. One image serves both as a
// colour attachment and as a sampled texture, so render-to-texture and MRT fall
// out of the same lookup; an address -> image page table resolves a sampled
// address to every live image whose footprint overlaps it.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "gpu/rhi/command.h"
#include "gpu/vulkan/vk_memory.h"

namespace gpu::vk {

// An image in the resource cache: a render target keyed by its guest base
// address, that also doubles as a sampleable texture (render-to-texture). This
// is the unit of the resource model -- RT-bind and texture-sample both resolve
// to the same Image via the address page table, so render-to-texture/MRT "just
// work".
struct RTarget {
  VkImage image = VK_NULL_HANDLE;
  ImageAllocation allocation;
  VkImageView view = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;  // for sampling this RT as a texture
  // Sampling an image while it is also a color attachment requires Vulkan's
  // attachment-feedback-loop extension. Keep a lazy copy instead so the shader
  // reads the attachment contents as they existed before the draw.
  VkImage feedback_image = VK_NULL_HANDLE;
  ImageAllocation feedback_allocation;
  VkImageView feedback_view = VK_NULL_HANDLE;
  VkDescriptorSet feedback_set = VK_NULL_HANDLE;
  std::unordered_map<uint32_t, VkImageView> sampled_views;
  std::unordered_map<uint32_t, VkImageView> feedback_sampled_views;
  VkImageLayout feedback_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  uint32_t w = 0, h = 0;
  VkFormat fmt =
      VK_FORMAT_B8G8R8A8_UNORM;  // identity: addr alone doesn't pin format
  bool is_depth = false;         // depth/stencil attachment (MRT/depth task)
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  // Rendered into since the last barrier that made it readable. A later draw
  // sampling this target needs an execution/memory dependency even when the
  // layout already matches, or it reads what was there BEFORE those writes --
  // which is how SotC's composite sampled its scene target and got black.
  bool dirty_for_read = false;
  // Layout the image holds once the last SUBMITTED work executes (stamped at
  // EndFrame submit). `layout` tracks the recording timeline, which runs
  // ahead of the GPU: a mid-frame submission (the compute path staging an
  // RT-backed CS input) executes before this frame's still-recording
  // barriers, so its own barriers must chain from -- and restore -- the
  // submitted state, never `layout`.
  VkImageLayout submitted_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  bool used_this_frame = false;
  uint32_t draws = 0;      // draws into this RT this frame
  int last_frame = -1000;  // frame number this RT was last rendered into
  bool clear_pending =
      false;  // a fullscreen black clear was requested; applied lazily
              // (as loadOp=CLEAR) only when later content redraws this RT
              // in the same frame.
  VkClearColorValue clear_value{{0.0f, 0.0f, 0.0f, 0.0f}};
  bool ever_rendered =
      false;  // false until first real render (then loadOp can LOAD)
  // Guest code addresses of the last recompiled pair that rendered into this
  // target. Render-target guest bases move between runs, so a poisoned target
  // can only be traced back to its producer within the run that observed it;
  // the reporting path (DELTA_GPU_RTSTAT) prints these.
  uint64_t last_vs = 0, last_ps = 0;
  // Bit i: cbuffer binding i of that draw resolved to readable guest memory.
  // Bindings that do not resolve are bound to a zero window, which is what
  // turns a tonemap's constants into 0 and its output into NaN.
  uint32_t last_cbuf_mask = 0;
  // Bit i: set-2 raw-buffer binding i of that draw resolved to readable guest
  // memory and was staged. A shader that indexes a buffer by hand reads zeros
  // for every bit that is clear, which is indistinguishable in the output from
  // a buffer that is genuinely zero -- so it has to be reported.
  uint32_t last_rawbuf_mask = 0;
};

extern std::unordered_map<uint64_t, RTarget>& g_rts;

// Depth/stencil attachment, keyed by its guest DB_Z_WRITE_BASE. Allocated on
// demand when a 3D draw binds a Z buffer. Internal format is always D32_SFLOAT
// (we never read depth back to guest memory, so only a valid depth format is
// needed).
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
struct DepthTarget {
  VkImage image = VK_NULL_HANDLE;
  ImageAllocation allocation;
  VkImageView view = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  std::unordered_map<uint32_t, VkImageView> sampled_views;
  uint32_t w = 0, h = 0;
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  // Rendered into since the last barrier that made it readable. A later draw
  // sampling this target needs an execution/memory dependency even when the
  // layout already matches, or it reads what was there BEFORE those writes --
  // which is how SotC's composite sampled its scene target and got black.
  bool dirty_for_read = false;
  // See RTarget::submitted_layout: the anchor for mid-frame copies.
  VkImageLayout submitted_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  int last_frame = -1000;
  bool used_this_frame = false;
  bool clear_pending = false;
  float clear_value = 1.0f;
};

extern std::unordered_map<uint64_t, DepthTarget>& g_depths;

// Address -> image page table (the resource model's core). Maps a 64 KiB guest
// page to the RT bases whose memory footprint covers it, so a sampled address
// resolves to every overlapping live image in O(pages) instead of scanning the
// whole cache. A page can be touched by several overlapping/aliased RTs
// (double-buffer pairs, a pool of cycled scene buffers), so each page holds a
// list.
constexpr uint32_t kRtPageShift = 16;  // 64 KiB
extern std::unordered_map<uint64_t, std::vector<uint64_t>>& g_rt_pages;

uint64_t RtByteSizeWH(uint32_t w, uint32_t h, VkFormat fmt);
uint64_t RtByteSize(const RTarget& rt);

RTarget* GetRT(uint64_t base, uint32_t w, uint32_t h, VkFormat fmt);
DepthTarget* GetDepthRT(uint64_t base, uint32_t w, uint32_t h);

// Sampling an image while it is also a colour attachment needs Vulkan's
// attachment-feedback-loop extension; copy it instead and sample the copy.
VkDescriptorSet SnapshotRT(RTarget& rt);

VkImageView SampledView(RTarget& rt, uint32_t swizzle, bool feedback = false);
VkImageView SampledView(DepthTarget& depth, uint32_t swizzle);

// Resolve a sampled guest address to the live image backing it (0 = none).
uint64_t ResolveSampledRT(uint64_t addr, uint32_t w, uint32_t h);
uint64_t ResolveSampledDepth(uint64_t addr, uint32_t w, uint32_t h);

// The open dynamic-rendering region, and which targets the frame has touched.
struct RenderRegion {
  uint64_t cur_rt = 0;        // primary RT (MRT0) of the open region (0 = none)
  uint64_t cur_mrt[8] = {0};  // all colour targets bound in the open region
  uint32_t cur_mrt_count = 0;
  uint64_t cur_depth = 0;  // depth target bound in the open region (0 = none)
  uint64_t last_rt = 0;    // last RT rendered to (present fallback)
  uint64_t first_rt = 0;   // first colour RT created (diagnostic selector)
  uint64_t busiest_rt = 0;
  uint32_t busiest_rt_draws = 0;
  bool open = false;
};

extern RenderRegion& g_region;

bool BeginRegion(const uint64_t* mrt_base,
                 const uint32_t* mrt_info,
                 uint32_t mrt_count,
                 uint32_t w,
                 uint32_t h,
                 uint64_t depth_base = 0,
                 float depth_clear = 1.0f);
void EndRegion();
void SetGuestViewport(const rhi::DrawInfo& d);

}  // namespace gpu::vk
