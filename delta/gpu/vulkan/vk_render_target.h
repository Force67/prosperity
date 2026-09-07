/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Render targets keyed by their guest address. One image serves both as a
// colour attachment and as a sampled texture, so render-to-texture and MRT fall
// out of the same lookup; an address -> image page table resolves a sampled
// address to every live image whose footprint overlaps it.

#include <vulkan/vulkan.h>
#include "base/arch.h"

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
  std::unordered_map<u32, VkImageView> sampled_views;
  // Views of the same image in a different (size-compatible) format, keyed by
  // swizzle | format<<16. See SampledViewAs.
  std::unordered_map<u32, VkImageView> alias_views;
  std::unordered_map<u32, VkImageView> feedback_sampled_views;
  VkImageLayout feedback_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  u32 w = 0, h = 0;
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
  u32 draws = 0;      // draws into this RT this frame
  int last_frame = -1000;  // frame number this RT was last rendered into
  bool clear_pending =
      false;  // a fullscreen black clear was requested; applied lazily
              // (as loadOp=CLEAR) only when later content redraws this RT
              // in the same frame.
  VkClearColorValue clear_value{{0.0f, 0.0f, 0.0f, 0.0f}};
  // Which mechanism asked for that clear (static string, DELTA_GPU_CLEARTRACE
  // prints it). A clear that wipes live content is only fixable once you know
  // whether the guest asked for it or one of our heuristics did.
  const char* clear_src = "none";
  bool ever_rendered =
      false;  // false until first real render (then loadOp can LOAD)
  // Guest code addresses of the last recompiled pair that rendered into this
  // target. Render-target guest bases move between runs, so a poisoned target
  // can only be traced back to its producer within the run that observed it;
  // the reporting path (DELTA_GPU_RTSTAT) prints these.
  u64 last_vs = 0, last_ps = 0;
  // Bit i: cbuffer binding i of that draw resolved to readable guest memory.
  // Bindings that do not resolve are bound to a zero window, which is what
  // turns a tonemap's constants into 0 and its output into NaN.
  u32 last_cbuf_mask = 0;
  // Bit i: set-2 raw-buffer binding i of that draw resolved to readable guest
  // memory and was staged. A shader that indexes a buffer by hand reads zeros
  // for every bit that is clear, which is indistinguishable in the output from
  // a buffer that is genuinely zero -- so it has to be reported.
  u32 last_rawbuf_mask = 0;
  // CB_COLORn_DCC_BASE of the last draw that bound this target: the
  // compression metadata a fast clear writes instead of the pixels. A write
  // over it (NoteDccWrite) is turned into the lazy clear at the next bind,
  // when the code it wrote can be read back.
  u64 dcc_base = 0;
  bool dcc_clear_pending = false;
  bool dcc_code_known = false;  // an immediate fill carried its value
  u32 dcc_clear_code = 0;
};

// The live target at each guest address. An address the guest renders to at
// several geometries owns one image per geometry, of which this holds the one
// the last draw rendered into; the others are parked in vk_render_target.cc, so
// resolving a target by address alone keeps working.
extern std::unordered_map<u64, RTarget>& g_rts;

// Depth/stencil attachment, keyed by its guest DB_Z_WRITE_BASE. A combined host
// image preserves the separate PS4 Z and stencil planes for raster and compute.
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
struct DepthTarget {
  VkImage image = VK_NULL_HANDLE;
  ImageAllocation allocation;
  VkImageView view = VK_NULL_HANDLE;  // depth-only sampled view
  // Stencil-plane sampled view, made on demand. A deferred renderer reads the
  // stencil plane as an R8_UINT texture to recover the material id it wrote
  // during the G-buffer pass; without this the address resolves to whatever
  // colour target happens to overlap it and the shader discards every pixel.
  VkImageView stencil_view = VK_NULL_HANDLE;
  VkImageView attachment_view = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  std::unordered_map<u32, VkImageView> sampled_views;
  u32 w = 0, h = 0;
  // DB_DEPTH_SIZE's padded geometry: how much guest memory this Z surface
  // actually owns, which is not the image size when the guest binds a
  // half-resolution depth buffer to a full-resolution pass.
  u32 guest_w = 0, guest_h = 0;
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout stencil_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  u64 stencil_base = 0;
  // Rendered into since the last barrier that made it readable. A later draw
  // sampling this target needs an execution/memory dependency even when the
  // layout already matches, or it reads what was there BEFORE those writes --
  // which is how SotC's composite sampled its scene target and got black.
  bool dirty_for_read = false;
  // See RTarget::submitted_layout: the anchor for mid-frame copies.
  VkImageLayout submitted_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout submitted_stencil_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  int last_frame = -1000;
  bool used_this_frame = false;
  bool stencil_used_this_frame = false;
  bool clear_pending = false;
  float clear_value = 1.0f;
  // DB_HTILE_DATA_BASE of the last draw that bound this target, and whether
  // something wrote over it since (the depth fast clear, see NoteDccWrite).
  // A target with a known HTILE base is cleared by those writes alone, not
  // on its first bind of every frame: the guest's frame does not line up with
  // ours, and Astro Bot's depth prepass lands in the frame before the one
  // that tests against it.
  u64 htile_base = 0;
  bool htile_clear_pending = false;
  bool htile_code_known = false;
  u32 htile_clear_code = 0;
};

extern std::unordered_map<u64, DepthTarget>& g_depths;
// Depth images of a base rendered at more than one geometry; only the one in
// g_depths answers to the address (see ActivateDepthVariant).
extern std::unordered_map<u64, std::vector<DepthTarget>> g_depth_variants;
// Colour images of a base rendered at more than one geometry (see
// ActivateRtVariant); only the one in g_rts answers to the address.
extern std::unordered_map<u64, std::vector<RTarget>> g_rt_variants;

// Address -> image page table (the resource model's core). Maps a 64 KiB guest
// page to the RT bases whose memory footprint covers it, so a sampled address
// resolves to every overlapping live image in O(pages) instead of scanning the
// whole cache. A page can be touched by several overlapping/aliased RTs
// (double-buffer pairs, a pool of cycled scene buffers), so each page holds a
// list.
constexpr u32 kRtPageShift = 16;  // 64 KiB
extern std::unordered_map<u64, std::vector<u64>>& g_rt_pages;

u64 RtByteSizeWH(u32 w, u32 h, VkFormat fmt);
u64 RtByteSize(const RTarget& rt);

RTarget* GetRT(u64 base, u32 w, u32 h, VkFormat fmt);
DepthTarget* GetDepthRT(u64 base,
                        u32 w,
                        u32 h,
                        u64 stencil_base = 0);

// Sampling an image while it is also a colour attachment needs Vulkan's
// attachment-feedback-loop extension; copy it instead and sample the copy.
VkDescriptorSet SnapshotRT(RTarget& rt);

VkImageView SampledView(RTarget& rt, u32 swizzle, bool feedback = false);
// Sampled view reinterpreted into `want` (same texel size) so the view's
// numeric type matches the shader's OpTypeImage sampled type.
// `used` takes the format the view was actually created with, which is not
// always `want`: the caller needs it to decide whether the binding may be
// filtered.
VkImageView SampledViewAs(RTarget& rt, u32 swizzle, VkFormat want,
                          VkFormat* used = nullptr);
VkImageView SampledView(DepthTarget& depth, u32 swizzle);

// Resolve a sampled guest address to the live image backing it (0 = none).
// A sample names a render target by address alone, but one base can hold
// several geometries (see ActivateRtVariant). Make the variant matching the
// sampled geometry the live one when the live target cannot serve the sample.
// Returns true if a usable target is live at `base` afterwards.
bool ActivateSampledRtVariant(u64 base, u32 w, u32 h);
bool ActivateSampledDepthVariant(u64 base, u32 w, u32 h);

u64 ResolveSampledRT(u64 addr, u32 w, u32 h);
u64 ResolveSampledDepth(u64 addr, u32 w, u32 h);
// The depth target whose STENCIL plane lives at `addr` (DB_STENCIL_WRITE_BASE),
// or 0. Keyed exactly, because the stencil plane is a separate guest surface.
u64 ResolveSampledStencil(u64 addr);
// Sampled view of a depth target's stencil plane (S8_UINT), created on demand.
VkImageView StencilSampledView(DepthTarget& depth);

// Preserve an already-rendered depth target for a later compute read before a
// same-frame clear destroys it. No-op when no compute range aliases the target.
bool PreserveCsDepthBeforeClear(u64 base);

// A write of `bytes` at `base` by a dispatch or a CP DMA fill. If it covers a
// live target's DCC metadata it is that target's fast clear: the hardware
// reads the clear code out of the metadata and never touches the pixels, so
// nothing else in the stream says the target was cleared. `fill` is the dword
// written when the packet carries it; a compute write is read back at the
// next bind.
void NoteDccWrite(u64 base, u64 bytes, const u32* fill);

// A dispatch wrote `bytes` at `base` through a raw buffer, i.e. not through
// any image the compute bridge could carry into a target. Guest memory under
// a target it overlaps is now newer than the image, so until a draw renders
// into it again a sample of that address has to read the memory: Astro Bot
// reuses the pages of its title-screen UI target for an exposure texel, and
// the image is what the old surface last held.
void NoteRawWrite(u64 base, u64 bytes);

// A dispatch is about to write `base` at geometry (w, h): make that
// geometry's image the live one so the write lands where a later sample at
// that geometry looks, whether or not it has been rendered before.
bool ActivateWrittenRtVariant(u64 base, u32 w, u32 h);

// The open dynamic-rendering region, and which targets the frame has touched.
struct RenderRegion {
  u64 cur_rt = 0;        // primary RT (MRT0) of the open region (0 = none)
  u64 cur_mrt[8] = {0};  // all colour targets bound in the open region
  u32 cur_mrt_count = 0;
  u32 cur_fmt[8] = {0};  // colour attachment formats the region opened with
  u32 cur_w[8] = {0};    // per-attachment extents the region opened with
  u32 cur_h[8] = {0};
  u32 cur_area_w = 0;  // render area the region opened with
  u32 cur_area_h = 0;
  u64 cur_depth = 0;  // depth target bound in the open region (0 = none)
  u64 cur_stencil = 0;
  u64 last_rt = 0;    // last RT rendered to (present fallback)
  u64 first_rt = 0;   // first colour RT created (diagnostic selector)
  u64 busiest_rt = 0;
  u32 busiest_rt_draws = 0;
  bool open = false;
  // The open region binds depth read-only because a draw in it samples the same
  // image. Tracked so a later draw that needs to write depth restarts.
  bool depth_read_only = false;
};

extern RenderRegion& g_region;

bool BeginRegion(const u64* mrt_base,
                 const u32* mrt_info,
                 u32 mrt_count,
                 u32 w,
                  u32 h,
                  u64 depth_base = 0,
                  float depth_clear = 1.0f,
                  u64 stencil_base = 0,
                  u8 stencil_clear = 0,
                  bool depth_read_only = false,
                  u32 depth_w = 0,
                  u32 depth_h = 0,
                  const u32* mrt_surf_w = nullptr,
                  const u32* mrt_surf_h = nullptr,
                  const u64* mrt_dcc_base = nullptr,
                  const u32 (*mrt_clear_word)[2] = nullptr,
                  u64 depth_htile_base = 0);

// The extent an attachment's image needs: its own surface geometry when that is
// believable, else the region the pass draws into.
// `tile` is the alignment the drawn region is allowed to be padded to: 256
// pixels of pitch for width, 64 rows of slice for height.
u32 RtSurfaceExtent(u32 surface, u32 drawn, u32 tile);

// The Z surface's own geometry when DB_DEPTH_SIZE gave one, else the colour
// target's. Sizing a depth image from the colour pass is wrong whenever the
// two differ, and the error shows up as a footprint that swallows neighbours.
inline u32 DepthW(const rhi::DrawInfo& d) {
  return d.depth_w ? d.depth_w : d.rt_w;
}
inline u32 DepthH(const rhi::DrawInfo& d) {
  return d.depth_h ? d.depth_h : d.rt_h;
}
void EndRegion();
void SetGuestViewport(const rhi::DrawInfo& d);

}  // namespace gpu::vk
