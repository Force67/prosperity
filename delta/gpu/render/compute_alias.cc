/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// Bridges between compute ranges and the live render targets and depth
// images that alias their guest memory.

#include "gpu/render/compute_internal.h"

#include <climits>
#include <cstring>

#include "base/algorithm.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/strings/format.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/guest_memory.h"
#include "gpu/render/frame.h"
#include "gpu/render/guest_format.h"
#include "gpu/render/render_target.h"
#include "gpu/render/trace.h"
#include "guest/session.h"
#include "profile/profile.h"

namespace gpu::render::cs {

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
  static const guest::SessionReset reset_warned(
      [] { guest::ResetResource(warned); });
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

// Device is lost: every later submit fails regardless of what it records, so
// work that only waits must not pretend its failure is a per-staging miss and
// keep recording.
bool g_cs_failed = false;

base::Vector<BridgeFloatImage> g_bridge_float_images;
u64 g_in_frame_bridge_n = 0;

bool BlitsBetween(rhi::Format a, rhi::Format b) {
  static base::HashMap<u64, bool> known;
  static const guest::SessionReset reset_known(
      [] { guest::ResetResource(known); });
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

const char* g_rec_why = "";
bool RecordRtStageInFrame(const InFrameTarget& t,
                          const ComputeInfo::Res& res,
                          CsRange& e,
                          const AliasedCopyPlan& plan) {
  struct Span {
    rhi::CommandList* c;
    ~Span() {
      if (c)
        c->ProfileEnd();
    }
  } span{g_frame.recording ? g_frame.list : nullptr};
  if (span.c)
    span.c->ProfileBegin(5);
  const CsAliasedImage& img = t.img;
  const u32 layers = base::Max(res.layers, 1u);
  g_rec_why = !e.buf                                      ? "no-buf"
              : (e.truth && plan.widen)                   ? "truth-widen"
              : (!e.truth && plan.depth16)                ? "depth16"
              : t.layout == rhi::TextureState::kUndefined ? "layout-undef"
              : (e.truth && plan.unpack)                  ? "truth-unpack"
                                                          : "layers";
  if (!kCsInFrameBridge || !e.buf || (e.truth && plan.widen) ||
      (!e.truth && plan.depth16) ||
      t.layout == rhi::TextureState::kUndefined || (e.truth && plan.unpack) ||
      (layers == 1 ? plan.layers != 1 || img.layers != 1
                   : plan.layers > layers || plan.unpack || img.is_stencil ||
                         plan.widen))
    return false;
  g_rec_why = "late";
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(res, tiled, linear))
    return false;
  const auto& level = linear.mips[0];
  // Layers sit one stored slice apart in the linear staging layout.
  const u64 layer_bytes = static_cast<u64>(level.pitch) * level.stored_height *
                          res.stage_elem_bytes;
  if (layers > 1 && level.size != layer_bytes * layers)
    return false;
  // Layers the target lacks read as zero, as on the host path.
  const u64 copy_bytes =
      layer_bytes * (plan.layers - 1) +
      static_cast<u64>(level.pitch) * plan.h * res.stage_elem_bytes;
  const TileTable* table = nullptr;
  rhi::Buffer* linear_buf = e.buf;
  // A staged stencil or 8/16-bit colour target holds a dword per texel: the
  // plane is copied out packed and the tiling shader widens it into the
  // staging. A stencil truth is retiled from the packed plane.
  const bool widen_packed = !e.truth && (img.is_stencil || plan.widen);
  const bool packed_truth = e.truth && img.is_stencil;
  const u32 packed_elem = img.is_stencil ? 1 : res.elem_bytes;
  gcn::TextureLayout32 packed;
  if ((widen_packed || packed_truth) &&
      (res.elem_bytes != packed_elem ||
       (widen_packed && res.stage_elem_bytes != 4) || plan.layers != 1 ||
       !gcn::BuildTextureLayout32(packed, res.width, res.height, res.pitch, 1,
                                  1, 8, res.pow2_pad, packed_elem) ||
       !(table = packed_truth ? GetTileTable(tiled, packed, /*packed=*/true)
                              : GetTileTable(packed, linear)) ||
       !(linear_buf = AcquireScratch(packed.size)) || res.size > e.cap))
    return false;
  if (e.truth && !packed_truth) {
    if (!(table = GetTileTable(tiled, linear)) ||
        !(linear_buf = AcquireScratch(res.size)))
      return false;
  } else if (!widen_packed && !packed_truth &&
             (level.offset + copy_bytes > e.cap || res.size > e.cap)) {
    return false;
  }
  rhi::Texture* float_image = nullptr;
  if (plan.unpack && (!BlitsBetween((t.fmt), rhi::Format::kRGBA32Float) ||
                      !(float_image = GetBridgeFloatImage(plan.w, plan.h))))
    return false;
  // Between frames the copy rides the compute batch, which goes on the queue
  // after the frame that rendered the target and before the next one.
  if (!g_frame.recording) {
    CsBatchBeginImpl();
    if (!CsBatchOpenOutOfFrame())
      return false;
  }
  EndRegion();
  rhi::CommandList* const c = g_frame.recording ? g_frame.list : g_cs_list;
  c->Barrier(kAccessAll, kAccessCopyRW);
  constexpr rhi::TextureState kSrc = rhi::TextureState::kCopySrc;
  AliasedImageBarrier(c, img, t.layout, kSrc);
  // Staging bytes the copy does not reach read as zero, as the host path's
  // cleared mirror did.
  if (widen_packed) {
    c->FillBuffer(e.buf, 0, res.size & ~u64(3), 0);
    c->Barrier(rhi::kAccessCopyWrite, kAccessComputeRW | kAccessCopyRW);
  } else if (level.offset || copy_bytes < res.size || plan.w < level.pitch) {
    c->FillBuffer(linear_buf, 0, res.size & ~u64(3), 0);
    c->Barrier(rhi::kAccessCopyWrite, rhi::kAccessCopyWrite);
  }
  rhi::BufferTextureCopy copy;
  copy.buffer_offset = level.offset;
  copy.row_length = level.pitch;
  copy.image_height = level.stored_height;
  if (widen_packed || packed_truth) {
    copy.buffer_offset = packed.mips[0].offset;
    copy.row_length = packed.mips[0].pitch;
    copy.image_height = packed.mips[0].stored_height;
  }
  copy.region.aspect = img.aspect;
  copy.region.layers = plan.layers;
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
  if (widen_packed)
    RecordImageTiling(c, *table, linear_buf, 0, packed.size, e.buf, res.size,
                      /*detile=*/true, 1, 1);
  else if (packed_truth)
    RecordImageTiling(c, *table, e.buf, 0, e.guest_bytes, linear_buf,
                      packed.size, /*detile=*/false, 1, 1);
  else if (table)
    RecordImageTiling(c, *table, e.buf, 0, e.guest_bytes, linear_buf, res.size,
                      /*detile=*/false, 1, layers, plan.depth16);
  AliasedImageBarrier(c, img, kSrc, t.layout);
  c->Barrier(rhi::kAccessCopyWrite, kAccessAll);
  e.frame_ref = g_frame.num;
  g_in_frame_bridge_n++;
  return CsSplitFrameChunk();
}

// The reverse of RecordRtStageInFrame: a dispatch's output copied into the
// live target it aliases, on the frame's list behind the draws recorded so
// far (between frames on the batch, ahead of the next frame).
bool RecordRtUploadInFrame(const InFrameTarget& t,
                           const ComputeInfo::Res& res,
                           CsRange& e,
                           const AliasedCopyPlan& plan) {
  const CsAliasedImage& img = t.img;
  if (!kCsInFrameBridge || !e.buf || plan.layers != 1 ||
      (e.truth && (plan.unpack || plan.widen)) ||
      (!e.truth && plan.depth16) || t.layout == rhi::TextureState::kUndefined)
    return false;
  gcn::TextureLayout32 tiled, linear, packed;
  if (!BuildCsImageLayouts(res, tiled, linear))
    return false;
  // A stencil plane and 8/16-bit colour go in packed: from a truth by a
  // detile straight to the packed form, from a widened staging by a retile.
  const bool narrow = img.is_stencil || plan.widen;
  const u32 packed_elem = img.is_stencil ? 1 : res.elem_bytes;
  const TileTable* table = nullptr;
  rhi::Buffer* src = e.buf;
  if (narrow) {
    if (res.elem_bytes != packed_elem ||
        (!e.truth && res.stage_elem_bytes != 4) ||
        !gcn::BuildTextureLayout32(packed, res.width, res.height, res.pitch, 1,
                                   1, 8, res.pow2_pad, packed_elem) ||
        !(table = e.truth ? GetTileTable(tiled, packed, /*packed=*/true)
                          : GetTileTable(packed, linear)) ||
        !(src = AcquireScratch((packed.size + 3) & ~u64(3))))
      return false;
  } else if (e.truth) {
    if (!(table = GetTileTable(tiled, linear)) ||
        !(src = AcquireScratch(res.size)))
      return false;
  } else if (linear.mips[0].offset +
                 u64(linear.mips[0].pitch) * plan.h * res.stage_elem_bytes >
             e.cap) {
    return false;
  }
  rhi::Texture* float_image = nullptr;
  if (plan.unpack && (!BlitsBetween(rhi::Format::kRGBA32Float, t.fmt) ||
                      !(float_image = GetBridgeFloatImage(plan.w, plan.h))))
    return false;
  if (!g_frame.recording) {
    CsBatchBeginImpl();
    if (!CsBatchOpenOutOfFrame())
      return false;
  }
  EndRegion();
  rhi::CommandList* const c = g_frame.recording ? g_frame.list : g_cs_list;
  c->Barrier(kAccessAll, kAccessComputeRW | kAccessCopyRW);
  const u64 packed_bytes = (packed.size + 3) & ~u64(3);
  if (narrow && e.truth)
    RecordImageTiling(c, *table, e.buf, 0, e.guest_bytes, src, packed_bytes,
                      /*detile=*/true, 1, 1);
  else if (narrow)
    RecordImageTiling(c, *table, src, 0, packed_bytes, e.buf, res.size,
                      /*detile=*/false, 1, 1);
  else if (e.truth)
    RecordImageTiling(c, *table, e.buf, 0, e.guest_bytes, src, res.size,
                      /*detile=*/true, 1, 1, plan.depth16);
  const auto& level = narrow ? packed.mips[0] : linear.mips[0];
  rhi::BufferTextureCopy copy;
  copy.buffer_offset = level.offset;
  copy.row_length = level.pitch;
  copy.image_height = level.stored_height;
  copy.region.aspect = img.aspect;
  copy.region.width = plan.w;
  copy.region.height = plan.h;
  constexpr rhi::TextureState kDst = rhi::TextureState::kCopyDst;
  if (plan.unpack) {
    rhi::TextureBarrier fb;
    fb.texture = float_image;
    fb.before = rhi::TextureState::kUndefined;
    fb.after = kDst;
    c->Barrier(0, 0, &fb, 1);
    c->CopyBufferToTexture(float_image, src, &copy, 1);
    fb.before = kDst;
    fb.after = rhi::TextureState::kCopySrc;
    c->Barrier(0, 0, &fb, 1);
    rhi::TextureRegion region;
    region.width = plan.w;
    region.height = plan.h;
    AliasedImageBarrier(c, img, t.layout, kDst);
    c->BlitTexture(img.texture, region, float_image, region,
                   rhi::Filter::kNearest);
  } else {
    AliasedImageBarrier(c, img, t.layout, kDst);
    c->CopyBufferToTexture(img.texture, src, &copy, 1);
  }
  AliasedImageBarrier(c, img, kDst, t.layout);
  c->Barrier(rhi::kAccessCopyWrite, kAccessAll);
  e.rt_seq = e.write_seq;
  e.frame_ref = g_frame.num;
  if (e.pending_batch && g_frame.recording)
    g_cs_chunk_needs_batch = true;
  g_in_frame_bridge_n++;
  return true;
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
  if (it != g_depths.end())
    return use_depth(it->second);
  // A stencil plane allocated at an address of its own.
  for (auto& [depth_base, depth] : g_depths) {
    (void)depth_base;
    if (depth.stencil_base != base || !depth.texture || depth.clear_pending)
      continue;
    t.layout = depth.stencil_layout;
    t.img = {depth.texture, depth.w, depth.h, 1,    rhi::kAspectStencil,
             t.layout,      false,   true,    depth.layers};
    return true;
  }
  return false;
}

// All on the GPU: the copy goes into the frame chunk, so it needs no wait for
// a batch still using the buffer, only that batch on the queue first.
bool StageCsRangeFromRtInFrame(const ComputeInfo::Res& res, CsRange& e) {
  DELTA_ZONE("gpu.cs_stage_from_rt");
  InFrameTarget live;
  AliasedCopyPlan live_plan;
  const char* why = nullptr;
  if (!LiveInFrameTarget(res.base, /*prefer_depth=*/res.dfmt == 4, live))
    why = "no-live-target";
  else if (!PlanAliasedCopy(live.img, res, "reads", live_plan))
    why = "plan";
  else if (!RecordRtStageInFrame(live, res, e, live_plan))
    why = g_rec_why;
  if (why && CsSyncReporting()) {
    static base::HashSet<u64> seen;
    static const guest::SessionReset reset_seen(
        [] { guest::ResetResource(seen); });
    if (seen.size() < 64 && seen.insert(res.base).second)
      BASE_LOGI("csbridge",
                "{:#x} {}x{}x{} dfmt={} in frame: {} (img {}x{} "
                "layers={} plan layers={})",
                res.base, res.width, res.height, res.layers, res.dfmt, why,
                live.img.w, live.img.h, live.img.layers, live_plan.layers);
  }
  return !why;
}

// Read shapes no target bridge can serve (the live image is another size):
// staged from guest memory without first waiting for the range to go idle.
base::HashSet<u64> g_rt_unbridgeable;
u64 RtShapeKey(const ComputeInfo::Res& res) {
  return res.base ^ (u64{res.width} << 40) ^ (u64{res.height} << 20) ^
         res.layers;
}

// The reverse: a CS result written to a range that a live render/depth target
// aliases must also land in the image, because draws sample the image,
// never guest memory (SotC's exposure/bloom compute writes the adapted scene
// into an RT the tonemap then samples; its depth downsample writes the
// half-res depth pyramid). e.buf already holds the linear pixel data the
// dispatch produced, so upload straight from it. Guest memory was refreshed
// by the caller either way; a shape mismatch just leaves the image stale.
bool UploadCsRangeToRt(u64 base, CsRange& e) {
  if (!e.image_staging)
    return true;
  // The frame list already copied this revision into the target
  // (CsRefreshRtInFrame); uploading it again would paint over whatever was
  // rendered into it since.
  if (e.rt_seq == e.write_seq)
    return true;
  // On the frame's own list, behind the draws already recorded.
  if (g_frame.recording && CsRefreshRtInFrame(base, e))
    return true;
  // A base rendered at several geometries: the dispatch names which one it
  // writes, and only the live image answers to the address.
  ActivateWrittenRtVariant(base, e.res.width, e.res.height);
  InFrameTarget live;
  if (!LiveInFrameTarget(base, /*prefer_depth=*/e.res.dfmt == 4, live))
    return true;  // nothing to refresh
  const CsAliasedImage& img = live.img;
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
  if (!RecordRtUploadInFrame(live, e.res, e, plan))
    return false;
  if (!img.is_depth) {
    RTarget& rt = g_rts[base];
    rt.ever_rendered = true;  // CS content is real content
    rt.last_frame = g_frame.num;
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

}  // namespace gpu::render::cs
