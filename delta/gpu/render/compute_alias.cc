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

// True when `base` names a live target the compute bridges apply to. The
// UNDEFINED-submitted-layout case (target created this frame, no submission
// yet) reports false: there is nothing real to copy either way yet.
// `for_write`: the dispatch produces the surface, so an image nothing has
// rendered into yet is still the one to fill. Its layout anchor may be missing:
// a parked variant (see ActivateRtVariant) has no submitted-layout stamp, but
// one untouched this frame has nothing recorded against it either, so the
// layout its last frame left it in IS the one on the GPU.
// `prefer_depth`: the resource is a single 32-bit channel, which is what a
// depth surface looks like to a dispatch. One address can hold a colour target
// AND a depth target (Astro Bot's bloom pyramid and its half-res scene depth
// share 0x53a500000 in different passes); the colour one answered first, so the
// depth downsample landed in the bloom image.
bool FindCsAliasedImage(u64 base,
                        CsAliasedImage& out,
                        bool for_write,
                        bool prefer_depth) {
  if (prefer_depth) {
    auto depth_it = g_depths.find(base);
    if (depth_it != g_depths.end() && depth_it->second.texture) {
      DepthTarget& depth = depth_it->second;
      rhi::TextureState anchor = depth.submitted_layout;
      if (anchor == rhi::TextureState::kUndefined && for_write &&
          depth.last_frame != g_frame.num)
        anchor = depth.layout;
      out = {depth.texture, depth.w, depth.h, 4,           rhi::kAspectDepth,
             anchor,        true,    false,   depth.layers};
      return out.submitted != rhi::TextureState::kUndefined;
    }
  }
  auto rt_it = g_rts.find(base);
  if (rt_it != g_rts.end()) {
    RTarget& rt = rt_it->second;
    if (!rt.texture || rt.is_depth || (!for_write && !rt.ever_rendered))
      return false;
    rhi::TextureState anchor = rt.submitted_layout;
    if (anchor == rhi::TextureState::kUndefined && for_write &&
        rt.last_frame != g_frame.num)
      anchor = rt.layout;
    out = {rt.texture,        rt.w,   rt.h,  FormatBytes(rt.fmt),
           rhi::kAspectColor, anchor, false, false};
    return out.submitted != rhi::TextureState::kUndefined;
  }
  auto depth_it = g_depths.find(base);
  if (depth_it != g_depths.end()) {
    DepthTarget& depth = depth_it->second;
    if (!depth.texture)
      return false;
    out = {depth.texture,
           depth.w,
           depth.h,
           4,  // kDepthFormat == D32_SFLOAT
           rhi::kAspectDepth,
           depth.submitted_layout,
           true,
           false,
           depth.layers};
    return out.submitted != rhi::TextureState::kUndefined;
  }
  for (auto& [depth_base, depth] : g_depths) {
    (void)depth_base;
    if (depth.stencil_base != base || !depth.texture)
      continue;
    out = {depth.texture,
           depth.w,
           depth.h,
           1,
           rhi::kAspectStencil,
           depth.submitted_stencil_layout,
           false,
           true};
    return out.submitted != rhi::TextureState::kUndefined;
  }
  return false;
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

CsBridgeScratch g_bridge;

bool EnsureBridgeScratch(u64 bytes) {
  if (g_bridge.cap >= bytes)
    return true;
  Device().Destroy(g_bridge.buf);
  g_bridge = CsBridgeScratch{};
  const u64 cap = (bytes + 0xFFFF) & ~u64(0xFFFF);
  g_bridge.buf = CreateCsBuffer(cap, rhi::kBufferCopySrc | rhi::kBufferCopyDst,
                                rhi::MemoryKind::kReadback);
  if (!g_bridge.buf)
    return false;
  g_bridge.map = g_bridge.buf->mapped();
  g_bridge.cap = cap;
  return true;
}

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
  // Layers sit one stored slice apart in the linear staging layout.
  const u64 layer_bytes = static_cast<u64>(level.pitch) * level.stored_height *
                          res.stage_elem_bytes;
  if (res.layers > 1 && (level.size != layer_bytes * res.layers ||
                         level.offset + level.size > e.cap || plan.unpack ||
                         plan.widen || img.is_stencil))
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
  if (truth && convert)
    return false;
  // A stencil truth goes through the tiling shader's packed form: one byte
  // per texel on the linear side, which is what the stencil aspect copies.
  const bool stencil_truth = truth && img.is_stencil;
  // Only the tiling shader converts float depth to UNORM16.
  if (plan.depth16 && !truth)
    return false;
  gcn::TextureLayout32 t_tiled, t_linear, s_linear;
  const TileTable* table = nullptr;
  if (truth && !BuildCsImageLayouts(res, t_tiled, t_linear))
    return false;
  if (stencil_truth &&
      (res.layers > 1 ||
       !gcn::BuildTextureLayout32(s_linear, res.width, res.height, res.pitch, 1,
                                  1, 8, res.pow2_pad, 1) ||
       !(table = GetTileTable(t_tiled, s_linear, /*packed=*/true))))
    return false;
  if (truth && !stencil_truth && !(table = GetTileTable(t_tiled, t_linear)))
    return false;
  if (truth && !CsRangeEnsureBuffer(g_truth_bridge_scratch, res.size))
    return false;
  rhi::Buffer* const copy_buf = convert ? g_bridge.buf
                                : truth ? g_truth_bridge_scratch.buf
                                        : e.buf;
  auto* scratch = static_cast<u32*>(g_bridge.map);
  if (convert && to_image) {
    gcn::DetileParallelRows(plan.h, [&](u32 y0, u32 y1) {
      for (u32 y = y0; y < y1; y++) {
        const u8* row =
            static_cast<const u8*>(e.map) + level.offset +
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
    });
  }
  // Host-zero any padding an image->buffer copy does not cover (host writes
  // are made available by the submission).
  if (!to_image && (convert || level.offset != 0 || copy_bytes < res.size))
    std::memset(e.map, 0, res.size);
  if (img.is_stencil && !truth) {
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

  base::String where;
  base::FormatTo(where, "bridge base={:#x}", (unsigned long long)res.base);
  if (!QueueCheck(where.c_str())) {
    BASE_LOGI("gpuvk", "cs {} bridge copy failed: queue check (base={:#x})",
              to_image ? "upload" : "staging", (unsigned long long)res.base);
    g_cs_failed = true;
    Device().ReportDeviceLoss();
    return false;
  }
  rhi::CommandList* c = BeginImmediate();
  if (!c)
    return false;
  // A frame chunk submitted just before this may still read the buffer.
  c->Barrier(kAccessAll, kAccessAll);
  // The dispatch result already lives in VRAM. An ordinary buffer->image copy
  // must read it there, without uploading the identical host readback again.
  // Stencil packing changes the host bytes, and image->buffer staging may have
  // zeroed padding, so those directions still need the host preparation copy.
  // Format conversion uses the separate scratch buffer.
  if (!truth && !convert && (!to_image || img.is_stencil))
    RecordStagingCopy(c, e, e.cap, /*to_device=*/true);
  if (truth && to_image)
    RecordImageTiling(c, *table, e.buf, 0, e.guest_bytes, copy_buf, res.size,
                      /*detile=*/true, 1, plan.layers, plan.depth16);
  if (truth && !to_image &&
      (plan.w < res.width || plan.h < res.height || plan.layers < res.layers ||
       stencil_truth)) {
    // The retile below covers the whole mip; texels the copy leaves alone
    // must not carry a previous conversion's bytes into the truth.
    if (stencil_truth)
      c->FillBuffer(copy_buf, 0, (s_linear.size + 3) & ~u64(3), 0);
    else
      c->FillBuffer(copy_buf, level.offset,
                    layer_bytes * base::Max(res.layers, 1u), 0);
    c->Barrier(rhi::kAccessCopyWrite, rhi::kAccessCopyWrite);
  }
  // Chain from, and restore, the SUBMITTED layout: this copy executes
  // before the current frame's still-recording barriers, whose layout chain
  // must stay intact.
  const rhi::TextureState transfer =
      to_image ? rhi::TextureState::kCopyDst : rhi::TextureState::kCopySrc;
  if (to_image) {
    // The buffer was last written by the dispatch (already waited) or the
    // host; make those writes available to the transfer.
    c->Barrier(rhi::kAccessComputeWrite | rhi::kAccessHostWrite,
               rhi::kAccessCopyRead);
  }
  AliasedImageBarrier(c, img, img.submitted, transfer);
  rhi::BufferTextureCopy copy;
  copy.buffer_offset = (img.is_stencil || convert) ? 0 : level.offset;
  copy.row_length = level.pitch;
  copy.image_height = level.stored_height;
  if (stencil_truth) {
    copy.buffer_offset = s_linear.mips[0].offset;
    copy.row_length = s_linear.mips[0].pitch;
    copy.image_height = s_linear.mips[0].stored_height;
  }
  copy.region.aspect = img.aspect;
  copy.region.layers = plan.layers;
  copy.region.width = plan.w;
  copy.region.height = plan.h;
  if (to_image)
    c->CopyBufferToTexture(img.texture, copy_buf, &copy, 1);
  else
    c->CopyTextureToBuffer(copy_buf, img.texture, &copy, 1);
  if (truth && !to_image) {
    RecordImageTiling(c, *table, e.buf, 0, e.guest_bytes, copy_buf, res.size,
                      /*detile=*/false, 1, base::Max(res.layers, 1u),
                      plan.depth16);
    e.mirror_current = false;
  }
  AliasedImageBarrier(c, img, transfer, img.submitted);
  if (!to_image)
    c->Barrier(rhi::kAccessCopyWrite,
               rhi::kAccessComputeRead | rhi::kAccessHostRead);
  const bool ok = EndImmediate(c, "imm.aliased_copy");
  g_out_rt_submits++;
  if (!ok) {
    // A device loss is not a "shape mismatch": the queue is dead and every
    // fallback the caller would keep recording with is already doomed. Latch
    // the failure so the caller stops instead of building more work on a
    // lost device.
    BASE_LOGI("gpuvk", "cs {} bridge copy failed (base={:#x})",
              to_image ? "upload" : "staging", (unsigned long long)res.base);
    g_cs_failed = true;
    Device().ReportDeviceLoss();
    return false;
  }
  if (trace::Recording())
    trace::RecordBridge(to_image ? "upload" : "stage", res.base,
                        reinterpret_cast<u64>(img.texture), plan.w, plan.h);
  if (img.is_stencil && !truth) {
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
            std::memcpy(
                &expanded,
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
              : plan.widen                                ? "widen"
              : plan.depth16                              ? "depth16"
              : t.layout == rhi::TextureState::kUndefined ? "layout-undef"
              : (e.truth && plan.unpack)                  ? "truth-unpack"
              : (e.truth && img.is_stencil)               ? "truth-stencil"
                                                          : "layers";
  if (!kCsInFrameBridge || !e.buf || plan.widen || plan.depth16 ||
      t.layout == rhi::TextureState::kUndefined ||
      (e.truth && (plan.unpack || img.is_stencil)) ||
      (layers == 1 ? plan.layers != 1 || img.layers != 1
                   : plan.layers > layers || plan.unpack || img.is_stencil))
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
  // A staged stencil holds a dword per texel: the plane is copied out packed,
  // one byte a texel, and the tiling shader widens it into the staging, which
  // the host path did on the CPU behind a wait for the whole queue.
  const bool widen_stencil = img.is_stencil && !e.truth;
  gcn::TextureLayout32 packed;
  if (widen_stencil &&
      (res.elem_bytes != 1 || res.stage_elem_bytes != 4 ||
       !gcn::BuildTextureLayout32(packed, res.width, res.height, res.pitch, 1,
                                  1, 8, res.pow2_pad, 1) ||
       !(table = GetTileTable(packed, linear)) ||
       !(linear_buf = AcquireScratch(packed.size)) || res.size > e.cap))
    return false;
  if (e.truth) {
    if (!(table = GetTileTable(tiled, linear)) ||
        !(linear_buf = AcquireScratch(res.size)))
      return false;
  } else if (!widen_stencil &&
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
  if (widen_stencil) {
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
  if (widen_stencil) {
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
  if (widen_stencil)
    RecordImageTiling(c, *table, linear_buf, 0, packed.size, e.buf, res.size,
                      /*detile=*/true, 1, 1);
  else if (table)
    RecordImageTiling(c, *table, e.buf, 0, e.guest_bytes, linear_buf, res.size,
                      /*detile=*/false, 1, layers);
  AliasedImageBarrier(c, img, kSrc, t.layout);
  c->Barrier(rhi::kAccessCopyWrite, kAccessAll);
  e.frame_ref = g_frame.num;
  e.mirror_current = false;
  e.readback_pending = false;
  g_in_frame_bridge_n++;
  return CsSplitFrameChunk();
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

// The host-side bridge: the buffer must be idle.
bool StageCsRangeFromRt(const ComputeInfo::Res& res, CsRange& e) {
  DELTA_ZONE("gpu.cs_stage_from_rt");
  // The draws that produced it this frame must be on the queue before the
  // copy is, or the dispatch reads last frame's pixels.
  if (RenderedInOpenChunk(res.base) && !CsSplitFrameChunk())
    return false;
  CsAliasedImage img;
  if (!FindCsAliasedImage(res.base, img, /*for_write=*/false,
                          /*prefer_depth=*/res.dfmt == 4))
    return false;
  AliasedCopyPlan plan;
  if (!PlanAliasedCopy(img, res, "reads", plan)) {
    g_rt_unbridgeable.insert(RtShapeKey(res));
    return false;
  }
  return RunAliasedCopy(img, res, e, /*to_image=*/false, plan);
}

// The reverse: a CS result written to a range that a live render/depth target
// aliases must also land in the image, because draws sample the image,
// never guest memory (SotC's exposure/bloom compute writes the adapted scene
// into an RT the tonemap then samples; its depth downsample writes the
// half-res depth pyramid). e.buf already holds the linear pixel data the
// dispatch produced, so upload straight from it. Guest memory was refreshed
// by the caller either way; a shape mismatch just leaves the image stale.
bool UploadCsRangeToRt(u64 base, CsRange& e) {
  CsAliasedImage img;
  if (!e.image_staging)
    return true;
  // The frame list already copied this revision into the target
  // (CsRefreshRtInFrame); uploading it again would paint over whatever was
  // rendered into it since.
  if (e.rt_seq == e.write_seq)
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
  if (!RunAliasedCopy(img, e.res, e, /*to_image=*/true, plan))
    return false;
  if (!img.is_depth) {
    RTarget& rt = g_rts[base];
    rt.ever_rendered = true;  // CS content is real content
    rt.last_frame = g_frame.num;
    // The copy chained from and restored the anchor, so that is the layout
    // on the GPU now whether or not a frame end had stamped it.
    if (rt.submitted_layout == rhi::TextureState::kUndefined)
      rt.submitted_layout = img.submitted;
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
