/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/render_target.h"
#include "gpu/gpu_perf.h"
#include "base/arch.h"

#include "gpu/render/renderer.h"
#include "gpu/render/compute.h"
#include "gpu/render/labels.h"
#include "gpu/render/device.h"
#include "gpu/guest_memory.h"
#include "gpu/render/guest_format.h"
#include "gpu/render/frame.h"
#include "gpu/render/texture_cache.h"
#include "gpu/render/trace.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <utl/options.h>
#include <base/algorithm.h>
#include <base/atomic.h>
#include <base/containers/map.h>
#include <base/containers/vector.h>
#include <base/math/value_bounds.h>
#include <base/memory/move.h>
#include <base/containers/hash_map.h>

namespace {
// DELTA_GPU_CLEARTRACE=<n>: print up to n clear-opening lines. A small cap only
// ever shows the loading screens; a lazy clear that lands on the wrong draw in
// a steady-state frame needs a cap large enough to reach that frame.
DELTA_OPTION(u32, kMaxRenderTargets, "DELTA_GPU_RT_MAX", 256);
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
// targets rendered in the same frame TIE, broken by page-table insertion
// order, i.e. by the guest addresses of the run. That is a per-run coin flip
// deciding which image a pass reads.
DELTA_OPTION(bool, kResolveTrace, "DELTA_GPU_RESOLVETRACE", false);
// DELTA_GPU_NOVARIANT=1: never swap in a parked geometry variant. One base
// rendered at two geometries owns two images, and which one answers to the
// address depends on draw order, so an artefact that alternates frame to
// frame has to be tested against variant switching before anything else.
DELTA_OPTION(bool, kNoVariant, "DELTA_GPU_NOVARIANT", false);
DELTA_OPTION(u64, kDepthResolveTrace, "DELTA_GPU_DEPTHRESOLVE", 0);
DELTA_OPTION(u64, kFrameClearRt, "DELTA_GPU_FRAMECLEAR", 0);
}  // namespace

namespace gpu::render {

using render::DrawInfo;

namespace {

bool IdentitySwizzle(const rhi::Swizzle (&sw)[4]) {
  for (rhi::Swizzle c : sw)
    if (c != rhi::Swizzle::kIdentity)
      return false;
  return true;
}

}  // namespace

// Transition `texture` from `current` to `state` and record the new state.
// `layers` covers every layer the barrier must reach.
void TransitionImage(rhi::CommandList* list,
                     rhi::Texture* texture,
                     rhi::TextureState& current,
                     rhi::TextureState state,
                     u8 aspect,
                     u32 layers) {
  // A depth or stencil plane that is sampled is in the read-only depth state.
  if (aspect != rhi::kAspectColor && state == rhi::TextureState::kShaderRead)
    state = rhi::TextureState::kDepthRead;
  rhi::TextureBarrier b;
  b.texture = texture;
  b.before = current;
  b.after = state;
  b.range.aspect = aspect;
  b.range.layers = layers;
  list->Barrier(0, 0, &b, 1);
  if (trace::Recording())
    trace::RecordBarrier(aspect == rhi::kAspectColor     ? "color"
                         : aspect == rhi::kAspectStencil ? "stencil"
                                                         : "depth",
                         reinterpret_cast<u64>(texture), b.before, state);
  current = state;
}

rhi::TextureView* SampledImageView(rhi::Texture* texture,
                                   rhi::TextureView* identity,
                                   rhi::Format format,
                                   u8 aspect,
                                   u32 swizzle,
                                   base::HashMap<u32, rhi::TextureView*>& views,
                                   rhi::ViewDim dim = rhi::ViewDim::k2D) {
  rhi::TextureViewDesc desc;
  TextureSwizzle(swizzle, desc.swizzle);
  if (IdentitySwizzle(desc.swizzle))
    return identity;
  const auto it = views.find(swizzle);
  if (it != views.end())
    return it->second;
  desc.dim = dim;
  desc.format = (format);
  desc.aspect = aspect;
  rhi::TextureView* view = Device().CreateView(texture, desc);
  if (!view)
    return nullptr;
  views.emplace(swizzle, view);
  return view;
}

// Sampled view of a colour target in `want` rather than the target's own
// format, when the two are the same size (so the reinterpretation is legal and
// the bytes line up). Falls back to the target's format when they are not.
rhi::TextureView* SampledViewAs(RTarget& rt, u32 swizzle, rhi::Format want,
                                rhi::Format* used) {
  const auto own = [&](rhi::TextureView* v) {
    if (used)
      *used = rt.fmt;
    return v;
  };
  // Equal size is not the whole rule: a mutable-format view may only take
  // another format of the same COMPATIBILITY CLASS, and a block-compressed one
  // shares no class with the uncompressed format every render target has. A
  // T# naming BC1/BC5 over a target is a resolution accident, and asking for
  // that view fails outright, leaving the binding on the white default
  // instead of the target's own content.
  if (want == rhi::Format::kUndefined || want == rt.fmt ||
      FormatBytes(want) != FormatBytes(rt.fmt) || FormatBlockCompressed(want))
    return own(SampledView(rt, swizzle));
  if (used)
    *used = want;
  const u32 key = swizzle | (static_cast<u32>(want) << 16);
  const auto it = rt.alias_views.find(key);
  if (it != rt.alias_views.end())
    return it->second;
  rhi::TextureViewDesc desc;
  desc.dim = rt.depth > 1 ? rhi::ViewDim::k3D : rhi::ViewDim::k2D;
  desc.format = (want);
  TextureSwizzle(swizzle, desc.swizzle);
  rhi::TextureView* v = Device().CreateView(rt.texture, desc);
  if (!v)
    return own(SampledView(rt, swizzle));
  rt.alias_views[key] = v;
  return v;
}

rhi::TextureView* SampledView(RTarget& rt, u32 swizzle, bool feedback) {
  // A target that became live at this address after the draw took its snapshot
  // (an alias switch, see ActivateRtVariant) has no copy of its own yet.
  // Sampling the attachment instead would be the feedback loop the copy exists
  // to avoid, so leave the caller its default texture.
  if (feedback && !rt.feedback_texture)
    return nullptr;
  if (feedback)
    return SampledImageView(rt.feedback_texture, rt.feedback_view, rt.fmt,
                            rhi::kAspectColor, swizzle,
                            rt.feedback_sampled_views);
  if (rt.depth > 1)
    return SampledImageView(rt.texture, rt.volume_view, rt.fmt,
                            rhi::kAspectColor, swizzle, rt.sampled_views,
                            rhi::ViewDim::k3D);
  return SampledImageView(rt.texture, rt.view, rt.fmt, rhi::kAspectColor,
                          swizzle, rt.sampled_views);
}

rhi::TextureView* SampledView(DepthTarget& depth, u32 swizzle) {
  return SampledImageView(depth.texture, depth.view, kDepthFormat,
                          rhi::kAspectDepth, swizzle, depth.sampled_views);
}

u64 RtByteSizeWH(u32 w, u32 h, rhi::Format fmt) {
  return (u64)w * h * FormatBytes(fmt);
}

// Register an RT's footprint pages so the page table can find it by overlap.
void RegisterRtPages(u64 base, u32 w, u32 h, rhi::Format fmt) {
  u64 lo = base >> kRtPageShift;
  u64 hi = (base + RtByteSizeWH(w, h, fmt) - 1) >> kRtPageShift;
  for (u64 p = lo; p <= hi; p++) {
    auto& v = g_rt_pages[p];
    bool seen = false;
    for (u64 b : v)
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
// A freshly created image's contents are UNDEFINED. That is invisible for a
// target whose first use is a clear or a full-screen draw, but not for one a
// pass READS before it writes: P.T.'s light buffer is consumed by a feedback
// pass (ps=0x80b54b4000) that un-premultiplies it, out.rgb = rgb / (1 - alpha),
// which can only ever INCREASE a texel. Fed undefined memory on its first
// frame, whatever garbage the allocation held is amplified every frame until it
// saturates fp16 and can never recover; the frame ends up with texels pinned at
// 65504 that then bloom into blown-white blobs. Guest memory a title renders
// into holds defined values; an undefined image is our artefact, so define it.
void ClearNewRt(RTarget& t) {
  rhi::CommandList* list = BeginImmediate();
  if (!list)
    return;
  rhi::TextureState layout = rhi::TextureState::kUndefined;
  TransitionImage(list, t.texture, layout, rhi::TextureState::kCopyDst);
  list->ClearTexture(t.texture, rhi::TextureState::kCopyDst, {},
                     rhi::ClearColor{});
  TransitionImage(list, t.texture, layout, rhi::TextureState::kShaderRead);
  if (EndImmediate(list)) {
    // Submitted and waited, so this layout is the real one on both timelines.
    t.layout = layout;
    t.submitted_layout = layout;
  }
}

bool CreateRtImage(RTarget& t,
                   u64 base,
                   u32 w,
                   u32 h,
                   rhi::Format fmt,
                   u32 depth = 1) {
  if (!w || !h || !depth)
    return false;
  // Robustness: reject render targets with implausible dimensions or an
  // undefined format. A garbage CB_COLOR base/scissor (e.g. a stray shader-pool
  // RT address on the PS5 path) would otherwise feed the driver an invalid
  // image and hard-crash it.
  if (w > 8192 || h > 8192 || fmt == rhi::Format::kUndefined) {
    BASE_LOGI("gpuvk", "skip bad RT {:#x} {}x{} fmt={}", (unsigned long)base,
              w, h, (int)fmt);
    return false;
  }
  t.w = w;
  t.h = h;
  t.depth = depth;
  t.fmt = fmt;
  rhi::TextureDesc td;
  td.dim = depth > 1 ? rhi::TextureDim::k3D : rhi::TextureDim::k2D;
  td.format = (fmt);
  td.width = w;
  td.height = h;
  td.depth = depth;
  td.usage = rhi::kTextureColorTarget | rhi::kTextureSampled |
             rhi::kTextureCopySrc | rhi::kTextureCopyDst;
  // An SRGB format cannot be a storage image, and asking for it makes every
  // view of this image invalid. A compute shader that wants to write this
  // surface reaches it through the linear alias anyway, which is what the
  // mutable format is here for.
  if (fmt != rhi::Format::kRGBA8Srgb && fmt != rhi::Format::kBGRA8Srgb)
    td.usage |= rhi::kTextureStorage;
  // The colour format a pass RENDERS with and the format a later shader SAMPLES
  // the same memory with are independent on PS4: a G-buffer plane written as
  // UINT is read back through a T# that may name a different numeric type, and
  // the view's numeric type has to match the shader's sampled type. A mutable
  // format lets SampledViewAs() hand out a view in the format the descriptor
  // asked for instead of the one the attachment was created with.
  td.usage |= rhi::kTextureMutableFormat;
  // A layered pass renders the slices of a volume through a 2D-array view.
  if (depth > 1)
    td.usage |= rhi::kTextureArrayCompatible;
  t.texture = Device().CreateTexture(td);
  if (!t.texture)
    return false;
  rhi::TextureViewDesc vd;
  vd.dim = depth > 1 ? rhi::ViewDim::k2DArray : rhi::ViewDim::k2D;
  vd.layers = depth;
  t.view = Device().CreateView(t.texture, vd);
  if (!t.view) {
    Device().Destroy(t.texture);
    t.texture = nullptr;
    return false;
  }
  if (depth > 1) {
    vd.dim = rhi::ViewDim::k3D;
    vd.layers = 1;
    t.volume_view = Device().CreateView(t.texture, vd);
  }
  // A group so this RT can be sampled (render-to-texture); a 2D-array view of
  // a volume is an attachment only.
  if (g_tex.descriptors_ready && depth == 1)
    t.set = SampledTextureGroup(t.view, g_tex.sampler);
  ClearNewRt(t);
  BASE_LOGI("gpuvk", "new RT {:#x} {}x{}x{} fmt={}", (unsigned long)base, w,
            h, depth, (int)fmt);
  if (Device().caps().debug_labels) {
    char name[64];
    std::snprintf(name, sizeof(name), "rt %#lx %ux%u fmt=%d",
                  (unsigned long)base, w, h, (int)fmt);
    Device().SetName(t.texture, name);
  }
  return true;
}

// The images of a base the guest renders to at more than one geometry, minus
// the one that is live in g_rts. P.T. renders a fullscreen pass and then a
// 512x512 pass to the same address inside one frame; sharing one image lands
// the small pass in a corner of the big one and every later sample of that
// address reads mostly stale pixels. Every lookup in the backend names a target
// by address alone, so g_rts keeps holding the live target and the other
// geometries wait here until a draw asks for them again.
base::HashMap<u64, base::Vector<RTarget>> g_rt_variants;
// Budgeted by MEMORY, not by count. A flat cap of three is generous for an
// address aliased at 1920x1080 and hopeless for one aliased at 48x48, and
// Gameface renders every glyph into the same scratch address at the glyph's own
// size, a dozen geometries and counting. 32 MiB still leaves the big case at
// three images; the count is only a backstop against an address that cycles
// forever, so it has to sit well above any real font (GTA:SA's four glyph
// targets each want more than 32 geometries and together they come to 201 KiB.
constexpr u64 kMaxRtVariantBytes = 32ull << 20;
constexpr size_t kMaxRtVariants = 256;

base::Vector<RTarget> g_retired_rts;

void RetireRt(RTarget& t) {
  g_retired_rts.push_back(base::move(t));
  // Cached multi-texture groups may name its views.
  ClearMultiTexCache();
}

void DestroyRt(RTarget& t) {
  rhi::Device& device = Device();
  for (auto* views : {&t.sampled_views, &t.alias_views,
                      &t.feedback_sampled_views})
    for (auto& [key, view] : *views)
      device.Destroy(view);
  for (rhi::Object* o :
       std::initializer_list<rhi::Object*>{t.set, t.feedback_set, t.view,
                                            t.volume_view, t.feedback_view})
    device.Destroy(o);
  device.Destroy(t.feedback_texture);
  device.Destroy(t.texture);
}

// Make the image of geometry (w, h, fmt) the live target at `base`, creating it
// on first use.
RTarget* ActivateRtVariant(RTarget& live,
                           u64 base,
                           u32 w,
                           u32 h,
                           rhi::Format fmt,
                           u32 depth = 1) {
  if (kNoVariant)
    return &live;
  auto& parked = g_rt_variants[base];
  RTarget* alt = nullptr;
  for (RTarget& v : parked)
    if (v.w == w && v.h == h && v.fmt == fmt && v.depth == depth) {
      alt = &v;
      break;
    }
  if (!alt) {
    u64 parked_bytes = 0;
    for (const RTarget& v : parked)
      parked_bytes += RtByteSizeWH(v.w, v.h, v.fmt);
    // Make room by retiring the variants rendered longest ago. Gameface
    // renders every glyph into one scratch address at the glyph's own size,
    // so without this the address fills up within a minute and every later
    // glyph is declined.
    while (parked.size() >= kMaxRtVariants ||
           parked_bytes + RtByteSizeWH(w, h, fmt) > kMaxRtVariantBytes) {
      size_t oldest = parked.size();
      for (size_t i = 0; i < parked.size(); i++)
        if (parked[i].last_frame < g_frame.num &&
            (oldest == parked.size() ||
             parked[i].last_frame < parked[oldest].last_frame))
          oldest = i;
      if (oldest == parked.size())
        break;
      parked_bytes -=
          RtByteSizeWH(parked[oldest].w, parked[oldest].h, parked[oldest].fmt);
      RetireRt(parked[oldest]);
      parked.erase(parked.begin() + oldest);
    }
    if (parked.size() >= kMaxRtVariants ||
        parked_bytes + RtByteSizeWH(w, h, fmt) > kMaxRtVariantBytes) {
      // Never hand back the live target at ITS geometry: the caller opens a
      // render area of w x h over it, and if that is bigger than the image the
      // region is invalid outright (VUID-VkRenderingInfo-pNext-06079/06080) and
      // the draw writes outside the attachment. Declining loses the draw, which
      // is at least a defined outcome.
      static int n = 0;
      if (n++ < 8)
        BASE_LOGI("gpuvk",
                  "RT {:#x}: {} variants ({} KiB) and no room for {}x{} fmt={} "
                  "-- declining the draw",
                  (unsigned long)base, parked.size(),
                  (unsigned long)(parked_bytes >> 10), w, h, (int)fmt);
      return nullptr;
    }
    RTarget t;
    if (!CreateRtImage(t, base, w, h, fmt, depth))
      return nullptr;
    BASE_LOGI("gpuvk",
              "RT alias {:#x}: have {}x{} fmt={}, requested {}x{} fmt={} -> "
              "own image",
              (unsigned long)base, live.w, live.h, (int)live.fmt, w, h,
              (int)fmt);
    parked.push_back(t);
    alt = &parked.back();
    RegisterRtPages(base, w, h, fmt);
  }
  base::Swap(live, *alt);
  if (trace::Recording())
    trace::RecordVariantSwap(base, reinterpret_cast<u64>(alt->texture), alt->w,
                             alt->h, reinterpret_cast<u64>(live.texture),
                             live.w, live.h);
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
  // The parked target keeps its submitted-layout stamp: EndFrame stamps
  // parked variants as well as live ones, so the stamp is the layout the GPU
  // holds whatever this frame has recorded since. Resetting it to UNDEFINED
  // here made every variant re-activated mid-frame unreadable to the compute
  // bridge until the next frame end, and Astro Bot's upscaler read its
  // 1080p input from guest memory (zeros) every frame because of it.
  return &live;
}

bool ActivateVolumeRt(u64 base, u32 w, u32 h, u32 depth) {
  const auto it = g_rts.find(base);
  if (depth <= 1 || it == g_rts.end())
    return false;
  RTarget& live = it->second;
  const auto matches = [&](const RTarget& t) {
    return t.w == w && t.h == h && t.depth == depth;
  };
  if (!matches(live)) {
    const auto parked = g_rt_variants.find(base);
    if (parked == g_rt_variants.end())
      return false;
    const auto alt = base::FindIf(parked->second.begin(), parked->second.end(),
                                  matches);
    if (alt == parked->second.end() ||
        !ActivateRtVariant(live, base, w, h, alt->fmt, depth))
      return false;
  }
  return live.ever_rendered;
}

// The extent an attachment's IMAGE needs, given the target's own surface
// geometry (CB_COLORn_PITCH/SLICE) and the region this pass draws into.
// The drawn region normally IS the target, padded: a 1920x1080 surface reports
// 1920x1088 because the slice is tile-aligned, and sizing the image from that
// would make every scanout eight rows taller than the display. So the surface
// only wins when the drawn region cannot be it even after rounding up a full
// tile, which is the case the screen scissor gets wrong (GTA:SA binding its
// 1792x960 G-buffer under a 256x256 scissor, losing 164 draws a frame into a
// 256x256 image).
u32 RtSurfaceExtent(u32 surface, u32 drawn, u32 tile) {
  const u32 padded = (drawn + tile - 1) & ~(tile - 1);
  return (surface > padded && surface <= 8192) ? surface : drawn;
}

// Find or create the render target at guest address `base` (dimensions w x h).
RTarget* GetRT(u64 base, u32 w, u32 h, rhi::Format fmt, u32 depth) {
  auto it = g_rts.find(base);
  if (it != g_rts.end()) {
    RTarget& live = it->second;
    if (live.w != w || live.h != h || live.fmt != fmt || live.depth != depth)
      return ActivateRtVariant(live, base, w, h, fmt, depth);
    return &live;
  }
  // A cap that a title exceeds does not degrade, it deletes: the target is
  // never created, every draw into it is dropped, and a later pass that SAMPLES
  // it resolves by overlap to whatever else happens to cover the address.
  // GTA:SA's world hit 64 during its level load, lost the shadow buffer at
  // 0x202b800000, and its lighting then sampled a shadow atlas that merely
  // overlapped, which is what made the scene black. Kept as a backstop
  // against a title that cycles addresses forever, not as a budget.
  if (g_rts.size() >= kMaxRenderTargets) {
    static int n = 0;
    if (n++ < 4)
      BASE_LOGI("gpuvk",
                "RT table full ({}), dropping {:#x} {}x{} fmt={} and every "
                "draw that targets it",
                (u32)kMaxRenderTargets, (unsigned long)base, w, h, (int)fmt);
    return nullptr;
  }
  RTarget t;
  if (!CreateRtImage(t, base, w, h, fmt, depth)) {
    static int n = 0;
    if (n++ < 8)
      BASE_LOGI("gpuvk", "RT image create FAILED {:#x} {}x{} fmt={}",
                (unsigned long)base, w, h, (int)fmt);
    return nullptr;
  }
  g_rts[base] = t;
  if (!g_region.first_rt)
    g_region.first_rt = base;
  RegisterRtPages(base, w, h, fmt);  // resource-model page table
  return &g_rts[base];
}

rhi::BindGroup* SnapshotRT(RTarget& rt) {
  if (!rt.feedback_texture) {
    rhi::TextureDesc td;
    td.format = (rt.fmt);
    td.width = rt.w;
    td.height = rt.h;
    td.usage = rhi::kTextureSampled | rhi::kTextureCopyDst;
    td.name = "rt feedback";
    rt.feedback_texture = Device().CreateTexture(td);
    if (!rt.feedback_texture)
      return nullptr;
    rt.feedback_view = Device().CreateView(rt.feedback_texture, {});
    rt.feedback_set = rt.feedback_view
                          ? SampledTextureGroup(rt.feedback_view, g_tex.sampler)
                          : nullptr;
    if (!rt.feedback_set) {
      Device().Destroy(rt.feedback_view);
      Device().Destroy(rt.feedback_texture);
      rt.feedback_view = nullptr;
      rt.feedback_texture = nullptr;
      return nullptr;
    }
  }
  rhi::CommandList* list = g_frame.list;
  TransitionImage(list, rt.texture, rt.layout, rhi::TextureState::kCopySrc);
  TransitionImage(list, rt.feedback_texture, rt.feedback_layout,
             rhi::TextureState::kCopyDst);
  rhi::TextureRegion region;
  region.width = rt.w;
  region.height = rt.h;
  list->CopyTexture(rt.feedback_texture, region, rt.texture, region);
  TransitionImage(list, rt.feedback_texture, rt.feedback_layout,
             rhi::TextureState::kShaderRead);
  return rt.feedback_set;
}

u64 RtByteSize(const RTarget& rt) {
  return RtByteSizeWH(rt.w, rt.h, rt.fmt);
}
// Find or create the depth target at guest address `base` (dimensions w x h).
// The images of a depth base the guest renders to at more than one geometry,
// minus the one live in g_depths, the same arrangement the colour targets
// use (see g_rt_variants). Without it, the first geometry to touch a base
// fixed it forever and every later pass at another size rendered into a corner
// of that image: P.T.'s 1920x1080 depth buffer had a 960x540 clear in its
// top-left quadrant and nothing anywhere else, so every pass that sampled the
// depth (SSAO first, then the whole post chain) read zero.
base::HashMap<u64, base::Vector<DepthTarget>> g_depth_variants;
constexpr size_t kMaxDepthVariants = 3;

bool CreateDepthImage(DepthTarget& t,
                      u64 base,
                      u32 w,
                      u32 h,
                      u64 stencil_base,
                      u32 layers = 1) {
  if (!w || !h || !layers)
    return false;
  t.w = w;
  t.h = h;
  t.layers = layers;
  t.stencil_base = stencil_base;
  rhi::TextureDesc td;
  td.format = (kDepthFormat);
  td.width = w;
  td.height = h;
  td.layers = layers;
  // Copy src/dst: the compute path bridges CS reads/writes of a live depth
  // target through image<->buffer copies (see vk_compute.cc).
  td.usage = rhi::kTextureDepthTarget | rhi::kTextureSampled |
             rhi::kTextureCopySrc | rhi::kTextureCopyDst;
  t.texture = Device().CreateTexture(td);
  if (!t.texture)
    return false;
  rhi::TextureViewDesc vd;
  vd.aspect = rhi::kAspectDepth;
  t.view = Device().CreateView(t.texture, vd);
  vd.aspect = rhi::kAspectDepth | rhi::kAspectStencil;
  t.attachment_view = t.view ? Device().CreateView(t.texture, vd) : nullptr;
  if (!t.attachment_view) {
    Device().Destroy(t.view);
    Device().Destroy(t.texture);
    t = {};
    return false;
  }
  if (g_tex.descriptors_ready)
    t.set = SampledTextureGroup(t.view, g_tex.sampler,
                                rhi::TextureState::kDepthRead);
  BASE_LOGI("gpuvk", "new depth {:#x} {}x{}x{}", (unsigned long)base, w, h,
            layers);
  if (Device().caps().debug_labels) {
    char name[64];
    std::snprintf(name, sizeof(name), "depth %#lx %ux%u", (unsigned long)base,
                  w, h);
    Device().SetName(t.texture, name);
  }
  return true;
}

// Make the image of geometry (w, h) the live depth target at `base`, creating
// it on first use. Mirrors ActivateRtVariant.
DELTA_OPTION(bool, kDepthVariants, "DELTA_GPU_DEPTHVARIANTS", true);

DepthTarget* ActivateDepthVariant(DepthTarget& live,
                                  u64 base,
                                  u32 w,
                                  u32 h,
                                  u64 stencil_base,
                                  u32 layers = 1) {
  // A depth attachment must COVER the render area: Vulkan lets it be larger,
  // never smaller (VUID-VkRenderingInfo-pNext-06079/06080). Handing the live
  // target back whatever its geometry is how P.T. began a 512x512 region with a
  // 64x64 depth view and LOST THE DEVICE (its environment-probe bake requests
  // one depth base at five geometries, so the three-variant cap is reached and
  // the cap path fires; the guest walks away with the device gone: no EndFrame,
  // no present, black window). Declining the draw (nullptr -> BeginRegion
  // false) is the only safe fallback: a dropped draw costs one pass, a lost
  // device costs the run.
  const auto covers = [&](DepthTarget* t) -> DepthTarget* {
    return (t && t->w >= w && t->h >= h && t->layers >= layers) ? t : nullptr;
  };
  if (!kDepthVariants)
    return covers(&live);
  auto& parked = g_depth_variants[base];
  DepthTarget* alt = nullptr;
  for (DepthTarget& v : parked)
    if (v.w == w && v.h == h && v.layers >= layers) {
      alt = &v;
      break;
    }
  if (!alt) {
    if (parked.size() >= kMaxDepthVariants)
      return covers(&live);
    DepthTarget t;
    // A variant that replaces the live one at the same geometry only because
    // it needs more layers keeps the larger count.
    const u32 want = live.w == w && live.h == h ? base::Max(layers, live.layers)
                                                : layers;
    if (!CreateDepthImage(t, base, w, h, stencil_base, want))
      return covers(&live);
    BASE_LOGI("gpuvk",
              "depth alias {:#x}: have {}x{}, requested {}x{} -> own image",
              (unsigned long)base, live.w, live.h, w, h);
    parked.push_back(t);
    alt = &parked.back();
  }
  base::Swap(live, *alt);
  if (live.last_frame != g_frame.num) {
    live.used_this_frame = false;
    live.stencil_used_this_frame = false;
  }
  return &live;  // see ActivateRtVariant: the parked stamp stays valid
}

// The depth counterpart of ActivateSampledRtVariant: a sample names a depth
// target by address, and the live geometry at that address is whichever pass
// ran last, routinely the small half-resolution one. P.T.'s SSAO samples the
// full-resolution scene depth and was handed the 960x540 variant instead.
bool ActivateSampledDepthVariant(u64 base, u32 w, u32 h) {
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

u64 g_render_serial = 0;
base::Vector<DepthTarget> g_retired_depths;

void RetireDepthTarget(const DepthTarget& t) {
  g_retired_depths.push_back(t);
  // Cached descriptor sets may name its views.
  ClearMultiTexCache();
}

void ReleaseRetiredTargets() {
  // Two BeginFrames of rest, like ReleaseRetiredTextures.
  static base::Vector<RTarget> aged_rts;
  for (RTarget& t : aged_rts)
    DestroyRt(t);
  aged_rts = base::move(g_retired_rts);
  g_retired_rts.clear();
  static base::Vector<DepthTarget> aged;
  for (DepthTarget& t : aged) {
    Device().Destroy(t.set);
    for (rhi::TextureView* v : {t.view, t.stencil_view, t.attachment_view})
      Device().Destroy(v);
    for (rhi::TextureView* v : t.layer_views)
      Device().Destroy(v);
    for (auto& [swizzle, v] : t.sampled_views)
      Device().Destroy(v);
    Device().Destroy(t.texture);
  }
  aged = base::move(g_retired_depths);
  g_retired_depths.clear();
}

// The attachment view of one array layer; layer 0 is the target's own.
rhi::TextureView* DepthLayerView(DepthTarget& t, u32 layer) {
  if (!layer || layer >= t.layers)
    return t.attachment_view;
  if (t.layer_views.size() < t.layers)
    t.layer_views.resize(t.layers, nullptr);
  rhi::TextureView*& view = t.layer_views[layer];
  if (!view) {
    rhi::TextureViewDesc vd;
    vd.aspect = rhi::kAspectDepth | rhi::kAspectStencil;
    vd.base_layer = layer;
    view = Device().CreateView(t.texture, vd);
  }
  return view ? view : t.attachment_view;
}

DepthTarget* GetDepthRT(u64 base,
                        u32 w,
                        u32 h,
                        u64 stencil_base,
                        u32 layers) {
  auto it = g_depths.find(base);
  if (it != g_depths.end()) {
    DepthTarget& live = it->second;
    if (stencil_base)
      live.stencil_base = stencil_base;
    if (!w || !h || (live.w == w && live.h == h && live.layers >= layers))
      return &live;
    // DB_DEPTH_VIEW names the slice being drawn, never the array's size: the
    // array grows in place, and the old image goes once the GPU is done.
    if (live.w == w && live.h == h) {
      u32 want = 4;
      while (want < layers)
        want *= 2;
      DepthTarget grown;
      if (!CreateDepthImage(grown, base, w, h, live.stencil_base, want))
        return nullptr;
      grown.htile_base = live.htile_base;
      grown.guest_w = live.guest_w;
      grown.guest_h = live.guest_h;
      grown.last_frame = live.last_frame;
      grown.used_this_frame = live.used_this_frame;
      // A new image holds nothing: each layer clears on its first bind.
      grown.clear_layers = ~0u;
      grown.clear_value = live.clear_value;
      RetireDepthTarget(live);
      live = grown;
      return &live;
    }
    return ActivateDepthVariant(live, base, w, h, stencil_base, layers);
  }
  if (g_depths.size() >= 32 || !w || !h)
    return nullptr;
  DepthTarget t;
  if (!CreateDepthImage(t, base, w, h, stencil_base, layers))
    return nullptr;
  g_depths[base] = t;
  return &g_depths[base];
}

// Resolve a sampled texture address to the live RT image that backs it (the
// resource model's "the page table collects all overlappers" lookup). The game
// cycles/aliases RT addresses (double-buffered room layers, a pool of scene
// buffers), so a composite often samples an address that does not exactly match
// the RT base it was rendered into. Gather every RT whose footprint touches the
// sampled region's pages and resolve by IDENTITY, not recency: an exact base
// whose size matches wins outright (the guest bound that buffer, so a more
// recently rendered overlapping buffer must never override it); only when no
// exact match exists fall back to the best-fitting overlapper (dimension match,
// then the freshest). Returns the g_rts key, or 0.
// This path also activates a variant, because only the RT-bind path used to:
// P.T. renders its fullscreen composite and a 64x64 pass to the same address,
// and when the small one was live the present blit resolved to nothing, a black
// screen with the whole scene one variant away.
bool ActivateSampledRtVariant(u64 base, u32 w, u32 h) {
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

u64 ResolveSampledRT(u64 addr, u32 w, u32 h) {
  if (!addr)
    return 0;
  u64 req_size = w && h ? (u64)w * h * 4 : 4;
  u64 a0 = addr, a1 = addr + req_size;
  // Exact-identity hit: the guest sampled this exact base and it is a live RT
  // of the requested size. That is unambiguously the right image, so return it
  // before any freshness comparison can pick an overlapping cycled buffer
  // instead.
  auto ex = g_rts.find(addr);
  if (ex != g_rts.end() && ex->second.ever_rendered &&
      ((!w || !h) || (ex->second.w == w && ex->second.h == h)))
    return addr;
  u64 best = 0;
  long best_score = -1;
  u32 candidates = 0, ties = 0;
  u64 tie_with = 0;
  auto consider = [&](u64 b0) {
    auto it = g_rts.find(b0);
    if (it == g_rts.end())
      return;
    const RTarget& rt = it->second;
    if (!rt.ever_rendered)
      return;  // never sample an RT with no content
    u64 b1 = b0 + RtByteSize(rt);
    if (!(a0 < b1 && b0 < a1))
      return;  // no interval overlap
    bool dim_match = (!w || !h) || (rt.w == w && rt.h == h);
    // A target may answer for an address only if it is the geometry being
    // sampled, or if it at least CONTAINS the sampled footprint. Bare interval
    // overlap is not enough: P.T. samples a 2048x1024 compute-written surface
    // whose 8 MB footprint runs across a pool of 480x270 targets, and seven of
    // them tied on freshness, so an arbitrary half-megabyte target answered
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
  for (u64 p = a0 >> kRtPageShift; p <= (a1 - 1) >> kRtPageShift; p++) {
    auto it = g_rt_pages.find(p);
    if (it != g_rt_pages.end())
      for (u64 b0 : it->second)
        consider(b0);
  }
  if (kResolveTrace && ties) {
    static base::Atomic<u64> n{0};
    if (n.fetch_add(1) < 40)
      BASE_LOGI("resolve",
                "AMBIGUOUS {:#x} {}x{}: {} candidates, {} tied at score {}; "
                "chose {:#x} over {:#x}",
                (unsigned long)addr, w, h, candidates, ties, best_score,
                (unsigned long)best, (unsigned long)tie_with);
    else if ((n.load() % 20000) == 0)
      BASE_LOGI("resolve", "{} ambiguous resolves so far",
                (unsigned long long)n.load());
  }
  return best;
}

// Depth images use a separate registry from color RTs. A GFX7 depth surface may
// be sampled through an R32_FLOAT descriptor whose base denotes an overlapping
// view rather than DB_Z_WRITE_BASE, so resolve typed depth aliases by footprint
// too.
u64 ResolveSampledStencil(u64 addr) {
  if (!addr)
    return 0;
  for (const auto& [base, depth] : g_depths)
    if (depth.stencil_base == addr && depth.last_frame > -1000)
      return base;
  return 0;
}

rhi::TextureView* StencilSampledView(DepthTarget& depth) {
  if (!depth.stencil_view) {
    rhi::TextureViewDesc vd;
    vd.aspect = rhi::kAspectStencil;
    depth.stencil_view = Device().CreateView(depth.texture, vd);
  }
  return depth.stencil_view;
}

u64 ResolveSampledDepth(u64 addr, u32 w, u32 h) {
  if (!addr)
    return 0;
  u64 req_size = w && h ? (u64)w * h * 4 : 4;
  u64 a1 = addr + req_size;
  u64 best = 0;
  long best_score = -1;
  for (const auto& [base, depth] : g_depths) {
    if (depth.last_frame <= -1000)
      continue;
    // A base can hold several geometries and only one of them is live, but a
    // sample names the address, not the geometry. Score against every variant:
    // otherwise a base whose small variant happens to be live stops covering
    // an address its full-resolution one does, and the sample falls through to
    // guest memory, which for a depth buffer is empty.
    u64 span = depth.guest_w && depth.guest_h
                        ? (u64)depth.guest_w * depth.guest_h * 4
                        : RtByteSizeWH(depth.w, depth.h, kDepthFormat);
    bool dim_match = (!w || !h) || (depth.w == w && depth.h == h);
    const auto parked = g_depth_variants.find(base);
    if (parked != g_depth_variants.end())
      for (const DepthTarget& v : parked->second) {
        span = base::Max(span, v.guest_w && v.guest_h
                                  ? (u64)v.guest_w * v.guest_h * 4
                                  : RtByteSizeWH(v.w, v.h, kDepthFormat));
        if (w && h && v.w == w && v.h == h)
          dim_match = true;
      }
    u64 b1 = base + span;
    if (!(addr < b1 && base < a1))
      continue;
    // As for colour targets: overlap alone lets a texture that merely borders
    // a depth allocation read the depth image. GTA:SA's material textures sit
    // right below its shadow maps and sampled them as red depth.
    if (!dim_match && !(base <= addr && b1 >= a1))
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
  // to a guest upload of undefined bytes, which reads as depth 0, i.e. the
  // far plane, and nothing downstream looks wrong.
  if (kDepthResolveTrace && addr == (u64)kDepthResolveTrace) {
    static int n = 0;
    if (n++ < 6) {
      base::String depths;
      for (const auto& [base, depth] : g_depths)
        base::FormatTo(depths, " [{:#x} {}x{} guest={}x{} lf={}]",
                       (unsigned long)base, depth.w, depth.h, depth.guest_w,
                       depth.guest_h, depth.last_frame);
      BASE_LOGI("dres", "addr={:#x} {}x{} req={} -> {:#x}; depths:{}",
                (unsigned long)addr, w, h, (unsigned long)req_size,
                (unsigned long)best, depths.c_str());
    }
  }
  return best;
}

void NoteDccWrite(u64 base, u64 bytes, const u32* fill) {
  if (!base || !bytes)
    return;
  const auto note = [&](RTarget& rt) {
    if (!rt.dcc_base || rt.dcc_base < base || rt.dcc_base >= base + bytes)
      return;
    rt.dcc_clear_pending = true;
    rt.dcc_code_known = fill != nullptr;
    rt.dcc_clear_code = fill ? *fill : 0;
  };
  for (auto& kv : g_rts) {
    note(kv.second);
    auto v = g_rt_variants.find(kv.first);
    if (v == g_rt_variants.end())
      continue;
    for (RTarget& alt : v->second)
      note(alt);
  }
  const auto note_depth = [&](DepthTarget& dt) {
    if (!dt.htile_base || dt.htile_base < base || dt.htile_base >= base + bytes)
      return;
    dt.htile_clear_pending = true;
    dt.htile_code_known = fill != nullptr;
    dt.htile_clear_code = fill ? *fill : 0;
  };
  for (auto& kv : g_depths) {
    note_depth(kv.second);
    auto v = g_depth_variants.find(kv.first);
    if (v == g_depth_variants.end())
      continue;
    for (DepthTarget& alt : v->second)
      note_depth(alt);
  }
}

// The depth counterpart of ResolveDccClear. HTILE's low nibble is ZMASK:
// 0 and 13 are the clear states (a title fills 0 or 0xfffffff0), 0xF is an
// expanded tile, the rest are partially compressed tiles no fill produces.
void ResolveHtileClear(DepthTarget& dt, u64 base, float depth_clear) {
  dt.htile_clear_pending = false;
  u32 code = dt.htile_clear_code;
  if (!dt.htile_code_known) {
    render::FlushCsWritesRange(render::DefaultRenderer(), dt.htile_base, 4, "htile");
    if (!gpu::IsReadableRange(dt.htile_base, 4))
      return;
    std::memcpy(&code, reinterpret_cast<const void*>(dt.htile_base), 4);
  }
  const u32 zmask = code & 0xF;
  const bool clear = zmask == 0 || zmask == 13;
  if (clear) {
    dt.clear_pending = true;
    dt.clear_value = depth_clear;
  }
  if (kClearTrace) {
    static int n = 0;
    if (n++ < kClearTrace)
      BASE_LOGI("clear",
                "f{} draw#{} depth {:#x} {}x{} HTILE {:#x} code={:08x} -> {}",
                g_frame.num, g_frame.draws, (unsigned long)base, dt.w, dt.h,
                (unsigned long)dt.htile_base, code,
                clear ? "clear" : "not a clear");
  }
}

void NoteSurfaceWrite(u64 base, u64 bytes) {
  if (!base || !bytes)
    return;
  const auto note = [&](RTarget& rt, u64 rt_base) {
    if (base >= rt_base + RtByteSize(rt) || rt_base >= base + bytes)
      return;
    rt.dcc_clear_pending = false;
    if (rt.clear_pending && rt.clear_src &&
        !std::strcmp(rt.clear_src, "dcc-clear"))
      rt.clear_pending = false;
  };
  for (auto& kv : g_rts) {
    note(kv.second, kv.first);
    auto v = g_rt_variants.find(kv.first);
    if (v == g_rt_variants.end())
      continue;
    for (RTarget& alt : v->second)
      note(alt, kv.first);
  }
}

void NoteRawWrite(u64 base, u64 bytes) {
  if (!base || !bytes)
    return;
  const auto note = [&](RTarget& rt, u64 rt_base) {
    // Rendered this frame: the image is the newer one, and a partial raw
    // write (a region update) must not turn its next bind into a clear.
    if (rt.last_frame == g_frame.num || !rt.ever_rendered)
      return;
    const u64 end = rt_base + RtByteSize(rt);
    if (base < end && rt_base < base + bytes)
      rt.ever_rendered = false;
  };
  for (auto& kv : g_rts) {
    note(kv.second, kv.first);
    auto v = g_rt_variants.find(kv.first);
    if (v == g_rt_variants.end())
      continue;
    for (RTarget& alt : v->second)
      note(alt, kv.first);
  }
}

bool OverlapsLiveTarget(u64 base, u64 bytes) {
  const u64 end = base + bytes;
  const auto hits = [&](u64 at, u64 n) { return at < end && base < at + n; };
  const auto depth_bytes = [](const DepthTarget& d) {
    return u64(d.w) * d.h * 4 * base::Max(1u, d.layers);
  };
  for (const auto& [at, rt] : g_rts)
    if (hits(at, RtByteSize(rt)))
      return true;
  for (const auto& [at, variants] : g_rt_variants)
    for (const RTarget& rt : variants)
      if (hits(at, RtByteSize(rt)))
        return true;
  for (const auto& [at, d] : g_depths)
    if (hits(at, depth_bytes(d)))
      return true;
  for (const auto& [at, variants] : g_depth_variants)
    for (const DepthTarget& d : variants)
      if (hits(at, depth_bytes(d)))
        return true;
  return false;
}

bool ActivateWrittenRtVariant(u64 base, u32 w, u32 h) {
  if (!base || !w || !h)
    return false;
  auto it = g_rts.find(base);
  if (it == g_rts.end())
    return false;
  RTarget& live = it->second;
  if (live.w == w && live.h == h)
    return true;
  auto parked = g_rt_variants.find(base);
  if (parked == g_rt_variants.end())
    return false;
  for (const RTarget& v : parked->second)
    if (v.w == w && v.h == h)
      return ActivateRtVariant(live, base, w, h, v.fmt) != nullptr;
  return false;
}

// The DCC clear codes (gfx8..gfx10): four fixed colours, one that names
// CB_COLORn_CLEAR_WORD0/1, and "uncompressed", which is not a clear.
bool DccClearColor(u32 code,
                   u32 info,
                   const u32* clear_word,
                   rhi::ClearColor& out) {
  switch (code) {
    case 0x00000000:
      out = rhi::ClearColor{{0.f, 0.f, 0.f, 0.f}};
      return true;
    case 0x40404040:
      out = rhi::ClearColor{{0.f, 0.f, 0.f, 1.f}};
      return true;
    case 0x80808080:
      out = rhi::ClearColor{{1.f, 1.f, 1.f, 0.f}};
      return true;
    case 0xC0C0C0C0:
      out = rhi::ClearColor{{1.f, 1.f, 1.f, 1.f}};
      return true;
    case 0x20202020:
      if (!clear_word)
        return false;
      out = ColorTargetClearValue(info, clear_word[0], clear_word[1]);
      return true;
    default:
      return false;
  }
}

// A FAST_CLEAR target reads cleared wherever its CMASK nibble is 0, and the
// eliminate pass that follows rendering writes the clear colour in and marks
// the tiles expanded (0xF). Our targets are never compressed and eliminate is
// a no-op, so do both here: a cleared CMASK at bind is the clear, and marking
// it expanded lets the guest's next clear show up as a write back to 0 however
// it lands, which is not always a packet we see.
void NoteCmaskBind(RTarget& rt) {
  const u64 bytes = base::Max<u64>(4, (u64(rt.w) * rt.h) / 128);
  if (!gpu::IsReadableRangeCached(rt.dcc_base, bytes))
    return;
  render::FlushCsWritesRange(render::DefaultRenderer(), rt.dcc_base, 4, "cmask");
  auto* cmask = reinterpret_cast<u32*>(rt.dcc_base);
  if (*cmask & 0xF)
    return;
  rt.dcc_clear_pending = true;
  rt.dcc_code_known = true;
  rt.dcc_clear_code = *cmask;
  std::memset(cmask, 0xFF, bytes);
}

// A pending DCC write is about to be followed by a draw into the target:
// find out what it wrote and make it the target's clear.
void ResolveDccClear(RTarget& rt, u64 base, u32 info, const u32* clear_word) {
  rt.dcc_clear_pending = false;
  u32 code = rt.dcc_clear_code;
  if (!rt.dcc_code_known) {
    render::FlushCsWritesRange(render::DefaultRenderer(), rt.dcc_base, 4, "cmask");
    if (!gpu::IsReadableRangeCached(rt.dcc_base, 4))
      return;
    std::memcpy(&code, reinterpret_cast<const void*>(rt.dcc_base), 4);
  }
  rhi::ClearColor value{};
  bool clear;
  if (rt.dcc_is_cmask) {
    // A CMASK nibble of 0 is a fast-cleared tile; 0xF is an expanded one.
    clear = (code & 0xF) == 0 && clear_word;
    if (clear)
      value = ColorTargetClearValue(info, clear_word[0], clear_word[1]);
  } else {
    clear = DccClearColor(code, info, clear_word, value);
  }
  if (clear) {
    rt.clear_pending = true;
    rt.clear_value = value;
    rt.clear_src = "dcc-clear";
  }
  if (kClearTrace) {
    static int n = 0;
    if (n++ < kClearTrace)
      BASE_LOGI("clear",
                "f{} draw#{} RT {:#x} {}x{} DCC {:#x} code={:08x} -> {}",
                g_frame.num, g_frame.draws, (unsigned long)base, rt.w, rt.h,
                (unsigned long)rt.dcc_base, code,
                clear ? "clear" : "not a clear");
  }
}

// End the current dynamic-rendering region. Attachments remain in attachment
// layouts until an actual sampled/transfer consumer requests a transition.
void EndRegion() {
  if (!g_region.open)
    return;
  PassProfEnd();
  g_frame.list->EndRenderPass();
  CmdEndLabel(g_frame.list);
  if (trace::Recording())
    trace::RegionEnd();
  g_region.open = false;
  g_region.cur_rt = 0;
  g_region.cur_mrt_count = 0;
  g_region.cur_depth = 0;
  g_region.cur_stencil = 0;
  if (const u64 slice = g_region.write_through) {
    g_region.write_through = 0;
    WriteRtToGuest(slice, g_region.write_through_tile);
  }
}

bool WriteRtToGuest(u64 base, u32 tile_mode) {
  auto it = g_rts.find(base);
  if (it == g_rts.end() || !it->second.texture || it->second.depth > 1)
    return false;
  RTarget& rt = it->second;
  const u32 elem = FormatBytes(rt.fmt);
  gcn::TextureLayout32 layout;
  if (!elem ||
      !gcn::BuildTextureLayout32(layout, rt.w, rt.h, rt.w, 1, 1, tile_mode,
                                 false, elem) ||
      !gpu::IsReadableRange(base, layout.size))
    return false;
  static rhi::Buffer* buf = nullptr;
  const u64 bytes = static_cast<u64>(rt.w) * rt.h * elem;
  if (!buf || buf->desc().size < bytes) {
    Device().WaitIdle();
    Device().Destroy(buf);
    rhi::BufferDesc desc;
    desc.size = bytes;
    desc.usage = rhi::kBufferCopyDst;
    desc.memory = rhi::MemoryKind::kReadback;
    desc.name = "rt write-through";
    buf = Device().CreateBuffer(desc);
    if (!buf)
      return false;
  }
  void* map = buf->mapped();
  render::FlushCsWritesRange(render::DefaultRenderer(), base, layout.size,
                          "rt-write-through");
  rhi::CommandList* list = BeginImmediate();
  if (!list)
    return false;
  const rhi::TextureState old_layout = rt.layout;
  TransitionImage(list, rt.texture, rt.layout, rhi::TextureState::kCopySrc);
  rhi::BufferTextureCopy copy;
  copy.region.width = rt.w;
  copy.region.height = rt.h;
  list->CopyTextureToBuffer(buf, rt.texture, &copy, 1);
  TransitionImage(list, rt.texture, rt.layout, old_layout);
  list->Barrier(rhi::kAccessCopyWrite, rhi::kAccessHostRead);
  const bool ok = EndImmediate(list);
  if (!ok ||
      !gcn::RetileTextureMip32(map, reinterpret_cast<void*>(base), layout, 0,
                               0))
    return false;
  InvalidateTexRange(base, layout.size);
  CsForgetGuestRange(base, layout.size);
  return true;
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
  // PA_CL_VPORT_ZSCALE/ZOFFSET are deliberately NOT applied here: doing so
  // collapses P.T.'s depth range (every fragment lands on one value, the scene
  // stops rendering), so whatever this title puts in those registers is not the
  // plain window_z transform. The registers are read into DrawInfo, and the
  // next person to try this needs to explain that first.
  const float min_depth = 0.0f, max_depth = 1.0f;
  g_frame.list->SetViewport(d.viewport_x_offset - d.viewport_x_scale,
                            d.viewport_y_offset - d.viewport_y_scale,
                            d.viewport_x_scale * 2.0f,
                            d.viewport_y_scale * 2.0f, min_depth, max_depth);
}

// Begin a dynamic-rendering region binding mrt_count color targets (mrt_base[0]
// is the primary). The common single-RT case (mrt_count == 1) binds exactly one
// attachment. depth_base != 0 additionally binds a depth attachment (cleared to
// depth_clear on its first use each frame, loaded thereafter); depth_base == 0
// leaves depth unbound (the 2D path).
bool BeginRegion(const u64* mrt_base,
                 const u32* mrt_info,
                 u32 mrt_count,
                 u32 w,
                  u32 h,
                  u64 depth_base,
                  float depth_clear,
                  u64 stencil_base,
                  u8 stencil_clear,
                  bool depth_read_only,
                 u32 depth_w,
                 u32 depth_h,
                 const u32* mrt_surf_w,
                 const u32* mrt_surf_h,
                 const u64* mrt_dcc_base,
                 const u32 (*mrt_clear_word)[2],
                 u64 depth_htile_base,
                 u32 depth_slice,
                 u32 layers,
                 bool meta_cmask) {
  ScopeNs _region_timer(&g_ns_region);
  // DELTA_GPU_QCHECK also gates this path: a checkpoint that fails here names
  // the DRAWS that ran before this region as the device loss, which is the
  // attribution the compute-side checkpoints cannot reach.
  if (QueueCheckArmed()) {
    char where[64];
    std::snprintf(where, sizeof(where), "region draw#%u rt=%#llx f%u",
                  g_frame.draws,
                  (unsigned long long)(mrt_count ? mrt_base[0] : 0),
                  g_frame.num);
    if (!QueueCheck(where))
      return false;
  }
  rhi::RenderPassDesc pass;
  rhi::ColorAttachment* colors = pass.colors;
  RTarget* targets[8]{};
  mrt_count = base::Min(mrt_count, 8u);
  for (u32 i = 0; i < 8; i++) {
    g_region.cur_fmt[i] = 0;
    g_region.cur_w[i] = 0;
    g_region.cur_h[i] = 0;
  }
  g_region.cur_area_w = w;
  g_region.cur_area_h = h;
  for (u32 i = 0; i < mrt_count; i++) {
    const u32 iw = RtSurfaceExtent(mrt_surf_w ? mrt_surf_w[i] : 0, w, 256);
    const u32 ih = RtSurfaceExtent(mrt_surf_h ? mrt_surf_h[i] : 0, h, 64);
    targets[i] =
        GetRT(mrt_base[i], iw, ih, ColorTargetFormat(mrt_info[i]), layers);
    if (!targets[i])
      return false;
    // A dispatch wrote these pixels since the image last saw them.
    if (!CsRefreshRtFromTruth(mrt_base[i]))
      render::FlushCsWritesRange(render::DefaultRenderer(), mrt_base[i],
                              RtByteSize(*targets[i]), "rt-bind");
    targets[i]->cb_info = mrt_info[i];
    if (mrt_dcc_base && mrt_dcc_base[i]) {
      targets[i]->dcc_base = mrt_dcc_base[i];
      targets[i]->dcc_is_cmask = meta_cmask;
    }
    if (meta_cmask && mrt_dcc_base && mrt_dcc_base[i])
      NoteCmaskBind(*targets[i]);
    if (targets[i]->dcc_clear_pending)
      ResolveDccClear(*targets[i], mrt_base[i], mrt_info[i],
                      mrt_clear_word ? mrt_clear_word[i] : nullptr);
    g_region.cur_fmt[i] = static_cast<u32>(ColorTargetFormat(mrt_info[i]));
    g_region.cur_w[i] = iw;
    g_region.cur_h[i] = ih;
  }
  // Depth gets the same surface-vs-drawn sizing as colour. A pass whose screen
  // scissor is smaller than the surface it binds (GTA:SA opens a 256x256
  // scissor over a 1792x960 Z buffer) otherwise builds a 256x256 depth image,
  // and everything that reads scene depth afterwards (the deferred lights'
  // world-position reconstruction, and every compute post pass, which falls
  // back to the zeroed guest bytes) reads a corner of the frame and garbage
  // elsewhere. RtSurfaceExtent only ever grows the image to the surface, so a
  // genuinely half-resolution Z bound to a full-resolution pass keeps the drawn
  // extent.
  // The depth views are single-layer, and every attachment of a layered pass
  // has to cover all its layers.
  if (layers > 1)
    depth_base = 0;
  const u32 dw = RtSurfaceExtent(depth_w, w, 256);
  const u32 dh = RtSurfaceExtent(depth_h, h, 64);
  DepthTarget* dt =
      depth_base ? GetDepthRT(depth_base, dw, dh, stencil_base, depth_slice + 1)
                 : nullptr;
  if (depth_base && !dt)
    return false;
  g_region.cur_mrt_count = 0;
  for (u32 i = 0; i < mrt_count; i++) {
    RTarget& rt = *targets[i];
    TransitionImage(g_frame.list, rt.texture, rt.layout,
               rhi::TextureState::kColorTarget);
    rt.dirty_for_read = true;
    auto& color = colors[i];
    color.view = rt.view;
    // Lazy clear (DELTA_GPU_LAZYCLEAR, default on): persist RT content across
    // frames (LOAD), clearing only when the game explicitly requested a clear
    // (clear_pending) or the RT was never rendered. Per-frame auto-clear wiped
    // baked-once content (room floor) whose redraw lands on a different frame
    // than its clear.
    // DELTA_GPU_FRAMECLEAR=<base>: force this target to load CLEAR on the FIRST
    // region of each frame (not every region, which would wipe a pass before a
    // later one in the same frame reads it). Setting clear_pending from EndFrame
    // does not work, it never reaches this gate.
    if (kFrameClearRt && mrt_base[i] == (u64)kFrameClearRt) {
      static int cleared_frame = -1;
      if (cleared_frame != g_frame.num) {
        cleared_frame = g_frame.num;
        rt.clear_pending = true;
        rt.clear_value = rhi::ClearColor{};
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
          BASE_LOGI("clear",
                    "f{} draw#{} RT {:#x} {}x{} loadOp=CLEAR value=({} {} {} "
                    "{}) pending={}({}) ever={}",
                    g_frame.num, g_frame.draws, (unsigned long)mrt_base[i],
                    rt.w, rt.h, rt.clear_value.f[0],
                    rt.clear_value.f[1], rt.clear_value.f[2],
                    rt.clear_value.f[3], (int)rt.clear_pending,
                    rt.clear_src, (int)rt.ever_rendered);
      }
      color.load = (rt.clear_pending || !rt.ever_rendered)
                       ? rhi::LoadOp::kClear
                       : rhi::LoadOp::kLoad;
    } else
      color.load =
          rt.used_this_frame ? rhi::LoadOp::kLoad : rhi::LoadOp::kClear;
    rt.clear_pending = false;
    rt.ever_rendered = true;
    color.clear = rt.clear_value;
    // DELTA_GPU_CLEARCOLOR / DELTA_GPU_CLEARRED: diagnostic knobs that force
    // every bound RT to clear to a solid colour this frame, to verify which RTs
    // are bound.
    if (kForceClear) {
      color.load = rhi::LoadOp::kClear;
      color.clear = rhi::ClearColor{{0.f, 1.f, 0.f, 1.f}};
    }
    if (kClearRed) {
      color.load = rhi::LoadOp::kClear;
      color.clear = rhi::ClearColor{{1.0f, 0.0f, 0.0f, 1.0f}};
    }
    rt.used_this_frame = true;
    rt.last_frame = g_frame.num;
    rt.render_serial = ++g_render_serial;
    g_region.cur_mrt[i] = mrt_base[i];
  }
  g_region.cur_mrt_count = mrt_count;
  RTarget* primary = mrt_count ? targets[0] : nullptr;
  u64 base = primary ? mrt_base[0] : 0;
  // Depth attachment (3D). Cleared to the guest DB_DEPTH_CLEAR value on its
  // first use this frame, then loaded so multiple regions in a frame share one
  // Z buffer.
  rhi::DepthAttachment& depth_att = pass.depth;
  if (dt && depth_w && depth_h) {
    // The IMAGE has to cover the render area, but the guest FOOTPRINT is the
    // Z surface's own padded geometry. Keeping them apart is what stops a
    // half-resolution depth buffer bound to a full-resolution pass from
    // claiming several megabytes it does not own, and swallowing a
    // neighbouring surface in the sampled-address lookup.
    dt->guest_w = depth_w;
    dt->guest_h = depth_h;
  }
  if (!dt && depth_base && kRegTrace) {
    static int n = 0;
    if (n++ < 8)
      BASE_LOGI("region",
                "depth {:#x} REQUESTED BUT NOT BOUND; every depth write in "
                "this region is dropped",
                (unsigned long)depth_base);
  }
  if (dt) {
    if (depth_htile_base)
      dt->htile_base = depth_htile_base;
    if (dt->htile_clear_pending)
      ResolveHtileClear(*dt, depth_base, depth_clear);
    if (dt->clear_pending && dt->layers > 1) {
      dt->clear_layers = ~0u;
      dt->clear_pending = false;
    }
    const bool layer_clear = (dt->clear_layers >> depth_slice) & 1;
    const bool clear_depth = dt->clear_pending || layer_clear ||
                             (!dt->used_this_frame && !dt->htile_base);
    // Async compute is currently serialized ahead of the next graphics frame.
    // Keep this frame's scene depth in its persistent CS range before a later
    // pass clears the shared depth image, or next frame's compute sees zero.
    if (clear_depth && dt->used_this_frame)
      PreserveCsDepthBeforeClear(depth_base);
    // A pass that samples the depth it tests against keeps the image in the
    // read-only layout, which is what makes attachment and sampled view legal
    // at the same time. A clear still needs write access, so never both.
    const bool read_only = depth_read_only && !clear_depth;
    TransitionImage(g_frame.list, dt->texture, dt->layout,
               read_only ? rhi::TextureState::kDepthRead
                         : rhi::TextureState::kDepthTarget,
               rhi::kAspectDepth, dt->layers);
    depth_att.view = DepthLayerView(*dt, depth_slice);
    depth_att.depth = true;
    depth_att.read_only = read_only;
    depth_att.depth_load =
        clear_depth ? rhi::LoadOp::kClear : rhi::LoadOp::kLoad;
    depth_att.clear_depth =
        dt->clear_pending || layer_clear ? dt->clear_value : depth_clear;
    if (kRegTrace) {
      static int n = 0;
      if (n++ < 200)
        BASE_LOGI("region",
                  "depth {:#x} {}x{} load={} store={} clear={} read_only={} "
                  "rt={:#x}",
                  (unsigned long)depth_base, dt->w, dt->h,
                  clear_depth ? "CLEAR" : "LOAD", read_only ? "NONE" : "STORE",
                  depth_att.clear_depth, (int)read_only, (unsigned long)base);
    }
    dt->clear_pending = false;
    dt->clear_layers &= ~(1u << depth_slice);
    dt->used_this_frame = true;
    dt->last_frame = g_frame.num;
    if (!read_only)
      dt->render_serial = ++g_render_serial;
    g_region.cur_depth = depth_base;
    g_region.cur_depth_slice = depth_slice;
    if (stencil_base) {
      const bool clear_stencil = !dt->stencil_used_this_frame;
      TransitionImage(g_frame.list, dt->texture, dt->stencil_layout,
                 rhi::TextureState::kDepthTarget, rhi::kAspectStencil,
                 dt->layers);
      depth_att.stencil = true;
      depth_att.stencil_load =
          clear_stencil ? rhi::LoadOp::kClear : rhi::LoadOp::kLoad;
      depth_att.clear_stencil = stencil_clear;
      dt->stencil_used_this_frame = true;
      g_region.cur_stencil = stencil_base;
    }
  }
  // A pending clear covers the whole surface, but loadOp=CLEAR only reaches
  // the render area. GTA:SA clears its depth and G-buffer inside a 256x256
  // capture pass, which left the rest of the frame holding last frame's depth.
  for (u32 i = 0; i < g_region.cur_mrt_count; i++) {
    RTarget& rt = *targets[i];
    if (colors[i].load != rhi::LoadOp::kClear || (rt.w <= w && rt.h <= h))
      continue;
    const rhi::TextureState att = rt.layout;
    TransitionImage(g_frame.list, rt.texture, rt.layout,
               rhi::TextureState::kCopyDst);
    g_frame.list->ClearTexture(rt.texture, rhi::TextureState::kCopyDst, {},
                               colors[i].clear);
    TransitionImage(g_frame.list, rt.texture, rt.layout, att);
    colors[i].load = rhi::LoadOp::kLoad;
  }
  if (dt && (dt->w > w || dt->h > h)) {
    const auto clear_aspect = [&](rhi::LoadOp& load, rhi::TextureState& layout,
                                  u8 aspect) {
      if (load != rhi::LoadOp::kClear)
        return;
      const rhi::TextureState att = layout;
      TransitionImage(g_frame.list, dt->texture, layout,
                 rhi::TextureState::kCopyDst, aspect, dt->layers);
      rhi::TextureRange range;
      range.aspect = aspect;
      range.base_layer = depth_slice;
      g_frame.list->ClearDepthStencil(dt->texture, rhi::TextureState::kCopyDst,
                                      range, depth_att.clear_depth,
                                      depth_att.clear_stencil);
      TransitionImage(g_frame.list, dt->texture, layout, att, aspect,
                 dt->layers);
      load = rhi::LoadOp::kLoad;
    };
    clear_aspect(depth_att.depth_load, dt->layout, rhi::kAspectDepth);
    if (stencil_base)
      clear_aspect(depth_att.stencil_load, dt->stencil_layout,
                   rhi::kAspectStencil);
  }
  pass.width = w;
  pass.height = h;
  pass.layers = layers;
  g_region.cur_layers = layers;
  pass.color_count = g_region.cur_mrt_count;
  if (!dt)
    depth_att = {};
  if (trace::Recording()) {
    trace::RegionInfo info;
    info.mrt_base = mrt_base;
    info.mrt_info = mrt_info;
    info.mrt_count = g_region.cur_mrt_count;
    info.width = w;
    info.height = h;
    info.depth_base = depth_base;
    info.stencil_base = stencil_base;
    for (u32 i = 0; i < g_region.cur_mrt_count; i++)
      if (colors[i].load == rhi::LoadOp::kClear)
        info.color_clear_mask |= 1u << i;
    info.depth_clear = dt && depth_att.depth_load == rhi::LoadOp::kClear;
    info.depth_clear_value = depth_att.clear_depth;
    trace::RegionBegin(info);
  }
  CmdBeginLabel(g_frame.list, "region rt=%#llx %ux%u mrt=%u depth=%#llx",
                (unsigned long long)base, w, h, g_region.cur_mrt_count,
                (unsigned long long)depth_base);
  g_frame.list->BeginRenderPass(pass);
  PassProfBegin(mrt_count ? mrt_base[0] : depth_base);
  g_region.open = true;
  g_region.depth_read_only = depth_base && depth_read_only;
  // Negative-height (y-up) viewport: GCN/PS4 rasterises y-up, so we do too.
  // This stores render-target content upright, so render-to-texture composites
  // (the scene->scanout copy, effect overlays) sample it with aligned UVs when
  // run through the game's real recompiled shader, and the presented scanout is
  // already upright (no readback flip needed; DELTA_GPU_FLIP defaults to 0).
  g_frame.list->SetViewport(0, static_cast<float>(h), static_cast<float>(w),
                            -static_cast<float>(h), 0, 1);
  g_frame.list->SetScissor(0, 0, w, h);
  if (primary) {
    primary->used_this_frame = true;
    primary->last_frame = g_frame.num;
    if (primary->w >= 700 && primary->w <= 900)
      g_frame.room_bake = true;
    g_region.last_rt = base;
  }
  g_region.cur_rt = base;
  if (kRegTrace && w < 1280)
    BASE_LOGI("reg", "f{} begin RT {:#x} {}x{} mrt={} clear={}", g_frame.num,
              (unsigned long)base, w, h, g_region.cur_mrt_count,
              g_region.cur_mrt_count &&
                  colors[0].load == rhi::LoadOp::kClear);
  return true;
}

}  // namespace gpu::render

namespace gpu::render {

void NoteMemoryFill(Renderer& renderer,
                    u64 base,
                    u64 bytes,
                    u32 value) {
  if (!renderer.available() || !bytes)
    return;
  if (trace::Recording())
    trace::RecordMemoryFill(base, bytes, value);
  NoteDccWrite(base, bytes, &value);
  const u64 end = base + bytes;
  const auto note = [&](RTarget& rt, u64 rt_base) {
    const u64 rt_end = rt_base + RtByteSize(rt);
    // Only a fill that covers the whole surface is a clear; a partial one is a
    // buffer update that happens to overlap.
    if (base > rt_base || end < rt_end)
      return;
    // The fill repeats one dword, so each texel holds that dword pattern in
    // the target's own guest format: decode it the way a fast-clear word is.
    // Reading it as four 8-bit channels made a float target (UE4's exposure
    // targets) clear to garbage, and the scene blew out from there on. A
    // target never bound as colour has no known encoding; zero is zero in all
    // of them, anything else reloads from the guest bytes instead.
    if (rt.cb_info) {
      rt.clear_value = ColorTargetClearValue(rt.cb_info, value, value);
    } else if (value == 0) {
      rt.clear_value = rhi::ClearColor{};
    } else {
      rt.ever_rendered = false;
      return;
    }
    rt.clear_pending = true;
    rt.clear_src = "memory-fill";
    static int n = 0;
    if (kGpuFilltrace && n++ < kGpuFilltrace)
      BASE_LOGI("fill",
                "f{} draw#{} RT {:#x} {}x{} {} info={:#x} cleared by fill "
                "{:08x} -> ({}, {}, {}, {}) (base={:#x} {} bytes)",
                g_frame.num, g_frame.draws, (unsigned long)rt_base, rt.w,
                rt.h, rhi::FormatName(rt.fmt), rt.cb_info, value,
                rt.clear_value.f[0], rt.clear_value.f[1], rt.clear_value.f[2],
                rt.clear_value.f[3], (unsigned long)base,
                (unsigned long)bytes);
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

}  // namespace gpu::render
