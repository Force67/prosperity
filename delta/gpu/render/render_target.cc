/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/render_target.h"
#include "base/arch.h"
#include "gpu/gpu_perf.h"
#include "guest/session.h"

#include "gpu/guest_memory.h"
#include "gpu/render/compute.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "gpu/render/guest_format.h"
#include "gpu/render/labels.h"
#include "gpu/render/renderer.h"
#include "gpu/render/texture_cache.h"
#include "gpu/render/trace.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/algorithm.h"
#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "options/options.h"

namespace {
DELTA_OPTION(u32, kMaxRenderTargets, "DELTA_GPU_RT_MAX", 256);
DELTA_OPTION(int, kClearTrace, "DELTA_GPU_CLEARTRACE", 0);
DELTA_OPTION(bool, kLazyClear, "DELTA_GPU_LAZYCLEAR", true);
DELTA_OPTION(bool, kClearRed, "DELTA_GPU_CLEARRED", false);
DELTA_OPTION(bool, kForceClear, "DELTA_GPU_CLEARCOLOR", false);
DELTA_OPTION(int, kGpuFilltrace, "DELTA_GPU_FILLTRACE", 0);
DELTA_OPTION(bool, kRegTrace, "DELTA_GPU_REGTRACE", false);
DELTA_OPTION(bool, kResolveTrace, "DELTA_GPU_RESOLVETRACE", false);
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

void TransitionImage(rhi::CommandList* list,
                     rhi::Texture* texture,
                     rhi::TextureState& current,
                     rhi::TextureState state,
                     u8 aspect,
                     u32 layers) {
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
  if (identity && IdentitySwizzle(desc.swizzle))
    return identity;
  const u32 key = swizzle | (static_cast<u32>(dim) << 24);
  const auto it = views.find(key);
  if (it != views.end())
    return it->second;
  desc.dim = dim;
  desc.format = (format);
  desc.aspect = aspect;
  rhi::TextureView* view = Device().CreateView(texture, desc);
  if (!view)
    return nullptr;
  views.emplace(key, view);
  return view;
}

rhi::TextureView* SampledViewAs(RTarget& rt,
                                u32 swizzle,
                                rhi::Format want,
                                rhi::Format* used,
                                bool arrayed) {
  const auto own = [&](rhi::TextureView* v) {
    if (used)
      *used = rt.fmt;
    return v;
  };
  if (want == rhi::Format::kUndefined || want == rt.fmt ||
      FormatBytes(want) != FormatBytes(rt.fmt) || FormatBlockCompressed(want))
    return own(SampledView(rt, swizzle, false, arrayed));
  if (used)
    *used = want;
  const u32 key = swizzle | (static_cast<u32>(want) << 16) |
                  (static_cast<u32>(arrayed) << 31);
  const auto it = rt.alias_views.find(key);
  if (it != rt.alias_views.end())
    return it->second;
  rhi::TextureViewDesc desc;
  desc.dim = rt.depth > 1 ? rhi::ViewDim::k3D
             : arrayed    ? rhi::ViewDim::k2DArray
                          : rhi::ViewDim::k2D;
  desc.format = (want);
  TextureSwizzle(swizzle, desc.swizzle);
  rhi::TextureView* v = Device().CreateView(rt.texture, desc);
  if (!v)
    return own(SampledView(rt, swizzle, false, arrayed));
  rt.alias_views[key] = v;
  return v;
}

rhi::TextureView* SampledView(RTarget& rt,
                              u32 swizzle,
                              bool feedback,
                              bool arrayed) {
  if (feedback && !rt.feedback_texture)
    return nullptr;
  if (feedback)
    return SampledImageView(
        rt.feedback_texture, arrayed ? nullptr : rt.feedback_view, rt.fmt,
        rhi::kAspectColor, swizzle, rt.feedback_sampled_views,
        arrayed ? rhi::ViewDim::k2DArray : rhi::ViewDim::k2D);
  if (rt.depth > 1)
    return SampledImageView(rt.texture, rt.volume_view, rt.fmt,
                            rhi::kAspectColor, swizzle, rt.sampled_views,
                            rhi::ViewDim::k3D);
  return SampledImageView(rt.texture, arrayed ? nullptr : rt.view, rt.fmt,
                          rhi::kAspectColor, swizzle, rt.sampled_views,
                          arrayed ? rhi::ViewDim::k2DArray : rhi::ViewDim::k2D);
}

rhi::TextureView* SampledView(DepthTarget& depth, u32 swizzle) {
  return SampledImageView(depth.texture, depth.view, kDepthFormat,
                          rhi::kAspectDepth, swizzle, depth.sampled_views);
}

u64 RtByteSizeWH(u32 w, u32 h, rhi::Format fmt) {
  return (u64)w * h * FormatBytes(fmt);
}

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

void ClearNewRt(RTarget& t) {
  const bool in_frame = g_frame.recording;
  rhi::CommandList* list = FrameOrBatchList();
  if (!list)
    return;
  TransitionImage(list, t.texture, t.layout, rhi::TextureState::kCopyDst);
  list->ClearTexture(t.texture, rhi::TextureState::kCopyDst, {},
                     rhi::ClearColor{});
  TransitionImage(list, t.texture, t.layout, rhi::TextureState::kShaderRead);
  if (!in_frame)
    t.submitted_layout = t.layout;
}

bool CreateRtImage(RTarget& t,
                   u64 base,
                   u32 w,
                   u32 h,
                   rhi::Format fmt,
                   u32 depth = 1) {
  if (!w || !h || !depth)
    return false;
  if (w > 8192 || h > 8192 || fmt == rhi::Format::kUndefined) {
    BASE_LOGI("gpuvk", "skip bad RT {:#x} {}x{} fmt={}", (unsigned long)base, w,
              h, (int)fmt);
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
  if (fmt != rhi::Format::kRGBA8Srgb && fmt != rhi::Format::kBGRA8Srgb)
    td.usage |= rhi::kTextureStorage;
  td.usage |= rhi::kTextureMutableFormat;
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
  if (g_tex.descriptors_ready && depth == 1)
    t.set = SampledTextureGroup(t.view, g_tex.sampler);
  ClearNewRt(t);
  BASE_LOGI("gpuvk", "new RT {:#x} {}x{}x{} fmt={}", (unsigned long)base, w, h,
            depth, (int)fmt);
  if (Device().caps().debug_labels) {
    char name[64];
    std::snprintf(name, sizeof(name), "rt %#lx %ux%u fmt=%d",
                  (unsigned long)base, w, h, (int)fmt);
    Device().SetName(t.texture, name);
  }
  return true;
}

base::HashMap<u64, base::Vector<RTarget>> g_rt_variants;
constexpr u64 kMaxRtVariantBytes = 32ull << 20;
constexpr size_t kMaxRtVariants = 256;

base::Vector<RTarget> g_retired_rts;

void RetireRt(RTarget& t) {
  g_retired_rts.push_back(base::move(t));
  ClearMultiTexCache();
}

void DestroyRt(RTarget& t) {
  rhi::Device& device = Device();
  for (auto* views :
       {&t.sampled_views, &t.alias_views, &t.feedback_sampled_views})
    for (auto& [key, view] : *views)
      device.Destroy(view);
  for (rhi::Object* o : std::initializer_list<rhi::Object*>{
           t.set, t.feedback_set, t.view, t.volume_view, t.feedback_view})
    device.Destroy(o);
  device.Destroy(t.feedback_texture);
  device.Destroy(t.texture);
}

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
      parked_bytes += RtByteSizeWH(v.w, v.h, v.fmt) * v.depth;
    const u64 retiring_bytes =
        RtByteSizeWH(live.w, live.h, live.fmt) * live.depth;
    while (parked.size() >= kMaxRtVariants ||
           parked_bytes + retiring_bytes > kMaxRtVariantBytes) {
      size_t oldest = parked.size();
      for (size_t i = 0; i < parked.size(); i++)
        if (parked[i].last_frame < g_frame.num &&
            (oldest == parked.size() ||
             parked[i].last_frame < parked[oldest].last_frame))
          oldest = i;
      if (oldest == parked.size())
        break;
      parked_bytes -=
          RtByteSizeWH(parked[oldest].w, parked[oldest].h, parked[oldest].fmt) *
          parked[oldest].depth;
      RetireRt(parked[oldest]);
      parked.erase(parked.begin() + oldest);
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
  const bool srgb_alias =
      (live.fmt == rhi::Format::kRGBA8Unorm &&
       fmt == rhi::Format::kRGBA8Srgb) ||
      (live.fmt == rhi::Format::kRGBA8Srgb &&
       fmt == rhi::Format::kRGBA8Unorm) ||
      (live.fmt == rhi::Format::kBGRA8Unorm &&
       fmt == rhi::Format::kBGRA8Srgb) ||
      (live.fmt == rhi::Format::kBGRA8Srgb && fmt == rhi::Format::kBGRA8Unorm);
  if (srgb_alias && live.w == w && live.h == h && live.depth == depth &&
      live.ever_rendered && g_frame.recording) {
    EndRegion();
    const rhi::TextureState old_state = live.layout;
    TransitionImage(g_frame.list, live.texture, live.layout,
                    rhi::TextureState::kCopySrc);
    TransitionImage(g_frame.list, alt->texture, alt->layout,
                    rhi::TextureState::kCopyDst);
    rhi::TextureRegion region;
    region.width = w;
    region.height = h;
    region.depth = depth;
    g_frame.list->CopyTexture(alt->texture, region, live.texture, region);
    TransitionImage(g_frame.list, live.texture, live.layout, old_state);
    alt->ever_rendered = true;
    alt->dirty_for_read = true;
    alt->render_serial = live.render_serial;
    alt->last_frame = live.last_frame;
    alt->used_this_frame = live.used_this_frame;
    alt->accumulated = live.accumulated;
    if (alt->dcc_base == live.dcc_base) {
      alt->dcc_clear_pending = live.dcc_clear_pending;
      alt->dcc_code_known = live.dcc_code_known;
      alt->dcc_clear_code = live.dcc_clear_code;
    }
    alt->clear_pending = live.clear_pending;
    alt->clear_value = live.clear_value;
    alt->clear_src = live.clear_src;
  }
  base::Swap(live, *alt);
  if (trace::Recording())
    trace::RecordVariantSwap(base, reinterpret_cast<u64>(alt->texture), alt->w,
                             alt->h, reinterpret_cast<u64>(live.texture),
                             live.w, live.h);
  if (live.last_frame != g_frame.num) {
    live.used_this_frame = false;
    live.draws = 0;
    live.clear_pending = false;
    live.clear_src = "none";
  }
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
    const auto alt =
        base::FindIf(parked->second.begin(), parked->second.end(), matches);
    if (alt == parked->second.end() ||
        !ActivateRtVariant(live, base, w, h, alt->fmt, depth))
      return false;
  }
  return live.ever_rendered;
}

u32 RtSurfaceExtent(u32 surface, u32 drawn, u32 tile) {
  const u32 padded = (drawn + tile - 1) & ~(tile - 1);
  return (surface > padded && surface <= 8192) ? surface : drawn;
}

RTarget* GetRT(u64 base, u32 w, u32 h, rhi::Format fmt, u32 depth) {
  auto it = g_rts.find(base);
  if (it != g_rts.end()) {
    RTarget& live = it->second;
    if (live.w != w || live.h != h || live.fmt != fmt || live.depth != depth)
      return ActivateRtVariant(live, base, w, h, fmt, depth);
    return &live;
  }
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
  RegisterRtPages(base, w, h, fmt);
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

DELTA_OPTION(bool, kDepthVariants, "DELTA_GPU_DEPTHVARIANTS", true);

DepthTarget* ActivateDepthVariant(DepthTarget& live,
                                  u64 base,
                                  u32 w,
                                  u32 h,
                                  u64 stencil_base,
                                  u32 layers = 1) {
  // A depth attachment must cover the render area
  // (VUID-VkRenderingInfo-pNext-06079).
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
    const u32 want =
        live.w == w && live.h == h ? base::Max(layers, live.layers) : layers;
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
  return &live;
}

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
  ClearMultiTexCache();
}

void ReleaseRetiredTargets() {
  static base::Vector<RTarget> aged_rts;
  static const guest::SessionReset reset_aged_rts(
      [] { guest::ResetResource(aged_rts); });
  for (RTarget& t : aged_rts)
    DestroyRt(t);
  aged_rts = base::move(g_retired_rts);
  g_retired_rts.clear();
  static base::Vector<DepthTarget> aged;
  static const guest::SessionReset reset_aged(
      [] { guest::ResetResource(aged); });
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

DepthTarget* GetDepthRT(u64 base, u32 w, u32 h, u64 stencil_base, u32 layers) {
  auto it = g_depths.find(base);
  if (it != g_depths.end()) {
    DepthTarget& live = it->second;
    if (stencil_base)
      live.stencil_base = stencil_base;
    if (!w || !h || (live.w == w && live.h == h && live.layers >= layers))
      return &live;
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
      return;
    u64 b1 = b0 + RtByteSize(rt);
    if (!(a0 < b1 && b0 < a1))
      return;
    bool dim_match = (!w || !h) || (rt.w == w && rt.h == h);
    if (!dim_match)
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
        if (p == base::Max(a0 >> kRtPageShift, b0 >> kRtPageShift))
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

base::Map<u64, bool>& MetadataBases() {
  static base::Map<u64, bool> bases;
  static const guest::SessionReset reset_bases(
      [] { guest::ResetResource(bases); });
  return bases;
}

void NoteMetadataBase(u64 base) {
  if (base)
    MetadataBases()[base] = true;
}

void NoteDccWrite(u64 base, u64 bytes, const u32* fill) {
  if (!base || !bytes)
    return;
  const auto& bases = MetadataBases();
  const auto it = bases.lower_bound(base);
  if (it == bases.end() || it->first >= base + bytes)
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

void NoteExpandedHtile(u64 base) {
  const auto note = [&](DepthTarget& dt) {
    if (dt.htile_base == base)
      dt.htile_clear_pending = false;
  };
  for (auto& kv : g_depths) {
    note(kv.second);
    auto v = g_depth_variants.find(kv.first);
    if (v != g_depth_variants.end())
      for (DepthTarget& alt : v->second)
        note(alt);
  }
}

void ResolveHtileClear(DepthTarget& dt, u64 base, float depth_clear) {
  dt.htile_clear_pending = false;
  u32 code = dt.htile_clear_code;
  if (!dt.htile_code_known) {
    render::FlushCsWritesRange(render::DefaultRenderer(), dt.htile_base, 4,
                               "htile");
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

void NoteSurfaceWrite(u64 base, u64 bytes, bool raw) {
  if (!base || !bytes)
    return;
  const auto note = [&](RTarget& rt, u64 rt_base) {
    if (base >= rt_base + RtByteSize(rt) || rt_base >= base + bytes)
      return;
    rt.dcc_clear_pending = false;
    if (rt.clear_pending && rt.clear_src &&
        !std::strcmp(rt.clear_src, "dcc-clear"))
      rt.clear_pending = false;
    if (raw && rt.last_frame != g_frame.num)
      rt.ever_rendered = false;
  };
  for (auto& [at, rt] : g_rts)
    note(rt, at);
  for (auto& [at, variants] : g_rt_variants)
    for (RTarget& alt : variants)
      note(alt, at);
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

bool ActivateRtVariantAs(u64 base, u32 w, u32 h, rhi::Format fmt) {
  auto it = g_rts.find(base);
  if (it == g_rts.end() || !w || !h || fmt == rhi::Format::kUndefined)
    return false;
  RTarget& live = it->second;
  if (live.w == w && live.h == h && live.fmt == fmt)
    return true;
  return ActivateRtVariant(live, base, w, h, fmt) != nullptr;
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

void NoteCmaskBind(RTarget& rt) {
  const u64 bytes = base::Max<u64>(4, (u64(rt.w) * rt.h) / 128);
  if (!gpu::IsReadableRangeCached(rt.dcc_base, bytes))
    return;
  if (CsClearPredicate(rt.dcc_base, bytes, false, &rt.clear_pred,
                       &rt.clear_pred_offset)) {
    rt.dcc_clear_pending = false;
    return;
  }
  render::FlushCsWritesRange(render::DefaultRenderer(), rt.dcc_base, 4,
                             "cmask-bind");
  auto* cmask = reinterpret_cast<u32*>(rt.dcc_base);
  if (*cmask & 0xF)
    return;
  rt.dcc_clear_pending = true;
  rt.dcc_code_known = true;
  rt.dcc_clear_code = *cmask;
  std::memset(cmask, 0xFF, bytes);
}

void ResolveDccClear(RTarget& rt, u64 base, u32 info, const u32* clear_word) {
  rt.dcc_clear_pending = false;
  u32 code = rt.dcc_clear_code;
  if (!rt.dcc_code_known) {
    render::FlushCsWritesRange(render::DefaultRenderer(), rt.dcc_base, 4,
                               "cmask-resolve");
    if (!gpu::IsReadableRangeCached(rt.dcc_base, 4))
      return;
    std::memcpy(&code, reinterpret_cast<const void*>(rt.dcc_base), 4);
  }
  rhi::ClearColor value{};
  bool clear;
  if (rt.dcc_is_cmask) {
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
      BASE_LOGI(
          "clear", "f{} draw#{} RT {:#x} {}x{} DCC {:#x} code={:08x} -> {}",
          g_frame.num, g_frame.draws, (unsigned long)base, rt.w, rt.h,
          (unsigned long)rt.dcc_base, code, clear ? "clear" : "not a clear");
  }
}

void EndRegion() {
  g_region.reads_synced = false;
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
  gcn::TextureLayout32 tiled, linear;
  if (!elem ||
      !gcn::BuildTextureLayout32(tiled, rt.w, rt.h, rt.w, 1, 1, tile_mode,
                                 false, elem) ||
      !gcn::BuildTextureLayout32(linear, rt.w, rt.h, rt.w, 1, 1, 8, false,
                                 elem) ||
      !gpu::IsReadableRange(base, tiled.size) ||
      !render::FlushCsWritesForGpu(render::DefaultRenderer(), base, tiled.size,
                                   "rt-write-through") ||
      !WriteTextureToGuest(rt.texture, rt.layout, base, tiled, linear))
    return false;
  InvalidateTexRange(base, tiled.size);
  CsForgetGuestRange(base, tiled.size);
  return true;
}

void SetGuestViewport(const DrawInfo& d) {
  if (!std::isfinite(d.viewport_x_scale) ||
      !std::isfinite(d.viewport_x_offset) ||
      !std::isfinite(d.viewport_y_scale) ||
      !std::isfinite(d.viewport_y_offset) || d.viewport_x_scale <= 0.0f ||
      d.viewport_y_scale == 0.0f)
    return;
  // window_z = ndc_z * ZSCALE + ZOFFSET matches Vulkan's depth range transform.
  const float min_depth = 0.0f, max_depth = 1.0f;
  g_frame.list->SetViewport(d.viewport_x_offset - d.viewport_x_scale,
                            d.viewport_y_offset - d.viewport_y_scale,
                            d.viewport_x_scale * 2.0f,
                            d.viewport_y_scale * 2.0f, min_depth, max_depth);
}

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
  ScopeNs region_timer(&g_ns_region);
  if (QueueCheckArmed()) {
    char where[64];
    std::snprintf(
        where, sizeof(where), "region draw#%u rt=%#llx f%u", g_frame.draws,
        (unsigned long long)(mrt_count ? mrt_base[0] : 0), g_frame.num);
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
    if (!CsRefreshRtFromTruth(mrt_base[i]))
      render::FlushCsWritesAt(render::DefaultRenderer(), mrt_base[i],
                              "rt-bind");
    targets[i]->cb_info = mrt_info[i];
    if (mrt_dcc_base && mrt_dcc_base[i]) {
      targets[i]->dcc_base = mrt_dcc_base[i];
      NoteMetadataBase(mrt_dcc_base[i]);
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
      if (kClearTrace && (rt.clear_pending || !rt.ever_rendered)) {
        static int n = 0;
        if (n++ < kClearTrace)
          BASE_LOGI("clear",
                    "f{} draw#{} RT {:#x} {}x{} loadOp=CLEAR value=({} {} {} "
                    "{}) pending={}({}) ever={}",
                    g_frame.num, g_frame.draws, (unsigned long)mrt_base[i],
                    rt.w, rt.h, rt.clear_value.f[0], rt.clear_value.f[1],
                    rt.clear_value.f[2], rt.clear_value.f[3],
                    (int)rt.clear_pending, rt.clear_src, (int)rt.ever_rendered);
      }
      color.load = (rt.clear_pending || !rt.ever_rendered) ? rhi::LoadOp::kClear
                                                           : rhi::LoadOp::kLoad;
    } else
      color.load =
          rt.used_this_frame ? rhi::LoadOp::kLoad : rhi::LoadOp::kClear;
    rt.clear_pending = false;
    rt.ever_rendered = true;
    color.clear = rt.clear_value;
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
  rhi::DepthAttachment& depth_att = pass.depth;
  if (dt && depth_w && depth_h) {
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
    if (depth_htile_base) {
      dt->htile_base = depth_htile_base;
      NoteMetadataBase(depth_htile_base);
    }
    if (dt->htile_clear_pending && !dt->htile_code_known && !depth_read_only &&
        CsClearPredicate(dt->htile_base, 0, true, &dt->clear_pred,
                         &dt->clear_pred_offset))
      dt->htile_clear_pending = false;
    if (dt->htile_clear_pending)
      ResolveHtileClear(*dt, depth_base, depth_clear);
    if (dt->clear_pending && dt->layers > 1) {
      dt->clear_layers = ~0u;
      dt->clear_pending = false;
    }
    const bool layer_clear = (dt->clear_layers >> depth_slice) & 1;
    const bool clear_depth = dt->clear_pending || layer_clear ||
                             (!dt->used_this_frame && !dt->htile_base);
    if ((clear_depth || dt->clear_pred) && dt->used_this_frame)
      PreserveCsDepthBeforeClear(depth_base);
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
  if (dt && dt->clear_pred && depth_att.depth_load != rhi::LoadOp::kClear &&
      (dt->w > w || dt->h > h)) {
    rhi::RenderPassDesc whole;
    whole.depth = depth_att;
    whole.depth.stencil = false;
    whole.width = dt->w;
    whole.height = dt->h;
    g_frame.list->BeginRenderPass(whole);
    g_frame.list->BeginConditional(dt->clear_pred, dt->clear_pred_offset,
                                   false);
    g_frame.list->ClearAttachment(~0u, {}, depth_clear, 0, rhi::kAspectDepth, 0,
                                  0, dt->w, dt->h);
    g_frame.list->EndConditional();
    g_frame.list->EndRenderPass();
    dt->clear_pred = nullptr;
  }
  g_frame.list->BeginRenderPass(pass);
  for (u32 i = 0; i < mrt_count; i++) {
    RTarget& rt = *targets[i];
    if (!rt.clear_pred)
      continue;
    const u32* word = mrt_clear_word ? mrt_clear_word[i] : nullptr;
    if (word && colors[i].load != rhi::LoadOp::kClear) {
      g_frame.list->BeginConditional(rt.clear_pred, rt.clear_pred_offset,
                                     false);
      g_frame.list->ClearAttachment(
          i, ColorTargetClearValue(mrt_info[i], word[0], word[1]), 0, 0,
          rhi::kAspectColor, 0, 0, w, h);
      g_frame.list->EndConditional();
    }
    rt.clear_pred = nullptr;
  }
  if (dt && dt->clear_pred) {
    if (depth_att.depth_load != rhi::LoadOp::kClear) {
      g_frame.list->BeginConditional(dt->clear_pred, dt->clear_pred_offset,
                                     false);
      g_frame.list->ClearAttachment(~0u, {}, depth_clear, 0, rhi::kAspectDepth,
                                    0, 0, w, h);
      g_frame.list->EndConditional();
    }
    dt->clear_pred = nullptr;
  }
  PassProfBegin(mrt_count ? mrt_base[0] : depth_base);
  g_region.open = true;
  g_region.depth_read_only = depth_base && depth_read_only;
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
              g_region.cur_mrt_count && colors[0].load == rhi::LoadOp::kClear);
  return true;
}

}  // namespace gpu::render

namespace gpu::render {

bool CanClearMemoryFill64(u64 base, u64 bytes) {
  if (!g_frame.recording || !bytes || (base & 7) || (bytes & 7) ||
      base + bytes < base)
    return false;
  const u64 end = base + bytes;
  const auto hits = [&](u64 at, u64 n) { return at < end && base < at + n; };
  for (const auto& [at, d] : g_depths)
    if (hits(at, u64(d.w) * d.h * 4 * base::Max(1u, d.layers)))
      return false;
  for (const auto& [at, variants] : g_depth_variants)
    for (const auto& d : variants)
      if (hits(at, u64(d.w) * d.h * 4 * base::Max(1u, d.layers)))
        return false;
  bool found = false;
  const auto supported = [&](u64 at, const RTarget& rt) {
    const u64 size = RtByteSize(rt);
    if (!hits(at, size))
      return true;
    found = true;
    return base <= at && end >= at + size && !((at - base) & 7) &&
           FormatBytes(rt.fmt) == 8 && rt.cb_info &&
           ColorTargetFormat(rt.cb_info) == rt.fmt;
  };
  for (const auto& [at, rt] : g_rts)
    if (!supported(at, rt))
      return false;
  for (const auto& [at, variants] : g_rt_variants)
    for (const auto& rt : variants)
      if (!supported(at, rt))
        return false;
  return found;
}

void NoteMemoryFill64(Renderer& renderer, u64 base, u64 bytes, u64 value) {
  if (!renderer.available() || !g_frame.recording)
    return;
  if (trace::Recording())
    trace::RecordMemoryFill(base, bytes, value, 8);
  EndRegion();
  const auto clear = [&](u64 at, RTarget& rt) {
    if (at >= base + bytes || base >= at + RtByteSize(rt))
      return;
    const auto color = ColorTargetClearValue(
        rt.cb_info, static_cast<u32>(value), static_cast<u32>(value >> 32));
    const auto from = rt.layout;
    TransitionImage(g_frame.list, rt.texture, rt.layout,
                    rhi::TextureState::kCopyDst);
    g_frame.list->ClearTexture(rt.texture, rhi::TextureState::kCopyDst, {},
                               color);
    if (from != rhi::TextureState::kUndefined)
      TransitionImage(g_frame.list, rt.texture, rt.layout, from);
    rt.clear_pending = false;
    rt.ever_rendered = true;
    rt.dirty_for_read = true;
    rt.last_frame = g_frame.num;
    rt.render_serial = ++g_render_serial;
  };
  for (auto& [at, rt] : g_rts)
    clear(at, rt);
  for (auto& [at, variants] : g_rt_variants)
    for (auto& rt : variants)
      clear(at, rt);
}

void NoteMemoryFill(Renderer& renderer, u64 base, u64 bytes, u32 value) {
  if (!renderer.available() || !bytes)
    return;
  if (trace::Recording())
    trace::RecordMemoryFill(base, bytes, value);
  NoteDccWrite(base, bytes, &value);
  const u64 end = base + bytes;
  const auto note = [&](RTarget& rt, u64 rt_base) {
    const u64 rt_end = rt_base + RtByteSize(rt);
    if (base > rt_base || end < rt_end)
      return;
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
                g_frame.num, g_frame.draws, (unsigned long)rt_base, rt.w, rt.h,
                rhi::FormatName(rt.fmt), rt.cb_info, value, rt.clear_value.f[0],
                rt.clear_value.f[1], rt.clear_value.f[2], rt.clear_value.f[3],
                (unsigned long)base, (unsigned long)bytes);
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

namespace {
const guest::SessionReset g_session_reset([] {
  guest::ResetResource(g_rt_variants);
  guest::ResetResource(g_retired_rts);
  guest::ResetResource(g_depth_variants);
  guest::ResetResource(g_render_serial);
  guest::ResetResource(g_retired_depths);
});
}  // namespace

}  // namespace gpu::render
