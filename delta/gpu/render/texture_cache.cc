/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/texture_cache.h"
#include "base/arch.h"

#include "gpu/gcn/gcn_detile.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gpu_check.h"
#include "gpu/gpu_perf.h"
#include "gpu/guest_memory.h"
#include "gpu/render/capture.h"
#include "gpu/render/compute.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "gpu/render/guest_format.h"
#include "gpu/render/hash.h"
#include "gpu/render/labels.h"
#include "gpu/render/render_target.h"
#include "gpu/render/renderer.h"
#include "gpu/render/upload_ring.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/algorithm.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/pair.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/time/time.h"
#include "options/options.h"
#include "profile/profile.h"

namespace {
DELTA_OPTION(int, kForceTile, "DELTA_GPU_FORCETILE", -1);
DELTA_OPTION(u32, kDumpMin, "DELTA_GPU_TEXDUMP_MIN", 128);
DELTA_OPTION(u64, kTextureMb, "DELTA_GPU_TEX_MB", 1024);
DELTA_OPTION(bool, kDefaultSampler, "DELTA_GPU_DEFSAMPLER", false);
DELTA_OPTION(bool, kDumpUpload, "DELTA_GPU_TEXDUMP_UPLOAD", false);
DELTA_OPTION(bool, kForceWhite, "DELTA_GPU_FORCEWHITE", false);
DELTA_OPTION(bool, kNoDetile, "DELTA_GPU_NODETILE", false);
DELTA_OPTION(bool, kTexDump, "DELTA_GPU_TEXDUMP", false);
DELTA_OPTION(u64, kDumpBase, "DELTA_GPU_TEXDUMP_BASE", 0);
DELTA_OPTION(bool, kForceNearest, "DELTA_GPU_FORCENEAREST", false);
DELTA_OPTION(bool, kTexFail, "DELTA_GPU_TEXFAIL", false);
DELTA_OPTION(bool, kTexMiss, "DELTA_GPU_TEXMISS", false);
DELTA_OPTION(bool, kIntegerRt, "DELTA_GPU_INT_RT", true);
DELTA_OPTION(bool, kTexRaw, "DELTA_GPU_TEXRAW", false);
DELTA_OPTION(u64, kTexWatch, "DELTA_GPU_TEXWATCH", 0);
DELTA_OPTION(int, kTexCensus, "DELTA_GPU_TEXCENSUS", 0);
// DELTA_GPU_FORCELOD=<n>: clamp every sampler to mip n. A streamed title
// uploads its mip chain progressively, so "this surface is fine" from a
// close-up view says nothing about the levels a minified one samples.
DELTA_OPTION(int, kForceLod, "DELTA_GPU_FORCELOD", -1);
// How far the content re-check of a texture that keeps holding still may back
// off. 1 restores the old every-frame validation.
DELTA_OPTION(u32, kMaxCheckInterval, "DELTA_GPU_TEXRECHECK", 32);
DELTA_OPTION(bool, kTexMemo, "DELTA_GPU_TEXMEMO", true);
DELTA_OPTION(bool, kTexPrevalidate, "DELTA_GPU_TEXPREVALIDATE", true);
}  // namespace

namespace gpu::render {

using render::DrawInfo;

struct TexImageKey {
  u64 base = 0;
  u32 w = 0, h = 0, tiling = 8, pitch = 0, layers = 1;
  u32 depth = 1;
  u32 mip_levels = 1;
  rhi::Format format = rhi::Format::kUndefined;
  bool pow2_pad = false;
  // A volume and a 2D surface at the same guest address are different image
  // types, so is_3d has to separate them here.
  bool is_3d = false;
  bool operator==(const TexImageKey& o) const {
    return base == o.base && w == o.w && h == o.h && tiling == o.tiling &&
           pitch == o.pitch && layers == o.layers && depth == o.depth &&
           mip_levels == o.mip_levels && format == o.format &&
           pow2_pad == o.pow2_pad && is_3d == o.is_3d;
  }
};

struct TexImageKeyHash {
  size_t operator()(const TexImageKey& k) const {
    u64 h = 1469598103934665603ull;
    h = HashWord(h, k.base);
    h = HashWord(h, k.w);
    h = HashWord(h, k.h);
    h = HashWord(h, k.tiling);
    h = HashWord(h, k.pitch);
    h = HashWord(h, k.layers);
    h = HashWord(h, k.depth);
    h = HashWord(h, k.mip_levels);
    h = HashWord(h, k.pow2_pad);
    h = HashWord(h, k.is_3d);
    h = HashWord(h, static_cast<u64>(k.format));
    return static_cast<size_t>(h);
  }
};

struct SamplerKey {
  u32 raw[4] = {};
  u32 image_min_lod = 0;
  bool valid = false;
  bool force_lod_zero = false;
  bool depth_compare = false;
  // Vulkan permits only NEAREST filtering on an integer-format image.
  bool integer = false;
  bool operator==(const SamplerKey& o) const {
    return valid == o.valid && image_min_lod == o.image_min_lod &&
           force_lod_zero == o.force_lod_zero &&
           depth_compare == o.depth_compare && integer == o.integer &&
           std::memcmp(raw, o.raw, sizeof(raw)) == 0;
  }
};

struct SamplerKeyHash {
  size_t operator()(const SamplerKey& k) const {
    u64 h = HashWord(1469598103934665603ull, k.valid);
    for (u32 word : k.raw)
      h = HashWord(h, word);
    h = HashWord(h, k.image_min_lod);
    h = HashWord(h, k.force_lod_zero);
    h = HashWord(h, k.depth_compare);
    h = HashWord(h, k.integer);
    return static_cast<size_t>(h);
  }
};

struct TexKey {
  TexImageKey image;
  u32 base_array = 0, view_layers = 1;
  u32 base_mip = 0, view_mips = 1;
  u32 swizzle = 0;  // packed T# DST_SEL (0 = identity)
  SamplerKey sampler;
  bool arrayed = false;
  bool operator==(const TexKey& o) const {
    return swizzle == o.swizzle && image == o.image &&
           base_array == o.base_array && view_layers == o.view_layers &&
           base_mip == o.base_mip && view_mips == o.view_mips &&
           sampler == o.sampler && arrayed == o.arrayed;
  }
};

struct TexKeyHash {
  size_t operator()(const TexKey& k) const {
    u64 h = TexImageKeyHash{}(k.image);
    h = HashWord(h, k.base_array);
    h = HashWord(h, k.view_layers);
    h = HashWord(h, k.base_mip);
    h = HashWord(h, k.view_mips);
    h = HashWord(h, k.swizzle);
    h = HashWord(h, SamplerKeyHash{}(k.sampler));
    return static_cast<size_t>(HashWord(h, k.arrayed));
  }
};

struct TexViewKey {
  TexImageKey image;
  u32 base_array = 0, view_layers = 1;
  u32 base_mip = 0, view_mips = 1;
  u32 swizzle = 0;  // packed T# DST_SEL_X/Y/Z/W
  bool arrayed = false;
  bool operator==(const TexViewKey& o) const {
    return swizzle == o.swizzle && image == o.image &&
           base_array == o.base_array && view_layers == o.view_layers &&
           base_mip == o.base_mip && view_mips == o.view_mips &&
           arrayed == o.arrayed;
  }
};

struct TexViewKeyHash {
  size_t operator()(const TexViewKey& k) const {
    u64 h = TexImageKeyHash{}(k.image);
    h = HashWord(h, k.base_array);
    h = HashWord(h, k.view_layers);
    h = HashWord(h, k.base_mip);
    h = HashWord(h, k.view_mips);
    h = HashWord(h, k.swizzle);
    return static_cast<size_t>(HashWord(h, k.arrayed));
  }
};

struct TexImageEntry {
  rhi::Texture* image = nullptr;
  u64 footprint = 0;
  u64 allocation_size = 0;
  u64 hash = 0;         // whole-content, from the periodic sweep
  u64 sample_hash = 0;  // 16 KB of windows, checked every frame
  int last_checked_frame = -1;
  int last_full_frame = -1;
  int last_used_frame = -1;
  // Frames between full sweeps. Doubles while the content holds still and snaps
  // back to 1 the moment it moves: reading every bound texture in full every
  // frame was 269 MB and 12 ms of a Dead Cells frame, and none of it changed.
  u32 check_interval = 1;
  // False while `hash` predates the image's current contents. A refresh already
  // reads the whole surface to upload it; hashing it again in the same breath
  // is a second cold pass over tens of megabytes, and the value is only ever
  // needed by the next full sweep, which recomputes it anyway.
  bool hash_valid = true;
  // Revision of the compute range the image was last copied from
  // (CsSupplyTexture); 0 when it holds guest memory.
  u64 cs_seq = 0;
  // This frame's validation, computed ahead on the worker pool for every image
  // the last frame used (PrevalidateTextures); pre_frame says for which frame.
  int pre_frame = -1;
  bool pre_readable = false;
  bool pre_full = false;  // pre_hash holds a whole-content hash
  u64 pre_sample = 0;
  u64 pre_hash = 0;
};

struct TexViewEntry {
  rhi::TextureView* view = nullptr;
};
struct TexEntry {
  rhi::BindGroup* set = nullptr;
};

base::HashMap<TexImageKey, TexImageEntry, TexImageKeyHash> g_tex_images;
base::HashMap<TexViewKey, TexViewEntry, TexViewKeyHash> g_tex_views;
base::HashMap<TexKey, TexEntry, TexKeyHash> g_tex_cache;
base::HashMap<SamplerKey, rhi::Sampler*, SamplerKeyHash> g_sampler_cache;
constexpr u32 kTexturePageShift = 16;
base::HashMap<u64, base::Vector<TexImageKey>> g_texture_pages;
u64 g_tex_image_bytes = 0;
base::Vector<TexImageEntry> g_retired_tex_images;
base::Vector<TexViewEntry> g_retired_tex_views;
base::Vector<TexEntry> g_retired_tex_sets;

void RegisterTexturePages(const TexImageKey& key, u64 bytes) {
  const u64 end = key.base + bytes;
  for (u64 page = key.base >> kTexturePageShift;
       page <= (end - 1) >> kTexturePageShift; page++)
    g_texture_pages[page].push_back(key);
}

void UnregisterTexturePages(const TexImageKey& key, u64 bytes) {
  const u64 end = key.base + bytes;
  for (u64 page = key.base >> kTexturePageShift;
       page <= (end - 1) >> kTexturePageShift; page++) {
    auto found = g_texture_pages.find(page);
    if (found == g_texture_pages.end())
      continue;
    auto& keys = found->second;
    base::EraseIf(keys, [&](const auto& k) { return k == key; });
    if (keys.empty())
      g_texture_pages.erase(found);
  }
}
TexKey TextureKey(u64 base,
                  u32 w,
                  u32 h,
                  u32 dfmt,
                  u32 nfmt,
                  u32 tiling,
                  u32 pitch,
                  u32 layers,
                  u32 base_array,
                  u32 view_layers,
                  u32 mip_levels,
                  u32 base_mip,
                  u32 view_mips,
                  u32 min_lod,
                  bool pow2_pad,
                  const u32* sampler,
                  bool sampler_valid,
                  bool arrayed,
                  bool force_lod_zero,
                  bool depth_compare,
                  u32 swizzle,
                  u32 depth,
                  bool is_3d) {
  if (is_3d)
    layers = 1;
  else
    depth = 1;
  if (layers && base_array < layers)
    view_layers = base::Min(view_layers, layers - base_array);
  if (!arrayed)
    view_layers = 1;
  if (mip_levels && base_mip < mip_levels)
    view_mips = base::Min(view_mips, mip_levels - base_mip);
  TexKey key;
  key.image = {base,     w,          h,
               tiling,   pitch,      layers,
               depth,    mip_levels, GuestTextureFormat(dfmt, nfmt),
               pow2_pad, is_3d};
  key.base_array = base_array;
  key.view_layers = view_layers;
  key.base_mip = base_mip;
  key.view_mips = view_mips;
  key.swizzle = swizzle;
  key.sampler.valid = sampler_valid && sampler;
  if (key.sampler.valid)
    std::memcpy(key.sampler.raw, sampler, sizeof(key.sampler.raw));
  key.sampler.image_min_lod = min_lod;
  key.sampler.force_lod_zero = force_lod_zero;
  key.sampler.depth_compare = depth_compare;
  // An integer-format view may only be sampled with NEAREST (no format feature
  // for linear filtering); the multi-binding path already keys this, and the
  // single-texture path has to agree or the same descriptor is rejected.
  key.sampler.integer = kIntegerRt && (nfmt == 4 || nfmt == 5);
  key.arrayed = arrayed;
  return key;
}

TexViewKey TextureViewKey(const TexKey& key) {
  return {key.image,     key.base_array, key.view_layers, key.base_mip,
          key.view_mips, key.swizzle,    key.arrayed};
}

u32 TextureTiling(u32 tiling) {
  return kForceTile >= 0 ? static_cast<u32>(kForceTile.get()) : tiling;
}

bool UploadTexPixelsImmediate(rhi::Texture* img,
                              u64 base,
                              const gcn::TextureLayout32& layout,
                              u32 texel_w,
                              u32 texel_h,
                              bool is_3d = false);  // defined below
void ClearMultiTexCache();

// Put the 1x1 depth default into kDepthRead holding 1.0.
void ClearDepthDefaultToFar() {
  rhi::CommandList* list = BeginImmediate();
  if (!list)
    return;
  rhi::TextureRange range;
  range.aspect = rhi::kAspectDepth;
  rhi::TextureBarrier b{g_tex.depth_default_img, rhi::TextureState::kUndefined,
                        rhi::TextureState::kCopyDst, range};
  list->Barrier(0, 0, &b, 1);
  list->ClearDepthStencil(g_tex.depth_default_img, rhi::TextureState::kCopyDst,
                          range, 1.0f, 0);
  b.before = rhi::TextureState::kCopyDst;
  b.after = rhi::TextureState::kDepthRead;
  list->Barrier(0, 0, &b, 1);
  EndImmediate(list, "imm.clear_depth");
}

namespace {

rhi::TextureView* MakeView(rhi::Texture* texture,
                           rhi::ViewDim dim,
                           rhi::Format format,
                           bool zero = false) {
  rhi::TextureViewDesc desc;
  desc.dim = dim;
  desc.format = format;
  if (zero)
    for (auto& c : desc.swizzle)
      c = rhi::Swizzle::kZero;
  return Device().CreateView(texture, desc);
}

}  // namespace

rhi::BindGroup* SampledTextureGroup(rhi::TextureView* view,
                                    rhi::Sampler* sampler,
                                    rhi::TextureState state) {
  rhi::BindGroupDesc desc;
  desc.layout = g_tex.layout;
  rhi::BindingWrite w;
  w.view = view;
  w.sampler = sampler;
  w.view_state = state;
  desc.writes.push_back(w);
  return Device().CreateBindGroup(desc);
}

bool CreateTextureDescriptors() {
  if (g_tex.descriptors_ready)
    return true;
  if (g_tex.layout)
    return false;
  rhi::Device& device = Device();
  // Binding 0 = combined image sampler. Both stages see it: a vertex texture
  // fetch takes its own binding out of the same set.
  rhi::BindGroupLayoutDesc layout;
  layout.bindings.push_back({0, rhi::BindingType::kSampledTexture,
                             rhi::kStageFragment | rhi::kStageVertex});
  g_tex.layout = device.CreateBindGroupLayout(layout);
  rhi::SamplerDesc sd;
  g_tex.sampler = device.CreateSampler(sd);
  sd.mag = sd.min = rhi::Filter::kNearest;
  g_tex.sampler_nearest = device.CreateSampler(sd);
  if (!g_tex.layout || !g_tex.sampler || !g_tex.sampler_nearest)
    return false;

  // 1x1 white default texture (for unresolved sampler bindings). Mutable so
  // the same texel can also be viewed as RGBA8_UINT: a binding the module
  // declared as an integer sampler may not take a UNORM view, and an
  // unresolved one still has to get a default.
  rhi::TextureDesc white;
  white.format = rhi::Format::kRGBA8Unorm;
  white.usage =
      rhi::kTextureSampled | rhi::kTextureCopyDst | rhi::kTextureMutableFormat;
  white.name = "white default";
  g_tex.white_img = device.CreateTexture(white);
  // Same default for a Dim3D binding, which cannot sample a 2D view.
  white.dim = rhi::TextureDim::k3D;
  white.name = "white default 3d";
  g_tex.white_3d_img = device.CreateTexture(white);
  if (!g_tex.white_img || !g_tex.white_3d_img)
    return false;
  u32 texel = 0xFFFFFFFFu;
  gcn::TextureLayout32 white_layout;
  gcn::BuildTextureLayout32(white_layout, 1, 1, 1, 1, 1, 31, false);
  if (!UploadTexPixelsImmediate(g_tex.white_img, reinterpret_cast<u64>(&texel),
                                white_layout, 1, 1) ||
      !UploadTexPixelsImmediate(g_tex.white_3d_img,
                                reinterpret_cast<u64>(&texel), white_layout, 1,
                                1, true))
    return false;
  // 1x1 D32 "far plane" for a compare sample that cannot resolve to a real
  // depth surface (see TextureBindings::depth_default_view).
  rhi::TextureDesc depth;
  depth.format = rhi::Format::kD32Float;
  depth.usage =
      rhi::kTextureSampled | rhi::kTextureDepthTarget | rhi::kTextureCopyDst;
  depth.name = "depth default";
  g_tex.depth_default_img = device.CreateTexture(depth);
  if (g_tex.depth_default_img) {
    rhi::TextureViewDesc dv;
    dv.aspect = rhi::kAspectDepth;
    g_tex.depth_default_view = device.CreateView(g_tex.depth_default_img, dv);
    if (g_tex.depth_default_view)
      ClearDepthDefaultToFar();
  }
  using rhi::Format;
  using rhi::ViewDim;
  g_tex.white_view =
      MakeView(g_tex.white_img, ViewDim::k2D, Format::kRGBA8Unorm);
  g_tex.white_uint_view =
      MakeView(g_tex.white_img, ViewDim::k2D, Format::kRGBA8Uint);
  g_tex.white_uint_array_view =
      MakeView(g_tex.white_img, ViewDim::k2DArray, Format::kRGBA8Uint);
  g_tex.white_array_view =
      MakeView(g_tex.white_img, ViewDim::k2DArray, Format::kRGBA8Unorm);
  g_tex.zero_view =
      MakeView(g_tex.white_img, ViewDim::k2D, Format::kRGBA8Unorm, true);
  g_tex.zero_array_view =
      MakeView(g_tex.white_img, ViewDim::k2DArray, Format::kRGBA8Unorm, true);
  g_tex.zero_3d_view =
      MakeView(g_tex.white_3d_img, ViewDim::k3D, Format::kRGBA8Unorm, true);
  g_tex.white_3d_view =
      MakeView(g_tex.white_3d_img, ViewDim::k3D, Format::kRGBA8Unorm);
  g_tex.white_uint_3d_view =
      MakeView(g_tex.white_3d_img, ViewDim::k3D, Format::kRGBA8Uint);
  if (!g_tex.white_view || !g_tex.white_uint_view ||
      !g_tex.white_uint_array_view || !g_tex.white_array_view ||
      !g_tex.zero_view || !g_tex.zero_array_view || !g_tex.zero_3d_view ||
      !g_tex.white_3d_view || !g_tex.white_uint_3d_view)
    return false;
  g_tex.white_set = SampledTextureGroup(g_tex.white_view, g_tex.sampler);
  g_tex.white_array_set =
      SampledTextureGroup(g_tex.white_array_view, g_tex.sampler);
  g_tex.zero_set = SampledTextureGroup(g_tex.zero_view, g_tex.sampler);
  g_tex.zero_array_set =
      SampledTextureGroup(g_tex.zero_array_view, g_tex.sampler);
  g_tex.white_3d_set = SampledTextureGroup(g_tex.white_3d_view, g_tex.sampler);
  g_tex.zero_3d_set = SampledTextureGroup(g_tex.zero_3d_view, g_tex.sampler);
  g_tex.descriptors_ready = g_tex.white_set && g_tex.white_array_set &&
                            g_tex.zero_set && g_tex.zero_array_set &&
                            g_tex.white_3d_set && g_tex.zero_3d_set;
  return g_tex.descriptors_ready;
}

rhi::Sampler* SamplerFor(const SamplerKey& key) {
  // DELTA_GPU_DEFSAMPLER: ignore every guest S# and use the default sampler,
  // to tell a mis-decoded sampler apart from a mis-bound image.
  // The default sampler is LINEAR, and an integer-format view may not be
  // filtered at all, so every exit that hands out a default has to hand out
  // the NEAREST one for those. An unresolved S# over an integer view taking the
  // LINEAR default was the overwhelming majority of GTA:SA's magFilter-04553,
  // 200k messages in a 200 s run.
  const auto fallback = [&] {
    return key.integer ? g_tex.sampler_nearest : g_tex.sampler;
  };
  if (kDefaultSampler)
    return fallback();
  if (!key.valid && !key.force_lod_zero && !key.depth_compare)
    return fallback();
  auto found = g_sampler_cache.find(key);
  if (found != g_sampler_cache.end())
    return found->second;
  if (g_sampler_cache.size() >= 4096)
    return fallback();

  auto address_mode = [](u32 mode) {
    switch (mode & 7) {
      case 0:
        return rhi::AddressMode::kRepeat;
      case 1:
        return rhi::AddressMode::kMirroredRepeat;
      case 3:
      case 5:
      case 7:
        return rhi::AddressMode::kMirrorClampToEdge;
      case 4:
      case 6:
        return rhi::AddressMode::kClampToBorder;
      default:
        return rhi::AddressMode::kClampToEdge;
    }
  };
  rhi::SamplerDesc ci;
  ci.address_u = address_mode(key.raw[0]);
  ci.address_v = address_mode(key.raw[0] >> 3);
  ci.address_w = address_mode(key.raw[0] >> 6);
  // DELTA_GPU_FORCENEAREST=1: diagnostic only. Force point sampling everywhere,
  // to separate "this pass reads the texel it addresses" from "its 2x2 bilinear
  // footprint straddles a neighbour". A pass that thresholds on what it samples
  // behaves completely differently under the two, and nothing else
  // distinguishes them from outside the shader.
  u32 mag = kForceNearest ? 0u : (key.raw[2] >> 20) & 3;
  u32 min = kForceNearest ? 0u : (key.raw[2] >> 22) & 3;
  ci.mag =
      (mag & 1) && !key.integer ? rhi::Filter::kLinear : rhi::Filter::kNearest;
  ci.min =
      (min & 1) && !key.integer ? rhi::Filter::kLinear : rhi::Filter::kNearest;
  u32 mip_filter = (key.raw[2] >> 26) & 3;
  ci.mip = mip_filter == 2 && !key.integer ? rhi::Filter::kLinear
                                           : rhi::Filter::kNearest;
  ci.min_lod = ci.max_lod = 0.0f;
  if (mip_filter) {
    u32 min_lod = base::Max(key.raw[1] & 0xFFF, key.image_min_lod);
    ci.min_lod = static_cast<float>(min_lod) / 256.0f;
    ci.max_lod = static_cast<float>((key.raw[1] >> 12) & 0xFFF) / 256.0f;
    ci.max_lod = base::Max(ci.min_lod, ci.max_lod);
  }
  if (key.force_lod_zero)
    ci.min_lod = ci.max_lod = 0.0f;
  if (kForceLod >= 0)
    ci.min_lod = ci.max_lod = static_cast<float>(kForceLod.get());
  i32 bias = static_cast<i32>(key.raw[2] << 18) >> 18;
  ci.lod_bias = static_cast<float>(bias) / 256.0f;
  switch ((key.raw[3] >> 30) & 3) {
    case 1:
      ci.border = rhi::BorderColor::kOpaqueBlack;
      break;
    case 2:
      ci.border = rhi::BorderColor::kOpaqueWhite;
      break;
    default:
      ci.border = rhi::BorderColor::kTransparentBlack;
      break;
  }
  ci.compare_enable = key.depth_compare;
  // DEPTH_COMPARE_FUNC uses the same order as rhi::CompareOp.
  ci.compare = static_cast<rhi::CompareOp>((key.raw[0] >> 12) & 7);
  // Anisotropy requires LINEAR on both filters, which an integer format has
  // just been denied.
  if (!key.integer && (mag >= 2 || min >= 2))
    ci.max_anisotropy = static_cast<float>(1u << ((key.raw[0] >> 9) & 7));
  rhi::Sampler* sampler = Device().CreateSampler(ci);
  if (!sampler)
    return fallback();
  g_sampler_cache.emplace(key, sampler);
  return sampler;
}

u64 TextureLinearBytes(const gcn::TextureLayout32& layout) {
  u64 bytes = 0;
  for (u32 mip = 0; mip < layout.mip_levels; mip++)
    bytes += static_cast<u64>(layout.mips[mip].width) *
             layout.mips[mip].height * layout.layers * layout.elem_bytes;
  return bytes;
}

void PackTexPixels(u8* linear,
                   u64 buffer_offset,
                   u64 base,
                   const gcn::TextureLayout32& layout,
                   u32 texel_w,
                   u32 texel_h,
                   rhi::BufferTextureCopy* copies,
                   bool is_3d) {
  const u32 elem = layout.elem_bytes;
  const u8* src = reinterpret_cast<const u8*>(base);
  u64 linear_offset = 0;
  for (u32 mip = 0; mip < layout.mip_levels; mip++) {
    const auto& level = layout.mips[mip];
    rhi::BufferTextureCopy& copy = copies[mip];
    copy = {};
    copy.buffer_offset = buffer_offset + linear_offset;
    // A volume's slices are the layout's layers, but the copy takes them as
    // `depth` z-slices of a single-layer image.
    copy.region.mip = mip;
    copy.region.layers = is_3d ? 1u : layout.layers;
    copy.region.width = base::Max(texel_w >> mip, 1u);
    copy.region.height = base::Max(texel_h >> mip, 1u);
    copy.region.depth = is_3d ? layout.layers : 1u;
    const u64 layer_bytes = static_cast<u64>(level.width) * level.height * elem;
    for (u32 layer = 0; layer < layout.layers; layer++) {
      u8* dst = linear + linear_offset + layer * layer_bytes;
      if (!kNoDetile) {
        gcn::DetileTextureMip32(src, dst, layout, mip, layer);
      } else {
        const u8* level_src =
            src + level.offset +
            static_cast<u64>(layer) * level.pitch * level.stored_height * elem;
        for (u32 y = 0; y < level.height; y++)
          std::memcpy(dst + static_cast<size_t>(y) * level.width * elem,
                      level_src + static_cast<size_t>(y) * level.pitch * elem,
                      static_cast<size_t>(level.width) * elem);
      }
    }
    linear_offset += layer_bytes * layout.layers;
  }
  // DELTA_GPU_TEXDUMP_BASE=<guest addr>: write the post-detile mip 0 of ONE
  // named texture to <dumpdir>/tex_<addr>.bin, whatever its size or format.
  // The size-capped text dump below cannot reach a 2048x2048 compressed
  // atlas, and a light cookie's decoded ALPHA is what shapes P.T.'s light
  // pools. There is no way to tell a ragged cookie from a ragged decode of
  // a smooth one without looking at the texels.
  // Dump the LAST upload seen, not the first: a once-only guard reports the
  // state at first sample, which for a surface the title fills later is zero
  // and says nothing about what the shader eventually reads.
  if (kDumpBase && base == kDumpBase.get()) {
    {
      char dp[256];
      std::snprintf(dp, sizeof(dp), "%s/tex_%lx.bin", DumpDir(),
                    (unsigned long)base);
      if (FILE* df = std::fopen(dp, "wb")) {
        std::fwrite(linear, 1,
                    static_cast<size_t>(layout.mips[0].width) *
                        layout.mips[0].height * elem,
                    df);
        std::fclose(df);
      }
      BASE_LOGI("texdumpbase",
                "{:#x} mip0 {}x{} texel={}x{} elem={} "
                "tiling={} mips={} -> {}",
                (unsigned long)base, layout.mips[0].width,
                layout.mips[0].height, texel_w, texel_h, elem,
                layout.tiling_idx, layout.mip_levels, dp);
    }
  }
  // DELTA_GPU_TEXDUMP_UPLOAD: dump the post-detile pixels that Vulkan
  // receives. This differs from DELTA_GPU_TEXDUMP, which inspects raw guest
  // memory and is intentionally useful for spotting tiling rather than
  // validating the detiler.
  // DELTA_GPU_TEXDUMP_MIN: smallest edge to dump (default 128, i.e. content
  // atlases only). Lower it to reach the tiny nine-slice UI surfaces, which are
  // a handful of texels each and are best read as text.
  static int dump_small_count = 0;
  if (kDumpUpload && dump_small_count < 24 && elem == 4 && texel_w <= 32 &&
      texel_h <= 32 && texel_w >= kDumpMin && texel_h >= kDumpMin) {
    dump_small_count++;
    base::String line;
    base::FormatTo(line,
                   "base={:#x} {}x{} layers={} tiling={} pitch={} "
                   "stored_h={} raw:",
                   (unsigned long)base, texel_w, texel_h, layout.layers,
                   layout.tiling_idx, layout.mips[0].pitch,
                   layout.mips[0].stored_height);
    base::FormatTo(line, "\n");
    char rp[256];
    std::snprintf(rp, sizeof(rp), "%s/rawblk_%02d_%ux%u.bin", DumpDir(),
                  dump_small_count - 1, texel_w, texel_h);
    if (FILE* rf = std::fopen(rp, "wb")) {
      std::fwrite(src, 1,
                  static_cast<size_t>(layout.mips[0].pitch) *
                      layout.mips[0].stored_height * elem,
                  rf);
      std::fclose(rf);
    }
    for (u32 y = 0; y < texel_h; y++) {
      base::FormatTo(line, "  ");
      for (u32 x = 0; x < texel_w; x++) {
        const u8* p = linear + (static_cast<u64>(y) * texel_w + x) * 4;
        base::FormatTo(line, "{:02x}{:02x}{:02x}{:02x} ", p[0], p[1], p[2],
                       p[3]);
      }
      base::FormatTo(line, "\n");
    }
    BASE_LOGI("texsmall", "{}", line.c_str());
  }
  static int dump_upload_count = 0;
  if (kDumpUpload && dump_upload_count < 32 && elem == 4 && texel_w >= 128 &&
      texel_h >= 128) {
    // Every layer, so an array/cube surface can be inspected face by face.
    for (u32 l = 1; l < layout.layers && l < 8; l++) {
      char lp[256];
      std::snprintf(lp, sizeof(lp), "%s/tex_layer%u_%02d_%#lx_%ux%u.ppm",
                    DumpDir(), l, dump_upload_count, (unsigned long)base,
                    texel_w, texel_h);
      if (FILE* lf = std::fopen(lp, "wb")) {
        std::fprintf(lf, "P6\n%u %u\n255\n", texel_w, texel_h);
        const u8* lp8 = linear + static_cast<u64>(l) * texel_w * texel_h * 4;
        for (u64 i = 0; i < static_cast<u64>(texel_w) * texel_h; i++) {
          std::fputc(lp8[i * 4], lf);
          std::fputc(lp8[i * 4 + 1], lf);
          std::fputc(lp8[i * 4 + 2], lf);
        }
        std::fclose(lf);
      }
    }
    u64 rgb_nonzero = 0, alpha_nonzero = 0;
    const u64 pixels = static_cast<u64>(texel_w) * texel_h;
    for (u64 i = 0; i < pixels; i++) {
      const u8* p = linear + i * 4;
      if (p[0] || p[1] || p[2])
        rgb_nonzero++;
      if (p[3])
        alpha_nonzero++;
    }
    char path[256];
    std::snprintf(path, sizeof(path), "%s/tex_upload_%02d_%#lx_%ux%u.ppm",
                  DumpDir(), dump_upload_count, (unsigned long)base, texel_w,
                  texel_h);
    FILE* file = std::fopen(path, "wb");
    if (file) {
      std::fprintf(file, "P6\n%u %u\n255\n", texel_w, texel_h);
      for (u64 i = 0; i < pixels; i++) {
        const u8* p = linear + i * 4;
        std::fputc(p[0], file);
        std::fputc(p[1], file);
        std::fputc(p[2], file);
      }
      std::fclose(file);
    }
    if (texel_w == 1024 && texel_h == 2048) {
      char alpha_path[256];
      std::snprintf(alpha_path, sizeof(alpha_path),
                    "%s/tex_upload_alpha_%#lx.pgm", DumpDir(),
                    (unsigned long)base);
      if (FILE* alpha = std::fopen(alpha_path, "wb")) {
        std::fprintf(alpha, "P5\n%u %u\n255\n", texel_w, texel_h);
        for (u64 i = 0; i < pixels; i++)
          std::fputc(linear[i * 4 + 3], alpha);
        std::fclose(alpha);
      }
    }
    BASE_LOGI("texupload", "{} base={:#x} {}x{} rgb={} alpha={}/{} -> {}",
              dump_upload_count, (unsigned long)base, texel_w, texel_h,
              (unsigned long)rgb_nonzero, (unsigned long)alpha_nonzero,
              (unsigned long)pixels, path);
    dump_upload_count++;
  }
}

// One-time initialization upload used before frame recording starts.
bool UploadTexPixelsImmediate(rhi::Texture* img,
                              u64 base,
                              const gcn::TextureLayout32& layout,
                              u32 texel_w,
                              u32 texel_h,
                              bool is_3d) {
  DELTA_ZONE("gpu.tex_upload_immediate");
  const u64 sz = TextureLinearBytes(layout);
  const u32 barrier_layers = is_3d ? 1 : layout.layers;
  rhi::BufferDesc desc;
  desc.size = sz;
  desc.usage = rhi::kBufferCopySrc;
  desc.memory = rhi::MemoryKind::kUpload;
  rhi::Buffer* staging = Device().CreateBuffer(desc);
  if (!staging)
    return false;
  rhi::BufferTextureCopy copies[16]{};
  PackTexPixels(staging->mapped(), 0, base, layout, texel_w, texel_h, copies,
                is_3d);
  rhi::CommandList* list = BeginImmediate();
  if (!list) {
    Device().Destroy(staging);
    return false;
  }
  rhi::TextureRange range;
  range.mips = layout.mip_levels;
  range.layers = barrier_layers;
  rhi::TextureBarrier b{img, rhi::TextureState::kUndefined,
                        rhi::TextureState::kCopyDst, range};
  list->Barrier(rhi::kAccessHostWrite, rhi::kAccessCopyRead, &b, 1);
  list->CopyBufferToTexture(img, staging, copies, layout.mip_levels);
  b.before = rhi::TextureState::kCopyDst;
  b.after = rhi::TextureState::kShaderRead;
  list->Barrier(0, 0, &b, 1);
  const u64 t0 = NowNs();
  const bool ok = EndImmediate(list, "imm.tex_upload");
  if (!ok)
    BASE_LOGI("gpuvk",
              "tex upload DEVICE FAULT: base={:#x} {}x{} mips={} layers={} "
              "bytes={}",
              (unsigned long)base, layout.mips[0].width, layout.mips[0].height,
              layout.mip_levels, layout.layers, (unsigned long long)sz);
  const u64 dt = NowNs() - t0;
  g_ns_tex_up += dt;
  g_fr_tex_up += dt;
  g_tex_ups++;
  Device().Destroy(staging);
  return ok;
}

bool RecordTexPixels(rhi::Texture* img,
                     rhi::TextureState old_state,
                     u64 base,
                     const gcn::TextureLayout32& layout,
                     u32 texel_w,
                     u32 texel_h,
                     bool is_3d) {
  DELTA_ZONE("gpu.tex_upload");
  const u64 bytes = TextureLinearBytes(layout);
  const u32 barrier_layers = is_3d ? 1 : layout.layers;
  TextureUploadSlice upload;
  if (!AllocateTextureUpload(g_frame.slot_idx, bytes,
                             base::Max<u32>(16, layout.elem_bytes), upload))
    return false;
  const u64 start = NowNs();
  rhi::BufferTextureCopy copies[16]{};
  PackTexPixels(upload.map, upload.offset, base, layout, texel_w, texel_h,
                copies, is_3d);
  EndRegion();
  rhi::TextureRange range;
  range.mips = layout.mip_levels;
  range.layers = barrier_layers;
  rhi::TextureBarrier b{img, old_state, rhi::TextureState::kCopyDst, range};
  g_frame.list->Barrier(0, 0, &b, 1);
  g_frame.list->CopyBufferToTexture(img, upload.buffer, copies,
                                    layout.mip_levels);
  b.before = rhi::TextureState::kCopyDst;
  b.after = rhi::TextureState::kShaderRead;
  g_frame.list->Barrier(0, 0, &b, 1);
  const u64 elapsed = NowNs() - start;
  g_ns_tex_up += elapsed;
  g_fr_tex_up += elapsed;
  g_tex_ups++;
  return true;
}

using TexImageKeySet = base::HashSet<TexImageKey, TexImageKeyHash>;

// One pass over the set and view caches for a whole batch of images: walking
// both caches once per image was most of what eviction cost.
void RetireTextureImages(const TexImageKeySet& keys) {
  if (keys.empty())
    return;
  BumpTextureEpoch();
  for (auto set = g_tex_cache.begin(); set != g_tex_cache.end();) {
    if (keys.count(set->first.image)) {
      g_retired_tex_sets.push_back(set->second);
      set = g_tex_cache.erase(set);
    } else {
      ++set;
    }
  }
  for (auto view = g_tex_views.begin(); view != g_tex_views.end();) {
    if (keys.count(view->first.image)) {
      g_retired_tex_views.push_back(view->second);
      view = g_tex_views.erase(view);
    } else {
      ++view;
    }
  }
  for (const TexImageKey& key : keys) {
    auto image = g_tex_images.find(key);
    if (image == g_tex_images.end())
      continue;
    UnregisterTexturePages(key, image->second.footprint);
    GPU_BUGCHECK(g_tex_image_bytes >= image->second.allocation_size,
                 "texture budget underflow: %llu live < %llu retiring",
                 (unsigned long long)g_tex_image_bytes,
                 (unsigned long long)image->second.allocation_size);
    g_tex_image_bytes -= image->second.allocation_size;
    g_retired_tex_images.push_back(image->second);
    g_tex_images.erase(image);
  }
  ClearMultiTexCache();
}

// Make room for one more image of `bytes` under `budget`: retire the least
// recently used images not bound this frame, in one batch, down to 1/16 of
// the budget and of the image cap below the limit, so the next uploads do not
// come straight back here. False when nothing can be evicted.
constexpr size_t kMaxTextureImages = 3000;

bool EvictTextures(u64 bytes, u64 budget) {
  base::Vector<base::Pair<int, TexImageKey>> old;
  for (const auto& [key, entry] : g_tex_images)
    if (entry.last_used_frame != g_frame.num)
      old.push_back({entry.last_used_frame, key});
  if (old.empty())
    return false;
  base::Sort(old.begin(), old.end(),
             [](const auto& a, const auto& b) { return a.first < b.first; });
  const u64 want_bytes = budget - base::Min(budget, budget / 16 + bytes);
  const size_t want_count = kMaxTextureImages - kMaxTextureImages / 16;
  u64 live_bytes = g_tex_image_bytes;
  size_t live_count = g_tex_images.size();
  TexImageKeySet retire;
  for (const auto& [frame, key] : old) {
    if (live_bytes <= want_bytes && live_count <= want_count)
      break;
    live_bytes -= g_tex_images[key].allocation_size;
    live_count--;
    retire.insert(key);
  }
  RetireTextureImages(retire);
  return true;
}

// The once-a-frame validation of every image last frame used, as one parallel
// pass instead of ~1.4k serial cold reads as the draws reach them. It runs at
// the frame's first texture resolve, a little earlier than the checks it
// replaces, which are themselves once a frame: a write after it is seen next
// frame, as one after the first use always was.
void PrevalidateTextures() {
  DELTA_ZONE("gpu.tex_prevalidate");
  struct Job {
    TexImageEntry* entry;
    u64 base;
    bool full;
  };
  static base::Vector<Job> jobs;
  jobs.clear();
  u64 bytes = 0;
  for (auto& [key, e] : g_tex_images) {
    if (e.last_used_frame != g_frame.num - 1 ||
        e.last_checked_frame == g_frame.num)
      continue;
    const bool full =
        g_frame.num - e.last_full_frame >= static_cast<int>(e.check_interval);
    jobs.push_back({&e, key.base, full});
    bytes += full ? e.footprint : base::Min<u64>(e.footprint, 16384);
  }
  if (jobs.empty())
    return;
  gcn::DetileParallelWork(
      static_cast<u32>(jobs.size()), bytes, [](u32 first, u32 end) {
        for (u32 i = first; i < end; i++) {
          const Job& job = jobs[i];
          TexImageEntry& e = *job.entry;
          e.pre_readable = gpu::IsReadableRange(job.base, e.footprint);
          e.pre_full = e.pre_readable && job.full;
          if (e.pre_readable)
            e.pre_sample = TexSampleHash(job.base, e.footprint);
          if (e.pre_full)
            e.pre_hash = TexHashSerial(job.base, e.footprint);
        }
      });
  for (const Job& job : jobs)
    job.entry->pre_frame = g_frame.num;
}

// Uploads/validates the image and returns its view for this binding, or null.
// `key_out` receives the binding's cache key.
static rhi::TextureView* ResolveTextureView(u64 base,
                                            u32 w,
                                            u32 h,
                                            u32 dfmt,
                                            u32 nfmt,
                                            u32 tiling,
                                            u32 pitch,
                                            u32 layers,
                                            u32 base_array,
                                            u32 view_layers,
                                            u32 mip_levels,
                                            u32 base_mip,
                                            u32 view_mips,
                                            u32 min_lod,
                                            bool pow2_pad,
                                            const u32* sampler,
                                            bool sampler_valid,
                                            bool arrayed,
                                            bool force_lod_zero,
                                            bool depth_compare,
                                            u32 swizzle,
                                            u32 depth,
                                            bool is_3d,
                                            TexKey& key_out) {
  ScopeNs lookup_timer(&g_ns_tex_lookup);
  g_tex_lookup_n++;
  constexpr u64 kMaxTextureBytes = 256ull * 1024 * 1024;
  // DELTA_GPU_TEXWATCH=<base>: the head of one texture's guest memory, once per
  // frame it is bound. A surface the guest fills after our first upload reads
  // as its initial contents forever, and no other trace distinguishes that from
  // memory the guest never wrote.
  if (kTexWatch && base == (u64)kTexWatch) {
    static int watched = -1;
    if (watched != g_frame.num && gpu::IsReadableRange(base, 32)) {
      watched = g_frame.num;
      const u8* p = reinterpret_cast<const u8*>(base);
      // Scan the whole surface, not just its head: a grading LUT legitimately
      // starts at black, so a zero prefix says nothing about whether the guest
      // filled it.
      const u64 span = u64(w) * h * depth * 4;
      u64 first_nz = span;
      if (gpu::IsReadableRange(base, span))
        for (u64 i = 0; i < span; i++)
          if (p[i]) {
            first_nz = i;
            break;
          }
      base::String line;
      base::FormatTo(line,
                     "f{} {:#x} {}x{}x{} span={} first_nz={}:", g_frame.num,
                     (unsigned long)base, w, h, depth, (unsigned long)span,
                     first_nz == span ? -1L : (long)first_nz);
      for (u32 i = 0; i < 32; i++)
        base::FormatTo(line, " {:02x}", p[i]);
      BASE_LOGI("texwatch", "{}", line.c_str());
    }
  }
  if (!w || !h || w > 8192 || h > 8192)
    return nullptr;
  rhi::Format format = GuestTextureFormat(dfmt, nfmt);
  if (format == rhi::Format::kUndefined)
    return nullptr;
  if (is_3d) {
    // maxImageDimension3D (commonly 2048) applies to every axis of a volume,
    // well below the 2D limits checked above.
    const u32 max_3d = Device().caps().max_texture_size_3d;
    if (!depth || depth > max_3d || w > max_3d || h > max_3d) {
      if (kTexFail) {
        static int n = 0;
        if (n++ < 12)
          BASE_LOGI("texfail", "volume {:#x} {}x{}x{} (device max {})",
                    (unsigned long)base, w, h, depth, max_3d);
      }
      return nullptr;
    }
    layers = 1;
  } else {
    depth = 1;
  }
  if (!layers || base_array >= layers) {
    if (kTexFail) {
      static int n = 0;
      if (n++ < 12)
        BASE_LOGI("texfail", "layers {:#x} layers={} base_array={}",
                  (unsigned long)base, layers, base_array);
    }
    return nullptr;
  }
  view_layers = base::Min(view_layers, layers - base_array);
  if (!arrayed)
    view_layers = 1;
  if (!view_layers)
    return nullptr;
  if (!mip_levels || base_mip >= mip_levels)
    return nullptr;
  view_mips = base::Min(view_mips, mip_levels - base_mip);
  if (!view_mips)
    return nullptr;
  // Diagnostic override used to identify incorrectly described guest surfaces.
  tiling = TextureTiling(tiling);
  // Block-compressed formats lay out and detile in 4x4-block ("element")
  // space. Mip chains only halve cleanly in block space for power-of-two
  // texel dimensions; decline others (rare) rather than mis-address them.
  const bool bc = GuestFormatBlockCompressed(dfmt);
  const u32 elem_bytes = GuestFormatElemBytes(dfmt);
  const bool gfx10 = tiling >= gcn::kGfx10TilingBase;
  if (bc && mip_levels > 1 && !gfx10 && ((w & (w - 1)) || (h & (h - 1))))
    return nullptr;
  // BCn volume images are an optional Vulkan feature; decline rather than
  // create an image the driver need not support.
  if (bc && is_3d)
    return nullptr;
  const u32 lw = bc ? (w + 3) / 4 : w;
  const u32 lh = bc ? (h + 3) / 4 : h;
  const u32 lpitch = bc ? ((pitch ? pitch : w) + 3) / 4 : (pitch ? pitch : w);
  gcn::TextureLayout32 layout;
  // DELTA_GPU_TEXFAIL: why a guest texture declines to upload. Skyrim's font
  // atlas fell out here and every glyph then sampled the white default, which
  // fills each glyph quad solid instead of masking it.
  // The layout's layer axis is the volume's slice axis.
  if (!gcn::BuildTextureLayout32(layout, lw, lh, lpitch, is_3d ? depth : layers,
                                 mip_levels, tiling, pow2_pad, elem_bytes)) {
    if (kTexFail) {
      static int n = 0;
      if (n++ < 12)
        BASE_LOGI(
            "texfail", "layout {:#x} {}x{} pitch={} tiling={} elem={} mips={}",
            (unsigned long)base, w, h, lpitch, tiling, elem_bytes, mip_levels);
    }
    return nullptr;
  }
  // gfx10 sizes each level in blocks as ceil(blocks / 2^mip), which is the
  // hardware's, but a level holds ceil(texels / 4) blocks of real data: a
  // 1600-wide chain is 13 blocks at 50 texels where halving says 12.
  if (bc && gfx10)
    for (u32 mip = 0; mip < mip_levels; mip++) {
      auto& level = layout.mips[mip];
      level.width = (base::Max(w >> mip, 1u) + 3) / 4;
      level.height = (base::Max(h >> mip, 1u) + 3) / 4;
    }
  u64 footprint = layout.size;
  if (footprint > kMaxTextureBytes) {
    if (kTexFail) {
      static int n = 0;
      if (n++ < 12)
        BASE_LOGI("texfail", "range {:#x} {}x{} footprint={} (max {})",
                  (unsigned long)base, w, h, (unsigned long)footprint,
                  (unsigned long)kMaxTextureBytes);
    }
    return nullptr;
  }
  // DELTA_GPU_TEXCENSUS=<seconds>: every distinct guest surface the GPU
  // samples, with its real footprint and whether that footprint holds any
  // non-zero byte. One watched address answers "is THIS texture empty"; the
  // census answers "which of the title's surfaces are filled and which are
  // not", which is what separates a broken upload from a broken descriptor.
  if (kTexCensus && gpu::IsReadableRange(base, footprint)) {
    struct Cell {
      u32 w, h, dfmt, nfmt, tiling, mips;
      u64 footprint, binds;
    };
    static base::Mutex m;
    static base::Map<u64, Cell> tbl;
    static auto last = base::TimeTicks::Now();
    base::LockGuard<base::Mutex> lk(m);
    Cell& c = tbl[base];
    c = {w, h, dfmt, nfmt, tiling, mip_levels, footprint, c.binds + 1};
    const auto now = base::TimeTicks::Now();
    if ((now - last).InSeconds() >= kTexCensus) {
      last = now;
      BASE_LOGI("texcensus", "{} surfaces", tbl.size());
      for (const auto& kv : tbl) {
        const auto* p = reinterpret_cast<const u64*>(kv.first);
        u64 nz = 0;
        if (gpu::IsReadableRange(kv.first, kv.second.footprint))
          for (u64 i = 0; i < kv.second.footprint / 8; i++)
            if (p[i])
              nz++;
        BASE_LOGI("texcensus",
                  "{:#x} {:5}x{:<5} dfmt={:2} nfmt={} tile={:2} "
                  "mips={:2} bytes={:<9} nzq={} binds={}",
                  (unsigned long)kv.first, kv.second.w, kv.second.h,
                  kv.second.dfmt, kv.second.nfmt, kv.second.tiling,
                  kv.second.mips, (unsigned long)kv.second.footprint,
                  (unsigned long)nz, (unsigned long)kv.second.binds);
      }
      std::fflush(stderr);
    }
  }
  TexKey& key = key_out;
  key = TextureKey(base, w, h, dfmt, nfmt, tiling, pitch, layers, base_array,
                   view_layers, mip_levels, base_mip, view_mips, min_lod,
                   pow2_pad, sampler, sampler_valid, arrayed, force_lod_zero,
                   depth_compare, swizzle, depth, is_3d);
  // DELTA_GPU_TEXRAW: write each large texture's raw tiled footprint, with its
  // layout in the name, so a swizzle can be worked out offline.
  if (kTexRaw && w >= 256 && h >= 128 &&
      gpu::IsReadableRange(base, footprint)) {
    static int rawn = 0;
    static base::HashSet<u64> raw_seen;
    if (rawn < 12 && raw_seen.insert(static_cast<u64>(w) << 32 | h).second) {
      char p[320];
      std::snprintf(p, sizeof(p),
                    "%s/raw_%02d_%ux%u_pitch%u_sh%u_tile%u_e%u.bin", DumpDir(),
                    rawn++, w, h, layout.mips[0].pitch,
                    layout.mips[0].stored_height, tiling, elem_bytes);
      if (FILE* f = std::fopen(p, "wb")) {
        std::fwrite(reinterpret_cast<const void*>(base), 1, footprint, f);
        std::fclose(f);
        BASE_LOGI("texraw", "{} ({} bytes)", p, (unsigned long long)footprint);
      }
    }
  }
  // Diagnostic (DELTA_GPU_TEXDUMP): in deep gameplay, dump the first few large
  // guest textures sampled, so a non-tutorial room's floor texture can be
  // inspected (is it loaded/brown or black/zero?). Counts non-zero pixels too.
  if (kTexDump && g_frame.num > 1200 && w >= 128 && h >= 128) {
    if (!gpu::IsReadableRange(base, footprint))
      return nullptr;
    static int tdn = 0;
    if (tdn < 16) {
      const u32* px = reinterpret_cast<const u32*>(base);
      u64 cnt = (u64)w * h, nz = 0, step = cnt > 8192 ? cnt / 8192 : 1;
      for (u64 i = 0; i < cnt; i += step)
        if (px[i] & 0x00FFFFFFu)
          nz++;
      char p[256];
      std::snprintf(p, sizeof(p), "%s/tex_%02d_%#lx_%ux%u.ppm", DumpDir(), tdn,
                    (unsigned long)base, w, h);
      FILE* f = std::fopen(p, "wb");
      if (f) {
        std::fprintf(f, "P6\n%u %u\n255\n", w, h);
        const u8* b = reinterpret_cast<const u8*>(base);
        for (u64 i = 0; i < cnt; i++) {
          std::fputc(b[i * 4], f);
          std::fputc(b[i * 4 + 1], f);
          std::fputc(b[i * 4 + 2], f);
        }
        std::fclose(f);
      }
      BASE_LOGI("texdump", "{} base={:#x} {}x{} nonzero={}/8192", tdn,
                (unsigned long)base, w, h, (unsigned long)nz);
      tdn++;
    }
  }
  auto image_it = g_tex_images.find(key.image);
  // A dispatch's output for this surface still in VRAM is copied straight into
  // the image; the guest bytes under it are stale and stay that way.
  // A raw range a dispatch wrote through a V# holds the guest layout itself.
  gcn::TextureLayout32 linear;
  const bool cs_image =
      (!is_3d || mip_levels == 1) &&
      CsSupplyTexture(base, layout, w, h, nullptr,
                      rhi::TextureState::kUndefined, nullptr);
  const bool cs_buffer =
      !cs_image && !bc && !is_3d &&
      gcn::BuildTextureLayout32(linear, lw, lh, lpitch, layers, mip_levels, 8,
                                pow2_pad, elem_bytes) &&
      CsSupplyTextureFromBuffer(base, layout, linear, w, h, nullptr,
                                rhi::TextureState::kUndefined, nullptr);
  const bool cs_supplies = cs_image || cs_buffer;
  const auto cs_supply = [&](rhi::Texture* img, rhi::TextureState state,
                             u64* seq) {
    return cs_image ? CsSupplyTexture(base, layout, w, h, img, state, seq)
                    : CsSupplyTextureFromBuffer(base, layout, linear, w, h,
                                                img, state, seq);
  };
  if (cs_supplies && image_it != g_tex_images.end()) {
    if (!cs_supply(image_it->second.image, rhi::TextureState::kShaderRead,
                   &image_it->second.cs_seq))
      return nullptr;
    image_it->second.last_checked_frame = g_frame.num;
    image_it->second.hash_valid = false;
  }
  if (!cs_supplies && !render::FlushCsWritesRange(render::DefaultRenderer(),
                                                  base, footprint, "tex"))
    return nullptr;
  static int prevalidated_frame = -1;
  if (kTexPrevalidate && prevalidated_frame != g_frame.num) {
    prevalidated_frame = g_frame.num;
    PrevalidateTextures();
  }
  if (image_it == g_tex_images.end() ||
      image_it->second.last_checked_frame != g_frame.num) {
    const bool pre = image_it != g_tex_images.end() &&
                     image_it->second.pre_frame == g_frame.num &&
                     image_it->second.footprint == footprint;
    // Mapping probes are syscall-heavy. Perform one alongside the
    // once-per-frame content validation rather than on every draw that reuses
    // this image.
    const u64 t_probe = NowNs();
    const bool readable = pre ? image_it->second.pre_readable
                              : gpu::IsReadableRange(base, footprint);
    g_ns_tex_probe += NowNs() - t_probe;
    g_tex_probe_n++;
    if (!readable) {
      // The commonest way a binding ends up on the white fallback, and until
      // now the only silent one: the descriptor is fine, the memory behind it
      // just is not mapped (yet).
      if (kTexFail) {
        static int n = 0;
        if (n++ < 12)
          BASE_LOGI("texfail",
                    "unmapped {:#x} {}x{}x{} tiling={} mips={} "
                    "footprint={}",
                    (unsigned long)base, w, h, is_3d ? depth : layers, tiling,
                    mip_levels, (unsigned long)footprint);
      }
      return nullptr;
    }
    const u64 t_hash = NowNs();
    // A brand new image takes its reference hash at the upload below.
    if (image_it != g_tex_images.end()) {
      TexImageEntry& e = image_it->second;
      e.last_checked_frame = g_frame.num;
      // Cheap windowed check every frame, whole-content sweep on the backoff.
      // The sweep alone would let a guest CPU write sit unnoticed for as long
      // as the interval (nothing calls InvalidateTexRange for those, only
      // compute writeback does), and the windows alone can miss a small one.
      const u64 sample = pre ? e.pre_sample : TexSampleHash(base, footprint);
      g_tex_hash_bytes += base::Min<u64>(footprint, 16384);
      bool changed = sample != e.sample_hash;
      if (!changed && g_frame.num - e.last_full_frame >=
                          static_cast<int>(e.check_interval)) {
        const u64 hsh =
            pre && e.pre_full ? e.pre_hash : TexHash(base, footprint);
        g_tex_hash_bytes += footprint;
        // With no reference hash the sweep has nothing to compare against, and
        // the windowed check has already said the surface is holding still –
        // so adopt this value as the reference rather than forcing a refresh.
        changed = e.hash_valid && hsh != e.hash;
        e.hash = hsh;
        e.hash_valid = true;
        e.last_full_frame = g_frame.num;
        if (!changed && e.check_interval < kMaxCheckInterval)
          e.check_interval *= 2;
      }
      if (changed) {
        // The reference hash is taken BEFORE the upload. Left to the next
        // sweep, a surface the guest was still writing while we read it (a
        // texture streaming in) got its finished contents adopted as the
        // reference and the half-written upload kept for good. Hashed first,
        // a write that lands after the hash reads as a change next sweep.
        e.hash = TexHash(base, footprint);
        g_tex_hash_bytes += footprint;
        if (!RecordTexPixels(e.image, rhi::TextureState::kShaderRead, base,
                             layout, w, h, is_3d)) {
          if (kTexFail) {
            static int n = 0;
            if (n++ < 12)
              BASE_LOGI("texfail", "refresh {:#x} {}x{}x{} tiling={}",
                        (unsigned long)base, w, h, is_3d ? depth : layers,
                        tiling);
          }
          g_ns_tex_hash += NowNs() - t_hash;
          return nullptr;
        }
        e.hash_valid = true;
        e.sample_hash = sample;
        e.last_full_frame = g_frame.num;
        e.check_interval = 1;
      }
    }
    g_ns_tex_hash += NowNs() - t_hash;
    g_tex_hash_n++;
  }
  if (image_it == g_tex_images.end()) {
    static const u64 kTextureBudget =
        base::Max<u64>(kTextureMb, 64) * 1024 * 1024;
    if (g_tex_images.size() >= kMaxTextureImages &&
        !EvictTextures(0, kTextureBudget))
      return nullptr;
    TexImageEntry image_entry;
    image_entry.footprint = footprint;
    image_entry.hash_valid = false;
    image_entry.sample_hash = TexSampleHash(base, footprint);
    image_entry.last_checked_frame = g_frame.num;
    image_entry.last_full_frame = g_frame.num;
    image_entry.last_used_frame = g_frame.num;
    // Budget on LIVE image bytes. Retired images are only destroyed two
    // BeginFrames later (ReleaseRetiredTextures), so actual allocated memory
    // can transiently exceed the budget by up to two frames of retirements;
    // capping allocated bytes instead would make creation fail outright at
    // the budget edge, since eviction cannot free memory mid-frame.
    const u64 bytes = TextureLinearBytes(layout);
    if (g_tex_image_bytes + bytes > kTextureBudget &&
        (!EvictTextures(bytes, kTextureBudget) ||
         g_tex_image_bytes + bytes > kTextureBudget))
      return nullptr;
    rhi::TextureDesc td;
    td.dim = is_3d ? rhi::TextureDim::k3D : rhi::TextureDim::k2D;
    td.format = (format);
    td.width = w;
    td.height = h;
    td.depth = is_3d ? depth : 1;
    td.layers = layers;
    td.mips = mip_levels;
    td.usage = rhi::kTextureSampled | rhi::kTextureCopyDst;
    image_entry.image = Device().CreateTexture(td);
    if (!image_entry.image)
      return nullptr;
    const bool cs_uploaded =
        cs_supplies && cs_supply(image_entry.image,
                                 rhi::TextureState::kUndefined,
                                 &image_entry.cs_seq);
    if (!cs_uploaded) {  // see the refresh above: hash before the upload
      image_entry.hash = TexHash(base, footprint);
      image_entry.hash_valid = true;
      g_tex_hash_bytes += footprint;
    }
    if (!cs_uploaded &&
        !RecordTexPixels(image_entry.image, rhi::TextureState::kUndefined, base,
                         layout, w, h, is_3d)) {
      if (kTexFail) {
        static int n = 0;
        if (n++ < 12)
          BASE_LOGI("texfail", "upload {:#x} {}x{}x{} tiling={}",
                    (unsigned long)base, w, h, is_3d ? depth : layers, tiling);
      }
      Device().Destroy(image_entry.image);
      return nullptr;
    }
    image_entry.allocation_size = bytes;
    if (Device().caps().debug_labels) {
      char name[96];
      std::snprintf(name, sizeof(name), "tex %#llx %ux%u mips=%u layers=%u",
                    (unsigned long long)base, w, h, mip_levels, layers);
      Device().SetName(image_entry.image, name);
    }
    image_it = g_tex_images.emplace(key.image, image_entry).first;
    g_tex_image_bytes += bytes;
    RegisterTexturePages(key.image, footprint);
  }
  image_it->second.last_used_frame = g_frame.num;

  TexViewKey view_key = TextureViewKey(key);
  auto view_it = g_tex_views.find(view_key);
  if (view_it == g_tex_views.end()) {
    if (g_tex_views.size() >= 12000)
      return nullptr;
    TexViewEntry view_entry;
    rhi::TextureViewDesc vd;
    vd.dim = is_3d     ? rhi::ViewDim::k3D
             : arrayed ? rhi::ViewDim::k2DArray
                       : rhi::ViewDim::k2D;
    vd.base_mip = base_mip;
    vd.mips = view_mips;
    vd.base_layer = is_3d ? 0u : base_array;
    vd.layers = (arrayed && !is_3d) ? view_layers : 1u;
    // T# DST_SEL: 0 = zero, 1 = one, 4..7 = R/G/B/A. A single-channel mask (a
    // font atlas) selects its coverage into the components the shader reads;
    // without this every glyph samples alpha = 1 and fills solid.
    TextureSwizzle(view_key.swizzle, vd.swizzle);
    view_entry.view = Device().CreateView(image_it->second.image, vd);
    if (!view_entry.view)
      return nullptr;
    view_it = g_tex_views.emplace(view_key, view_entry).first;
  }
  return view_it->second.view;
}

// Upload a guest texture (linear 32bpp RGBA) and return a descriptor set bound
// to it. Cached by guest base; re-uploaded when the guest pixels change (the
// room art is composed/loaded into the same buffer after the first sample, so a
// once-only cache would serve a stale black frame).
rhi::BindGroup* GetTexture(u64 base,
                           u32 w,
                           u32 h,
                           u32 dfmt,
                           u32 nfmt,
                           u32 tiling,
                           u32 pitch,
                           u32 layers,
                           u32 base_array,
                           u32 view_layers,
                           u32 mip_levels,
                           u32 base_mip,
                           u32 view_mips,
                           u32 min_lod,
                           bool pow2_pad,
                           const u32* sampler,
                           bool sampler_valid,
                           bool arrayed,
                           bool force_lod_zero,
                           bool depth_compare,
                           u32 swizzle,
                           u32 depth,
                           bool is_3d) {
  TexKey key;
  rhi::TextureView* view = ResolveTextureView(
      base, w, h, dfmt, nfmt, tiling, pitch, layers, base_array, view_layers,
      mip_levels, base_mip, view_mips, min_lod, pow2_pad, sampler,
      sampler_valid, arrayed, force_lod_zero, depth_compare, swizzle, depth,
      is_3d, key);
  if (!view)
    return nullptr;
  auto it = g_tex_cache.find(key);
  if (it != g_tex_cache.end())
    return it->second.set;
  if (g_tex_cache.size() >= 12000)
    return nullptr;

  // See GetMultiTexSet: a guest texture is never a depth format, so a compare
  // sample of one reads the far-plane default rather than an undefined
  // comparison against a colour view.
  const bool cmp_default =
      key.sampler.depth_compare && g_tex.depth_default_view;
  TexEntry e;
  e.set = SampledTextureGroup(cmp_default ? g_tex.depth_default_view : view,
                              SamplerFor(key.sampler),
                              cmp_default ? rhi::TextureState::kDepthRead
                                          : rhi::TextureState::kShaderRead);
  if (!e.set)
    return nullptr;
  g_tex_cache.emplace(key, e);
  return e.set;
}

// Image view for a guest texture (ensures it is cached/uploaded via
// GetTexture).
bool GuestTextureUploadSupported(u32 dfmt, u32 nfmt) {
  // The detiler preserves packed 32-bpp texels; Vulkan applies the descriptor's
  // numeric interpretation when sampling.
  return GuestTextureFormat(dfmt, nfmt) != rhi::Format::kUndefined;
}

static rhi::TextureView* ResolveTexViewUncached(const DrawInfo::DrawTex& t);

namespace {
u64 g_tex_epoch = 1;

// The fields of a binding that decide its view; `src` and the rest do not.
struct TexMemoKey {
  u64 base;
  u32 w, h, dfmt, nfmt, tiling, pitch, layers, base_array, view_layers,
      mip_levels, base_mip, view_mips, min_lod, swizzle, depth;
  u32 sampler[4];
  u8 flags;
  bool operator==(const TexMemoKey&) const = default;
};

TexMemoKey MemoKeyOf(const DrawInfo::DrawTex& t) {
  TexMemoKey k;
  std::memset(&k, 0, sizeof(k));
  k.base = t.base;
  k.w = t.w, k.h = t.h, k.dfmt = t.dfmt, k.nfmt = t.nfmt, k.tiling = t.tiling;
  k.pitch = t.pitch, k.layers = t.layers, k.base_array = t.base_array;
  k.view_layers = t.view_layers, k.mip_levels = t.mip_levels;
  k.base_mip = t.base_mip, k.view_mips = t.view_mips, k.min_lod = t.min_lod;
  k.swizzle = t.swizzle, k.depth = t.depth;
  std::memcpy(k.sampler, t.sampler, sizeof(k.sampler));
  k.flags = u8(t.pow2_pad) | u8(t.sampler_valid) << 1 | u8(t.arrayed) << 2 |
            u8(t.force_lod_zero) << 3 | u8(t.depth_compare) << 4 |
            u8(t.is_3d) << 5;
  return k;
}

// Consecutive draws rebind the same few textures: ~12k lookups a frame over
// ~1.4k images in GTA:SA. A hit skips the key build, the layout, three map
// lookups and the compute checks, which can only change their answer at an
// epoch bump. New frames bump it too, so validation stays once per frame.
struct TexMemo {
  TexMemoKey key;
  u64 epoch = 0;
  rhi::TextureView* view = nullptr;
};
constexpr u32 kTexMemoSlots = 1024;
TexMemo g_tex_memo[kTexMemoSlots];
int g_tex_memo_frame = -1;

u32 TexMemoSlot(const TexMemoKey& k) {
  u64 h = k.base * 0x9E3779B97F4A7C15ull;
  h ^= (u64(k.w) << 32 | k.h) * 0xC2B2AE3D27D4EB4Full;
  h ^= (u64(k.dfmt) << 40 | u64(k.nfmt) << 32 | k.sampler[0]) *
       0x165667B19E3779F9ull;
  h ^= (u64(k.base_mip) << 32 | k.view_mips | u64(k.flags) << 56) ^
       (u64(k.sampler[2]) << 16) ^ k.swizzle;
  return static_cast<u32>(h ^ (h >> 29)) & (kTexMemoSlots - 1);
}
}  // namespace

void BumpTextureEpoch() {
  g_tex_epoch++;
}

u64 TextureEpoch() {
  return g_tex_epoch;
}

rhi::TextureView* TexViewFor(const DrawInfo::DrawTex& t) {
  if (g_tex_memo_frame != g_frame.num) {
    g_tex_memo_frame = g_frame.num;
    g_tex_epoch++;
  }
  const TexMemoKey memo_key = MemoKeyOf(t);
  TexMemo& memo = g_tex_memo[TexMemoSlot(memo_key)];
  if (kTexMemo && memo.epoch == g_tex_epoch && memo.key == memo_key)
    return memo.view;
  rhi::TextureView* view = ResolveTexViewUncached(t);
  // Failures are not remembered: each exit reports itself when asked to.
  if (view)
    memo = {memo_key, g_tex_epoch, view};
  return view;
}

static rhi::TextureView* ResolveTexViewUncached(const DrawInfo::DrawTex& t) {
  // Each exit here leaves the binding on the white fallback, so each one needs
  // to be able to say so (DELTA_GPU_TEXFAIL): an unsupported format and an
  // unmapped surface look identical from the draw side.
  const auto fail = [&](const char* why) {
    if (kTexFail) {
      static int n = 0;
      if (n++ < 16)
        BASE_LOGI("texfail",
                  "{} {:#x} {}x{}x{} dfmt={} nfmt={} tiling={} "
                  "mips={} 3d={} arr={}",
                  why, (unsigned long)t.base, t.w, t.h,
                  t.is_3d ? t.depth : t.layers, t.dfmt, t.nfmt, t.tiling,
                  t.mip_levels, (int)t.is_3d, (int)t.arrayed);
    }
    return static_cast<rhi::TextureView*>(nullptr);
  };
  if (!t.base || !t.w || !t.h)
    return fail("degenerate");
  if (!GuestTextureUploadSupported(t.dfmt, t.nfmt))
    return fail("format");
  TexKey key;
  rhi::TextureView* view = ResolveTextureView(
      t.base, t.w, t.h, t.dfmt, t.nfmt, t.tiling, t.pitch, t.layers,
      t.base_array, t.view_layers, t.mip_levels, t.base_mip, t.view_mips,
      t.min_lod, t.pow2_pad, t.sampler, t.sampler_valid, t.arrayed,
      t.force_lod_zero, t.depth_compare, t.swizzle, t.depth, t.is_3d, key);
  return view ? view : fail("get-texture");
}

// An N-sampler descriptor set (set 0, bindings 0..kMaxTex-1) for a recomp PS
// that samples >1 texture. Cached by the combination of the textures'
// (base,w,h); any binding we cannot resolve gets the 1x1 white default so a
// diffuse*lightmap shader with a missing map shows the diffuse instead of going
// black.
struct MultiTexSet {
  rhi::BindGroup* set = nullptr;
};

// Keyed on exactly what the set's descriptors hold: per binding the view (a
// resolved texture, or the default its declared type calls for), its state and
// the sampler. Only the first num_texs slots are meaningful; the arrays are
// left uninitialised so building a key per draw writes only those.
struct MultiTexKey {
  u32 num_texs = 0;
  rhi::TextureView* view[kMaxTex];
  rhi::Sampler* sampler[kMaxTex];
  rhi::TextureState state[kMaxTex];
  bool operator==(const MultiTexKey& o) const {
    if (num_texs != o.num_texs)
      return false;
    for (u32 i = 0; i < num_texs; i++)
      if (view[i] != o.view[i] || sampler[i] != o.sampler[i] ||
          state[i] != o.state[i])
        return false;
    return true;
  }
};

struct MultiTexKeyHash {
  size_t operator()(const MultiTexKey& k) const {
    u64 h = HashWord(1469598103934665603ull, k.num_texs);
    for (u32 i = 0; i < k.num_texs; i++) {
      h = HashWord(h, reinterpret_cast<u64>(k.view[i]));
      h = HashWord(h, reinterpret_cast<u64>(k.sampler[i]));
      h = HashWord(h, static_cast<u64>(k.state[i]));
    }
    return static_cast<size_t>(h);
  }
};

base::HashMap<MultiTexKey, MultiTexSet, MultiTexKeyHash> g_mtex_cache;
base::Vector<MultiTexSet> g_retired_mtex;
void ClearMultiTexCache() {
  for (const auto& [key, entry] : g_mtex_cache) {
    (void)key;
    if (entry.set)
      g_retired_mtex.push_back(entry);
  }
  g_mtex_cache.clear();
}

void ReleaseRetiredTextures() {
  // Two-stage aging for the pipelined frame: an object retired while frame N
  // recorded may still be referenced by BOTH in-flight command buffers (N-1
  // until N's EndFrame, N until N+1's EndFrame). Objects therefore rest one
  // extra BeginFrame in the `aged` generation before being destroyed, by
  // then every command buffer that could reference them has been fence-waited.
  static base::Vector<MultiTexSet> aged_mtex;
  static base::Vector<TexEntry> aged_tex_sets;
  static base::Vector<TexViewEntry> aged_tex_views;
  static base::Vector<TexImageEntry> aged_tex_images;
  for (const MultiTexSet& entry : aged_mtex)
    Device().Destroy(entry.set);
  for (const TexEntry& e : aged_tex_sets)
    Device().Destroy(e.set);
  for (const TexViewEntry& e : aged_tex_views)
    Device().Destroy(e.view);
  for (const TexImageEntry& e : aged_tex_images)
    Device().Destroy(e.image);
  aged_mtex = base::move(g_retired_mtex);
  aged_tex_sets = base::move(g_retired_tex_sets);
  aged_tex_views = base::move(g_retired_tex_views);
  aged_tex_images = base::move(g_retired_tex_images);
  g_retired_mtex.clear();
  g_retired_tex_sets.clear();
  g_retired_tex_views.clear();
  g_retired_tex_images.clear();
}

rhi::Sampler* SamplerForTex(const DrawInfo::DrawTex& t,
                            rhi::Format view_format) {
  SamplerKey sampler;
  sampler.valid = t.sampler_valid;
  std::memcpy(sampler.raw, t.sampler, sizeof(sampler.raw));
  sampler.image_min_lod = t.min_lod;
  sampler.force_lod_zero = t.force_lod_zero;
  sampler.integer = (kIntegerRt && (t.nfmt == 4 || t.nfmt == 5)) ||
                    IsIntegerColorFormat(view_format);
  return SamplerFor(sampler);
}

rhi::TextureView* DefaultTexView(const DrawInfo::DrawTex& t, bool want_uint) {
  if (t.is_3d)
    return want_uint ? g_tex.white_uint_3d_view : g_tex.white_3d_view;
  if (t.arrayed)
    return want_uint ? g_tex.white_uint_array_view : g_tex.white_array_view;
  return want_uint ? g_tex.white_uint_view : g_tex.white_view;
}

rhi::BindGroup* GetMultiTexSet(const DrawInfo& d,
                               rhi::BindGroupLayout* set_layout,
                               u32 num_bindings,
                               rhi::TextureView* const* resolved_views,
                               const rhi::TextureState* resolved_layouts,
                               const rhi::Format* resolved_formats,
                               const u64* depth_src) {
  ScopeNs set_timer(&g_ns_tex_set);
  g_tex_set_n++;
  // What the draw resolved, and what the layout declares. A shader may declare
  // more samplers than the draw tracked textures for, and a binding left
  // unwritten is read as an undefined descriptor. Validation names it
  // (VUID-vkCmdDrawIndexed-None-08114) and a driver may fault on it. Cover
  // every declared binding; the ones past what resolved take the default.
  const u32 resolved = base::Min(d.num_texs, kMaxTex);
  // A binding past what the draw resolved has no T# to describe it, so the
  // default it takes has to match what the SHADER declared: a 2D default in a
  // binding the module built as a volume is the same undefined read by another
  // name, and so is a UNORM one under an integer sampler.
  const auto declared = [&](u32 i) -> const gcn::ShaderTex* {
    if (!d.recomp)
      return nullptr;
    if (i < d.recomp->ps_texs.size())
      return &d.recomp->ps_texs[i];
    const size_t vs_i = i - d.recomp->ps_texs.size();
    return vs_i < d.recomp->vs_texs.size() ? &d.recomp->vs_texs[vs_i] : nullptr;
  };
  MultiTexKey key;
  key.num_texs = base::Min(base::Max(resolved, num_bindings), kMaxTex);
  // DELTA_GPU_FORCEWHITE: bind the 1x1 white default for every sampler
  // (diagnostic). Doom64's world textures are built by compute dispatches we
  // don't execute, so the atlases are all-zero and the alpha-blended world
  // samples transparent-black (= invisible). Forcing white makes the geometry
  // render opaque, proving the 3D transform/raster/depth path works and
  // isolating the blackness to the texture data.
  rhi::TextureView* views[kMaxTex];
  rhi::TextureState layouts[kMaxTex];
  // DELTA_GPU_TEXMISS: report every sampler binding that falls back to the 1x1
  // white default (the source of "everything renders white" chains) with the
  // descriptor state that failed to resolve.
  static int tex_miss_logged = 0;
  for (u32 i = 0; i < key.num_texs; i++) {
    rhi::TextureView* v =
        (i < resolved && !kForceWhite) ? resolved_views[i] : nullptr;
    bool arrayed = i < resolved && d.texs[i].arrayed;
    bool is_3d =
        i < resolved ? d.texs[i].is_3d : (declared(i) && declared(i)->is_3d);
    if (i >= resolved && declared(i) && declared(i)->storage)
      return nullptr;  // as for an unresolved storage binding below
    if (kTexMiss && !v && i < resolved && tex_miss_logged < 64) {
      tex_miss_logged++;
      const auto& t = d.texs[i];
      // The descriptor's provenance decides what kind of failure this is: a
      // slot that is zero now was never published, one holding a plausible T#
      // means the resolver read the wrong place, and src=0 means the shader
      // took it from inline user data.
      char mem[96] = "";
      if (t.src && gpu::IsReadableRange(t.src, 32)) {
        const auto* w = reinterpret_cast<const u32*>(t.src);
        std::snprintf(mem, sizeof(mem), " [%08x %08x %08x %08x]", w[0], w[1],
                      w[2], w[3]);
      }
      BASE_LOGI("texmiss",
                "ps={:#x} bind={} base={:#x} {}x{} dfmt={} nfmt={} "
                "tiling={} layers={} mips={} arrayed={} 3d={} decl={} uint={} "
                "src={:#x}{}",
                (unsigned long)d.ps_addr, i, (unsigned long)t.base, t.w, t.h,
                t.dfmt, t.nfmt, t.tiling, t.layers, t.mip_levels,
                (int)t.arrayed, (int)t.is_3d, (int)(declared(i) != nullptr),
                (int)(declared(i) && declared(i)->is_uint),
                (unsigned long)t.src, mem);
    }
    if (d.texs[i].storage && !v)
      return nullptr;
    // A binding the module declared with an integer sampled type cannot take
    // the UNORM default: the numeric types have to match
    // (VUID-vkCmdDrawIndexed-format-07753).
    const bool want_uint = declared(i) && declared(i)->is_uint;
    // Shape AND numeric type: a default that matches one but not the other is
    // the same undefined read the resolved case would have been.
    rhi::TextureView* fallback =
        is_3d ? (want_uint ? g_tex.white_uint_3d_view : g_tex.white_3d_view)
        : arrayed
            ? (want_uint ? g_tex.white_uint_array_view : g_tex.white_array_view)
            : (want_uint ? g_tex.white_uint_view : g_tex.white_view);
    views[i] = v ? v : fallback;
    layouts[i] = v ? resolved_layouts[i] : rhi::TextureState::kShaderRead;
  }
  thread_local rhi::BindingWrite writes[kMaxTex];
  for (u32 i = 0; i < key.num_texs; i++) {
    // Past `resolved` there is no T#: d.texs[i] holds whatever the previous
    // draw left there, so nothing about it may be read. Those bindings take a
    // default sampler over the default view.
    const bool have_tex = i < resolved;
    const bool storage_i = have_tex && d.texs[i].storage;
    SamplerKey sampler;
    if (have_tex) {
      sampler.valid = d.texs[i].sampler_valid;
      std::memcpy(sampler.raw, d.texs[i].sampler, sizeof(sampler.raw));
      sampler.image_min_lod = d.texs[i].min_lod;
      sampler.force_lod_zero = d.texs[i].force_lod_zero;
      // A depth comparison is only defined on a view whose format supports
      // it. This binding may have resolved to a colour target instead of the
      // depth surface the guest named, and a compare sampler on that is
      // undefined (VUID-vkCmdDrawIndexed-None-06479): GCN compares against
      // the first component of any format, Vulkan does not. Sample it plainly.
      sampler.depth_compare =
          d.texs[i].depth_compare && depth_src && depth_src[i];
      // The T# says how the SHADER reads the texels; the view says what the
      // hardware may do with them. A binding that resolved to a render target
      // takes that target's format, and an integer one is not filterable at
      // all, because VK_FILTER_LINEAR on it is undefined
      // (VUID-vkCmdDrawIndexed-magFilter-04553), whatever the T# asked for.
      sampler.integer =
          !d.texs[i].storage &&
          ((kIntegerRt && (d.texs[i].nfmt == 4 || d.texs[i].nfmt == 5)) ||
           (resolved_formats && IsIntegerColorFormat(resolved_formats[i])));
    }
    // A compare sample that did not resolve to a depth surface has nothing
    // valid to read: Vulkan defines the comparison only on a format that
    // supports it, and every colour target and guest texture is the wrong kind.
    // Bind the 1x1 far-plane depth default and keep the comparison, which reads
    // as "nothing occludes this" instead of as undefined.
    rhi::BindingWrite& w = writes[i];
    w = {};
    w.binding = i;
    w.view = views[i];
    w.view_state = layouts[i];
    if (have_tex && d.texs[i].depth_compare && !storage_i &&
        !(depth_src && depth_src[i]) && g_tex.depth_default_view) {
      w.view = g_tex.depth_default_view;
      w.view_state = rhi::TextureState::kDepthRead;
      sampler.depth_compare = true;
    }
    if (storage_i)
      w.view_state = rhi::TextureState::kGeneral;
    else
      w.sampler = SamplerFor(sampler);
    key.view[i] = w.view;
    key.sampler[i] = w.sampler;
    key.state[i] = w.view_state;
  }
  auto ci = g_mtex_cache.find(key);
  if (ci != g_mtex_cache.end())
    return ci->second.set;
  if (g_mtex_cache.size() > 3500)
    return nullptr;
  rhi::BindGroupDesc group;
  group.layout = set_layout;
  group.writes.assign(writes, writes + key.num_texs);
  MultiTexSet entry;
  entry.set = Device().CreateBindGroup(group);
  if (!entry.set)
    return nullptr;
  g_mtex_cache[key] = entry;
  return entry.set;
}

// Mark cached textures overlapping a written range for validation. Their next
// use refreshes the existing compatible image in place.
void InvalidateTexRange(u64 base, u64 size) {
  if (!size || base > UINT64_MAX - size)
    return;
  BumpTextureEpoch();
  u64 end = base + size;
  base::Vector<TexImageKey> overlap;
  for (u64 page = base >> kTexturePageShift;
       page <= (end - 1) >> kTexturePageShift; page++) {
    auto found = g_texture_pages.find(page);
    if (found == g_texture_pages.end())
      continue;
    for (const TexImageKey& key : found->second) {
      if (base::Find(overlap.begin(), overlap.end(), key) == overlap.end())
        overlap.push_back(key);
    }
  }
  for (const TexImageKey& key : overlap) {
    auto found = g_tex_images.find(key);
    if (found != g_tex_images.end() && key.base < end &&
        base < key.base + found->second.footprint) {
      found->second.last_checked_frame = -1;
      found->second.last_full_frame = -1;
      found->second.check_interval = 1;
      // Its prevalidated hashes read the bytes from before this change.
      found->second.pre_frame = -1;
    }
  }
}

}  // namespace gpu::render
