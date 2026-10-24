/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/image/texture_cache.h"
#include "base/arch.h"
#include "guest/session.h"
#include "memory_debug/memory_debug.h"

#include "gpu/gcn/gcn_detile.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gpu_check.h"
#include "gpu/gpu_perf.h"
#include "gpu/guest_memory.h"
#include "gpu/render/debug/capture.h"
#include "gpu/render/compute/compute.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "gpu/render/image/guest_format.h"
#include "gpu/render/hash.h"
#include "gpu/render/debug/labels.h"
#include "gpu/render/image/render_target.h"
#include "gpu/render/renderer.h"
#include "gpu/render/memory/upload_ring.h"
#include "gpu/write_tracker.h"

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
#include "base/memory/unique_pointer.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/time/time.h"
#include "options/options.h"
#include "profile/profile.h"

namespace {
DELTA_OPTION(int, kForceTile, "DELTA_GPU_FORCETILE", -1);
DELTA_OPTION(u64, kTextureMb, "DELTA_GPU_TEX_MB", 1024);
DELTA_OPTION(bool, kDefaultSampler, "DELTA_GPU_DEFSAMPLER", false);
DELTA_OPTION(bool, kForceWhite, "DELTA_GPU_FORCEWHITE", false);
DELTA_OPTION(bool, kTexDump, "DELTA_GPU_TEXDUMP", false);
DELTA_OPTION(bool, kForceNearest, "DELTA_GPU_FORCENEAREST", false);
DELTA_OPTION(bool, kTexFail, "DELTA_GPU_TEXFAIL", false);
DELTA_OPTION(bool, kTexMiss, "DELTA_GPU_TEXMISS", false);
DELTA_OPTION(bool, kIntegerRt, "DELTA_GPU_INT_RT", true);
DELTA_OPTION(bool, kTexRaw, "DELTA_GPU_TEXRAW", false);
DELTA_OPTION(u64, kTexWatch, "DELTA_GPU_TEXWATCH", 0);
DELTA_OPTION(int, kTexCensus, "DELTA_GPU_TEXCENSUS", 0);
DELTA_OPTION(int, kForceLod, "DELTA_GPU_FORCELOD", -1);
DELTA_OPTION(bool, kTexMemo, "DELTA_GPU_TEXMEMO", true);
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
  u32 swizzle = 0;
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
  u32 swizzle = 0;
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
  int last_checked_frame = -1;
  int last_used_frame = -1;
  u64 cs_seq = 0;
  u32 tail_first_mip = 0;
  bool tail_clean = false;
  bool wt_clean = false;
  gcn::TextureLayout32 layout;
};

struct TexViewEntry {
  rhi::TextureView* view = nullptr;
};
struct TexEntry {
  rhi::BindGroup* set = nullptr;
};

namespace {

struct TexMemoKey {
  u64 base;
  u32 w, h, dfmt, nfmt, tiling, pitch, layers, base_array, view_layers,
      mip_levels, base_mip, view_mips, min_lod, swizzle, depth;
  u32 sampler[4];
  u8 flags;
  bool operator==(const TexMemoKey&) const = default;
};

struct TexMemo {
  TexMemoKey key;
  u64 epoch = 0;
  rhi::TextureView* view = nullptr;
};
constexpr u32 kTexMemoSlots = 1024;

struct MultiTexSet {
  rhi::BindGroup* set = nullptr;
};

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

}  // namespace

constexpr u32 kTexturePageShift = 16;

// Texture cache state of one guest session; the session reset clears all of it.
struct TextureCacheState {
  base::HashMap<TexImageKey, TexImageEntry, TexImageKeyHash> images;
  base::HashMap<TexViewKey, TexViewEntry, TexViewKeyHash> views;
  base::HashMap<TexKey, TexEntry, TexKeyHash> sets;
  base::HashMap<SamplerKey, rhi::Sampler*, SamplerKeyHash> samplers;
  base::HashMap<u64, base::Vector<TexImageKey>> pages;
  u64 image_bytes = 0;
  base::Vector<TexImageEntry> retired_images;
  base::Vector<TexViewEntry> retired_views;
  base::Vector<TexEntry> retired_sets;
  u64 epoch = 1;
  TexMemo memo[kTexMemoSlots];
  int memo_frame = -1;
  base::HashMap<MultiTexKey, MultiTexSet, MultiTexKeyHash> multi_sets;
  base::Vector<MultiTexSet> retired_multi_sets;
};
TextureCacheState g_texture_cache;

u64 TextureImageBytes() {
  return g_texture_cache.image_bytes;
}

void RegisterTexturePages(const TexImageKey& key, u64 bytes) {
  const u64 end = key.base + bytes;
  for (u64 page = key.base >> kTexturePageShift;
       page <= (end - 1) >> kTexturePageShift; page++)
    g_texture_cache.pages[page].push_back(key);
}

bool Repacked111110(u32 dfmt, u32 nfmt) {
  return dfmt == 7 && nfmt == 7;
}

rhi::Format SampledFormat(u32 dfmt, u32 nfmt) {
  return Repacked111110(dfmt, nfmt) ? rhi::Format::kB10G11R11Float
                                    : GuestTextureFormat(dfmt, nfmt);
}

void UnregisterTexturePages(const TexImageKey& key, u64 bytes) {
  const u64 end = key.base + bytes;
  for (u64 page = key.base >> kTexturePageShift;
       page <= (end - 1) >> kTexturePageShift; page++) {
    auto found = g_texture_cache.pages.find(page);
    if (found == g_texture_cache.pages.end())
      continue;
    auto& keys = found->second;
    base::EraseIf(keys, [&](const auto& k) { return k == key; });
    if (keys.empty())
      g_texture_cache.pages.erase(found);
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
               depth,    mip_levels, SampledFormat(dfmt, nfmt),
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

// IMG_DATA_FORMAT_11_11_10 (R in the low 10 bits, G and B 11 bits above it)
// has no host format. It is uploaded as B10G11R11: R gains a zero mantissa bit,
// B loses its lowest one. Mip 0 only, detiled on the CPU.
bool UploadRepacked111110(rhi::Texture* img,
                          u64 base,
                          const gcn::TextureLayout32& layout,
                          u32 texel_w,
                          u32 texel_h,
                          bool is_3d) {
  const auto& level = layout.mips[0];
  if (layout.elem_bytes != 4 || level.width < texel_w ||
      level.height < texel_h)
    return false;
  const u64 slice = u64(level.width) * level.height * 4;
  rhi::BufferDesc desc;
  desc.size = slice * layout.layers;
  desc.usage = rhi::kBufferCopySrc;
  desc.memory = rhi::MemoryKind::kUpload;
  rhi::Buffer* staging = Device().CreateBuffer(desc);
  if (!staging)
    return false;
  u8* const pixels = staging->mapped();
  for (u32 layer = 0; layer < layout.layers; layer++) {
    u8* const dst = pixels + slice * layer;
    if (!gcn::DetileTextureMip32(reinterpret_cast<const void*>(base), dst,
                                 layout, 0, layer)) {
      Device().Destroy(staging);
      return false;
    }
    for (u64 at = 0; at < slice; at += 4) {
      u32 w;
      std::memcpy(&w, dst + at, 4);
      const u32 r = (w & 0x3FFu) << 1, g = (w >> 10) & 0x7FFu,
                b = (w >> 22) & 0x3FFu;
      w = r | g << 11 | b << 22;
      std::memcpy(dst + at, &w, 4);
    }
  }
  rhi::CommandList* list = BeginImmediate();
  if (!list) {
    Device().Destroy(staging);
    return false;
  }
  rhi::BufferTextureCopy copy;
  copy.row_length = level.width;
  copy.image_height = level.height;
  copy.region.width = texel_w;
  copy.region.height = texel_h;
  copy.region.layers = is_3d ? 1 : layout.layers;
  copy.region.depth = is_3d ? layout.layers : 1;
  rhi::TextureRange range;
  range.layers = is_3d ? 1 : layout.layers;
  rhi::TextureBarrier b{img, rhi::TextureState::kUndefined,
                        rhi::TextureState::kCopyDst, range};
  list->Barrier(rhi::kAccessHostWrite, rhi::kAccessCopyRead, &b, 1);
  list->CopyBufferToTexture(img, staging, &copy, 1);
  b.before = rhi::TextureState::kCopyDst;
  b.after = rhi::TextureState::kShaderRead;
  list->Barrier(0, 0, &b, 1);
  const bool ok = EndImmediate(list, "imm.tex_repack");
  Device().Destroy(staging);
  return ok;
}

bool UploadTexPixelsImmediate(rhi::Texture* img,
                              u64 base,
                              const gcn::TextureLayout32& layout,
                              u32 texel_w,
                              u32 texel_h,
                              bool is_3d = false);
void ClearMultiTexCache();

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

  rhi::TextureDesc white;
  white.format = rhi::Format::kRGBA8Unorm;
  white.usage =
      rhi::kTextureSampled | rhi::kTextureCopyDst | rhi::kTextureMutableFormat;
  white.name = "white default";
  g_tex.white_img = device.CreateTexture(white);
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
  const auto fallback = [&] {
    return key.integer ? g_tex.sampler_nearest : g_tex.sampler;
  };
  if (kDefaultSampler)
    return fallback();
  if (!key.valid && !key.force_lod_zero && !key.depth_compare)
    return fallback();
  auto found = g_texture_cache.samplers.find(key);
  if (found != g_texture_cache.samplers.end())
    return found->second;
  if (g_texture_cache.samplers.size() >= 4096)
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
  ci.compare = static_cast<rhi::CompareOp>((key.raw[0] >> 12) & 7);
  if (!key.integer && (mag >= 2 || min >= 2))
    ci.max_anisotropy = static_cast<float>(1u << ((key.raw[0] >> 9) & 7));
  rhi::Sampler* sampler = Device().CreateSampler(ci);
  if (!sampler)
    return fallback();
  g_texture_cache.samplers.emplace(key, sampler);
  return sampler;
}

u64 TextureLinearBytes(const gcn::TextureLayout32& layout, u32 first_mip = 0) {
  u64 bytes = 0;
  for (u32 mip = first_mip; mip < layout.mip_levels; mip++)
    bytes += static_cast<u64>(layout.mips[mip].width) *
             layout.mips[mip].height * layout.layers * layout.elem_bytes;
  return bytes;
}

bool UploadTexPixelsImmediate(rhi::Texture* img,
                              u64 base,
                              const gcn::TextureLayout32& layout,
                              u32 texel_w,
                              u32 texel_h,
                              bool is_3d) {
  DELTA_ZONE("gpu.tex_upload_immediate");
  const u64 sz = TextureLinearBytes(layout);
  rhi::BufferDesc desc;
  desc.size = sz;
  desc.usage = rhi::kBufferCopySrc;
  desc.memory = rhi::MemoryKind::kUpload;
  rhi::Buffer* staging = Device().CreateBuffer(desc);
  if (!staging)
    return false;
  std::memcpy(staging->mapped(), reinterpret_cast<const void*>(base), sz);
  rhi::CommandList* list = BeginImmediate();
  if (!list) {
    Device().Destroy(staging);
    return false;
  }
  rhi::BufferTextureCopy copy;
  copy.region.width = texel_w;
  copy.region.height = texel_h;
  copy.region.layers = is_3d ? 1 : layout.layers;
  copy.region.depth = is_3d ? layout.layers : 1;
  rhi::TextureRange range;
  range.layers = is_3d ? 1 : layout.layers;
  rhi::TextureBarrier b{img, rhi::TextureState::kUndefined,
                        rhi::TextureState::kCopyDst, range};
  list->Barrier(rhi::kAccessHostWrite, rhi::kAccessCopyRead, &b, 1);
  list->CopyBufferToTexture(img, staging, &copy, 1);
  b.before = rhi::TextureState::kCopyDst;
  b.after = rhi::TextureState::kShaderRead;
  list->Barrier(0, 0, &b, 1);
  const bool ok = EndImmediate(list, "imm.tex_upload");
  Device().Destroy(staging);
  return ok;
}

bool RecordTexPixels(rhi::Texture* img,
                     rhi::TextureState old_state,
                     u64 base,
                     const gcn::TextureLayout32& layout,
                     u32 texel_w,
                     u32 texel_h,
                     bool is_3d,
                     u32 first_mip = 0,
                     bool repack111110 = false) {
  DELTA_ZONE("gpu.tex_upload");
  const u64 start = NowNs();
  if (!DetileTextureOnGpu(base, layout, texel_w, texel_h, img, old_state, is_3d,
                          first_mip, repack111110)) {
    if (repack111110) {
      static int n = 0;
      if (n++ < 4)
        BASE_LOGW("gpuvk", "texture {:#x} tiling={}: 11_11_10 repacked on the CPU",
                  (unsigned long)base, layout.tiling_idx);
      return UploadRepacked111110(img, base, layout, texel_w, texel_h, is_3d);
    }
    static int n = 0;
    if (n++ < 16)
      BASE_LOGW("gpuvk", "texture {:#x} {}x{} tiling={} elem={}: no GPU upload",
                (unsigned long)base, texel_w, texel_h, layout.tiling_idx,
                layout.elem_bytes);
    return false;
  }
  const u64 elapsed = NowNs() - start;
  g_stats.ns_tex_up += elapsed;
  g_frame_ns.tex_up += elapsed;
  g_stats.tex_ups++;
  return true;
}

using TexImageKeySet = base::HashSet<TexImageKey, TexImageKeyHash>;

void RetireTextureImages(const TexImageKeySet& keys) {
  if (keys.empty())
    return;
  BumpTextureEpoch();
  for (auto set = g_texture_cache.sets.begin();
       set != g_texture_cache.sets.end();) {
    if (keys.count(set->first.image)) {
      g_texture_cache.retired_sets.push_back(set->second);
      set = g_texture_cache.sets.erase(set);
    } else {
      ++set;
    }
  }
  for (auto view = g_texture_cache.views.begin();
       view != g_texture_cache.views.end();) {
    if (keys.count(view->first.image)) {
      g_texture_cache.retired_views.push_back(view->second);
      view = g_texture_cache.views.erase(view);
    } else {
      ++view;
    }
  }
  for (const TexImageKey& key : keys) {
    auto image = g_texture_cache.images.find(key);
    if (image == g_texture_cache.images.end())
      continue;
    UnregisterTexturePages(key, image->second.footprint);
    GPU_BUGCHECK(g_texture_cache.image_bytes >= image->second.allocation_size,
                 "texture budget underflow: %llu live < %llu retiring",
                 (unsigned long long)g_texture_cache.image_bytes,
                 (unsigned long long)image->second.allocation_size);
    g_texture_cache.image_bytes -= image->second.allocation_size;
    g_texture_cache.retired_images.push_back(image->second);
    g_texture_cache.images.erase(image);
  }
  ClearMultiTexCache();
}

constexpr size_t kMaxTextureImages = 3000;

bool EvictTextures(u64 bytes, u64 budget) {
  base::Vector<base::Pair<int, TexImageKey>> old;
  for (const auto& [key, entry] : g_texture_cache.images)
    if (entry.last_used_frame != g_frame.num)
      old.push_back({entry.last_used_frame, key});
  if (old.empty())
    return false;
  base::Sort(old.begin(), old.end(),
             [](const auto& a, const auto& b) { return a.first < b.first; });
  const u64 want_bytes = budget - base::Min(budget, budget / 16 + bytes);
  const size_t want_count = kMaxTextureImages - kMaxTextureImages / 16;
  u64 live_bytes = g_texture_cache.image_bytes;
  size_t live_count = g_texture_cache.images.size();
  TexImageKeySet retire;
  for (const auto& [frame, key] : old) {
    if (live_bytes <= want_bytes && live_count <= want_count)
      break;
    live_bytes -= g_texture_cache.images[key].allocation_size;
    live_count--;
    retire.insert(key);
  }
  RetireTextureImages(retire);
  return true;
}

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
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kMetadata);
  ScopeNs lookup_timer(&g_stats.ns_tex_lookup);
  g_stats.tex_lookup_n++;
  constexpr u64 kMaxTextureBytes = 512ull * 1024 * 1024;
  if (kTexWatch && base == (u64)kTexWatch) {
    static int watched = -1;
    if (watched != g_frame.num && gpu::IsReadableRange(base, 32)) {
      watched = g_frame.num;
      const u8* p = reinterpret_cast<const u8*>(base);
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
  const u32 max_size = base::Min(Device().caps().max_texture_size, 16384u);
  if (!w || !h || w > max_size || h > max_size)
    return nullptr;
  const rhi::Format format = SampledFormat(dfmt, nfmt);
  const bool repack = Repacked111110(dfmt, nfmt);
  if (format == rhi::Format::kUndefined)
    return nullptr;
  if (is_3d) {
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
  tiling = TextureTiling(tiling);
  const bool bc = GuestFormatBlockCompressed(dfmt);
  const u32 elem_bytes = GuestFormatElemBytes(dfmt);
  const bool gfx10 = tiling >= gcn::kGfx10TilingBase;
  if (bc && mip_levels > 1 && !gfx10 && ((w & (w - 1)) || (h & (h - 1))))
    return nullptr;
  if (bc && is_3d)
    return nullptr;
  const u32 lw = bc ? (w + 3) / 4 : w;
  const u32 lh = bc ? (h + 3) / 4 : h;
  const u32 lpitch = bc ? ((pitch ? pitch : w) + 3) / 4 : (pitch ? pitch : w);
  TexKey& key = key_out;
  key = TextureKey(base, w, h, dfmt, nfmt, tiling, pitch, layers, base_array,
                   view_layers, mip_levels, base_mip, view_mips, min_lod,
                   pow2_pad, sampler, sampler_valid, arrayed, force_lod_zero,
                   depth_compare, swizzle, depth, is_3d);
  auto image_it = g_texture_cache.images.find(key.image);
  base::UniquePointer<gcn::TextureLayout32> new_layout;
  if (image_it == g_texture_cache.images.end()) {
    new_layout = base::MakeUnique<gcn::TextureLayout32>();
    auto& layout = *new_layout;
    if (!gcn::BuildTextureLayout32(layout, lw, lh, lpitch,
                                   is_3d ? depth : layers, mip_levels, tiling,
                                   pow2_pad, elem_bytes)) {
      if (kTexFail) {
        static int n = 0;
        if (n++ < 12)
          BASE_LOGI("texfail",
                    "layout {:#x} {}x{} pitch={} tiling={} elem={} mips={}",
                    (unsigned long)base, w, h, lpitch, tiling, elem_bytes,
                    mip_levels);
      }
      return nullptr;
    }
    if (bc && gfx10)
      for (u32 mip = 0; mip < mip_levels; mip++) {
        auto& level = layout.mips[mip];
        level.width = (base::Max(w >> mip, 1u) + 3) / 4;
        level.height = (base::Max(h >> mip, 1u) + 3) / 4;
      }
  }
  const auto& layout = image_it == g_texture_cache.images.end()
                           ? *new_layout
                           : image_it->second.layout;
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
  if (kTexCensus && gpu::IsReadableRange(base, footprint)) {
    struct Cell {
      u32 w, h, dfmt, nfmt, tiling, mips;
      u64 footprint, binds;
    };
    static base::Mutex m;
    static base::Map<u64, Cell> tbl;
    static const guest::SessionReset reset_tbl(
        [] { guest::ResetResource(tbl); });
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
  if (kTexRaw && w >= 256 && h >= 128 &&
      gpu::IsReadableRange(base, footprint)) {
    static int rawn = 0;
    static base::HashSet<u64> raw_seen;
    static const guest::SessionReset reset_raw_seen(
        [] { guest::ResetResource(raw_seen); });
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
  gcn::TextureLayout32 linear;
  u32 supplied_mips = layout.mip_levels;
  u64* checked_seq = image_it != g_texture_cache.images.end() &&
                             image_it->second.last_checked_frame == g_frame.num
                         ? &image_it->second.cs_seq
                         : nullptr;
  const bool cs_image = (!is_3d || mip_levels == 1) &&
                        CsSupplyTexture(base, layout, w, h, nullptr,
                                        rhi::TextureState::kUndefined,
                                        checked_seq, bc, &supplied_mips);
  const bool cs_buffer =
      !cs_image && !bc && !is_3d &&
      gcn::BuildTextureLayout32(linear, lw, lh, lpitch, layers, mip_levels, 8,
                                pow2_pad, elem_bytes) &&
      CsSupplyTextureFromBuffer(base, layout, linear, w, h, nullptr,
                                rhi::TextureState::kUndefined, nullptr);
  const bool cs_supplies = cs_image || cs_buffer;
  const auto cs_supply = [&](TexImageEntry& entry, rhi::TextureState state) {
    if (cs_image) {
      if (supplied_mips < layout.mip_levels &&
          entry.last_checked_frame != g_frame.num &&
          (entry.tail_first_mip != supplied_mips || !entry.tail_clean)) {
        if (!RecordTexPixels(entry.image, state, base, layout, w, h, is_3d,
                             supplied_mips))
          return false;
        state = rhi::TextureState::kShaderRead;
        entry.tail_first_mip = supplied_mips;
        CollectGuestWrites();
        const u64 offset = layout.mips[supplied_mips].offset;
        entry.tail_clean =
            GuestWriteTracker().Arm(base + offset, footprint - offset);
        g_stats.tex_refresh_n++;
        g_stats.tex_refresh_bytes += footprint - offset;
      } else if (supplied_mips == layout.mip_levels) {
        entry.tail_first_mip = 0;
      }
      return CsSupplyTexture(base, layout, w, h, entry.image, state,
                             &entry.cs_seq, bc, &supplied_mips);
    }
    return CsSupplyTextureFromBuffer(base, layout, linear, w, h, entry.image,
                                     state, &entry.cs_seq);
  };
  if (cs_supplies && image_it != g_texture_cache.images.end()) {
    if (!cs_supply(image_it->second, rhi::TextureState::kShaderRead))
      return nullptr;
    image_it->second.last_checked_frame = g_frame.num;
    image_it->second.wt_clean = false;
  }
  if (!cs_supplies && !render::FlushCsWritesForGpu(render::DefaultRenderer(),
                                                   base, footprint, "tex"))
    return nullptr;
  const bool known = image_it != g_texture_cache.images.end();
  if (!known || image_it->second.last_checked_frame != g_frame.num) {
    const u64 t_probe = NowNs();
    const bool readable = (known && image_it->second.wt_clean) ||
                          gpu::IsReadableRange(base, footprint);
    g_stats.ns_tex_probe += NowNs() - t_probe;
    g_stats.tex_probe_n++;
    if (!readable) {
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
  }
  if (known && image_it->second.last_checked_frame != g_frame.num) {
    TexImageEntry& e = image_it->second;
    CollectGuestWritesForSubmission();
    e.last_checked_frame = g_frame.num;
    if (!e.wt_clean) {
      const u64 t_refresh = NowNs();
      if (!RecordTexPixels(e.image, rhi::TextureState::kShaderRead, base,
                           layout, w, h, is_3d, 0, repack)) {
        if (kTexFail) {
          static int n = 0;
          if (n++ < 12)
            BASE_LOGI("texfail", "refresh {:#x} {}x{}x{} tiling={}",
                      (unsigned long)base, w, h, is_3d ? depth : layers,
                      tiling);
        }
        return nullptr;
      }
      CollectGuestWrites();
      e.wt_clean = GuestWriteTracker().Arm(base, footprint);
      g_stats.ns_tex_refresh += NowNs() - t_refresh;
      g_stats.tex_refresh_n++;
      g_stats.tex_refresh_bytes += footprint;
    }
  }
  if (image_it == g_texture_cache.images.end()) {
    static const u64 kTextureBudget =
        base::Max<u64>(kTextureMb, 64) * 1024 * 1024;
    if (g_texture_cache.images.size() >= kMaxTextureImages &&
        !EvictTextures(0, kTextureBudget))
      return nullptr;
    TexImageEntry image_entry;
    image_entry.footprint = footprint;
    image_entry.layout = layout;
    image_entry.last_checked_frame = -1;
    image_entry.last_used_frame = g_frame.num;
    const u64 bytes = TextureLinearBytes(layout);
    if (g_texture_cache.image_bytes + bytes > kTextureBudget &&
        (!EvictTextures(bytes, kTextureBudget) ||
         g_texture_cache.image_bytes + bytes > kTextureBudget))
      return nullptr;
    rhi::TextureDesc td;
    td.dim = is_3d ? rhi::TextureDim::k3D : rhi::TextureDim::k2D;
    td.format = (format);
    td.width = w;
    td.height = h;
    td.depth = is_3d ? depth : 1;
    td.layers = layers;
    td.mips = mip_levels;
    td.usage =
        rhi::kTextureSampled | rhi::kTextureCopyDst | rhi::kTextureCopySrc;
    image_entry.image = Device().CreateTexture(td);
    if (!image_entry.image)
      return nullptr;
    const bool cs_uploaded =
        cs_supplies && cs_supply(image_entry, rhi::TextureState::kUndefined);
    if (!cs_uploaded &&
        !RecordTexPixels(image_entry.image, rhi::TextureState::kUndefined,
                         base, layout, w, h, is_3d, 0, repack)) {
      if (kTexFail) {
        static int n = 0;
        if (n++ < 12)
          BASE_LOGI("texfail", "upload {:#x} {}x{}x{} tiling={}",
                    (unsigned long)base, w, h, is_3d ? depth : layers, tiling);
      }
      Device().Destroy(image_entry.image);
      return nullptr;
    }
    if (!cs_uploaded) {
      CollectGuestWrites();
      image_entry.wt_clean = GuestWriteTracker().Arm(base, footprint);
    }
    image_entry.last_checked_frame = g_frame.num;
    image_entry.allocation_size = bytes;
    if (Device().caps().debug_labels) {
      char name[96];
      std::snprintf(name, sizeof(name), "tex %#llx %ux%u mips=%u layers=%u",
                    (unsigned long long)base, w, h, mip_levels, layers);
      Device().SetName(image_entry.image, name);
    }
    image_it = g_texture_cache.images.emplace(key.image, image_entry).first;
    g_texture_cache.image_bytes += bytes;
    RegisterTexturePages(key.image, footprint);
  }
  image_it->second.last_used_frame = g_frame.num;

  TexViewKey view_key = TextureViewKey(key);
  auto view_it = g_texture_cache.views.find(view_key);
  if (view_it == g_texture_cache.views.end()) {
    if (g_texture_cache.views.size() >= 12000)
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
    TextureSwizzle(view_key.swizzle, vd.swizzle);
    view_entry.view = Device().CreateView(image_it->second.image, vd);
    if (!view_entry.view)
      return nullptr;
    view_it = g_texture_cache.views.emplace(view_key, view_entry).first;
  }
  return view_it->second.view;
}

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
  auto it = g_texture_cache.sets.find(key);
  if (it != g_texture_cache.sets.end())
    return it->second.set;
  if (g_texture_cache.sets.size() >= 12000)
    return nullptr;

  const bool cmp_default =
      key.sampler.depth_compare && g_tex.depth_default_view;
  TexEntry e;
  e.set = SampledTextureGroup(cmp_default ? g_tex.depth_default_view : view,
                              SamplerFor(key.sampler),
                              cmp_default ? rhi::TextureState::kDepthRead
                                          : rhi::TextureState::kShaderRead);
  if (!e.set)
    return nullptr;
  g_texture_cache.sets.emplace(key, e);
  return e.set;
}

bool GuestTextureUploadSupported(u32 dfmt, u32 nfmt) {
  return SampledFormat(dfmt, nfmt) != rhi::Format::kUndefined;
}

static rhi::TextureView* ResolveTexViewUncached(const DrawInfo::DrawTex& t);

namespace {
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
  g_texture_cache.epoch++;
}

u64 TextureEpoch() {
  return g_texture_cache.epoch;
}

rhi::TextureView* TexViewFor(const DrawInfo::DrawTex& t) {
  if (g_texture_cache.memo_frame != g_frame.num) {
    g_texture_cache.memo_frame = g_frame.num;
    g_texture_cache.epoch++;
  }
  const TexMemoKey memo_key = MemoKeyOf(t);
  TexMemo& memo = g_texture_cache.memo[TexMemoSlot(memo_key)];
  if (kTexMemo && memo.epoch == g_texture_cache.epoch && memo.key == memo_key)
    return memo.view;
  rhi::TextureView* view = ResolveTexViewUncached(t);
  if (view)
    memo = {memo_key, g_texture_cache.epoch, view};
  return view;
}

static rhi::TextureView* ResolveTexViewUncached(const DrawInfo::DrawTex& t) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kMetadata);
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

void ClearMultiTexCache() {
  for (const auto& [key, entry] : g_texture_cache.multi_sets) {
    (void)key;
    if (entry.set)
      g_texture_cache.retired_multi_sets.push_back(entry);
  }
  g_texture_cache.multi_sets.clear();
}

void ReleaseRetiredTextures() {
  static base::Vector<MultiTexSet> aged_mtex;
  static const guest::SessionReset reset_aged_mtex(
      [] { guest::ResetResource(aged_mtex); });
  static base::Vector<TexEntry> aged_tex_sets;
  static const guest::SessionReset reset_aged_tex_sets(
      [] { guest::ResetResource(aged_tex_sets); });
  static base::Vector<TexViewEntry> aged_tex_views;
  static const guest::SessionReset reset_aged_tex_views(
      [] { guest::ResetResource(aged_tex_views); });
  static base::Vector<TexImageEntry> aged_tex_images;
  static const guest::SessionReset reset_aged_tex_images(
      [] { guest::ResetResource(aged_tex_images); });
  for (const MultiTexSet& entry : aged_mtex)
    Device().Destroy(entry.set);
  for (const TexEntry& e : aged_tex_sets)
    Device().Destroy(e.set);
  for (const TexViewEntry& e : aged_tex_views)
    Device().Destroy(e.view);
  for (const TexImageEntry& e : aged_tex_images)
    Device().Destroy(e.image);
  aged_mtex = base::move(g_texture_cache.retired_multi_sets);
  aged_tex_sets = base::move(g_texture_cache.retired_sets);
  aged_tex_views = base::move(g_texture_cache.retired_views);
  aged_tex_images = base::move(g_texture_cache.retired_images);
  g_texture_cache.retired_multi_sets.clear();
  g_texture_cache.retired_sets.clear();
  g_texture_cache.retired_views.clear();
  g_texture_cache.retired_images.clear();
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
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kMetadata);
  ScopeNs set_timer(&g_stats.ns_tex_set);
  g_stats.tex_set_n++;
  const u32 resolved = base::Min(d.num_texs, kMaxTex);
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
  rhi::TextureView* views[kMaxTex];
  rhi::TextureState layouts[kMaxTex];
  static int tex_miss_logged = 0;
  for (u32 i = 0; i < key.num_texs; i++) {
    rhi::TextureView* v =
        (i < resolved && !kForceWhite) ? resolved_views[i] : nullptr;
    bool arrayed = i < resolved && d.texs[i].arrayed;
    bool is_3d =
        i < resolved ? d.texs[i].is_3d : (declared(i) && declared(i)->is_3d);
    if (i >= resolved && declared(i) && declared(i)->storage)
      return nullptr;
    if (kTexMiss && !v && i < resolved && tex_miss_logged < 64) {
      tex_miss_logged++;
      const auto& t = d.texs[i];
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
    const bool want_uint = declared(i) && declared(i)->is_uint;
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
    const bool have_tex = i < resolved;
    const bool storage_i = have_tex && d.texs[i].storage;
    SamplerKey sampler;
    if (have_tex) {
      sampler.valid = d.texs[i].sampler_valid;
      std::memcpy(sampler.raw, d.texs[i].sampler, sizeof(sampler.raw));
      sampler.image_min_lod = d.texs[i].min_lod;
      sampler.force_lod_zero = d.texs[i].force_lod_zero;
      sampler.depth_compare =
          d.texs[i].depth_compare && depth_src && depth_src[i];
      sampler.integer =
          !d.texs[i].storage &&
          ((kIntegerRt && (d.texs[i].nfmt == 4 || d.texs[i].nfmt == 5)) ||
           (resolved_formats && IsIntegerColorFormat(resolved_formats[i])));
    }
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
  auto ci = g_texture_cache.multi_sets.find(key);
  if (ci != g_texture_cache.multi_sets.end())
    return ci->second.set;
  if (g_texture_cache.multi_sets.size() > 3500)
    return nullptr;
  rhi::BindGroupDesc group;
  group.layout = set_layout;
  group.writes.assign(writes, writes + key.num_texs);
  MultiTexSet entry;
  entry.set = Device().CreateBindGroup(group);
  if (!entry.set)
    return nullptr;
  g_texture_cache.multi_sets[key] = entry;
  return entry.set;
}

void InvalidateTexRange(u64 base, u64 size) {
  if (!size || base > UINT64_MAX - size)
    return;
  BumpTextureEpoch();
  u64 end = base + size;
  base::Vector<TexImageKey> overlap;
  for (u64 page = base >> kTexturePageShift;
       page <= (end - 1) >> kTexturePageShift; page++) {
    auto found = g_texture_cache.pages.find(page);
    if (found == g_texture_cache.pages.end())
      continue;
    for (const TexImageKey& key : found->second) {
      if (base::Find(overlap.begin(), overlap.end(), key) == overlap.end())
        overlap.push_back(key);
    }
  }
  for (const TexImageKey& key : overlap) {
    auto found = g_texture_cache.images.find(key);
    if (found != g_texture_cache.images.end() && key.base < end &&
        base < key.base + found->second.footprint) {
      found->second.last_checked_frame = -1;
      found->second.wt_clean = false;
      found->second.tail_clean = false;
    }
  }
}

namespace {
const guest::SessionReset g_session_reset([] {
  guest::ResetResource(g_texture_cache);
});
}  // namespace

}  // namespace gpu::render
