/*
 * PS4Delta : PS4 emulation and research project
 */

#include "gpu/ps4/compute_dispatch.h"
#include "base/arch.h"
#include "gpu/guest_memory.h"
#include "guest/session.h"

#include <cstring>

#include "base/logging.h"
#include "host_memory/host_memory.h"
#include "options/options.h"

#include "base/algorithm.h"
#include "base/containers/hash_map.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/shared_pointer.h"
#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_detile.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps4/cmd_trace.h"
#include "gpu/ps4/draw_state.h"
#include "gpu/ps4/guest_address.h"
#include "gpu/ps4/render_queue.h"
#include "gpu/ps4/shader_cache.h"
#include "gpu/render/image/render_target.h"
#include "profile/profile.h"

namespace {
DELTA_OPTION(bool, kNoCs, "DELTA_GPU_NOCS", false);
DELTA_OPTION(bool, kCsOnRenderThread, "DELTA_GPU_CS_RENDER_THREAD", true);
DELTA_OPTION(bool, kCsNativeImages, "DELTA_GPU_CS_NATIVE_IMAGES", true);
}  // namespace

namespace gpu::ps4 {
namespace {

constexpr u64 kMaxResource = 256ull * 1024 * 1024;
constexpr u64 kMaxUnboundedBuffer = 16ull * 1024 * 1024;
constexpr u64 kZeroFillBytes = 16;

bool IsMappedGuestRange(u64 address, u64 bytes) {
  return IsGuestRange(address, bytes) &&
         gpu::IsReadableRangeCached(address, bytes);
}

u64 ExpandedHtileBase(const render::ComputeInfo& ci,
                      const u32* threads,
                      u32 user_sgpr,
                      u32 tgid_enable) {
  if (user_sgpr != 16 || (tgid_enable & 3) != 3 || threads[0] != 8 ||
      threads[1] != 8 || threads[2] != 1 || !ci.groups[2] ||
      !IsMappedGuestRange(ci.cs_addr, 291 * sizeof(u32)))
    return 0;
  const auto* code = reinterpret_cast<const u32*>(ci.cs_addr);
  if (code[0] != 0xbeeb03ff || code[1] != 0x92)
    return 0;
  u64 hash = 1469598103934665603ull;
  for (u32 i = 0; i < 291; ++i)
    hash = (hash ^ code[i]) * 1099511628211ull;
  if (hash != 0xe043b9bd71a7d375ull)
    return 0;
  const gcn::VBuffer dst = gcn::DecodeVBuffer(ci.user_data + 8);
  if (!dst.base || dst.stride != 4 || !dst.num_records || dst.dfmt != 4 ||
      (dst.nfmt != 4 && dst.nfmt != 5 && dst.nfmt != 7) ||
      (ci.user_data[9] >> 31) || (ci.user_data[11] & (1u << 23)))
    return 0;
  for (u32 i = 0; i < ci.num_res; ++i) {
    const auto& res = ci.res[i];
    if (res.base == dst.base && res.written && !res.zero_fill &&
        !res.image_staging && res.size >= 4 && !res.view_offset)
      return dst.base;
  }
  return 0;
}

struct ResourceRange {
  u64 base = 0;
  u64 size = 0;
  u64 guest_size = 0;
  gcn::TImage image;
  bool image_staging = false;
  bool zero_fill = false;
  u32 elem_bytes = 4;
  u32 stage_elem_bytes = 4;
  u64 write_bytes = 0;
  bool ok = true;
};

ResourceRange ResolveImageResource(u64 cs_addr,
                                   const gcn::CsResource& res,
                                   const u32* descriptor,
                                   bool trace) {
  ResourceRange out;
  const gcn::TImage t = gcn::DecodeTImage(descriptor);
  const bool r8 = t.dfmt == 1 && (t.nfmt == 0 || t.nfmt == 1 || t.nfmt == 4);
  const bool rgba8 = t.dfmt == 10 && (t.nfmt == 0 || t.nfmt == 1 ||
                                      t.nfmt == 4 || t.nfmt == 5 || t.nfmt == 9);
  const bool r32 = t.dfmt == 4 && (t.nfmt == 4 || t.nfmt == 5 || t.nfmt == 7);
  const bool rg16f = t.dfmt == 5 && t.nfmt == 7;
  const bool rg16i = t.dfmt == 5 && (t.nfmt == 4 || t.nfmt == 5);
  const bool r16f = t.dfmt == 2 && (t.nfmt == 0 || t.nfmt == 7);
  const bool rg8 = t.dfmt == 3 && (t.nfmt == 0 || t.nfmt == 1 || t.nfmt == 4);
  const bool rgba16f = t.dfmt == 12 && t.nfmt == 7;
  const bool r11g11b10f = (t.dfmt == 6 || t.dfmt == 7) && t.nfmt == 7;
  const bool block64 = (t.dfmt == 11 && (t.nfmt == 4 || t.nfmt == 5 ||
                                         t.nfmt == 7)) ||
                       (t.dfmt == 12 && (t.nfmt == 4 || t.nfmt == 5));
  const bool block128 =
      t.dfmt == 14 && (t.nfmt == 4 || t.nfmt == 5 || t.nfmt == 7);
  const bool supported_type = t.type == 8 || t.type == 9 || t.type == 10 ||
                              t.type == 11 || t.type == 12 || t.type == 13;
  const bool supported_format = r8 || rgba8 || r32 || rg16f || rg16i || r16f ||
                                rg8 || rgba16f || r11g11b10f || block64 ||
                                block128;
  out.elem_bytes = (rgba16f || block64) ? 8
                   : block128           ? 16
                   : (r16f || rg8)      ? 2
                   : r8                 ? 1
                                        : 4;
  out.stage_elem_bytes = r11g11b10f ? 16 : base::Max(out.elem_bytes, 4u);

  if (!supported_type || !supported_format) {
    if (trace)
      TraceCsZeroFill(cs_addr, res.binding, t, descriptor);
    out.zero_fill = true;
    out.size = kZeroFillBytes;
    return out;
  }

  gcn::TextureLayout32 layout;
  if (!t.valid ||
      !gcn::BuildTextureLayout32(layout, t.width, t.height, t.pitch,
                                 t.is_3d ? t.depth : t.layers, t.mip_levels,
                                 t.tiling_idx, t.pow2_pad, out.elem_bytes)) {
    if (trace)
      TraceCsUnsupportedImage(cs_addr, res.binding, t, descriptor);
    out.ok = false;
    return out;
  }
  out.base = t.base;
  out.guest_size = layout.size;
  // The shader reads these packed straight from guest memory (see
  // EmitCsMimgStaged), so there is nothing to widen.
  const bool packed = !res.written && out.elem_bytes < 4 &&
                      !((descriptor[3] >> 16) & 0xF);
  out.image_staging = !gcn::TilingIsLinear(t.tiling_idx) ||
                      (out.elem_bytes != out.stage_elem_bytes && !packed);
  if (!out.image_staging) {
    out.size = out.guest_size;
    return out;
  }
  gcn::TextureLayout32 linear;
  const u32 stage_tiling = t.tiling_idx == 31 ? 31 : 8;
  if (!gcn::BuildTextureLayout32(
          linear, t.width, t.height, t.pitch, t.is_3d ? t.depth : t.layers,
          t.mip_levels, stage_tiling, t.pow2_pad, out.stage_elem_bytes)) {
    out.ok = false;
    return out;
  }
  out.size = linear.size;
  out.image = t;
  return out;
}

ResourceRange ResolveBufferResource(u64 cs_addr,
                                    const gcn::CsResource& res,
                                    const u32* descriptor,
                                    bool trace) {
  ResourceRange out;
  if (res.kind == 2) {
    out.base = (static_cast<u64>(descriptor[1] & 0xFFFF) << 32) | descriptor[0];
    out.size = res.min_bytes;
    if (!IsMappedGuestRange(out.base, 1)) {
      out.zero_fill = true;
      out.base = 0;
      out.size = base::Max<u64>(out.size, kZeroFillBytes);
    }
    return out;
  }
  const gcn::VBuffer v = gcn::DecodeVBuffer(descriptor);
  out.base = v.base;
  out.size = v.stride ? (u64)v.stride * v.num_records : v.num_records;
  if (out.size < res.min_bytes)
    out.size = res.min_bytes;
  const bool swizzled = (descriptor[1] >> 31) || ((descriptor[3] >> 23) & 1);
  if (res.written && res.store_extent &&
      res.store_extent != gcn::kUnknownStoreExtent && v.stride &&
      v.num_records && !swizzled) {
    const u64 end =
        static_cast<u64>(v.num_records - 1) * v.stride + res.store_extent;
    if (end < out.size)
      out.write_bytes = end;
  }
  if (!IsMappedGuestRange(out.base, 1)) {
    out.zero_fill = true;
    out.base = 0;
    out.size = kZeroFillBytes;
    out.write_bytes = 0;
  } else if (out.size > kMaxResource) {
    const u64 declared = out.size;
    out.size = host_memory::MappedMemoryPrefix(
        reinterpret_cast<const void*>(out.base), kMaxUnboundedBuffer);
    if (trace)
      TraceCsWindowedBuffer(cs_addr, res.binding, declared, out.size);
  }
  return out;
}

}  // namespace

namespace {
ComputeShaderState ComputeStateOf(const Regs& regs) {
  ComputeShaderState state;
  state.cs_addr = (static_cast<u64>(regs[mmCOMPUTE_PGM_HI] & 0xFF) << 32 |
                   regs[mmCOMPUTE_PGM_LO])
                  << 8;
  state.thread_x = regs[mmCOMPUTE_NUM_THREAD_X] & 0xFFFF;
  state.thread_y = regs[mmCOMPUTE_NUM_THREAD_Y] & 0xFFFF;
  state.thread_z = regs[mmCOMPUTE_NUM_THREAD_Z] & 0xFFFF;
  const u32 rsrc2 = regs[mmCOMPUTE_PGM_RSRC2];
  state.user_sgpr = (rsrc2 >> 1) & 0x1F;
  state.tgid_enable = (rsrc2 >> 7) & 0x7;
  state.lds_dwords = (rsrc2 >> 15) & 0x1FF;
  return state;
}
}  // namespace

constexpr u32 kFillKernel[] = {0xbeeb03ff, 0x00000009, 0xc2020500, 0x8f6a8608,
                               0xd2ba0000, 0x0401006a, 0xbf8c007f, 0x7e020204,
                               0xd1a80004, 0x00020002, 0xe0102000, 0x80000100,
                               0xbf810000};

DELTA_OPTION(bool, kFillKernels, "DELTA_GPU_FILL_KERNELS", true);

bool RunFillKernelAsFill(render::Renderer& renderer,
                         u64 cs_addr,
                         const u32 groups[3],
                         const u32 threads[3],
                         u32 user_sgpr,
                         u32 tgid_enable,
                         const u32* user_data) {
  if (!kFillKernels || threads[0] != 64 || threads[1] != 1 || threads[2] != 1 ||
      groups[1] != 1 || groups[2] != 1 || user_sgpr != 8 ||
      !(tgid_enable & 1) || !IsMappedGuestRange(cs_addr, sizeof(kFillKernel)) ||
      std::memcmp(reinterpret_cast<const void*>(cs_addr), kFillKernel,
                  sizeof(kFillKernel)))
    return false;
  const gcn::VBuffer dst = gcn::DecodeVBuffer(user_data);
  const gcn::VBuffer src = gcn::DecodeVBuffer(user_data + 4);
  const bool raw_dwords = !(user_data[1] >> 31) && !(user_data[3] & (1u << 23));
  if (!raw_dwords || dst.stride != 4 || dst.dfmt != 4 ||
      (dst.nfmt != 4 && dst.nfmt != 5 && dst.nfmt != 7))
    return false;
  const u64 bytes = base::Min<u64>(u64(groups[0]) * 64, dst.num_records) * 4;
  if (!bytes || !IsMappedGuestRange(dst.base, bytes) ||
      !IsMappedGuestRange(src.base, 4))
    return false;
  OwnRenderer("fill");
  if (render::OverlapsLiveTarget(dst.base, bytes) ||
      !render::GuestBytesSettled(src.base, 4))
    return false;
  u32 value;
  std::memcpy(&value, reinterpret_cast<const void*>(src.base), 4);
  render::ApplyMemoryFill(renderer, dst.base, bytes, value);
  return true;
}

constexpr u32 kRepeatFillKernel[] = {
    0xbeeb03ff, 0x0000001c, 0x8f008610, 0x4a000000, 0xc2400d00, 0xbf8c007f,
    0x7d880000, 0xbe82246a, 0x7e020c01, 0xbf88002c, 0x7e025501, 0x100202ff,
    0x4f800000, 0x7e020f01, 0xd2d20002, 0x00020201, 0xd2d40003, 0x00020201,
    0x4c080480, 0xd10a0002, 0x00020680, 0xd2000002, 0x000a0504, 0xd2d40002,
    0x00020302, 0x4c060501, 0x4a020501, 0xd2000001, 0x000a0701, 0xd2d40001,
    0x00020101, 0xd2d20001, 0x00000301, 0x4c040300, 0xd18c0002, 0x00020300,
    0xd18c000c, 0x00000302, 0x4e020401, 0x87ea0c02, 0x00020302, 0x4a040201,
    0xd2000001, 0x000a0302, 0xd10a006a, 0x00000280, 0x000202c1, 0xe0002000,
    0x80010101, 0x7d88000a, 0xbe80246a, 0xbf8c0f70, 0xe0102000, 0x80020100,
    0xbf810000,
};

bool RunRepeatFillKernelAsFill(render::Renderer& renderer,
                               u64 cs_addr,
                               const u32 groups[3],
                               const u32 threads[3],
                               u32 user_sgpr,
                               u32 tgid_enable,
                               const u32* user_data) {
  if (!kFillKernels || threads[0] != 64 || threads[1] != 1 || threads[2] != 1 ||
      groups[1] != 1 || groups[2] != 1 || user_sgpr != 16 || tgid_enable != 1 ||
      !IsMappedGuestRange(cs_addr, sizeof(kRepeatFillKernel)) ||
      std::memcmp(reinterpret_cast<const void*>(cs_addr), kRepeatFillKernel,
                  sizeof(kRepeatFillKernel)))
    return false;
  const gcn::VBuffer src = gcn::DecodeVBuffer(user_data + 4);
  const gcn::VBuffer dst = gcn::DecodeVBuffer(user_data + 8);
  const gcn::VBuffer params = gcn::DecodeVBuffer(user_data + 12);
  if (src.stride != 4 || (src.num_records != 1 && src.num_records != 2) ||
      dst.stride != 4 || src.dfmt != 4 || dst.dfmt != 4 ||
      src.nfmt != dst.nfmt ||
      (dst.nfmt != 4 && dst.nfmt != 5 && dst.nfmt != 7) ||
      (user_data[7] & 7) != 4 || (user_data[5] >> 31) || (user_data[9] >> 31) ||
      (user_data[7] & (1u << 23)) || (user_data[11] & (1u << 23)) ||
      u64(params.stride) * params.num_records < 8 ||
      !IsMappedGuestRange(params.base, 8) || !IsMappedGuestRange(src.base, 4))
    return false;
  OwnRenderer("fill");
  if (!render::GuestBytesSettled(params.base, 8))
    return false;
  u32 config[2];
  std::memcpy(config, reinterpret_cast<const void*>(params.base),
              sizeof(config));
  if (config[1] != src.num_records)
    return false;
  const u64 bytes =
      base::Min<u64>(base::Min<u64>(u64(groups[0]) * 64, config[0]),
                     dst.num_records) *
      4;
  const bool wide = config[1] == 2;
  if (!bytes || !IsMappedGuestRange(dst.base, bytes) ||
      (wide ? !render::CanClearMemoryFill64(dst.base, bytes)
            : render::OverlapsLiveTarget(dst.base, bytes)) ||
      !IsMappedGuestRange(src.base, wide ? 8 : 4) ||
      !render::GuestBytesSettled(src.base, wide ? 8 : 4))
    return false;
  if (wide) {
    u64 value;
    std::memcpy(&value, reinterpret_cast<const void*>(src.base), 8);
    render::ApplyMemoryFill64(renderer, dst.base, bytes, value);
    return true;
  }
  u32 value;
  std::memcpy(&value, reinterpret_cast<const void*>(src.base), 4);
  render::ApplyMemoryFill(renderer, dst.base, bytes, value);
  return true;
}

namespace {

// Bindings whose guest memory one lane may write while another reads it.
u64 SharedBindings(const render::ComputeInfo& ci) {
  u64 mask = 0;
  for (u32 w = 0; w < ci.num_res; w++) {
    const render::ComputeInfo::Res& writer = ci.res[w];
    if (!writer.written || writer.binding >= 64)
      continue;
    for (u32 r = 0; r < ci.num_res; r++) {
      const render::ComputeInfo::Res& reader = ci.res[r];
      if (!reader.read || reader.zero_fill || reader.binding >= 64 ||
          reader.base >= writer.base + writer.guest_size ||
          writer.base >= reader.base + reader.guest_size)
        continue;
      mask |= (1ull << writer.binding) | (1ull << reader.binding);
    }
  }
  return mask;
}

// An array binds natively only as a view of a few of its first layers (one
// cube): a whole array can be hundreds of megabytes the texture cache would
// upload.
bool Plain2d(const base::Vector<gcn::TImage>& images, u32 b, bool arrayed) {
  if (b >= images.size())
    return false;
  const gcn::TImage& t = images[b];
  return t.valid && !t.is_3d && !t.is_1d && t.arrayed == arrayed &&
         !t.storage &&
         (!arrayed || (t.view_layers <= 6 && t.base_array + t.view_layers <= 32));
}

// A cube array too large to stage, even at 4 bytes a texel, is one no dispatch
// of ours ever wrote, so its layers hold whatever the memory held before.
bool CubeStageable(const gcn::TImage& t) {
  gcn::TextureLayout32 layout;
  return gcn::BuildTextureLayout32(layout, t.width, t.height, t.pitch, t.layers,
                                   t.mip_levels, t.tiling_idx, t.pow2_pad, 4) &&
         layout.size <= kMaxResource;
}

// `shader` with native image bindings wherever the bound texture is a 2D
// image or array, or `shader` itself when there is none. Fills `images`.
ComputeShaderState NativeVariant(const ComputeShaderState& shader,
                                 const gcn::RecompiledCs& module,
                                 const u32* user_data,
                                 base::Vector<gcn::TImage>& images) {
  if (!kCsNativeImages || !module.ok || !module.native_candidates ||
      !render::CsBindsImages())
    return shader;
  const auto cs_program = gcn::CachedProgram(shader.cs_addr, 4096);
  gcn::TrackTextures(images, cs_program, user_data, false, shader.cs_addr);
  const gcn::MimgBindingPlan plan = gcn::PlanMimgBindings(
      *cs_program, gcn::ComputeReachability(*cs_program).data());
  ComputeShaderState native_shader = shader;
  for (const gcn::CsResource& r : module.resources) {
    const auto it = plan.binding_by_pc.find(r.use_pc);
    if (r.binding >= 64 || !((module.native_candidates >> r.binding) & 1) ||
        it == plan.binding_by_pc.end())
      continue;
    const u32 b = it->second;
    const bool cube = b < images.size() && images[b].type == 11 && b < 32;
    if (cube && !CubeStageable(images[b]))
      continue;
    if (Plain2d(images, b,
                cube || ((module.native_arrayed >> r.binding) & 1))) {
      native_shader.native_mask |= 1ull << r.binding;
      native_shader.cube_mask |= u64(cube) << b;
    }
  }
  if (!native_shader.native_mask)
    return shader;
  for (u32 i = 0; i < images.size() && i < 64; i++)
    if (images[i].nfmt == 4 || images[i].nfmt == 5)
      native_shader.uint_mask |= 1ull << i;
  return native_shader;
}

// The shared-memory mask the dispatch will compute for these descriptors.
u64 PredictSharedMask(u64 cs_addr,
                      const gcn::RecompiledCs& rc,
                      u64 native,
                      const base::Vector<gcn::ResolvedCsResource>& resolved) {
  static constexpr u32 kNullDescriptor[8] = {};
  render::ComputeInfo ci;
  for (const gcn::CsResource& r : rc.resources) {
    if (ci.num_res >= gcn::kMaxCsResources)
      return 0;
    const bool valid = r.binding < resolved.size() && resolved[r.binding].valid;
    const u32* descriptor =
        valid ? resolved[r.binding].descriptor : kNullDescriptor;
    ResourceRange range;
    if (r.binding < 64 && ((native >> r.binding) & 1)) {
      range.zero_fill = true;
      range.size = 16;
    } else {
      range = r.kind == 1
                  ? ResolveImageResource(cs_addr, r, descriptor, false)
                  : ResolveBufferResource(cs_addr, r, descriptor, false);
    }
    if (!range.ok)
      return 0;
    render::ComputeInfo::Res& out = ci.res[ci.num_res++];
    out.base = range.base;
    out.guest_size = range.guest_size || range.zero_fill ? range.guest_size
                                                         : range.size;
    out.binding = r.binding;
    out.read = r.read;
    out.written = r.written && !range.zero_fill;
    out.zero_fill = range.zero_fill;
  }
  return SharedBindings(ci);
}

// `binding << 16 | T# dword1 bits 20..29` for each image `rc` stages.
base::Vector<u32> StagedFormats(
    const gcn::RecompiledCs& rc,
    u64 native,
    const base::Vector<gcn::ResolvedCsResource>& resolved) {
  base::Vector<u32> formats;
  for (const gcn::CsResource& r : rc.resources)
    if (r.kind == 1 && !(r.binding < 64 && ((native >> r.binding) & 1)) &&
        r.binding < resolved.size() && resolved[r.binding].valid)
      formats.push_back(r.binding << 16 |
                        ((resolved[r.binding].descriptor[1] >> 20) & 0x3FF));
  return formats;
}

base::HashMap<u64, u32>& DispatchCounts() {
  static base::HashMap<u64, u32> counts;
  static const guest::SessionReset reset(
      [] { guest::ResetResource(counts); });
  return counts;
}

u32 DispatchCount(u64 cs_addr) {
  const auto it = DispatchCounts().find(cs_addr);
  return it == DispatchCounts().end() ? 0 : it->second;
}

}  // namespace

void PrefetchComputeDispatch(const Regs& regs) {
  const ComputeShaderState state = ComputeStateOf(regs);
  if (kNoCs || !IsGuestAddress(state.cs_addr) || !state.thread_x ||
      !state.thread_y)
    return;
  // Shaders dispatched often already have the variants they use.
  if (DispatchCount(state.cs_addr) >= 16)
    return;
  const u32* user_data = regs.At(mmCOMPUTE_USER_DATA_0);
  PrefetchComputeShader(state, [&](const gcn::RecompiledCs& module) {
    base::Vector<gcn::TImage> images;
    ComputeShaderState chosen = NativeVariant(state, module, user_data, images);
    thread_local base::Vector<gcn::ResolvedCsResource> resolved;
    gcn::ResolveCsResources(resolved, *gcn::CachedProgram(state.cs_addr, 4096),
                            module, user_data, state.cs_addr);
    chosen.formats = StagedFormats(module, chosen.native_mask, resolved);
    if (module.orders_shared_memory)
      chosen.shared_mask =
          PredictSharedMask(state.cs_addr, module, chosen.native_mask, resolved);
    return chosen;
  });
}

void DispatchCompute(render::Renderer& renderer,
                     const Regs& regs,
                     const u32* body,
                     u32 count,
                     u64 indirect_args) {
  DELTA_ZONE("ps4.dispatch");
  u32 groups[3] = {count >= 1 ? body[0] : 0, count >= 2 ? body[1] : 0,
                   count >= 3 ? body[2] : 0};
  const u32 initiator = count >= 4 ? body[3] : 5;
  u32 group_base[3]{};
  if (!(initiator & 4))
    for (u32 axis = 0; axis < 3; ++axis) {
      group_base[axis] = regs[mmCOMPUTE_START_X + axis];
      if (groups[axis] <= group_base[axis])
        return;
      groups[axis] -= group_base[axis];
    }
  const u64 cs_addr = (static_cast<u64>(regs[mmCOMPUTE_PGM_HI] & 0xFF) << 32 |
                       regs[mmCOMPUTE_PGM_LO])
                      << 8;
  const u32 threads[3] = {regs[mmCOMPUTE_NUM_THREAD_X] & 0xFFFF,
                          regs[mmCOMPUTE_NUM_THREAD_Y] & 0xFFFF,
                          regs[mmCOMPUTE_NUM_THREAD_Z] & 0xFFFF};
  const u32 rsrc2 = regs[mmCOMPUTE_PGM_RSRC2];
  const u32 user_sgpr = (rsrc2 >> 1) & 0x1F;
  const u32 tgid_enable = (rsrc2 >> 7) & 0x7;
  const u32 lds_dwords = (rsrc2 >> 15) & 0x1FF;
  TraceComputeShader(regs, cs_addr, groups, threads, user_sgpr, tgid_enable,
                     lds_dwords);

  if (!IsGuestAddress(cs_addr) || !threads[0] || !threads[1] || !groups[0] ||
      !groups[1]) {
    TraceDroppedDispatch(cs_addr, groups, threads,
                         !IsGuestAddress(cs_addr)
                             ? "program address out of guest range"
                         : !threads[0] || !threads[1] ? "no threadgroup size"
                                                      : "zero group count");
    return;
  }
  if (kNoCs || !renderer.available())
    return;
  if (!indirect_args && !group_base[0] && !group_base[1] && !group_base[2] &&
      (RunFillKernelAsFill(renderer, cs_addr, groups, threads, user_sgpr,
                           tgid_enable, regs.At(mmCOMPUTE_USER_DATA_0)) ||
       RunRepeatFillKernelAsFill(renderer, cs_addr, groups, threads, user_sgpr,
                                 tgid_enable, regs.At(mmCOMPUTE_USER_DATA_0))))
    return;

  const ComputeShaderState shader{cs_addr,    threads[0], threads[1],
                                  threads[2], user_sgpr,  tgid_enable,
                                  lds_dwords};
  const u32* user_data = regs.At(mmCOMPUTE_USER_DATA_0);
  const auto cs_program = gcn::CachedProgram(cs_addr, 4096);
  const gcn::RecompiledCs* module = &GetComputeShader(shader);
  ComputeShaderState chosen = shader;
  base::Vector<gcn::TImage> images;
  const ComputeShaderState native_shader =
      NativeVariant(shader, *module, user_data, images);
  if (native_shader.native_mask) {
    const gcn::RecompiledCs& candidate = GetComputeShader(native_shader);
    bool usable = candidate.ok;
    for (u32 i = 0; usable && i < candidate.textures.size(); i++)
      usable = candidate.textures[i] < 0 ||
               (i < images.size() && Plain2d(images, i, images[i].arrayed));
    if (usable) {
      module = &candidate;
      chosen = native_shader;
    }
  }
  const u64 native = chosen.native_mask;
  thread_local base::Vector<gcn::ResolvedCsResource> resolved;
  if (module->ok) {
    gcn::ResolveCsResources(resolved, *cs_program, *module, user_data, cs_addr);
    const auto count_resolved = [&] {
      u32 n = 0;
      for (const auto& r : module->resources)
        n += r.binding < resolved.size() && resolved[r.binding].valid;
      return n;
    };
    const u32 resolved_n = count_resolved();
    static base::HashSet<u64> retry_useless;
    static const guest::SessionReset reset_retry_useless(
        [] { guest::ResetResource(retry_useless); });
    if (resolved_n < module->resources.size() &&
        !retry_useless.contains(cs_addr)) {
      OwnRenderer("cs-unresolved");
      if (!render::FlushCsWrites(renderer, "cs-unresolved") &&
          !renderer.available())
        return;
      gcn::ResolveCsResources(resolved, *cs_program, *module, user_data,
                              cs_addr);
      if (count_resolved() > resolved_n) {
        static u32 helped = 0;
        if (helped++ < 8)
          BASE_LOGI("csgpu", "cs @{:#x}: the unresolved-binding retry helped",
                    cs_addr);
      } else {
        retry_useless.insert(cs_addr);
      }
    }
    ComputeShaderState pinned_state = chosen;
    pinned_state.formats = StagedFormats(*module, native, resolved);
    if (!pinned_state.formats.empty()) {
      const gcn::RecompiledCs& pinned = GetComputeShader(pinned_state);
      if (pinned.ok && pinned.resources.size() == module->resources.size()) {
        module = &pinned;
        chosen = pinned_state;
      }
    }
  }
  DispatchCounts()[cs_addr]++;
  const gcn::RecompiledCs& rc = *module;
  if (!rc.ok) {
    static base::HashSet<u64> reported;
    static const guest::SessionReset reset_reported(
        [] { guest::ResetResource(reported); });
    if (reported.size() < 8 && reported.insert(cs_addr).second)
      BASE_LOGW("csgpu", "unsupported CS @{:#x} groups=[{} {} {}], skipped",
                cs_addr, groups[0], groups[1], groups[2]);
    return;
  }

  const auto job = base::MakeShared<render::ComputeInfo>();
  render::ComputeInfo& ci = *job;
  for (u32 i = 0; i < rc.textures.size(); i++) {
    if (rc.textures[i] < 0 || ci.num_tex >= render::ComputeInfo::kMaxTextures)
      continue;
    render::ComputeInfo::Tex& tex = ci.tex[ci.num_tex++];
    tex.binding = static_cast<u32>(rc.textures[i]);
    FillDrawTex(tex.tex, images[i]);
  }
  ci.cs_addr = cs_addr;
  ci.indirect_args = indirect_args;
  base::Copy(group_base, group_base + 3, ci.group_base);
  ci.groups[0] = groups[0];
  ci.groups[1] = groups[1];
  ci.groups[2] = groups[2];
  ci.recomp = &rc;
  for (int i = 0; i < 16; i++)
    ci.user_data[i] = user_data[i];


  const bool trace = ShouldTraceCsResources(cs_addr);
  static constexpr u32 kNullDescriptor[8] = {};
  for (const auto& r : rc.resources) {
    const bool binding_resolved =
        r.binding < resolved.size() && resolved[r.binding].valid;
    if (!binding_resolved) {
      if (trace)
        TraceCsUnresolved(cs_addr, r);
      TraceCsCode(cs_addr);
    }
    const u32* descriptor =
        binding_resolved ? resolved[r.binding].descriptor : kNullDescriptor;
    ResourceRange range;
    if (r.binding < 64 && ((native >> r.binding) & 1)) {
      range.zero_fill = true;
      range.size = 16;
    } else {
      range = r.kind == 1
                  ? ResolveImageResource(cs_addr, r, descriptor, trace)
                  : ResolveBufferResource(cs_addr, r, descriptor, trace);
    }
    if (!range.ok)
      return;
    if (!range.guest_size && !range.zero_fill)
      range.guest_size = range.size;

    if (trace || CsWatchCovers(range.base, range.size))
      TraceCsResource(cs_addr, r, range.base, range.size, range.guest_size,
                      range.image_staging, range.image, range.elem_bytes,
                      range.stage_elem_bytes);

    if (range.size < r.min_bytes || range.size > kMaxResource ||
        range.guest_size > kMaxResource ||
        (!range.zero_fill &&
         !IsMappedGuestRange(range.base, range.guest_size)) ||
        ci.num_res >= gcn::kMaxCsResources) {
      if (trace)
        TraceCsInvalidRange(cs_addr, r.binding, range.base, range.guest_size);
      return;
    }

    render::ComputeInfo::Res& out = ci.res[ci.num_res++];
    out.base = range.base;
    out.size = range.size;
    out.guest_size = range.guest_size;
    out.binding = r.binding;
    out.shader_writes = r.written;
    out.read = r.read;
    out.written = r.written && !range.zero_fill;
    out.write_bytes = range.write_bytes;
    out.zero_fill = range.zero_fill;
    out.image_staging = range.image_staging;
    if (!range.image_staging)
      continue;
    out.width = range.image.width;
    out.height = range.image.height;
    out.pitch = range.image.pitch;
    out.layers = range.image.is_3d ? range.image.depth : range.image.layers;
    out.mip_levels = range.image.mip_levels;
    out.tiling_idx = range.image.tiling_idx;
    out.elem_bytes = range.elem_bytes;
    out.stage_elem_bytes = range.stage_elem_bytes;
    out.dfmt = range.image.dfmt;
    out.nfmt = range.image.nfmt;
    out.pow2_pad = range.image.pow2_pad;
  }
  if (rc.orders_shared_memory) {
    chosen.shared_mask = SharedBindings(ci);
    if (chosen.shared_mask) {
      const gcn::RecompiledCs& ordered = GetComputeShader(chosen);
      if (ordered.ok && ordered.resources.size() == rc.resources.size())
        ci.recomp = &ordered;
    }
  }
  ci.gds_binding = rc.gds_binding;
  if (!ci.num_res)
    return;
  if (!indirect_args && !group_base[0] && !group_base[1] && !group_base[2])
    ci.expanded_htile_base =
        ExpandedHtileBase(ci, threads, user_sgpr, tgid_enable);

  for (u32 i = 0; i < ci.num_tex; i++)
    if (ci.tex[i].tex.storage)
      TraceAddrWatch("DISPATCH_IMAGE_STORE", ci.tex[i].tex.base,
                     u64(ci.tex[i].tex.pitch ? ci.tex[i].tex.pitch
                                             : ci.tex[i].tex.w) *
                         ci.tex[i].tex.h * ci.tex[i].tex.depth *
                         ci.tex[i].tex.layers * 16,
                     static_cast<u32>(cs_addr), 8);
  for (u32 i = 0; i < ci.num_res; i++)
    if (ci.res[i].written)
      TraceAddrWatch("DISPATCH_WRITE", ci.res[i].base,
                     ci.res[i].write_bytes ? ci.res[i].write_bytes
                                           : ci.res[i].size,
                     static_cast<u32>(cs_addr), 8);
  RenderQueue& queue = GuestRenderQueue();
  if (kCsOnRenderThread && queue.running()) {
    for (u32 i = 0; i < ci.num_res; i++)
      if (ci.res[i].written)
        queue.NotePendingWrite(ci.res[i].base,
                               ci.res[i].write_bytes  ? ci.res[i].write_bytes
                               : ci.res[i].guest_size ? ci.res[i].guest_size
                                                      : ci.res[i].size);
    queue.PushCall([&renderer, job, trace, num_res = ci.num_res] {
      const bool executed = render::Dispatch(renderer, *job);
      if (trace)
        TraceCsDispatch(job->cs_addr, executed, num_res);
    });
    return;
  }
  OwnRenderer("dispatch");
  const bool executed = render::Dispatch(renderer, ci);
  if (trace)
    TraceCsDispatch(cs_addr, executed, ci.num_res);
}

void DispatchComputeDirect(render::Renderer& renderer,
                           const Regs& regs,
                           const u32* body,
                           u32 count) {
  // An earlier DMA_DATA in the stream can patch the group counts in place
  // (DRIVECLUB copies GPU-computed counts over placeholders), so they are only
  // valid when the GPU reaches the packet: read them there.
  const u64 args = reinterpret_cast<uintptr_t>(body);
  if (count >= 4 && WalkReadWouldWait(args, 12) &&
      render::CanDispatchIndirect(args)) {
    const u32 initiator = body[3];
    if ((initiator & 4) || (!regs[mmCOMPUTE_START_X] &&
                            !regs[mmCOMPUTE_START_Y] &&
                            !regs[mmCOMPUTE_START_Z])) {
      const u32 placeholder[] = {1, 1, 1, initiator};
      DispatchCompute(renderer, regs, placeholder, 4, args);
      return;
    }
  }
  DispatchCompute(renderer, regs, body, count);
}

void DispatchComputeIndirect(render::Renderer& renderer,
                             const Regs& regs,
                             const u32* body,
                             u32 count,
                             u64 argument_base) {
  if (count < 2 || (count == 2 && !argument_base))
    return;
  const u64 args = count >= 3 ? body[0] | (u64(body[1] & 0xff) << 32)
                              : argument_base + body[0];
  if ((args & 3) || !IsGuestAddress(args) ||
      !gpu::IsReadableRangeCached(args, 12))
    return;
  const u32 initiator = body[count >= 3 ? 2 : 1];
  const bool starts_at_zero =
      (initiator & 4) || (!regs[mmCOMPUTE_START_X] &&
                          !regs[mmCOMPUTE_START_Y] && !regs[mmCOMPUTE_START_Z]);
  if (starts_at_zero &&
      (render::CanDispatchIndirect(args) || WalkReadWouldWait(args, 12))) {
    const u32 placeholder[] = {1, 1, 1, initiator};
    DispatchCompute(renderer, regs, placeholder, 4, args);
    return;
  }
  FlushForWalkRead(renderer, args, 12, "indirect-cs");
  if (!render::GuestBytesSettled(args, 12))
    return;
  u32 dimensions[4];
  std::memcpy(dimensions, reinterpret_cast<const void*>(args), 12);
  dimensions[3] = initiator;
  DispatchCompute(renderer, regs, dimensions, 4);
}

}  // namespace gpu::ps4
