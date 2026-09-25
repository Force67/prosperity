/*
 * PS4Delta : PS4 emulation and research project
 *
 * Compute dispatches, from COMPUTE_* registers to the renderer. See
 * compute_dispatch.h.
 */

#include "gpu/ps4/compute_dispatch.h"
#include "base/arch.h"

#include <algorithm>
#include <unordered_set>

#include <base/logging.h>
#include <utl/mem.h>
#include <utl/options.h>

#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_detile.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps4/cmd_trace.h"
#include "gpu/ps4/guest_address.h"
#include "gpu/ps4/shader_cache.h"

namespace {
DELTA_OPTION(bool, kNoCs, "DELTA_GPU_NOCS", false);
}  // namespace

namespace gpu::ps4 {
namespace {

// Sanity caps. A storage buffer beyond kMaxResource is a descriptor we misread;
// a buffer whose declared size is unbounded is windowed down to the mapped
// prefix instead of being refused.
constexpr u64 kMaxResource = 256ull * 1024 * 1024;
constexpr u64 kMaxUnboundedBuffer = 16ull * 1024 * 1024;
// An all-zero descriptor stands in for a binding the shader guards and never
// takes; the translator returns zero for it, so a token allocation is enough.
constexpr u64 kZeroFillBytes = 16;

bool IsMappedGuestRange(u64 address, u64 bytes) {
  return IsGuestRange(address, bytes) &&
         utl::isMemoryRangeMapped(reinterpret_cast<const void*>(address),
                                  bytes);
}

// One resource's live guest range and how it has to be staged.
struct ResourceRange {
  u64 base = 0;
  u64 size = 0;        // bytes the dispatch reads/writes through it
  u64 guest_size = 0;  // bytes it occupies in guest memory
  gcn::TImage image;
  bool image_staging = false;  // tiled or reformatted: stage, don't alias
  bool zero_fill = false;      // no live range: hand the shader zeros
  u32 elem_bytes = 4;
  u32 stage_elem_bytes = 4;
  bool ok = true;
};

// A block-compressed surface written by a shader is described as an
// uncompressed integer image whose texel is one BC block: 64 bpp (32_32 /
// 16_16_16_16) for BC1/BC4, 128 bpp (32_32_32_32) for BC2/BC3/BC5. P.T.'s
// texture streamer uploads every streamed surface this way, a compute copy from
// the linear staging area it inflated texture.qar into to the tiled surface the
// draws sample. Rejecting the alias zero-filled both bindings, so the copy
// wrote a 16-byte dummy and every streamed texture stayed empty.
ResourceRange ResolveImageResource(u64 cs_addr,
                                   const gcn::CsResource& res,
                                   const u32* descriptor,
                                   bool trace) {
  ResourceRange out;
  const gcn::TImage t = gcn::DecodeTImage(descriptor);
  const bool r8 = t.dfmt == 1 && (t.nfmt == 0 || t.nfmt == 4);
  const bool rgba8 = t.dfmt == 10 && (t.nfmt == 0 || t.nfmt == 4);
  const bool r32 = t.dfmt == 4 && (t.nfmt == 4 || t.nfmt == 5 || t.nfmt == 7);
  const bool rg16f = t.dfmt == 5 && t.nfmt == 7;
  const bool r16f = t.dfmt == 2 && t.nfmt == 7;
  const bool rg8 = t.dfmt == 3 && t.nfmt == 0;
  const bool rgba16f = t.dfmt == 12 && t.nfmt == 7;
  const bool r11g11b10f = t.dfmt == 6 && t.nfmt == 7;
  const bool block64 =
      (t.dfmt == 11 || t.dfmt == 12) && (t.nfmt == 4 || t.nfmt == 5);
  const bool block128 = t.dfmt == 14 && (t.nfmt == 4 || t.nfmt == 5);
  const bool supported_type = t.type == 8 || t.type == 9 || t.type == 10 ||
                              t.type == 11 || t.type == 12 || t.type == 13;
  const bool supported_format = r8 || rgba8 || r32 || rg16f || r16f || rg8 ||
                                rgba16f || r11g11b10f || block64 || block128;
  out.elem_bytes = (rgba16f || block64) ? 8
                   : block128           ? 16
                   : (r16f || rg8)      ? 2
                   : r8                 ? 1
                                        : 4;
  out.stage_elem_bytes = r11g11b10f ? 16 : std::max(out.elem_bytes, 4u);

  if (!supported_type || !supported_format) {
    // Invalid/null T# values can be present on paths the guest shader does not
    // take. The translator guards these descriptors and returns zero.
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
  out.image_staging = !gcn::TilingIsLinear(t.tiling_idx) ||
                      out.elem_bytes != out.stage_elem_bytes;
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
  if (res.kind == 2) {  // scalar-load pointer into an SRT/descriptor table
    out.base =
        (static_cast<u64>(descriptor[1] & 0xFFFF) << 32) | descriptor[0];
    out.size = res.min_bytes;
    if (!IsMappedGuestRange(out.base, 1)) {
      out.zero_fill = true;
      out.base = 0;
      out.size = std::max<u64>(out.size, kZeroFillBytes);
    }
    return out;
  }
  // Buffer V#: stride*num_records, else the min hint from immediate offsets.
  const gcn::VBuffer v = gcn::DecodeVBuffer(descriptor);
  out.base = v.base;
  out.size = v.stride ? (u64)v.stride * v.num_records : v.num_records;
  if (out.size < res.min_bytes)
    out.size = res.min_bytes;
  if (!IsMappedGuestRange(out.base, 1)) {
    out.zero_fill = true;
    out.base = 0;
    out.size = kZeroFillBytes;
  } else if (out.size > kMaxResource) {
    const u64 declared = out.size;
    out.size = utl::mappedMemoryPrefix(reinterpret_cast<const void*>(out.base),
                                       kMaxUnboundedBuffer);
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
  state.user_sgpr = (rsrc2 >> 1) & 0x1F;   // num_user_regs [37:33]
  state.tgid_enable = (rsrc2 >> 7) & 0x7;  // tgid_enable [41:39]
  state.lds_dwords = (rsrc2 >> 15) & 0x1FF;
  return state;
}
}  // namespace

void PrefetchComputeDispatch(const Regs& regs) {
  const ComputeShaderState state = ComputeStateOf(regs);
  if (!kNoCs && IsGuestAddress(state.cs_addr) && state.thread_x &&
      state.thread_y)
    PrefetchComputeShader(state);
}

void DispatchCompute(rhi::Renderer& renderer,
                     const Regs& regs,
                     const u32* body,
                     u32 count) {
  const u32 groups[3] = {count >= 1 ? body[0] : 0,
                              count >= 2 ? body[1] : 0,
                              count >= 3 ? body[2] : 0};
  const u64 cs_addr =
      (static_cast<u64>(regs[mmCOMPUTE_PGM_HI] & 0xFF) << 32 |
       regs[mmCOMPUTE_PGM_LO])
      << 8;
  const u32 threads[3] = {regs[mmCOMPUTE_NUM_THREAD_X] & 0xFFFF,
                               regs[mmCOMPUTE_NUM_THREAD_Y] & 0xFFFF,
                               regs[mmCOMPUTE_NUM_THREAD_Z] & 0xFFFF};
  const u32 rsrc2 = regs[mmCOMPUTE_PGM_RSRC2];
  const u32 user_sgpr = (rsrc2 >> 1) & 0x1F;   // num_user_regs [37:33]
  const u32 tgid_enable = (rsrc2 >> 7) & 0x7;  // tgid_enable [41:39]
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

  const gcn::RecompiledCs& rc =
      GetComputeShader({cs_addr, threads[0], threads[1], threads[2], user_sgpr,
                        tgid_enable, lds_dwords});
  if (!rc.ok) {
    // Loud, not silently corrupting memory: an unsupported CS is content the
    // frame is missing, and one report per shader says which.
    static std::unordered_set<u64> reported;
    if (reported.size() < 8 && reported.insert(cs_addr).second)
      BASE_LOGW("csgpu", "unsupported CS @{:#x} groups=[{} {} {}], skipped",
                cs_addr, groups[0], groups[1], groups[2]);
    return;
  }

  const u32* user_data = regs.At(mmCOMPUTE_USER_DATA_0);
  rhi::ComputeInfo ci;
  ci.cs_addr = cs_addr;
  ci.groups[0] = groups[0];
  ci.groups[1] = groups[1];
  ci.groups[2] = groups[2];
  ci.recomp = &rc;
  for (int i = 0; i < 16; i++)
    ci.user_data[i] = user_data[i];

  const auto cs_program = gcn::CachedProgram(cs_addr, 4096);
  auto resolved = gcn::ResolveCsResources(*cs_program, rc, user_data);
  // Descriptor chains live in guest memory, which an earlier dispatch may have
  // written, and writebacks are lazy. If any binding failed to resolve, land
  // pending compute writes in guest memory and re-resolve once before falling
  // back to a dummy (the fallback zeroes a real input and silently corrupts
  // whatever pipeline this CS belongs to). Best-effort: a range that could not
  // be written back leaves its chain stale, but the dummy still beats dropping
  // the dispatch. Only a dead renderer makes retrying pointless.
  bool any_unresolved = false;
  for (const auto& r : rc.resources)
    any_unresolved |=
        r.binding >= resolved.size() || !resolved[r.binding].valid;
  if (any_unresolved) {
    if (!rhi::FlushCsWrites(renderer) && !renderer.available())
      return;
    resolved = gcn::ResolveCsResources(*cs_program, rc, user_data);
  }

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
    ResourceRange range =
        r.kind == 1 ? ResolveImageResource(cs_addr, r, descriptor, trace)
                    : ResolveBufferResource(cs_addr, r, descriptor, trace);
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

    rhi::ComputeInfo::Res& out = ci.res[ci.num_res++];
    out.base = range.base;
    out.size = range.size;
    out.guest_size = range.guest_size;
    out.binding = r.binding;
    out.shader_writes = r.written;
    out.read = r.read;
    out.written = r.written && !range.zero_fill;
    out.zero_fill = range.zero_fill;
    out.image_staging = range.image_staging;
    if (!range.image_staging)
      continue;
    out.width = range.image.width;
    out.height = range.image.height;
    out.pitch = range.image.pitch;
    // A volume's slice count lives in `depth`, not `layers` (which stays 1).
    // Handing the renderer the raw layer count made it rebuild the staging
    // layout at 1/depth of the real size and the dispatch failed outright. Both
    // of P.T.'s volume uploads are its colour-grading LUT, so its tonemap
    // graded every frame through an all-zero table: a black screen from a
    // 16 KiB texture.
    out.layers = range.image.is_3d ? range.image.depth : range.image.layers;
    out.mip_levels = range.image.mip_levels;
    out.tiling_idx = range.image.tiling_idx;
    out.elem_bytes = range.elem_bytes;
    out.stage_elem_bytes = range.stage_elem_bytes;
    out.dfmt = range.image.dfmt;
    out.pow2_pad = range.image.pow2_pad;
  }
  if (!ci.num_res)
    return;

  const bool executed = rhi::Dispatch(renderer, ci);
  if (trace)
    TraceCsDispatch(cs_addr, executed, ci.num_res);
}

}  // namespace gpu::ps4
