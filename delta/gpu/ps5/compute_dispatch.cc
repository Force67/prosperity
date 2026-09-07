/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * Compute dispatches, from COMPUTE_* registers to the renderer. See
 * compute_dispatch.h.
 */

#include "gpu/ps5/compute_dispatch.h"
#include "base/arch.h"

#include <algorithm>
#include <cstring>

#include <base/logging.h>
#include <utl/options.h>

#include "gpu/gcn/gcn_detile.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/guest_memory.h"
#include "gpu/ps5/cmd_trace.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include "gpu/ps5/shader_cache.h"

namespace {
DELTA_OPTION(bool, kNoCs, "DELTA_GPU_NOCS", false);
DELTA_OPTION(const char*, kCsProbe, "DELTA_GPU_CSPROBE", nullptr);
}  // namespace

namespace gpu::ps5 {
namespace {

constexpr u64 kMaxShaderBytes = 4096 * sizeof(u32);
// A storage buffer beyond this is a descriptor we misread.
constexpr u64 kMaxResource = 256ull * 1024 * 1024;

// One resource's live guest range and how it has to be staged.
struct ResourceRange {
  u64 base = 0;
  u64 size = 0;        // bytes the dispatch reads/writes through it
  u64 guest_size = 0;  // bytes it occupies in guest memory
  gcn::TImage image;
  bool image_staging = false;  // tiled or reformatted: stage, don't alias
  bool zero_fill = false;      // no live range: hand the shader zeros
  bool prefer_import = false;  // alias the guest pages, do not copy them
  u32 elem_bytes = 4;
  u32 stage_elem_bytes = 4;
  bool ok = true;
};

// An image the compute backend can stage in and out again. The formats listed
// are the ones whose texels survive the round trip through a linear SSBO; a
// swizzle mode with no address equation, or a layout we cannot rebuild, would
// scramble what the staging copy writes back into guest memory.
ResourceRange ResolveImageResource(u64 cs_addr,
                                   const gcn::CsResource& res,
                                   const u32* descriptor) {
  ResourceRange out;
  const gcn::TImage t = rdna::DecodeTImage(descriptor);
  const bool r8 = t.dfmt == 1 && (t.nfmt == 0 || t.nfmt == 4);
  const bool rgba8 = t.dfmt == 10 && (t.nfmt == 0 || t.nfmt == 4);
  const bool r32 = t.dfmt == 4 && (t.nfmt == 4 || t.nfmt == 5 || t.nfmt == 7);
  const bool rg16f = t.dfmt == 5 && t.nfmt == 7;
  const bool r16f = t.dfmt == 2 && t.nfmt == 7;
  const bool rg8 = t.dfmt == 3 && (t.nfmt == 0 || t.nfmt == 4);
  const bool rgba16 = t.dfmt == 12 && (t.nfmt == 0 || t.nfmt == 7);
  const bool r11g11b10f = t.dfmt == 6 && t.nfmt == 7;
  // 32_32: two full dwords per texel, so it stages unchanged like the other
  // 32-bit-per-channel forms, just twice as wide.
  const bool rg32 = t.dfmt == 11 && (t.nfmt == 4 || t.nfmt == 5 || t.nfmt == 7);
  const bool block128 =
      t.dfmt == 14 && (t.nfmt == 4 || t.nfmt == 5 || t.nfmt == 7);
  out.elem_bytes = block128            ? 16u
                   : (rgba16 || rg32) ? 8u
                   : (r16f || rg8)     ? 2u
                   : r8                ? 1u
                                       : 4u;
  out.stage_elem_bytes = r11g11b10f ? 16u : std::max(out.elem_bytes, 4u);

  // type 10 is a volume: DecodeTImage already reports its depth as layers and
  // the shared emitter addresses 3D slice-major, so it stages like the 2D
  // forms.
  const bool supported_type = t.type >= 8 && t.type <= 13;
  const bool supported_format = r8 || rgba8 || r32 || rg16f || r16f || rg8 ||
                                rgba16 || r11g11b10f || rg32 || block128;
  gcn::TextureLayout32 layout;
  if (!supported_type || !supported_format ||
      !gcn::TilingSupported(t.tiling_idx) || !t.valid ||
      !gcn::BuildTextureLayout32(layout, t.width, t.height, t.pitch, t.layers,
                                 t.mip_levels, t.tiling_idx, t.pow2_pad,
                                 out.elem_bytes)) {
    // One descriptor we cannot stage used to skip the whole dispatch, taking
    // every other binding's work with it. Hand this one zeros instead and let
    // the rest of the shader run.
    TraceCsUnsupportedImage(cs_addr, res.binding, t);
    out.zero_fill = true;
    out.size = std::max<u64>(res.min_bytes, 16);
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
  if (!gcn::BuildTextureLayout32(linear, t.width, t.height, t.pitch, t.layers,
                                 t.mip_levels, 8, t.pow2_pad,
                                 out.stage_elem_bytes)) {
    TraceCsNoLinearStaging(cs_addr, res.binding, t);
    out.ok = false;
    return out;
  }
  out.size = linear.size;
  out.image = t;
  return out;
}

ResourceRange ResolveBufferResource(const gcn::CsResource& res,
                                    const u32* descriptor) {
  ResourceRange out;
  if (res.kind == 2) {  // a raw pointer into an SRT/descriptor table
    out.base = (static_cast<u64>(descriptor[1] & 0xFFFF) << 32) | descriptor[0];
    out.size = res.min_bytes;
    // A global_* base has no extent in the instruction. Its own allocation is
    // the bound the hardware would fault at, so use that and let the range be
    // imported instead of copied.
    u64 pool_base = 0, pool_end = 0;
    if (res.min_bytes >= 0x10000 && out.base &&
        GpuPoolRange(out.base, pool_base, pool_end)) {
      constexpr u64 kMaxWindow = 64ull << 20;
      out.size = std::min<u64>(pool_end - out.base, kMaxWindow);
      out.prefer_import = true;
    }
    return out;
  }
  const rdna::VBuffer v = rdna::DecodeVBuffer(descriptor);
  out.base = v.base;
  out.size = v.stride ? static_cast<u64>(v.stride) * v.num_records
                      : v.num_records;
  out.size = std::max<u64>(out.size, res.min_bytes);
  return out;
}

}  // namespace

void DispatchCompute(rhi::Renderer& renderer,
                     const Regs& regs,
                     const u32* body,
                     u32 count) {
  const u32 groups[3] = {count >= 1 ? body[0] : 0, count >= 2 ? body[1] : 0,
                         count >= 3 ? body[2] : 0};
  const u64 cs_addr = (static_cast<u64>(regs[mmCOMPUTE_PGM_HI] & 0xFF) << 32 |
                       regs[mmCOMPUTE_PGM_LO])
                      << 8;
  const u32 threads[3] = {regs[mmCOMPUTE_NUM_THREAD_X] & 0xFFFF,
                          regs[mmCOMPUTE_NUM_THREAD_Y] & 0xFFFF,
                          regs[mmCOMPUTE_NUM_THREAD_Z] & 0xFFFF};
  const u32 rsrc2 = regs[mmCOMPUTE_PGM_RSRC2];
  const u32 user_sgpr = (rsrc2 >> 1) & 0x1F;

  char probe_buf[32];
  std::snprintf(probe_buf, sizeof(probe_buf), "%llx",
                (unsigned long long)cs_addr);
  if (kCsProbe && std::strstr(probe_buf, kCsProbe)) {
    base::String line;
    base::FormatTo(line, "cs={:#x} groups=[{} {} {}] tg=[{} {} {}] "
                        "user_sgpr={} rsrc2={:#x} ud:",
                   cs_addr, groups[0], groups[1], groups[2], threads[0],
                   threads[1], threads[2], user_sgpr, rsrc2);
    const u32* ud = regs.At(mmCOMPUTE_USER_DATA_0);
    for (int k = 0; k < 16; k++)
      base::FormatTo(line, " {:08x}", ud[k]);
    BASE_LOGI("csprobe", "{}", line.c_str());
  }

  NoteDispatch(cs_addr, threads, rsrc2);
  TraceComputeShader(regs, cs_addr, groups, threads, rsrc2);

  if (!IsGuestAddress(cs_addr) || !threads[0] || !threads[1] || !groups[0] ||
      !groups[1])
    return;
  if (kNoCs || !renderer.available() ||
      !gpu::IsReadableRange(cs_addr, kMaxShaderBytes))
    return;

  // Recompile the CS to a Vulkan compute pipeline (cached), resolve the guest
  // ranges its descriptors name, and run it on the shared compute backend.
  const gcn::RecompiledCs& rc = GetComputeShader({.cs_addr = cs_addr,
                                                  .thread_x = threads[0],
                                                  .thread_y = threads[1],
                                                  .thread_z = threads[2],
                                                  .user_sgpr = user_sgpr,
                                                  .tgid_enable =
                                                      (rsrc2 >> 7) & 0x7,
                                                  .lds_dwords =
                                                      (rsrc2 >> 15) & 0x1FF});
  if (!rc.ok) {
    TraceCsUnsupported(cs_addr, groups, threads, user_sgpr);
    return;
  }

  const u32* ud = regs.At(mmCOMPUTE_USER_DATA_0);
  const u32 ud_dwords = std::min(user_sgpr, 16u);
  rhi::ComputeInfo ci;
  ci.cs_addr = cs_addr;
  ci.groups[0] = groups[0];
  ci.groups[1] = groups[1];
  ci.groups[2] = groups[2];
  ci.recomp = &rc;
  for (int k = 0; k < 16; k++)
    ci.user_data[k] = ud[k];

  // Descriptors an SRT chain produced are not in the user-data window; replay
  // the shader's scalar ops to recover the one each resource's instruction
  // actually used.
  const auto resolved = rdna::ResolveBuffers(
      reinterpret_cast<const u32*>(cs_addr), ud, ud_dwords, 0);

  for (const gcn::CsResource& r : rc.resources) {
    const u32 dwords = r.kind == 1 ? 8u : r.kind == 2 ? 2u : 4u;
    // Compute seeds user data straight into s0.., so a plan naming an SGPR past
    // the loaded window names one an SRT load produced, which nothing here
    // replays. The replay seeds from this same user data, so what it recovered
    // is at least as good: preferring the window would bind whatever the CPU
    // left there for a shader that loaded a descriptor over its own user data.
    const u32* desc = nullptr;
    if (const auto it = resolved.find(r.use_pc);
        it != resolved.end() && it->second.descriptor_valid &&
        it->second.descriptor_dwords >= dwords) {
      desc = it->second.descriptor;
    } else if (r.base_sgpr + dwords <= ud_dwords) {
      desc = &ud[r.base_sgpr];
    } else {
      TraceCsUnresolved(cs_addr, r, ud_dwords);
      return;
    }

    ResourceRange range;
    if (std::all_of(desc, desc + dwords, [](u32 w) { return w == 0; })) {
      // A null descriptor is a real binding on a path this launch does not
      // take; the translator guards it and reads zero.
      range.zero_fill = true;
      range.size = std::max<u64>(r.min_bytes, 16);
    } else if (r.kind == 1) {
      range = ResolveImageResource(cs_addr, r, desc);
    } else {
      range = ResolveBufferResource(r, desc);
    }
    if (!range.ok)
      return;
    if (!range.guest_size && !range.zero_fill)
      range.guest_size = range.size;
    TraceCsResource(cs_addr, r, range.base, range.size, range.guest_size,
                    range.zero_fill);
    // A dispatch writes guest memory, so a range that does not check out skips
    // the whole dispatch rather than binding something wrong.
    if (!range.zero_fill &&
        (range.size < r.min_bytes || range.size > kMaxResource ||
         range.guest_size > kMaxResource || !IsGpuAddress(range.base) ||
         !gpu::IsReadableRange(range.base, range.guest_size))) {
      TraceCsInvalidRange(cs_addr, r, range.base, range.guest_size);
      return;
    }
    if (ci.num_res >= rhi::ComputeInfo::kMaxResources) {
      TraceCsTooManyResources(cs_addr, rhi::ComputeInfo::kMaxResources);
      return;
    }
    rhi::ComputeInfo::Res& out = ci.res[ci.num_res++];
    out.base = range.base;
    out.size = range.size;
    out.guest_size = range.guest_size;
    out.binding = r.binding;
    out.prefer_import = range.prefer_import;
    out.shader_writes = r.written;
    out.written = r.written && !range.zero_fill;
    out.zero_fill = range.zero_fill;
    out.image_staging = range.image_staging;
    if (!range.image_staging)
      continue;
    out.width = range.image.width;
    out.height = range.image.height;
    out.pitch = range.image.pitch;
    out.layers = range.image.layers;
    out.mip_levels = range.image.mip_levels;
    out.tiling_idx = range.image.tiling_idx;
    out.elem_bytes = range.elem_bytes;
    out.stage_elem_bytes = range.stage_elem_bytes;
    out.dfmt = range.image.dfmt;
    out.pow2_pad = range.image.pow2_pad;
  }
  ci.gds_binding = rc.gds_binding;
  if (!ci.num_res)
    return;
  const bool dispatched = rhi::Dispatch(renderer, ci);
  if (!dispatched)
    TraceCsDispatchFailed(cs_addr, ci.num_res);
  TraceCsDispatch(cs_addr, dispatched, ci.num_res);
}

}  // namespace gpu::ps5
