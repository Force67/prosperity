/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * One draw, from gfx10.3 registers and descriptors to the renderer. See
 * draw_state.h.
 */

#include "gpu/ps5/draw_state.h"
#include "base/arch.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <utl/options.h>

#include "gpu/gcn/gcn_translate.h"
#include "gpu/guest_memory.h"
#include "gpu/ps4/pm4.h"
#include "gpu/ps5/cmd_trace.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include "gpu/ps5/shader_cache.h"
#include "gpu/vulkan/vk_perf.h"

namespace {
DELTA_OPTION(bool, kNoDepth, "DELTA_GPU_NODEPTH", false);
DELTA_OPTION(bool, kNoStickyRt, "DELTA_GPU_NOSTICKYRT", false);
DELTA_OPTION(bool, kRecompOn, "DELTA_PS5_RECOMP", true);
DELTA_OPTION(u64, kSkipVs, "DELTA_PS5_SKIPVS", 0);
DELTA_OPTION(u32, kUdBase, "DELTA_PS5_UDBASE", 8);
DELTA_OPTION(bool, kGsIsVs, "DELTA_PS5_GSVS", false);
}  // namespace

namespace gpu::ps5 {
namespace {

constexpr u64 kMaxShaderBytes = 4096 * sizeof(u32);
constexpr u32 kMaxAttrs = 8;

using ResolvedBuffers = std::unordered_map<u32, rdna::BufferResource>;

// --- render-target dimensions ----------------------------------------------

// Derived from the viewport the title programmed: the target spans
// [center-halfSize, center+halfSize] where halfSize = |VPORT_xSCALE|. The
// screen scissor is a "no-clip" max (e.g. 16384) and is NOT the target size.
// Reading the viewport keeps this title-agnostic (no fixed default resolution);
// a 0 scale (no viewport yet) yields 0 so the draw is skipped until real state
// is set.
u32 FbDim(const Regs& regs, u32 scale_reg) {
  float s;
  std::memcpy(&s, regs.At(scale_reg), 4);
  if (s < 0.0f)
    s = -s;
  return static_cast<u32>(s * 2.0f + 0.5f);
}

// CB_COLOR0_ATTRIB2 carries the bound target's real size (MIP0_WIDTH-1 in
// [27:14], MIP0_HEIGHT-1 in [13:0]). Titles whose context state arrives as AGC
// state blocks never program the viewport registers, so the scale-derived size
// reads 0 and every draw is skipped for a zero-sized target; the target's own
// dimensions are the authority in that case.
u32 FbAttrib2Dim(const Regs& regs, bool want_width) {
  const u32 a2 = regs[mmCB_COLOR0_ATTRIB2];
  if (!a2)
    return 0;
  return want_width ? ((a2 >> 14) & 0x3FFF) + 1 : (a2 & 0x3FFF) + 1;
}

u32 FbWidth(const Regs& regs) {
  const u32 v = FbDim(regs, mmPA_CL_VPORT_XSCALE);
  return v ? v : FbAttrib2Dim(regs, true);
}

u32 FbHeight(const Regs& regs) {
  const u32 v = FbDim(regs, mmPA_CL_VPORT_YSCALE);
  return v ? v : FbAttrib2Dim(regs, false);
}

u32 UserSgprCount(u32 rsrc2) {
  const u32 count = ((rsrc2 >> 1) & 0x1F) | (((rsrc2 >> 27) & 1) << 5);
  return std::min(count, 32u);
}

// The T#'s four DST_SEL channel selects, packed 3 bits each for the renderer's
// image view. The identity selection (R,G,B,A) packs to 0 so views that need no
// swizzle keep sharing one cache entry.
u32 PackDstSel(const gcn::TImage& t) {
  const u32 p = (t.dst_sel[0] & 7) | ((t.dst_sel[1] & 7) << 3) |
                ((t.dst_sel[2] & 7) << 6) | ((t.dst_sel[3] & 7) << 9);
  return p == (4u | (5u << 3) | (6u << 6) | (7u << 9)) ? 0u : p | (1u << 12);
}

// --- shader addresses ------------------------------------------------------

// The programs a draw runs, and the user-data window each reads its descriptors
// out of.
struct ShaderBinding {
  u64 vs_addr = 0;
  u64 ps_addr = 0;
  const u32* vs_user_data = nullptr;
  const u32* ps_user_data = nullptr;
};

// Some titles program the shader PGM_LO/HI at non-standard SH offsets (via the
// inline op 0x93), so the fixed GS/PS registers read 0. Fallback: scan the SH
// register block for PGM pairs whose address is a 256-aligned GPU-aperture
// pointer to plausible RDNA2 ISA. It is proven that for the titles that need
// this, NO register in the whole file points at shader code -- the AGC binds
// them outside the PM4 stream -- so this stays inert until that path is
// decoded, and the probes it feeds are what will decode it.
void RecoverShaderAddresses(const Regs& regs, ShaderBinding& binding) {
  u64 found[16];
  u32 found_reg[16];
  u32 nf = 0;
  const auto try_addr = [&](u32 off, u64 a) {
    if (nf >= 16 || !IsGpuAddress(a) || (a & 0xFF) ||
        !gpu::IsReadableRangeCached(a, sizeof(u32)))
      return;
    const u32 word0 = *reinterpret_cast<const u32*>(a);
    if ((word0 >> 24) < 0x7e || word0 == 0xffffffffu)
      return;  // ISA plausibility
    for (u32 k = 0; k < nf; k++)
      if (found[k] == a)
        return;
    found_reg[nf] = off;
    found[nf++] = a;
  };
  for (u32 o = 0; o + 1 < 0x300; o++) {
    const u64 lo = regs[kShRegBase + o], hi = regs[kShRegBase + o + 1];
    try_addr(o, (lo << 8) | ((hi & 0xFF) << 40));
    try_addr(o, ((hi & 0xFFFF) << 32) | lo);
  }
  if (nf >= 1 && !IsGuestAddress(binding.vs_addr))
    binding.vs_addr = found[0];
  if (nf >= 2 && !IsGuestAddress(binding.ps_addr))
    binding.ps_addr = found[1];
  TraceShaderScan(regs, found_reg, found, nf);
  TraceUserDataPointers(binding.vs_user_data, binding.ps_user_data);
}

ShaderBinding ResolveShaderBinding(const Regs& regs) {
  ShaderBinding binding;
  // gfx10.3 has no HW VS: the vertex program is the merged NGG shader, whose
  // address is written to the ES (front half) and/or GS (back half) PGM_LO.
  // The ES slot is the VERTEX program. Usually both slots hold the same merged
  // address and the choice does not matter, but a pass whose GS half is a
  // primitive shader programs them separately -- and then the GS program is
  // one that reads its vertices back out of LDS after the ES half filled it,
  // which as a Vulkan vertex shader exports a degenerate position. Astro Bot's
  // fullscreen passes are all of that shape. DELTA_PS5_GSVS=1 restores the
  // old preference for a bisect.
  const u64 es_addr = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_ES);
  const u64 gs_addr = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_GS);
  binding.vs_addr = kGsIsVs ? gs_addr : es_addr;
  if (!IsGuestAddress(binding.vs_addr))
    binding.vs_addr = kGsIsVs ? es_addr : gs_addr;
  binding.ps_addr = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_PS);

  // A pipeline that only populates the ES half leaves the GS user-data window
  // empty, and then every cbuffer and texture the vertex stage names resolves
  // against zeros. Take the half that actually holds something.
  const u32* gs_user_data = regs.At(mmSPI_SHADER_USER_DATA_GS_0);
  const u32* es_user_data = regs.At(mmSPI_SHADER_USER_DATA_ES_0);
  bool gs_populated = false;
  for (u32 i = 0; i < 16 && !gs_populated; i++)
    gs_populated = gs_user_data[i] != 0;
  binding.vs_user_data = gs_populated ? gs_user_data : es_user_data;
  binding.ps_user_data = regs.At(mmSPI_SHADER_USER_DATA_PS_0);
  TraceUserData(gs_user_data, es_user_data);
  TraceVsUserData(binding.vs_addr, gs_user_data, es_user_data,
                  gs_populated);

  if (!IsGuestAddress(binding.vs_addr) || !IsGuestAddress(binding.ps_addr))
    RecoverShaderAddresses(regs, binding);
  return binding;
}

// --- packet decode ---------------------------------------------------------

// DRAW_INDEX_2 carries the buffer inline (maxSize, baseLo, baseHi, index_count,
// initiator); DRAW_INDEX_OFFSET_2 (maxSize, index_offset, index_count,
// initiator) indexes the buffer INDEX_BASE last set, starting index_offset
// indices in. Minecraft draws its world geometry almost entirely with the
// latter (291k packets a run against 33k AUTO), and leaving it undecoded left
// every one of those draws with no index buffer and a vertex count taken from
// the V#'s num_records -- a shared ~210k-record ring -- so they all tripped the
// vertex-count cap and were dropped.
void ResolveIndexBuffer(const DrawPacket& packet, rhi::DrawInfo& d) {
  const u64 index_size =
      packet.index_type == 1 ? 4 : packet.index_type == 2 ? 1 : 2;
  const u32* body = packet.body;
  if (packet.op == IT_DRAW_INDEX_OFFSET_2 && packet.count >= 3 &&
      packet.index_base) {
    const u32 offset = body[1], count = body[2];
    const u64 base = packet.index_base + static_cast<u64>(offset) * index_size;
    if (IsGuestAddress(base) && count && count <= 0x100000 &&
        (!packet.index_max ||
         static_cast<u64>(offset) + count <= packet.index_max) &&
        gpu::IsReadableRangeCached(base, count * index_size)) {
      d.index_data = reinterpret_cast<const void*>(base);
      d.index_count = count;
      d.index_type = packet.index_type;
    }
  }
  if (packet.op == IT_DRAW_INDEX_2 && packet.count >= 4) {
    const u64 base = (static_cast<u64>(body[2] & 0xFFFF) << 32) | body[1];
    const u32 count = body[3];
    if (IsGuestAddress(base) && count && count <= 0x100000 &&
        gpu::IsReadableRangeCached(base, count * index_size)) {
      d.index_data = reinterpret_cast<const void*>(base);
      d.index_count = count;
      d.index_type = packet.index_type;
    }
  }
  TraceIndexBuffer(packet.op, d, index_size);
}

// --- register state --------------------------------------------------------

// Bind CB_COLORn only when its write mask and INFO format are valid: a stale
// base remains programmed during depth-only passes.
void ResolveRenderTargets(const Regs& regs, rhi::DrawInfo& d) {
  d.rt_w = FbWidth(regs);
  d.rt_h = FbHeight(regs);
  const u32 target_mask = regs[mmCB_TARGET_MASK];
  for (int rt = 0; rt < 8; rt++) {
    const u64 base = regs.CbColorBase(rt);
    const u32 info = regs[mmCB_COLOR0_INFO + rt * kCbColorStride];
    if (((target_mask >> (rt * 4)) & 0xF) && ((info >> 2) & 0x1F) &&
        IsGuestAddress(base)) {
      d.mrt_base[rt] = base;
      d.mrt_info[rt] = info;
      d.mrt_count = rt + 1;
    }
  }
  d.rt_base = d.mrt_count ? d.mrt_base[0] : 0;

  // A draw whose target mask asks for colour but whose CB_COLOR0_INFO reads
  // zero keeps the last target that was valid. Skyrim's logo and menu passes
  // are bound through a path we do not see yet (the driver's default-state
  // block zeroes CB_COLOR0 and only some passes get a re-bind), and dropping
  // them entirely is certainly wrong where reusing the target is only maybe.
  // DELTA_GPU_NOSTICKYRT restores the drop.
  static u64 last_base = 0;
  static u32 last_info = 0, last_w = 0, last_h = 0;
  if (d.mrt_count) {
    last_base = d.mrt_base[0];
    last_info = d.mrt_info[0];
    last_w = d.rt_w;
    last_h = d.rt_h;
  } else if (!kNoStickyRt && (target_mask & 0xF) && last_base &&
             d.rt_w == last_w && d.rt_h == last_h) {
    d.mrt_base[0] = last_base;
    d.mrt_info[0] = last_info;
    d.mrt_count = 1;
    d.rt_base = last_base;
  }
  if (!d.mrt_count && (target_mask & 0xF))
    TraceNoRenderTarget(regs, target_mask);
}

// Per-MRT blend (CB_BLENDn_CONTROL, bit 30 = enable).
void ResolveColorState(const Regs& regs, rhi::DrawInfo& d) {
  d.blend_control = regs[mmCB_BLEND0_CONTROL];
  d.blend_enable = (d.blend_control >> 30) & 1u;
  d.mrt_blend[0] = d.blend_control;
  if (d.blend_enable)
    d.mrt_blend_mask |= 1u;
  for (u32 rt = 1; rt < 8; rt++) {
    const u32 control = regs[mmCB_BLEND0_CONTROL + rt * kCbBlendStride];
    d.mrt_blend[rt] = control;
    if ((control >> 30) & 1u)
      d.mrt_blend_mask |= (1u << rt);
  }
  d.target_mask = regs[mmCB_TARGET_MASK];
  d.color_control = regs[mmCB_COLOR_CONTROL];
}

// Depth/stencil (gfx10 Z base = (WRITE_BASE | WRITE_BASE_HI<<32) << 8). Mirrors
// the PS4 path: DELTA_GPU_NODEPTH is the shared kill switch, otherwise the
// title's own DB_Z_INFO / DB_DEPTH_CONTROL state decides. 2D titles (Isaac)
// leave DB_Z_INFO's format field invalid, so no depth attachment binds.
void ResolveDepthState(const Regs& regs, rhi::DrawInfo& d) {
  const u32 control = regs[mmDB_DEPTH_CONTROL];
  const u32 z_info = kNoDepth ? 0 : regs[mmDB_Z_INFO];
  const u64 z_base =
      ((static_cast<u64>(regs[mmDB_Z_WRITE_BASE_HI]) << 32) |
       regs[mmDB_Z_WRITE_BASE])
      << 8;
  d.depth_valid = (z_info & 0x3) != 0;
  if (!d.depth_valid || !IsGuestAddress(z_base) ||
      !(((control >> 1) & 1u) || ((control >> 2) & 1u))) {
    d.depth_valid = false;
    return;
  }
  d.depth_base = z_base;
  d.depth_test_enable = (control >> 1) & 1u;
  d.depth_write_enable = (control >> 2) & 1u;
  d.depth_func = (control >> 4) & 0x7;
  std::memcpy(&d.depth_clear, regs.At(mmDB_DEPTH_CLEAR), 4);
  if (!(d.depth_clear >= 0.0f && d.depth_clear <= 1.0f))
    d.depth_clear = 1.0f;
}

void ResolveRasterState(const Regs& regs, rhi::DrawInfo& d) {
  const u32 mode = regs[mmPA_SU_SC_MODE_CNTL];
  d.cull_mode = mode & 0x3;
  d.front_ccw = ((mode >> 2) & 1u) == 0;
  std::memcpy(&d.viewport_x_scale, regs.At(mmPA_CL_VPORT_XSCALE), 4);
  std::memcpy(&d.viewport_x_offset, regs.At(mmPA_CL_VPORT_XOFFSET), 4);
  std::memcpy(&d.viewport_y_scale, regs.At(mmPA_CL_VPORT_YSCALE), 4);
  std::memcpy(&d.viewport_y_offset, regs.At(mmPA_CL_VPORT_YOFFSET), 4);
  // Titles whose context state arrives as AGC state blocks never program
  // PA_CL_VPORT_*, and the renderer rejects a zero scale -- so no viewport is
  // ever set and every draw rasterises nothing. Default to the bound target's
  // full extent, y-up (negative height) like a programmed one, so RT-as-texture
  // composites still sample aligned.
  if (!(d.viewport_x_scale > 0.0f) && d.rt_w && d.rt_h) {
    d.viewport_x_scale = d.rt_w * 0.5f;
    d.viewport_x_offset = d.rt_w * 0.5f;
    d.viewport_y_scale = d.rt_h * -0.5f;
    d.viewport_y_offset = d.rt_h * 0.5f;
  }
}

// --- the recompiled-shader path --------------------------------------------

// Resolve each attribute's V#, then group the attributes into vertex bindings:
// same-stride V#s within one stride of each other interleave in one binding,
// others get their own (a textured sprite VS streams pos/colour and uv/params
// from two buffers, so a single interleaved binding would feed the PS garbage
// UVs). Attributes that do not decode to a valid V# are skipped -- a partial
// fetch still rasterises.
void BindVertexAttributes(const gcn::Recompiled& rc,
                          const ResolvedBuffers& resolved,
                          const u32* vs_user_data,
                          u32 vs_user_sgprs,
                          rhi::DrawInfo& d) {
  rdna::VBuffer attr_vbs[kMaxAttrs];
  const gcn::ShaderAttr* attr_res[kMaxAttrs];
  u32 attr_n = 0;
  for (size_t i = 0; i < rc.attrs.size() && i < kMaxAttrs; i++) {
    const gcn::ShaderAttr& a = rc.attrs[i];
    rdna::VBuffer vb{};
    u32 fetch_soffset = 0;
    const char* how = "replay";
    if (a.use_pc != ~0u) {
      const auto it = resolved.find(a.use_pc);
      TraceAttrReplay(static_cast<u32>(i), a.use_pc, it != resolved.end(),
                      it != resolved.end() && it->second.descriptor_valid);
      if (it == resolved.end() || !it->second.descriptor_valid)
        continue;
      vb = rdna::DecodeVBuffer(it->second.descriptor);
      fetch_soffset = it->second.soffset;
    } else {
      // The V# is either inline in user data at table_sgpr (AGC frequently
      // passes it directly) or reached through a table pointer there.
      const u32 slot =
          a.table_sgpr >= kUdBase ? a.table_sgpr - kUdBase : a.table_sgpr;
      if (slot + 3 >= vs_user_sgprs)
        continue;
      vb = rdna::DecodeVBuffer(&vs_user_data[slot]);
      how = "inline";
      if (!IsGuestAddress(vb.base) || !rdna::PlausibleVBuffer(vb)) {
        const u64 table =
            (static_cast<u64>(vs_user_data[slot + 1] & 0xFFFF) << 32) |
            vs_user_data[slot];
        const u64 entry = table + a.vbuf_dword_off * 4;
        if (IsGuestAddress(table) && gpu::IsReadableRangeCached(entry, 16)) {
          vb = rdna::DecodeVBuffer(reinterpret_cast<const u32*>(entry));
          how = "table";
        }
      }
    }
    TraceAttr(static_cast<u32>(i), a, vb, fetch_soffset, how);
    if (!IsGuestAddress(vb.base) || !rdna::PlausibleVBuffer(vb))
      continue;  // unresolved: keep the rest
    // Where this attribute sits inside the vertex. The V# only names the
    // stream: the fetch adds its own byte offset, either as the immediate or
    // (Sony's compiler) through the soffset scalar. Fold it into the base so
    // the grouping below turns it back into a per-attribute offset.
    const u64 field_off = a.inst_offset + fetch_soffset;
    if (field_off < vb.stride)
      vb.base += field_off;
    attr_vbs[attr_n] = vb;
    attr_res[attr_n++] = &a;
  }

  u32 attr_binding[kMaxAttrs] = {};
  for (u32 i = 0; i < attr_n; i++) {
    const rdna::VBuffer& vb = attr_vbs[i];
    int sel = -1;
    for (u32 j = 0; j < d.num_vbufs; j++) {
      if (d.vbufs[j].stride != vb.stride)
        continue;
      const u64 b = reinterpret_cast<u64>(d.vbufs[j].data);
      const u64 lo = b < vb.base ? b : vb.base;
      const u64 hi = b < vb.base ? vb.base : b;
      if (hi - lo < vb.stride) {
        sel = static_cast<int>(j);
        break;
      }
    }
    if (sel < 0) {
      if (d.num_vbufs >= 8)
        break;
      sel = static_cast<int>(d.num_vbufs);
      d.vbufs[d.num_vbufs++] = {reinterpret_cast<const void*>(vb.base),
                                vb.stride, vb.num_records};
    } else {
      auto& bind = d.vbufs[sel];
      if (vb.base < reinterpret_cast<u64>(bind.data))
        bind.data = reinterpret_cast<const void*>(vb.base);
      bind.num_records = std::min(bind.num_records, vb.num_records);
    }
    attr_binding[i] = static_cast<u32>(sel);
  }

  // Offsets are relative to each binding's final (lowest) base.
  for (u32 i = 0; i < attr_n && d.num_vattrs < 8; i++) {
    const rdna::VBuffer& vb = attr_vbs[i];
    const u32 b = attr_binding[i];
    if (b >= d.num_vbufs)
      continue;
    const u64 off = vb.base - reinterpret_cast<u64>(d.vbufs[b].data);
    if (off >= d.vbufs[b].stride)
      continue;
    // A typed fetch (tbuffer_load_format_*) carries its own format and the
    // hardware ignores the V#'s; Skyrim's world positions come in that way.
    u32 dfmt = vb.dfmt, nfmt = vb.nfmt;
    if (attr_res[i]->inst_format)
      rdna::DecodeBufferFormat(attr_res[i]->inst_format, dfmt, nfmt);
    d.vattrs[d.num_vattrs++] = {attr_res[i]->location,  b,
                                static_cast<u32>(off),  attr_res[i]->num_comps,
                                dfmt,                   nfmt};
  }

  if (!d.num_vbufs)
    return;
  // Mirror binding 0 into the legacy single-stream fields; the vertex count is
  // bounded by the smallest binding's record count.
  d.vertex_data = d.vbufs[0].data;
  d.vertex_stride = d.vbufs[0].stride;
  u32 count = UINT32_MAX;
  for (u32 j = 0; j < d.num_vbufs; j++)
    count = std::min(count, d.vbufs[j].num_records);
  d.vertex_count = count;
}

void ResolveCbufferBindings(const std::vector<gcn::ShaderCbuf>& cbufs,
                            const ResolvedBuffers& resolved,
                            bool vertex_stage,
                            rhi::DrawInfo& d) {
  for (const gcn::ShaderCbuf& cb : cbufs) {
    if (cb.binding >= gpu::gcn::kMaxCbufBindings)
      continue;
    const auto it = resolved.find(cb.use_pc);
    if (it == resolved.end())
      continue;
    // The window need not start at the buffer: a shader reading one constant
    // far into a large buffer is bound at that dword and indexes relative to it
    // (ShaderCbuf::first_dword).
    const u64 base =
        (it->second.base & ~u64{3}) + static_cast<u64>(cb.first_dword) * 4;
    const u64 bytes = static_cast<u64>(cb.num_dwords) * 4;
    TraceCbufBinding(vertex_stage, cb.binding, cb.use_pc, base, cb.num_dwords);
    if (!IsGuestAddress(base) || !gpu::IsReadableRangeCached(base, bytes))
      continue;
    d.cbufs[cb.binding] = {base, static_cast<u32>(bytes)};
    d.num_cbufs = std::max(d.num_cbufs, cb.binding + 1);
    if (vertex_stage && !cb.first_dword && bytes >= sizeof(d.mvp)) {
      d.cbuf_base = base;
      d.cbuf_size = static_cast<u32>(bytes);
      std::memcpy(d.mvp, reinterpret_cast<const void*>(base), sizeof(d.mvp));
    }
  }
}

// Raw MUBUF buffers the shader indexes itself (set 2). Same per-pc descriptor
// replay as the cbuffers.
void ResolveRawBuffers(const std::vector<gcn::ShaderBuffer>& buffers,
                       const ResolvedBuffers& resolved,
                       const u32* user_data,
                       u32 user_sgprs,
                       rhi::DrawInfo& d,
                       bool vertex_stage) {
  for (const gcn::ShaderBuffer& sb : buffers) {
    if (sb.binding >= rhi::DrawInfo::kMaxBuffers)
      continue;
    const auto it = resolved.find(sb.use_pc);
    // A global_load names its window with a raw 64-bit pointer pair rather than
    // a V#, tagged srsrc_sgpr >= kRawBufTag (rdna_translate.cc kFlatBaseTag) so
    // it can never be V#-decoded by accident. Its length is not expressed
    // anywhere, so bind a fixed window and let the mapping check drop it if the
    // guest does not actually own that much.
    constexpr u32 kRawBufTag = 0x100;
    if (sb.srsrc_sgpr >= kRawBufTag) {
      constexpr u32 kRawBufWindow = 1u << 20;
      u64 base = 0;
      if (it != resolved.end() && it->second.descriptor_valid &&
          it->second.descriptor_dwords >= 2)
        base = it->second.descriptor[0] |
               (static_cast<u64>(it->second.descriptor[1] & 0xFFFF) << 32);
      else if (const u32 s = sb.srsrc_sgpr - kRawBufTag; s + 1 < user_sgprs)
        base = user_data[s] |
               (static_cast<u64>(user_data[s + 1] & 0xFFFF) << 32);
      const bool ok = IsGuestAddress(base) &&
                      gpu::IsReadableRangeCached(base, kRawBufWindow);
      TraceRawBufBinding(vertex_stage, sb.binding, sb.use_pc, sb.srsrc_sgpr,
                         it != resolved.end(), base, ok ? kRawBufWindow : 0);
      if (!ok)
        continue;
      d.bufs[sb.binding] = {base, kRawBufWindow};
      d.num_bufs = std::max(d.num_bufs, sb.binding + 1);
      continue;
    }
    rdna::VBuffer vb{};
    if (it != resolved.end() && it->second.descriptor_valid)
      vb = rdna::DecodeVBuffer(it->second.descriptor);
    else if (sb.srsrc_sgpr + 3 < user_sgprs)
      vb = rdna::DecodeVBuffer(&user_data[sb.srsrc_sgpr]);
    const u64 bytes =
        vb.stride ? static_cast<u64>(vb.stride) * vb.num_records
                  : vb.num_records;
    const bool ok =
        IsGuestAddress(vb.base) && bytes && bytes <= 0xFFFFFFFFull;
    TraceRawBufBinding(vertex_stage, sb.binding, sb.use_pc, sb.srsrc_sgpr,
                       it != resolved.end() && it->second.descriptor_valid,
                       vb.base, ok ? bytes : 0);
    if (!ok)
      continue;
    // Hand the shader the descriptor the replay walked to. A V# that arrives
    // through an s_load is invisible to the recompiled module (descriptor
    // loads emit nothing), so an in-shader read of the descriptor's own fields
    // -- the STRIDE an indexed buffer_load multiplies by -- came back zero and
    // every vertex of a draw read the same record.
    if (it != resolved.end() && it->second.descriptor_valid &&
        it->second.descriptor_dwords >= 4 && sb.srsrc_sgpr >= kUdBase &&
        sb.srsrc_sgpr + 4 <= kUdBase + 16) {
      u32* ud = vertex_stage ? d.vs_user_data : d.ps_user_data;
      std::memcpy(&ud[sb.srsrc_sgpr - kUdBase], it->second.descriptor,
                  4 * sizeof(u32));
    }
    d.bufs[sb.binding] = {vb.base, static_cast<u32>(bytes)};
    d.num_bufs = std::max(d.num_bufs, sb.binding + 1);
  }
}

// Resolve the live gfx10.3 T#/S# each PS sampler reads, in the recompiler's
// set-0 binding order (rdna::TrackTextures re-derives the same plan), so
// texs[i] maps to PS sampler binding i.
void ResolvePsTextures(u64 ps_addr,
                       const u32* ps_user_data,
                       u32 ps_user_sgprs,
                       rhi::DrawInfo& d) {
  const auto texs = rdna::TrackTextures(
      reinterpret_cast<const u32*>(ps_addr), ps_user_data, ps_user_sgprs);
  if (texs.empty())
    return;
  // The single-texture render path reads the legacy tex* mirror of texs[0], so
  // populate it too (the PS4 path does the same).
  d.tex_base = texs[0].valid ? texs[0].base : 0;
  d.tex_swizzle = PackDstSel(texs[0]);
  d.tex_w = texs[0].width;
  d.tex_h = texs[0].height;
  d.tex_dfmt = texs[0].dfmt;
  d.tex_nfmt = texs[0].nfmt;
  d.tex_tiling = texs[0].tiling_idx;
  d.tex_pitch = texs[0].pitch;
  d.tex_layers = texs[0].layers;
  d.tex_base_array = texs[0].base_array;
  d.tex_view_layers = texs[0].view_layers;
  d.tex_mip_levels = texs[0].mip_levels;
  d.tex_base_mip = texs[0].base_mip;
  d.tex_view_mips = texs[0].view_mips;
  d.tex_min_lod = texs[0].min_lod;
  std::memcpy(d.tex_sampler, texs[0].sampler, sizeof(d.tex_sampler));
  d.tex_pow2_pad = texs[0].pow2_pad;
  d.tex_sampler_valid = texs[0].sampler_valid;
  d.tex_arrayed = texs[0].arrayed;
  d.tex_is_3d = texs[0].is_3d;
  d.tex_force_lod_zero = texs[0].force_lod_zero;
  d.tex_depth_compare = texs[0].depth_compare;
  d.tex_null_descriptor = texs[0].null_descriptor;

  for (size_t i = 0; i < texs.size() && i < 16; i++) {
    const gcn::TImage& s = texs[i];
    rhi::DrawInfo::DrawTex& dt = d.texs[i];
    TraceRejectedTexture(static_cast<u32>(i), s);
    dt.base = s.valid ? s.base : 0;
    dt.w = s.width;
    dt.h = s.height;
    dt.tiling = s.tiling_idx;
    dt.pitch = s.pitch;
    dt.dfmt = s.dfmt;
    dt.nfmt = s.nfmt;
    dt.layers = s.layers;
    dt.base_array = s.base_array;
    dt.view_layers = s.view_layers;
    dt.mip_levels = s.mip_levels;
    dt.base_mip = s.base_mip;
    dt.view_mips = s.view_mips;
    dt.min_lod = s.min_lod;
    std::memcpy(dt.sampler, s.sampler, sizeof(dt.sampler));
    dt.sampler_valid = s.sampler_valid;
    dt.arrayed = s.arrayed;
    dt.is_3d = s.is_3d;
    dt.force_lod_zero = s.force_lod_zero;
    dt.depth_compare = s.depth_compare;
    dt.storage = s.storage;
    dt.null_descriptor = s.null_descriptor;
    dt.swizzle = PackDstSel(s);
  }
  d.num_texs = static_cast<u32>(std::min<size_t>(texs.size(), 16));
  TraceDrawTextures(d);
}

// Recompile the VS/PS pair (cached) and resolve the live vertex-attribute
// buffers, constant buffers and textures from the RDNA2 descriptors in user
// data. Leaves d.recomp null when the draw cannot run, which the caller reports.
// `shader_attrs` is how many attributes the shader wanted, for that report.
void ResolveRecompiledShaders(const Regs& regs,
                              const ShaderBinding& binding,
                              rhi::DrawInfo& d,
                              size_t& shader_attrs) {
  // An unprogrammed RSRC2 reports no user SGPRs at all, which makes every
  // cbuffer and vertex descriptor unreachable; the window picked above is the
  // better evidence that they exist.
  u32 vs_user_sgprs = UserSgprCount(regs[mmSPI_SHADER_PGM_RSRC2_GS]);
  if (!vs_user_sgprs) {
    for (u32 i = 0; i < 16; i++)
      if (binding.vs_user_data[i])
        vs_user_sgprs = 16;
  }
  const u32 ps_user_sgprs = UserSgprCount(regs[mmSPI_SHADER_PGM_RSRC2_PS]);
  // Fetch-shader pointer (a heuristic default: GS user data[0..1]; the AGC
  // input-usage table is authoritative and a follow-up).
  const u64 fetch = vs_user_sgprs >= 2
                        ? (static_cast<u64>(binding.vs_user_data[1] & 0xFFFF)
                           << 32) |
                              binding.vs_user_data[0]
                        : 0;

  NoteDrawDetail();
  if (!kRecompOn || !IsGuestAddress(binding.vs_addr) ||
      (binding.ps_addr && !IsGuestAddress(binding.ps_addr)))
    return;
  if (!gpu::IsReadableRangeCached(binding.vs_addr, kMaxShaderBytes) ||
      (binding.ps_addr &&
       !gpu::IsReadableRangeCached(binding.ps_addr, kMaxShaderBytes)))
    return;

  const gcn::Recompiled& rc = GetGraphicsShader(
      {.vs_addr = binding.vs_addr,
       .ps_addr = binding.ps_addr,
       .fetch_addr = fetch,
       .vs_user_sgprs = vs_user_sgprs,
       .ps_user_sgprs = ps_user_sgprs,
       .ps_input_ena = regs[mmSPI_PS_INPUT_ENA],
       .ps_in_cntl = regs.At(mmSPI_PS_INPUT_CNTL_0),
       .ps_num_interp = regs[mmSPI_PS_IN_CONTROL] & 0x3F,
       // DX_CLIP_SPACE_DEF (bit 19) picks the guest's clip-z convention.
       .gl_clip = !((regs[mmPA_CL_CLIP_CNTL] >> 19) & 1),
       .vs_user_data = binding.vs_user_data,
       .ps_user_data = binding.ps_user_data});
  if (!rc.ok)
    return;

  // The merged ES/GS NGG vertex shader reads its GS user data starting at wave
  // SGPR udBase (sgpr 0..udBase-1 are ES/system), but the AGC latches it into
  // SPI_SHADER_USER_DATA_GS_0 which we index from 0, so shader SGPR N maps to
  // user_data[N-udBase] for both attributes and cbuffers. Defaults to 8 (the
  // observed merged-NGG layout); DELTA_PS5_UDBASE overrides.
  // TODO: derive from RSRC2.
  const ResolvedBuffers vs_resources =
      rdna::ResolveBuffers(reinterpret_cast<const u32*>(binding.vs_addr),
                           binding.vs_user_data, vs_user_sgprs, kUdBase);
  TraceAttrPlan(rc.attrs.size(), vs_user_sgprs, vs_resources.size(),
                binding.vs_user_data);
  BindVertexAttributes(rc, vs_resources, binding.vs_user_data, vs_user_sgprs,
                       d);
  shader_attrs = rc.attrs.size();

  // A fetch VS needs at least one attribute resolved; a procedural VS (no
  // recovered attrs, seeds from VertexIndex) draws without a vertex buffer. A
  // fullscreen pass binds NO vertex buffer at all even though its shader
  // contains fetch instructions (the same shader is used both ways), so
  // demanding a resolved attribute there discarded the whole pass. Let it
  // through with no vertex inputs declared instead.
  bool good = d.num_vattrs > 0 || rc.attrs.empty();
  if (!good && !d.num_vbufs) {
    d.num_vattrs = 0;
    good = true;
  }
  // Before the drop below, not after: replaying the PS's scalar ops reads guest
  // memory a compute dispatch may still own, which lands its pending writes
  // (gcn::g_flush_guest_range). A dropped draw still owes the next one that
  // flush.
  const ResolvedBuffers ps_resources =
      binding.ps_addr
          ? rdna::ResolveBuffers(
                reinterpret_cast<const u32*>(binding.ps_addr),
                binding.ps_user_data, ps_user_sgprs)
          : ResolvedBuffers{};
  if (!good) {
    d.num_vattrs = 0;
    return;
  }
  ResolveCbufferBindings(rc.vs_cbufs, vs_resources, true, d);
  ResolveRawBuffers(rc.vs_bufs, vs_resources, binding.vs_user_data,
                    vs_user_sgprs, d, true);
  if (binding.ps_addr) {
    ResolveCbufferBindings(rc.ps_cbufs, ps_resources, false, d);
    ResolveRawBuffers(rc.ps_bufs, ps_resources, binding.ps_user_data,
                      ps_user_sgprs, d, false);
    if (!rc.ps_texs.empty())
      ResolvePsTextures(binding.ps_addr, binding.ps_user_data, ps_user_sgprs,
                        d);
  }
  d.vs_addr = binding.vs_addr;
  d.ps_addr = binding.ps_addr;
  d.recomp = &rc;
}

}  // namespace

bool BuildDrawInfo(const Regs& regs,
                   const DrawPacket& packet,
                   rhi::DrawInfo& d) {
  vk::ScopeNs _build_timer(&vk::g_ns_build_draw);
  vk::g_build_draw_n++;
  const ShaderBinding binding = ResolveShaderBinding(regs);
  TraceShaderListing(binding.vs_addr);
  TraceShaderListing(binding.ps_addr);
  // DELTA_PS5_SKIPVS=<hexaddr>: drop draws using this VS (draw-isolation
  // bisect).
  if (kSkipVs && binding.vs_addr == kSkipVs)
    return false;

  std::memcpy(d.vs_user_data, binding.vs_user_data, sizeof(d.vs_user_data));
  std::memcpy(d.ps_user_data, binding.ps_user_data, sizeof(d.ps_user_data));
  d.prim_type = regs[mmVGT_PRIMITIVE_TYPE];
  d.instance_count = packet.num_instances;
  ResolveIndexBuffer(packet, d);
  ResolveRenderTargets(regs, d);
  ResolveColorState(regs, d);
  ResolveDepthState(regs, d);
  ResolveRasterState(regs, d);
  TraceRenderTarget(regs, d, binding.vs_addr, binding.ps_addr);

  size_t shader_attrs = 0;
  ResolveRecompiledShaders(regs, binding, d, shader_attrs);

  const u32 auto_vertex_count =
      packet.op == IT_DRAW_INDEX_AUTO && packet.count >= 1 ? packet.body[0] : 0;
  if (auto_vertex_count && auto_vertex_count <= 0x100000)
    d.vertex_count = auto_vertex_count;

  TraceRtProbe(regs, d);
  if (!d.recomp) {
    // No usable shader pair: the draw is discarded here, which looks exactly
    // like the title never issuing it.
    NoteDrawDropped(d, binding.vs_addr, binding.ps_addr, shader_attrs);
    return false;
  }
  TraceVertexDump(d, binding.vs_user_data, binding.vs_addr, binding.ps_addr);
  return true;
}

}  // namespace gpu::ps5
