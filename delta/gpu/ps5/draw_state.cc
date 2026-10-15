/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/ps5/draw_state.h"
#include "base/arch.h"
#include "guest/session.h"

#include <cstring>

#include "base/containers/array.h"
#include "base/logging.h"
#include "options/options.h"

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gpu_perf.h"
#include "gpu/guest_memory.h"
#include "gpu/ps4/pm4.h"
#include "gpu/ps5/cmd_trace.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include "gpu/ps5/shader_cache.h"

namespace {
DELTA_OPTION(bool, kNoDepth, "DELTA_GPU_NODEPTH", false);
DELTA_OPTION(bool, kNoStickyRt, "DELTA_GPU_NOSTICKYRT", false);
DELTA_OPTION(bool, kRecompOn, "DELTA_PS5_RECOMP", true);
DELTA_OPTION(u64, kSkipVs, "DELTA_PS5_SKIPVS", 0);
DELTA_OPTION(u32, kUdBase, "DELTA_PS5_UDBASE", 8);
DELTA_OPTION(bool, kGsIsVs, "DELTA_PS5_GSVS", false);
DELTA_OPTION(bool, kCbResolve, "DELTA_GPU_CBRESOLVE", false);
DELTA_OPTION(bool, kDimTrace, "DELTA_GPU_TEXDIM", false);
}  // namespace

namespace gpu::ps5 {
namespace {

constexpr u64 kMaxShaderBytes = 4096 * sizeof(u32);
constexpr u32 kMaxAttrs = 8;

using ResolvedBuffers = base::HashMap<u32, rdna::BufferResource>;

u32 FbDim(const Regs& regs, u32 scale_reg) {
  float s;
  std::memcpy(&s, regs.At(scale_reg), 4);
  if (s < 0.0f)
    s = -s;
  return static_cast<u32>(s * 2.0f + 0.5f);
}

u32 SurfaceDim(const Regs& regs, int rt, bool want_width) {
  const u32 a2 = regs[mmCB_COLOR0_ATTRIB2 + rt * kCbColorExtStride];
  if (!a2)
    return 0;
  return want_width ? ((a2 >> 14) & 0x3FFF) + 1 : (a2 & 0x3FFF) + 1;
}

u32 FbAttrib2Dim(const Regs& regs, bool want_width) {
  return SurfaceDim(regs, 0, want_width);
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
  return base::Min(count, 32u);
}

u32 PackDstSel(const gcn::TImage& t) {
  const u32 p = (t.dst_sel[0] & 7) | ((t.dst_sel[1] & 7) << 3) |
                ((t.dst_sel[2] & 7) << 6) | ((t.dst_sel[3] & 7) << 9);
  return p == (4u | (5u << 3) | (6u << 6) | (7u << 9)) ? 0u : p | (1u << 12);
}

struct ShaderBinding {
  u64 vs_addr = 0;
  u64 ps_addr = 0;
  u64 es_addr = 0;
  u64 gs_addr = 0;
  const u32* vs_user_data = nullptr;
  const u32* ps_user_data = nullptr;
};

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
      return;
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
  TraceShaderScan(found_reg, found, nf);
  TraceUserDataPointers(binding.vs_user_data, binding.ps_user_data);
}

ShaderBinding ResolveShaderBinding(const Regs& regs) {
  ShaderBinding binding;
  const u64 es_addr = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_ES);
  const u64 gs_addr = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_GS);
  binding.vs_addr = kGsIsVs ? gs_addr : es_addr;
  if (!IsGuestAddress(binding.vs_addr))
    binding.vs_addr = kGsIsVs ? es_addr : gs_addr;
  binding.ps_addr = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_PS);
  binding.es_addr = es_addr;
  binding.gs_addr = gs_addr;
  TraceNggState(regs, es_addr, gs_addr);

  const u32* gs_user_data = regs.At(mmSPI_SHADER_USER_DATA_GS_0);
  const u32* es_user_data = regs.At(mmSPI_SHADER_USER_DATA_ES_0);
  bool gs_populated = false;
  for (u32 i = 0; i < 16 && !gs_populated; i++)
    gs_populated = gs_user_data[i] != 0;
  binding.vs_user_data = gs_populated ? gs_user_data : es_user_data;
  binding.ps_user_data = regs.At(mmSPI_SHADER_USER_DATA_PS_0);
  TraceUserData(gs_user_data, es_user_data);
  TraceVsUserData(binding.vs_addr, gs_user_data, es_user_data, gs_populated);

  if (!IsGuestAddress(binding.vs_addr) || !IsGuestAddress(binding.ps_addr))
    RecoverShaderAddresses(regs, binding);
  return binding;
}

void ResolveIndexBuffer(const DrawPacket& packet, render::DrawInfo& d) {
  const u64 index_size = packet.index_type == 1   ? 4
                         : packet.index_type == 2 ? 1
                                                  : 2;
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

void ResolveRenderTargets(const Regs& regs, render::DrawInfo& d) {
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
      d.mrt_surf_w[rt] = SurfaceDim(regs, rt, true);
      d.mrt_surf_h[rt] = SurfaceDim(regs, rt, false);
      d.mrt_clear_word[rt][0] =
          regs[mmCB_COLOR0_CLEAR_WORD0 + rt * kCbColorStride];
      d.mrt_clear_word[rt][1] =
          regs[mmCB_COLOR0_CLEAR_WORD1 + rt * kCbColorStride];
      if ((info >> 28) & 1u) {
        const u64 dcc =
            ((static_cast<u64>(regs[mmCB_COLOR0_DCC_BASE_EXT + rt]) << 32) |
             regs[mmCB_COLOR0_DCC_BASE + rt * kCbColorStride])
            << 8;
        if (IsGuestAddress(dcc))
          d.mrt_dcc_base[rt] = dcc;
      }
    }
  }
  d.rt_base = d.mrt_count ? d.mrt_base[0] : 0;

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

void ResolveColorState(const Regs& regs, render::DrawInfo& d) {
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
  for (u32 c = 0; c < 4; c++) {
    const u32 raw = regs[mmCB_BLEND_RED + c];
    std::memcpy(&d.blend_constants[c], &raw, sizeof(float));
  }
  d.target_mask = regs[mmCB_TARGET_MASK];
  d.color_control = regs[mmCB_COLOR_CONTROL];
}

void ResolveDepthState(const Regs& regs, render::DrawInfo& d) {
  const u32 control = regs[mmDB_DEPTH_CONTROL];
  d.depth_control = control;
  d.render_control = regs[mmDB_RENDER_CONTROL];
  d.depth_clear_draw = (d.render_control & 1u) != 0;
  const u32 z_info = kNoDepth ? 0 : regs[mmDB_Z_INFO];
  const u64 z_base = ((static_cast<u64>(regs[mmDB_Z_WRITE_BASE_HI]) << 32) |
                      regs[mmDB_Z_WRITE_BASE])
                     << 8;
  d.depth_valid = (z_info & 0x3) != 0;
  if (!d.depth_valid || !IsGuestAddress(z_base) ||
      !((control & 1u) || ((control >> 1) & 1u) || ((control >> 2) & 1u))) {
    d.depth_valid = false;
    return;
  }
  d.depth_base = z_base;
  d.depth_slice = regs[mmDB_DEPTH_VIEW] & 0x7FF;
  const u64 htile = ((static_cast<u64>(regs[mmDB_HTILE_DATA_BASE_HI]) << 32) |
                     regs[mmDB_HTILE_DATA_BASE])
                    << 8;
  if (IsGuestAddress(htile))
    d.depth_htile_base = htile;
  d.depth_test_enable = (control >> 1) & 1u;
  d.depth_write_enable = (control >> 2) & 1u;
  d.depth_func = (control >> 4) & 0x7;
  std::memcpy(&d.depth_clear, regs.At(mmDB_DEPTH_CLEAR), 4);
  if (!(d.depth_clear >= 0.0f && d.depth_clear <= 1.0f))
    d.depth_clear = 1.0f;
  const u64 stencil_base =
      ((static_cast<u64>(regs[mmDB_STENCIL_WRITE_BASE_HI]) << 32) |
       regs[mmDB_STENCIL_WRITE_BASE])
      << 8;
  d.stencil_enable = (control & 1u) && (regs[mmDB_STENCIL_INFO] & 1u) &&
                     IsGuestAddress(stencil_base);
  d.stencil_clear_draw =
      d.stencil_enable && ((d.render_control >> 1) & 1u) != 0;
  if (d.stencil_enable) {
    d.stencil_base = stencil_base;
    d.stencil_backface_enable = (control >> 7) & 1u;
    d.stencil_clear = regs[mmDB_STENCIL_CLEAR] & 0xFF;
    d.stencil_control = regs[mmDB_STENCIL_CONTROL];
    d.stencil_refmask = regs[mmDB_STENCILREFMASK];
    d.stencil_refmask_bf = regs[mmDB_STENCILREFMASK_BF];
  }
}

void ResolveRasterState(const Regs& regs, render::DrawInfo& d) {
  const u32 mode = regs[mmPA_SU_SC_MODE_CNTL];
  d.cull_mode = mode & 0x3;
  d.front_ccw = ((mode >> 2) & 1u) == 0;
  std::memcpy(&d.viewport_x_scale, regs.At(mmPA_CL_VPORT_XSCALE), 4);
  std::memcpy(&d.viewport_x_offset, regs.At(mmPA_CL_VPORT_XOFFSET), 4);
  std::memcpy(&d.viewport_y_scale, regs.At(mmPA_CL_VPORT_YSCALE), 4);
  std::memcpy(&d.viewport_y_offset, regs.At(mmPA_CL_VPORT_YOFFSET), 4);
  if (!(d.viewport_x_scale > 0.0f) && d.rt_w && d.rt_h) {
    d.viewport_x_scale = d.rt_w * 0.5f;
    d.viewport_x_offset = d.rt_w * 0.5f;
    d.viewport_y_scale = d.rt_h * -0.5f;
    d.viewport_y_offset = d.rt_h * 0.5f;
  }
}

void BindVertexAttributes(const gcn::Recompiled& rc,
                          const ResolvedBuffers& resolved,
                          const u32* vs_user_data,
                          u32 vs_user_sgprs,
                          render::DrawInfo& d) {
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
      continue;
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
      bind.num_records = base::Min(bind.num_records, vb.num_records);
    }
    attr_binding[i] = static_cast<u32>(sel);
  }

  for (u32 i = 0; i < attr_n && d.num_vattrs < 8; i++) {
    const rdna::VBuffer& vb = attr_vbs[i];
    const u32 b = attr_binding[i];
    if (b >= d.num_vbufs)
      continue;
    const u64 off = vb.base - reinterpret_cast<u64>(d.vbufs[b].data);
    if (off >= d.vbufs[b].stride)
      continue;
    u32 dfmt = vb.dfmt, nfmt = vb.nfmt;
    if (attr_res[i]->inst_format)
      rdna::DecodeBufferFormat(attr_res[i]->inst_format, dfmt, nfmt);
    d.vattrs[d.num_vattrs++] = {
        attr_res[i]->location,  b,    static_cast<u32>(off),
        attr_res[i]->num_comps, dfmt, nfmt};
  }

  if (!d.num_vbufs)
    return;
  d.vertex_data = d.vbufs[0].data;
  d.vertex_stride = d.vbufs[0].stride;
  u32 count = UINT32_MAX;
  for (u32 j = 0; j < d.num_vbufs; j++)
    count = base::Min(count, d.vbufs[j].num_records);
  d.vertex_count = count;
}

void ResolveCbufferBindings(const base::Vector<gcn::ShaderCbuf>& cbufs,
                            const ResolvedBuffers& resolved,
                            bool vertex_stage,
                            u64 stage_addr,
                            render::DrawInfo& d,
                            const ResolvedBuffers* gs_resolved = nullptr) {
  u32 planned = 0, replayed = 0, mapped = 0;
  for (const gcn::ShaderCbuf& cb : cbufs) {
    if (cb.binding >= base::ArraySize(d.cbufs))
      continue;
    planned++;
    const auto& resources = cb.from_gs && gs_resolved ? *gs_resolved : resolved;
    const auto it = resources.find(cb.use_pc);
    if (it == resources.end())
      continue;
    replayed++;
    const u64 base =
        (it->second.base & ~u64{3}) + static_cast<u64>(cb.first_dword) * 4;
    const u64 bytes = static_cast<u64>(cb.num_dwords) * 4;
    TraceCbufBinding(vertex_stage, cb.binding, cb.use_pc, base, cb.num_dwords);
    if (!IsGuestAddress(base) || !gpu::IsReadableRangeCached(base, bytes))
      continue;
    mapped++;
    d.cbufs[cb.binding] = {base, static_cast<u32>(bytes)};
    d.num_cbufs = base::Max(d.num_cbufs, cb.binding + 1);
    if (vertex_stage && !cb.first_dword && bytes >= sizeof(d.mvp)) {
      d.cbuf_base = base;
      d.cbuf_size = static_cast<u32>(bytes);
      std::memcpy(d.mvp, reinterpret_cast<const void*>(base), sizeof(d.mvp));
    }
  }
  if (kCbResolve && planned) {
    static base::HashSet<u64> seen;
    static const guest::SessionReset reset_seen(
        [] { guest::ResetResource(seen); });
    const u64 key = stage_addr * 2 + vertex_stage;
    if (seen.size() < 512 && seen.insert(key).second)
      BASE_LOGI("cbresolve", "{} {:#x} planned={} replayed={} mapped={}",
                vertex_stage ? "vs" : "ps", (unsigned long)stage_addr, planned,
                replayed, mapped);
  }
}

void ResolveRawBuffers(const base::Vector<gcn::ShaderBuffer>& buffers,
                       const ResolvedBuffers& resolved,
                       const u32* user_data,
                       u32 user_sgprs,
                       render::DrawInfo& d,
                       bool vertex_stage,
                       const ResolvedBuffers* gs_resolved = nullptr) {
  for (const gcn::ShaderBuffer& sb : buffers) {
    if (sb.binding >= render::DrawInfo::kMaxBuffers)
      continue;
    const auto& resources = sb.from_gs && gs_resolved ? *gs_resolved : resolved;
    const auto it = resources.find(sb.use_pc);
    constexpr u32 kRawBufTag = 0x100;
    if (sb.srsrc_sgpr >= kRawBufTag) {
      constexpr u32 kRawBufWindow = 1u << 20;
      u64 base = 0;
      if (it != resources.end() && it->second.descriptor_valid &&
          it->second.descriptor_dwords >= 2)
        base = it->second.descriptor[0] |
               (static_cast<u64>(it->second.descriptor[1] & 0xFFFF) << 32);
      else if (const u32 s = sb.srsrc_sgpr - kRawBufTag; s + 1 < user_sgprs)
        base =
            user_data[s] | (static_cast<u64>(user_data[s + 1] & 0xFFFF) << 32);
      const bool ok = IsGuestAddress(base) &&
                      gpu::IsReadableRangeCached(base, kRawBufWindow);
      TraceRawBufBinding(vertex_stage, sb.binding, sb.use_pc, sb.srsrc_sgpr,
                         it != resources.end(), base, ok ? kRawBufWindow : 0);
      if (!ok)
        continue;
      d.bufs[sb.binding] = {base, kRawBufWindow};
      d.num_bufs = base::Max(d.num_bufs, sb.binding + 1);
      continue;
    }
    rdna::VBuffer vb{};
    if (it != resources.end() && it->second.descriptor_valid)
      vb = rdna::DecodeVBuffer(it->second.descriptor);
    else if (sb.srsrc_sgpr + 3 < user_sgprs)
      vb = rdna::DecodeVBuffer(&user_data[sb.srsrc_sgpr]);
    const u64 bytes = vb.stride ? static_cast<u64>(vb.stride) * vb.num_records
                                : vb.num_records;
    const bool ok = IsGuestAddress(vb.base) && bytes && bytes <= 0xFFFFFFFFull;
    TraceRawBufBinding(vertex_stage, sb.binding, sb.use_pc, sb.srsrc_sgpr,
                       it != resources.end() && it->second.descriptor_valid,
                       vb.base, ok ? bytes : 0);
    if (!ok)
      continue;
    d.bufs[sb.binding] = {vb.base, static_cast<u32>(bytes)};
    d.num_bufs = base::Max(d.num_bufs, sb.binding + 1);
  }
}

void FillDrawTex(u32 slot, const gcn::TImage& s, render::DrawInfo& d);

void ResolvePsTextures(u64 ps_addr,
                       const u32* ps_user_data,
                       u32 ps_user_sgprs,
                       render::DrawInfo& d) {
  const auto texs = rdna::TrackTextures(reinterpret_cast<const u32*>(ps_addr),
                                        ps_user_data, ps_user_sgprs);
  if (texs.empty())
    return;
  d.tex_base = texs[0].valid ? texs[0].base : 0;
  d.tex_swizzle = PackDstSel(texs[0]);
  d.tex_w = texs[0].width;
  d.tex_h = texs[0].height;
  d.tex_dfmt = texs[0].dfmt;
  d.tex_nfmt = texs[0].nfmt;
  d.tex_tiling = texs[0].tiling_idx;
  d.tex_pitch = texs[0].pitch;
  d.tex_layers = texs[0].layers;
  d.tex_depth = texs[0].depth;
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

  constexpr size_t kMax = render::DrawInfo::kMaxDrawTextures;
  for (size_t i = 0; i < texs.size() && i < kMax; i++)
    FillDrawTex(static_cast<u32>(i), texs[i], d);
  d.num_texs = static_cast<u32>(base::Min<size_t>(texs.size(), kMax));
  TraceDrawTextures(d);
}

void FillDrawTex(u32 slot, const gcn::TImage& s, render::DrawInfo& d) {
  TraceRejectedTexture(slot, s);
  TexFromImage(s, d.texs[slot]);
}

}  // namespace

void TexFromImage(const gcn::TImage& s, render::DrawInfo::DrawTex& dt) {
  dt = {};
  dt.base = s.valid ? s.base : 0;
  dt.w = s.width;
  dt.h = s.height;
  dt.tiling = s.tiling_idx;
  dt.pitch = s.pitch;
  dt.dfmt = s.dfmt;
  dt.nfmt = s.nfmt;
  dt.layers = s.layers;
  dt.depth = s.depth;
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

namespace {

void ReconcileTextureDims(const base::Vector<gcn::ShaderTex>& plan,
                          u32 first_slot,
                          render::DrawInfo& d) {
  for (const gcn::ShaderTex& st : plan) {
    const u32 slot = first_slot + st.binding;
    if (slot >= d.num_texs)
      continue;
    render::DrawInfo::DrawTex& dt = d.texs[slot];
    if (dt.is_3d == st.is_3d)
      continue;
    if (kDimTrace)
      BASE_LOGI("texdim", "binding {} module {} descriptor {} {}x{} layers={}",
                slot, st.is_3d ? "3D" : "2D", dt.is_3d ? "3D" : "2D", dt.w,
                dt.h, dt.layers);
    dt.is_3d = st.is_3d;
    if (st.is_3d) {
      dt.depth = dt.layers;
      dt.arrayed = false;
      dt.base_array = 0;
      dt.view_layers = 1;
    } else {
      dt.depth = 1;
      dt.layers = 1;
      dt.view_layers = 1;
    }
    if (slot == 0) {
      d.tex_is_3d = dt.is_3d;
      d.tex_arrayed = dt.arrayed;
      d.tex_layers = dt.layers;
      d.tex_base_array = dt.base_array;
      d.tex_view_layers = dt.view_layers;
    }
  }
}

void AppendStageTextures(u64 code,
                         const u32* user_data,
                         u32 user_sgprs,
                         u32 ud_base,
                         render::DrawInfo& d,
                         u64 system_user_data_addr = 0) {
  const auto texs =
      rdna::TrackTextures(reinterpret_cast<const u32*>(code), user_data,
                          user_sgprs, ud_base, system_user_data_addr);
  for (size_t i = 0;
       i < texs.size() && d.num_texs < render::DrawInfo::kMaxDrawTextures; i++)
    FillDrawTex(d.num_texs++, texs[i], d);
  TraceDrawTextures(d);
}

void ResolveRecompiledShaders(const Regs& regs,
                              const ShaderBinding& binding,
                              render::DrawInfo& d,
                              size_t& shader_attrs) {
  u32 vs_user_sgprs = UserSgprCount(regs[mmSPI_SHADER_PGM_RSRC2_GS]);
  if (!vs_user_sgprs) {
    for (u32 i = 0; i < 16; i++)
      if (binding.vs_user_data[i])
        vs_user_sgprs = 16;
  }
  const u32 ps_user_sgprs = UserSgprCount(regs[mmSPI_SHADER_PGM_RSRC2_PS]);
  const u64 fetch =
      vs_user_sgprs >= 2
          ? (static_cast<u64>(binding.vs_user_data[1] & 0xFFFF) << 32) |
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

  rdna::NggConfig ngg;
  const u64 gs_user_data_addr =
      regs[mmSPI_SHADER_USER_DATA_ADDR_LO_GS] |
      (static_cast<u64>(regs[mmSPI_SHADER_USER_DATA_ADDR_HI_GS]) << 32);
  const auto* es_code = reinterpret_cast<const u32*>(binding.es_addr);
  const bool point_ngg =
      d.prim_type == 1 && !d.index_count && binding.vs_addr == binding.es_addr;
  const bool split_ngg =
      point_ngg && binding.es_addr != binding.gs_addr &&
      IsGuestAddress(binding.gs_addr) &&
      gpu::IsReadableRangeCached(binding.gs_addr, kMaxShaderBytes) &&
      rdna::HasNggTransfer(es_code);
  const bool unified_ngg =
      point_ngg && !split_ngg && rdna::HasNggPrimitiveExports(es_code);
  if (split_ngg || unified_ngg) {
    const u32 vertices_per_input = regs[mmVGT_GS_MAX_VERT_OUT] & 0x7ff;
    const u32 primitives_per_input = regs[mmGE_NGG_SUBGRP_CNTL] & 0x1ff;
    const u32 per_input =
        base::Max({1u, vertices_per_input, primitives_per_input});
    const u32 lanes = per_input <= 64 ? 64u : 128u;
    const u32 inputs = base::Min((regs[mmVGT_GS_ONCHIP_CNTL] >> 11) & 0x7ff,
                                 lanes / per_input);
    const u32 vertices = inputs * vertices_per_input;
    const u32 primitives = inputs * primitives_per_input;
    const u32 lds = ((regs[mmSPI_SHADER_PGM_RSRC2_GS] >> 19) & 0xff) * 128;
    if (inputs && vertices && primitives && lds && vertices <= 256 &&
        primitives <= 256) {
      const u32 threads =
          (base::Max({inputs, vertices, primitives}) + 63) & ~63u;
      ngg = {
          split_ngg ? reinterpret_cast<const u32*>(binding.gs_addr) : es_code,
          threads,
          inputs,
          vertices,
          primitives,
          lds,
          gs_user_data_addr,
          split_ngg};
    }
  }
  const gcn::Recompiled& rc =
      GetGraphicsShader({.vs_addr = binding.vs_addr,
                         .ps_addr = binding.ps_addr,
                         .fetch_addr = fetch,
                         .vs_user_sgprs = vs_user_sgprs,
                         .ps_user_sgprs = ps_user_sgprs,
                         .ps_input_ena = regs[mmSPI_PS_INPUT_ENA],
                         .ps_in_cntl = regs.At(mmSPI_PS_INPUT_CNTL_0),
                         .ps_num_interp = regs[mmSPI_PS_IN_CONTROL] & 0x3F,
                         .gl_clip = !((regs[mmPA_CL_CLIP_CNTL] >> 19) & 1),
                         .vs_user_data = binding.vs_user_data,
                         .ps_user_data = binding.ps_user_data,
                         .ngg = ngg});
  if (!rc.ok)
    return;

  const ResolvedBuffers vs_resources =
      rc.direct_vs
          ? ResolvedBuffers{}
          : rdna::ResolveBuffers(reinterpret_cast<const u32*>(binding.vs_addr),
                                 binding.vs_user_data, vs_user_sgprs, kUdBase,
                                 4096, ngg.gs_code ? gs_user_data_addr : 0);
  TraceAttrPlan(rc.attrs.size(), vs_user_sgprs, vs_resources.size(),
                binding.vs_user_data);
  BindVertexAttributes(rc, vs_resources, binding.vs_user_data, vs_user_sgprs,
                       d);
  shader_attrs = rc.attrs.size();

  bool good = d.num_vattrs > 0 || rc.attrs.empty();
  if (!good && !d.num_vbufs) {
    d.num_vattrs = 0;
    good = true;
  }
  const ResolvedBuffers ps_resources =
      binding.ps_addr && !rc.direct
          ? rdna::ResolveBuffers(reinterpret_cast<const u32*>(binding.ps_addr),
                                 binding.ps_user_data, ps_user_sgprs)
          : ResolvedBuffers{};
  if (!good) {
    d.num_vattrs = 0;
    return;
  }
  const ResolvedBuffers gs_resources =
      rc.mesh_spirv.empty() || rc.direct_vs
          ? ResolvedBuffers{}
          : rdna::ResolveBuffers(ngg.gs_code, binding.vs_user_data,
                                 vs_user_sgprs, kUdBase, 4096,
                                 gs_user_data_addr);
  if (!rc.direct_vs) {
    ResolveCbufferBindings(rc.vs_cbufs, vs_resources, true, binding.vs_addr, d,
                           &gs_resources);
    ResolveRawBuffers(rc.vs_bufs, vs_resources, binding.vs_user_data,
                      vs_user_sgprs, d, true, &gs_resources);
  }
  if (binding.ps_addr) {
    if (!rc.direct) {
      ResolveCbufferBindings(rc.ps_cbufs, ps_resources, false, binding.ps_addr,
                             d);
      ResolveRawBuffers(rc.ps_bufs, ps_resources, binding.ps_user_data,
                        ps_user_sgprs, d, false);
    }
    if (!rc.ps_texs.empty())
      ResolvePsTextures(binding.ps_addr, binding.ps_user_data, ps_user_sgprs,
                        d);
    ReconcileTextureDims(rc.ps_texs, 0, d);
  }
  const u32 vs_tex_slot = static_cast<u32>(rc.ps_texs.size());
  if (!rc.vs_texs.empty()) {
    AppendStageTextures(rc.mesh_spirv.empty()
                            ? binding.vs_addr
                            : reinterpret_cast<u64>(ngg.gs_code),
                        binding.vs_user_data, vs_user_sgprs, 8, d,
                        rc.mesh_spirv.empty() ? 0 : gs_user_data_addr);
    ReconcileTextureDims(rc.vs_texs, vs_tex_slot, d);
  }
  d.vs_addr = binding.vs_addr;
  d.ps_addr = binding.ps_addr;
  d.es_addr = binding.es_addr;
  d.gs_addr = binding.gs_addr;
  d.gs_user_data_addr = gs_user_data_addr;
  d.recomp = &rc;
}

}  // namespace

bool BuildDrawInfo(const Regs& regs,
                   const DrawPacket& packet,
                   render::DrawInfo& d) {
  ScopeNs build_timer(&g_stats.ns_build_draw);
  g_stats.build_draw_n++;
  const ShaderBinding binding = ResolveShaderBinding(regs);
  TraceShaderListing(binding.vs_addr);
  TraceShaderListing(binding.ps_addr);
  if (kSkipVs && binding.vs_addr == kSkipVs)
    return false;

  std::memcpy(d.vs_user_data, binding.vs_user_data, sizeof(d.vs_user_data));
  std::memcpy(d.ps_user_data, binding.ps_user_data, sizeof(d.ps_user_data));
  d.prim_type = regs[mmVGT_PRIMITIVE_TYPE];
  d.instance_count = packet.num_instances;
  d.indirect_args = packet.indirect_args;
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
    NoteDrawDropped(d, binding.vs_addr, binding.ps_addr, shader_attrs);
    return false;
  }
  TraceVertexDump(d, binding.vs_user_data, binding.vs_addr, binding.ps_addr);
  return true;
}

}  // namespace gpu::ps5
