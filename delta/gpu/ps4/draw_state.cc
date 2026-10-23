/*
 * PS4Delta : PS4 emulation and research project
 */

#include "gpu/ps4/draw_state.h"
#include "base/arch.h"
#include "gpu/guest_memory.h"
#include "gpu/ps4/render_queue.h"
#include "guest/session.h"

#include <cstdlib>
#include <cstring>

#include "host_memory/host_memory.h"
#include "options/options.h"

#include "base/algorithm.h"
#include "base/containers/array.h"
#include "base/containers/hash_map.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/shared_pointer.h"
#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gpu_perf.h"
#include "gpu/ps4/cmd_trace.h"
#include "gpu/ps4/guest_address.h"
#include "gpu/ps4/pm4.h"
#include "gpu/ps4/shader_cache.h"
#include "profile/profile.h"

namespace {

DELTA_OPTION(u64, kCntlApply, "DELTA_GPU_PSCNTL_APPLY", 1);
DELTA_OPTION(bool, kForceDepth, "DELTA_GPU_FORCEDEPTH", false);
DELTA_OPTION(bool, kIntegerRt, "DELTA_GPU_INT_RT", true);
DELTA_OPTION(bool, kNoDepth, "DELTA_GPU_NODEPTH", false);
DELTA_OPTION(bool, kNoStencil, "DELTA_GPU_NOSTENCIL", false);
DELTA_OPTION(bool, kPreflushResources, "DELTA_GPU_PREFLUSHRES", false);
DELTA_OPTION(bool, kSkipStale, "DELTA_GPU_SKIPSTALE", false);
DELTA_OPTION(const char*, kSkipShList, "DELTA_GPU_SKIPSH", nullptr);
// Hex VS addresses: draw only these (isolates what one shader renders).
DELTA_OPTION(const char*, kOnlyVsList, "DELTA_GPU_ONLYVS", nullptr);

}  // namespace

namespace gpu::ps4 {
namespace {

constexpr u32 kMaxTrackedTextures = render::DrawInfo::kMaxDrawTextures;
constexpr u32 kMaxElementCount = 0x100000;

struct TextureMasks {
  u32 tex_3d = 0;
  u32 tex_1d = 0;
  u32 tex_uint = 0;
  u32 tex_cube = 0;
};

u32 CbFormatBytes(u32 format) {
  switch (format) {
    case 1:
      return 1;
    case 2:
    case 3:
    case 16:
    case 17:
    case 18:
    case 19:
      return 2;
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 20:
    case 21:
      return 4;
    case 11:
    case 12:
      return 8;
    case 14:
      return 16;
    default:
      return 0;
  }
}

u32 FbWidth(const Regs& regs) {
  const u32 w = regs[mmPA_SC_SCREEN_SCISSOR_BR] & 0xFFFF;
  return w ? w : 1920;
}
u32 FbHeight(const Regs& regs) {
  const u32 h = regs[mmPA_SC_SCREEN_SCISSOR_BR] >> 16;
  return h ? h : 1080;
}

u64 UserDataPointer(const u32* user_data, u32 index) {
  return (static_cast<u64>(user_data[index + 1] & 0xFFFF) << 32) |
         user_data[index];
}

}  // namespace

void FillDrawTex(render::DrawInfo::DrawTex& dt, const gcn::TImage& t) {
  dt.base = t.valid ? t.base : 0;
  dt.w = t.width;
  dt.h = t.height;
  dt.dfmt = t.dfmt;
  dt.nfmt = t.nfmt;
  dt.tiling = t.tiling_idx;
  dt.pitch = t.pitch;
  dt.depth = t.depth;
  dt.layers = t.layers;
  dt.base_array = t.base_array;
  dt.view_layers = t.view_layers;
  dt.mip_levels = t.mip_levels;
  dt.base_mip = t.base_mip;
  dt.view_mips = t.view_mips;
  dt.min_lod = t.min_lod;
  dt.pow2_pad = t.pow2_pad;
  std::memcpy(dt.sampler, t.sampler, sizeof(dt.sampler));
  dt.sampler_valid = t.sampler_valid;
  dt.arrayed = t.arrayed;
  dt.is_3d = t.is_3d;
  dt.is_1d = t.is_1d;
  dt.force_lod_zero = t.force_lod_zero;
  dt.depth_compare = t.depth_compare;
  dt.storage = t.storage;
  dt.null_descriptor = t.null_descriptor;
  dt.swizzle = gcn::PackDstSel(t.dst_sel);
  dt.src = t.src;
}

namespace {

void BindTexture(render::DrawInfo& d,
                 const gcn::TImage& t,
                 u32 slot,
                 TextureMasks& masks) {
  FillDrawTex(d.texs[slot], t);
  if (t.is_3d)
    masks.tex_3d |= 1u << slot;
  if (t.is_1d)
    masks.tex_1d |= 1u << slot;
  if (t.type == 11)
    masks.tex_cube |= 1u << slot;
  if (kIntegerRt && !t.storage && (t.nfmt == 4 || t.nfmt == 5))
    masks.tex_uint |= 1u << slot;
}

void ReadIndirectArgs(render::Renderer& renderer,
                      const DrawPacket& packet,
                      u64 args,
                      render::DrawInfo& d) {
  const u32 op = packet.op;
  const u32 need = op == IT_DRAW_INDIRECT ? 16u : 20u;
  FlushForWalkRead(renderer, args, need, "indirect");
  const bool mapped = host_memory::IsMemoryRangeMapped(
      reinterpret_cast<const void*>(args), need);
  u32 a[5] = {};
  if (mapped)
    std::memcpy(a, reinterpret_cast<const void*>(args), need);
  TraceIndirectArgs(op, args, packet.indirect_base, packet.body[0], mapped, a);
  if (!mapped || !a[0] || a[0] > kMaxElementCount)
    return;
  if (op == IT_DRAW_INDIRECT) {
    d.vertex_count = a[0];
  } else if (packet.index_base) {
    const u32 index_bytes = packet.index_type == 1   ? 4
                            : packet.index_type == 2 ? 1
                                                     : 2;
    const u64 base = packet.index_base + static_cast<u64>(a[2]) * index_bytes;
    if (host_memory::IsMemoryRangeMapped(
            reinterpret_cast<const void*>(base),
            static_cast<u64>(a[0]) * index_bytes)) {
      d.index_data = reinterpret_cast<const void*>(base);
      d.index_count = a[0];
      d.index_type = packet.index_type;
    }
  }
  if (a[1])
    d.instance_count = a[1];
}

void ResolveIndexBuffer(render::Renderer& renderer,
                        const DrawPacket& packet,
                        render::DrawInfo& d) {
  const u32 op = packet.op, count = packet.count;
  const u32* body = packet.body;

  if (op == IT_DRAW_INDEX_2 && count >= 4) {
    const u64 base = (static_cast<u64>(body[2] & 0xFF) << 32) | body[1];
    const u32 index_count = body[3];
    const bool accepted =
        IsGuestAddress(base) && index_count && index_count <= kMaxElementCount;
    TraceIndexBuffer(base, index_count, packet.index_type, accepted);
    if (accepted) {
      d.index_data = reinterpret_cast<const void*>(base);
      d.index_count = index_count;
      d.index_type = packet.index_type;
    }
  }

  if ((op == IT_DRAW_INDIRECT || op == IT_DRAW_INDEX_INDIRECT) && count >= 1 &&
      packet.indirect_base) {
    const u64 args = packet.indirect_base + body[0];
    const bool indexed = op == IT_DRAW_INDEX_INDIRECT;
    const u32 index_bytes = packet.index_type == 1   ? 4
                            : packet.index_type == 2 ? 1
                                                     : 2;
    const bool whole_buffer =
        packet.index_base && packet.index_max &&
        packet.index_max <= kMaxElementCount &&
        host_memory::IsMemoryRangeMapped(
            reinterpret_cast<const void*>(packet.index_base),
            static_cast<u64>(packet.index_max) * index_bytes);
    if (!(args & 3) && gpu::IsReadableRangeCached(args, indexed ? 20 : 16) &&
        render::CanDrawIndirect(d.prim_type, indexed) &&
        (!indexed || whole_buffer)) {
      d.indirect_args = args;
      if (indexed) {
        d.index_data = reinterpret_cast<const void*>(packet.index_base);
        d.index_count = packet.index_max;
        d.index_type = packet.index_type;
      } else {
        d.vertex_count = 3;
      }
    } else {
      ReadIndirectArgs(renderer, packet, args, d);
    }
  }

  if (op == IT_DRAW_INDEX_OFFSET_2 && count >= 3) {
    const u32 index_offset = body[1], index_count = body[2];
    const u32 index_bytes = packet.index_type == 1   ? 4
                            : packet.index_type == 2 ? 1
                                                     : 2;
    const u64 base =
        packet.index_base + static_cast<u64>(index_offset) * index_bytes;
    const bool accepted =
        packet.index_base && index_count && index_count <= kMaxElementCount &&
        gpu::IsReadableRangeCached(base,
                                   static_cast<u64>(index_count) * index_bytes);
    TraceIndexOffsetArgs(body[0], index_offset, index_count, packet.index_base,
                         base, packet.index_type, accepted);
    if (accepted) {
      d.index_data = reinterpret_cast<const void*>(base);
      d.index_count = index_count;
      d.index_type = packet.index_type;
    }
  }
}

u32 ResolveRenderTargets(const Regs& regs,
                         render::DrawInfo& d,
                         u64 vs_addr,
                         u64 ps_addr) {
  d.rt_w = FbWidth(regs);
  d.rt_h = FbHeight(regs);
  d.scissor_tl = regs[mmPA_SC_VPORT_SCISSOR_0_TL];
  d.scissor_br = regs[mmPA_SC_VPORT_SCISSOR_0_BR];

  const u32 pitch = ((regs[mmCB_COLOR0_PITCH] & 0x7FFu) + 1u) * 8u;
  const u64 slice =
      static_cast<u64>((regs[mmCB_COLOR0_SLICE] & 0x3FFFFFu) + 1u) * 64u;
  d.rt_surf_w = pitch;
  d.rt_surf_h = pitch ? static_cast<u32>(slice / pitch) : 0u;

  u32 mrt_uint_mask = 0;
  const u32 target_mask = regs[mmCB_TARGET_MASK];
  for (int rt = 0; rt < 8; rt++) {
    const u64 base = regs.CbColorBase(rt);
    const u32 info = regs[mmCB_COLOR0_INFO + rt * kCbColorStride];
    if (!((target_mask >> (rt * 4)) & 0xF) || !((info >> 2) & 0x1F) ||
        !IsGuestAddress(base))
      continue;
    d.mrt_base[rt] = base;
    d.mrt_info[rt] = info;
    d.mrt_count = rt + 1;
    d.mrt_bound_mask |= 1u << rt;
    const u32 rt_pitch =
        ((regs[mmCB_COLOR0_PITCH + rt * kCbColorStride] & 0x7FFu) + 1u) * 8u;
    const u64 rt_slice =
        static_cast<u64>(
            (regs[mmCB_COLOR0_SLICE + rt * kCbColorStride] & 0x3FFFFFu) + 1u) *
        64u;
    d.mrt_surf_w[rt] = rt_pitch;
    d.mrt_surf_h[rt] = rt_pitch ? static_cast<u32>(rt_slice / rt_pitch) : 0u;
    TraceAddrWatch("DRAW_COLOR_TARGET", base,
                   rt_slice * CbFormatBytes((info >> 2) & 0x1Fu) *
                       (((regs[mmCB_COLOR0_VIEW + rt * kCbColorStride] >> 13) &
                         0x7FFu) + 1u),
                   regs.ShaderAddr(mmSPI_SHADER_PGM_LO_PS) & 0xFFFFFFFFu, 8);
    if (rt == 0) {
      d.rt_tile_mode = regs[mmCB_COLOR0_ATTRIB] & 0x1Fu;
      const u32 view = regs[mmCB_COLOR0_VIEW];
      const u32 first = view & 0x7FFu, last = (view >> 13) & 0x7FFu;
      const u32 bytes = CbFormatBytes((info >> 2) & 0x1Fu);
      if (first && first == last && bytes) {
        d.rt_array_base = base;
        d.mrt_base[rt] = base + first * rt_slice * bytes;
      }
    }
    if ((info >> 13) & 1u) {
      d.mrt_dcc_base[rt] =
          static_cast<u64>(regs[mmCB_COLOR0_CMASK + rt * kCbColorStride]) << 8;
      d.mrt_meta_cmask = true;
    }
    d.mrt_clear_word[rt][0] =
        regs[mmCB_COLOR0_CLEAR_WORD0 + rt * kCbColorStride];
    d.mrt_clear_word[rt][1] =
        regs[mmCB_COLOR0_CLEAR_WORD1 + rt * kCbColorStride];
    const u32 nfmt = (info >> 8) & 0x7;
    if (kIntegerRt && (nfmt == 4 || nfmt == 5))
      mrt_uint_mask |= 1u << rt;
  }
  d.rt_base = d.mrt_count ? d.mrt_base[0] : 0;
  TraceMrtSlotZeroGap(d, target_mask, vs_addr, ps_addr);
  TraceNoMrtBound(regs, d, target_mask, vs_addr, ps_addr);
  return mrt_uint_mask;
}

void ResolveColorState(const Regs& regs, u64 ps_addr, render::DrawInfo& d) {
  d.blend_control = regs[mmCB_BLEND0_CONTROL];
  d.blend_enable = (d.blend_control >> 30) & 1u;
  d.mrt_blend[0] = d.blend_control;
  if (d.blend_enable)
    d.mrt_blend_mask |= 1u;
  for (u32 rt = 1; rt < 8; rt++) {
    const u32 blend = regs[mmCB_BLEND0_CONTROL + rt * kCbBlendStride];
    d.mrt_blend[rt] = blend;
    if ((blend >> 30) & 1u)
      d.mrt_blend_mask |= 1u << rt;
  }
  for (u32 c = 0; c < 4; c++) {
    const u32 raw = regs[mmCB_BLEND_RED + c];
    std::memcpy(&d.blend_constants[c], &raw, sizeof(float));
  }
  d.target_mask = regs[mmCB_TARGET_MASK];
  d.shader_mask = regs[mmCB_SHADER_MASK];
  d.col_format = regs[mmSPI_SHADER_COL_FORMAT];

  d.is_clear_rect =
      d.prim_type == 17 && !ps_addr && !d.num_vattrs && d.mrt_count != 0;
  if (d.is_clear_rect) {
    d.clear_tl = regs[mmPA_SC_GENERIC_SCISSOR_TL];
    d.clear_br = regs[mmPA_SC_GENERIC_SCISSOR_BR];
    d.clear_window_tl = regs[mmPA_SC_WINDOW_SCISSOR_TL];
    d.clear_window_br = regs[mmPA_SC_WINDOW_SCISSOR_BR];
    d.clear_screen_tl = regs[mmPA_SC_SCREEN_SCISSOR_TL];
    d.clear_screen_br = regs[mmPA_SC_SCREEN_SCISSOR_BR];
  }
  d.color_control = regs[mmCB_COLOR_CONTROL];
}

void ResolveDepthState(const Regs& regs, render::DrawInfo& d) {
  const u32 depth_control = regs[mmDB_DEPTH_CONTROL];
  d.depth_control = depth_control;
  const u32 z_info = kNoDepth ? 0 : regs[mmDB_Z_INFO];
  const u32 stencil_info = kNoDepth ? 0 : regs[mmDB_STENCIL_INFO];
  const u64 z_base = static_cast<u64>(regs[mmDB_Z_WRITE_BASE]) << 8;
  const u64 stencil_base = static_cast<u64>(regs[mmDB_STENCIL_WRITE_BASE]) << 8;
  TraceDepthState(regs, z_info);

  const u32 render_control = regs[mmDB_RENDER_CONTROL];
  d.render_control = render_control;
  const u32 size = regs[mmDB_DEPTH_SIZE];
  d.depth_w = ((size & 0x7FFu) + 1u) * 8u;
  d.depth_h = (((size >> 11) & 0x7FFu) + 1u) * 8u;
  d.depth_clear_draw = (render_control & 1u) != 0;
  d.stencil_clear_draw = ((render_control >> 1) & 1u) != 0;
  d.depth_valid = (z_info & 0x3) != 0;
  const bool tests_or_writes = (depth_control >> 1) & 1u ||
                               (depth_control >> 2) & 1u ||
                               (depth_control & 1u);
  if (!d.depth_valid || !IsGuestAddress(z_base) || !tests_or_writes) {
    d.depth_valid = false;
  } else {
    d.depth_base = z_base;
    d.depth_test_enable = (depth_control >> 1) & 1u;
    d.depth_write_enable = (depth_control >> 2) & 1u;
    d.depth_func = (depth_control >> 4) & 0x7;
    std::memcpy(&d.depth_clear, regs.At(mmDB_DEPTH_CLEAR), 4);
    if (!(d.depth_clear >= 0.0f && d.depth_clear <= 1.0f))
      d.depth_clear = 1.0f;
    if ((z_info >> 29) & 1u) {
      const u64 htile = static_cast<u64>(regs[mmDB_HTILE_DATA_BASE]) << 8;
      if (IsGuestAddress(htile))
        d.depth_htile_base = htile;
    }
    d.stencil_enable = !kNoStencil && (depth_control & 1u) &&
                       (stencil_info & 1u) && IsGuestAddress(stencil_base);
    if (d.stencil_enable) {
      d.stencil_base = stencil_base;
      d.stencil_backface_enable = (depth_control >> 7) & 1u;
      d.stencil_clear = regs[mmDB_STENCIL_CLEAR] & 0xFF;
      d.stencil_control = regs[mmDB_STENCIL_CONTROL];
      d.stencil_refmask = regs[mmDB_STENCILREFMASK];
      d.stencil_refmask_bf = regs[mmDB_STENCILREFMASK_BF];
    }
  }
  if (kForceDepth && !d.depth_base && d.rt_base) {
    d.depth_base = d.rt_base;
    d.depth_test_enable = true;
    d.depth_write_enable = true;
    d.depth_func = 7;
    d.depth_clear = 1.0f;
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
  std::memcpy(&d.viewport_z_scale, regs.At(mmPA_CL_VPORT_ZSCALE), 4);
  std::memcpy(&d.viewport_z_offset, regs.At(mmPA_CL_VPORT_ZOFFSET), 4);
}

u64 FetchShaderAddress(const u32* vud, u64 vs_addr) {
  if (!IsGuestAddress(vs_addr) ||
      !gcn::CallsFetchShader(*gcn::CachedProgram(vs_addr, 4096)))
    return 0;
  const u64 fetch = UserDataPointer(vud, 0);
  return IsGuestAddress(fetch) ? fetch : 0;
}

void ResolveDefaultSources(const u32* vud,
                           u64 fetch_addr,
                           render::DrawInfo& d) {
  const u64 cbuf = UserDataPointer(vud, 4);
  if (IsGuestAddress(cbuf)) {
    std::memcpy(d.mvp, reinterpret_cast<const void*>(cbuf), 64);
    d.cbuf_base = cbuf;
  }
  if (!fetch_addr)
    return;
  thread_local base::Vector<gcn::VBuffer> vbs;
  gcn::TrackVertexBuffers(vbs, *gcn::CachedProgram(fetch_addr, 64), vud);
  if (vbs.empty())
    return;
  d.vertex_data = reinterpret_cast<const void*>(vbs[0].base);
  d.vertex_count = vbs[0].num_records;
  d.vertex_stride = vbs[0].stride;
}

base::SharedPointer<const gcn::Program> ResolvePsTextures(
    render::Renderer& renderer,
    const Regs& regs,
    u64 ps_addr,
    u32 frame,
    render::DrawInfo& d,
    TextureMasks& masks) {
  if (!IsGuestAddress(ps_addr))
    return nullptr;
  if (kPreflushResources) {
    OwnRenderer();
    render::FlushCsWrites(renderer);
  }
  auto ps_prog = gcn::CachedProgram(ps_addr, 4096);
  const bool trace = ShouldTraceTextureTracking(frame, ps_addr);
  thread_local base::Vector<gcn::TImage> texs;
  gcn::TrackTextures(texs, ps_prog, regs.At(mmSPI_SHADER_USER_DATA_PS_0), trace,
                     ps_addr);
  if (texs.empty())
    return ps_prog;

  const gcn::TImage& first = texs[0];
  d.tex_base = first.valid ? first.base : 0;
  d.tex_w = first.width;
  d.tex_h = first.height;
  d.tex_dfmt = first.dfmt;
  d.tex_nfmt = first.nfmt;
  d.tex_tiling = first.tiling_idx;
  d.tex_pitch = first.pitch;
  d.tex_depth = first.depth;
  d.tex_layers = first.layers;
  d.tex_base_array = first.base_array;
  d.tex_view_layers = first.view_layers;
  d.tex_mip_levels = first.mip_levels;
  d.tex_base_mip = first.base_mip;
  d.tex_view_mips = first.view_mips;
  d.tex_min_lod = first.min_lod;
  std::memcpy(d.tex_sampler, first.sampler, sizeof(d.tex_sampler));
  d.tex_pow2_pad = first.pow2_pad;
  d.tex_sampler_valid = first.sampler_valid;
  d.tex_arrayed = first.arrayed;
  d.tex_is_3d = first.is_3d;
  d.tex_is_1d = first.is_1d;
  d.tex_force_lod_zero = first.force_lod_zero;
  d.tex_depth_compare = first.depth_compare;
  d.tex_null_descriptor = first.null_descriptor;
  d.tex_swizzle = gcn::PackDstSel(first.dst_sel);

  d.num_texs =
      static_cast<u32>(base::Min<size_t>(texs.size(), kMaxTrackedTextures));
  for (u32 i = 0; i < d.num_texs; i++)
    BindTexture(d, texs[i], i, masks);
  TraceTextureFormat(first, d);
  return ps_prog;
}

void ResolveVsTextures(const u32* vud,
                       u64 vs_addr,
                       render::DrawInfo& d,
                       TextureMasks& masks) {
  if (!IsGuestAddress(vs_addr))
    return;
  thread_local base::Vector<gcn::TImage> texs;
  gcn::TrackTextures(texs, gcn::CachedProgram(vs_addr, 4096), vud, false,
                     vs_addr);
  for (const auto& t : texs) {
    if (d.num_texs >= kMaxTrackedTextures)
      break;
    BindTexture(d, t, d.num_texs++, masks);
  }
}

base::Vector<u64> ParseShaderList(const char* spec) {
  base::Vector<u64> list;
  for (const char* p = spec ? spec : ""; *p;) {
    char* end;
    const u64 addr = std::strtoull(p, &end, 16);
    if (end == p)
      break;
    list.push_back(addr);
    p = *end == ',' ? end + 1 : end;
  }
  return list;
}

bool ShaderSkipped(u64 vs_addr, u64 ps_addr) {
  static const base::Vector<u64> kSkipped = ParseShaderList(kSkipShList);
  static const base::Vector<u64> kOnly = ParseShaderList(kOnlyVsList);
  if (!kOnly.empty() &&
      base::Find(kOnly.begin(), kOnly.end(), vs_addr) == kOnly.end())
    return true;
  return !kSkipped.empty() && (base::Find(kSkipped.begin(), kSkipped.end(),
                                          vs_addr) != kSkipped.end() ||
                               base::Find(kSkipped.begin(), kSkipped.end(),
                                          ps_addr) != kSkipped.end());
}

RecompStatus BindVertexAttributes(const gcn::Recompiled& rc,
                                  const gcn::VBuffer* attr_vbs,
                                  u32 attr_count,
                                  render::DrawInfo& d) {
  u32 attr_binding[render::DrawInfo::kMaxVertexAttrs] = {};
  for (u32 i = 0; i < attr_count; i++) {
    const gcn::VBuffer& vb = attr_vbs[i];
    const bool per_instance = rc.attrs[i].per_instance;
    int sel = -1;
    for (u32 j = 0; j < d.num_vbufs; j++) {
      if (d.vbufs[j].stride != vb.stride ||
          d.vbufs[j].per_instance != per_instance)
        continue;
      const u64 bound = reinterpret_cast<u64>(d.vbufs[j].data);
      const u64 lo = base::Min(bound, vb.base);
      const u64 hi = base::Max(bound, vb.base);
      if (hi - lo < vb.stride) {
        sel = static_cast<int>(j);
        break;
      }
    }
    if (sel < 0) {
      if (d.num_vbufs >= 8)
        return RecompStatus::kAttrBindings;
      sel = static_cast<int>(d.num_vbufs);
      d.vbufs[d.num_vbufs++] = {reinterpret_cast<const void*>(vb.base),
                                vb.stride, vb.num_records, per_instance};
    } else {
      auto& bind = d.vbufs[sel];
      if (vb.base < reinterpret_cast<u64>(bind.data))
        bind.data = reinterpret_cast<const void*>(vb.base);
      bind.num_records = base::Min(bind.num_records, vb.num_records);
    }
    attr_binding[i] = static_cast<u32>(sel);
  }
  for (u32 i = 0; i < attr_count; i++) {
    const gcn::ShaderAttr& a = rc.attrs[i];
    const gcn::VBuffer& vb = attr_vbs[i];
    const u32 binding = attr_binding[i];
    const u64 offset = vb.base - reinterpret_cast<u64>(d.vbufs[binding].data);
    if (d.vbufs[binding].stride && offset >= d.vbufs[binding].stride)
      return RecompStatus::kAttrOffset;
    const u32 dfmt = a.inst_dfmt ? a.inst_dfmt : vb.dfmt;
    const u32 nfmt = a.inst_dfmt ? a.inst_nfmt : vb.nfmt;
    d.vattrs[d.num_vattrs++] = {a.location,  binding, static_cast<u32>(offset),
                                a.num_comps, dfmt,    nfmt};
  }

  u32 primary = 0;
  while (primary < d.num_vbufs &&
         (!d.vbufs[primary].stride || d.vbufs[primary].per_instance))
    primary++;
  d.vertex_data = d.vbufs[primary < d.num_vbufs ? primary : 0].data;
  d.vertex_stride = primary < d.num_vbufs ? d.vbufs[primary].stride : 0;
  u32 records = UINT32_MAX;
  for (u32 j = 0; j < d.num_vbufs; j++)
    if (d.vbufs[j].stride && !d.vbufs[j].per_instance)
      records = base::Min(records, d.vbufs[j].num_records);
  d.vertex_count = records == UINT32_MAX ? 0 : records;
  return RecompStatus::kOk;
}

u64 IntAttrMask(const gcn::Recompiled& rc,
                const gcn::VBuffer* attr_vbs,
                u32 attr_count) {
  u64 mask = 0;
  for (u32 i = 0; i < attr_count; i++) {
    const gcn::ShaderAttr& a = rc.attrs[i];
    const u32 dfmt = a.inst_dfmt ? a.inst_dfmt : attr_vbs[i].dfmt;
    const u32 nfmt = a.inst_dfmt ? a.inst_nfmt : attr_vbs[i].nfmt;
    const bool narrow = dfmt == 1 || dfmt == 2 || dfmt == 3 || dfmt == 5 ||
                        dfmt == 10 || dfmt == 12;
    if (narrow && a.location < 16 && (nfmt == 4 || nfmt == 5))
      mask |= 1u << (a.location + (nfmt == 5 ? 16 : 0));
    if (gcn::PackedAttrFormat(dfmt, nfmt) && a.location < 16)
      mask |= gcn::PackedAttrBits(a.location, nfmt);
  }
  return mask;
}

void ResolveCbufferBindings(
    const base::Vector<gcn::ShaderCbuf>& cbufs,
    const u32* user_data,
    const base::SharedPointer<const gcn::Program>& program,
    u64 code_base,
    bool vertex_stage,
    render::DrawInfo& d,
    bool& resolved_vs_cbuf) {
  if (cbufs.empty())
    return;
  thread_local gcn::CbufList resolved;
  gcn::ResolveCbuffers(resolved, program, user_data, code_base);
  for (const auto& cb : cbufs) {
    if (cb.binding >= base::ArraySize(d.cbufs))
      continue;
    gcn::VBuffer vb{};
    if (const gcn::VBuffer* found = gcn::FindCbuf(
            resolved, gcn::CbufKey(cb.ud_sgpr, cb.pointer, cb.version)))
      vb = *found;
    else if (cb.pointer && cb.ud_sgpr + 1 < 16)
      vb.base = UserDataPointer(user_data, cb.ud_sgpr);
    else if (!cb.pointer && cb.ud_sgpr + 3 < 16)
      vb = gcn::DecodeVBuffer(&user_data[cb.ud_sgpr]);
    u64 bytes = vb.stride ? (u64)vb.stride * vb.num_records : vb.num_records;
    if (cb.pointer)
      bytes = (u64)cb.num_dwords * 4;
    if (!IsGuestRange(vb.base, bytes) || bytes > 0xFFFFFFFFull)
      continue;
    d.cbufs[cb.binding] = {vb.base, static_cast<u32>(bytes)};
    d.num_cbufs = base::Max(d.num_cbufs, cb.binding + 1);
    if (cb.pointer)
      continue;
    if (vertex_stage && !resolved_vs_cbuf) {
      resolved_vs_cbuf = true;
      d.cbuf_base = vb.base;
      d.cbuf_size = static_cast<u32>(bytes);
      if (bytes >= sizeof(d.mvp) &&
          gpu::IsReadableRangeCached(vb.base, sizeof(d.mvp)))
        std::memcpy(d.mvp, reinterpret_cast<const void*>(vb.base),
                    sizeof(d.mvp));
    }
  }
}

void ResolveRawBuffers(const base::Vector<gcn::ShaderBuffer>& buffers,
                       const u32* user_data,
                       const base::SharedPointer<const gcn::Program>& program,
                       u64 code_base,
                       const char* stage,
                       u64 vs_addr,
                       render::DrawInfo& d) {
  if (buffers.empty())
    return;
  thread_local base::Vector<gcn::VBuffer> resolved;
  gcn::ResolveShaderBuffers(resolved, program, buffers, user_data, code_base);
  for (size_t i = 0; i < buffers.size(); i++) {
    const gcn::ShaderBuffer& sb = buffers[i];
    if (sb.binding >= render::DrawInfo::kMaxBuffers)
      continue;
    gcn::VBuffer vb = resolved[i];
    if (!vb.base && sb.srsrc_sgpr + 3 < 16)
      vb = gcn::DecodeVBuffer(&user_data[sb.srsrc_sgpr]);
    const u64 bytes =
        vb.stride ? (u64)vb.stride * vb.num_records : vb.num_records;
    const bool ok = IsGuestRange(vb.base, bytes) && bytes <= 0xFFFFFFFFull;
    TraceRawBuffer(stage, vs_addr, sb, vb, bytes, ok, d);
    if (!ok)
      continue;
    d.bufs[sb.binding] = {vb.base, static_cast<u32>(bytes), vb.stride};
    d.num_bufs = base::Max(d.num_bufs, sb.binding + 1);
  }
}

GraphicsShaderState ShaderStateOf(const Regs& regs,
                                  u64 vs_addr,
                                  const gcn::GsPipeline* gs,
                                  u64 ps_addr,
                                  u64 fetch_addr,
                                  const u32* ps_in_cntl,
                                  const TextureMasks& masks,
                                  u32 mrt_uint_mask,
                                  u32 mrt_bound_mask) {
  GraphicsShaderState state;
  state.vs_addr = gs ? regs.ShaderAddr(mmSPI_SHADER_PGM_LO_VS) : vs_addr;
  state.gs = gs;
  state.ps_addr = ps_addr;
  state.fetch_addr = fetch_addr;
  state.ps_input_ena = regs[mmSPI_PS_INPUT_ENA];
  state.ps_in_cntl = ps_in_cntl;
  state.honour_ps_in_cntl =
      kCntlApply != 0 && (kCntlApply == 1 || ps_addr == (u64)kCntlApply);
  state.ps_num_interp = regs[mmSPI_PS_IN_CONTROL] & 0x3F;
  state.tex_3d_mask = masks.tex_3d;
  state.tex_1d_mask = masks.tex_1d;
  state.tex_cube_mask = masks.tex_cube;
  state.tex_uint_mask = masks.tex_uint;
  state.mrt_uint_mask = mrt_uint_mask;
  state.mrt_bound_mask = mrt_bound_mask;
  if (ps_addr)
    state.col_format = regs[mmSPI_SHADER_COL_FORMAT];
  state.gl_clip = !((regs[mmPA_CL_CLIP_CNTL] >> 19) & 1);
  return state;
}

RecompStatus ResolveRecompiledShaders(
    const Regs& regs,
    u64 vs_addr,
    const u32* vud,
    const gcn::GsPipeline* gs,
    u64 ps_addr,
    u64 fetch_addr,
    const base::SharedPointer<const gcn::Program>& ps_prog,
    const TextureMasks& masks,
    u32 mrt_uint_mask,
    u32 mrt_bound_mask,
    render::DrawInfo& d) {
  if (ShaderSkipped(vs_addr, ps_addr))
    return RecompStatus::kSkipped;
  if (!IsGuestAddress(vs_addr) || (ps_addr && !IsGuestAddress(ps_addr)))
    return RecompStatus::kBadAddress;

  u32 ps_in_cntl[32];
  for (u32 i = 0; i < 32; i++)
    ps_in_cntl[i] = regs[mmSPI_PS_INPUT_CNTL_0 + i];
  const u32 ps_input_ena = regs[mmSPI_PS_INPUT_ENA];
  TracePsInputCntl(regs, ps_addr, ps_input_ena, ps_in_cntl);
  TraceDepthBaseWatch(regs);

  GraphicsShaderState state =
      ShaderStateOf(regs, vs_addr, gs, ps_addr, fetch_addr, ps_in_cntl, masks,
                    mrt_uint_mask, mrt_bound_mask);
  const gcn::Recompiled* rc = &GetGraphicsShader(regs, state);
  if (!rc->ok)
    return RecompStatus::kRejected;

  d.ps4_neo = gcn::DefaultIsaMode() == gcn::IsaMode::kNeo;
  const u32* pud = regs.At(mmSPI_SHADER_USER_DATA_PS_0);
  const auto vs_prog = gcn::CachedProgram(vs_addr, 4096);
  thread_local base::Vector<gcn::VBuffer> direct_vbs;
  gcn::ResolveDirectVertexBuffers(direct_vbs, vs_prog, rc->attrs, vud, vs_addr);

  gcn::VBuffer attr_vbs[render::DrawInfo::kMaxVertexAttrs];
  u32 attr_count = 0;
  for (size_t i = 0;
       i < rc->attrs.size() && i < render::DrawInfo::kMaxVertexAttrs; i++) {
    const gcn::ShaderAttr& a = rc->attrs[i];
    if (!a.direct_fetch && a.table_sgpr + 1 >= 16)
      return RecompStatus::kBadAttrs;
    gcn::VBuffer vb;
    if (a.direct_fetch) {
      vb = direct_vbs[i];
    } else {
      const u64 table =
          UserDataPointer(vud, a.table_sgpr) + a.vbuf_dword_off * 4;
      if (!IsGuestAddress(table) || !gpu::IsReadableRangeCached(table, 16))
        return RecompStatus::kBadAttrs;
      vb = gcn::DecodeVBuffer(reinterpret_cast<const u32*>(table));
    }
    if (!IsGuestAddress(vb.base))
      return RecompStatus::kBadAttrs;
    attr_vbs[attr_count++] = vb;
  }
  state.int_attr_mask = IntAttrMask(*rc, attr_vbs, attr_count);
  if (state.int_attr_mask) {
    rc = &GetGraphicsShader(regs, state);
    if (!rc->ok)
      return RecompStatus::kRejected;
  }

  if (attr_count) {
    const RecompStatus status =
        BindVertexAttributes(*rc, attr_vbs, attr_count, d);
    if (status != RecompStatus::kOk) {
      d.num_vattrs = 0;
      d.num_vbufs = 0;
      return status;
    }
  }

  bool resolved_vs_cbuf = false;
  ResolveCbufferBindings(rc->vs_cbufs, vud, vs_prog, vs_addr, true, d,
                         resolved_vs_cbuf);
  const auto ps_program =
      ps_prog ? ps_prog
              : (IsGuestAddress(ps_addr) ? gcn::CachedProgram(ps_addr, 4096)
                                         : nullptr);
  if (ps_program)
    ResolveCbufferBindings(rc->ps_cbufs, pud, ps_program, ps_addr, false, d,
                           resolved_vs_cbuf);
  if (gs && !rc->gs_cbufs.empty())
    ResolveCbufferBindings(
        rc->gs_cbufs, regs.At(mmSPI_SHADER_USER_DATA_GS_0),
        gcn::CachedProgram(reinterpret_cast<u64>(gs->gs_code), 4096),
        reinterpret_cast<u64>(gs->gs_code), false, d, resolved_vs_cbuf);
  for (const auto& cb : rc->vs_cbufs)
    d.num_cbufs = base::Max(d.num_cbufs, cb.binding + 1);
  for (const auto& cb : rc->gs_cbufs)
    d.num_cbufs = base::Max(d.num_cbufs, cb.binding + 1);
  for (const auto& cb : rc->ps_cbufs)
    d.num_cbufs = base::Max(d.num_cbufs, cb.binding + 1);
  ResolveRawBuffers(rc->vs_bufs, vud, vs_prog, vs_addr, "vs", vs_addr, d);
  if (ps_program)
    ResolveRawBuffers(rc->ps_bufs, pud, ps_program, ps_addr, "ps", vs_addr, d);

  d.vs_addr = vs_addr;
  d.ps_addr = ps_addr;
  d.recomp = rc;
  return RecompStatus::kOk;
}

bool ResolveGsPipeline(const Regs& regs, u32 prim_type, gcn::GsPipeline& gs) {
  const u32 stages = regs[mmVGT_SHADER_STAGES_EN];
  if (!((stages >> 5) & 1) || ((stages >> 3) & 3) != 2 || (stages & 7))
    return false;
  const u64 es_addr = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_ES);
  const u64 gs_addr = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_GS);
  if (!IsGuestAddress(es_addr) || !IsGuestAddress(gs_addr))
    return false;
  gs.es_code = reinterpret_cast<const u32*>(es_addr);
  gs.gs_code = reinterpret_cast<const u32*>(gs_addr);
  gs.es_user_data = regs.At(mmSPI_SHADER_USER_DATA_ES_0);
  gs.es_user_sgprs = (regs[mmSPI_SHADER_PGM_RSRC2_ES] >> 1) & 0x1F;
  gs.gs_user_sgprs = (regs[mmSPI_SHADER_PGM_RSRC2_GS] >> 1) & 0x1F;
  gs.input_prim = prim_type;
  gs.out_prim = regs[mmVGT_GS_OUT_PRIM_TYPE] & 0x3F;
  gs.max_vert_out = regs[mmVGT_GS_MAX_VERT_OUT] & 0x7FF;
  gs.esgs_dwords = regs[mmVGT_ESGS_RING_ITEMSIZE] & 0x7FFF;
  gs.gsvs_dwords = regs[mmVGT_GSVS_RING_ITEMSIZE] & 0x7FFF;
  const u32 instancing = regs[mmVGT_GS_INSTANCE_CNT];
  gs.instances = (instancing & 1) ? base::Max(1u, (instancing >> 2) & 0x7F) : 1;
  return true;
}

}  // namespace

bool BuildDrawInfo(render::Renderer& renderer,
                   const Regs& regs,
                   const DrawPacket& packet,
                   render::DrawInfo& d) {
  DELTA_ZONE("ps4.build_draw_info");
  ScopeNs build_timer(&g_stats.ns_build_draw);
  g_stats.build_draw_n++;
  const gcn::ScalarReplayScope replays;
  d.prim_type = regs[mmVGT_PRIMITIVE_TYPE];
  gcn::GsPipeline gs;
  const bool has_gs = ResolveGsPipeline(regs, d.prim_type, gs);
  const u64 vs_addr = has_gs ? regs.ShaderAddr(mmSPI_SHADER_PGM_LO_ES)
                             : regs.ShaderAddr(mmSPI_SHADER_PGM_LO_VS);
  const u32* vud = regs.At(has_gs ? mmSPI_SHADER_USER_DATA_ES_0
                                  : mmSPI_SHADER_USER_DATA_VS_0);
  const u64 ps_addr = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_PS);
  std::memcpy(d.vs_user_data, vud, 16 * sizeof(u32));
  std::memcpy(d.ps_user_data, regs.At(mmSPI_SHADER_USER_DATA_PS_0),
              16 * sizeof(u32));
  d.instance_count = packet.num_instances;
  const u32 auto_vertex_count =
      packet.op == IT_DRAW_INDEX_AUTO && packet.count >= 1 ? packet.body[0] : 0;
  TraceDrawOpcode(packet.op, d.prim_type, auto_vertex_count);

  ResolveIndexBuffer(renderer, packet, d);
  const u32 mrt_uint_mask = ResolveRenderTargets(regs, d, vs_addr, ps_addr);
  ResolveColorState(regs, ps_addr, d);
  ResolveDepthState(regs, d);
  ResolveRasterState(regs, d);

  const u64 fetch_addr = FetchShaderAddress(vud, vs_addr);
  ResolveDefaultSources(vud, fetch_addr, d);

  TextureMasks masks;
  const auto ps_prog =
      ResolvePsTextures(renderer, regs, ps_addr, packet.frame, d, masks);
  ResolveVsTextures(vud, vs_addr, d, masks);

  const RecompStatus status = ResolveRecompiledShaders(
      regs, vs_addr, vud, has_gs ? &gs : nullptr, ps_addr, fetch_addr, ps_prog,
      masks, mrt_uint_mask, d.mrt_bound_mask, d);
  if (auto_vertex_count && auto_vertex_count <= kMaxElementCount)
    d.vertex_count = auto_vertex_count;
  if (status == RecompStatus::kOk && d.recomp->writes_layer)
    d.rt_layers = ((regs[mmCB_COLOR0_VIEW] >> 13) & 0x7FF) + 1;

  TraceBlitDraw(regs, d, vs_addr, ps_addr);
  TraceDrawList(regs, d, vs_addr, ps_addr, fetch_addr, status);
  TraceColorMasks(regs, d, ps_addr, status);
  TraceSpriteDraw(d);
  TraceVertexAttrs(d);
  TraceWorldGeometry(regs, d);

  if (kSkipStale && d.tex_base && d.tex_w >= 2048)
    return false;
  return d.vertex_data || (d.recomp && d.recomp->ok && d.num_vattrs == 0);
}

void PrefetchDrawShaders(const Regs& regs) {
  gcn::GsPipeline gs;
  const bool has_gs = ResolveGsPipeline(regs, regs[mmVGT_PRIMITIVE_TYPE], gs);
  const u64 vs_addr = has_gs ? regs.ShaderAddr(mmSPI_SHADER_PGM_LO_ES)
                             : regs.ShaderAddr(mmSPI_SHADER_PGM_LO_VS);
  const u64 ps_addr = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_PS);
  static base::HashSet<u64> seen;
  static const guest::SessionReset reset_seen(
      [] { guest::ResetResource(seen); });
  if (!seen.insert(vs_addr * 0x9e3779b97f4a7c15ull ^ ps_addr).second)
    return;
  if (ShaderSkipped(vs_addr, ps_addr) || !IsGuestAddress(vs_addr) ||
      (ps_addr && !IsGuestAddress(ps_addr)))
    return;
  const u32* vud = regs.At(has_gs ? mmSPI_SHADER_USER_DATA_ES_0
                                  : mmSPI_SHADER_USER_DATA_VS_0);
  thread_local render::DrawInfo d;
  d.Reset();
  const u32 mrt_uint_mask = ResolveRenderTargets(regs, d, vs_addr, ps_addr);
  TextureMasks masks;
  if (IsGuestAddress(ps_addr)) {
    thread_local base::Vector<gcn::TImage> texs;
    gcn::TrackTextures(texs, gcn::CachedProgram(ps_addr, 4096),
                       regs.At(mmSPI_SHADER_USER_DATA_PS_0), false, ps_addr);
    for (u32 i = 0; i < texs.size() && i < kMaxTrackedTextures; i++)
      BindTexture(d, texs[i], i, masks);
    d.num_texs =
        static_cast<u32>(base::Min<size_t>(texs.size(), kMaxTrackedTextures));
  }
  ResolveVsTextures(vud, vs_addr, d, masks);
  u32 ps_in_cntl[32];
  for (u32 i = 0; i < 32; i++)
    ps_in_cntl[i] = regs[mmSPI_PS_INPUT_CNTL_0 + i];
  PrefetchGraphicsShader(
      regs, ShaderStateOf(regs, vs_addr, has_gs ? &gs : nullptr, ps_addr,
                          FetchShaderAddress(vud, vs_addr), ps_in_cntl, masks,
                          mrt_uint_mask, d.mrt_bound_mask));
}

}  // namespace gpu::ps4
