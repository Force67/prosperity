/*
 * PS4Delta : PS4 emulation and research project
 *
 * How Liverpool register state becomes one renderer draw. See draw_state.h.
 */

#include "gpu/ps4/draw_state.h"
#include "base/arch.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <utl/mem.h>
#include <utl/options.h>

#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps4/cmd_trace.h"
#include "gpu/ps4/guest_address.h"
#include "gpu/ps4/pm4.h"
#include "gpu/ps4/shader_cache.h"

namespace {

DELTA_OPTION(u64, kCntlApply, "DELTA_GPU_PSCNTL_APPLY", 1);
DELTA_OPTION(bool, kForceDepth, "DELTA_GPU_FORCEDEPTH", false);
DELTA_OPTION(bool, kIntegerRt, "DELTA_GPU_INT_RT", true);
DELTA_OPTION(bool, kNoDepth, "DELTA_GPU_NODEPTH", false);
DELTA_OPTION(bool, kNoStencil, "DELTA_GPU_NOSTENCIL", false);
DELTA_OPTION(bool, kPreflushResources, "DELTA_GPU_PREFLUSHRES", false);
DELTA_OPTION(bool, kRecompOn, "DELTA_GPU_RECOMP", true);
DELTA_OPTION(bool, kSkipStale, "DELTA_GPU_SKIPSTALE", false);
DELTA_OPTION(const char*, kSkipShList, "DELTA_GPU_SKIPSH", nullptr);

}  // namespace

namespace gpu::ps4 {
namespace {

// Sampler bindings tracked per draw. This has to be the number the recompiler
// and the descriptor set layout use, not a smaller one of its own: texs[i] IS
// the module's set-0 binding i, and a vertex texture is numbered after the last
// pixel one. Truncating the pixel list here left every vertex binding one short
// per dropped entry, so a shader with 17+ samplers had a volume view land on a
// binding its module had declared 2D, and its last bindings written by nothing
// at all (VUID-vkCmdDrawIndexed-viewType-07752 / -None-08114).
constexpr u32 kMaxTrackedTextures = rhi::DrawInfo::kMaxDrawTextures;
// An index or vertex count beyond this is a decode error, not a draw.
constexpr u32 kMaxElementCount = 0x100000;

// Which sampled images need a different module than their code alone implies.
struct TextureMasks {
  u32 tex_3d = 0;
  u32 tex_1d = 0;
  u32 tex_uint = 0;
};

// The CB registers carry a pitch and a slice, not a width and a height, so the
// screen scissor is the only place the render-target geometry appears.
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

void FillDrawTex(rhi::DrawInfo::DrawTex& dt, const gcn::TImage& t) {
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

// Bind one tracked image, recording what its descriptor forces the module to
// be built with.
void BindTexture(rhi::DrawInfo& d,
                 const gcn::TImage& t,
                 u32 slot,
                 TextureMasks& masks) {
  FillDrawTex(d.texs[slot], t);
  if (t.is_3d)
    masks.tex_3d |= 1u << slot;
  if (t.is_1d)
    masks.tex_1d |= 1u << slot;
  // T# NUM_FORMAT 4/5 are UINT/SINT: the texels are packed bits, not a colour,
  // and must reach the shader unconverted.
  if (kIntegerRt && !t.storage && (t.nfmt == 4 || t.nfmt == 5))
    masks.tex_uint |= 1u << slot;
}

// --- packet decode ---------------------------------------------------------

// The draws are indexed triangle lists; without the index buffer (drawing raw
// vertices as a strip) batched sprites smear into long diagonal triangles.
// DRAW_INDEX_AUTO has no index buffer (sequential verts).
void ResolveIndexBuffer(rhi::Renderer& renderer,
                        const DrawPacket& packet,
                        rhi::DrawInfo& d) {
  const u32 op = packet.op, count = packet.count;
  const u32* body = packet.body;

  // DRAW_INDEX_2 (5 dwords): maxSize, baseLo, baseHi, index_count, initiator.
  if (op == IT_DRAW_INDEX_2 && count >= 4) {
    const u64 base =
        (static_cast<u64>(body[2] & 0xFF) << 32) | body[1];
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

  // DRAW_INDIRECT / DRAW_INDEX_INDIRECT take their counts from a struct in
  // guest memory at IT_SET_BASE(1) + dataOffset, not from the packet:
  //   indirect:       vertexCount, instanceCount, startVertex, startInstance
  //   index-indirect: indexCount,  instanceCount, startIndex, baseVertex, ...
  if ((op == IT_DRAW_INDIRECT || op == IT_DRAW_INDEX_INDIRECT) && count >= 1 &&
      packet.indirect_base) {
    const u64 args = packet.indirect_base + body[0];
    const u32 need = op == IT_DRAW_INDIRECT ? 16u : 20u;
    // The args are usually produced ON the GPU (a compute pass writes the
    // counts a later draw consumes). Compute results stay GPU-resident and are
    // written back lazily, so reading guest memory without flushing that range
    // first yields the stale zeros the buffer was allocated with.
    rhi::FlushCsWritesRange(renderer, args, need);
    const bool mapped =
        utl::isMemoryRangeMapped(reinterpret_cast<const void*>(args), need);
    u32 a[5] = {};
    if (mapped)
      std::memcpy(a, reinterpret_cast<const void*>(args), need);
    TraceIndirectArgs(op, args, packet.indirect_base, body[0], mapped, a);
    if (mapped && a[0] && a[0] <= kMaxElementCount) {
      if (op == IT_DRAW_INDIRECT) {
        d.vertex_count = a[0];
      } else if (packet.index_base) {
        const u32 index_bytes = packet.index_type == 1 ? 4 : 2;
        const u64 base =
            packet.index_base + static_cast<u64>(a[2]) * index_bytes;
        if (utl::isMemoryRangeMapped(
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
  }

  // DRAW_INDEX_OFFSET_2 (maxSize, indexOffset, indexCount, initiator) draws
  // from the index buffer IT_INDEX_BASE already set, starting indexOffset
  // entries in. SotC submits ~47% of its draws this way, and leaving the packet
  // undecoded left them with no index buffer at all: they fell back to the
  // vertex count, which a hand-fetch V# reports as 1, and then declined for
  // having fewer than 3 vertices.
  if (op == IT_DRAW_INDEX_OFFSET_2 && count >= 3) {
    const u32 index_offset = body[1], index_count = body[2];
    const u32 index_bytes = packet.index_type == 1 ? 4 : 2;
    const u64 base =
        packet.index_base + static_cast<u64>(index_offset) * index_bytes;
    const bool accepted = packet.index_base && index_count &&
                          index_count <= kMaxElementCount &&
                          utl::isMemoryRangeMapped(
                              reinterpret_cast<const void*>(base),
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

// --- register state --------------------------------------------------------

// A stale CB_COLORn_BASE remains programmed during depth-only passes, so a
// target counts as bound only when its write mask and its CB_COLORn_INFO format
// are both live. Returns the mask of targets with an integer texel format: the
// PS must declare an integer output for one, so that belongs to the module's
// identity. Kept as raw register bits rather than a VkFormat: this layer must
// not depend on the backend.
u32 ResolveRenderTargets(const Regs& regs,
                              rhi::DrawInfo& d,
                              u64 vs_addr,
                              u64 ps_addr) {
  d.rt_w = FbWidth(regs);
  d.rt_h = FbHeight(regs);
  d.scissor_tl = regs[mmPA_SC_VPORT_SCISSOR_0_TL];
  d.scissor_br = regs[mmPA_SC_VPORT_SCISSOR_0_BR];

  // Surface geometry of MRT0, independent of how much of it this draw touches.
  // PITCH_TILE_MAX counts 8-texel tiles minus one; SLICE_TILE_MAX counts
  // 64-texel tiles of the whole slice, so height = slice / pitch.
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
    // Each target carries its own geometry; the screen scissor is one global
    // that says how much of them this pass draws into.
    const u32 rt_pitch =
        ((regs[mmCB_COLOR0_PITCH + rt * kCbColorStride] & 0x7FFu) + 1u) * 8u;
    const u64 rt_slice =
        static_cast<u64>(
            (regs[mmCB_COLOR0_SLICE + rt * kCbColorStride] & 0x3FFFFFu) + 1u) *
        64u;
    d.mrt_surf_w[rt] = rt_pitch;
    d.mrt_surf_h[rt] = rt_pitch ? static_cast<u32>(rt_slice / rt_pitch) : 0u;
    const u32 nfmt = (info >> 8) & 0x7;
    if (kIntegerRt && (nfmt == 4 || nfmt == 5))
      mrt_uint_mask |= 1u << rt;
  }
  d.rt_base = d.mrt_count ? d.mrt_base[0] : 0;
  // A gap in the bound targets (slot 0 masked off while a higher slot is live)
  // leaves the primary null and drops the whole draw downstream, taking with it
  // a pass that renders only into MRT1.
  TraceMrtSlotZeroGap(d, target_mask, vs_addr, ps_addr);
  TraceNoMrtBound(regs, d, target_mask, vs_addr, ps_addr);
  return mrt_uint_mask;
}

// Blend, write masks and GNM's fast clear. Bit 30 of CB_BLENDn_CONTROL is the
// per-target blend enable; when clear the draw writes opaquely. Isaac's room
// vignette and its additive overlays rely on that: rendered with a single
// hardcoded blend they came out opaque and blacked out the scene.
void ResolveColorState(const Regs& regs, u64 ps_addr, rhi::DrawInfo& d) {
  d.blend_control = regs[mmCB_BLEND0_CONTROL];
  d.blend_enable = (d.blend_control >> 30) & 1u;
  // Target 0 mirrors blend_control/blend_enable, so the single-RT path is
  // unchanged; an MRT draw blends each attachment as the guest programmed it.
  d.mrt_blend[0] = d.blend_control;
  if (d.blend_enable)
    d.mrt_blend_mask |= 1u;
  for (u32 rt = 1; rt < 8; rt++) {
    const u32 blend = regs[mmCB_BLEND0_CONTROL + rt * kCbBlendStride];
    d.mrt_blend[rt] = blend;
    if ((blend >> 30) & 1u)
      d.mrt_blend_mask |= 1u << rt;
  }
  d.target_mask = regs[mmCB_TARGET_MASK];
  d.shader_mask = regs[mmCB_SHADER_MASK];

  // GNM fast clear: RECT_LIST (VGT prim 17), no pixel shader, no vertex
  // attributes, at least one colour target bound. The colour is in
  // CB_COLORn_CLEAR_WORD0/1, encoded in each target's own format. A depth-only
  // pass looks similar but binds no colour target, so mrt_count separates them.
  d.is_clear_rect =
      d.prim_type == 17 && !ps_addr && !d.num_vattrs && d.mrt_count != 0;
  if (d.is_clear_rect) {
    d.clear_tl = regs[mmPA_SC_GENERIC_SCISSOR_TL];
    d.clear_br = regs[mmPA_SC_GENERIC_SCISSOR_BR];
    d.clear_window_tl = regs[mmPA_SC_WINDOW_SCISSOR_TL];
    d.clear_window_br = regs[mmPA_SC_WINDOW_SCISSOR_BR];
    d.clear_screen_tl = regs[mmPA_SC_SCREEN_SCISSOR_TL];
    d.clear_screen_br = regs[mmPA_SC_SCREEN_SCISSOR_BR];
    for (u32 rt = 0; rt < 8; rt++) {
      d.mrt_clear_word[rt][0] =
          regs[mmCB_COLOR0_CLEAR_WORD0 + rt * kCbColorStride];
      d.mrt_clear_word[rt][1] =
          regs[mmCB_COLOR0_CLEAR_WORD1 + rt * kCbColorStride];
    }
  }
  d.color_control = regs[mmCB_COLOR_CONTROL];
}

// A 3D title binds a Z buffer and Z-tests the world; 2D titles leave DB_Z_INFO
// invalid so no depth attachment is bound. Z and stencil render into one Vulkan
// depth/stencil image; compute bridges expose their separate guest bases when
// later shaders consume either plane.
void ResolveDepthState(const Regs& regs, rhi::DrawInfo& d) {
  const u32 depth_control = regs[mmDB_DEPTH_CONTROL];
  d.depth_control = depth_control;
  const u32 z_info = kNoDepth ? 0 : regs[mmDB_Z_INFO];
  const u32 stencil_info = kNoDepth ? 0 : regs[mmDB_STENCIL_INFO];
  const u64 z_base = static_cast<u64>(regs[mmDB_Z_WRITE_BASE]) << 8;
  const u64 stencil_base =
      static_cast<u64>(regs[mmDB_STENCIL_WRITE_BASE]) << 8;
  TraceDepthState(regs, z_info);

  const u32 render_control = regs[mmDB_RENDER_CONTROL];
  d.render_control = render_control;
  const u32 size = regs[mmDB_DEPTH_SIZE];
  d.depth_w = ((size & 0x7FFu) + 1u) * 8u;          // PITCH_TILE_MAX
  d.depth_h = (((size >> 11) & 0x7FFu) + 1u) * 8u;  // HEIGHT_TILE_MAX
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
  // DELTA_GPU_FORCEDEPTH: exercise the depth attachment/clear/pipeline path on
  // titles that bind no Z buffer, to validate it end to end. func=ALWAYS so no
  // fragment is hidden (visible output unchanged); depth_base keys a synthetic
  // depth image off the RT.
  if (kForceDepth && !d.depth_base && d.rt_base) {
    d.depth_base = d.rt_base;
    d.depth_test_enable = true;
    d.depth_write_enable = true;
    d.depth_func = 7;  // ALWAYS
    d.depth_clear = 1.0f;
  }
}

void ResolveRasterState(const Regs& regs, rhi::DrawInfo& d) {
  const u32 mode = regs[mmPA_SU_SC_MODE_CNTL];
  d.cull_mode = mode & 0x3;  // CULL_FRONT[0] | CULL_BACK[1]
  d.front_ccw = ((mode >> 2) & 1u) == 0;
  std::memcpy(&d.viewport_x_scale, regs.At(mmPA_CL_VPORT_XSCALE), 4);
  std::memcpy(&d.viewport_x_offset, regs.At(mmPA_CL_VPORT_XOFFSET), 4);
  std::memcpy(&d.viewport_y_scale, regs.At(mmPA_CL_VPORT_YSCALE), 4);
  std::memcpy(&d.viewport_y_offset, regs.At(mmPA_CL_VPORT_YOFFSET), 4);
  std::memcpy(&d.viewport_z_scale, regs.At(mmPA_CL_VPORT_ZSCALE), 4);
  std::memcpy(&d.viewport_z_offset, regs.At(mmPA_CL_VPORT_ZOFFSET), 4);
}

// --- the heuristic (pre-recompiler) fallback -------------------------------

// The fetch shader the VS calls, or 0. s0:s1 is only a fetch-shader pointer
// when the VS actually calls one (s_swappc_b64). SotC parks its per-draw
// shader-resource-table pointer there instead: treating THAT as a fetch shader
// tracked phantom vertex buffers out of live constant data, and hashing the
// pointee into the shader key made a fresh key nearly every draw: 96% of all
// recompile-cache misses (4183 of 4352 over 210s).
u64 FetchShaderAddress(const Regs& regs, u64 vs_addr) {
  if (!IsGuestAddress(vs_addr) ||
      !gcn::CallsFetchShader(*gcn::CachedProgram(vs_addr, 4096)))
    return 0;
  const u64 fetch =
      UserDataPointer(regs.At(mmSPI_SHADER_USER_DATA_VS_0), 0);
  return IsGuestAddress(fetch) ? fetch : 0;
}

// The transform buffer and vertex streams the heuristic quad path uses when the
// recompiled path below declines the draw.
void ResolveHeuristicSources(const Regs& regs,
                             u64 fetch_addr,
                             rhi::DrawInfo& d) {
  const u32* vud = regs.At(mmSPI_SHADER_USER_DATA_VS_0);
  // Default to the sgpr[4..7] V# (the common VS cbuffer slot); the recompiled
  // path re-resolves it from the SGPR the VS actually reads.
  const u64 cbuf = UserDataPointer(vud, 4);
  if (IsGuestAddress(cbuf)) {
    std::memcpy(d.mvp, reinterpret_cast<const void*>(cbuf), 64);
    d.cbuf_base = cbuf;
  }
  if (!fetch_addr)
    return;
  auto vbs = gcn::TrackVertexBuffers(*gcn::CachedProgram(fetch_addr, 64), vud);
  if (vbs.empty())
    return;
  d.vertex_data = reinterpret_cast<const void*>(vbs[0].base);
  d.vertex_count = vbs[0].num_records;
  d.vertex_stride = vbs[0].stride;
  d.pos_offset = 0;
  // Per-attribute offsets from the fetch shader's V# bases: each attribute's V#
  // points at vertexBase + attributeOffset. dfmt 11 = 32_32 (float2 uv), dfmt
  // 10 = 8_8_8_8 (rgba8 colour). Generalises the old hardcoded sprite offsets
  // (pos@0, color@0x10, uv@0x1c), which remain the fallback.
  u32 uv_off = 0, col_off = 0;
  for (size_t i = 1; i < vbs.size(); i++) {
    const u64 off = vbs[i].base - vbs[0].base;
    if (off == 0 || off >= d.vertex_stride)
      continue;
    if (vbs[i].dfmt == 11 && !uv_off)
      uv_off = static_cast<u32>(off);
    else if (vbs[i].dfmt == 10 && !col_off)
      col_off = static_cast<u32>(off);
  }
  d.uv_offset = uv_off ? uv_off : (d.vertex_stride >= 0x1c ? 0x1c : 0);
  d.color_offset =
      col_off ? col_off : (d.vertex_stride >= 0x1c ? 0x10 : 0xFFFFFFFFu);
}

// --- sampled images --------------------------------------------------------

// The images the pixel shader samples, in its set-0 binding order. Returns the
// decoded program, which the recompiled path reuses.
std::shared_ptr<const gcn::Program> ResolvePsTextures(rhi::Renderer& renderer,
                                                      const Regs& regs,
                                                      u64 ps_addr,
                                                      u32 frame,
                                                      rhi::DrawInfo& d,
                                                      TextureMasks& masks) {
  if (!IsGuestAddress(ps_addr))
    return nullptr;
  if (kPreflushResources)
    rhi::FlushCsWrites(renderer);
  auto ps_prog = gcn::CachedProgram(ps_addr, 4096);
  const bool trace = ShouldTraceTextureTracking(frame, ps_addr);
  auto texs = gcn::TrackTextures(ps_prog, regs.At(mmSPI_SHADER_USER_DATA_PS_0),
                                 trace, ps_addr);
  if (texs.empty())
    return ps_prog;

  // Preserve every valid GFX7 T# address. Format support is relevant only when
  // uploading guest memory; a T# with R32F/RG16F/RGBA16F semantics may alias a
  // live renderer RT and must still resolve to that image.
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
  // Isaac's textured sprite vertex format is {pos.xyzw @0, color @0x10, uv.xy
  // @0x1c} in a 64-byte vertex, so the UV lives in the position buffer.
  d.uv_data = d.vertex_data;
  d.uv_stride = d.vertex_stride;

  d.num_texs =
      static_cast<u32>(std::min<size_t>(texs.size(), kMaxTrackedTextures));
  for (u32 i = 0; i < d.num_texs; i++)
    BindTexture(d, texs[i], i, masks);
  TraceTextureFormat(first, d);
  return ps_prog;
}

// A vertex texture fetch takes its own set-0 bindings, numbered after the PS's,
// so its descriptors continue the same list. Bloodborne's character sheet draws
// that way; without it the draw is rejected and falls back to the heuristic
// quad renderer, which paints the atlas as a staircase.
void ResolveVsTextures(const Regs& regs,
                       u64 vs_addr,
                       rhi::DrawInfo& d,
                       TextureMasks& masks) {
  if (!IsGuestAddress(vs_addr))
    return;
  auto texs =
      gcn::TrackTextures(gcn::CachedProgram(vs_addr, 4096),
                         regs.At(mmSPI_SHADER_USER_DATA_VS_0), false, vs_addr);
  for (const auto& t : texs) {
    if (d.num_texs >= kMaxTrackedTextures)
      break;
    BindTexture(d, t, d.num_texs++, masks);
  }
}

// --- the recompiled-shader path --------------------------------------------

// DELTA_GPU_SKIPSH=addr[,addr...] (hex): refuse to recompile draws whose VS or
// PS lives at one of these guest addresses, for shader-hang bisection.
bool ShaderSkipped(u64 vs_addr, u64 ps_addr) {
  static const std::vector<u64> kSkipped = [] {
    std::vector<u64> list;
    if (const char* spec = kSkipShList)
      for (const char* p = spec; *p;) {
        char* end;
        const u64 addr = std::strtoull(p, &end, 16);
        if (end == p)
          break;
        list.push_back(addr);
        p = *end == ',' ? end + 1 : end;
      }
    return list;
  }();
  return !kSkipped.empty() && (std::find(kSkipped.begin(), kSkipped.end(),
                                         vs_addr) != kSkipped.end() ||
                               std::find(kSkipped.begin(), kSkipped.end(),
                                         ps_addr) != kSkipped.end());
}

// Group the attributes' V#s into vertex bindings. Attributes that interleave in
// one buffer (same stride, base within one stride of the binding base) share a
// binding with distinct offsets; attributes fed from a separate buffer get
// their own. SotC streams position/normal/uv from distinct buffers with
// distinct strides, which the old single-stream model declined outright.
RecompStatus BindVertexAttributes(const gcn::Recompiled& rc,
                                  const gcn::VBuffer* attr_vbs,
                                  u32 attr_count,
                                  rhi::DrawInfo& d) {
  u32 attr_binding[rhi::DrawInfo::kMaxVertexAttrs] = {};
  for (u32 i = 0; i < attr_count; i++) {
    const gcn::VBuffer& vb = attr_vbs[i];
    int sel = -1;
    for (u32 j = 0; j < d.num_vbufs; j++) {
      if (d.vbufs[j].stride != vb.stride)
        continue;
      const u64 bound = reinterpret_cast<u64>(d.vbufs[j].data);
      const u64 lo = std::min(bound, vb.base);
      const u64 hi = std::max(bound, vb.base);
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
                                vb.stride, vb.num_records};
    } else {
      auto& bind = d.vbufs[sel];
      if (vb.base < reinterpret_cast<u64>(bind.data))
        bind.data = reinterpret_cast<const void*>(vb.base);
      bind.num_records = std::min(bind.num_records, vb.num_records);
    }
    attr_binding[i] = static_cast<u32>(sel);
  }
  // Second pass: offsets are relative to each binding's final (lowest) base.
  for (u32 i = 0; i < attr_count; i++) {
    const gcn::ShaderAttr& a = rc.attrs[i];
    const gcn::VBuffer& vb = attr_vbs[i];
    const u32 binding = attr_binding[i];
    const u64 offset =
        vb.base - reinterpret_cast<u64>(d.vbufs[binding].data);
    // Strided bindings must keep every attribute inside one record; a stride-0
    // (constant) binding has no record extent to bound.
    if (d.vbufs[binding].stride && offset >= d.vbufs[binding].stride)
      return RecompStatus::kAttrOffset;
    // A typed fetch (tbuffer_load_format_*) states its own format and the
    // hardware ignores the V#'s; only untyped fetches read it.
    const u32 dfmt = a.inst_dfmt ? a.inst_dfmt : vb.dfmt;
    const u32 nfmt = a.inst_dfmt ? a.inst_nfmt : vb.nfmt;
    d.vattrs[d.num_vattrs++] = {
        a.location,  binding, static_cast<u32>(offset),
        a.num_comps, dfmt,    nfmt};
  }

  // vertex_data/vertex_stride mirror the first per-vertex (strided) binding for
  // the heuristic fallback and clear detection; vertex_count is bounded by the
  // smallest strided binding's record count (stride-0 constant bindings do not
  // constrain it).
  u32 primary = 0;
  while (primary < d.num_vbufs && !d.vbufs[primary].stride)
    primary++;
  d.vertex_data = d.vbufs[primary < d.num_vbufs ? primary : 0].data;
  d.vertex_stride = primary < d.num_vbufs ? d.vbufs[primary].stride : 0;
  u32 records = UINT32_MAX;
  for (u32 j = 0; j < d.num_vbufs; j++)
    if (d.vbufs[j].stride)
      records = std::min(records, d.vbufs[j].num_records);
  d.vertex_count = records == UINT32_MAX ? 0 : records;
  return RecompStatus::kOk;
}

// Resolve every cbuffer V# an emitted stage reads, following EUD/SRT pointer
// chains (FOX loads the V# through an extended-user-data pointer, so it is not
// sitting directly in user data at cb.ud_sgpr). Bindings are assigned by the
// translator and shared across both stages in descriptor set 1.
void ResolveCbufferBindings(const std::vector<gcn::ShaderCbuf>& cbufs,
                            const u32* user_data,
                            const std::shared_ptr<const gcn::Program>& program,
                            bool vertex_stage,
                            rhi::DrawInfo& d,
                            bool& resolved_vs_cbuf) {
  auto resolved = gcn::ResolveCbuffers(program, user_data);
  for (const auto& cb : cbufs) {
    if (cb.binding >= 8)
      continue;
    gcn::VBuffer vb{};
    auto it = resolved.find(cb.ud_sgpr | (cb.pointer ? 0x100u : 0u));
    if (it != resolved.end())
      vb = it->second;  // EUD-resolved V# (handles indirection)
    else if (cb.pointer && cb.ud_sgpr + 1 < 16)
      vb.base = UserDataPointer(user_data, cb.ud_sgpr);  // flat pointer inline
    else if (!cb.pointer && cb.ud_sgpr + 3 < 16)
      vb = gcn::DecodeVBuffer(&user_data[cb.ud_sgpr]);  // inline V#
    // An s_load table carries no size; the shader's own highest read bounds the
    // window.
    u64 bytes =
        vb.stride ? (u64)vb.stride * vb.num_records : vb.num_records;
    if (cb.pointer)
      bytes = (u64)cb.num_dwords * 4;
    if (!IsGuestRange(vb.base, bytes) || bytes > 0xFFFFFFFFull)
      continue;
    d.cbufs[cb.binding] = {vb.base, static_cast<u32>(bytes)};
    d.num_cbufs = std::max(d.num_cbufs, cb.binding + 1);
    if (cb.pointer)
      continue;  // an SRT root is not a transform buffer
    if (vertex_stage && !resolved_vs_cbuf) {
      resolved_vs_cbuf = true;
      d.cbuf_base = vb.base;
      d.cbuf_size = static_cast<u32>(bytes);
      if (bytes >= sizeof(d.mvp))
        std::memcpy(d.mvp, reinterpret_cast<const void*>(vb.base),
                    sizeof(d.mvp));
    }
  }
}

// The live V# behind every raw buffer an emitted stage reads with MUBUF. Same
// scalar replay as the cbuffers: the descriptor is read at the instruction that
// consumes it, so an SRT-chained V# lands here as well as an inline one.
void ResolveRawBuffers(const std::vector<gcn::ShaderBuffer>& buffers,
                       const u32* user_data,
                       const std::shared_ptr<const gcn::Program>& program,
                       const char* stage,
                       u64 vs_addr,
                       rhi::DrawInfo& d) {
  if (buffers.empty())
    return;
  const auto resolved = gcn::ResolveShaderBuffers(program, buffers, user_data);
  for (size_t i = 0; i < buffers.size(); i++) {
    const gcn::ShaderBuffer& sb = buffers[i];
    if (sb.binding >= rhi::DrawInfo::kMaxBuffers)
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
    d.bufs[sb.binding] = {vb.base, static_cast<u32>(bytes)};
    d.num_bufs = std::max(d.num_bufs, sb.binding + 1);
  }
}

// Run the game's own shaders for this draw: recompile the VS/PS pair (cached)
// and resolve the live buffers they read. The heuristic fields stay populated
// as the fallback for a draw this declines.
RecompStatus ResolveRecompiledShaders(
    const Regs& regs,
    u64 vs_addr,
    u64 ps_addr,
    u64 fetch_addr,
    const std::shared_ptr<const gcn::Program>& ps_prog,
    const TextureMasks& masks,
    u32 mrt_uint_mask,
    rhi::DrawInfo& d) {
  if (ShaderSkipped(vs_addr, ps_addr))
    return RecompStatus::kSkipped;
  if (!kRecompOn)
    return RecompStatus::kDisabled;
  if (!IsGuestAddress(vs_addr) || (ps_addr && !IsGuestAddress(ps_addr)))
    return RecompStatus::kBadAddress;

  u32 ps_in_cntl[32];
  for (u32 i = 0; i < 32; i++)
    ps_in_cntl[i] = regs[mmSPI_PS_INPUT_CNTL_0 + i];
  const u32 ps_input_ena = regs[mmSPI_PS_INPUT_ENA];
  TracePsInputCntl(regs, ps_addr, ps_input_ena, ps_in_cntl);
  TraceDepthBaseWatch(regs);

  GraphicsShaderState state;
  state.vs_addr = vs_addr;
  state.ps_addr = ps_addr;
  state.fetch_addr = fetch_addr;
  state.ps_input_ena = ps_input_ena;
  state.ps_in_cntl = ps_in_cntl;
  // A PS names an input SLOT; SPI_PS_INPUT_CNTL_<slot>.OFFSET names the VS
  // parameter export that slot reads, and it is routinely not the identity.
  // GTA:SA's UI shader has slot 1 -> param 3: assuming identity handed it the
  // clip position as a texture coordinate and the vertex colour as a scale, so
  // every UI quad sampled texel (0,0) and multiplied it by zero -- a black
  // screen over a main menu that was otherwise drawing correctly.
  // DELTA_GPU_PSCNTL_APPLY=0 goes back to the identity, =<ps addr> applies the
  // mapping to one shader (Isaac/Undertale/Doom64 are unchanged either way).
  state.honour_ps_in_cntl = kCntlApply != 0 && (kCntlApply == 1 ||
                                                ps_addr == (u64)kCntlApply);
  state.ps_num_interp = regs[mmSPI_PS_IN_CONTROL] & 0x3F;
  state.tex_3d_mask = masks.tex_3d;
  state.tex_1d_mask = masks.tex_1d;
  state.tex_uint_mask = masks.tex_uint;
  state.mrt_uint_mask = mrt_uint_mask;
  state.gl_clip = !((regs[mmPA_CL_CLIP_CNTL] >> 19) & 1);

  const gcn::Recompiled& rc = GetGraphicsShader(regs, state);
  if (!rc.ok)
    return RecompStatus::kRejected;

  d.ps4_neo = gcn::DefaultIsaMode() == gcn::IsaMode::kNeo;
  const u32* vud = regs.At(mmSPI_SHADER_USER_DATA_VS_0);
  const u32* pud = regs.At(mmSPI_SHADER_USER_DATA_PS_0);
  const auto vs_prog = gcn::CachedProgram(vs_addr, 4096);
  const auto direct_vbs =
      gcn::ResolveDirectVertexBuffers(vs_prog, rc.attrs, vud);

  gcn::VBuffer attr_vbs[rhi::DrawInfo::kMaxVertexAttrs];
  u32 attr_count = 0;
  for (size_t i = 0;
       i < rc.attrs.size() && i < rhi::DrawInfo::kMaxVertexAttrs; i++) {
    const gcn::ShaderAttr& a = rc.attrs[i];
    if (!a.direct_fetch && a.table_sgpr + 1 >= 16)
      return RecompStatus::kBadAttrs;
    gcn::VBuffer vb;
    if (a.direct_fetch) {
      vb = direct_vbs[i];
    } else {
      const u64 table = UserDataPointer(vud, a.table_sgpr);
      if (!IsGuestAddress(table))
        return RecompStatus::kBadAttrs;
      vb = gcn::DecodeVBuffer(
          reinterpret_cast<const u32*>(table + a.vbuf_dword_off * 4));
    }
    if (!IsGuestAddress(vb.base))
      return RecompStatus::kBadAttrs;
    // A stride-0 V# is a per-draw constant input (all vertices read the same
    // record) and becomes a stride-0 Vulkan binding. Instance stepping uses a
    // normal stride indexed by the fetch shader, so a zero stride is
    // unambiguously a constant.
    attr_vbs[attr_count++] = vb;
  }

  if (attr_count) {
    const RecompStatus status =
        BindVertexAttributes(rc, attr_vbs, attr_count, d);
    if (status != RecompStatus::kOk) {
      d.num_vattrs = 0;
      d.num_vbufs = 0;
      return status;
    }
  }

  bool resolved_vs_cbuf = false;
  ResolveCbufferBindings(rc.vs_cbufs, vud, vs_prog, true, d, resolved_vs_cbuf);
  const auto ps_program =
      ps_prog ? ps_prog
              : (IsGuestAddress(ps_addr) ? gcn::CachedProgram(ps_addr, 4096)
                                         : nullptr);
  if (ps_program)
    ResolveCbufferBindings(rc.ps_cbufs, pud, ps_program, false, d,
                           resolved_vs_cbuf);
  for (const auto& cb : rc.vs_cbufs)
    d.num_cbufs = std::max(d.num_cbufs, cb.binding + 1);
  for (const auto& cb : rc.ps_cbufs)
    d.num_cbufs = std::max(d.num_cbufs, cb.binding + 1);
  ResolveRawBuffers(rc.vs_bufs, vud, vs_prog, "vs", vs_addr, d);
  if (ps_program)
    ResolveRawBuffers(rc.ps_bufs, pud, ps_program, "ps", vs_addr, d);

  d.vs_addr = vs_addr;
  d.ps_addr = ps_addr;
  d.recomp = &rc;
  return RecompStatus::kOk;
}

}  // namespace

bool BuildDrawInfo(rhi::Renderer& renderer,
                   const Regs& regs,
                   const DrawPacket& packet,
                   rhi::DrawInfo& d) {
  const u64 vs_addr = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_VS);
  const u64 ps_addr = regs.ShaderAddr(mmSPI_SHADER_PGM_LO_PS);
  std::memcpy(d.vs_user_data, regs.At(mmSPI_SHADER_USER_DATA_VS_0),
              16 * sizeof(u32));
  std::memcpy(d.ps_user_data, regs.At(mmSPI_SHADER_USER_DATA_PS_0),
              16 * sizeof(u32));
  d.prim_type = regs[mmVGT_PRIMITIVE_TYPE];
  d.instance_count = packet.num_instances;
  const u32 auto_vertex_count =
      packet.op == IT_DRAW_INDEX_AUTO && packet.count >= 1 ? packet.body[0] : 0;
  TraceDrawOpcode(packet.op, d.prim_type, auto_vertex_count);

  ResolveIndexBuffer(renderer, packet, d);
  const u32 mrt_uint_mask =
      ResolveRenderTargets(regs, d, vs_addr, ps_addr);
  ResolveColorState(regs, ps_addr, d);
  ResolveDepthState(regs, d);
  ResolveRasterState(regs, d);

  const u64 fetch_addr = FetchShaderAddress(regs, vs_addr);
  ResolveHeuristicSources(regs, fetch_addr, d);

  TextureMasks masks;
  const auto ps_prog =
      ResolvePsTextures(renderer, regs, ps_addr, packet.frame, d, masks);
  ResolveVsTextures(regs, vs_addr, d, masks);

  const RecompStatus status = ResolveRecompiledShaders(
      regs, vs_addr, ps_addr, fetch_addr, ps_prog, masks, mrt_uint_mask, d);
  if (auto_vertex_count && auto_vertex_count <= kMaxElementCount)
    d.vertex_count = auto_vertex_count;

  TraceBlitDraw(regs, d, vs_addr, ps_addr);
  TraceDrawList(regs, d, vs_addr, ps_addr, fetch_addr, status);
  TraceColorMasks(regs, d, ps_addr, status);
  TraceSpriteDraw(d);
  TraceVertexAttrs(d);
  TraceWorldGeometry(regs, d);

  // DELTA_GPU_SKIPSTALE: drop draws that sample a very wide (>=2048) buffer, to
  // hide a title's stale full-screen video-buffer blit (Doom64's undecoded 4K
  // menu background) so the menu items drawn on top become readable.
  if (kSkipStale && d.tex_base && d.tex_w >= 2048)
    return false;
  return d.vertex_data || (d.recomp && d.recomp->ok && d.num_vattrs == 0);
}

}  // namespace gpu::ps4
