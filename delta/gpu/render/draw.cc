/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// The renderer's draw entry point: applies the diagnostic draw knobs, tries the
// recompiled-shader path, and falls back to the heuristic quad path (repack the
// guest vertices into pos/colour/uv and draw them with a fixed shader pair) for
// draws that path cannot run.

#include "gpu/render/renderer.h"
#include "base/arch.h"

#include "gpu/render/labels.h"
#include "gpu/render/device.h"
#include "gpu/render/draw_recomp.h"
#include "gpu/render/guest_format.h"
#include "gpu/render/frame.h"
#include "gpu/render/index_upload.h"
#include "gpu/gpu_perf.h"
#include "gpu/guest_memory.h"
#include "gpu/render/pipeline_cache.h"
#include "gpu/render/render_target.h"
#include "gpu/render/texture_cache.h"
#include "gpu/render/trace.h"
#include "gpu/render/upload_ring.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <base/logging.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(int, kMaxDraw, "DELTA_GPU_MAXDRAW", -1);
DELTA_OPTION(int, kOnlyDraw, "DELTA_GPU_ONLYDRAW", -1);
DELTA_OPTION(u32, kOnlyIc, "DELTA_GPU_ONLYIC", 0);
DELTA_OPTION(bool, kRecompPath, "DELTA_GPU_RECOMP", true);
DELTA_OPTION(bool, kDrawTraceAll, "DELTA_GPU_DRAWTRACE", false);
DELTA_OPTION(bool, kNoCull, "DELTA_GPU_NOCULL", false);
DELTA_OPTION(bool, kNoDepth, "DELTA_GPU_NODEPTH", false);
DELTA_OPTION(bool, kNoMask, "DELTA_GPU_NOMASK", false);
DELTA_OPTION(bool, kSwapTex, "DELTA_GPU_SWAPTEX01", false);
}  // namespace

namespace gpu::render {

void Draw(Renderer& renderer, const DrawInfo& d_in) {
  if (!g_frame.recording)
    return;
  // DELTA_GPU_SWAPTEX01: bisect a suspected sampler-binding order mismatch by
  // exchanging the first two textures of every multi-texture draw.
  std::optional<DrawInfo> swapped;
  if (kSwapTex && d_in.num_texs >= 2) {
    swapped = d_in;
    std::swap(swapped->texs[0], swapped->texs[1]);
  }
  const DrawInfo& d_sw = swapped ? *swapped : d_in;
  // DELTA_GPU_MAXDRAW=<n> / DELTA_GPU_ONLYDRAW=<n>: build a frame up one draw
  // at a time, or isolate a single one, to see what each pass contributes.
  // DELTA_GPU_ONLYIC=<n>: render only draws with this index count. Draw indices
  // move between frames; an index count names one pass reliably.
  if (kOnlyIc && d_sw.index_count != kOnlyIc) {
    g_frame.draws++;
    return;
  }
  if (kMaxDraw >= 0 && (int)g_frame.draws >= kMaxDraw)
    return;
  if (kOnlyDraw >= 0 && (int)g_frame.draws != kOnlyDraw) {
    g_frame.draws++;
    return;
  }
  // Diagnostic kill-switches for bisecting "renders nothing" chains:
  // DELTA_GPU_NODEPTH disables depth test/write, DELTA_GPU_NOCULL disables
  // face culling, DELTA_GPU_NOMASK forces full color write masks.
  // A pass that samples the depth buffer it also has bound is reading it as a
  // texture, which is legal while depth testing and writes are off. Detaching
  // the unused attachment lets the draw run instead of being declined as
  // self-sampling: Skyrim's grading pass does exactly this, and dropping it
  // left the display buffer showing ungraded content on those frames, so the
  // frame alternated between graded and raw as the pass came and went.
  bool detach_depth = false;
  if (d_sw.depth_base && !d_sw.depth_test_enable && !d_sw.depth_write_enable) {
    if (d_sw.tex_base == d_sw.depth_base)
      detach_depth = true;
    for (u32 i = 0; i < d_sw.num_texs && !detach_depth; i++)
      if (d_sw.texs[i].base == d_sw.depth_base)
        detach_depth = true;
  }
  // Copied only when patched: DrawInfo is ~10 KB and this runs every draw.
  std::optional<DrawInfo> patched_copy;
  const bool patched = kNoDepth || kNoCull || kNoMask || detach_depth;
  if (patched) {
    patched_copy = d_sw;
    DrawInfo& dd = *patched_copy;
    if (kNoDepth) {
      dd.depth_test_enable = false;
      dd.depth_write_enable = false;
    }
    if (kNoCull)
      dd.cull_mode = 0;
    if (kNoMask) {
      dd.target_mask = 0xFFFFFFFFu;
      dd.color_control = 0x10;
    }
    if (detach_depth) {
      dd.depth_base = 0;
      dd.depth_test_enable = false;
    }
  }
  const DrawInfo& d = patched ? *patched_copy : d_sw;
  if (d.index_count > g_frame.max_idx)
    g_frame.max_idx = d.index_count;
  ScopeNs draw_timer(&g_ns_draw);
  ScopeNs frame_draw_timer(&g_fr_draw);
  g_win_draws++;
  if (d.index_data && d.index_count) {
    const u64 index_bytes = static_cast<u64>(d.index_count) *
                                 GuestIndexElementBytes(d.index_type);
    if (!FlushCsWritesRange(renderer, reinterpret_cast<u64>(d.index_data),
                            index_bytes))
      return;
  }
  // Recompiled-shader path: run the game's actual VS/PS. Falls through to the
  // heuristic quad path when the draw can't be handled. On by default now that
  // it renders gameplay correctly; DELTA_GPU_RECOMP=0 forces the old heuristic
  // path.
  if (ShaderFilterDrops(d.ps_addr)) {
    g_frame.draws++;
    return;
  }
  const bool recompiled = kRecompPath && d.recomp && DrawRecomp(renderer, d);
  if (!renderer.available())
    return;
  if (kDrawTraceAll) {
    static u32 traced = 0;
    if (traced++ < 100)
      BASE_LOGI("dt", "f{} rt={:#x} count={} indexed={} nv={} mrt={} mask={:#x} "
                      "psmask={:#x} prim={} vp=[{:.1f} {:.1f} {:.1f} {:.1f}] "
                      "depth={:#x} handled={:d}",
                g_frame.num, (unsigned long)d.rt_base, d.vertex_count,
                d.index_count, d.num_vattrs, d.mrt_count, d.target_mask,
                d.recomp ? d.recomp->ps_mrt_mask : 0, d.prim_type,
                d.viewport_x_scale, d.viewport_x_offset, d.viewport_y_scale,
                d.viewport_y_offset, (unsigned long)d.depth_base,
                recompiled);
  }
  if (recompiled)
    return;
  if (!d.vertex_data || !d.vertex_stride)
    return;
  // No colour target: this path renders ONE colour attachment and nothing else,
  // so a depth-only pass has nothing it can contribute here. It used to open a
  // region on an RT keyed at address 0 whose geometry followed whichever draw
  // touched it last, and then render a 1020x1020 shadow pass into a 1470x827
  // attachment, rendering outside the attachment, which is undefined and
  // takes the device down during GTA:SA's level load.
  if (!d.rt_base)
    return;
  if (!FlushCsWritesRange(renderer, reinterpret_cast<u64>(d.vertex_data),
                          static_cast<u64>(d.vertex_stride) *
                              (d.vertex_count ? d.vertex_count : 1)))
    return;
  // Indexed triangle list (the common GNM draw): the index buffer selects which
  // vertices form each triangle. Find how many vertices the indices reference
  // so we repack exactly that many (the V# num_records can be the whole shared
  // batch).
  bool indexed = d.index_data && d.index_count >= 3;
  u32 nv = d.vertex_count;
  if (indexed) {
    if (d.index_count > 1500000u)
      return;
    const u32 max_index =
        MaxGuestIndex(d.index_data, d.index_count, d.index_type);
    if (max_index >= 200000u)
      return;
    nv = max_index + 1;
  }
  if (nv < 3 || nv > 200000u)
    return;                                   // sane cap
  if (!IsReadableRange(reinterpret_cast<u64>(d.vertex_data),
                       static_cast<u64>(nv) * d.vertex_stride))
    return;
  u64 need = (u64)nv * 32;  // pos.xy + color.rgba + uv.xy
  if (g_ring.vb_offset + need > g_ring.vb_end)
    return;  // ring full this frame
  const u64 index_bytes =
      indexed ? static_cast<u64>(d.index_count) *
                    UploadedIndexElementBytes(d.index_type)
              : 0;
  const u64 index_align = d.index_type == 1 ? 4 : 2;
  const u64 aligned_ioff =
      (g_ring.ib_offset + index_align - 1) & ~(index_align - 1);
  if (indexed && aligned_ioff + index_bytes > g_ring.ib_end)
    return;
  // Repack pos / color / uv interleaved into the vertex ring (stride 32).
  auto* base = static_cast<const u8*>(d.vertex_data);
  auto* dst = reinterpret_cast<float*>(g_ring.vb_map + g_ring.vb_offset);
  for (u32 v = 0; v < nv; v++) {
    const u8* vert = base + (size_t)v * d.vertex_stride;
    auto* p = reinterpret_cast<const float*>(vert + d.pos_offset);
    dst[v * 8 + 0] = p[0];
    dst[v * 8 + 1] = p[1];
    if (d.color_offset != 0xFFFFFFFFu) {
      auto* c = reinterpret_cast<const float*>(vert + d.color_offset);
      dst[v * 8 + 2] = c[0];
      dst[v * 8 + 3] = c[1];
      dst[v * 8 + 4] = c[2];
      dst[v * 8 + 5] = 1.0f;
    } else {
      dst[v * 8 + 2] = dst[v * 8 + 3] = dst[v * 8 + 4] = dst[v * 8 + 5] = 1.0f;
    }
    if (d.uv_data && d.uv_stride) {
      auto* u = reinterpret_cast<const float*>(vert + d.uv_offset);
      dst[v * 8 + 6] = u[0];
      dst[v * 8 + 7] = u[1];
    } else {
      dst[v * 8 + 6] = dst[v * 8 + 7] = 0.0f;
    }
  }

  // Resolve the sampled texture address to a render target via overlap (the
  // resource-model page-table lookup): an exact RT base, or an address whose
  // footprint overlaps a live RT, binds that RT's image instead of stale guest
  // memory. This replaces the old per-symptom FRESHRT/CYCLEREDIR/ROOMALPHA
  // address heuristics with one principled, game-agnostic lookup.
  u64 tex_base = d.tex_base;
  if (tex_base && !d.tex_arrayed && !g_rts.count(tex_base)) {
    u64 r = ResolveSampledRT(tex_base, d.tex_w, d.tex_h);
    if (r)
      tex_base = r;
  }
  // Is this a render-to-texture sample (the draw samples another render
  // target)?
  bool rt_as_tex = !d.tex_arrayed && tex_base && tex_base != d.rt_base &&
                   g_rts.count(tex_base);
  bool room_src =
      rt_as_tex && g_rts[tex_base].w >= 700 && g_rts[tex_base].w <= 900;
  if (room_src)
    g_frame.had_room = true;

  // Upload guest texture (independent of the render region) if not
  // RT-as-texture.
  rhi::BindGroup* tex_set = nullptr;
  if (d.tex_base && g_quad.tex_pipeline && !rt_as_tex && !d.tex_arrayed &&
      GuestTextureUploadSupported(d.tex_dfmt, d.tex_nfmt))
    tex_set = (GetTexture(
        d.tex_base, d.tex_w, d.tex_h, d.tex_dfmt, d.tex_nfmt, d.tex_tiling,
        d.tex_pitch, d.tex_layers, d.tex_base_array, d.tex_view_layers,
        d.tex_mip_levels, d.tex_base_mip, d.tex_view_mips, d.tex_min_lod,
        d.tex_pow2_pad, d.tex_sampler, d.tex_sampler_valid, false,
        d.tex_force_lod_zero, d.tex_depth_compare, d.tex_swizzle));

  // Switch render target if this draw targets a different RT than the open
  // region (or the open region is multi-target/has a depth attachment: the
  // heuristic path renders to a single color attachment with no depth).
  const bool transition_source =
      rt_as_tex &&
      g_rts[tex_base].layout != rhi::TextureState::kShaderRead;
  if (g_region.cur_rt != d.rt_base || g_region.cur_mrt_count != 1 ||
      g_region.cur_depth != 0 || transition_source) {
    EndRegion();
    if (rt_as_tex) {  // make the sampled RT shader-readable before we render
      auto& src = g_rts[tex_base];
      if (src.layout != rhi::TextureState::kShaderRead)
        TransitionImage(g_frame.list, src.texture, src.layout,
                        rhi::TextureState::kShaderRead);
    }
    rhi::Format rt_format = ColorTargetFormat(d.mrt_info[0]);
    RTarget* rt = GetRT(d.rt_base, RtSurfaceExtent(d.mrt_surf_w[0], d.rt_w, 256),
                        RtSurfaceExtent(d.mrt_surf_h[0], d.rt_h, 64), rt_format);
    if (!rt) {
      g_frame.draws++;
      return;
    }
    if (!BeginRegion(d.mrt_base, d.mrt_info, 1, d.rt_w, d.rt_h, 0, 1.0f, 0, 0,
                     false, 0, 0, d.mrt_surf_w, d.mrt_surf_h)) {
      g_frame.draws++;
      return;
    }
  }
  if (rt_as_tex &&
      g_rts[tex_base].layout == rhi::TextureState::kShaderRead)
    tex_set = g_rts[tex_base].set;

  g_frame.heuristic++;
  SetGuestViewport(d);
  rhi::CommandList* list = g_frame.list;
  const u64 off = g_ring.vb_offset;
  if (tex_set) {
    // Per-draw blend from the guest's CB_BLEND0_CONTROL, real vertex UVs.
    list->SetPipeline(GetPipeline(true, d.blend_control, d.blend_enable,
                                  ColorTargetFormat(d.mrt_info[0])));
    float pc[17];
    std::memcpy(pc, d.mvp, 64);
    reinterpret_cast<u32*>(pc)[16] =
        0u;  // clipUV: real per-vertex uv/colour
    list->SetPushConstants(0, 68, pc);
    list->SetBindGroup(0, tex_set);
  } else {
    list->SetPipeline(GetPipeline(false, d.blend_control, d.blend_enable,
                                  ColorTargetFormat(d.mrt_info[0])));
    list->SetPushConstants(0, 64, d.mvp);
  }
  list->SetBlendConstants(d.blend_constants);
  list->SetVertexBuffers(0, 1, &g_ring.vb, &off);
  CmdInsertLabel(g_frame.list, "quad vs=%#llx ps=%#llx n=%u",
                 (unsigned long long)d.vs_addr, (unsigned long long)d.ps_addr,
                 indexed ? d.index_count : nv);
  if (indexed) {
    u64 ioff = aligned_ioff;
    CopyGuestIndices(g_ring.ib_map + ioff, d.index_data, d.index_count,
                     d.index_type);
    list->SetIndexBuffer(g_ring.ib, ioff,
                         d.index_type == 1 ? rhi::IndexType::kUint32
                                           : rhi::IndexType::kUint16);
    list->DrawIndexed(d.index_count, d.instance_count ? d.instance_count : 1,
                      0, 0, 0);
    g_ring.ib_offset = ioff + index_bytes;
  } else {
    list->Draw(nv, d.instance_count ? d.instance_count : 1, 0, 0);
  }
  g_ring.vb_offset += need;
  g_frame.draws++;
  if (g_region.cur_rt) {
    auto& rt = g_rts[g_region.cur_rt];
    if (++rt.draws > g_region.busiest_rt_draws) {
      g_region.busiest_rt_draws = rt.draws;
      g_region.busiest_rt = g_region.cur_rt;
    }
  }
  if (trace::Recording())
    trace::RecordDraw(d, "quad", nullptr);
}

}  // namespace gpu::render
