/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// The renderer's draw entry point: applies the diagnostic draw knobs, tries the
// recompiled-shader path, and falls back to the heuristic quad path (repack the
// guest vertices into pos/colour/uv and draw them with a fixed shader pair) for
// draws that path cannot run.

#include "gpu/rhi/renderer.h"

#include "gpu/vulkan/vk_debug.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_draw_recomp.h"
#include "gpu/vulkan/vk_format.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_index_upload.h"
#include "gpu/vulkan/vk_perf.h"
#include "gpu/vulkan/vk_pipeline_cache.h"
#include "gpu/vulkan/vk_render_target.h"
#include "gpu/vulkan/vk_texture_cache.h"
#include "gpu/vulkan/vk_upload_ring.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gpu::rhi {
using namespace gpu::vk;

void Draw(Renderer& renderer, const DrawInfo& d_in) {
  if (!g_frame.recording)
    return;
  // DELTA_GPU_SWAPTEX01: bisect a suspected sampler-binding order mismatch by
  // exchanging the first two textures of every multi-texture draw.
  static const bool kSwapTex = std::getenv("DELTA_GPU_SWAPTEX01") != nullptr;
  DrawInfo swapped;
  if (kSwapTex && d_in.num_texs >= 2) {
    swapped = d_in;
    std::swap(swapped.texs[0], swapped.texs[1]);
  }
  const DrawInfo& d_sw = (kSwapTex && d_in.num_texs >= 2) ? swapped : d_in;
  // DELTA_GPU_MAXDRAW=<n> / DELTA_GPU_ONLYDRAW=<n>: build a frame up one draw
  // at a time, or isolate a single one, to see what each pass contributes.
  static const int kMaxDraw = [] {
    const char* e = std::getenv("DELTA_GPU_MAXDRAW");
    return e ? std::atoi(e) : -1;
  }();
  static const int kOnlyDraw = [] {
    const char* e = std::getenv("DELTA_GPU_ONLYDRAW");
    return e ? std::atoi(e) : -1;
  }();
  // DELTA_GPU_ONLYIC=<n>: render only draws with this index count. Draw indices
  // move between frames; an index count names one pass reliably.
  static const uint32_t kOnlyIc = [] {
    const char* e = std::getenv("DELTA_GPU_ONLYIC");
    return e ? static_cast<uint32_t>(std::strtoul(e, nullptr, 0)) : 0u;
  }();
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
  static const bool kNoDepth = std::getenv("DELTA_GPU_NODEPTH") != nullptr;
  static const bool kNoCull = std::getenv("DELTA_GPU_NOCULL") != nullptr;
  static const bool kNoMask = std::getenv("DELTA_GPU_NOMASK") != nullptr;
  // A pass that samples the depth buffer it also has bound is reading it as a
  // texture, which is legal while depth testing and writes are off. Detaching
  // the unused attachment lets the draw run instead of being declined as
  // self-sampling: Skyrim's grading pass does exactly this, and dropping it
  // left the display buffer showing ungraded content on those frames -- the
  // frame alternated between graded and raw as the pass came and went.
  bool detach_depth = false;
  if (d_sw.depth_base && !d_sw.depth_test_enable && !d_sw.depth_write_enable) {
    if (d_sw.tex_base == d_sw.depth_base)
      detach_depth = true;
    for (uint32_t i = 0; i < d_sw.num_texs && !detach_depth; i++)
      if (d_sw.texs[i].base == d_sw.depth_base)
        detach_depth = true;
  }
  DrawInfo dd;
  const bool patched = kNoDepth || kNoCull || kNoMask || detach_depth;
  if (patched) {
    dd = d_sw;
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
  const DrawInfo& d = patched ? dd : d_sw;
  if (d.index_count > g_frame.max_idx)
    g_frame.max_idx = d.index_count;
  ScopeNs draw_timer(&g_ns_draw);
  ScopeNs frame_draw_timer(&g_fr_draw);
  if (d.index_data && d.index_count) {
    const uint64_t index_bytes = static_cast<uint64_t>(d.index_count) *
                                 GuestIndexElementBytes(d.index_type);
    if (!FlushCsWritesRange(renderer, reinterpret_cast<uint64_t>(d.index_data),
                            index_bytes))
      return;
  }
  // Recompiled-shader path: run the game's actual VS/PS. Falls through to the
  // heuristic quad path when the draw can't be handled. On by default now that
  // it renders gameplay correctly; DELTA_GPU_RECOMP=0 forces the old heuristic
  // path.
  static const bool kRecompPath = [] {
    const char* e = std::getenv("DELTA_GPU_RECOMP");
    return !e || std::strcmp(e, "0") != 0;
  }();
  const bool recompiled = kRecompPath && d.recomp && DrawRecomp(renderer, d);
  if (!renderer.available())
    return;
  static const bool kDrawTraceAll =
      std::getenv("DELTA_GPU_DRAWTRACE") != nullptr;
  if (kDrawTraceAll) {
    static uint32_t traced = 0;
    if (traced++ < 100)
      std::fprintf(stderr,
                   "[dt] f%d rt=%#lx count=%u indexed=%u nv=%u mrt=%u mask=%#x "
                   "psmask=%#x prim=%u vp=[%.1f %.1f %.1f %.1f] depth=%#lx "
                   "handled=%d\n",
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
  if (!FlushCsWritesRange(renderer, reinterpret_cast<uint64_t>(d.vertex_data),
                          static_cast<uint64_t>(d.vertex_stride) *
                              (d.vertex_count ? d.vertex_count : 1)))
    return;
  // Indexed triangle list (the common GNM draw): the index buffer selects which
  // vertices form each triangle. Find how many vertices the indices reference
  // so we repack exactly that many (the V# num_records can be the whole shared
  // batch).
  bool indexed = d.index_data && d.index_count >= 3;
  uint32_t nv = d.vertex_count;
  if (indexed) {
    if (d.index_count > 1500000u)
      return;
    const uint32_t max_index =
        MaxGuestIndex(d.index_data, d.index_count, d.index_type);
    if (max_index >= 200000u)
      return;
    nv = max_index + 1;
  }
  if (nv < 3 || nv > 200000u)
    return;                                   // sane cap
  VkDeviceSize need = (VkDeviceSize)nv * 32;  // pos.xy + color.rgba + uv.xy
  if (g_ring.vb_offset + need > g_ring.vb_end)
    return;  // ring full this frame
  const VkDeviceSize index_bytes =
      indexed ? static_cast<VkDeviceSize>(d.index_count) *
                    UploadedIndexElementBytes(d.index_type)
              : 0;
  const VkDeviceSize index_align = d.index_type == 1 ? 4 : 2;
  const VkDeviceSize aligned_ioff =
      (g_ring.ib_offset + index_align - 1) & ~(index_align - 1);
  if (indexed && aligned_ioff + index_bytes > g_ring.ib_end)
    return;
  // Repack pos / color / uv interleaved into the vertex ring (stride 32).
  auto* base = static_cast<const uint8_t*>(d.vertex_data);
  auto* dst = reinterpret_cast<float*>(g_ring.vb_map + g_ring.vb_offset);
  for (uint32_t v = 0; v < nv; v++) {
    const uint8_t* vert = base + (size_t)v * d.vertex_stride;
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
  uint64_t tex_base = d.tex_base;
  if (tex_base && !d.tex_arrayed && !g_rts.count(tex_base)) {
    uint64_t r = ResolveSampledRT(tex_base, d.tex_w, d.tex_h);
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
  VkDescriptorSet tex_set = VK_NULL_HANDLE;
  if (d.tex_base && g_quad.tex_pipeline && !rt_as_tex && !d.tex_arrayed &&
      GuestTextureUploadSupported(d.tex_dfmt, d.tex_nfmt))
    tex_set = GetTexture(
        d.tex_base, d.tex_w, d.tex_h, d.tex_dfmt, d.tex_nfmt, d.tex_tiling,
        d.tex_pitch, d.tex_layers, d.tex_base_array, d.tex_view_layers,
        d.tex_mip_levels, d.tex_base_mip, d.tex_view_mips, d.tex_min_lod,
        d.tex_pow2_pad, d.tex_sampler, d.tex_sampler_valid, false,
        d.tex_force_lod_zero, d.tex_depth_compare, d.tex_swizzle);

  // Switch render target if this draw targets a different RT than the open
  // region (or the open region is multi-target/has a depth attachment: the
  // heuristic path renders to a single color attachment with no depth).
  const bool transition_source =
      rt_as_tex &&
      g_rts[tex_base].layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  if (g_region.cur_rt != d.rt_base || g_region.cur_mrt_count != 1 ||
      g_region.cur_depth != 0 || transition_source) {
    EndRegion();
    if (rt_as_tex) {  // make the sampled RT shader-readable before we render
      auto& src = g_rts[tex_base];
      if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        ImageBarrier(g_frame.cmd, src.image, src.layout,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     ColorImageAccess(src.layout), VK_ACCESS_SHADER_READ_BIT);
        src.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
    }
    VkFormat rt_format = ColorTargetFormat(d.mrt_info[0]);
    RTarget* rt = GetRT(d.rt_base, d.rt_w, d.rt_h, rt_format);
    if (!rt) {
      g_frame.draws++;
      return;
    }
    if (!BeginRegion(d.mrt_base, d.mrt_info, 1, d.rt_w, d.rt_h)) {
      g_frame.draws++;
      return;
    }
  }
  if (rt_as_tex &&
      g_rts[tex_base].layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    tex_set = g_rts[tex_base].set;

  g_frame.heuristic++;
  SetGuestViewport(d);
  VkDeviceSize off = g_ring.vb_offset;
  if (tex_set) {
    // Per-draw blend from the guest's CB_BLEND0_CONTROL, real vertex UVs.
    vkCmdBindPipeline(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      GetPipeline(true, d.blend_control, d.blend_enable,
                                  ColorTargetFormat(d.mrt_info[0])));
    float pc[17];
    std::memcpy(pc, d.mvp, 64);
    reinterpret_cast<uint32_t*>(pc)[16] =
        0u;  // clipUV: real per-vertex uv/colour
    vkCmdPushConstants(g_frame.cmd, g_quad.tex_layout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, 68, pc);
    vkCmdBindDescriptorSets(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            g_quad.tex_layout, 0, 1, &tex_set, 0, nullptr);
  } else {
    vkCmdBindPipeline(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      GetPipeline(false, d.blend_control, d.blend_enable,
                                  ColorTargetFormat(d.mrt_info[0])));
    vkCmdPushConstants(g_frame.cmd, g_quad.layout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, 64, d.mvp);
  }
  vkCmdBindVertexBuffers(g_frame.cmd, 0, 1, &g_ring.vb, &off);
  CmdInsertLabel(g_frame.cmd, "quad vs=%#llx ps=%#llx n=%u",
                 (unsigned long long)d.vs_addr, (unsigned long long)d.ps_addr,
                 indexed ? d.index_count : nv);
  if (indexed) {
    VkDeviceSize ioff = aligned_ioff;
    CopyGuestIndices(g_ring.ib_map + ioff, d.index_data, d.index_count,
                     d.index_type);
    const VkIndexType vk_index_type =
        d.index_type == 1 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
    vkCmdBindIndexBuffer(g_frame.cmd, g_ring.ib, ioff, vk_index_type);
    vkCmdDrawIndexed(g_frame.cmd, d.index_count,
                     d.instance_count ? d.instance_count : 1, 0, 0, 0);
    g_ring.ib_offset = ioff + index_bytes;
  } else {
    vkCmdDraw(g_frame.cmd, nv, d.instance_count ? d.instance_count : 1, 0, 0);
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
}

}  // namespace gpu::rhi
