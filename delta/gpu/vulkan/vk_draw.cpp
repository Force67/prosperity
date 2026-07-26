/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// The renderer's draw entry point: applies the diagnostic draw knobs, tries the
// recompiled-shader path, and falls back to the heuristic quad path (repack the
// guest vertices into pos/colour/uv and draw them with a fixed shader pair) for
// draws that path cannot run.

#include "rhi/renderer.h"

#include "vulkan/vk_device.h"
#include "vulkan/vk_draw_recomp.h"
#include "vulkan/vk_format.h"
#include "vulkan/vk_frame.h"
#include "vulkan/vk_perf.h"
#include "vulkan/vk_pipeline_cache.h"
#include "vulkan/vk_render_target.h"
#include "vulkan/vk_texture_cache.h"
#include "vulkan/vk_upload_ring.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gpu::rhi {
using namespace gpu::vk;

void draw(const DrawInfo &d_in) {
  if (!g_frame.recording) return;
  // DELTA_GPU_SWAPTEX01: bisect a suspected sampler-binding order mismatch by
  // exchanging the first two textures of every multi-texture draw.
  static const bool swapTex = std::getenv("DELTA_GPU_SWAPTEX01") != nullptr;
  DrawInfo swapped;
  if (swapTex && d_in.nTexs >= 2) {
    swapped = d_in;
    std::swap(swapped.texs[0], swapped.texs[1]);
  }
  const DrawInfo &d_sw = (swapTex && d_in.nTexs >= 2) ? swapped : d_in;
  // DELTA_GPU_MAXDRAW=<n> / DELTA_GPU_ONLYDRAW=<n>: build a frame up one draw at
  // a time, or isolate a single one, to see what each pass contributes.
  static const int maxDraw = [] {
    const char *e = std::getenv("DELTA_GPU_MAXDRAW"); return e ? std::atoi(e) : -1;
  }();
  static const int onlyDraw = [] {
    const char *e = std::getenv("DELTA_GPU_ONLYDRAW");
    return e ? std::atoi(e) : -1;
  }();
  if (maxDraw >= 0 && (int)g_frame.draws >= maxDraw)
    return;
  if (onlyDraw >= 0 && (int)g_frame.draws != onlyDraw) {
    g_frame.draws++;
    return;
  }
  // Diagnostic kill-switches for bisecting "renders nothing" chains:
  // DELTA_GPU_NODEPTH disables depth test/write, DELTA_GPU_NOCULL disables
  // face culling, DELTA_GPU_NOMASK forces full color write masks.
  static const bool noDepth = std::getenv("DELTA_GPU_NODEPTH") != nullptr;
  static const bool noCull = std::getenv("DELTA_GPU_NOCULL") != nullptr;
  static const bool noMask = std::getenv("DELTA_GPU_NOMASK") != nullptr;
  // A pass that samples the depth buffer it also has bound is reading it as a
  // texture, which is legal while depth testing and writes are off. Detaching
  // the unused attachment lets the draw run instead of being declined as
  // self-sampling: Skyrim's grading pass does exactly this, and dropping it
  // left the display buffer showing ungraded content on those frames -- the
  // frame alternated between graded and raw as the pass came and went.
  bool detachDepth = false;
  if (d_sw.depthBase && !d_sw.depthTestEnable && !d_sw.depthWriteEnable) {
    if (d_sw.texBase == d_sw.depthBase)
      detachDepth = true;
    for (uint32_t i = 0; i < d_sw.nTexs && !detachDepth; i++)
      if (d_sw.texs[i].base == d_sw.depthBase)
        detachDepth = true;
  }
  DrawInfo dd;
  const bool patched = noDepth || noCull || noMask || detachDepth;
  if (patched) {
    dd = d_sw;
    if (noDepth) { dd.depthTestEnable = false; dd.depthWriteEnable = false; }
    if (noCull) dd.cullMode = 0;
    if (noMask) { dd.targetMask = 0xFFFFFFFFu; dd.colorControl = 0x10; }
    if (detachDepth) { dd.depthBase = 0; dd.depthTestEnable = false; }
  }
  const DrawInfo &d = patched ? dd : d_sw;
  if (d.indexCount > g_frame.maxIdx) g_frame.maxIdx = d.indexCount;
  ScopeNs _t(&g_nsDraw);
  ScopeNs _tf(&g_frDraw);
  // Recompiled-shader path: run the game's actual VS/PS. Falls through to the
  // heuristic quad path when the draw can't be handled. On by default now that it
  // renders gameplay correctly; DELTA_GPU_RECOMP=0 forces the old heuristic path.
  static const bool recompPath = [] {
    const char *e = std::getenv("DELTA_GPU_RECOMP");
    return !e || std::strcmp(e, "0") != 0;
  }();
  const bool recompiled = recompPath && d.recomp && drawRecomp(d);
  static const bool drawTraceAll = std::getenv("DELTA_GPU_DRAWTRACE") != nullptr;
  if (drawTraceAll) {
    static uint32_t traced = 0;
    if (traced++ < 100)
      std::fprintf(stderr,
                   "[dt] f%d rt=%#lx count=%u indexed=%u nv=%u mrt=%u mask=%#x "
                   "psmask=%#x prim=%u vp=[%.1f %.1f %.1f %.1f] depth=%#lx "
                   "handled=%d\n",
                   g_frame.num, (unsigned long)d.rtBase, d.vertexCount,
                   d.indexCount, d.nvattrs, d.mrtCount, d.targetMask,
                   d.recomp ? d.recomp->ps_mrt_mask : 0, d.primType,
                   d.viewportXScale, d.viewportXOffset, d.viewportYScale,
                   d.viewportYOffset, (unsigned long)d.depthBase, recompiled);
  }
  if (recompiled) return;
  if (!d.vertexData || !d.vertexStride)
    return;
  flushCsWritesRange(reinterpret_cast<uint64_t>(d.vertexData),
                     static_cast<uint64_t>(d.vertexStride) *
                         (d.vertexCount ? d.vertexCount : 1));
  // Indexed triangle list (the common GNM draw): the index buffer selects which
  // vertices form each triangle. Find how many vertices the indices reference so
  // we repack exactly that many (the V# num_records can be the whole shared batch).
  const uint16_t *idx16 = nullptr;
  const uint32_t *idx32 = nullptr;
  bool indexed = d.indexData && d.indexCount >= 3;
  uint32_t nv = d.vertexCount;
  if (indexed) {
    if (d.indexCount > 1500000u) return;
    uint32_t maxIdx = 0;
    if (d.indexType == 1) {
      idx32 = static_cast<const uint32_t *>(d.indexData);
      for (uint32_t i = 0; i < d.indexCount; i++) maxIdx = idx32[i] > maxIdx ? idx32[i] : maxIdx;
    } else {
      idx16 = static_cast<const uint16_t *>(d.indexData);
      for (uint32_t i = 0; i < d.indexCount; i++) maxIdx = idx16[i] > maxIdx ? idx16[i] : maxIdx;
    }
    nv = maxIdx + 1;
  }
  if (nv < 3 || nv > 200000u) return;  // sane cap
  VkDeviceSize need = (VkDeviceSize)nv * 32;  // pos.xy + color.rgba + uv.xy
  if (g_ring.vbOffset + need > g_ring.vbEnd)
    return;  // ring full this frame
  if (indexed && g_ring.ibOffset + (VkDeviceSize)d.indexCount * 4 > g_ring.ibEnd)
    return;
  // Repack pos / color / uv interleaved into the vertex ring (stride 32).
  auto *base = static_cast<const uint8_t *>(d.vertexData);
  auto *dst = reinterpret_cast<float *>(g_ring.vbMap + g_ring.vbOffset);
  for (uint32_t v = 0; v < nv; v++) {
    const uint8_t *vert = base + (size_t)v * d.vertexStride;
    auto *p = reinterpret_cast<const float *>(vert + d.posOffset);
    dst[v * 8 + 0] = p[0];
    dst[v * 8 + 1] = p[1];
    if (d.colorOffset != 0xFFFFFFFFu) {
      auto *c = reinterpret_cast<const float *>(vert + d.colorOffset);
      dst[v * 8 + 2] = c[0];
      dst[v * 8 + 3] = c[1];
      dst[v * 8 + 4] = c[2];
      dst[v * 8 + 5] = 1.0f;
    } else {
      dst[v * 8 + 2] = dst[v * 8 + 3] = dst[v * 8 + 4] = dst[v * 8 + 5] = 1.0f;
    }
    if (d.uvData && d.uvStride) {
      auto *u = reinterpret_cast<const float *>(vert + d.uvOffset);
      dst[v * 8 + 6] = u[0];
      dst[v * 8 + 7] = u[1];
    } else {
      dst[v * 8 + 6] = dst[v * 8 + 7] = 0.0f;
    }
  }

  // Resolve the sampled texture address to a render target via overlap (the
  // resource-model page-table lookup): an exact RT base, or an address whose
  // footprint overlaps a live RT, binds that RT's image instead of stale guest
  // memory. This replaces the old per-symptom FRESHRT/CYCLEREDIR/ROOMALPHA address
  // heuristics with one principled, game-agnostic lookup.
  uint64_t texBase = d.texBase;
  if (texBase && !d.texArrayed && !g_rts.count(texBase)) {
    uint64_t r = resolveSampledRT(texBase, d.texW, d.texH);
    if (r) texBase = r;
  }
  // Is this a render-to-texture sample (the draw samples another render target)?
  bool rtAsTex = !d.texArrayed && texBase && texBase != d.rtBase && g_rts.count(texBase);
  bool roomSrc = rtAsTex && g_rts[texBase].w >= 700 && g_rts[texBase].w <= 900;
  if (roomSrc) g_frame.hadRoom = true;

  // Upload guest texture (independent of the render region) if not RT-as-texture.
  VkDescriptorSet texSet = VK_NULL_HANDLE;
  if (d.texBase && g_quad.texPipeline && !rtAsTex && !d.texArrayed &&
      guestTextureUploadSupported(d.texDfmt, d.texNfmt))
    texSet = getTexture(d.texBase, d.texW, d.texH, d.texDfmt, d.texNfmt,
                         d.texTiling, d.texPitch,
                         d.texLayers, d.texBaseArray, d.texViewLayers,
                         d.texMipLevels, d.texBaseMip, d.texViewMips,
                          d.texMinLod, d.texPow2Pad, d.texSampler,
                          d.texSamplerValid, false, d.texForceLodZero,
                          d.texDepthCompare, d.texSwizzle);

  // Switch render target if this draw targets a different RT than the open region (or
  // the open region is multi-target/has a depth attachment: the heuristic path renders
  // to a single color attachment with no depth).
  if (g_region.curRt != d.rtBase || g_region.curMrtCount != 1 || g_region.curDepth != 0) {
    endRegion();
    if (rtAsTex) {  // make the sampled RT shader-readable before we render
      auto &src = g_rts[texBase];
      if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        imageBarrier(g_frame.cmd, src.image, src.layout,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        src.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
    }
    VkFormat rtFormat = colorTargetFormat(d.mrtInfo[0]);
    RTarget *rt = getRT(d.rtBase, d.rtW, d.rtH, rtFormat);
    if (!rt) { g_frame.draws++; return; }
    beginRegion(d.mrtBase, d.mrtInfo, 1, d.rtW, d.rtH);  // heuristic path is single-RT
  }
  if (rtAsTex && g_rts[texBase].layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    texSet = g_rts[texBase].set;

  g_frame.heuristic++;
  setGuestViewport(d);
  VkDeviceSize off = g_ring.vbOffset;
  if (texSet) {
    // Per-draw blend from the guest's CB_BLEND0_CONTROL, real vertex UVs.
    vkCmdBindPipeline(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      getPipeline(true, d.blendControl, d.blendEnable,
                                  colorTargetFormat(d.mrtInfo[0])));
    float pc[17];
    std::memcpy(pc, d.mvp, 64);
    reinterpret_cast<uint32_t *>(pc)[16] = 0u;  // clipUV: real per-vertex uv/colour
    vkCmdPushConstants(g_frame.cmd, g_quad.texLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 68, pc);
    vkCmdBindDescriptorSets(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_quad.texLayout, 0,
                            1, &texSet, 0, nullptr);
  } else {
    vkCmdBindPipeline(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      getPipeline(false, d.blendControl, d.blendEnable,
                                  colorTargetFormat(d.mrtInfo[0])));
    vkCmdPushConstants(g_frame.cmd, g_quad.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 64, d.mvp);
  }
  vkCmdBindVertexBuffers(g_frame.cmd, 0, 1, &g_ring.vb, &off);
  if (indexed) {
    // Widen the guest indices (16- or 32-bit) into the 32-bit index ring and draw.
    VkDeviceSize ioff = g_ring.ibOffset;
    auto *idst = reinterpret_cast<uint32_t *>(g_ring.ibMap + ioff);
    if (idx32)
      std::memcpy(idst, idx32, (size_t)d.indexCount * 4);
    else
      for (uint32_t i = 0; i < d.indexCount; i++) idst[i] = idx16[i];
    vkCmdBindIndexBuffer(g_frame.cmd, g_ring.ib, ioff, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(g_frame.cmd, d.indexCount, d.instanceCount ? d.instanceCount : 1, 0, 0, 0);
    g_ring.ibOffset += (VkDeviceSize)d.indexCount * 4;
  } else {
    vkCmdDraw(g_frame.cmd, nv, d.instanceCount ? d.instanceCount : 1, 0, 0);
  }
  g_ring.vbOffset += need;
  g_frame.draws++;
  if (g_region.curRt) {
    auto &rt = g_rts[g_region.curRt];
    if (++rt.draws > g_region.busiestRtDraws) { g_region.busiestRtDraws = rt.draws; g_region.busiestRt = g_region.curRt; }
  }
}

}  // namespace gpu::rhi
