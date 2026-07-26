/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "vulkan/vk_draw_recomp.h"

#include "gcn/gcn_translate.h"
#include "guest_memory.h"
#include "rhi/renderer.h"
#include "vulkan/vk_device.h"
#include "vulkan/vk_format.h"
#include "vulkan/vk_frame.h"
#include "vulkan/vk_pipeline_cache.h"
#include "vulkan/vk_render_target.h"
#include "vulkan/vk_texture_cache.h"
#include "vulkan/vk_upload_ring.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gpu::vk {

using rhi::DrawInfo;
using rhi::flushCsWritesRange;

namespace {

// Why a draw declined the recompiled path (falls back to the heuristic). Tallied
// per reason so the remaining heuristic draws can be driven to zero; dumped with the
// periodic frame log.
enum DeclineReason { DR_NORECOMP, DR_NOTEXPIPE, DR_SELF, DR_RING,
                     DR_GUESTTEX, DR_MIDREGION, DR_NOPIPE, DR_MAX };
static const char *kDeclineName[DR_MAX] = {
    "norecomp", "notexpipe", "self", "ring", "guesttex", "midregion", "nopipe"};
uint32_t g_decline[DR_MAX] = {0};
inline bool decline(DeclineReason r) { g_decline[r]++; return false; }

}  // namespace

void reportDeclines() {
  std::fprintf(stderr, "[gpuvk]   decline:");
  for (int i = 0; i < DR_MAX; i++)
    if (g_decline[i])
      std::fprintf(stderr, " %s=%u", kDeclineName[i], g_decline[i]);
  std::fprintf(stderr, "\n");
}

// Issue a draw running the game's recompiled VS/PS. Returns false if the draw
// can't be handled (the caller falls back to the heuristic path).
bool drawRecomp(const DrawInfo &d) {
  bool indexed = d.indexData && d.indexCount >= 3;
  uint32_t drawCount = indexed ? d.indexCount : d.vertexCount;
  static const bool drawTrace = std::getenv("DELTA_GPU_DRAWTRACE") != nullptr;
  if (drawTrace && drawCount >= 300) {
    static int n = 0;
    if (n++ < 40)
      std::fprintf(stderr, "[dt] enter recomp count=%u rt=%#lx tex=%#lx nTexs=%u nvattrs=%u ok=%d\n",
                   drawCount, (unsigned long)d.rtBase, (unsigned long)d.texBase,
                   d.nTexs, d.nvattrs, d.recomp ? d.recomp->ok : 0);
  }
  if (!d.recomp || !d.recomp->ok || drawCount < 3)
    return decline(DR_NORECOMP);
  const bool hasStorageImage = std::any_of(
      d.recomp->ps_texs.begin(), d.recomp->ps_texs.end(),
      [](const gcn::ShaderTex &tex) { return tex.storage; });
  if (!d.mrtCount && !d.depthBase && !hasStorageImage) {
    g_frame.draws++;
    return true;
  }
  if (!g_quad.texPipeline) return decline(DR_NOTEXPIPE);  // need the descriptor infra (dsPool/dsLayout)
  // Resolve the sampled texture address to an overlapping live RT (resource-model
  // page-table lookup), so an RT-as-texture sample binds the live image for
  // cycled/aliased RT addresses instead of stale guest memory. Additive: an exact
  // RT base resolves to itself.
  uint64_t texBase = d.texBase;
  if (texBase && !d.texArrayed && !g_rts.count(texBase) &&
      !g_depths.count(texBase)) {
    bool depthFormat = d.texDfmt == 4 && d.texNfmt == 7;
    uint64_t r = depthFormat ? resolveSampledDepth(texBase, d.texW, d.texH) : 0;
    if (!r) r = resolveSampledRT(texBase, d.texW, d.texH);
    if (!r && !depthFormat) r = resolveSampledDepth(texBase, d.texW, d.texH);
    if (r) texBase = r;
  }
  bool colorAsTex = !d.texArrayed && texBase && texBase != d.rtBase &&
                    g_rts.count(texBase);
  bool feedbackAsTex = !d.texArrayed && texBase && texBase == d.rtBase &&
                       g_rts.count(texBase) && g_rts[texBase].everRendered;
  bool depthAsTex = !d.texArrayed && texBase && texBase != d.depthBase &&
                    g_depths.count(texBase);
  bool rtAsTex = colorAsTex || feedbackAsTex || depthAsTex;
  if (colorAsTex && g_rts[texBase].w >= 700 && g_rts[texBase].w <= 900)
    g_frame.hadRoom = true;
  if (texBase && (texBase == d.depthBase ||
                  (texBase == d.rtBase && !feedbackAsTex))) {
    // DELTA_GPU_SELFTRACE: which pass reads the target it is drawing into. We
    // drop those, and dropping one every frame leaves whatever it was meant to
    // produce stale.
    static const bool selfTrace = std::getenv("DELTA_GPU_SELFTRACE") != nullptr;
    static int n = 0;
    if (selfTrace && n++ < 20)
      std::fprintf(stderr,
                   "[self] draw#%u ps=%#lx rt=%#lx tex=%#lx depth=%#lx dtest=%d "
                   "dwrite=%d ntex=%u\n",
                   g_frame.draws, (unsigned long)d.psAddr, (unsigned long)d.rtBase,
                   (unsigned long)texBase, (unsigned long)d.depthBase,
                   (int)d.depthTestEnable, (int)d.depthWriteEnable, d.nTexs);
    return decline(DR_SELF);
  }
  // Indexed draws derive the copied vertex range from their indices. DRAW_INDEX_AUTO
  // consumes the packet's sequential vertex count directly.
  const uint16_t *i16 = nullptr; const uint32_t *i32 = nullptr;
  uint32_t nv = d.vertexCount;
  if (indexed) {
    uint32_t maxIdx = 0;
    if (d.indexType == 1) { i32 = (const uint32_t *)d.indexData;
      for (uint32_t i = 0; i < d.indexCount; i++) maxIdx = i32[i] > maxIdx ? i32[i] : maxIdx; }
    else { i16 = (const uint16_t *)d.indexData;
      for (uint32_t i = 0; i < d.indexCount; i++) maxIdx = i16[i] > maxIdx ? i16[i] : maxIdx; }
    nv = maxIdx + 1;
  }
  if (nv > 200000u || (d.nvattrs && (!d.vertexData || !d.vertexStride)))
    return decline(DR_NORECOMP);

  // A fullscreen, untextured, near-black REPLACE draw is the game CLEARING an RT.
  // Don't render it (that wipes the RT immediately); record a LAZY clear instead --
  // realised as loadOp=CLEAR only when content actually redraws this RT this frame
  // (see beginRegion). Baked-once content (the room floor) whose clear and redraw land
  // on different frames then survives. A COLOURED fullscreen REPLACE is real content
  // (e.g. the per-frame minimap redraw) and must NOT be treated as a clear.
  static const bool noWipe = [] {
    const char *e = std::getenv("DELTA_GPU_NOWIPE");
    return !e || std::strcmp(e, "0") != 0;
  }();
  static const bool lazyClear2 = [] { const char *e = std::getenv("DELTA_GPU_LAZYCLEAR");
    return !e || std::strcmp(e, "0") != 0; }();
  // The extent test below reads the position attribute as float32s. A title
  // whose positions are not floats (Skyrim's UI uses 16_16_SScaled) would have
  // its geometry read as garbage, land a bogus fullscreen extent, and get
  // swallowed as a "clear" -- which is exactly what turned its whole frame
  // black. Only consider draws whose position really is float.
  bool floatPos = false;
  for (uint32_t a = 0; a < d.nvattrs; a++) {
    if (d.vattrs[a].location != 0) continue;
    const uint32_t df = d.vattrs[a].dfmt;
    floatPos = d.vattrs[a].nfmt == 7 &&
               (df == 4 || df == 11 || df == 13 || df == 14);
    break;
  }
  if (noWipe && floatPos && d.vertexData && d.nvattrs &&
      d.recomp->ps_texs.empty() && nv <= 8) {
    uint32_t cdst = (d.blendControl >> 8) & 0x1F, csrc = d.blendControl & 0x1F;
    bool replace = d.blendEnable && csrc == 1 && cdst == 0;
    if (replace) {
      const auto *vb = static_cast<const uint8_t *>(d.vertexData);
      bool nearBlack = true;
      float clearColor[4] = {0, 0, 0, 0};
      for (uint32_t a = 0; a < d.nvattrs; a++) {
        if (d.vattrs[a].num_comps == 4 && d.vattrs[a].offset != 0) {
          // Colour may live in its own binding; read from that binding's base.
          const auto *cbuf = static_cast<const uint8_t *>(
              d.vbufs[d.vattrs[a].binding].data);
          const uint8_t *cb0 = cbuf + d.vattrs[a].offset;  // vertex 0's colour
          if (d.vattrs[a].dfmt == 10) {
            for (int i = 0; i < 4; i++) clearColor[i] = cb0[i] / 255.f;
          } else {
            const float *c = reinterpret_cast<const float *>(cb0);
            for (int i = 0; i < 4; i++) clearColor[i] = c[i];
          }
          if (clearColor[0] > 0.02f || clearColor[1] > 0.02f || clearColor[2] > 0.02f)
            nearBlack = false;
          break;
        }
      }
      const float *m = d.mvp;
      float nx0=1e9f,ny0=1e9f,nx1=-1e9f,ny1=-1e9f;
      for (uint32_t v = 0; v < nv; v++) {
        const float *p = reinterpret_cast<const float *>(vb + (size_t)v * d.vertexStride);
        float cw = m[3]*p[0]+m[7]*p[1]+m[15]; if (cw==0) cw=1;
        float nx=(m[0]*p[0]+m[4]*p[1]+m[12])/cw, ny=(m[1]*p[0]+m[5]*p[1]+m[13])/cw;
        nx0=nx<nx0?nx:nx0; nx1=nx>nx1?nx:nx1; ny0=ny<ny0?ny:ny0; ny1=ny>ny1?ny:ny1;
      }
      bool fullscreenBlack = nearBlack && (nx1-nx0) >= 1.8f && (ny1-ny0) >= 1.8f;
      if (fullscreenBlack && lazyClear2) {
        RTarget *rt = d.rtBase
                          ? getRT(d.rtBase, d.rtW, d.rtH,
                                  colorTargetFormat(d.mrtInfo[0]))
                          : nullptr;
        if (rt) {
          rt->clearPending = true;
          std::memcpy(rt->clearValue.float32, clearColor, sizeof(clearColor));
        }
        // This draw also performs the guest's reverse-Z clear (depth write enabled,
        // ZFUNC=ALWAYS). Suppressing its color write must not discard that depth
        // effect, or stale depth rejects the following layer composites.
        if (d.depthBase && d.depthWriteEnable && d.depthFunc == 7) {
          DepthTarget *dt = getDepthRT(d.depthBase, d.rtW, d.rtH);
          if (dt) {
            dt->clearPending = true;
            dt->clearValue = d.depthClear;
          }
        }
        g_frame.draws++;
        return true;  // suppressed; the clear is applied lazily on the next redraw
      }
      // Legacy single-frame behaviour (DELTA_GPU_LAZYCLEAR=0): only suppress if the
      // RT already holds content this frame.
      auto rit = g_rts.find(d.rtBase);
      if (fullscreenBlack && rit != g_rts.end() && rit->second.draws > 0 &&
          rit->second.lastFrame == g_frame.num) {
        g_frame.draws++;
        return true;
      }
    }
  }

  // Lay out one contiguous ring range per vertex binding. Binding 0 sits at the
  // ring offset (single-stream draws are byte-identical to before); additional
  // bindings are 16-byte aligned so no attribute straddles a coarse boundary.
  const uint32_t nbind = d.nvattrs ? std::min(d.nvbufs, 8u) : 0;
  VkDeviceSize bindOff[8] = {}, bindSize[8] = {};
  VkDeviceSize vneed = 0;
  for (uint32_t j = 0; j < nbind; j++) {
    if (j) vneed = (vneed + 15) & ~VkDeviceSize(15);
    bindOff[j] = vneed;
    if (d.vbufs[j].stride) {
      bindSize[j] = (VkDeviceSize)nv * d.vbufs[j].stride;
    } else {
      // Stride-0 (constant) binding: upload a single record large enough to cover
      // every attribute that reads it; the pipeline binds it with stride 0 so all
      // vertices fetch this one record.
      uint32_t rec = 0;
      for (uint32_t a = 0; a < d.nvattrs; a++)
        if (d.vattrs[a].binding == j)
          rec = std::max(rec, d.vattrs[a].offset + vfmtBytes(d.vattrs[a].dfmt));
      bindSize[j] = rec;
    }
    vneed += bindSize[j];
  }
  if (g_ring.vbOffset + vneed > g_ring.vbEnd) return decline(DR_RING);
  if (indexed && g_ring.ibOffset + (VkDeviceSize)d.indexCount * 4 > g_ring.ibEnd)
    return decline(DR_RING);

  RecompPipe *rp = getRecompPipe(d);
  if (!rp) return decline(DR_NOPIPE);
  // Guest-texture source resolved up front; an RT-as-texture source is resolved after
  // the region switch (transitioning it to readable must happen outside a region).
  VkDescriptorSet texSet = VK_NULL_HANDLE;
  if (rp->textured && !rp->multiTex && !rtAsTex) {
    if (guestTextureUploadSupported(d.texDfmt, d.texNfmt))
      texSet = getTexture(d.texBase, d.texW, d.texH, d.texDfmt, d.texNfmt,
                           d.texTiling,
                          d.texPitch, d.texLayers, d.texBaseArray,
                          d.texViewLayers, d.texMipLevels, d.texBaseMip,
                           d.texViewMips, d.texMinLod, d.texPow2Pad, d.texSampler,
                           d.texSamplerValid, d.texArrayed, d.texForceLodZero,
                           d.texDepthCompare, d.texSwizzle);
    if (!texSet) texSet = d.texArrayed ? g_tex.whiteArraySet : g_tex.whiteSet;
    if (!texSet) return decline(DR_GUESTTEX);
  }

  uint64_t multiColor[kMaxTex] = {};
  uint64_t multiDepth[kMaxTex] = {};
  uint64_t multiFeedback[kMaxTex] = {};
  uint64_t multiStorage[kMaxTex] = {};
  VkImageView multiViews[kMaxTex] = {};
  VkImageLayout multiLayouts[kMaxTex];
  bool multiTransitionSource = false;
  uint32_t multiN = std::min(d.nTexs, kMaxTex);
  if (rp->multiTex) {
    auto isBoundTarget = [&](uint64_t base) {
      uint32_t count = std::min(d.mrtCount, 8u);
      for (uint32_t m = 0; m < count; m++)
        if (d.mrtBase[m] == base) return true;
      return false;
    };
    for (uint32_t i = 0; i < kMaxTex; i++)
      multiLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    for (uint32_t i = 0; i < multiN; i++) {
      const auto &t = d.texs[i];
      uint64_t base = t.base;
      if (t.storage) {
        if (base && !g_rts.count(base)) {
          uint64_t resolved = resolveSampledRT(base, t.w, t.h);
          if (resolved) base = resolved;
        }
        if (base && !g_rts.count(base)) {
          const VkFormat format = guestTextureFormat(t.dfmt, t.nfmt);
          if (format != VK_FORMAT_UNDEFINED) getRT(base, t.w, t.h, format);
        }
        if (base && g_rts.count(base)) {
          multiStorage[i] = base;
          multiTransitionSource |=
              g_rts[base].layout != VK_IMAGE_LAYOUT_GENERAL;
        }
        continue;
      }
      if (base && !t.arrayed && !g_rts.count(base) && !g_depths.count(base)) {
        bool depthFormat = t.dfmt == 4 && t.nfmt == 7;
        uint64_t resolved = depthFormat ? resolveSampledDepth(base, t.w, t.h) : 0;
        if (!resolved) resolved = resolveSampledRT(base, t.w, t.h);
        if (!resolved && !depthFormat)
          resolved = resolveSampledDepth(base, t.w, t.h);
        if (resolved) base = resolved;
      }
      if (base && !t.arrayed && isBoundTarget(base) && g_rts.count(base) &&
          g_rts[base].everRendered) {
        multiFeedback[i] = base;
        multiTransitionSource = true;
      } else if (base && !t.arrayed && g_rts.count(base) &&
                 g_rts[base].everRendered) {
        multiColor[i] = base;
        multiTransitionSource |=
            g_rts[base].layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      } else if (base && !t.arrayed && base != d.depthBase && g_depths.count(base)) {
        multiDepth[i] = base;
        multiTransitionSource |=
            g_depths[base].layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
      } else {
        multiViews[i] = texViewFor(t);
      }
    }
  }

  // DELTA_GPU_TEXBIND=<frame>: how each sampler of each draw resolved -- to a
  // live render target, a depth target, guest memory, or the 1x1 white default.
  // A post-processing chain that samples its own previous target reads zero the
  // moment one of those lands on guest memory.
  {
    static const int texBindFrame = [] {
      const char *e = std::getenv("DELTA_GPU_TEXBIND"); return e ? std::atoi(e) : -1;
    }();
    if (texBindFrame >= 0 && (int)g_frame.num == texBindFrame)
      std::fprintf(stderr, "[blend] draw#%u rt=%#lx blend=%u ctl=%#x mask=%#x mrt=%u\n",
                   g_frame.draws, (unsigned long)d.rtBase, d.blendEnable,
                   d.blendControl, d.targetMask, d.mrtCount);
    if (texBindFrame >= 0 && (int)g_frame.num == texBindFrame && !rp->multiTex)
      std::fprintf(stderr,
                   "[texbind] draw#%u rt=%#lx %ux%u LEGACY tex=%#lx %ux%u "
                    "rtAsTex=%u color=%u feedback=%u depth=%u set=%u\n",
                   g_frame.draws, (unsigned long)d.rtBase, d.rtW, d.rtH,
                   (unsigned long)d.texBase, d.texW, d.texH, (unsigned)rtAsTex,
                   (unsigned)colorAsTex, (unsigned)feedbackAsTex,
                    (unsigned)depthAsTex, (unsigned)(texSet != VK_NULL_HANDLE));
    if (texBindFrame >= 0 && (int)g_frame.num == texBindFrame && multiN &&
        rp->multiTex) {
      std::fprintf(stderr, "[texbind] draw#%u rt=%#lx %ux%u ntex=%u:",
                   g_frame.draws, (unsigned long)d.rtBase, d.rtW, d.rtH, multiN);
      for (uint32_t i = 0; i < multiN; i++) {
        const char *how = multiColor[i]   ? "RT"
                          : multiFeedback[i] ? "feedback"
                          : multiDepth[i]    ? "depth"
                          : multiStorage[i]  ? "storage"
                          : multiViews[i]    ? "guest"
                                             : "WHITE";
        std::fprintf(stderr, " [%u]%s@%#lx %ux%u layers=%u mips=%u fmt=%u/%u rt?=%u er=%u",
                     i, how, (unsigned long)d.texs[i].base, d.texs[i].w,
                     d.texs[i].h, d.texs[i].layers, d.texs[i].mip_levels,
                     d.texs[i].dfmt, d.texs[i].nfmt,
                     (unsigned)g_rts.count(d.texs[i].base),
                     (unsigned)(g_rts.count(d.texs[i].base)
                                    ? g_rts[d.texs[i].base].everRendered
                                    : 0));
      }
      std::fprintf(stderr, "\n");
    }
  }

  // Copy each vertex binding's source range into the ring and, for indexed
  // draws, the indices.
  VkDeviceSize voff = g_ring.vbOffset, ioff = g_ring.ibOffset;
  for (uint32_t j = 0; j < nbind; j++) {
    if (!bindSize[j]) continue;
    flushCsWritesRange(reinterpret_cast<uint64_t>(d.vbufs[j].data), bindSize[j]);
    std::memcpy(g_ring.vbMap + voff + bindOff[j], d.vbufs[j].data, (size_t)bindSize[j]);
  }
  if (indexed) {
    auto *idst = reinterpret_cast<uint32_t *>(g_ring.ibMap + ioff);
    if (i32) std::memcpy(idst, i32, (size_t)d.indexCount * 4);
    else for (uint32_t i = 0; i < d.indexCount; i++) idst[i] = i16[i];
  }

  // Switch render target. Re-begin when the primary target or the MRT count changes
  // (the open region's attachment count must match the pipeline's), or when a new
  // RT-as-texture source still needs a read transition. Barriers cannot be recorded
  // inside dynamic rendering; consecutive layer composites often keep the same target
  // while switching sources, so that source change must also close/reopen the region.
  uint32_t mrtN = std::min(d.mrtCount, 8u);
  bool transitionSource = rp->multiTex ? multiTransitionSource :
      feedbackAsTex ||
          (colorAsTex && g_rts[texBase].layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ||
          (depthAsTex && g_depths[texBase].layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
  bool pendingDepthClear = d.depthBase && g_depths.count(d.depthBase) &&
                           g_depths[d.depthBase].clearPending;
  bool restartRegion = g_region.curRt != d.rtBase || g_region.curMrtCount != mrtN ||
                       g_region.curDepth != d.depthBase || transitionSource ||
                       pendingDepthClear;
  if (restartRegion) {
    endRegion();
    if (!rp->multiTex && colorAsTex && transitionSource) {
      auto &src = g_rts[texBase];
      if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        imageBarrier(g_frame.cmd, src.image, src.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        src.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
    }
    if (!rp->multiTex && depthAsTex && transitionSource) {
      auto &src = g_depths[texBase];
      if (src.layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL) {
        depthBarrier(g_frame.cmd, src.image, src.layout,
                     VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_ACCESS_SHADER_READ_BIT);
        src.layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
      }
    }
    if (!rp->multiTex && feedbackAsTex) {
      texSet = snapshotRT(g_rts[texBase]);
      if (!texSet) return decline(DR_MIDREGION);
    }
    if (rp->multiTex) {
      for (uint32_t i = 0; i < multiN; i++) {
        if (multiStorage[i]) {
          auto &dst = g_rts[multiStorage[i]];
          if (dst.layout != VK_IMAGE_LAYOUT_GENERAL) {
            VkAccessFlags srcAccess =
                dst.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    ? VK_ACCESS_SHADER_READ_BIT
                : dst.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                    ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                : dst.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                    ? VK_ACCESS_TRANSFER_READ_BIT
                    : 0;
            imageBarrier(g_frame.cmd, dst.image, dst.layout, VK_IMAGE_LAYOUT_GENERAL,
                         srcAccess, VK_ACCESS_SHADER_WRITE_BIT);
            dst.layout = VK_IMAGE_LAYOUT_GENERAL;
          }
        } else if (multiColor[i]) {
          auto &src = g_rts[multiColor[i]];
          if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            VkAccessFlags srcAccess =
                src.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                    ? VK_ACCESS_TRANSFER_READ_BIT
                    : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            imageBarrier(g_frame.cmd, src.image, src.layout,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         srcAccess, VK_ACCESS_SHADER_READ_BIT);
            src.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          }
        } else if (multiDepth[i]) {
          auto &src = g_depths[multiDepth[i]];
          if (src.layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL) {
            depthBarrier(g_frame.cmd, src.image, src.layout,
                         VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         VK_ACCESS_SHADER_READ_BIT);
            src.layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
          }
        } else if (multiFeedback[i]) {
          bool alreadyCopied = false;
          for (uint32_t prior = 0; prior < i; prior++)
            alreadyCopied |= multiFeedback[prior] == multiFeedback[i];
          if (!alreadyCopied && !snapshotRT(g_rts[multiFeedback[i]]))
            return decline(DR_MIDREGION);
        }
      }
    }
    if (rp->multiTex) {
      for (uint32_t i = 0; i < multiN; i++) {
        if (multiStorage[i]) {
          multiViews[i] = g_rts[multiStorage[i]].view;
          multiLayouts[i] = VK_IMAGE_LAYOUT_GENERAL;
        } else if (multiFeedback[i]) {
          auto &src = g_rts[multiFeedback[i]];
          multiViews[i] = sampledView(src, d.texs[i].swizzle, true);
          multiLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        } else if (multiColor[i]) {
          multiViews[i] = sampledView(g_rts[multiColor[i]], d.texs[i].swizzle);
          multiLayouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        } else if (multiDepth[i]) {
          multiViews[i] =
              sampledView(g_depths[multiDepth[i]], d.texs[i].swizzle);
          multiLayouts[i] = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        }
      }
      texSet = getMultiTexSet(d, rp->texSetLayout, multiViews, multiLayouts);
      if (!texSet)
        return decline(DR_GUESTTEX);
    }
    RTarget *rt = d.rtBase ? getRT(d.rtBase, d.rtW, d.rtH,
                                   colorTargetFormat(d.mrtInfo[0]))
                           : nullptr;
    if (d.rtBase && !rt)
      return true; // RT cap hit: treat as handled (dropped)
    beginRegion(d.mrtBase, d.mrtInfo, mrtN, d.rtW, d.rtH, d.depthBase,
                d.depthClear);
  }
  if (rp->multiTex) {
    if (!texSet) {
      for (uint32_t i = 0; i < multiN; i++) {
        if (multiStorage[i]) {
          multiViews[i] = g_rts[multiStorage[i]].view;
          multiLayouts[i] = VK_IMAGE_LAYOUT_GENERAL;
        } else if (multiFeedback[i]) {
          multiViews[i] =
              sampledView(g_rts[multiFeedback[i]], d.texs[i].swizzle, true);
        } else if (multiColor[i]) {
          multiViews[i] = sampledView(g_rts[multiColor[i]], d.texs[i].swizzle);
        } else if (multiDepth[i]) {
          multiViews[i] =
              sampledView(g_depths[multiDepth[i]], d.texs[i].swizzle);
          multiLayouts[i] = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        }
      }
      texSet = getMultiTexSet(d, rp->texSetLayout, multiViews, multiLayouts);
    }
    if (!texSet)
      return decline(DR_GUESTTEX);
  } else if (feedbackAsTex) {
    VkImageView views[kMaxTex] = {};
    VkImageLayout layouts[kMaxTex] = {};
    views[0] = sampledView(g_rts[texBase], d.texSwizzle, true);
    layouts[0] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    texSet = getMultiTexSet(d, g_tex.dsLayout, views, layouts);
    if (!texSet)
      return decline(DR_MIDREGION);
  } else if (colorAsTex) {
    auto &src = g_rts[texBase];
    if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
      return decline(DR_MIDREGION);
    VkImageView views[kMaxTex] = {};
    VkImageLayout layouts[kMaxTex] = {};
    views[0] = sampledView(src, d.texSwizzle);
    layouts[0] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    texSet = getMultiTexSet(d, g_tex.dsLayout, views, layouts);
    if (!texSet)
      return decline(DR_MIDREGION);
  } else if (depthAsTex) {
    auto &src = g_depths[texBase];
    if (src.layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL)
      return decline(DR_MIDREGION);
    VkImageView views[kMaxTex] = {};
    VkImageLayout layouts[kMaxTex] = {};
    views[0] = sampledView(src, d.texSwizzle);
    layouts[0] = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    texSet = getMultiTexSet(d, g_tex.dsLayout, views, layouts);
    if (!texSet)
      return decline(DR_MIDREGION);
  }

  setGuestViewport(d);
  vkCmdBindPipeline(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp->pipe);
  vkCmdPushConstants(g_frame.cmd, rp->layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(d.vsUserData), d.vsUserData);
  vkCmdPushConstants(g_frame.cmd, rp->layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof(d.psUserData), d.psUserData);
  // Copy each guest cbuffer window into the per-frame ring and bind set 1. Vulkan
  // requires one dynamic offset for every dynamic descriptor in the set layout.
  VkDeviceSize cbOff = (g_ring.uboOffset + g_ring.uboAlign - 1) & ~(VkDeviceSize)(g_ring.uboAlign - 1);
  VkDeviceSize cbStride =
      (kCbufWindow + g_ring.uboAlign - 1) & ~(VkDeviceSize)(g_ring.uboAlign - 1);
  if (cbOff + cbStride * kCbufBindings > g_ring.uboEnd)
    return decline(DR_RING);
  uint32_t dynOff[kCbufBindings];
  VkDeviceSize next = cbOff;
  for (uint32_t i = 0; i < kCbufBindings; i++) {
    const auto &cb = d.cbufs[i];
    const uint32_t readable = std::min(cb.size, kCbufWindow);
    const bool haveCbuf = readable && gpu::IsReadableRange(cb.base, readable);
    if (!haveCbuf && i != 0) {
      dynOff[i] = 0; // shared zero window (see beginFrame)
      continue;
    }
    uint8_t *cbDst = g_ring.uboMap + next;
    uint32_t n;
    if (haveCbuf)
      flushCsWritesRange(cb.base, kCbufWindow);
    if (haveCbuf) {
      // Upload as much of the window as the base's page holds, not just the
      // recompiler's planned size. A shader that indexes its constants
      // dynamically -- a UI batch picking a per-quad transform out of an array
      // -- reads past the planned size, and the truncated copy left those
      // entries zero: Skyrim's menu drew its sprite atlas at screen size over
      // everything. Clamped to the page so a cbuffer at the end of a mapping
      // cannot fault.
      static const bool tightCbuf =
          std::getenv("DELTA_GPU_TIGHTCBUF") != nullptr;
      const uint32_t planned = cb.size < kCbufWindow ? cb.size : kCbufWindow;
      const uint64_t pageEnd = (cb.base + 0x1000) & ~uint64_t{0xFFF};
      const uint32_t avail = static_cast<uint32_t>(
          std::min<uint64_t>(kCbufWindow, pageEnd - cb.base));
      n = tightCbuf ? planned : std::max(planned, avail);
      std::memcpy(cbDst, reinterpret_cast<const void *>(cb.base), n);
    } else { // binding 0 without a resolved cbuffer: the heuristic MVP
      n = sizeof(d.mvp);
      std::memcpy(cbDst, d.mvp, n);
    }
    if (n < kCbufWindow)
      std::memset(cbDst + n, 0, kCbufWindow - n);
    dynOff[i] = static_cast<uint32_t>(next);
    next += cbStride;
  }
  g_ring.uboOffset = next;
  vkCmdBindDescriptorSets(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp->layout,
                          1, 1, &g_ring.uboSet, kCbufBindings, dynOff);
  if (texSet)
    vkCmdBindDescriptorSets(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp->layout, 0, 1, &texSet, 0, nullptr);
  if (nbind) {
    VkBuffer bufs[8];
    VkDeviceSize offs[8];
    for (uint32_t j = 0; j < nbind; j++) { bufs[j] = g_ring.vb; offs[j] = voff + bindOff[j]; }
    vkCmdBindVertexBuffers(g_frame.cmd, 0, nbind, bufs, offs);
  }
  if (indexed)
    vkCmdBindIndexBuffer(g_frame.cmd, g_ring.ib, ioff, VK_INDEX_TYPE_UINT32);
  if (drawTrace && drawCount >= 300) {
    static int n = 0;
    if (n++ < 40)
      std::fprintf(stderr, "[dt] RECOMP DREW count=%u rt=%#lx nv=%u multiTex=%d\n",
                   drawCount, (unsigned long)d.rtBase, nv, rp->multiTex);
  }
  if (indexed)
    vkCmdDrawIndexed(g_frame.cmd, d.indexCount, d.instanceCount ? d.instanceCount : 1,
                     0, 0, 0);
  else
    vkCmdDraw(g_frame.cmd, d.vertexCount, d.instanceCount ? d.instanceCount : 1, 0, 0);
  // DELTA_GPU_DRAWRT=<base>: report what was actually issued for one target.
  {
    static const uint64_t want = [] {
      const char *e = std::getenv("DELTA_GPU_DRAWRT");
      return e ? std::strtoull(e, nullptr, 0) : 0ull;
    }();
    static int shown = 0;
    // DELTA_GPU_DRAWRT=1 logs EVERY draw (the whole frame graph) instead of one
    // target's draws, so a broken producer/consumer link is visible directly.
    const bool all = want == 1;
    // DELTA_GPU_DRAWRT_FRAME=N: only this frame, so the graph is a steady-state
    // frame rather than the opening composites.
    static const int wantFrame = [] {
      const char *e = std::getenv("DELTA_GPU_DRAWRT_FRAME");
      return e ? std::atoi(e) : 0;
    }();
    if (want && (all || g_region.curRt == want) && shown < (all ? 60 : 8) &&
        (!wantFrame || (int)g_frame.num == wantFrame)) {
      shown++;
      std::fprintf(stderr,
                   "[drawrt] rt=%#lx %ux%u indexed=%d vcount=%u icount=%u "
                   "prim=%u tmask=%#x nvbufs=%u stride=%u mrt=%u "
                   "vp=%g,%g scale %g,%g off vs=%#lx ps=%#lx\n",
                   (unsigned long)g_region.curRt, d.rtW, d.rtH, (int)indexed,
                   d.vertexCount, d.indexCount, d.primType, d.targetMask,
                   d.nvbufs, d.vertexStride, d.mrtCount, d.viewportXScale,
                   d.viewportYScale, d.viewportXOffset, d.viewportYOffset,
                   (unsigned long)d.vsAddr, (unsigned long)d.psAddr);
      for (uint32_t i = 0; i < multiN; i++) {
        const auto &t = d.texs[i];
        if (!t.base) continue;
        std::fprintf(stderr,
                     "[drawrt]  tex%u %#lx %ux%u dfmt=%u tiling=%u -> %s%#lx\n",
                     i, (unsigned long)t.base, t.w, t.h, t.dfmt, t.tiling,
                     multiColor[i]      ? "rt "
                     : multiFeedback[i] ? "fb "
                     : multiDepth[i]    ? "depth "
                     : multiViews[i]    ? "guest "
                                        : "MISS ",
                     (unsigned long)(multiColor[i]      ? multiColor[i]
                                     : multiFeedback[i] ? multiFeedback[i]
                                     : multiDepth[i]    ? multiDepth[i]
                                                        : 0));
      }
      for (uint32_t c = 0; c < kCbufBindings; c++) {
        const auto &cb = d.cbufs[c];
        if (!gpu::IsReadableRange(cb.base, std::min(cb.size, 192u)))
          continue;
        const uint32_t *w = reinterpret_cast<const uint32_t *>(cb.base);
        std::fprintf(stderr, "[drawrt]  cb%u %#lx sz=%u:", c,
                     (unsigned long)cb.base, cb.size);
        for (uint32_t k = 0; k < 8 && k * 4 < cb.size; k++)
          std::fprintf(stderr, " %g", *reinterpret_cast<const float *>(&w[k]));
        // The 3D VS loads its transform with s_buffer_load_dwordx16 at byte
        // offset 0x80; show that window when the binding is big enough to hold
        // it.
        if (cb.size >= 192) {
          std::fprintf(stderr, "\n[drawrt]   @0x80:");
          for (uint32_t k = 32; k < 48; k++)
            std::fprintf(stderr, " %g",
                         *reinterpret_cast<const float *>(&w[k]));
        }
        std::fprintf(stderr, "\n");
      }
    }
  }
  for (uint32_t i = 0; i < multiN; i++) {
    if (!multiStorage[i]) continue;
    RTarget &target = g_rts[multiStorage[i]];
    target.everRendered = true;
    target.usedThisFrame = true;
    target.lastFrame = g_frame.num;
  }
  g_ring.vbOffset += vneed;
  if (indexed)
    g_ring.ibOffset += (VkDeviceSize)d.indexCount * 4;
  g_frame.draws++;
  if (g_region.curRt) { auto &rt = g_rts[g_region.curRt];
    if (++rt.draws > g_region.busiestRtDraws) { g_region.busiestRtDraws = rt.draws; g_region.busiestRt = g_region.curRt; } }
  return true;
}

}  // namespace gpu::vk
