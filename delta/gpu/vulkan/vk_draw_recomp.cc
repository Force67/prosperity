/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_draw_recomp.h"

#include "gpu/guest_memory.h"
#include "gpu/ps4/gcn/gcn_translate.h"
#include "gpu/rhi/renderer.h"
#include "gpu/vulkan/vk_debug.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_format.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_index_upload.h"
#include "gpu/vulkan/vk_pipeline_cache.h"
#include "gpu/vulkan/vk_render_target.h"
#include "gpu/vulkan/vk_texture_cache.h"
#include "gpu/vulkan/vk_upload_ring.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace gpu::vk {

using rhi::DrawInfo;
using rhi::FlushCsWritesRange;

namespace {

// Why a draw declined the recompiled path (falls back to the heuristic).
// Tallied per reason so the remaining heuristic draws can be driven to zero;
// dumped with the periodic frame log.
enum DeclineReason {
  kNoRecomp,
  kNoTexPipe,
  kSelf,
  kRing,
  kGuestTex,
  kMidRegion,
  kNoPipe,
  kMaxDeclineReason
};
static const char* kDeclineName[kMaxDeclineReason] = {
    "norecomp", "notexpipe", "self", "ring", "guesttex", "midregion", "nopipe"};
uint32_t g_decline[kMaxDeclineReason] = {0};
inline bool Decline(DeclineReason r) {
  g_decline[r]++;
  return false;
}

struct ReadableRangeKey {
  uint64_t base;
  uint32_t size;
  bool operator==(const ReadableRangeKey&) const = default;
};

struct ReadableRangeKeyHash {
  size_t operator()(const ReadableRangeKey& key) const {
    return static_cast<size_t>(key.base ^ (key.base >> 32) ^ key.size);
  }
};

bool IsReadableThisFrame(uint64_t base, uint32_t size) {
  static int frame = -1;
  static std::unordered_map<ReadableRangeKey, bool, ReadableRangeKeyHash> cache;
  if (frame != g_frame.num) {
    frame = g_frame.num;
    cache.clear();
  }
  const ReadableRangeKey key{base, size};
  auto found = cache.find(key);
  if (found != cache.end())
    return found->second;
  return cache.emplace(key, gpu::IsReadableRange(base, size)).first->second;
}

}  // namespace

void ReportDeclines() {
  std::fprintf(stderr, "[gpuvk]   decline:");
  for (int i = 0; i < kMaxDeclineReason; i++)
    if (g_decline[i])
      std::fprintf(stderr, " %s=%u", kDeclineName[i], g_decline[i]);
  std::fprintf(stderr, "\n");
}

// Issue a draw running the game's recompiled VS/PS. Returns false if the draw
// can't be handled (the caller falls back to the heuristic path).
bool DrawRecomp(rhi::Renderer& renderer, const DrawInfo& d) {
  bool indexed = d.index_data && d.index_count >= 3;
  uint32_t draw_count = indexed ? d.index_count : d.vertex_count;
  static const bool kDrawTrace = std::getenv("DELTA_GPU_DRAWTRACE") != nullptr;
  if (kDrawTrace && draw_count >= 300) {
    static int n = 0;
    if (n++ < 40)
      std::fprintf(stderr,
                   "[dt] enter recomp count=%u rt=%#lx tex=%#lx num_texs=%u "
                   "num_vattrs=%u ok=%d\n",
                   draw_count, (unsigned long)d.rt_base,
                   (unsigned long)d.tex_base, d.num_texs, d.num_vattrs,
                   d.recomp ? d.recomp->ok : 0);
  }
  if (!d.recomp || !d.recomp->ok || draw_count < 3)
    return Decline(kNoRecomp);
  const bool has_storage_image =
      std::any_of(d.recomp->ps_texs.begin(), d.recomp->ps_texs.end(),
                  [](const gcn::ShaderTex& tex) { return tex.storage; });
  if (!d.mrt_count && !d.depth_base && !has_storage_image) {
    g_frame.draws++;
    return true;
  }
  if (!g_quad.tex_pipeline)
    return Decline(
        kNoTexPipe);  // need the descriptor infra (ds_pool/ds_layout)
  // Resolve the sampled texture address to an overlapping live RT
  // (resource-model page-table lookup), so an RT-as-texture sample binds the
  // live image for cycled/aliased RT addresses instead of stale guest memory.
  // Additive: an exact RT base resolves to itself.
  uint64_t tex_base = d.tex_base;
  if (tex_base && !d.tex_arrayed && !g_rts.count(tex_base) &&
      !g_depths.count(tex_base)) {
    bool depth_format = d.tex_dfmt == 4 && d.tex_nfmt == 7;
    uint64_t r =
        depth_format ? ResolveSampledDepth(tex_base, d.tex_w, d.tex_h) : 0;
    if (!r)
      r = ResolveSampledRT(tex_base, d.tex_w, d.tex_h);
    if (!r && !depth_format)
      r = ResolveSampledDepth(tex_base, d.tex_w, d.tex_h);
    if (r)
      tex_base = r;
  }
  bool color_as_tex = !d.tex_arrayed && tex_base && tex_base != d.rt_base &&
                      g_rts.count(tex_base);
  bool feedback_as_tex = !d.tex_arrayed && tex_base && tex_base == d.rt_base &&
                         g_rts.count(tex_base) && g_rts[tex_base].ever_rendered;
  bool depth_as_tex = !d.tex_arrayed && tex_base && tex_base != d.depth_base &&
                      g_depths.count(tex_base);
  bool rt_as_tex = color_as_tex || feedback_as_tex || depth_as_tex;
  if (color_as_tex && g_rts[tex_base].w >= 700 && g_rts[tex_base].w <= 900)
    g_frame.had_room = true;
  if (tex_base && (tex_base == d.depth_base ||
                   (tex_base == d.rt_base && !feedback_as_tex))) {
    // DELTA_GPU_SELFTRACE: which pass reads the target it is drawing into. We
    // drop those, and dropping one every frame leaves whatever it was meant to
    // produce stale.
    static const bool kSelfTrace =
        std::getenv("DELTA_GPU_SELFTRACE") != nullptr;
    static int n = 0;
    if (kSelfTrace && n++ < 20)
      std::fprintf(
          stderr,
          "[self] draw#%u ps=%#lx rt=%#lx tex=%#lx depth=%#lx dtest=%d "
          "dwrite=%d ntex=%u\n",
          g_frame.draws, (unsigned long)d.ps_addr, (unsigned long)d.rt_base,
          (unsigned long)tex_base, (unsigned long)d.depth_base,
          (int)d.depth_test_enable, (int)d.depth_write_enable, d.num_texs);
    return Decline(kSelf);
  }
  // Indexed draws derive the copied vertex range from their indices.
  // DRAW_INDEX_AUTO consumes the packet's sequential vertex count directly.
  uint32_t nv = d.vertex_count;
  if (indexed) {
    const uint32_t max_index =
        MaxGuestIndex(d.index_data, d.index_count, d.index_type);
    if (max_index >= 200000u)
      return Decline(kNoRecomp);
    nv = max_index + 1;
  }
  if (nv > 200000u || (d.num_vattrs && (!d.vertex_data || !d.vertex_stride)))
    return Decline(kNoRecomp);
  if (d.vertex_data && d.vertex_stride &&
      !FlushCsWritesRange(renderer, reinterpret_cast<uint64_t>(d.vertex_data),
                          static_cast<uint64_t>(nv) * d.vertex_stride))
    return Decline(kNoRecomp);

  // A fullscreen, untextured, near-black REPLACE draw is the game CLEARING an
  // RT. Don't render it (that wipes the RT immediately); record a LAZY clear
  // instead -- realised as loadOp=CLEAR only when content actually redraws this
  // RT this frame (see BeginRegion). Baked-once content (the room floor) whose
  // clear and redraw land on different frames then survives. A COLOURED
  // fullscreen REPLACE is real content (e.g. the per-frame minimap redraw) and
  // must NOT be treated as a clear.
  static const bool kNoWipe = [] {
    const char* e = std::getenv("DELTA_GPU_NOWIPE");
    return !e || std::strcmp(e, "0") != 0;
  }();
  static const bool kLazyClear2 = [] {
    const char* e = std::getenv("DELTA_GPU_LAZYCLEAR");
    return !e || std::strcmp(e, "0") != 0;
  }();
  // The extent test below reads the position attribute as float32s. A title
  // whose positions are not floats (Skyrim's UI uses 16_16_SScaled) would have
  // its geometry read as garbage, land a bogus fullscreen extent, and get
  // swallowed as a "clear" -- which is exactly what turned its whole frame
  // black. Only consider draws whose position really is float.
  bool float_pos = false;
  for (uint32_t a = 0; a < d.num_vattrs; a++) {
    if (d.vattrs[a].location != 0)
      continue;
    const uint32_t df = d.vattrs[a].dfmt;
    float_pos =
        d.vattrs[a].nfmt == 7 && (df == 4 || df == 11 || df == 13 || df == 14);
    break;
  }
  if (kNoWipe && float_pos && d.vertex_data && d.num_vattrs &&
      d.recomp->ps_texs.empty() && nv <= 8) {
    uint32_t cdst = (d.blend_control >> 8) & 0x1F,
             csrc = d.blend_control & 0x1F;
    bool replace = d.blend_enable && csrc == 1 && cdst == 0;
    if (replace) {
      const auto* vb = static_cast<const uint8_t*>(d.vertex_data);
      bool near_black = true;
      float clear_color[4] = {0, 0, 0, 0};
      for (uint32_t a = 0; a < d.num_vattrs; a++) {
        if (d.vattrs[a].num_comps == 4 && d.vattrs[a].offset != 0) {
          // Colour may live in its own binding; read from that binding's base.
          const auto* cbuf =
              static_cast<const uint8_t*>(d.vbufs[d.vattrs[a].binding].data);
          const uint8_t* cb0 = cbuf + d.vattrs[a].offset;  // vertex 0's colour
          if (d.vattrs[a].dfmt == 10) {
            for (int i = 0; i < 4; i++)
              clear_color[i] = cb0[i] / 255.f;
          } else {
            const float* c = reinterpret_cast<const float*>(cb0);
            for (int i = 0; i < 4; i++)
              clear_color[i] = c[i];
          }
          if (clear_color[0] > 0.02f || clear_color[1] > 0.02f ||
              clear_color[2] > 0.02f)
            near_black = false;
          break;
        }
      }
      const float* m = d.mvp;
      float nx0 = 1e9f, ny0 = 1e9f, nx1 = -1e9f, ny1 = -1e9f;
      for (uint32_t v = 0; v < nv; v++) {
        const float* p =
            reinterpret_cast<const float*>(vb + (size_t)v * d.vertex_stride);
        float cw = m[3] * p[0] + m[7] * p[1] + m[15];
        if (cw == 0)
          cw = 1;
        float nx = (m[0] * p[0] + m[4] * p[1] + m[12]) / cw,
              ny = (m[1] * p[0] + m[5] * p[1] + m[13]) / cw;
        nx0 = nx < nx0 ? nx : nx0;
        nx1 = nx > nx1 ? nx : nx1;
        ny0 = ny < ny0 ? ny : ny0;
        ny1 = ny > ny1 ? ny : ny1;
      }
      bool fullscreen_black =
          near_black && (nx1 - nx0) >= 1.8f && (ny1 - ny0) >= 1.8f;
      if (fullscreen_black && kLazyClear2) {
        RTarget* rt = d.rt_base ? GetRT(d.rt_base, d.rt_w, d.rt_h,
                                        ColorTargetFormat(d.mrt_info[0]))
                                : nullptr;
        if (rt) {
          rt->clear_pending = true;
          std::memcpy(rt->clear_value.float32, clear_color,
                      sizeof(clear_color));
        }
        // This draw also performs the guest's reverse-Z clear (depth write
        // enabled, ZFUNC=ALWAYS). Suppressing its color write must not discard
        // that depth effect, or stale depth rejects the following layer
        // composites.
        if (d.depth_base && d.depth_write_enable && d.depth_func == 7) {
          DepthTarget* dt = GetDepthRT(d.depth_base, d.rt_w, d.rt_h);
          if (dt) {
            dt->clear_pending = true;
            dt->clear_value = d.depth_clear;
          }
        }
        g_frame.draws++;
        return true;  // suppressed; the clear is applied lazily on the next
                      // redraw
      }
      // Legacy single-frame behaviour (DELTA_GPU_LAZYCLEAR=0): only suppress if
      // the RT already holds content this frame.
      auto rit = g_rts.find(d.rt_base);
      if (fullscreen_black && rit != g_rts.end() && rit->second.draws > 0 &&
          rit->second.last_frame == g_frame.num) {
        g_frame.draws++;
        return true;
      }
    }
  }

  // Lay out one contiguous ring range per vertex binding. Binding 0 sits at the
  // ring offset (single-stream draws are byte-identical to before); additional
  // bindings are 16-byte aligned so no attribute straddles a coarse boundary.
  const uint32_t nbind = d.num_vattrs ? std::min(d.num_vbufs, 8u) : 0;
  VkDeviceSize bind_off[8] = {}, bind_size[8] = {};
  VkDeviceSize vneed = 0;
  for (uint32_t j = 0; j < nbind; j++) {
    if (j)
      vneed = (vneed + 15) & ~VkDeviceSize(15);
    bind_off[j] = vneed;
    if (d.vbufs[j].stride) {
      bind_size[j] = (VkDeviceSize)nv * d.vbufs[j].stride;
    } else {
      // Stride-0 (constant) binding: upload a single record large enough to
      // cover every attribute that reads it; the pipeline binds it with stride
      // 0 so all vertices fetch this one record.
      uint32_t rec = 0;
      for (uint32_t a = 0; a < d.num_vattrs; a++)
        if (d.vattrs[a].binding == j)
          rec = std::max(
              rec, d.vattrs[a].offset + VertexFormatBytes(d.vattrs[a].dfmt));
      bind_size[j] = rec;
    }
    vneed += bind_size[j];
  }
  if (g_ring.vb_offset + vneed > g_ring.vb_end)
    return Decline(kRing);
  const VkDeviceSize index_bytes =
      indexed ? static_cast<VkDeviceSize>(d.index_count) *
                    UploadedIndexElementBytes(d.index_type)
              : 0;
  const VkDeviceSize index_align = d.index_type == 1 ? 4 : 2;
  const VkDeviceSize aligned_ioff =
      (g_ring.ib_offset + index_align - 1) & ~(index_align - 1);
  if (indexed && aligned_ioff + index_bytes > g_ring.ib_end)
    return Decline(kRing);

  RecompPipe* rp = GetRecompPipe(d);
  if (!rp)
    return Decline(kNoPipe);
  // Guest-texture source resolved up front; an RT-as-texture source is resolved
  // after the region switch (transitioning it to readable must happen outside a
  // region).
  VkDescriptorSet tex_set = VK_NULL_HANDLE;
  if (rp->textured && !rp->multi_tex && !rt_as_tex) {
    if (GuestTextureUploadSupported(d.tex_dfmt, d.tex_nfmt))
      tex_set = GetTexture(
          d.tex_base, d.tex_w, d.tex_h, d.tex_dfmt, d.tex_nfmt, d.tex_tiling,
          d.tex_pitch, d.tex_layers, d.tex_base_array, d.tex_view_layers,
          d.tex_mip_levels, d.tex_base_mip, d.tex_view_mips, d.tex_min_lod,
          d.tex_pow2_pad, d.tex_sampler, d.tex_sampler_valid, d.tex_arrayed,
          d.tex_force_lod_zero, d.tex_depth_compare, d.tex_swizzle);
    if (!tex_set)
      tex_set = d.tex_arrayed ? g_tex.white_array_set : g_tex.white_set;
    if (!tex_set)
      return Decline(kGuestTex);
  }

  uint64_t multi_color[kMaxTex] = {};
  uint64_t multi_depth[kMaxTex] = {};
  uint64_t multi_feedback[kMaxTex] = {};
  uint64_t multi_storage[kMaxTex] = {};
  VkImageView multi_views[kMaxTex] = {};
  VkImageLayout multi_layouts[kMaxTex];
  bool multi_transition_source = false;
  uint32_t multi_n = std::min(d.num_texs, kMaxTex);
  if (rp->multi_tex) {
    auto is_bound_target = [&](uint64_t base) {
      uint32_t count = std::min(d.mrt_count, 8u);
      for (uint32_t m = 0; m < count; m++)
        if (d.mrt_base[m] == base)
          return true;
      return false;
    };
    for (uint32_t i = 0; i < kMaxTex; i++)
      multi_layouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    for (uint32_t i = 0; i < multi_n; i++) {
      const auto& t = d.texs[i];
      uint64_t base = t.base;
      if (t.storage) {
        if (base && !g_rts.count(base)) {
          uint64_t resolved = ResolveSampledRT(base, t.w, t.h);
          if (resolved)
            base = resolved;
        }
        if (base && !g_rts.count(base)) {
          const VkFormat format = GuestTextureFormat(t.dfmt, t.nfmt);
          if (format != VK_FORMAT_UNDEFINED)
            GetRT(base, t.w, t.h, format);
        }
        if (base && g_rts.count(base)) {
          multi_storage[i] = base;
          multi_transition_source |=
              g_rts[base].layout != VK_IMAGE_LAYOUT_GENERAL;
        }
        continue;
      }
      if (base && !t.arrayed && !g_rts.count(base) && !g_depths.count(base)) {
        bool depth_format = t.dfmt == 4 && t.nfmt == 7;
        uint64_t resolved =
            depth_format ? ResolveSampledDepth(base, t.w, t.h) : 0;
        if (!resolved)
          resolved = ResolveSampledRT(base, t.w, t.h);
        if (!resolved && !depth_format)
          resolved = ResolveSampledDepth(base, t.w, t.h);
        if (resolved)
          base = resolved;
      }
      if (base && !t.arrayed && is_bound_target(base) && g_rts.count(base) &&
          g_rts[base].ever_rendered) {
        multi_feedback[i] = base;
        multi_transition_source = true;
      } else if (base && !t.arrayed && g_rts.count(base) &&
                 g_rts[base].ever_rendered) {
        multi_color[i] = base;
        multi_transition_source |=
            g_rts[base].layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      } else if (base && !t.arrayed && base != d.depth_base &&
                 g_depths.count(base)) {
        multi_depth[i] = base;
        multi_transition_source |=
            g_depths[base].layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
      } else {
        multi_views[i] = TexViewFor(t);
      }
    }
  }

  // DELTA_GPU_TEXBIND=<frame>: how each sampler of each draw resolved -- to a
  // live render target, a depth target, guest memory, or the 1x1 white default.
  // A post-processing chain that samples its own previous target reads zero the
  // moment one of those lands on guest memory.
  {
    static const int kTexBindFrame = [] {
      const char* e = std::getenv("DELTA_GPU_TEXBIND");
      return e ? std::atoi(e) : -1;
    }();
    if (kTexBindFrame >= 0 && (int)g_frame.num == kTexBindFrame)
      std::fprintf(stderr,
                   "[blend] draw#%u rt=%#lx blend=%u ctl=%#x mask=%#x mrt=%u\n",
                   g_frame.draws, (unsigned long)d.rt_base, d.blend_enable,
                   d.blend_control, d.target_mask, d.mrt_count);
    if (kTexBindFrame >= 0 && (int)g_frame.num == kTexBindFrame &&
        !rp->multi_tex)
      std::fprintf(stderr,
                   "[texbind] draw#%u rt=%#lx %ux%u LEGACY tex=%#lx %ux%u "
                   "rtAsTex=%u color=%u feedback=%u depth=%u set=%u\n",
                   g_frame.draws, (unsigned long)d.rt_base, d.rt_w, d.rt_h,
                   (unsigned long)d.tex_base, d.tex_w, d.tex_h,
                   (unsigned)rt_as_tex, (unsigned)color_as_tex,
                   (unsigned)feedback_as_tex, (unsigned)depth_as_tex,
                   (unsigned)(tex_set != VK_NULL_HANDLE));
    if (kTexBindFrame >= 0 && (int)g_frame.num == kTexBindFrame && multi_n &&
        rp->multi_tex) {
      std::fprintf(stderr,
                   "[texbind] draw#%u rt=%#lx %ux%u ntex=%u:", g_frame.draws,
                   (unsigned long)d.rt_base, d.rt_w, d.rt_h, multi_n);
      for (uint32_t i = 0; i < multi_n; i++) {
        const char* how = multi_color[i]      ? "RT"
                          : multi_feedback[i] ? "feedback"
                          : multi_depth[i]    ? "depth"
                          : multi_storage[i]  ? "storage"
                          : multi_views[i]    ? "guest"
                                              : "WHITE";
        std::fprintf(
            stderr,
            " [%u]%s@%#lx %ux%u layers=%u mips=%u fmt=%u/%u rt?=%u er=%u", i,
            how, (unsigned long)d.texs[i].base, d.texs[i].w, d.texs[i].h,
            d.texs[i].layers, d.texs[i].mip_levels, d.texs[i].dfmt,
            d.texs[i].nfmt, (unsigned)g_rts.count(d.texs[i].base),
            (unsigned)(g_rts.count(d.texs[i].base)
                           ? g_rts[d.texs[i].base].ever_rendered
                           : 0));
      }
      std::fprintf(stderr, "\n");
    }
  }

  VkDeviceSize voff = g_ring.vb_offset, ioff = aligned_ioff;

  // Switch render target. Re-begin when the primary target or the MRT count
  // changes (the open region's attachment count must match the pipeline's), or
  // when a new RT-as-texture source still needs a read transition. Barriers
  // cannot be recorded inside dynamic rendering; consecutive layer composites
  // often keep the same target while switching sources, so that source change
  // must also close/reopen the region.
  uint32_t mrt_n = std::min(d.mrt_count, 8u);
  bool transition_source =
      rp->multi_tex
          ? multi_transition_source
          : feedback_as_tex ||
                (color_as_tex &&
                 g_rts[tex_base].layout !=
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ||
                (depth_as_tex && g_depths[tex_base].layout !=
                                     VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
  bool pending_depth_clear = d.depth_base && g_depths.count(d.depth_base) &&
                             g_depths[d.depth_base].clear_pending;
  bool restart_region = g_region.cur_rt != d.rt_base ||
                        g_region.cur_mrt_count != mrt_n ||
                        g_region.cur_depth != d.depth_base ||
                        transition_source || pending_depth_clear;
  if (restart_region) {
    EndRegion();
    if (!rp->multi_tex && color_as_tex && transition_source) {
      auto& src = g_rts[tex_base];
      if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        ImageBarrier(g_frame.cmd, src.image, src.layout,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     ColorImageAccess(src.layout), VK_ACCESS_SHADER_READ_BIT);
        src.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
    }
    if (!rp->multi_tex && depth_as_tex && transition_source) {
      auto& src = g_depths[tex_base];
      if (src.layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL) {
        DepthBarrier(g_frame.cmd, src.image, src.layout,
                     VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_ACCESS_SHADER_READ_BIT);
        src.layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
      }
    }
    if (!rp->multi_tex && feedback_as_tex) {
      tex_set = SnapshotRT(g_rts[tex_base]);
      if (!tex_set)
        return Decline(kMidRegion);
    }
    if (rp->multi_tex) {
      for (uint32_t i = 0; i < multi_n; i++) {
        if (multi_storage[i]) {
          auto& dst = g_rts[multi_storage[i]];
          if (dst.layout != VK_IMAGE_LAYOUT_GENERAL) {
            // Storage images are read *and* written (imageLoad/imageStore), so
            // the destination access needs both bits, and the source access
            // must cover every layout an RT can arrive from (a hand-rolled
            // subset missed TRANSFER_DST, leaving the transition unordered
            // against the clear that put it there).
            ImageBarrier(g_frame.cmd, dst.image, dst.layout,
                         VK_IMAGE_LAYOUT_GENERAL, ColorImageAccess(dst.layout),
                         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            dst.layout = VK_IMAGE_LAYOUT_GENERAL;
          }
        } else if (multi_color[i]) {
          auto& src = g_rts[multi_color[i]];
          if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            ImageBarrier(g_frame.cmd, src.image, src.layout,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         ColorImageAccess(src.layout),
                         VK_ACCESS_SHADER_READ_BIT);
            src.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          }
        } else if (multi_depth[i]) {
          auto& src = g_depths[multi_depth[i]];
          if (src.layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL) {
            DepthBarrier(g_frame.cmd, src.image, src.layout,
                         VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                         VK_ACCESS_SHADER_READ_BIT);
            src.layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
          }
        } else if (multi_feedback[i]) {
          bool already_copied = false;
          for (uint32_t prior = 0; prior < i; prior++)
            already_copied |= multi_feedback[prior] == multi_feedback[i];
          if (!already_copied && !SnapshotRT(g_rts[multi_feedback[i]]))
            return Decline(kMidRegion);
        }
      }
    }
    if (rp->multi_tex) {
      for (uint32_t i = 0; i < multi_n; i++) {
        if (multi_storage[i]) {
          multi_views[i] = g_rts[multi_storage[i]].view;
          multi_layouts[i] = VK_IMAGE_LAYOUT_GENERAL;
        } else if (multi_feedback[i]) {
          auto& src = g_rts[multi_feedback[i]];
          multi_views[i] = SampledView(src, d.texs[i].swizzle, true);
          multi_layouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        } else if (multi_color[i]) {
          multi_views[i] =
              SampledView(g_rts[multi_color[i]], d.texs[i].swizzle);
          multi_layouts[i] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        } else if (multi_depth[i]) {
          multi_views[i] =
              SampledView(g_depths[multi_depth[i]], d.texs[i].swizzle);
          multi_layouts[i] = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        }
      }
      tex_set =
          GetMultiTexSet(d, rp->tex_set_layout, multi_views, multi_layouts);
      if (!tex_set)
        return Decline(kGuestTex);
    }
    RTarget* rt = d.rt_base ? GetRT(d.rt_base, d.rt_w, d.rt_h,
                                    ColorTargetFormat(d.mrt_info[0]))
                            : nullptr;
    if (d.rt_base && !rt)
      return true;  // RT cap hit: treat as handled (dropped)
    if (!BeginRegion(d.mrt_base, d.mrt_info, mrt_n, d.rt_w, d.rt_h,
                     d.depth_base, d.depth_clear))
      return true;
  }
  if (rp->multi_tex) {
    if (!tex_set) {
      for (uint32_t i = 0; i < multi_n; i++) {
        if (multi_storage[i]) {
          multi_views[i] = g_rts[multi_storage[i]].view;
          multi_layouts[i] = VK_IMAGE_LAYOUT_GENERAL;
        } else if (multi_feedback[i]) {
          multi_views[i] =
              SampledView(g_rts[multi_feedback[i]], d.texs[i].swizzle, true);
        } else if (multi_color[i]) {
          multi_views[i] =
              SampledView(g_rts[multi_color[i]], d.texs[i].swizzle);
        } else if (multi_depth[i]) {
          multi_views[i] =
              SampledView(g_depths[multi_depth[i]], d.texs[i].swizzle);
          multi_layouts[i] = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        }
      }
      tex_set =
          GetMultiTexSet(d, rp->tex_set_layout, multi_views, multi_layouts);
    }
    if (!tex_set)
      return Decline(kGuestTex);
  } else if (feedback_as_tex) {
    VkImageView views[kMaxTex] = {};
    VkImageLayout layouts[kMaxTex] = {};
    views[0] = SampledView(g_rts[tex_base], d.tex_swizzle, true);
    layouts[0] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    tex_set = GetMultiTexSet(d, g_tex.ds_layout, views, layouts);
    if (!tex_set)
      return Decline(kMidRegion);
  } else if (color_as_tex) {
    auto& src = g_rts[tex_base];
    if (src.layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
      return Decline(kMidRegion);
    VkImageView views[kMaxTex] = {};
    VkImageLayout layouts[kMaxTex] = {};
    views[0] = SampledView(src, d.tex_swizzle);
    layouts[0] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    tex_set = GetMultiTexSet(d, g_tex.ds_layout, views, layouts);
    if (!tex_set)
      return Decline(kMidRegion);
  } else if (depth_as_tex) {
    auto& src = g_depths[tex_base];
    if (src.layout != VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL)
      return Decline(kMidRegion);
    VkImageView views[kMaxTex] = {};
    VkImageLayout layouts[kMaxTex] = {};
    views[0] = SampledView(src, d.tex_swizzle);
    layouts[0] = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    tex_set = GetMultiTexSet(d, g_tex.ds_layout, views, layouts);
    if (!tex_set)
      return Decline(kMidRegion);
  }

  SetGuestViewport(d);
  vkCmdBindPipeline(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, rp->pipe);
  vkCmdPushConstants(g_frame.cmd, rp->layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(d.vs_user_data), d.vs_user_data);
  vkCmdPushConstants(g_frame.cmd, rp->layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof(d.ps_user_data), d.ps_user_data);
  // Copy each guest cbuffer window into the per-frame ring and bind set 1.
  // Vulkan requires one dynamic offset for every dynamic descriptor in the set
  // layout.
  VkDeviceSize cb_off = (g_ring.ubo_offset + g_ring.ubo_align - 1) &
                        ~(VkDeviceSize)(g_ring.ubo_align - 1);
  VkDeviceSize cb_stride = g_ring.ubo_stride;
  if (cb_off + cb_stride * kCbufBindings > g_ring.ubo_end)
    return Decline(kRing);
  uint32_t dyn_off[kCbufBindings];
  VkDeviceSize next = cb_off;
  for (uint32_t i = 0; i < kCbufBindings; i++) {
    const auto& cb = d.cbufs[i];
    const uint32_t readable = std::min(cb.size, kCbufWindow);
    const bool have_cbuf = readable && IsReadableThisFrame(cb.base, readable);
    if (!have_cbuf && i != 0) {
      dyn_off[i] = 0;  // shared zero window (see BeginFrame)
      continue;
    }
    uint8_t* cb_dst = g_ring.ubo_map + next;
    uint32_t n;
    if (have_cbuf && !FlushCsWritesRange(renderer, cb.base, kCbufWindow))
      return Decline(kNoRecomp);
    if (have_cbuf) {
      // Upload as much of the window as the base's page holds, not just the
      // recompiler's planned size. A shader that indexes its constants
      // dynamically -- a UI batch picking a per-quad transform out of an array
      // -- reads past the planned size, and the truncated copy left those
      // entries zero: Skyrim's menu drew its sprite atlas at screen size over
      // everything. Clamped to the page so a cbuffer at the end of a mapping
      // cannot fault.
      static const bool kTightCbuf =
          std::getenv("DELTA_GPU_TIGHTCBUF") != nullptr;
      const uint32_t planned = cb.size < kCbufWindow ? cb.size : kCbufWindow;
      const uint64_t page_end = (cb.base + 0x1000) & ~uint64_t{0xFFF};
      const uint32_t avail = static_cast<uint32_t>(
          std::min<uint64_t>(kCbufWindow, page_end - cb.base));
      n = kTightCbuf ? planned : std::max(planned, avail);
      std::memcpy(cb_dst, reinterpret_cast<const void*>(cb.base), n);
    } else {  // binding 0 without a resolved cbuffer: the heuristic MVP
      n = sizeof(d.mvp);
      std::memcpy(cb_dst, d.mvp, n);
    }
    const size_t window = static_cast<size_t>(next / cb_stride);
    if (window >= g_ring.ubo_written.size())
      return Decline(kRing);
    const uint32_t previous = g_ring.ubo_written[window];
    if (n < previous)
      std::memset(cb_dst + n, 0, previous - n);
    g_ring.ubo_written[window] = n;
    dyn_off[i] = static_cast<uint32_t>(next);
    next += cb_stride;
  }
  g_ring.ubo_offset = next;
  vkCmdBindDescriptorSets(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          rp->layout, 1, 1, &g_ring.ubo_set, kCbufBindings,
                          dyn_off);
  if (tex_set)
    vkCmdBindDescriptorSets(g_frame.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            rp->layout, 0, 1, &tex_set, 0, nullptr);
  // Commit ring uploads only after every fallible pipeline, texture, region and
  // cbuffer decision has succeeded.
  for (uint32_t j = 0; j < nbind; j++) {
    if (!bind_size[j])
      continue;
    if (!FlushCsWritesRange(renderer,
                            reinterpret_cast<uint64_t>(d.vbufs[j].data),
                            bind_size[j]))
      return Decline(kNoRecomp);
    std::memcpy(g_ring.vb_map + voff + bind_off[j], d.vbufs[j].data,
                (size_t)bind_size[j]);
  }
  if (indexed) {
    CopyGuestIndices(g_ring.ib_map + ioff, d.index_data, d.index_count,
                     d.index_type);
  }
  if (nbind) {
    VkBuffer bufs[8];
    VkDeviceSize offs[8];
    for (uint32_t j = 0; j < nbind; j++) {
      bufs[j] = g_ring.vb;
      offs[j] = voff + bind_off[j];
    }
    vkCmdBindVertexBuffers(g_frame.cmd, 0, nbind, bufs, offs);
  }
  if (indexed)
    vkCmdBindIndexBuffer(
        g_frame.cmd, g_ring.ib, ioff,
        d.index_type == 1 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
  if (kDrawTrace && draw_count >= 300) {
    static int n = 0;
    if (n++ < 40)
      std::fprintf(stderr,
                   "[dt] RECOMP DREW count=%u rt=%#lx nv=%u multi_tex=%d\n",
                   draw_count, (unsigned long)d.rt_base, nv, rp->multi_tex);
  }
  CmdInsertLabel(g_frame.cmd, "recomp vs=%#llx ps=%#llx n=%u%s",
                 (unsigned long long)d.vs_addr, (unsigned long long)d.ps_addr,
                 indexed ? d.index_count : d.vertex_count,
                 indexed ? " indexed" : "");
  if (indexed)
    vkCmdDrawIndexed(g_frame.cmd, d.index_count,
                     d.instance_count ? d.instance_count : 1, 0, 0, 0);
  else
    vkCmdDraw(g_frame.cmd, d.vertex_count,
              d.instance_count ? d.instance_count : 1, 0, 0);
  // DELTA_GPU_DRAWRT=<base>: report what was actually issued for one target.
  {
    static const uint64_t kWant = [] {
      const char* e = std::getenv("DELTA_GPU_DRAWRT");
      return e ? std::strtoull(e, nullptr, 0) : 0ull;
    }();
    static int shown = 0;
    // DELTA_GPU_DRAWRT=1 logs EVERY draw (the whole frame graph) instead of one
    // target's draws, so a broken producer/consumer link is visible directly.
    const bool all = kWant == 1;
    // DELTA_GPU_DRAWRT_FRAME=N: only this frame, so the graph is a steady-state
    // frame rather than the opening composites.
    static const int kWantFrame = [] {
      const char* e = std::getenv("DELTA_GPU_DRAWRT_FRAME");
      return e ? std::atoi(e) : 0;
    }();
    if (kWant && (all || g_region.cur_rt == kWant) && shown < (all ? 60 : 8) &&
        (!kWantFrame || (int)g_frame.num == kWantFrame)) {
      shown++;
      std::fprintf(stderr,
                   "[drawrt] rt=%#lx %ux%u indexed=%d vcount=%u icount=%u "
                   "prim=%u tmask=%#x num_vbufs=%u stride=%u mrt=%u "
                   "vp=%g,%g scale %g,%g off vs=%#lx ps=%#lx\n",
                   (unsigned long)g_region.cur_rt, d.rt_w, d.rt_h, (int)indexed,
                   d.vertex_count, d.index_count, d.prim_type, d.target_mask,
                   d.num_vbufs, d.vertex_stride, d.mrt_count,
                   d.viewport_x_scale, d.viewport_y_scale, d.viewport_x_offset,
                   d.viewport_y_offset, (unsigned long)d.vs_addr,
                   (unsigned long)d.ps_addr);
      for (uint32_t i = 0; i < multi_n; i++) {
        const auto& t = d.texs[i];
        if (!t.base)
          continue;
        std::fprintf(stderr,
                     "[drawrt]  tex%u %#lx %ux%u dfmt=%u tiling=%u -> %s%#lx\n",
                     i, (unsigned long)t.base, t.w, t.h, t.dfmt, t.tiling,
                     multi_color[i]      ? "rt "
                     : multi_feedback[i] ? "fb "
                     : multi_depth[i]    ? "depth "
                     : multi_views[i]    ? "guest "
                                         : "MISS ",
                     (unsigned long)(multi_color[i]      ? multi_color[i]
                                     : multi_feedback[i] ? multi_feedback[i]
                                     : multi_depth[i]    ? multi_depth[i]
                                                         : 0));
      }
      for (uint32_t c = 0; c < kCbufBindings; c++) {
        const auto& cb = d.cbufs[c];
        if (!gpu::IsReadableRange(cb.base, std::min(cb.size, 192u)))
          continue;
        const uint32_t* w = reinterpret_cast<const uint32_t*>(cb.base);
        std::fprintf(stderr, "[drawrt]  cb%u %#lx sz=%u:", c,
                     (unsigned long)cb.base, cb.size);
        for (uint32_t k = 0; k < 8 && k * 4 < cb.size; k++)
          std::fprintf(stderr, " %g", *reinterpret_cast<const float*>(&w[k]));
        // The 3D VS loads its transform with s_buffer_load_dwordx16 at byte
        // offset 0x80; show that window when the binding is big enough to hold
        // it.
        if (cb.size >= 192) {
          std::fprintf(stderr, "\n[drawrt]   @0x80:");
          for (uint32_t k = 32; k < 48; k++)
            std::fprintf(stderr, " %g", *reinterpret_cast<const float*>(&w[k]));
        }
        std::fprintf(stderr, "\n");
      }
    }
  }
  for (uint32_t i = 0; i < multi_n; i++) {
    if (!multi_storage[i])
      continue;
    RTarget& target = g_rts[multi_storage[i]];
    target.ever_rendered = true;
    target.used_this_frame = true;
    target.last_frame = g_frame.num;
  }
  g_ring.vb_offset += vneed;
  if (indexed)
    g_ring.ib_offset = ioff + index_bytes;
  g_frame.draws++;
  if (g_region.cur_rt) {
    auto& rt = g_rts[g_region.cur_rt];
    if (++rt.draws > g_region.busiest_rt_draws) {
      g_region.busiest_rt_draws = rt.draws;
      g_region.busiest_rt = g_region.cur_rt;
    }
  }
  return true;
}

}  // namespace gpu::vk
