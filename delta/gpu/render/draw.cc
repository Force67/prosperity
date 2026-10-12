/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// The renderer's draw entry point: applies the diagnostic draw knobs and runs
// the draw through the recompiled guest shaders.

#include "base/arch.h"
#include "gpu/render/renderer.h"

#include "gpu/gpu_perf.h"
#include "gpu/guest_memory.h"
#include "gpu/render/device.h"
#include "gpu/render/draw_recomp.h"
#include "gpu/render/frame.h"
#include "gpu/render/guest_format.h"
#include "gpu/render/labels.h"
#include "gpu/render/pipeline_cache.h"
#include "gpu/render/render_target.h"
#include "gpu/render/texture_cache.h"
#include "gpu/render/trace.h"
#include "gpu/render/upload_ring.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "base/algorithm.h"
#include "base/logging.h"
#include "base/memory/unique_pointer.h"
#include "options/options.h"
#include "profile/profile.h"

namespace {
DELTA_OPTION(int, kMaxDraw, "DELTA_GPU_MAXDRAW", -1);
DELTA_OPTION(int, kOnlyDraw, "DELTA_GPU_ONLYDRAW", -1);
DELTA_OPTION(u32, kOnlyIc, "DELTA_GPU_ONLYIC", 0);
DELTA_OPTION(bool, kDrawTraceAll, "DELTA_GPU_DRAWTRACE", false);
DELTA_OPTION(bool, kNoCull, "DELTA_GPU_NOCULL", false);
DELTA_OPTION(bool, kNoDepth, "DELTA_GPU_NODEPTH", false);
DELTA_OPTION(bool, kNoMask, "DELTA_GPU_NOMASK", false);
DELTA_OPTION(bool, kSwapTex, "DELTA_GPU_SWAPTEX01", false);
}  // namespace

namespace gpu::render {

void Draw(Renderer& renderer, const DrawInfo& d_in) {
  DELTA_ZONE("gpu.draw");
  if (!g_frame.recording)
    return;
  // DELTA_GPU_SWAPTEX01: bisect a suspected sampler-binding order mismatch by
  // exchanging the first two textures of every multi-texture draw.
  base::UniquePointer<DrawInfo> swapped;
  if (kSwapTex && d_in.num_texs >= 2) {
    swapped = base::MakeUnique<DrawInfo>(d_in);
    base::Swap(swapped->texs[0], swapped->texs[1]);
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
  base::UniquePointer<DrawInfo> patched_copy;
  const bool patched = kNoDepth || kNoCull || kNoMask || detach_depth;
  if (patched) {
    patched_copy = base::MakeUnique<DrawInfo>(d_sw);
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
  if (ShaderFilterDrops(d.ps_addr)) {
    g_frame.draws++;
    return;
  }
  const bool recompiled = d.recomp && DrawRecomp(renderer, d);
  if (!renderer.available())
    return;
  if (kDrawTraceAll) {
    static u32 traced = 0;
    if (traced++ < 100)
      BASE_LOGI("dt",
                "f{} rt={:#x} count={} indexed={} nv={} mrt={} mask={:#x} "
                "psmask={:#x} prim={} vp=[{:.1f} {:.1f} {:.1f} {:.1f}] "
                "depth={:#x} handled={:d}",
                g_frame.num, (unsigned long)d.rt_base, d.vertex_count,
                d.index_count, d.num_vattrs, d.mrt_count, d.target_mask,
                d.recomp ? d.recomp->ps_mrt_mask : 0, d.prim_type,
                d.viewport_x_scale, d.viewport_x_offset, d.viewport_y_scale,
                d.viewport_y_offset, (unsigned long)d.depth_base, recompiled);
  }
}

}  // namespace gpu::render
