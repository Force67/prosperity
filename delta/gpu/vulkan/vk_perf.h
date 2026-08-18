/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Where a frame's wall time goes: the per-window and per-frame stage
// accumulators the renderer feeds, the periodic FPS report, and the on-screen
// stacked-column overlay that draws the history into the presented image.

#include <chrono>
#include "base/arch.h"

namespace gpu::vk {

// Window accumulators (ns), reset by each FPS report. Reveals where the
// per-frame wall time goes: our GPU code (draw + EndFrame, including the
// readback stall and synchronous texture uploads) vs the guest/FEX time
// outside it.
extern u64 g_ns_draw, g_ns_end, g_ns_readback, g_ns_tex_up;
extern u64 g_ns_cs, g_cs_bytes;
extern u64 g_ns_cs_in, g_ns_cs_gpu, g_ns_cs_out;
extern u64 g_ns_submit, g_ns_present;
extern u64 g_ns_gpu_exec;
extern u32 g_cs_count, g_tex_ups;
// Draws the renderer was handed this window, and how many it declined. A
// per-frame cost only means something next to the count it is spread over.
extern u64 g_win_draws, g_win_declines;
// DrawRecomp split into the four decisions it makes per draw, so the one that
// costs is named rather than guessed at. DELTA_GPU_DRAWPROF=1 reports them.
extern u64 g_ns_dr_pre, g_ns_dr_pipe, g_ns_dr_tex, g_ns_dr_bind;
// The two candidates inside the texture bind: re-hashing guest content to see
// whether it changed, and probing that the range is mapped.
extern u64 g_ns_tex_hash, g_tex_hash_bytes, g_ns_tex_probe;
extern u64 g_tex_hash_n, g_tex_probe_n, g_ns_tex_lookup, g_tex_lookup_n;
extern u64 g_ns_tex_set, g_tex_set_n, g_ns_region, g_ns_cs_flush;
// Turning a draw packet's registers into a DrawInfo.
extern u64 g_ns_build_draw, g_build_draw_n;
// The present path: the presenter thread's own time, and what the frame loop
// waits for it before reusing the scanout buffer it lent.
extern u64 g_ns_gfx_present, g_ns_borrow_wait;
extern u32 g_gpu_exec_samples;
extern u32 g_cs_stage_n, g_cs_flush_n;
extern u64 g_cs_stage_bytes;
// Compute writeback coverage: how much of what we copy back to guest memory the
// dispatch actually changed. The gap is memory the CPU owns and we were
// reverting -- see CsRangeFlushOne.
extern u64 g_cs_wb_bytes_written, g_cs_wb_bytes_total;

// Submit+wait round trips a frame, broken down by what asked for each.
void CsSyncReport(double frames);

// Per-frame accumulators (ns), reset when a frame's sample is pushed.
extern u64 g_fr_draw, g_fr_submit, g_fr_wait, g_fr_present, g_fr_tex_up;

inline u64 NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct ScopeNs {
  u64 t0;
  u64* acc;
  explicit ScopeNs(u64* a) : t0(NowNs()), acc(a) {}
  ~ScopeNs() { *acc += NowNs() - t0; }
};

// Close out this frame's stage sample and start the next.
void PushStageSample();
// `rgba` when the buffer is in R,G,B,A byte order rather than the scanout
// default B,G,R,A.
void DrawPerfOverlay(u8* pixels, u32 w, u32 h, bool rgba = false);
void ReportFps();

}  // namespace gpu::vk
