/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Where a frame's wall time goes: the per-window and per-frame stage
// accumulators the renderer feeds, the periodic FPS report, and the on-screen
// stacked-column overlay that draws the history into the presented image.

#include <chrono>
#include <cstdint>

namespace gpu::vk {

// Window accumulators (ns), reset by each FPS report. Reveals where the
// per-frame wall time goes: our GPU code (draw + EndFrame, including the
// readback stall and synchronous texture uploads) vs the guest/FEX time
// outside it.
extern uint64_t g_ns_draw, g_ns_end, g_ns_readback, g_ns_tex_up;
extern uint64_t g_ns_cs, g_cs_bytes;
extern uint64_t g_ns_cs_in, g_ns_cs_gpu, g_ns_cs_out;
extern uint64_t g_ns_submit, g_ns_present;
extern uint64_t g_ns_gpu_exec;
extern uint32_t g_cs_count, g_tex_ups;
extern uint32_t g_gpu_exec_samples;
extern uint32_t g_cs_stage_n, g_cs_flush_n;
extern uint64_t g_cs_stage_bytes;
// Compute writeback coverage: how much of what we copy back to guest memory the
// dispatch actually changed. The gap is memory the CPU owns and we were
// reverting -- see CsRangeFlushOne.
extern uint64_t g_cs_wb_bytes_written, g_cs_wb_bytes_total;

// Per-frame accumulators (ns), reset when a frame's sample is pushed.
extern uint64_t g_fr_draw, g_fr_submit, g_fr_wait, g_fr_present, g_fr_tex_up;

inline uint64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct ScopeNs {
  uint64_t t0;
  uint64_t* acc;
  explicit ScopeNs(uint64_t* a) : t0(NowNs()), acc(a) {}
  ~ScopeNs() { *acc += NowNs() - t0; }
};

// Close out this frame's stage sample and start the next.
void PushStageSample();
void DrawPerfOverlay(uint8_t* bgra, uint32_t w, uint32_t h);
void ReportFps();

}  // namespace gpu::vk
