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
// per-frame wall time goes: our GPU code (draw + endFrame, including the
// readback stall and synchronous texture uploads) vs the guest/FEX time
// outside it.
extern uint64_t g_nsDraw, g_nsEnd, g_nsReadback, g_nsTexUp;
extern uint64_t g_nsCs, g_csBytes;
extern uint64_t g_nsCsIn, g_nsCsGpu, g_nsCsOut;
extern uint64_t g_nsSubmit, g_nsPresent;
extern uint32_t g_csCount, g_texUps;
extern uint32_t g_csStageN, g_csFlushN;
extern uint64_t g_csStageBytes;

// Per-frame accumulators (ns), reset when a frame's sample is pushed.
extern uint64_t g_frDraw, g_frSubmit, g_frWait, g_frPresent, g_frTexUp;

inline uint64_t nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct ScopeNs {
  uint64_t t0;
  uint64_t *acc;
  explicit ScopeNs(uint64_t *a) : t0(nowNs()), acc(a) {}
  ~ScopeNs() { *acc += nowNs() - t0; }
};

// Close out this frame's stage sample and start the next.
void pushStageSample();
void drawPerfOverlay(uint8_t *bgra, uint32_t w, uint32_t h);
void reportFps();

}  // namespace gpu::vk
