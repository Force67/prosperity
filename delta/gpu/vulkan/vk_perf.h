/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Reporting for the counters in gpu/gpu_perf.h: the periodic FPS line, the
// submit/wait census, and the on-screen stacked-column overlay drawn into the
// presented image. It reads counters the whole module writes, so it lives on
// the frame loop's side of the module rather than at its root.

#include "base/arch.h"

namespace gpu::vk {

// Submit+wait round trips a frame, broken down by what asked for each.
void CsSyncReport(double frames);
// Close out this frame's stage sample and start the next.
void PushStageSample();
// `rgba` when the buffer is in R,G,B,A byte order rather than the scanout
// default B,G,R,A.
void DrawPerfOverlay(u8* pixels, u32 w, u32 h, bool rgba = false);
void ReportFps();

}  // namespace gpu::vk
