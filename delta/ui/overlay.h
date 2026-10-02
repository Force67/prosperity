#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Desktop controls, log, and busy overlays, plus home-screen frame building.
 * The Vulkan UI backend renders the resulting ImGui draw data.
 */

#include "base/arch.h"

namespace ui {

// Create the ImGui context if needed (safe to call repeatedly).
void OverlayEnsureImGui();
void OverlayShutdownImGui();

// Per-frame perf stats, pushed by the GPU renderer (delta_gpu).
void OverlaySetPerf(float fps, float gpu_ms, float frame_ms);

// Build the overlay ImDrawData at display size (w by h). vram in bytes (0 =
// unknown). Call once per frame before OverlayVkRender.
void OverlayBuildFrame(u32 w,
                       u32 h,
                       u64 vram_used,
                       u64 vram_total,
                       bool frame_stalled = false);

// Show/hide the overlay (bound to F1 by the window event pump).
void OverlayToggle();

}  // namespace ui
