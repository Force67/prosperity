#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * On-screen overlay content: a keyboard->DualSense legend, a memory-pressure
 * gauge (VRAM + system RAM), and a CPU/GPU utilization gauge. Built as a Dear
 * ImGui draw list here and rendered through a Vulkan pipeline (overlay_vk.cc).
 * Desktop/Linux only; the Android build has its own touch overlay.
 */

#include "base/arch.h"

namespace host {

// Create the ImGui context if needed (safe to call repeatedly).
void OverlayEnsureImGui();

// Per-frame perf stats, pushed by the GPU renderer (delta_gpu).
void OverlaySetPerf(float fps, float gpu_ms, float frame_ms);

// Build the overlay ImDrawData at display size (w by h). vram in bytes (0 =
// unknown). Call once per frame before OverlayVkRender.
void OverlayBuildFrame(u32 w, u32 h, u64 vram_used, u64 vram_total);

// Show/hide the overlay (bound to F1 by the window event pump).
void OverlayToggle();

}  // namespace host
