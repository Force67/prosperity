#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * On-screen overlay content: a keyboard->DualSense legend, a memory-pressure
 * gauge (VRAM + system RAM), and a CPU/GPU utilization gauge. Built as a Dear
 * ImGui draw list here and rendered through a Vulkan pipeline (overlay_vk.cpp).
 * Desktop/Linux only; the Android build has its own touch overlay.
 */

#include <cstdint>

namespace gfx {

// Create the ImGui context if needed (safe to call repeatedly).
void overlayEnsureImGui();

// Per-frame perf stats, pushed by the GPU renderer (delta_gpu).
void overlaySetPerf(float fps, float gpuMs, float frameMs);

// Build the overlay ImDrawData at display size (w by h). vram in bytes (0 =
// unknown). Call once per frame before overlayVkRender.
void overlayBuildFrame(uint32_t w, uint32_t h, uint64_t vramUsed,
                       uint64_t vramTotal);

// Show/hide the overlay (bound to F1 by the window event pump).
void overlayToggle();

}  // namespace gfx
