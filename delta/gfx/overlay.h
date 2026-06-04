#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Small on-screen legend mapping keyboard keys to DualSense buttons, drawn with
 * a raw Dear ImGui draw list and software-rasterised into the present
 * framebuffer (the same CPU-blend hook the Android gamepad overlay uses).
 * Desktop/Linux only; the Android build has its own touch overlay.
 */

#include <cstdint>

namespace gfx {

// Composite the controls legend into a CPU RGBA/BGRA8 framebuffer (w by h, tight
// w*4 pitch). No-op while hidden. `bgra` selects the channel order.
void overlayDraw(uint8_t *fb, uint32_t w, uint32_t h, bool bgra);

// Show/hide the legend (bound to F1 by the window event pump).
void overlayToggle();

}  // namespace gfx
