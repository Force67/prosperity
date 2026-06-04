#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdint>

// SDL3 window backed by a Vulkan swapchain. present() uploads a CPU framebuffer
// and blits it to the swapchain, scaling to the window size. The VideoOut flip
// path (/dev/dce) drives present() with the guest scanout framebuffer.
namespace gfx {

enum class PixelFormat {
  rgba8,  // R8G8B8A8 unorm, byte order R,G,B,A
  bgra8,  // B8G8R8A8 unorm (PS4 scanout default)
};

// Create the window, Vulkan device and swapchain. Returns false on failure.
bool init(const char *title, uint32_t width, uint32_t height);

// True once a window + swapchain exist (init succeeded and not shut down).
bool available();

// Upload pixels (w by h, srcPitch bytes per row, 0 means w*4) and present them,
// scaling to the current window size.
void present(const void *pixels, uint32_t w, uint32_t h, uint32_t srcPitch = 0,
             PixelFormat fmt = PixelFormat::rgba8);

// Drain window events. Returns false once the user asks to close the window.
bool pumpEvents();

// Keyboard-to-gamepad state (an optional input adapter). Buttons are booleans
// and sticks are 0..255 with 128 centred. Maps a WASD/arrows layout to a DS4.
struct PadKeys {
  bool cross = false, circle = false, square = false, triangle = false;
  bool up = false, down = false, left = false, right = false;
  bool l1 = false, r1 = false, l2 = false, r2 = false;
  bool options = false, touchpad = false;
  uint8_t lx = 128, ly = 128, rx = 128, ry = 128;
};
// Fill `out` from the current keyboard state. Returns false if no window exists.
bool pollKeyboardPad(PadKeys &out);

void shutdown();

}  // namespace gfx
