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
// Idempotent: returns true immediately if a window already exists.
bool init(const char *title, uint32_t width, uint32_t height);

// Idempotent bring-up for the presenting thread: create the window on the first
// call, then report availability. Stops retrying after a failed attempt.
bool ensure(const char *title, uint32_t width, uint32_t height);

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

// Drive haptics on the active controller. large/small are the DS4 motor
// intensities (0..255). Routed to SDL gamepad rumble (PC) or the device
// vibrator (Android); a no-op when no haptic device is present.
void setRumble(uint8_t largeMotor, uint8_t smallMotor);

// Harness signal shared between the GPU renderer and the input layer. The renderer
// raises it once sustained gameplay (room rendering) is on screen, so the headless
// autoskip (DELTA_PAD_AUTOSKIP) stops pressing menu buttons and stays in the run
// instead of bouncing back out through the pause menu. Latches on (a run started).
void setInGameplay(bool v);
bool inGameplay();

void shutdown();

}  // namespace gfx
