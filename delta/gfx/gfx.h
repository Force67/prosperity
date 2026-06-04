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

void shutdown();

}  // namespace gfx
