/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Handing a finished frame to the window. gfx::present blocks on the window
// swapchain and, on a software driver, rasterizes the blit on the CPU, so a
// dedicated thread owns the window and always shows the newest complete frame.

#include <cstdint>

namespace gpu::vk {

void PresentAsync(const uint8_t* pixels, uint32_t w, uint32_t h);

}  // namespace gpu::vk
