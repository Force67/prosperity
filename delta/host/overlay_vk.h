#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Vulkan backend for the Dear ImGui overlay. Renders the overlay's ImDrawData
 * into the swapchain image with a small dedicated pipeline (embedded SPIR-V),
 * replacing the earlier software rasteriser. Driven by window_sdl.cc; the overlay
 * content + metrics live in overlay.cc.
 */

#include "base/arch.h"

#include <vulkan/vulkan.h>
#include <base/containers/vector.h>

namespace host {

bool overlayVkInit(VkPhysicalDevice phys, VkDevice device, VkQueue queue,
                   u32 queueFamily, VkCommandPool pool, VkFormat swapFormat);
void overlayVkSetSwapchain(const base::Vector<VkImage> &images, VkExtent2D extent,
                           VkFormat format);
// Records the overlay render pass into `cmd` for swapchain image `imageIndex`.
// The overlay frame must have been built (overlayBuildFrame) first. Returns true
// if it recorded a pass leaving the image in PRESENT_SRC (caller then skips its
// own transition); false if nothing was drawn.
bool overlayVkRender(VkCommandBuffer cmd, u32 imageIndex);
void overlayVkShutdown();
bool overlayVkReady();

}  // namespace host
