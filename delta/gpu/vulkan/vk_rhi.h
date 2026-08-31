#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The Vulkan implementation of rhi::Device.
 *
 * It brings up its own instance, device and queue rather than sharing
 * vk_device.cc's `g_dev`: a device you can create two of, in one process, is
 * what lets a test stand one up and what lets the conformance suite run the
 * same work through this backend and through D3D12. The existing renderer
 * still uses `g_dev`; the two meet when it migrates onto this seam.
 */

#include <memory>

#include "base/arch.h"
#include "gpu/rhi/device.h"

namespace gpu::vk {

// Null when no usable Vulkan device exists (no loader, no ICD, or no device
// supporting dynamic rendering and timeline semaphores).
std::unique_ptr<rhi::Device> CreateVulkanDevice();

}  // namespace gpu::vk
