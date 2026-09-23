#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The Direct3D 12 implementation of rhi::Device.
 *
 * It is written as Windows D3D12 code and compiled as such. On Linux it builds
 * against vkd3d's headers and runs on vkd3d's D3D12-over-Vulkan
 * implementation, which is what lets the backend be developed and tested here
 * rather than only on the machine it eventually ships to.
 *
 * Built only when D3D12 is available (DELTA_HAVE_D3D12); the factory below is
 * declared unconditionally and returns null when it is not.
 */

#include <memory>

#include "gpu/rhi/device.h"

namespace gpu::d3d12 {

// Null when D3D12 was not compiled in, or no device could be created.
std::unique_ptr<rhi::Device> CreateD3D12Device();

}  // namespace gpu::d3d12
