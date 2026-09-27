#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The Direct3D 12 implementation of rhi::Device.
 *
 * It is Windows D3D12 code. On Linux it builds against vkd3d's headers and
 * runs on vkd3d's D3D12-over-Vulkan implementation, so the backend is tested
 * where it is developed. Shaders arrive as SPIR-V; they are lowered to HLSL
 * with SPIRV-Cross and compiled to DXIL with DXC when a pipeline is created.
 */


#include "base/arch.h"
#include "gpu/rhi/device.h"
#include <base/memory/unique_pointer.h>

namespace gpu::d3d12 {

struct D3D12Options {
  // Substring of the adapter name to prefer. Windows only: vkd3d takes its
  // Vulkan device from VKD3D_VULKAN_DEVICE.
  const char* gpu_filter = nullptr;
  bool debug_layer = false;
  bool debug_labels = false;  // PIX events and object names
};

// Null when D3D12 is not compiled in, no device exists, or DXC cannot be
// loaded. Call from a plain host thread.
base::UniquePointer<rhi::Device> CreateD3D12Device(const D3D12Options& options);

}  // namespace gpu::d3d12
