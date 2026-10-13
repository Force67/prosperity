#pragma once

#include "base/arch.h"
#include "base/memory/unique_pointer.h"
#include "gpu/rhi/device.h"

namespace gpu::d3d12 {

struct D3D12Options {
  const char* gpu_filter = nullptr;
  bool debug_layer = false;
  bool debug_labels = false;
};

base::UniquePointer<rhi::Device> CreateD3D12Device(const D3D12Options& options);

}  // namespace gpu::d3d12
