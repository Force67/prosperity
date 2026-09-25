/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/backend.h"

#include <string>

#include "gpu/vulkan/vk_rhi.h"
#if DELTA_HAVE_D3D12
#include "gpu/d3d12/d3d12_rhi.h"
#endif

namespace gpu::render {

std::vector<rhi::Backend> CompiledBackends() {
  return {
      rhi::Backend::kVulkan,
#if DELTA_HAVE_D3D12
      rhi::Backend::kD3D12,
#endif
  };
}

std::unique_ptr<rhi::Device> CreateBackendDevice(
    rhi::Backend backend,
    const BackendOptions& options) {
  const std::string cache =
      options.cache_dir ? std::string(options.cache_dir) + "/pipeline.bin"
                        : std::string();
  switch (backend) {
    case rhi::Backend::kVulkan: {
      vk::VulkanOptions vo;
      vo.gpu_filter = options.gpu_filter;
      vo.debug_utils = options.debug_labels;
      vo.validation_layer = options.validation_layer;
      vo.sync_validation = options.sync_validation;
      vo.checkpoints = options.checkpoints;
      vo.pipeline_cache_path = cache.empty() ? nullptr : cache.c_str();
      return vk::CreateVulkanDevice(vo);
    }
#if DELTA_HAVE_D3D12
    case rhi::Backend::kD3D12: {
      d3d12::D3D12Options d3;
      d3.gpu_filter = options.gpu_filter;
      d3.debug_layer = options.validation_layer != nullptr;
      d3.debug_labels = options.debug_labels;
      return d3d12::CreateD3D12Device(d3);
    }
#endif
    default:
      return nullptr;
  }
}

}  // namespace gpu::render
