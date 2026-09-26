/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/backend.h"

#include <string>

#include "gpu/vulkan/vk_rhi.h"

namespace gpu::render {

std::vector<rhi::Backend> CompiledBackends() {
  return {rhi::Backend::kVulkan};
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
      vo.validation_layer =
          options.validation ? "VK_LAYER_KHRONOS_validation" : nullptr;
      vo.on_message = options.on_message;
      vo.sync_validation = options.sync_validation;
      vo.checkpoints = options.checkpoints;
      vo.pipeline_cache_path = cache.empty() ? nullptr : cache.c_str();
      return vk::CreateVulkanDevice(vo);
    }
    default:
      return nullptr;
  }
}

}  // namespace gpu::render
