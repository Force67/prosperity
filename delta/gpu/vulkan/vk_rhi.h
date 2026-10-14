/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include <vulkan/vulkan.h>

#include "base/arch.h"
#include "base/memory/unique_pointer.h"
#include "gpu/rhi/device.h"

namespace gpu::vk {

struct VulkanOptions {
  const char* gpu_filter = nullptr;
  bool debug_utils = false;
  const char* validation_layer = nullptr;
  bool sync_validation = false;
  bool checkpoints = false;
  bool unified_layouts = false;
  const char* pipeline_cache_path = nullptr;
  void (*on_message)(const char*,
                     const char*,
                     const char*,
                     const char*) = nullptr;
};

// Call from a host thread: some ICDs misbehave on guest threads.
base::UniquePointer<rhi::Device> CreateVulkanDevice(
    const VulkanOptions& options);

struct NativeDevice {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  u32 queue_family = 0;
  VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
  u32 timestamp_valid_bits = 0;
  bool push_descriptor = false;
  bool device_fault = false;
  bool checkpoints = false;
  VkPhysicalDeviceMeshShaderPropertiesEXT mesh_limits{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
};
const NativeDevice& Native(rhi::Device* device);
VkBuffer Native(rhi::Buffer* buffer);
VkImage Native(rhi::Texture* texture);
VkImageView Native(rhi::TextureView* view);
VkSampler Native(rhi::Sampler* sampler);
VkDescriptorSetLayout Native(rhi::BindGroupLayout* layout);
VkDescriptorSet Native(rhi::BindGroup* group);
VkPipelineLayout Native(rhi::PipelineLayout* layout);
VkPipeline Native(rhi::Pipeline* pipeline);
VkCommandBuffer Native(rhi::CommandList* list);

VkFormat ToVkFormat(rhi::Format format);
VkImageLayout ToVkLayout(rhi::TextureState state,
                         u8 aspect,
                         bool exported = false);
VkImageLayout FixedLayout(rhi::Texture* texture, VkImageLayout optimal);
extern bool g_unified_layouts;
inline bool Exported(rhi::Texture* texture) {
  return texture->desc().usage & rhi::kTextureExportable;
}

}  // namespace gpu::vk
