/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The Vulkan implementation of rhi::Device.

#include <vulkan/vulkan.h>


#include "base/arch.h"
#include "gpu/rhi/device.h"
#include <base/memory/unique_pointer.h>

namespace gpu::vk {

struct VulkanOptions {
  // Substring of the adapter name to prefer (DELTA_VK_GPU).
  const char* gpu_filter = nullptr;
  bool debug_utils = false;  // names + labels for capture tools
  const char* validation_layer = nullptr;  // enable this layer when present
  bool sync_validation = false;
  bool checkpoints = false;  // VK_NV_device_diagnostic_checkpoints
  // Where the driver's pipeline cache blob persists; empty disables it.
  const char* pipeline_cache_path = nullptr;
  // Receives validation-layer messages: severity, id, label stack, text.
  void (*on_message)(const char*, const char*, const char*, const char*) =
      nullptr;
};

// Null when no usable device exists. Call from a plain host thread: some ICDs
// misbehave on a guest thread.
base::UniquePointer<rhi::Device> CreateVulkanDevice(const VulkanOptions& options);

// The native objects behind the abstraction, for code that has not moved onto
// it yet. Each takes an object the Vulkan device created.
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
// Queue access for code that submits outside Device::Submit.
void LockQueue(rhi::Device* device);
void UnlockQueue(rhi::Device* device);

VkFormat ToVkFormat(rhi::Format format);
rhi::Format FromVkFormat(VkFormat format);
VkImageLayout ToVkLayout(rhi::TextureState state, u8 aspect);
rhi::TextureState FromVkLayout(VkImageLayout layout);

}  // namespace gpu::vk
