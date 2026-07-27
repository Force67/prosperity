/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Device-local memory suballocation for optimal images. Small images share
// blocks; images requiring dedicated memory retain an exact allocation.

#include <vulkan/vulkan.h>

#include <cstdint>

namespace gpu::vk {

struct ImageAllocation {
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  VkDeviceSize size = 0;
  uint32_t memory_type = 0;
  bool dedicated = false;
};

bool AllocateImageMemory(VkImage image, ImageAllocation& allocation);
void FreeImageMemory(ImageAllocation& allocation);

}  // namespace gpu::vk
