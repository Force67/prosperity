/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_memory.h"

#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_memory_span.h"

#include <vector>

namespace gpu::vk {
namespace {

struct ImageBlock {
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  uint32_t memory_type = 0;
  MemorySpanAllocator spans;
};

std::vector<ImageBlock> g_image_blocks;

bool AllocateFromBlock(ImageBlock& block,
                       VkDeviceSize size,
                       VkDeviceSize alignment,
                       ImageAllocation& allocation) {
  uint64_t offset = 0;
  if (!block.spans.Allocate(size, alignment, offset))
    return false;
  allocation = {block.memory, offset, size, block.memory_type, false};
  return true;
}

}  // namespace

bool AllocateImageMemory(VkImage image, ImageAllocation& allocation) {
  VkMemoryDedicatedRequirements dedicated{
      VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
  VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
  requirements.pNext = &dedicated;
  VkImageMemoryRequirementsInfo2 info{
      VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
  info.image = image;
  vkGetImageMemoryRequirements2(g_dev.device, &info, &requirements);
  const VkMemoryRequirements& memory = requirements.memoryRequirements;
  const uint32_t memory_type = FindMemoryType(
      memory.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  constexpr VkDeviceSize kBlockSize = 256ull * 1024 * 1024;
  const bool use_dedicated =
      dedicated.requiresDedicatedAllocation || memory.size > kBlockSize / 2;

  if (!use_dedicated) {
    for (auto& block : g_image_blocks) {
      if (block.memory_type != memory_type ||
          !AllocateFromBlock(block, memory.size, memory.alignment, allocation))
        continue;
      if (vkBindImageMemory(g_dev.device, image, allocation.memory,
                            allocation.offset) == VK_SUCCESS)
        return true;
      FreeImageMemory(allocation);
      return false;
    }

    ImageBlock block;
    block.size = kBlockSize;
    block.memory_type = memory_type;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = block.size;
    ai.memoryTypeIndex = memory_type;
    if (vkAllocateMemory(g_dev.device, &ai, nullptr, &block.memory) ==
        VK_SUCCESS) {
      block.spans.Reset(block.size);
      g_image_blocks.push_back(std::move(block));
      if (AllocateFromBlock(g_image_blocks.back(), memory.size,
                            memory.alignment, allocation)) {
        if (vkBindImageMemory(g_dev.device, image, allocation.memory,
                              allocation.offset) == VK_SUCCESS)
          return true;
        FreeImageMemory(allocation);
        return false;
      }
    }
  }

  VkMemoryDedicatedAllocateInfo dedicated_info{
      VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
  dedicated_info.image = image;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.pNext = &dedicated_info;
  ai.allocationSize = memory.size;
  ai.memoryTypeIndex = memory_type;
  VkDeviceMemory device_memory = VK_NULL_HANDLE;
  if (vkAllocateMemory(g_dev.device, &ai, nullptr, &device_memory) !=
      VK_SUCCESS)
    return false;
  allocation = {device_memory, 0, memory.size, memory_type, true};
  if (vkBindImageMemory(g_dev.device, image, device_memory, 0) != VK_SUCCESS) {
    vkFreeMemory(g_dev.device, device_memory, nullptr);
    allocation = {};
    return false;
  }
  return true;
}

void FreeImageMemory(ImageAllocation& allocation) {
  if (!allocation.memory)
    return;
  if (allocation.dedicated) {
    vkFreeMemory(g_dev.device, allocation.memory, nullptr);
  } else {
    for (auto& block : g_image_blocks)
      if (block.memory == allocation.memory) {
        block.spans.Free(allocation.offset, allocation.size);
        break;
      }
  }
  allocation = {};
}

}  // namespace gpu::vk
