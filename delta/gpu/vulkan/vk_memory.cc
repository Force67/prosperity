/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_memory.h"

#include "gpu/gpu_check.h"
#include "gpu/vulkan/vk_device.h"

#include <utility>

namespace gpu::vk {

bool ImageMemoryPool::AllocateFromBlock(Block& block,
                                        VkDeviceSize size,
                                        VkDeviceSize alignment,
                                        ImageAllocation& allocation) {
  uint64_t offset = 0;
  if (!block.spans.Allocate(size, alignment, offset))
    return false;
  allocation = {block.memory, offset, size, block.memory_type, false};
  return true;
}

bool ImageMemoryPool::Allocate(const DeviceState& device,
                               VkImage image,
                               ImageAllocation& allocation) {
  if (!device.device || (device_ && device_ != device.device))
    return false;
  device_ = device.device;
  VkMemoryDedicatedRequirements dedicated{
      VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
  VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
  requirements.pNext = &dedicated;
  VkImageMemoryRequirementsInfo2 info{
      VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
  info.image = image;
  vkGetImageMemoryRequirements2(device.device, &info, &requirements);
  const VkMemoryRequirements& memory = requirements.memoryRequirements;
  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(device.phys, &properties);
  GPU_BUGCHECK(memory.memoryTypeBits != 0,
               "image reports no allowed memory types");
  uint32_t memory_type = 0;
  bool found_type = false;
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
    if ((memory.memoryTypeBits & (1u << i)) &&
        (properties.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      memory_type = i;
      found_type = true;
      break;
    }
  // No device-local type accepts this image (integrated / exotic heaps): any
  // allowed type beats the old silent default of index 0, which may not even
  // be in memoryTypeBits.
  if (!found_type)
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
      if (memory.memoryTypeBits & (1u << i)) {
        memory_type = i;
        break;
      }
  constexpr VkDeviceSize kBlockSize = 256ull * 1024 * 1024;
  const bool use_dedicated =
      dedicated.requiresDedicatedAllocation || memory.size > kBlockSize / 2;

  if (!use_dedicated) {
    // A bind failure against pooled memory releases the span and falls
    // through to a dedicated allocation instead of failing the image: the
    // driver may accept the same image with its own memory. One pooled bind
    // failure skips the fresh-block attempt too -- another suballocation is
    // no more likely to bind.
    bool pooled_bind_failed = false;
    for (auto& block : blocks_) {
      if (block.memory_type != memory_type ||
          !AllocateFromBlock(block, memory.size, memory.alignment, allocation))
        continue;
      if (vkBindImageMemory(device.device, image, allocation.memory,
                            allocation.offset) == VK_SUCCESS)
        return true;
      Free(device, allocation);
      pooled_bind_failed = true;
      break;
    }

    if (!pooled_bind_failed) {
      Block block;
      block.size = kBlockSize;
      block.memory_type = memory_type;
      VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      ai.allocationSize = block.size;
      ai.memoryTypeIndex = memory_type;
      if (vkAllocateMemory(device.device, &ai, nullptr, &block.memory) ==
          VK_SUCCESS) {
        block.spans.Reset(block.size);
        blocks_.push_back(std::move(block));
        // A fresh empty block always has room for a <= kBlockSize/2 image.
        const bool suballocated = AllocateFromBlock(
            blocks_.back(), memory.size, memory.alignment, allocation);
        GPU_BUGCHECK(suballocated, "suballocation failed in a fresh block");
        if (vkBindImageMemory(device.device, image, allocation.memory,
                              allocation.offset) == VK_SUCCESS)
          return true;
        Free(device, allocation);
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
  if (vkAllocateMemory(device.device, &ai, nullptr, &device_memory) !=
      VK_SUCCESS)
    return false;
  allocation = {device_memory, 0, memory.size, memory_type, true};
  if (vkBindImageMemory(device.device, image, device_memory, 0) != VK_SUCCESS) {
    vkFreeMemory(device.device, device_memory, nullptr);
    allocation = {};
    return false;
  }
  dedicated_allocations_.insert(device_memory);
  return true;
}

void ImageMemoryPool::Free(const DeviceState& device,
                           ImageAllocation& allocation) {
  if (!allocation.memory || device.device != device_)
    return;
  if (allocation.dedicated) {
    const auto found = dedicated_allocations_.find(allocation.memory);
    GPU_BUGCHECK(found != dedicated_allocations_.end(),
                 "dedicated allocation freed twice or never allocated");
    vkFreeMemory(device.device, allocation.memory, nullptr);
    dedicated_allocations_.erase(found);
  } else {
    bool found = false;
    for (auto& block : blocks_)
      if (block.memory == allocation.memory) {
        block.spans.Free(allocation.offset, allocation.size);
        found = true;
        break;
      }
    GPU_BUGCHECK(found, "pooled allocation freed against an unknown block");
  }
  allocation = {};
}

}  // namespace gpu::vk
