/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Device-local memory suballocation for optimal images. Small images share
// blocks; images requiring dedicated memory retain an exact allocation.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "gpu/vulkan/vk_memory_span.h"

namespace gpu::vk {

struct DeviceState;

struct ImageAllocation {
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  VkDeviceSize size = 0;
  uint32_t memory_type = 0;
  bool dedicated = false;
};

class ImageMemoryPool {
 public:
  ImageMemoryPool() = default;
  ImageMemoryPool(const ImageMemoryPool&) = delete;
  ImageMemoryPool& operator=(const ImageMemoryPool&) = delete;

  bool Allocate(const DeviceState& device,
                VkImage image,
                ImageAllocation& allocation);
  void Free(const DeviceState& device, ImageAllocation& allocation);

 private:
  struct Block {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    uint32_t memory_type = 0;
    MemorySpanAllocator spans;
  };

  static bool AllocateFromBlock(Block& block,
                                VkDeviceSize size,
                                VkDeviceSize alignment,
                                ImageAllocation& allocation);
  VkDevice device_ = VK_NULL_HANDLE;
  std::unordered_set<VkDeviceMemory> dedicated_allocations_;
  std::vector<Block> blocks_;
};

// Transitional alias into rhi::BackendState while callers are migrated to
// receive the image pool explicitly.
extern ImageMemoryPool& g_image_memory;

}  // namespace gpu::vk
