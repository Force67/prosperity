#pragma once
#include <vulkan/vulkan.h>
#include <span>
#include "gpu/render/command.h"

namespace gpu::vk {
// Caller must retire the preceding compute batch before updating this table.
// Guest pointers are translated only within these readable, disjoint spans.
bool PrepareGuestMemory(std::span<const render::GuestMemoryRange> ranges,
                        VkDescriptorBufferInfo& table,
                        bool writes = false);
// Called after the dispatch fence. Copies back only GPU-written dwords and
// returns the ranges whose cached graphics/compute copies must be invalidated.
std::vector<render::GuestMemoryRange> FinishGuestMemoryWrites();
}  // namespace gpu::vk
