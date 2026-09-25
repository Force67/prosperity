#pragma once
#include <span>
#include "gpu/render/command.h"
#include "gpu/rhi/device.h"

namespace gpu::vk {
// Caller must retire the preceding compute batch before updating this table.
// Guest pointers are translated only within these readable, disjoint spans.
// `table` and `table_bytes` are the translation table the shader binds.
bool PrepareGuestMemory(std::span<const render::GuestMemoryRange> ranges,
                        rhi::Buffer*& table,
                        u64& table_bytes,
                        bool writes = false);
// Called after the dispatch fence. Copies back only GPU-written dwords and
// returns the ranges whose cached graphics/compute copies must be invalidated.
std::vector<render::GuestMemoryRange> FinishGuestMemoryWrites();
}  // namespace gpu::vk
