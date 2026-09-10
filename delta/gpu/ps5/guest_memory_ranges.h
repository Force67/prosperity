#pragma once
#include <vector>
#include "gpu/rhi/command.h"

namespace gpu::ps5 {
// Readable mappings inside the GPU pools, plus mappings containing explicit
// raw-resource addresses outside those pools (for example module constants).
std::vector<rhi::GuestMemoryRange> GuestMemoryRanges(
    const std::vector<u64>& addresses);
}  // namespace gpu::ps5
