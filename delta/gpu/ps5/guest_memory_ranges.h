#pragma once
#include "base/containers/vector.h"
#include "gpu/render/command.h"

namespace gpu::ps5 {
// Readable mappings inside the GPU pools, plus mappings containing explicit
// raw-resource addresses outside those pools (for example module constants).
base::Vector<render::GuestMemoryRange> GuestMemoryRanges(
    const base::Vector<u64>& addresses);
}  // namespace gpu::ps5
