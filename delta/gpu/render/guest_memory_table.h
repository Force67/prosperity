#pragma once
#include "base/containers/span.h"
#include "base/containers/vector.h"
#include "gpu/render/command.h"
#include "gpu/rhi/device.h"

namespace gpu::render {
// Caller must retire the preceding compute batch before updating this table.
// Guest pointers are translated only within these readable, disjoint spans.
// `table` and `table_bytes` are the translation table the shader binds.
bool PrepareGuestMemory(base::Span<const render::GuestMemoryRange> ranges,
                        rhi::Buffer*& table,
                        u64& table_bytes,
                        bool writes = false);
// Called after the dispatch fence. Copies back only GPU-written dwords and
// returns the ranges whose cached graphics/compute copies must be invalidated.
base::Vector<render::GuestMemoryRange> FinishGuestMemoryWrites();
}  // namespace gpu::render
