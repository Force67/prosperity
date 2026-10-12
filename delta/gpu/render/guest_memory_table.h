#pragma once
#include "base/containers/span.h"
#include "base/containers/vector.h"
#include "gpu/render/command.h"
#include "gpu/rhi/device.h"

namespace gpu::render {
// A guest address window a pointer-chasing shader reads through the buffer
// that holds its current bytes: a compute range's own buffer, or a copy made
// for the shader (an image tiled into guest layout, `mirror`). Addresses
// outside every window translate to nothing, as a buffer cache page without a
// buffer does.
struct GuestMemoryWindow {
  u64 base = 0, size = 0, address = 0;
  bool writable = false;
  bool mirror = false;
};
// `windows` are sorted and disjoint. `table` and `table_bytes` are the
// translation table the shader binds, out of `set`, which stays the
// dispatch's until ReleaseGuestMemory.
bool PrepareGuestMemory(base::Span<const GuestMemoryWindow> windows,
                        rhi::Buffer*& table,
                        u64& table_bytes,
                        bool writes,
                        u32& set);
// After the dispatch fence: which windows the shader stored into, by index
// into that PrepareGuestMemory's `windows`.
base::Vector<u32> TakeWrittenWindows(u32 set);
// After the dispatch fence: guest addresses the shader reached outside every
// window (up to 63).
base::Vector<u64> TakeGuestMisses(u32 set);
void ReleaseGuestMemory(u32 set);
}  // namespace gpu::render
