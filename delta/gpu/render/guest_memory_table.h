#pragma once
#include "base/containers/span.h"
#include "base/containers/vector.h"
#include "gpu/render/command.h"
#include "gpu/rhi/device.h"

namespace gpu::render {
struct GuestMemoryWindow {
  u64 base = 0, size = 0, address = 0;
  bool writable = false;
  bool mirror = false;
};
bool PrepareGuestMemory(base::Span<const GuestMemoryWindow> windows,
                        rhi::Buffer*& table,
                        u64& table_bytes,
                        bool writes,
                        u32& set);
base::Vector<u32> TakeWrittenWindows(u32 set);
base::Vector<u64> TakeGuestMisses(u32 set);
void ReleaseGuestMemory(u32 set);
}  // namespace gpu::render
