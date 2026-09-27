/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Aligned free-span allocation used inside one Vulkan device-memory block.

#include "base/arch.h"
#include "base/containers/vector.h"

namespace gpu::vk {

struct MemorySpan {
  u64 offset = 0;
  u64 size = 0;
};

class MemorySpanAllocator {
 public:
  explicit MemorySpanAllocator(u64 capacity = 0);

  void Reset(u64 capacity);
  bool Allocate(u64 size, u64 alignment, u64& offset);
  void Free(u64 offset, u64 size);
  u64 FreeBytes() const;

 private:
  base::Vector<MemorySpan> free_;
};

}  // namespace gpu::vk
