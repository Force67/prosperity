/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Aligned free-span allocation used inside one Vulkan device-memory block.

#include <cstdint>
#include <vector>

namespace gpu::vk {

struct MemorySpan {
  uint64_t offset = 0;
  uint64_t size = 0;
};

class MemorySpanAllocator {
 public:
  explicit MemorySpanAllocator(uint64_t capacity = 0);

  void Reset(uint64_t capacity);
  bool Allocate(uint64_t size, uint64_t alignment, uint64_t& offset);
  void Free(uint64_t offset, uint64_t size);
  uint64_t FreeBytes() const;

 private:
  std::vector<MemorySpan> free_;
};

}  // namespace gpu::vk
