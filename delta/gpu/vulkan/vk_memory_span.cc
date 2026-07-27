/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_memory_span.h"

#include <algorithm>
#include <limits>

namespace gpu::vk {

MemorySpanAllocator::MemorySpanAllocator(uint64_t capacity) {
  Reset(capacity);
}

void MemorySpanAllocator::Reset(uint64_t capacity) {
  free_.clear();
  if (capacity)
    free_.push_back({0, capacity});
}

bool MemorySpanAllocator::Allocate(uint64_t size,
                                   uint64_t alignment,
                                   uint64_t& offset) {
  if (!size || !alignment || (alignment & (alignment - 1)))
    return false;
  for (size_t i = 0; i < free_.size(); i++) {
    const MemorySpan span = free_[i];
    if (span.offset > std::numeric_limits<uint64_t>::max() - alignment + 1)
      continue;
    const uint64_t aligned = (span.offset + alignment - 1) & ~(alignment - 1);
    const uint64_t padding = aligned - span.offset;
    if (padding > span.size || size > span.size - padding)
      continue;
    const uint64_t after_offset = aligned + size;
    const uint64_t after = span.offset + span.size - after_offset;
    free_.erase(free_.begin() + i);
    if (after)
      free_.insert(free_.begin() + i, {after_offset, after});
    if (padding)
      free_.insert(free_.begin() + i, {span.offset, padding});
    offset = aligned;
    return true;
  }
  return false;
}

void MemorySpanAllocator::Free(uint64_t offset, uint64_t size) {
  if (!size || offset > std::numeric_limits<uint64_t>::max() - size)
    return;
  free_.push_back({offset, size});
  std::sort(free_.begin(), free_.end(),
            [](const MemorySpan& a, const MemorySpan& b) {
              return a.offset < b.offset;
            });
  size_t out = 0;
  for (const MemorySpan& span : free_) {
    if (out && free_[out - 1].offset + free_[out - 1].size >= span.offset) {
      const uint64_t end = std::max(free_[out - 1].offset + free_[out - 1].size,
                                    span.offset + span.size);
      free_[out - 1].size = end - free_[out - 1].offset;
    } else {
      free_[out++] = span;
    }
  }
  free_.resize(out);
}

uint64_t MemorySpanAllocator::FreeBytes() const {
  uint64_t bytes = 0;
  for (const MemorySpan& span : free_)
    bytes += span.size;
  return bytes;
}

}  // namespace gpu::vk
