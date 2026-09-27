/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_memory_span.h"
#include "base/arch.h"

#include "gpu/gpu_check.h"

#include <limits>
#include <base/algorithm.h>
#include <base/math/value_bounds.h>

namespace gpu::vk {

MemorySpanAllocator::MemorySpanAllocator(u64 capacity) {
  Reset(capacity);
}

void MemorySpanAllocator::Reset(u64 capacity) {
  free_.clear();
  if (capacity)
    free_.push_back({0, capacity});
}

bool MemorySpanAllocator::Allocate(u64 size,
                                   u64 alignment,
                                   u64& offset) {
  if (!size || !alignment || (alignment & (alignment - 1)))
    return false;
  for (size_t i = 0; i < free_.size(); i++) {
    const MemorySpan span = free_[i];
    if (span.offset > std::numeric_limits<u64>::max() - alignment + 1)
      continue;
    const u64 aligned = (span.offset + alignment - 1) & ~(alignment - 1);
    const u64 padding = aligned - span.offset;
    if (padding > span.size || size > span.size - padding)
      continue;
    const u64 after_offset = aligned + size;
    const u64 after = span.offset + span.size - after_offset;
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

void MemorySpanAllocator::Free(u64 offset, u64 size) {
  if (!size || offset > std::numeric_limits<u64>::max() - size)
    return;
  free_.push_back({offset, size});
  base::Sort(free_.begin(), free_.end(),
            [](const MemorySpan& a, const MemorySpan& b) {
              return a.offset < b.offset;
            });
  size_t out = 0;
  for (const MemorySpan& span : free_) {
    // Strict overlap between free spans means the same bytes were freed
    // twice (or a free span was allocated over): the allocator's map is
    // corrupt and every later Allocate may hand out live memory.
    GPU_BUGCHECK(!out ||
                     free_[out - 1].offset + free_[out - 1].size <= span.offset,
                 "double free: span overlaps an already-free span");
    if (out && free_[out - 1].offset + free_[out - 1].size >= span.offset) {
      const u64 end = base::Max(free_[out - 1].offset + free_[out - 1].size,
                                    span.offset + span.size);
      free_[out - 1].size = end - free_[out - 1].offset;
    } else {
      free_[out++] = span;
    }
  }
  free_.resize(out);
}

u64 MemorySpanAllocator::FreeBytes() const {
  u64 bytes = 0;
  for (const MemorySpan& span : free_)
    bytes += span.size;
  return bytes;
}

}  // namespace gpu::vk
