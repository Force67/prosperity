#include "gpu/vulkan/vk_memory_span.h"

#include <gtest/gtest.h>

namespace gpu::vk {
namespace {

TEST(MemorySpanAllocator, AllocatesAlignedSpans) {
  MemorySpanAllocator allocator(1024);
  uint64_t first = 0, second = 0;
  ASSERT_TRUE(allocator.Allocate(3, 1, first));
  ASSERT_TRUE(allocator.Allocate(64, 64, second));
  EXPECT_EQ(first, 0u);
  EXPECT_EQ(second, 64u);
  EXPECT_EQ(allocator.FreeBytes(), 1024u - 3u - 64u);
}

TEST(MemorySpanAllocator, CoalescesReturnedSpans) {
  MemorySpanAllocator allocator(1024);
  uint64_t first = 0, second = 0, third = 0;
  ASSERT_TRUE(allocator.Allocate(128, 1, first));
  ASSERT_TRUE(allocator.Allocate(256, 1, second));
  ASSERT_TRUE(allocator.Allocate(64, 1, third));
  allocator.Free(second, 256);
  allocator.Free(first, 128);
  allocator.Free(third, 64);
  EXPECT_EQ(allocator.FreeBytes(), 1024u);
  uint64_t whole = 1;
  EXPECT_TRUE(allocator.Allocate(1024, 1, whole));
  EXPECT_EQ(whole, 0u);
}

TEST(MemorySpanAllocator, RejectsInvalidAndExhaustedRequests) {
  MemorySpanAllocator allocator(128);
  uint64_t offset = 0;
  EXPECT_FALSE(allocator.Allocate(0, 1, offset));
  EXPECT_FALSE(allocator.Allocate(1, 3, offset));
  EXPECT_FALSE(allocator.Allocate(129, 1, offset));
  ASSERT_TRUE(allocator.Allocate(128, 1, offset));
  EXPECT_FALSE(allocator.Allocate(1, 1, offset));
}

}  // namespace
}  // namespace gpu::vk
