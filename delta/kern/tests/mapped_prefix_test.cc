#include <gtest/gtest.h>

#include <sys/mman.h>
#include <unistd.h>

#include "host_memory/host_memory.h"

namespace {
const size_t kPage = static_cast<size_t>(::sysconf(_SC_PAGE_SIZE));

// Maps `pages` pages followed by an unmapped one.
unsigned char* MapBeforeHole(size_t pages) {
  void* p = ::mmap(nullptr, (pages + 1) * kPage, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED)
    return nullptr;
  auto* bytes = static_cast<unsigned char*>(p);
  ::munmap(bytes + pages * kPage, kPage);
  return bytes;
}

TEST(MappedMemoryPrefix, StopsAtTheHoleAfterUntrackedPages) {
  unsigned char* p = MapBeforeHole(5);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(host_memory::MappedMemoryPrefix(p + 100, 64 * kPage),
            5 * kPage - 100);
  EXPECT_EQ(host_memory::MappedMemoryPrefix(p, 3 * kPage + 7),
            3 * kPage + 7);
  ::munmap(p, 5 * kPage);
}

TEST(MappedMemoryPrefix, ContinuesPastTrackedPagesIntoUntrackedOnes) {
  unsigned char* p = MapBeforeHole(6);
  ASSERT_NE(p, nullptr);
  host_memory::TrackMemoryMapping(p, 2 * kPage);
  EXPECT_EQ(host_memory::MappedMemoryPrefix(p, 64 * kPage), 6 * kPage);
  EXPECT_EQ(host_memory::MappedMemoryPrefix(p, kPage + 1), kPage + 1);
  host_memory::ForgetMemoryMapping(p, 2 * kPage);
  ::munmap(p, 6 * kPage);
}

TEST(MappedMemoryPrefix, UnmappedStartHasNoPrefix) {
  unsigned char* p = MapBeforeHole(1);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(host_memory::MappedMemoryPrefix(p + kPage, 4 * kPage), 0u);
  ::munmap(p, kPage);
}
}  // namespace
