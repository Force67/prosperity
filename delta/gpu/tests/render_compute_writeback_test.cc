#include "gpu/render/compute_writeback.h"

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>

#include <array>

namespace gpu::render {
namespace {

TEST(ComputeWriteback, PreservesCpuWordsAndImagePadding) {
  std::array<u8, 64> guest{}, staged{}, shadow{};
  staged[0] = 1;
  staged[4] = 2;
  guest[4] = 3;
  guest[60] = 4;
  ComputeWritebackCounts counts;
  MergeComputeWritebackBlock(guest.data(), staged.data(), shadow.data(), counts);
  EXPECT_EQ(guest[0], 1);
  EXPECT_EQ(guest[4], 3);
  EXPECT_EQ(guest[60], 4);
  EXPECT_EQ(staged, guest);
  EXPECT_EQ(shadow, guest);
  EXPECT_EQ(counts.wrote, 4);
  EXPECT_EQ(counts.conflicts, 1);
}

TEST(ComputeWriteback, DoesNotWriteUntouchedGuestPages) {
  const auto page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  void* mapping = mmap(nullptr, page * 2, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  struct Unmap {
    void* mapping;
    size_t size;
    ~Unmap() { munmap(mapping, size); }
  } unmap{mapping, page * 2};
  auto* guest = static_cast<u8*>(mapping) + page - 16;
  std::array<u8, 64> staged{}, shadow{};
  staged[0] = 1;
  ASSERT_EQ(mprotect(static_cast<u8*>(mapping) + page, page, PROT_READ), 0);
  ComputeWritebackCounts counts;
  MergeComputeWritebackBlock(guest, staged.data(), shadow.data(), counts);
  EXPECT_EQ(guest[0], 1);
  EXPECT_EQ(counts.wrote, 4);
  EXPECT_EQ(counts.adopted, 0);
}

}  // namespace
}  // namespace gpu::render
