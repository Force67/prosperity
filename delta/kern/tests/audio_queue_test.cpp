#include <array>
#include <cstring>
#include <limits>

#include <gtest/gtest.h>

#include "kern/ps5/audio_queue.h"

namespace {
struct Queue {
  alignas(8) std::array<u8, 0x4000> memory{};
  template <class T> void Set(size_t at, T value) {
    std::memcpy(memory.data() + at, &value, sizeof(value));
  }
  u64 Consumed() const {
    u64 value;
    std::memcpy(&value, memory.data() + 0x2500, sizeof(value));
    return value;
  }
  u32 Take() {
    return krnl::ps5::ConsumeAudioOut2Block(memory.data(), memory.size());
  }
  Queue() {
    Set(0x1c88, 256u);
    Set(0x1c8c, 3u);
  }
};

TEST(AudioOut2Queue, ReleasesOnlySubmittedBlocks) {
  Queue q;
  EXPECT_EQ(q.Take(), 0u);
  q.Set(0x1c80, u64{3});
  for (u64 i = 1; i <= 3; ++i) {
    EXPECT_EQ(q.Take(), 256u);
    EXPECT_EQ(q.Consumed(), i);
  }
  EXPECT_EQ(q.Take(), 0u);
  EXPECT_EQ(q.Consumed(), 3u);
}

TEST(AudioOut2Queue, SequenceNumbersRemain64BitAndWrap) {
  Queue q;
  for (u64 start : {u64{0xffffffff}, std::numeric_limits<u64>::max()}) {
    q.Set(0x2500, start);
    q.Set(0x1c80, start + 1);
    EXPECT_EQ(q.Take(), 256u);
    EXPECT_EQ(q.Consumed(), start + 1);
    EXPECT_EQ(q.Take(), 0u);
  }
}

TEST(AudioOut2Queue, InvalidOrIncompleteControlDoesNotGrant) {
  Queue q;
  EXPECT_EQ(krnl::ps5::ConsumeAudioOut2Block(nullptr, 0), 0u);
  q.Set(0x1c80, u64{4});  // Impossible occupancy for a three-block ring.
  EXPECT_EQ(q.Take(), 0u);
  q.Set(0x1c80, u64{1});
  EXPECT_EQ(krnl::ps5::ConsumeAudioOut2Block(q.memory.data(), 0x2504), 0u);
  q.Set(0x1c88, 0u);
  EXPECT_EQ(q.Take(), 0u);
  q.Set(0x1c88, 257u);
  EXPECT_EQ(q.Take(), 0u);
  q.Set(0x1c88, 256u);
  q.Set(0x1c8c, 0u);
  EXPECT_EQ(q.Take(), 0u);
  EXPECT_EQ(q.Consumed(), 0u);
}
}  // namespace
