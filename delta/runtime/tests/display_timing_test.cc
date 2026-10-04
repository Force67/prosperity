#include <gtest/gtest.h>

#include <array>

#include "base/atomic.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "guest/display.h"
#include "guest/session.h"

namespace {
class DisplayTimingTest : public testing::Test {
 protected:
  void SetUp() override {
    guest::BeginSession();
    guest::ResetSessionResources();
    guest::display::kEmulateTiming.SetFromString("true");
  }
  void TearDown() override {
    guest::RequestStop();
    guest::JoinThreads();
    guest::ResetSessionResources();
    guest::display::kEmulateTiming.Reset();
    guest::BeginSession();
  }
};

TEST_F(DisplayTimingTest, DisabledCompletesSynchronously) {
  guest::display::kEmulateTiming.SetFromString("false");
  bool completed = false;
  ASSERT_TRUE(guest::display::QueueFlip(1, [&] { completed = true; }));
  EXPECT_TRUE(completed);
  EXPECT_EQ(guest::LiveThreads(), 0u);
}

TEST_F(DisplayTimingTest, VblankAdvancesWithoutFlipsAndWaitsForNextTick) {
  const u64 first = guest::display::VblankCount();
  guest::display::WaitVblank();
  EXPECT_GT(guest::display::VblankCount(), first);
  EXPECT_LE(guest::display::VblankTimeNs(),
            static_cast<u64>(base::TickClock::NowNs()));
  EXPECT_EQ(guest::LiveThreads(), 0u);
}

TEST_F(DisplayTimingTest, QueuePacesFramesAndAppliesBackpressure) {
  base::Atomic<u32> count{0};
  std::array<u64, 8> ticks{};
  for (u32 i = 0; i < ticks.size(); ++i)
    ASSERT_TRUE(guest::display::QueueFlip(1, [&, i] {
      ticks[i] = guest::display::VblankCount();
      count.fetch_add(1);
    }));
  EXPECT_GE(count.load(), 6u);
  const auto deadline = base::TimeTicks::Now() + base::Seconds(2);
  while (count.load() < ticks.size() && base::TimeTicks::Now() < deadline)
    base::SleepForMilliseconds(1);
  ASSERT_EQ(count.load(), ticks.size());
  for (u32 i = 1; i < ticks.size(); ++i)
    EXPECT_GT(ticks[i], ticks[i - 1]);
}

TEST_F(DisplayTimingTest, FlipRateOneUsesEverySecondVblank) {
  guest::display::SetFlipRate(1);
  base::Atomic<u32> count{0};
  std::array<u64, 3> ticks{};
  for (u32 i = 0; i < ticks.size(); ++i)
    ASSERT_TRUE(guest::display::QueueFlip(1, [&, i] {
      ticks[i] = guest::display::VblankCount();
      count.fetch_add(1);
    }));
  const auto deadline = base::TimeTicks::Now() + base::Seconds(2);
  while (count.load() < ticks.size() && base::TimeTicks::Now() < deadline)
    base::SleepForMilliseconds(1);
  ASSERT_EQ(count.load(), ticks.size());
  EXPECT_GE(ticks[1] - ticks[0], 2u);
  EXPECT_GE(ticks[2] - ticks[1], 2u);
}

TEST_F(DisplayTimingTest,
       StopCancelsPendingCompletionAndNextSessionStartsFresh) {
  guest::display::SetFlipRate(2);
  base::Atomic<bool> completed{false};
  ASSERT_TRUE(guest::display::QueueFlip(1, [&] { completed.store(true); }));
  guest::RequestStop();
  guest::JoinThreads();
  EXPECT_FALSE(completed.load());
  EXPECT_EQ(guest::LiveThreads(), 0u);
  guest::ResetSessionResources();
  guest::BeginSession();
  EXPECT_EQ(guest::display::VblankCount(), 0u);
}

}  // namespace
