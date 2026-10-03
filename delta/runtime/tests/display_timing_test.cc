#include <gtest/gtest.h>

#include <array>

#include "base/atomic.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "guest/display.h"
#include "guest/session.h"
#include "runtime/vprx/ps4/lib_sce_video_out/lib_sce_video_out.h"
#include "runtime/vprx/vprx.h"

namespace {
void InitVprx() {
  static const bool initialized = [] {
    runtime::vprx::Init();
    return true;
  }();
  (void)initialized;
}

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

TEST_F(DisplayTimingTest, Ps5EopCompletesTemporaryLabelBeforeReturning) {
  InitVprx();
  using SubmitEop = int PS4ABI (*)(int, int, int, i64, void*);
  auto submit = reinterpret_cast<SubmitEop>(
      runtime::vprx::LookupForced("libSceVideoOut", 0x8FCC65FBDD80D2AE));
  ASSERT_NE(submit, nullptr);
  guest::display::SetFlipRate(2);
  u64 label = 0;
  const u64 first = guest::display::VblankCount();
  ASSERT_EQ(submit(1, -1, 1, 0, &label), 0);
  EXPECT_EQ(label, 1u);
  EXPECT_GE(guest::display::VblankCount() - first, 3u);
}

struct FlipStatus {
  u64 count, process_time, tsc;
  i64 flip_arg;
  u64 submit_tsc, reserved0;
  i32 gc_queue_num, flip_pending_num, current_buffer;
  u32 reserved1;
};

class VideoOutTimingTest : public DisplayTimingTest,
                           public testing::WithParamInterface<bool> {
 protected:
  using Submit = int PS4ABI (*)(int, int, int, i64);
  using Status = int PS4ABI (*)(int, void*);
  using Pending = int PS4ABI (*)(int);
  Submit submit_ = sceVideoOutSubmitFlip;
  Status status_ = sceVideoOutGetFlipStatus;
  Pending pending_ = sceVideoOutIsFlipPending;

  void SetUp() override {
    DisplayTimingTest::SetUp();
    if (GetParam()) {
      InitVprx();
      submit_ = reinterpret_cast<Submit>(
          runtime::vprx::LookupForced("libSceVideoOut", 0x538E8DC0E889A72B));
      status_ = reinterpret_cast<Status>(
          runtime::vprx::LookupForced("libSceVideoOut", 0x49B537770A7CD254));
      pending_ = reinterpret_cast<Pending>(
          runtime::vprx::LookupForced("libSceVideoOut", 0xCE05E27C74FD12B6));
      ASSERT_NE(submit_, nullptr);
      ASSERT_NE(status_, nullptr);
      ASSERT_NE(pending_, nullptr);
    }
  }
};

TEST_P(VideoOutTimingTest, StatusChangesOnlyWhenFlipCompletes) {
  guest::display::SetFlipRate(2);
  ASSERT_EQ(submit_(1, 3, 1, 42), 0);
  FlipStatus status{};
  ASSERT_EQ(status_(1, &status), 0);
  EXPECT_EQ(status.count, 0u);
  EXPECT_EQ(status.current_buffer, -1);
  EXPECT_EQ(status.flip_pending_num, 1);
  EXPECT_EQ(pending_(1), 1);
  const auto deadline = base::TimeTicks::Now() + base::Seconds(2);
  while (pending_(1) && base::TimeTicks::Now() < deadline)
    base::SleepForMilliseconds(1);
  ASSERT_EQ(status_(1, &status), 0);
  EXPECT_EQ(status.count, 1u);
  EXPECT_EQ(status.current_buffer, 3);
  EXPECT_EQ(status.flip_arg, 42);
  EXPECT_EQ(status.flip_pending_num, 0);
  EXPECT_GE(status.tsc, status.submit_tsc);
}

TEST_P(VideoOutTimingTest, DisabledRetainsImmediateCompletion) {
  guest::display::kEmulateTiming.SetFromString("false");
  ASSERT_EQ(submit_(1, 2, 1, 99), 0);
  FlipStatus status{};
  ASSERT_EQ(status_(1, &status), 0);
  EXPECT_EQ(status.count, 1u);
  EXPECT_EQ(status.current_buffer, 2);
  EXPECT_EQ(status.flip_arg, 99);
  EXPECT_EQ(pending_(1), 0);
}

INSTANTIATE_TEST_SUITE_P(Ps4AndPs5, VideoOutTimingTest, testing::Bool());
}  // namespace
