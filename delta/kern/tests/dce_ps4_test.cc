#include <gtest/gtest.h>

#include "base/threading/thread.h"
#include "base/time/time.h"
#include "guest/display.h"
#include "guest/session.h"
#include "kern/lv2/error_table.h"
#include "kern/object_table.h"
#include "kern/ps4/dev/dce_dev.h"
#include "options/options.h"

namespace {
class DcePs4Test : public testing::Test {
 protected:
  void SetUp() override {
    options::Init();
    guest::BeginSession();
    guest::ResetSessionResources();
    guest::display::kEmulateTiming.SetFromString("true");
    ASSERT_TRUE(device.Init("dce", 0, 0));
  }
  void TearDown() override {
    guest::RequestStop();
    guest::JoinThreads();
    guest::ResetSessionResources();
    guest::display::kEmulateTiming.Reset();
    guest::BeginSession();
  }
  struct Status {
    i64 arg;
    u64 reserved, count, time, tsc;
    i32 index;
    u32 pending, gpu_pending, reserved1;
    u64 submit_tsc, reserved2;
  };
  Status GetStatus() {
    Status status{};
    u64 query[] = {0xa, 1, reinterpret_cast<u64>(&status), sizeof(status),
                   0,   0};
    EXPECT_EQ(device.Ioctl(0xc0308203, query), 0);
    return status;
  }
  kern::ObjectTable objects;
  kern::DceDevice& device = *new kern::DceDevice(objects);
};

TEST_F(DcePs4Test, EopWaitsForGpuAndVblankAndReportsPendingFlips) {
  u64 result = ~u64(0), stack_slot = 123;
  u64 flip[] = {1,
                2,
                1,
                42,
                0,
                1,
                reinterpret_cast<u64>(&stack_slot),
                0,
                reinterpret_cast<u64>(&result)};
  ASSERT_EQ(device.Ioctl(0xc0488204, flip), 0);
  EXPECT_EQ(result, 0u);
  auto status = GetStatus();
  EXPECT_EQ(status.count, 0u);
  EXPECT_EQ(status.gpu_pending, 1u);
  EXPECT_EQ(status.pending, 0u);
  EXPECT_EQ(kern::DceSubmittedBuffer(), 2u);
  const u64 tick = guest::display::VblankCount();
  ASSERT_TRUE(kern::DceCompleteGpuFlips());
  const auto deadline = base::TimeTicks::Now() + base::Seconds(2);
  do {
    status = GetStatus();
    if (status.count)
      break;
    base::SleepForMilliseconds(1);
  } while (base::TimeTicks::Now() < deadline);
  EXPECT_EQ(status.count, 1u);
  EXPECT_GT(guest::display::VblankCount(), tick);
  EXPECT_EQ(status.index, 2);
  EXPECT_EQ(status.arg, 42);
  EXPECT_EQ(status.pending, 0u);
  EXPECT_EQ(status.gpu_pending, 0u);
  EXPECT_EQ(stack_slot, 123u);
  ASSERT_TRUE(kern::DceCompleteGpuFlips());
  EXPECT_EQ(GetStatus().count, 1u);
}

TEST_F(DcePs4Test, DisabledTimingCompletesAfterGpuWithoutWorker) {
  guest::display::kEmulateTiming.SetFromString("false");
  ASSERT_TRUE(kern::DceSubmitFlip(1, 1, 17, true));
  EXPECT_EQ(GetStatus().count, 0u);
  ASSERT_TRUE(kern::DceCompleteGpuFlips());
  EXPECT_EQ(GetStatus().count, 1u);
  EXPECT_EQ(GetStatus().arg, 17);
  EXPECT_EQ(guest::LiveThreads(), 0u);
}

TEST_F(DcePs4Test, NativeFlipModesCompleteAfterGpuWithTimingDisabled) {
  guest::display::kEmulateTiming.SetFromString("false");
  u64 count = 0;
  for (u64 mode = 1; mode <= 6; ++mode) {
    u64 result = ~u64(0), stack_slot = 123;
    u64 flip[] = {1,
                  mode - 1,
                  mode,
                  mode,
                  0,
                  1,
                  reinterpret_cast<u64>(&stack_slot),
                  0,
                  reinterpret_cast<u64>(&result)};
    ASSERT_EQ(device.Ioctl(0xc0488204, flip), 0);
    EXPECT_EQ(result, 0u);
    EXPECT_EQ(GetStatus().count, count);
    EXPECT_EQ(GetStatus().gpu_pending, 1u);
    ASSERT_TRUE(kern::DceCompleteGpuFlips());
    const auto status = GetStatus();
    EXPECT_EQ(status.count, ++count);
    EXPECT_EQ(status.index, mode - 1);
    EXPECT_EQ(status.arg, mode);
    EXPECT_EQ(status.pending, 0u);
    EXPECT_EQ(status.gpu_pending, 0u);
    EXPECT_EQ(stack_slot, 123u);
  }
  EXPECT_EQ(guest::LiveThreads(), 0u);
}

TEST_F(DcePs4Test, InvalidFlipRequestsPreserveOutputAndQueue) {
  u64 result = ~u64(0);
  u64 flip[] = {1, 0, 1, 42, 0, 1, 0, 0, reinterpret_cast<u64>(&result)};
  const auto rejected = [&] {
    EXPECT_EQ(device.Ioctl(0xc0488204, flip),
              -static_cast<i32>(kern::SysError::eINVAL));
    EXPECT_EQ(result, ~u64(0));
    const auto status = GetStatus();
    EXPECT_EQ(status.count, 0u);
    EXPECT_EQ(status.pending, 0u);
    EXPECT_EQ(status.gpu_pending, 0u);
  };
  for (u64 index : {16u, 17u}) {
    flip[1] = index;
    rejected();
  }
  flip[1] = 0;
  for (u64 mode : {0u, 7u}) {
    flip[2] = mode;
    rejected();
  }
  flip[2] = 1;
  flip[8] = 0;
  rejected();
}

TEST_F(DcePs4Test, ReleasesPreviousLabelAndResetClearsState) {
  guest::display::kEmulateTiming.SetFromString("false");
  u64 offset = 0, size = 0x4000;
  u64 query[] = {
      9, 1, reinterpret_cast<u64>(&offset), reinterpret_cast<u64>(&size), 0, 0};
  ASSERT_EQ(device.Ioctl(0xc0308203, query), 0);
  auto* pool = device.Map(nullptr, size, 0, 0, offset);
  ASSERT_NE(pool, reinterpret_cast<u8*>(-1));
  auto* labels = reinterpret_cast<u64*>(pool + 0xa0);
  labels[0] = 1;
  labels[1] = 1;
  ASSERT_TRUE(kern::DceSubmitFlip(0, 1, 1));
  EXPECT_EQ(labels[0], 1u);
  ASSERT_TRUE(kern::DceSubmitFlip(1, 1, 2));
  EXPECT_EQ(labels[0], 0u);
  EXPECT_EQ(labels[1], 1u);
  ASSERT_TRUE(kern::DceSubmitFlip(2, 1, 3, true));
  guest::ResetSessionResources();
  EXPECT_FALSE(kern::DceActive());
  EXPECT_EQ(GetStatus().count, 0u);
  EXPECT_EQ(GetStatus().gpu_pending, 0u);
  EXPECT_EQ(GetStatus().index, -1);
}

TEST_F(DcePs4Test, MapsOnlyAllocatedPoolRanges) {
  auto* failed = reinterpret_cast<u8*>(-1);
  EXPECT_EQ(device.Map(nullptr, 0x4000, 0, 0, 0), failed);
  u64 offset = 0, size = 0x4000;
  u64 query[] = {
      9, 1, reinterpret_cast<u64>(&offset), reinterpret_cast<u64>(&size), 0, 0};
  ASSERT_EQ(device.Ioctl(0xc0308203, query), 0);
  auto* pool = device.Map(nullptr, size, 0, 0, offset);
  ASSERT_NE(pool, failed);
  EXPECT_EQ(device.Map(nullptr, 8, 0, 0, offset + size - 8), pool + size - 8);
  EXPECT_EQ(device.Map(nullptr, size + 1, 0, 0, offset), failed);
  EXPECT_EQ(device.Map(nullptr, 8, 0, 0, offset + size), failed);
  EXPECT_EQ(device.Map(nullptr, 8, 0, 0, SIZE_MAX - 3), failed);
  EXPECT_EQ(device.Map(nullptr, SIZE_MAX, 0, 0, offset + 1), failed);
}
}  // namespace
