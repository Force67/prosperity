#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "base/containers/array.h"
#include "base/threading/thread.h"
#include "runtime/vprx/ps4/lib_sce_pad/lib_sce_pad.h"

TEST(Pad, NeutralSticksFallInsideControllerDeadzone) {
  alignas(u64) base::Array<u8, 0x78> data{};
  alignas(u32) base::Array<u8, 0x1c> info{};
  ASSERT_EQ(scePadReadState(1, data.data()), 0);
  ASSERT_EQ(scePadGetControllerInformation(1, info.data()), 0);
  for (int i = 4; i < 8; ++i)
    EXPECT_LE(std::abs(2 * data[i] - 255), 2 * info[i < 6 ? 8 : 9]);
}

TEST(Pad, SampleTimestampsAdvanceInMicroseconds) {
  alignas(u64) base::Array<u8, 0x78> data{};
  ASSERT_EQ(scePadReadState(1, data.data()), 0);
  u64 first = 0;
  std::memcpy(&first, data.data() + 0x50, sizeof(first));
  base::SleepForMilliseconds(10);
  ASSERT_EQ(scePadReadState(1, data.data()), 0);
  u64 second = 0;
  std::memcpy(&second, data.data() + 0x50, sizeof(second));
  ASSERT_GT(second, first);
  EXPECT_GE(second - first, 1000u);
}
