// The PS5 streamwrite ring buffer: create, open, write, read back, close.

#include <gtest/gtest.h>
#include <cstring>

#include "base/containers/vector.h"
#include "guest_abi.h"
#include "kern/lv2/ps5/handlers.h"
#include "kern/process.h"

namespace {
constexpr int kEinval = 22;
constexpr u64 kBlock = 0x10000;

int Call(int op, int fd, const void* arg, size_t size) {
  return kern::Ps5Streamwrite(op, fd, size, arg);
}

TEST(Streamwrite, RoundTripAndWrap) {
  kern::Process process;  // a game: the category gate lets it through
  u32 handle = 0;
  const u64 create[3] = {2 * kBlock, reinterpret_cast<u64>(&handle), 0};
  ASSERT_EQ(Call(1, -1, create, sizeof(create)), 0);
  ASSERT_EQ(handle, 0x10000u);
  ASSERT_EQ(Call(1, -1, create, sizeof(create)), -16);  // one per app

  base::Vector<u8> data(3 * kBlock);
  for (size_t i = 0; i < data.size(); i++)
    data[i] = static_cast<u8>(i / kBlock + 1);
  u64 offset = 0, held = 0;
  u64 write[5] = {reinterpret_cast<u64>(data.data()), kBlock, 0,
                  reinterpret_cast<u64>(&offset), reinterpret_cast<u64>(&held)};
  EXPECT_EQ(Call(20, handle, write, sizeof(write)), -kEinval);  // not open

  const u64 zero = 0;
  ASSERT_EQ(Call(19, handle, &zero, 8), 0);
  for (int i = 0; i < 3; i++) {
    write[0] = reinterpret_cast<u64>(data.data() + i * kBlock);
    ASSERT_EQ(Call(20, handle, write, sizeof(write)), 0);
    EXPECT_EQ(offset, i * kBlock);
  }
  EXPECT_EQ(held, 2 * kBlock);  // the first block was overwritten

  base::Vector<u8> back(kBlock);
  u64 read[4] = {reinterpret_cast<u64>(back.data()), 2 * kBlock, kBlock, 0};
  ASSERT_EQ(Call(21, handle, read, sizeof(read)), 0);
  EXPECT_EQ(back[0], 3);
  read[1] = 0;  // gone
  EXPECT_EQ(Call(21, handle, read, sizeof(read)), -kEinval);

  u64 stat[8] = {};
  const u64 stat_arg[2] = {reinterpret_cast<u64>(stat), 0};
  ASSERT_EQ(Call(18, handle, stat_arg, sizeof(stat_arg)), 0);
  EXPECT_EQ(stat[0], 2 * kBlock);
  EXPECT_EQ(stat[2], 3 * kBlock);
  EXPECT_EQ(stat[3], 2 * kBlock);

  EXPECT_EQ(Call(17, handle, &zero, 8), -16);  // still open
  ASSERT_EQ(Call(22, handle, &zero, 8), 0);
  ASSERT_EQ(Call(17, handle, &zero, 8), 0);
  EXPECT_EQ(Call(5, handle, &zero, 8), -78);  // unassigned operation
}
}  // namespace
