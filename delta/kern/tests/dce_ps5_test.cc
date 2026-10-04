#include <gtest/gtest.h>

#include "guest/display.h"
#include "guest/session.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_event.h"
#include "kern/object_table.h"
#include "kern/ps5/dev/dce_dev.h"
#include "options/options.h"

namespace {
struct Request {
  u32 op, flags = 0, handle = 1, size = 0;
  u64 data = 0, value = 0, result = 0, result_size = 0;
};

class DcePs5Test : public testing::Test {
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
  i32 Call(Request request) { return device.Ioctl(0x80308217, &request); }
  kern::ObjectTable objects;
  kern::DceDevicePs5& device = *new kern::DceDevicePs5(objects);
};

class DcePs5SubmissionTest : public DcePs5Test,
                             public testing::WithParamInterface<u32> {};

TEST_F(DcePs5Test, PoolQueryReturnsAddressAndLabelsAreMapped) {
  u64 address = 0, size = 0;
  ASSERT_EQ(Call({.op = 9,
                  .size = 8,
                  .data = reinterpret_cast<u64>(&address),
                  .result = reinterpret_cast<u64>(&size),
                  .result_size = 8}),
            0);
  EXPECT_EQ(size, 0x4000u);
  EXPECT_EQ(reinterpret_cast<u64>(device.Map(nullptr, size, 0, 0, 0)), address);
  auto* labels = reinterpret_cast<u64*>(address + 0xa0);
  for (int i = 0; i < 16; ++i)
    EXPECT_EQ(labels[i], 0u);
  EXPECT_EQ(Call({.op = 0x33, .value = address + 0xa0}), 0);
  EXPECT_EQ(Call({.op = 0x33, .value = address}),
            -static_cast<i32>(kern::SysError::eINVAL));
}

TEST_P(DcePs5SubmissionTest,
       EopFlipPacesAndCompletesStackLabelBeforeReturning) {
  u64 buffer[6]{};
  buffer[2] = 0x1200000000ull;
  ASSERT_EQ(Call({.op = 0x37,
                  .size = sizeof(buffer),
                  .data = reinterpret_cast<u64>(buffer),
                  .value = 2}),
            0);
  EXPECT_TRUE(kern::DcePs5IsDisplayBuffer(buffer[2]));
  u64 pool = 0, size = 0;
  ASSERT_EQ(Call({.op = 9,
                  .size = 8,
                  .data = reinterpret_cast<u64>(&pool),
                  .result = reinterpret_cast<u64>(&size),
                  .result_size = 8}),
            0);
  u64 label = 0, result = ~u64(0);
  u64 flip[9]{};
  flip[1] = 2;
  flip[2] = 1 | (static_cast<u64>(GetParam()) << 32);
  flip[3] = 42;
  flip[6] = reinterpret_cast<u64>(&label);
  const u64 before = guest::display::VblankCount();
  ASSERT_EQ(Call({.op = 0x36,
                  .size = sizeof(flip),
                  .data = reinterpret_cast<u64>(flip),
                  .result = reinterpret_cast<u64>(&result),
                  .result_size = 8}),
            0);
  EXPECT_GT(guest::display::VblankCount(), before);
  EXPECT_EQ(label, 1u);
  EXPECT_EQ(reinterpret_cast<u64*>(pool + 0xa0)[2], 1u);
  EXPECT_EQ(result, 0u);
  EXPECT_EQ(kern::DcePs5ScanoutBuffer(), buffer[2]);
  u64 status[16]{};
  ASSERT_EQ(Call({.op = 0xa,
                  .size = sizeof(status),
                  .data = reinterpret_cast<u64>(status)}),
            0);
  EXPECT_EQ(status[0], 42u);
  EXPECT_EQ(status[2], 1u);
  EXPECT_EQ(reinterpret_cast<i32*>(status)[10], 2);
  guest::RequestStop();
  guest::JoinThreads();
  guest::ResetSessionResources();
  EXPECT_FALSE(kern::DcePs5Active());
  EXPECT_FALSE(kern::DcePs5IsDisplayBuffer(buffer[2]));
}

TEST_P(DcePs5SubmissionTest, BlankFlipCompletesWithoutARegisteredBuffer) {
  u64 label = 0;
  u64 flip[9]{};
  flip[1] = ~u64(0);
  flip[2] = 1 | (static_cast<u64>(GetParam()) << 32);
  flip[3] = 42;
  flip[6] = reinterpret_cast<u64>(&label);
  ASSERT_EQ(Call({.op = 0x36,
                  .size = sizeof(flip),
                  .data = reinterpret_cast<u64>(flip)}),
            0);
  EXPECT_EQ(label, 1u);
  EXPECT_EQ(kern::DcePs5ScanoutBuffer(), 0u);
  u64 status[16]{};
  ASSERT_EQ(Call({.op = 0xa,
                  .size = sizeof(status),
                  .data = reinterpret_cast<u64>(status)}),
            0);
  EXPECT_EQ(status[0], 42u);
  EXPECT_EQ(status[2], 1u);
  EXPECT_EQ(reinterpret_cast<i32*>(status)[10], -1);
}

INSTANTIATE_TEST_SUITE_P(Firmware,
                         DcePs5SubmissionTest,
                         testing::Values(0u, 1u, 5u, 9u));

TEST_P(DcePs5SubmissionTest, NativeFlipModesCompleteWithTimingDisabled) {
  guest::display::kEmulateTiming.SetFromString("false");
  u64 buffer[6]{};
  buffer[2] = 0x1200000000ull;
  ASSERT_EQ(Call({.op = 0x37,
                  .size = sizeof(buffer),
                  .data = reinterpret_cast<u64>(buffer)}),
            0);
  u64 count = 0;
  for (const u32 mode : {1u, 2u, 3u, 4u, 5u, 6u, 8u, 9u}) {
    u64 label = 0;
    u64 flip[9]{};
    flip[2] = mode | (static_cast<u64>(GetParam()) << 32);
    flip[3] = mode;
    flip[6] = reinterpret_cast<u64>(&label);
    ASSERT_EQ(Call({.op = 0x36,
                    .size = sizeof(flip),
                    .data = reinterpret_cast<u64>(flip)}),
              0);
    EXPECT_EQ(label, 1u);
    u64 status[16]{};
    ASSERT_EQ(Call({.op = 0xa,
                    .size = sizeof(status),
                    .data = reinterpret_cast<u64>(status)}),
              0);
    EXPECT_EQ(status[0], mode);
    EXPECT_EQ(status[2], ++count);
    EXPECT_EQ(reinterpret_cast<i32*>(status)[10], 0);
  }
  EXPECT_EQ(guest::LiveThreads(), 0u);
}

TEST_F(DcePs5Test, RejectsInvalidPayloadsAndUnknownRequests) {
  EXPECT_EQ(Call({.op = 9}), -static_cast<i32>(kern::SysError::eINVAL));
  EXPECT_EQ(Call({.op = 9, .size = 8, .data = 1}),
            -static_cast<i32>(kern::SysError::eFAULT));
  EXPECT_EQ(Call({.op = 0xff}), -static_cast<i32>(kern::SysError::eINVAL));
  EXPECT_EQ(Call({.op = 6, .value = 3}),
            -static_cast<i32>(kern::SysError::eINVAL));
  u64 flip[9]{};
  flip[1] = 16;
  EXPECT_EQ(Call({.op = 0x36,
                  .size = sizeof(flip),
                  .data = reinterpret_cast<u64>(flip)}),
            -static_cast<i32>(kern::SysError::eINVAL));
  flip[1] = 0;
  for (const u64 mode : {0ull, 7ull, 10ull, (2ull << 32) | 1ull}) {
    flip[2] = mode;
    EXPECT_EQ(Call({.op = 0x36,
                    .size = sizeof(flip),
                    .data = reinterpret_cast<u64>(flip)}),
              -static_cast<i32>(kern::SysError::eINVAL));
  }
}
}  // namespace

TEST_F(DcePs5Test, VblankAndFlipEventsHaveSeparateSources) {
  guest::RequestStop();
  auto& queue = *new kern::Equeue(objects, "videoout");
  constexpr u64 flip = 0x6010000000000;
  constexpr u64 vblank = 0x7010000000000;
  queue.AddEvent(flip, -13, nullptr);
  queue.AddEvent(vblank, -13, nullptr);
  kern::kevent_t events[2]{};
  const kern::ktimespec timeout{};

  queue.Trigger(-1, -13, 123, true);
  ASSERT_EQ(queue.Kevent(nullptr, 0, events, 2, &timeout), 1);
  EXPECT_EQ(events[0].ident, vblank);
  EXPECT_EQ(events[0].data, 123);

  queue.Trigger(-1, -13, 456);
  ASSERT_EQ(queue.Kevent(nullptr, 0, events, 2, &timeout), 1);
  EXPECT_EQ(events[0].ident, flip);
  EXPECT_EQ(events[0].data, 456);
}

TEST_F(DcePs5Test, UnregisterBuffersClearsScanoutAndAllowsReplacement) {
  u64 buffer[6]{};
  buffer[2] = 0x1200000000ull;
  ASSERT_EQ(Call({.op = 0x37,
                  .size = sizeof(buffer),
                  .data = reinterpret_cast<u64>(buffer)}),
            0);
  EXPECT_TRUE(kern::DcePs5IsDisplayBuffer(buffer[2]));
  u64 other[6]{};
  other[2] = 0x1400000000ull;
  ASSERT_EQ(Call({.op = 0x37,
                  .size = sizeof(other),
                  .data = reinterpret_cast<u64>(other),
                  .value = 1}),
            0);
  ASSERT_EQ(Call({.op = 4}), 0);
  EXPECT_TRUE(kern::DcePs5IsDisplayBuffer(other[2]));
  ASSERT_EQ(Call({.op = 4, .value = 1}), 0);
  EXPECT_FALSE(kern::DcePs5IsDisplayBuffer(other[2]));
  EXPECT_EQ(Call({.op = 5}), 0);
  EXPECT_FALSE(kern::DcePs5IsDisplayBuffer(buffer[2]));
  EXPECT_EQ(kern::DcePs5ScanoutBuffer(), 0u);
  buffer[2] = 0x1300000000ull;
  ASSERT_EQ(Call({.op = 0x37,
                  .size = sizeof(buffer),
                  .data = reinterpret_cast<u64>(buffer)}),
            0);
  EXPECT_EQ(kern::DcePs5ScanoutBuffer(), buffer[2]);
  EXPECT_EQ(Call({.op = 4, .value = 16}),
            -static_cast<i32>(kern::SysError::eINVAL));
}
