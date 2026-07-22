#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "kern/ps4/lv2/sys_thread.h"

namespace {
using namespace std::chrono_literals;

struct GuestTimespec {
  int64_t sec;
  int64_t nsec;
};

bool WaitFor(const std::atomic<int> &value, int expected) {
  const auto deadline = std::chrono::steady_clock::now() + 1s;
  while (value.load() != expected &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(1ms);
  return value.load() == expected;
}

TEST(UmtxOp, WakeHonorsRequestedCount) {
  uint32_t word = 0;
  GuestTimespec timeout{2, 0};
  std::atomic<int> ready{0};
  std::atomic<int> returned{0};
  std::array<int, 4> results{};
  std::vector<std::jthread> waiters;

  for (size_t i = 0; i < results.size(); ++i) {
    waiters.emplace_back([&, i] {
      ready.fetch_add(1);
      results[i] = krnl::sys_umtx_op(&word, 15, 0, nullptr, &timeout);
      returned.fetch_add(1);
    });
  }

  ASSERT_TRUE(WaitFor(ready, 4));
  std::this_thread::sleep_for(20ms);

  EXPECT_EQ(krnl::sys_umtx_op(&word, 16, 1, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(returned, 1));
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(returned.load(), 1);

  // FreeBSD 9 wakes one waiter for a zero count as a consequence of its queue
  // loop, even though normal callers use positive counts.
  EXPECT_EQ(krnl::sys_umtx_op(&word, 16, 0, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(returned, 2));
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(returned.load(), 2);

  EXPECT_EQ(krnl::sys_umtx_op(&word, 16, 2, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(returned, 4));
  for (int result : results)
    EXPECT_EQ(result, 0);
}

TEST(UmtxOp, WakeBeforeWaitIsLost) {
  uint32_t word = 0;
  GuestTimespec timeout{0, 30'000'000};

  EXPECT_EQ(krnl::sys_umtx_op(&word, 16, 1, nullptr, nullptr), 0);
  EXPECT_EQ(krnl::sys_umtx_op(&word, 15, 0, nullptr, &timeout), -60);
}

TEST(UmtxOp, BucketCollisionDoesNotReleaseAnotherAddress) {
  alignas(16) std::array<uint32_t, 1025> words{};
  auto *first = &words[0];
  auto *collision = &words[1024]; // 0x1000 apart: same wait-bucket hash.
  GuestTimespec timeout{2, 0};
  std::atomic<int> ready{0};
  std::atomic<int> firstReturned{0};
  std::atomic<int> collisionReturned{0};
  int firstResult = 0;
  int collisionResult = 0;

  std::jthread firstWaiter([&] {
    ready.fetch_add(1);
    firstResult = krnl::sys_umtx_op(first, 15, 0, nullptr, &timeout);
    firstReturned.store(1);
  });
  std::jthread collisionWaiter([&] {
    ready.fetch_add(1);
    collisionResult =
        krnl::sys_umtx_op(collision, 15, 0, nullptr, &timeout);
    collisionReturned.store(1);
  });

  ASSERT_TRUE(WaitFor(ready, 2));
  std::this_thread::sleep_for(20ms);

  EXPECT_EQ(krnl::sys_umtx_op(first, 16, 1, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(firstReturned, 1));
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(collisionReturned.load(), 0);

  EXPECT_EQ(krnl::sys_umtx_op(collision, 16, 1, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(collisionReturned, 1));
  EXPECT_EQ(firstResult, 0);
  EXPECT_EQ(collisionResult, 0);
}
} // namespace
