#include <gtest/gtest.h>
#include "base/arch.h"

#include "base/atomic.h"
#include "base/containers/array.h"
#include "base/containers/vector.h"
#include "base/functional/function.h"
#include "base/memory/move.h"
#include "base/memory/unique_pointer.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "kern/lv2/sys_thread.h"

namespace {
// Joins on scope exit, like std::jthread.
struct JoinedThread {
  base::Thread thread;
  JoinedThread(const char* name, base::Function<void()> body)
      : thread(name, base::move(body), true) {}
  ~JoinedThread() { thread.Join(); }
};

struct GuestTimespec {
  i64 sec;
  i64 nsec;
};

bool WaitFor(const base::Atomic<int>& value, int expected) {
  const auto deadline = base::TimeTicks::Now() + base::Seconds(1);
  while (value.load() != expected && base::TimeTicks::Now() < deadline)
    base::SleepForMilliseconds(1);
  return value.load() == expected;
}

TEST(UmtxOp, WakeHonorsRequestedCount) {
  u32 word = 0;
  GuestTimespec timeout{2, 0};
  base::Atomic<int> ready{0};
  base::Atomic<int> returned{0};
  base::Array<int, 4> results{};
  base::Vector<base::UniquePointer<JoinedThread>> waiters;

  for (size_t i = 0; i < results.size(); ++i) {
    waiters.push_back(base::MakeUnique<JoinedThread>("waiter", [&, i] {
      ready.fetch_add(1);
      results[i] = kern::sys_umtx_op(&word, 15, 0, nullptr, &timeout);
      returned.fetch_add(1);
    }));
  }

  ASSERT_TRUE(WaitFor(ready, 4));
  base::SleepForMilliseconds(20);

  EXPECT_EQ(kern::sys_umtx_op(&word, 16, 1, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(returned, 1));
  base::SleepForMilliseconds(20);
  EXPECT_EQ(returned.load(), 1);

  // FreeBSD 9 wakes one waiter for a zero count as a consequence of its queue
  // loop, even though normal callers use positive counts.
  EXPECT_EQ(kern::sys_umtx_op(&word, 16, 0, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(returned, 2));
  base::SleepForMilliseconds(20);
  EXPECT_EQ(returned.load(), 2);

  EXPECT_EQ(kern::sys_umtx_op(&word, 16, 2, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(returned, 4));
  for (int result : results)
    EXPECT_EQ(result, 0);
}

TEST(UmtxOp, WakeBeforeWaitIsLost) {
  u32 word = 0;
  GuestTimespec timeout{0, 30'000'000};

  EXPECT_EQ(kern::sys_umtx_op(&word, 16, 1, nullptr, nullptr), 0);
  EXPECT_EQ(kern::sys_umtx_op(&word, 15, 0, nullptr, &timeout), -60);
}

TEST(UmtxOp, BucketCollisionDoesNotReleaseAnotherAddress) {
  alignas(16) base::Array<u32, 1025> words{};
  auto* first = &words[0];
  auto* collision = &words[1024];  // 0x1000 apart: same wait-bucket hash.
  GuestTimespec timeout{2, 0};
  base::Atomic<int> ready{0};
  base::Atomic<int> first_returned{0};
  base::Atomic<int> collision_returned{0};
  int first_result = 0;
  int collision_result = 0;

  JoinedThread first_waiter("first", [&] {
    ready.fetch_add(1);
    first_result = kern::sys_umtx_op(first, 15, 0, nullptr, &timeout);
    first_returned.store(1);
  });
  JoinedThread collision_waiter("collision", [&] {
    ready.fetch_add(1);
    collision_result = kern::sys_umtx_op(collision, 15, 0, nullptr, &timeout);
    collision_returned.store(1);
  });

  ASSERT_TRUE(WaitFor(ready, 2));
  base::SleepForMilliseconds(20);

  EXPECT_EQ(kern::sys_umtx_op(first, 16, 1, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(first_returned, 1));
  base::SleepForMilliseconds(20);
  EXPECT_EQ(collision_returned.load(), 0);

  EXPECT_EQ(kern::sys_umtx_op(collision, 16, 1, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(collision_returned, 1));
  EXPECT_EQ(first_result, 0);
  EXPECT_EQ(collision_result, 0);
}
}  // namespace
