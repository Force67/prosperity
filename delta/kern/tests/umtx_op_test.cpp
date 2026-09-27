#include <gtest/gtest.h>
#include "base/arch.h"


#include "kern/lv2/sys_thread.h"
#include <base/atomic.h>
#include <base/containers/array.h>
#include <base/containers/vector.h>
#include <base/time/time.h>
#include <base/threading/thread.h>
#include <base/memory/unique_pointer.h>
#include <base/functional/function.h>
#include <base/memory/move.h>

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

bool WaitFor(const base::Atomic<int> &value, int expected) {
  const auto deadline = base::TimeTicks::Now() + base::Seconds(1);
  while (value.load() != expected &&
         base::TimeTicks::Now() < deadline)
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
      results[i] = krnl::sys_umtx_op(&word, 15, 0, nullptr, &timeout);
      returned.fetch_add(1);
    }));
  }

  ASSERT_TRUE(WaitFor(ready, 4));
  base::SleepForMilliseconds(20);

  EXPECT_EQ(krnl::sys_umtx_op(&word, 16, 1, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(returned, 1));
  base::SleepForMilliseconds(20);
  EXPECT_EQ(returned.load(), 1);

  // FreeBSD 9 wakes one waiter for a zero count as a consequence of its queue
  // loop, even though normal callers use positive counts.
  EXPECT_EQ(krnl::sys_umtx_op(&word, 16, 0, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(returned, 2));
  base::SleepForMilliseconds(20);
  EXPECT_EQ(returned.load(), 2);

  EXPECT_EQ(krnl::sys_umtx_op(&word, 16, 2, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(returned, 4));
  for (int result : results)
    EXPECT_EQ(result, 0);
}

TEST(UmtxOp, WakeBeforeWaitIsLost) {
  u32 word = 0;
  GuestTimespec timeout{0, 30'000'000};

  EXPECT_EQ(krnl::sys_umtx_op(&word, 16, 1, nullptr, nullptr), 0);
  EXPECT_EQ(krnl::sys_umtx_op(&word, 15, 0, nullptr, &timeout), -60);
}

TEST(UmtxOp, BucketCollisionDoesNotReleaseAnotherAddress) {
  alignas(16) base::Array<u32, 1025> words{};
  auto *first = &words[0];
  auto *collision = &words[1024]; // 0x1000 apart: same wait-bucket hash.
  GuestTimespec timeout{2, 0};
  base::Atomic<int> ready{0};
  base::Atomic<int> firstReturned{0};
  base::Atomic<int> collisionReturned{0};
  int firstResult = 0;
  int collisionResult = 0;

  JoinedThread firstWaiter("first", [&] {
    ready.fetch_add(1);
    firstResult = krnl::sys_umtx_op(first, 15, 0, nullptr, &timeout);
    firstReturned.store(1);
  });
  JoinedThread collisionWaiter("collision", [&] {
    ready.fetch_add(1);
    collisionResult =
        krnl::sys_umtx_op(collision, 15, 0, nullptr, &timeout);
    collisionReturned.store(1);
  });

  ASSERT_TRUE(WaitFor(ready, 2));
  base::SleepForMilliseconds(20);

  EXPECT_EQ(krnl::sys_umtx_op(first, 16, 1, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(firstReturned, 1));
  base::SleepForMilliseconds(20);
  EXPECT_EQ(collisionReturned.load(), 0);

  EXPECT_EQ(krnl::sys_umtx_op(collision, 16, 1, nullptr, nullptr), 0);
  ASSERT_TRUE(WaitFor(collisionReturned, 1));
  EXPECT_EQ(firstResult, 0);
  EXPECT_EQ(collisionResult, 0);
}
} // namespace
