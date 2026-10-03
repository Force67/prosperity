#include <gtest/gtest.h>

#include "base/atomic.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "guest/session.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_mem.h"

TEST(MdbgService, StopInterruptsWaitWithoutDeliveringAnEvent) {
  for (int session = 0; session < 3; ++session) {
    guest::BeginSession();
    base::Atomic<bool> entered{false};
    base::Atomic<bool> returned{false};
    int result = 0;
    base::Thread thread(
        "mdbg-wait",
        [&] {
          entered.store(true);
          result = kern::sys_mdbg_service(8, nullptr, nullptr, nullptr);
          returned.store(true);
        },
        true);
    while (!entered.load())
      base::SleepForMilliseconds(1);
    base::SleepForMilliseconds(40);
    EXPECT_FALSE(returned.load());
    guest::RequestStop();
    const auto deadline = base::TimeTicks::Now() + base::Seconds(1);
    while (!returned.load() && base::TimeTicks::Now() < deadline)
      base::SleepForMilliseconds(1);
    EXPECT_TRUE(returned.load());
    thread.Join();
    EXPECT_EQ(result, -static_cast<int>(kern::SysError::eINTR));
  }
  guest::BeginSession();
}
