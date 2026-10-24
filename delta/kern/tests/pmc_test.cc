// opmc_* counts real events on the host PMU.

#include <gtest/gtest.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "guest_abi.h"
#include "kern/lv2/sys_debug.h"

namespace {
// Whether this process can count at all: a profiler holding every counter
// (Tracy's system tracing, unless TRACY_NO_SYS_TRACE is set) leaves none.
bool HostCounts() {
  perf_event_attr a{};
  a.size = sizeof(a);
  a.type = PERF_TYPE_HARDWARE;
  a.config = PERF_COUNT_HW_INSTRUCTIONS;
  a.exclude_kernel = 1;
  const int fd =
      static_cast<int>(syscall(SYS_perf_event_open, &a, 0, -1, -1, 0));
  if (fd < 0)
    return false;
  volatile u64 sink = 0;
  for (int i = 0; i < 1000; i++)
    sink = sink + i;
  u64 v = 0;
  const bool ok = read(fd, &v, sizeof(v)) == sizeof(v) && v;
  close(fd);
  return ok;
}
}  // namespace

TEST(Pmc, CountsRetiredInstructions) {
  if (!HostCounts())
    GTEST_SKIP() << "the host PMU is not available to this process";
  constexpr u64 kRetiredInstructions = 0xc0;
  ASSERT_EQ(kern::sys_opmc_set_ctl(0, 0, kRetiredInstructions), 0);
  EXPECT_EQ(kern::sys_opmc_set_ctl(0, 2, 1ull << 22), -22);  // enable is ours
  EXPECT_EQ(kern::sys_opmc_set_ctl(0, 6, 0), -22);
  EXPECT_EQ(kern::sys_opmc_set_ctl(0, 1, 1ull << 40), -22);  // reserved bit
  ASSERT_EQ(kern::sys_opmc_enable(0), 0);
  volatile u64 sink = 0;
  for (int i = 0; i < 100000; i++)
    sink = sink + i;
  u64 count = 0;
  ASSERT_EQ(kern::sys_opmc_get_ctr(0, 0, &count), 0);
  EXPECT_GT(count, 100000u);
  ASSERT_EQ(kern::sys_opmc_set_ctr(0, 0, 5), 0);
  ASSERT_EQ(kern::sys_opmc_disable(0), 0);
  ASSERT_EQ(kern::sys_opmc_get_ctr(0, 0, &count), 0);
  EXPECT_LT(count, 1000u);  // counting stopped right after the reset to 5
}
