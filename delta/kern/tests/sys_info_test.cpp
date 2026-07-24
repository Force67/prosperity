#include <cstdint>

#include <gtest/gtest.h>

#include "kern/lv2/sys_info.h"

TEST(SysInfo, ReportsPs4PageSize) {
  int mib[] = {6, 7};
  uint32_t page_size = 0;
  size_t result_size = sizeof(page_size);

  EXPECT_EQ(krnl::sys_sysctl(mib, 2, &page_size, &result_size, nullptr, 0), 0);
  EXPECT_EQ(result_size, sizeof(page_size));
  EXPECT_EQ(page_size, 0x4000u);
}
