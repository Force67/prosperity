#include <sys/stat.h>

#include <gtest/gtest.h>

#include "kern/lv2/sys_mem.h"
#include "kern/process.h"
#include "kern/ps5/dev/dma_dev.h"

TEST(DmaDevice, Ps5MainAllocationsUseTheReportedPool) {
  kern::Process process;
  process.SetPlatform(kern::Process::Platform::kPs5);
  auto& device = *new kern::DmaDevicePs5(process.GetObjTable());

  u64 total = 0;
  ASSERT_EQ(device.Ioctl(0x4008800A, &total), 0);
  ASSERT_EQ(total, 0x320000000ull);

  struct stat backing {};
  ASSERT_EQ(fstat(kern::DmemBackingFd(), &backing), 0);
  EXPECT_EQ(static_cast<u64>(backing.st_size), total);
  EXPECT_EQ(kern::DmemBackingSize(), total);

  u64 allocation[] = {0, 0, total - 0x10000000, 0x4000, 12};
  ASSERT_EQ(device.Ioctl(0xC0288011, allocation), 0);
  EXPECT_EQ(allocation[0], 0x10000000u);
  EXPECT_EQ(allocation[0] + allocation[2], total);

  auto* mapping =
      kern::sys_mmap(nullptr, 0x4000, 3, 0, device.handle(), allocation[0]);
  ASSERT_NE(mapping, reinterpret_cast<u8*>(-1));
  u64 info[9]{};
  ASSERT_EQ(kern::sys_virtual_query(mapping, 0, info, sizeof(info)), 0);
  EXPECT_EQ(info[2], allocation[0]);
  EXPECT_EQ(kern::DmemTypeForOffset(info[2]), 12);
  *mapping = 42;
  auto mapped_region = process.GetVma().Get(mapping);
  ASSERT_TRUE(mapped_region);

  u64 release[] = {info[2], 0x4000};
  ASSERT_EQ(device.Ioctl(0x80108002, release), 0);
  u64 reuse[] = {0, 0, 0x4000, 0x4000, 12};
  ASSERT_EQ(device.Ioctl(0xC0288011, reuse), 0);
  EXPECT_EQ(reuse[0], allocation[0]);
  auto* replacement =
      kern::sys_mmap(nullptr, 0x4000, 3, 0, device.handle(), reuse[0]);
  ASSERT_NE(replacement, reinterpret_cast<u8*>(-1));
  *replacement = 23;
  EXPECT_NE(*mapping, 23);
  EXPECT_FALSE(process.GetVma().Get(mapping));
  EXPECT_EQ(mapped_region->ptr, mapping);
  EXPECT_EQ(mapped_region->phys_offset, allocation[0]);
}
