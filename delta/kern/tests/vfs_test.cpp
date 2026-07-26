#include <cstdint>

#include <gtest/gtest.h>

#include "kern/vfs.h"

TEST(Vfs, ReportsRootDirectory) {
  int64_t size = -1;
  bool is_dir = false;

  EXPECT_TRUE(krnl::vfs::stat("/", size, is_dir));
  EXPECT_EQ(size, 0);
  EXPECT_TRUE(is_dir);
}

TEST(Vfs, UnmountRemovesGuestPrefix) {
  krnl::vfs::mount("/vfs-unmount-test", "/tmp");
  EXPECT_FALSE(krnl::vfs::resolve("/vfs-unmount-test/file").empty());

  krnl::vfs::unmount("/vfs-unmount-test");
  EXPECT_TRUE(krnl::vfs::resolve("/vfs-unmount-test/file").empty());
}
