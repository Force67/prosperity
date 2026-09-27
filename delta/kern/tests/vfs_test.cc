#include "base/arch.h"

#include <gtest/gtest.h>

#include "kern/vfs.h"

TEST(Vfs, ReportsRootDirectory) {
  i64 size = -1;
  bool is_dir = false;

  EXPECT_TRUE(kern::vfs::Stat("/", size, is_dir));
  EXPECT_EQ(size, 0);
  EXPECT_TRUE(is_dir);
}

TEST(Vfs, UnmountRemovesGuestPrefix) {
  kern::vfs::Mount("/vfs-unmount-test", "/tmp");
  EXPECT_FALSE(kern::vfs::Resolve("/vfs-unmount-test/file").empty());

  kern::vfs::Unmount("/vfs-unmount-test");
  EXPECT_TRUE(kern::vfs::Resolve("/vfs-unmount-test/file").empty());
}
