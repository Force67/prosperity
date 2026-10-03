#include "gpu/ps5/guest_memory_ranges.h"

#include <string>

#include <gtest/gtest.h>

namespace gpu::ps5 {
namespace {

TEST(HostMapping, ParsesSharedFileBacking) {
  HostMapping mapping{};
  ASSERT_TRUE(
      ParseHostMapping("8000000000-8000010000 rw-s 00002000 00:1f 123456 "
                       "/memfd:direct (deleted)\n",
                       mapping));
  EXPECT_EQ(mapping.begin, 0x8000000000ull);
  EXPECT_EQ(mapping.end, 0x8000010000ull);
  EXPECT_EQ(mapping.offset, 0x2000ull);
  EXPECT_EQ(mapping.major, 0u);
  EXPECT_EQ(mapping.minor, 0x1fu);
  EXPECT_EQ(mapping.inode, 123456ull);
  EXPECT_STREQ(mapping.perm, "rw-s");
}

TEST(HostMapping, ParsesAnonymousAndProtectedMappings) {
  for (const char* permissions : {"r--p", "---p", "r-xp"}) {
    const std::string line =
        std::string("1000-2000 ") + permissions + " 00000000 00:00 0\n";
    HostMapping mapping{};
    ASSERT_TRUE(ParseHostMapping(line, mapping));
    EXPECT_STREQ(mapping.perm, permissions);
    EXPECT_EQ(mapping.inode, 0ull);
  }
}

TEST(HostMapping, RejectsMalformedAndOverflowingFields) {
  for (const char* line :
       {"", "1000-2000", "2000-1000 rw-p 0 00:00 0",
        "1000-2000 rw-p 0 00:00 12bad",
        "1000-2000 rw-p 0 00:00 18446744073709551616",
        "1000-2000 rw-p 0 100000000:00 0", "1000-2000 nope 0 00:00 0"}) {
    HostMapping mapping{};
    EXPECT_FALSE(ParseHostMapping(line, mapping)) << line;
  }
}

}  // namespace
}  // namespace gpu::ps5
