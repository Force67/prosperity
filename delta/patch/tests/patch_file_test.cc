#include <gtest/gtest.h>

#include <cstring>

#include "patch/game_patches.h"
#include "patch/patch_file.h"

namespace {

using patch::Patch;

base::Vector<Patch> Parse(const char* text, bool expect_ok = true) {
  base::Vector<Patch> out;
  base::String error;
  EXPECT_EQ(patch::ParsePatchFile(base::StringRef(text), &out, &error),
            expect_ok)
      << error.c_str();
  return out;
}

TEST(PatchFileTest, ParsesBlocksAndEntryTypes) {
  const auto patches = Parse(R"(Version: 1.0

EBOOT-aa:
  "60 FPS":  # comment
    Games:
      "Some Game":
        CUSA00001: [ 01.00 ]
    Author: someone
    Patch:
      - [ byte, 0x10, 0 ]
      - [ le32, 0x20, 0x11223344 ]
      - [ be16, 32, 0x1234 ]
      - [ bytes, 0x40, be01000000 ]
      - [ c_utf8, 0x50, "a, b" ]
  Other:
    Patch:
      - [ lef32, 0x0, 1.5 ]
EBOOT-bb:
  Third:
    Patch:
      - [ le64, 0x8, 1 ]
)");
  ASSERT_EQ(patches.size(), 3u);
  EXPECT_STREQ(patches[0].image.c_str(), "EBOOT-aa");
  EXPECT_STREQ(patches[0].name.c_str(), "60 FPS");
  ASSERT_EQ(patches[0].writes.size(), 5u);
  EXPECT_EQ(patches[0].writes[1].address, 0x20u);
  const u8 le32[] = {0x44, 0x33, 0x22, 0x11};
  EXPECT_EQ(0, memcmp(patches[0].writes[1].bytes.data(), le32, 4));
  EXPECT_EQ(patches[0].writes[2].address, 32u);
  EXPECT_EQ(patches[0].writes[2].bytes[0], 0x12);
  EXPECT_EQ(patches[0].writes[3].bytes.size(), 5u);
  EXPECT_EQ(patches[0].writes[3].bytes[0], 0xbe);
  EXPECT_EQ(patches[0].writes[4].bytes.size(), 5u);
  EXPECT_STREQ(patches[1].name.c_str(), "Other");
  const u8 f15[] = {0x00, 0x00, 0xc0, 0x3f};
  EXPECT_EQ(0, memcmp(patches[1].writes[0].bytes.data(), f15, 4));
  EXPECT_STREQ(patches[2].image.c_str(), "EBOOT-bb");
  EXPECT_EQ(patches[2].writes[0].bytes.size(), 8u);
}

TEST(PatchFileTest, RejectsMalformedEntries) {
  Parse("EBOOT-aa:\n  P:\n    Patch:\n      - [ le16, 0x0, 0x10000 ]\n", false);
  Parse("EBOOT-aa:\n  P:\n    Patch:\n      - [ word, 0x0, 1 ]\n", false);
  Parse("EBOOT-aa:\n  P:\n    Patch:\n      - [ bytes, 0x0, abc ]\n", false);
  Parse("EBOOT-aa:\n  P:\n    Patch:\n      - [ byte, 0x0 ]\n", false);
  Parse("  P:\n    Patch:\n      - [ byte, 0x0, 1 ]\n", false);
}

TEST(PatchFileTest, ApplyStaysInsideTheImage) {
  u8 image[16] = {};
  auto patches =
      Parse("EBOOT-aa:\n  P:\n    Patch:\n      - [ le32, 0xc, 0xaabbccdd ]\n");
  const auto base = reinterpret_cast<uintptr_t>(image);
  ASSERT_TRUE(patch::ApplyPatch(patches[0], base, sizeof(image)));
  EXPECT_EQ(image[12], 0xdd);
  EXPECT_EQ(image[15], 0xaa);
  patches[0].writes[0].address = 0xd;
  image[15] = 0;
  EXPECT_FALSE(patch::ApplyPatch(patches[0], base, sizeof(image)));
  EXPECT_EQ(image[15], 0);
}

TEST(PatchFileTest, ImageKeyHashesSegmentBytes) {
  const u8 a[] = {'a', 'b'}, b[] = {'c'};
  const patch::Segment segments[] = {{a, 2}, {b, 1}};
  EXPECT_STREQ(patch::ImageKey(segments, 2).c_str(),
               "EBOOT-a9993e364706816aba3e25717850c26c9cd0d89d");
}

}  // namespace
