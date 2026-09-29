#include "base/arch.h"
#include "gpu/render/index_upload.h"

#include <gtest/gtest.h>

#include <cstring>

namespace gpu::render {
namespace {

TEST(IndexUpload, PreservesNativeWidths) {
  const u16 indices16[] = {1, 9, 3};
  const u32 indices32[] = {2, 70000, 4};
  u16 output16[3] = {};
  u32 output32[3] = {};
  CopyGuestIndices(output16, indices16, 3, 0);
  CopyGuestIndices(output32, indices32, 3, 1);
  EXPECT_EQ(std::memcmp(output16, indices16, sizeof(indices16)), 0);
  EXPECT_EQ(std::memcmp(output32, indices32, sizeof(indices32)), 0);
  EXPECT_EQ(MaxGuestIndex(indices16, 3, 0), 9u);
  EXPECT_EQ(MaxGuestIndex(indices32, 3, 1), 70000u);
}

TEST(IndexUpload, WidensEightBitIndices) {
  const u8 input[] = {0, 255, 7};
  u16 output[3] = {};
  CopyGuestIndices(output, input, 3, 2);
  EXPECT_EQ(output[0], 0u);
  EXPECT_EQ(output[1], 255u);
  EXPECT_EQ(output[2], 7u);
  EXPECT_EQ(MaxGuestIndex(input, 3, 2), 255u);
  EXPECT_EQ(GuestIndexElementBytes(2), 1u);
  EXPECT_EQ(UploadedIndexElementBytes(2), 2u);
}

TEST(IndexUpload, SplitsQuadsIntoTwoTriangles) {
  const u16 input[] = {10, 11, 12, 13, 20, 21, 22, 23};
  u16 output[12] = {};
  CopyGuestQuadIndices(output, input, 8, 0);
  const u16 expected[] = {10, 11, 12, 10, 12, 13, 20, 21, 22, 20, 22, 23};
  EXPECT_EQ(std::memcmp(output, expected, sizeof(expected)), 0);
}

}  // namespace
}  // namespace gpu::render
