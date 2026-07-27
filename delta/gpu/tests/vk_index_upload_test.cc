#include "gpu/vulkan/vk_index_upload.h"

#include <gtest/gtest.h>

#include <cstring>

namespace gpu::vk {
namespace {

TEST(IndexUpload, PreservesNativeWidths) {
  const uint16_t indices16[] = {1, 9, 3};
  const uint32_t indices32[] = {2, 70000, 4};
  uint16_t output16[3] = {};
  uint32_t output32[3] = {};
  CopyGuestIndices(output16, indices16, 3, 0);
  CopyGuestIndices(output32, indices32, 3, 1);
  EXPECT_EQ(std::memcmp(output16, indices16, sizeof(indices16)), 0);
  EXPECT_EQ(std::memcmp(output32, indices32, sizeof(indices32)), 0);
  EXPECT_EQ(MaxGuestIndex(indices16, 3, 0), 9u);
  EXPECT_EQ(MaxGuestIndex(indices32, 3, 1), 70000u);
}

TEST(IndexUpload, WidensEightBitIndices) {
  const uint8_t input[] = {0, 255, 7};
  uint16_t output[3] = {};
  CopyGuestIndices(output, input, 3, 2);
  EXPECT_EQ(output[0], 0u);
  EXPECT_EQ(output[1], 255u);
  EXPECT_EQ(output[2], 7u);
  EXPECT_EQ(MaxGuestIndex(input, 3, 2), 255u);
  EXPECT_EQ(GuestIndexElementBytes(2), 1u);
  EXPECT_EQ(UploadedIndexElementBytes(2), 2u);
}

}  // namespace
}  // namespace gpu::vk
