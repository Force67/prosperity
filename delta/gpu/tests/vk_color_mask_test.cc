#include <gtest/gtest.h>

#include "gpu/vulkan/vk_format.h"

TEST(VkColorMask, TargetMaskAppliesWithoutFrontendShaderMask) {
  EXPECT_EQ(gpu::vk::ColorWriteMask(0x737, 0, 7, 0), 7u);
  EXPECT_EQ(gpu::vk::ColorWriteMask(0x737, 0, 7, 1), 3u);
  EXPECT_EQ(gpu::vk::ColorWriteMask(0x737, 0, 7, 2), 7u);
  EXPECT_EQ(gpu::vk::ColorWriteMask(0, 0, 0xff, 0), 0u);
  EXPECT_EQ(gpu::vk::ColorWriteMask(0x80000000, 0, 0x80, 7), 8u);
}

TEST(VkColorMask, UnexportedAttachmentsCannotWrite) {
  EXPECT_EQ(gpu::vk::ColorWriteMask(0xffffffff, 0, 0, 0), 0u);
  EXPECT_EQ(gpu::vk::ColorWriteMask(0xffffffff, 0, 1, 1), 0u);
  EXPECT_EQ(gpu::vk::ColorWriteMask(0xffffffff, 0, 0xff, 8), 0u);
}

TEST(VkColorMask, SuppliedShaderMaskIntersectsTargetMask) {
  EXPECT_EQ(gpu::vk::ColorWriteMask(0x737, 0x215, 7, 0), 5u);
  EXPECT_EQ(gpu::vk::ColorWriteMask(0x737, 0x215, 7, 1), 1u);
  EXPECT_EQ(gpu::vk::ColorWriteMask(0x737, 0x215, 7, 2), 2u);
  EXPECT_EQ(gpu::vk::ColorWriteMask(0x737, 0xf, 7, 1), 0u);
}
