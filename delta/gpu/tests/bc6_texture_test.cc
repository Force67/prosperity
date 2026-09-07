#include <array>

#include <gtest/gtest.h>

#include "gpu/ps5/rdna/rdna_resource.h"
#include "gpu/rhi/renderer.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_format.h"
#include "gpu/vulkan/vk_texture_cache.h"

// The renderer links the PS4 command processor; this test issues no EOPs.
extern "C" void prosperity_gpu_end_of_pipe() {}

TEST(Bc6Texture, ObservedHdrBackgroundDescriptorIsValid) {
  // Astro's actual background texture, previously rejected into white.
  u32 descriptor[8] = {
      0x8096fe00, 0xcb300000, 0x01ffc3ff, 0x909003ac, 0, 0, 0, 0};
  for (u32 format : {179u, 180u}) {
    descriptor[1] = (descriptor[1] & ~(0x1ffu << 20)) | (format << 20);
    const auto image = gpu::rdna::DecodeTImage(descriptor, false);
    ASSERT_TRUE(image.valid);
    EXPECT_EQ(image.base, 0x8096fe0000ull);
    EXPECT_EQ(image.width, 4096u);
    EXPECT_EQ(image.height, 2048u);
    EXPECT_EQ(image.dfmt, 40u);
    EXPECT_EQ(image.nfmt, format == 179 ? 0u : 1u);
    const auto host = gpu::vk::GuestTextureFormat(image.dfmt, image.nfmt);
    EXPECT_EQ(host, format == 179 ? VK_FORMAT_BC6H_UFLOAT_BLOCK
                                  : VK_FORMAT_BC6H_SFLOAT_BLOCK);
    EXPECT_TRUE(gpu::vk::FormatBlockCompressed(host));
    EXPECT_TRUE(gpu::vk::GuestFormatBlockCompressed(image.dfmt));
    EXPECT_EQ(gpu::vk::GuestFormatElemBytes(image.dfmt), 16u);
  }
}

TEST(Bc6Texture, NativeUploadsSupportSignedUnsignedTiledMipArrays) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(gpu::vk::g_dev.phys, &features);
  if (!features.textureCompressionBC)
    GTEST_SKIP() << "Device does not support BC textures";
  ASSERT_TRUE(gpu::vk::CreateTextureDescriptors());
  // Zero BC6 blocks encode black. Separate allocations avoid cache aliasing.
  alignas(65536) static std::array<std::array<u8, 1048576>, 4> images{};
  u32 allocation = 0;
  for (u32 nfmt : {0u, 1u}) {
    for (u32 tiling : {8u, 0x109u}) {
      const auto set = gpu::vk::GetTexture(
          reinterpret_cast<u64>(images[allocation++].data()),
          16, 8, 40, nfmt, tiling, 16, 2, 0, 2, 3, 0, 3,
          0, false, nullptr, false, true);
      EXPECT_NE(set, VK_NULL_HANDLE) << "nfmt=" << nfmt << " tiling=" << tiling;
    }
  }
}
