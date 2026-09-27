
#include <gtest/gtest.h>

#include "base/containers/array.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include "gpu/render/device.h"
#include "gpu/render/frame.h"
#include "gpu/render/guest_format.h"
#include "gpu/render/renderer.h"
#include "gpu/render/texture_cache.h"

// The renderer links the PS4 command processor; this test issues no EOPs.
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" void prosperity_gpu_end_of_pipe() {}

TEST(Bc6Texture, ObservedHdrBackgroundDescriptorIsValid) {
  // Astro's actual background texture, previously rejected into white.
  u32 descriptor[8] = {0x8096fe00, 0xcb300000, 0x01ffc3ff, 0x909003ac,
                       0,          0,          0,          0};
  for (u32 format : {179u, 180u}) {
    descriptor[1] = (descriptor[1] & ~(0x1ffu << 20)) | (format << 20);
    const auto image = gpu::rdna::DecodeTImage(descriptor, false);
    ASSERT_TRUE(image.valid);
    EXPECT_EQ(image.base, 0x8096fe0000ull);
    EXPECT_EQ(image.width, 4096u);
    EXPECT_EQ(image.height, 2048u);
    EXPECT_EQ(image.dfmt, 40u);
    EXPECT_EQ(image.nfmt, format == 179 ? 0u : 1u);
    const auto host = gpu::render::GuestTextureFormat(image.dfmt, image.nfmt);
    EXPECT_EQ(host, format == 179 ? gpu::rhi::Format::kBC6HUfloat
                                  : gpu::rhi::Format::kBC6HSfloat);
    EXPECT_TRUE(gpu::render::FormatBlockCompressed(host));
    EXPECT_TRUE(gpu::render::GuestFormatBlockCompressed(image.dfmt));
    EXPECT_EQ(gpu::render::GuestFormatElemBytes(image.dfmt), 16u);
  }
}

TEST(Bc6Texture, NativeUploadsSupportSignedUnsignedTiledMipArrays) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  if (!gpu::render::Device().SupportsFormat(gpu::rhi::Format::kBC6HUfloat,
                                            gpu::rhi::kTextureSampled))
    GTEST_SKIP() << "Device does not support BC textures";
  ASSERT_TRUE(gpu::render::CreateTextureDescriptors());
  // Zero BC6 blocks encode black. Separate allocations avoid cache aliasing.
  alignas(65536) static base::Array<base::Array<u8, 1048576>, 4> images{};
  u32 allocation = 0;
  for (u32 nfmt : {0u, 1u}) {
    for (u32 tiling : {8u, 0x109u}) {
      const auto set = gpu::render::GetTexture(
          reinterpret_cast<u64>(images[allocation++].data()), 16, 8, 40, nfmt,
          tiling, 16, 2, 0, 2, 3, 0, 3, 0, false, nullptr, false, true);
      EXPECT_NE(set, nullptr) << "nfmt=" << nfmt << " tiling=" << tiling;
    }
  }
}
