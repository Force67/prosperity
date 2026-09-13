#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>

#include <gtest/gtest.h>

#include "gpu/gcn/gcn_detile.h"
#include "gpu/ps5/compute_dispatch.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/vulkan/vk_tiling.h"

namespace {
TEST(Gfx10GpuTiling, MatchesCpuAcrossMipsLayersAndPadding) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  for (u32 mode :
       {1u, 2u, 5u, 6u, 9u, 10u, 17u, 18u, 21u, 22u, 24u, 25u, 26u, 27u}) {
    for (u32 elem : {1u, 2u, 4u, 8u, 16u}) {
      SCOPED_TRACE(testing::Message() << "mode=" << mode << " elem=" << elem);
      gpu::gcn::TextureLayout32 tiled, linear;
      const u32 stage_elem = std::max(4u, elem);
      const u32 layers = mode == 27 && elem == 8 ? 72 : 3;
      ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(tiled, 263, 137, 271, layers,
                                                 6, 0x100 + mode, false, elem));
      ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(linear, 263, 137, 271, layers,
                                                 6, 8, false, stage_elem));
      std::vector<u8> source(tiled.size), expected(linear.size, 0);
      for (size_t i = 0; i < source.size(); i++)
        source[i] = static_cast<u8>((i * 137) ^ (i >> 7) ^ (i >> 15));
      for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
        const auto& l = linear.mips[mip];
        for (u32 layer = 0; layer < tiled.layers; layer++) {
          const u64 offset =
              l.offset + u64(layer) * l.pitch * l.stored_height * stage_elem;
          std::vector<u8> tight(u64(l.width) * l.height * elem);
          ASSERT_TRUE(gpu::gcn::DetileTextureMip32Pitched(
              source.data(), tight.data(), l.width * elem, tiled,
              mip, layer));
          for (u32 y = 0; y < l.height; y++)
            for (u32 x = 0; x < l.width; x++)
              std::memcpy(expected.data() + offset +
                              (u64(y) * l.pitch + x) * stage_elem,
                          tight.data() + (u64(y) * l.width + x) * elem, elem);
        }
      }
      std::vector<u8> actual(linear.size, 0x5a);
      ASSERT_TRUE(gpu::vk::ConvertGfx10Image(tiled, linear, source.data(),
                                             actual.data(), true));
      ASSERT_EQ(actual, expected);
      for (size_t i = 0; i < actual.size(); i++)
        actual[i] ^= 0x97;
      std::vector<u8> retiled(tiled.size, 0xa5), reference = retiled;
      for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
        const auto& l = linear.mips[mip];
        for (u32 layer = 0; layer < tiled.layers; layer++) {
          const u64 offset =
              l.offset + u64(layer) * l.pitch * l.stored_height * stage_elem;
          std::vector<u8> tight(u64(l.width) * l.height * elem);
          for (u32 y = 0; y < l.height; y++)
            for (u32 x = 0; x < l.width; x++)
              std::memcpy(tight.data() + (u64(y) * l.width + x) * elem,
                          actual.data() + offset +
                              (u64(y) * l.pitch + x) * stage_elem,
                          elem);
          ASSERT_TRUE(gpu::gcn::RetileTextureMip32Pitched(
              tight.data(), l.width * elem, reference.data(), tiled,
              mip, layer));
        }
      }
      ASSERT_TRUE(gpu::vk::ConvertGfx10Image(tiled, linear, actual.data(),
                                             retiled.data(), false));
      ASSERT_EQ(retiled, reference);
    }
  }
}

// Run through descriptor resolution, SPIR-V, Vulkan, and guest writeback.
// Three adjacent texels catch incorrect per-channel strides. A sparse DMASK
// must leave the other channels untouched.
class RdnaComputeImage : public testing::TestWithParam<u32> {};

TEST_P(RdnaComputeImage, Raw32LoadStorePreservesUnwrittenChannels) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";

  // Distinct guest allocations prevent the backend's per-frame resource
  // cache from treating a later test's CPU initialization as unchanged data.
  alignas(65536) static std::array<std::array<u32, 16384>, 6> sources{}, dests{};
  const u32 format = GetParam();
  const u32 allocation = format < 65 ? format - 62 : format - 72;
  auto& source = sources[allocation];
  auto& dest = dests[allocation];
  alignas(256) static std::array<u32, 4096> code{};
  constexpr std::array<u32, 12> values = {
      0x3e800000, 0xc0000000, 0x40800000, 0x3f800000,
      0x3f000000, 0xbf800000, 0x41000000, 0x3f800000,
      0x3f400000, 0xc0400000, 0x41800000, 0x3f800000};
  std::copy(values.begin(), values.end(), source.begin());
  constexpr u32 untouched = 0x3f123456;
  const u32 channels = format < 65 ? 2 : 4;
  dest.fill(untouched);
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(source.data()), sizeof(source));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));

  // v1=0; image_load v[4:7], v[0:1], s[0:7];
  // image_store v[4:5], v[0:1], s[8:15], dmask:0x5; s_endpgm.
  code[0] = 0x7e020280;
  code[1] = 0xf0000000 | (((1u << channels) - 1) << 8);
  code[2] = 0x00000400;
  code[3] = 0xf0200000 | ((channels == 4 ? 5u : 1u) << 8);
  code[4] = 0x00020400;
  code[5] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();

  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 3;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
  const auto descriptor = [&](u32 first, const void* memory) {
    const u64 address = reinterpret_cast<u64>(memory);
    regs[first] = address >> 8;
    regs[first + 1] = ((address >> 40) & 0xff) | (format << 20) | (2u << 30);
    regs[first + 2] = 0;  // width=3, height=1
    regs[first + 3] = 0x80000fac;  // 1D, linear, RGBA
  };
  descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, source.data());
  descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, dest.data());
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
  for (u32 x = 0; x < 3; x++) {
    EXPECT_EQ(dest[x * channels], values[x * channels]) << "texel " << x;
    EXPECT_EQ(dest[x * channels + 1], untouched) << "texel " << x;
    if (channels == 4) {
      EXPECT_EQ(dest[x * 4 + 2], values[x * 4 + 1]) << "texel " << x;
      EXPECT_EQ(dest[x * 4 + 3], untouched) << "texel " << x;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(RgAndRgbaUintSintFloat, RdnaComputeImage,
                        testing::Values(62u, 63u, 64u, 75u, 76u, 77u));

TEST(RdnaComputeImageConversion, R8UnormStoreWritesBackByteImage) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static std::array<u32, 16384> source{};
  alignas(65536) static std::array<u8, 65536> dest{};
  alignas(256) static std::array<u32, 4096> code{};
  source[0] = std::bit_cast<u32>(0.f);
  source[1] = std::bit_cast<u32>(1.f);
  source[2] = std::bit_cast<u32>(64.f / 255.f);
  dest.fill(0xa5);
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(source.data()), sizeof(source));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  code[0] = 0x7e020280;  // v1=0
  code[1] = 0xf0000100; code[2] = 0x00000400;  // image_load R32_FLOAT
  code[3] = 0xf0200100; code[4] = 0x00020400;  // image_store R8_UNORM
  code[5] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 3;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
  const auto descriptor = [&](u32 first, const void* memory, u32 format) {
    const u64 address = reinterpret_cast<u64>(memory);
    regs[first] = address >> 8;
    regs[first + 1] = ((address >> 40) & 0xff) | (format << 20) | (2u << 30);
    regs[first + 3] = 0x80000fac;  // width=3, 1D linear
  };
  descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, source.data(), 22);
  descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, dest.data(), 1);
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
  EXPECT_EQ(dest[0], 0);
  EXPECT_EQ(dest[1], 255);
  EXPECT_EQ(dest[2], 64);
  EXPECT_EQ(dest[3], 0xa5);  // no overwrite of the adjacent guest byte
}

class RdnaR16Unorm : public testing::TestWithParam<u32> {};

TEST(RdnaComputeImageConversion, LargeNarrowStoresPreserveUntouchedTexels) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static std::array<std::array<u8, 1024 * 1024>, 4> memory{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(memory.data()), sizeof(memory));
  alignas(256) static std::array<u32, 4096> code{};
  code[0] = 0x7e020280;  // v1=0
  code[1] = 0xf0000108; code[2] = 0x00000400;  // image_load 2D
  code[3] = 0xf0200108; code[4] = 0x00020400;  // image_store 2D
  code[5] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  constexpr u32 width = 257, height = 256;
  for (u32 elem : {1u, 2u}) {
    auto& source = memory[(elem - 1) * 2];
    auto& dest = memory[(elem - 1) * 2 + 1];
    const float values[] = {-1.f, 2.f, .5f};
    std::memcpy(source.data(), values, sizeof(values));
    dest.fill(0xa5);
    gpu::gcn::TextureLayout32 layout;
    ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(
        layout, width, height, width, 1, 1, 0x118, false, elem));
    ASSERT_LE(layout.size, dest.size());
    std::vector<u8> tight(u64(width) * height * elem, 0xa5);
    const u32 expected_values[] = {0, elem == 1 ? 255u : 65535u,
                                     elem == 1 ? 128u : 32768u};
    for (u32 x = 0; x < 3; x++)
      std::memcpy(tight.data() + x * elem, &expected_values[x], elem);
    std::vector<u8> expected(dest.begin(), dest.end());
    ASSERT_TRUE(gpu::gcn::RetileTextureMip32(
        tight.data(), expected.data(), layout, 0, 0));
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 3;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
    const auto descriptor = [&](u32 first, const void* data, u32 format,
                                u32 swizzle) {
      const u64 address = reinterpret_cast<u64>(data);
      regs[first] = address >> 8;
      regs[first + 1] = ((address >> 40) & 0xff) | (format << 20) |
                        (((width - 1) & 3) << 30);
      regs[first + 2] = ((width - 1) >> 2) | ((height - 1) << 14);
      regs[first + 3] = 0x90000fac | (swizzle << 20);
    };
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, source.data(), 22, 0);
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, dest.data(),
                elem == 1 ? 1 : 7, 24);
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    EXPECT_EQ(std::vector<u8>(dest.begin(), dest.end()), expected)
        << "elem=" << elem;
  }
}

TEST_P(RdnaR16Unorm, MipArrayLoadAndClampedStorePreserveAdjacentTexels) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  // Include the 64 KB Z swizzle used by Astro's shadow images. Distinct
  // allocations keep the backend cache from reusing a previous test's data.
  alignas(65536) static std::array<std::array<u8, 131072>, 8> memory{};
  const u32 swizzle = GetParam();
  const u32 first = swizzle ? 4 : 0;
  auto& packed = memory[first];
  auto& loaded = memory[first + 1];
  auto& floats = memory[first + 2];
  auto& stored = memory[first + 3];
  gpu::gcn::TextureLayout32 layout16, layout32;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(layout16, 4, 2, 4, 2, 3,
                                             0x100 | swizzle, false, 2));
  ASSERT_TRUE(
      gpu::gcn::BuildTextureLayout32(layout32, 4, 2, 4, 2, 3, 0x100, false, 4));
  ASSERT_LE(layout16.size, packed.size());
  const std::array<u16, 8> values = {0,     1,     32768, 65535,
                                     16384, 49151, 42,    60000};
  const std::array<float, 8> input = {-1.f, 2.f, .5f,  .25f,
                                      0.f,  1.f, .75f, .125f};
  const std::array<u16, 8> sentinel = {0xa5a5, 0xa5a5, 0xa5a5, 0xa5a5,
                                       0xa5a5, 0xa5a5, 0xa5a5, 0xa5a5};
  for (u32 mip = 0; mip < 3; ++mip) {
    for (u32 layer = 0; layer < 2; ++layer) {
      ASSERT_TRUE(gpu::gcn::RetileTextureMip32(values.data(), packed.data(),
                                               layout16, mip, layer));
      ASSERT_TRUE(gpu::gcn::RetileTextureMip32(input.data(), floats.data(),
                                               layout32, mip, layer));
      ASSERT_TRUE(gpu::gcn::RetileTextureMip32(sentinel.data(), stored.data(),
                                               layout16, mip, layer));
    }
  }
  alignas(256) static std::array<u32, 4096> code{};
  // image_load/store with DIM=2D_ARRAY: v0=x, v1=y, v2=layer.
  code[0] = 0xf0000128;
  code[1] = 0x00000400;
  code[2] = 0xf0200128;
  code[3] = 0x00020400;
  code[4] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  for (u32 mip = 0; mip < 3; ++mip) {
    const auto transfer = [&](auto& src, u32 sfmt, u32 stile, auto& dst,
                              u32 dfmt, u32 dtile) {
      gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(src.data()), src.size());
      gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dst.data()), dst.size());
      gpu::ps5::Regs regs;
      const u64 pc = reinterpret_cast<u64>(code.data());
      regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
      regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
      // Leave the fourth column untouched to catch texel/row stride mistakes.
      regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] =
          mip ? std::max(4u >> mip, 1u) : 3;
      regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = std::max(2u >> mip, 1u);
      regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 2;
      regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
      const auto descriptor = [&](u32 first, const void* data, u32 fmt,
                                  u32 tile) {
        const u64 address = reinterpret_cast<u64>(data);
        regs[first] = address >> 8;
        regs[first + 1] = ((address >> 40) & 0xff) | (fmt << 20) | (3u << 30);
        regs[first + 2] = 1 << 14;                                  // 4x2
        regs[first + 3] = 0xd0020fac | (mip << 12) | (tile << 20);  // 2D array
        regs[first + 4] = 1;       // two layers
        regs[first + 5] = 2 << 4;  // three physical mip levels
      };
      descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, src.data(), sfmt, stile);
      descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, dst.data(), dfmt, dtile);
      const u32 dispatch[] = {1, 1, 1, 1};
      gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
      ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    };
    transfer(packed, 7, swizzle, loaded, 22, 0);
    transfer(floats, 22, 0, stored, 7, swizzle);
    for (u32 layer = 0; layer < 2; ++layer) {
      std::array<float, 8> result{};
      std::array<u16, 8> output{};
      ASSERT_TRUE(gpu::gcn::DetileTextureMip32(loaded.data(), result.data(),
                                               layout32, mip, layer));
      ASSERT_TRUE(gpu::gcn::DetileTextureMip32(stored.data(), output.data(),
                                               layout16, mip, layer));
      const std::array<u16, 8> expected = {0, 65535, 32768, 0xa5a5,
                                           0, 65535, 49151, 0xa5a5};
      for (u32 i = 0; i < std::max(4u >> mip, 1u) * std::max(2u >> mip, 1u);
           ++i) {
        if (i % 4 != 3)
          EXPECT_NEAR(result[i], values[i] / 65535.f, 1e-7f);
        EXPECT_EQ(output[i], expected[i])
            << "mip=" << mip << " layer=" << layer << " texel=" << i;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(LinearAndShadowTiling,
                         RdnaR16Unorm,
                         testing::Values(0u, 24u));

// BC6 mode 11 stores two explicit 10-bit RGB endpoints. Equal endpoints
// make a constant block, independent of the interpolation indices.
class RdnaBc6Image : public testing::TestWithParam<u32> {};

TEST_P(RdnaBc6Image, HdrLoadAndCubeSamplePreserveBlocksMipsAndLayers) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  const u32 param = GetParam();
  const bool is_signed = param & 1;
  const bool cube = param & 4;
  const u32 layer_count = cube ? 6 : 2;
  const u32 tile = param & 2 ? 5 : 0;
  alignas(65536) static std::array<std::array<u8, 131072>, 8> inputs{},
      outputs{};
  auto& input = inputs[param];
  auto& output = outputs[param];
  gpu::gcn::TextureLayout32 blocks, pixels;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(blocks, 2, 2, 2, layer_count, 4,
                                             0x100 | tile, false, 16));
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(pixels, 8, 8, 8, layer_count, 4,
                                             0x100, false, 16));
  ASSERT_LE(blocks.size, input.size());
  ASSERT_LE(pixels.size, output.size());
  const u32 quantized[3] = {0, is_signed ? 256u : 512u,
                            is_signed ? 768u : 1023u};
  const float expected[3] = {0.f, is_signed ? 1.5302734375f : 1.5146484375f,
                             is_signed ? -1.5302734375f : 65504.f};
  for (u32 mip = 0; mip < 4; ++mip) {
    for (u32 layer = 0; layer < layer_count; ++layer) {
      std::array<u8, 64> tight{};
      const auto& level = blocks.mips[mip];
      for (u32 b = 0; b < level.width * level.height; ++b) {
        u8* encoded = tight.data() + b * 16;
        u32 bit = 0;
        const auto put = [&](u32 value, u32 count) {
          for (u32 i = 0; i < count; ++i, ++bit)
            encoded[bit / 8] |= ((value >> i) & 1) << (bit % 8);
        };
        put(3, 5);
        for (u32 endpoint = 0; endpoint < 2; ++endpoint)
          for (u32 c = 0; c < 3; ++c)
            put(quantized[(mip + layer + b + c) % 3], 10);
      }
      ASSERT_TRUE(gpu::gcn::RetileTextureMip32(tight.data(), input.data(),
                                               blocks, mip, layer));
    }
  }
  const auto original = input;
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(inputs.data()), sizeof(inputs));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(outputs.data()), sizeof(outputs));
  alignas(256) static std::array<u32, 4096> code{};
  code[0] = 0xf0000f28;
  code[1] = 0x00000400;  // image_load RGBA, 2D array
  code[2] = 0xf0200f28;
  code[3] = 0x00020400;  // image_store RGBA32_FLOAT
  code[4] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  for (u32 mip = 0; mip < 4; ++mip) {
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    const u32 dim = 8u >> mip;
    if (cube) {
      const u32 scale = std::bit_cast<u32>(1.f / dim);
      const u32 bias = std::bit_cast<u32>(1.f + .5f / dim);
      const u32 sample_code[] = {
          0x7e100d00, 0x7e120d01,
          0x7e140d02,  // float x, y, face
          0x101010ff, scale,
          0x061010ff, std::bit_cast<u32>(1.f + 1.f / dim),
          0x101212ff, scale,
          0x061212ff, bias,
          0x7e160280,              // LOD 0, relative to the descriptor base mip
          0xf0900f18, 0x00800408,  // image_sample_l cube
          0xf0200f28, 0x00020400,  // image_store 2D array
          0xbf810000};
      std::copy(std::begin(sample_code), std::end(sample_code), code.begin());
      gpu::rdna::NextProgramGeneration();
    }
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = dim;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = dim;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = layer_count;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
    const auto descriptor = [&](u32 first, const void* data, u32 fmt, u32 sw) {
      const u64 address = reinterpret_cast<u64>(data);
      regs[first] = address >> 8;
      regs[first + 1] = ((address >> 40) & 0xff) | (fmt << 20) | (3u << 30);
      regs[first + 2] = 1 | (7 << 14);  // 8x8 texels
      regs[first + 3] = 0xd0030fac | (mip << 12) | (sw << 20);
      regs[first + 4] = layer_count - 1;
      if (cube && first == gpu::ps5::mmCOMPUTE_USER_DATA_0) {
        regs[first + 3] = (regs[first + 3] & 0x0fffffff) | 0xb0000000;
        regs[first + 4] = 0;  // a single cube still occupies all six faces
      }
      regs[first + 5] = 3 << 4;  // four physical mip levels
    };
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, input.data(),
               is_signed ? 180 : 179, tile);
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, output.data(), 77, 0);
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    for (u32 layer = 0; layer < layer_count; ++layer) {
      std::array<float, 8 * 8 * 4> result{};
      ASSERT_TRUE(gpu::gcn::DetileTextureMip32(output.data(), result.data(),
                                               pixels, mip, layer));
      for (u32 y = 0; y < dim; ++y) {
        for (u32 x = 0; x < dim; ++x) {
          const u32 b = (y / 4) * blocks.mips[mip].width + x / 4;
          for (u32 c = 0; c < 4; ++c) {
            float value = c == 3 ? 1.f : expected[(mip + layer + b + c) % 3];
            // Sample half a texel to the right; the two neighboring blocks
            // contribute equally at their boundary. This checks HDR filtering.
            if (cube && dim == 8 && x == 3 && c != 3)
              value = .5f * (value + expected[(mip + layer + b + 1 + c) % 3]);
            EXPECT_FLOAT_EQ(result[(y * dim + x) * 4 + c], value)
                << "mip=" << mip << " layer=" << layer << " xy=" << x << ','
                << y;
          }
        }
      }
    }
  }
  EXPECT_EQ(input,
            original);  // Sampling cannot write decompressed texels back.
}

INSTANTIATE_TEST_SUITE_P(SignedUnsignedLinearTiled,
                         RdnaBc6Image,
                         testing::Values(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u));

TEST(RdnaComputeImageConversion, Rgba16UnormLoadAndMaskedClampedStore) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static std::array<u32, 16384> packed{}, floats{}, result{}, output{};
  packed[0] = 0xffff0000;
  packed[2] = 0x80004000;
  packed[4] = 0x0001ffff;
  result.fill(0x12345678);
  output.fill(0x76543210);
  const std::array<float, 12> values = {
      -1.f, 2.f, 0.f, 0.f, 0.5f, 0.25f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f};
  for (u32 i = 0; i < values.size(); i++)
    floats[i] = std::bit_cast<u32>(values[i]);
  alignas(256) static std::array<u32, 4096> code{};
  code[0] = 0x7e020280;
  code[1] = 0xf0000f00; code[2] = 0x00000400;
  code[3] = 0xf0200500; code[4] = 0x00020400;
  code[5] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  const auto transfer = [&](const void* src, u32 sfmt, void* dst, u32 dfmt) {
    gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(src), 65536);
    gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dst), 65536);
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 3;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
    const auto descriptor = [&](u32 first, const void* memory, u32 format) {
      const u64 address = reinterpret_cast<u64>(memory);
      regs[first] = address >> 8;
      regs[first + 1] = ((address >> 40) & 0xff) | (format << 20) | (2u << 30);
      regs[first + 3] = 0x80000fac;
    };
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, src, sfmt);
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, dst, dfmt);
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    return gpu::rhi::FlushCsWrites(renderer);
  };
  ASSERT_TRUE(transfer(packed.data(), 65, result.data(), 77));
  for (u32 x = 0; x < 3; x++) {
    EXPECT_FLOAT_EQ(std::bit_cast<float>(result[x * 4]),
                    float(packed[x * 2] & 0xffff) / 65535.f);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(result[x * 4 + 2]),
                    float(packed[x * 2] >> 16) / 65535.f);
    EXPECT_EQ(result[x * 4 + 1], 0x12345678u);
    EXPECT_EQ(result[x * 4 + 3], 0x12345678u);
  }
  ASSERT_TRUE(transfer(floats.data(), 77, output.data(), 65));
  const std::array<u32, 6> expected = {
      0x76540000, 0x7654ffff, 0x76548000, 0x76544000, 0x7654ffff, 0x76540000};
  for (u32 i = 0; i < expected.size(); i++)
    EXPECT_EQ(output[i], expected[i]) << "dword " << i;
}

TEST(RdnaComputeAlu, Scalar64CompareUsesHighDwordAndInlineSignExtension) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static std::array<u32, 16384> output{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(output.data()), sizeof(output));
  u32 pc_dw = 0;
  const auto compare = [&](u32 op, u32 a, u32 b, u32 dst) {
    code[pc_dw++] = 0xbf000000 | (op << 16) | (b << 8) | a;
    code[pc_dw++] = 0x85108081;  // s_cselect_b32 s16, 1, 0
    code[pc_dw++] = 0x7e000210 | (dst << 17);  // v_mov_b32 vdst, s16
  };
  compare(0x12, 8, 10, 4);   // same low half, different high half
  compare(0x13, 8, 10, 5);
  compare(0x12, 12, 193, 6);  // -1 inline is sign extended to 64 bits
  compare(0x13, 12, 193, 7);
  // The VCC-targeting SDWA compare must not invalidate the output T# in s0.
  code[pc_dw++] = 0x7c041ef9; code[pc_dw++] = 0x86860080;
  code[pc_dw++] = 0xf0200f00; code[pc_dw++] = 0x00000400;
  code[pc_dw++] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
  const u64 address = reinterpret_cast<u64>(output.data());
  const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
  regs[ud] = address >> 8;
  regs[ud + 1] = ((address >> 40) & 0xff) | (75 << 20);
  regs[ud + 3] = 0x80000fac;
  regs[ud + 9] = 1;
  regs[ud + 11] = 2;
  regs[ud + 12] = regs[ud + 13] = 0xffffffff;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
  const std::array<u32, 4> expected = {0, 1, 1, 0};
  for (u32 i = 0; i < 4; ++i)
    EXPECT_EQ(output[i], expected[i]);
}

TEST(RdnaComputeAlu, XadUsesXorBeforeWrappingAdd) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";

  alignas(65536) static std::array<u32, 16384> dest{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  // v_xad_u32 v4, v0, -1, 2; image_store v4, v0, s[0:7]; s_endpgm.
  code[0] = 0xd7450004;
  code[1] = 256 | (193 << 9) | (130 << 18);
  code[2] = 0xf0200100;
  code[3] = 0x00000400;
  code[4] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 3;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 8 << 1;
  const u64 address = reinterpret_cast<u64>(dest.data());
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0] = address >> 8;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 1] =
      ((address >> 40) & 0xff) | (20 << 20) | (2u << 30);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 3] = 0x80000fac;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
  EXPECT_EQ(dest[0], 1u);
  EXPECT_EQ(dest[1], 0u);
  EXPECT_EQ(dest[2], 0xffffffffu);
}

TEST(RdnaComputeAlu, BitfieldMaskWrapsWidthAndOffset) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static std::array<u32, 16384> dest{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  u32 pc_dw = 0;
  for (u32 i = 0; i < 4; i++) {
    code[pc_dw++] = 0xd7630000 | (4 + i);
    code[pc_dw++] = (8 + i * 2) | ((9 + i * 2) << 9);
  }
  code[pc_dw++] = 0xf0200f00; code[pc_dw++] = 0x00000400;
  code[pc_dw++] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
  const u64 address = reinterpret_cast<u64>(dest.data());
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0] = address >> 8;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 1] =
      ((address >> 40) & 0xff) | (77 << 20);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 3] = 0x80000fac;
  const std::array<u32, 8> operands = {0, 0, 31, 0, 36, 40, 8, 28};
  for (u32 i = 0; i < operands.size(); i++)
    regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8 + i] = operands[i];
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
  const std::array<u32, 4> expected = {0, 0x7fffffff, 0xf00, 0xf0000000};
  for (u32 i = 0; i < expected.size(); i++)
    EXPECT_EQ(dest[i], expected[i]);
}

TEST(RdnaComputeAlu, FmaMixSelectsPrecisionHalvesAndSourceModifiers) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static std::array<u32, 16384> dest{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  u32 pc_dw = 0;
  const auto mix = [&](u32 dst, u32 a, u32 b, u32 c,
                       u32 half, u32 select, u32 abs, u32 neg, bool clamp) {
    code[pc_dw++] = 0xcc200000 | dst | (abs << 8) | (select << 11) |
                      ((half & 4) << 12) | (u32(clamp) << 15);
    code[pc_dw++] = a | (b << 9) | (c << 18) | ((half & 3) << 27) | (neg << 29);
  };
  mix(4, 8, 9, 10, 6, 6, 0, 4, false);  // 2*3 - 1 = 5
  mix(5, 8, 9, 10, 6, 0, 2, 0, false);  // 2*abs(-4) - 2 = 6
  mix(6, 8, 242, 240, 0, 0, 0, 0, true);  // clamp(2*1 + .5) = 1
  mix(7, 242, 9, 10, 6, 6, 0, 4, false);  // 1*3 - 1 = 2
  code[pc_dw++] = 0xf0200f00; code[pc_dw++] = 0x00000400;
  code[pc_dw++] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
  const u64 address = reinterpret_cast<u64>(dest.data());
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0] = address >> 8;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 1] =
      ((address >> 40) & 0xff) | (77 << 20);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 3] = 0x80000fac;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8] = std::bit_cast<u32>(2.f);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 9] = 0x4200c400;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 10] = 0x3c00c000;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
  const std::array<float, 4> expected = {5.f, 6.f, 1.f, 2.f};
  for (u32 i = 0; i < expected.size(); i++)
    EXPECT_FLOAT_EQ(std::bit_cast<float>(dest[i]), expected[i]);
}

TEST(RdnaComputeAlu, FmaMixHalfResultsPreserveTheOtherHalf) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static std::array<u32, 16384> dest{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  u32 pc_dw = 0;
  const auto mix = [&](u32 op, u32 dst, u32 a, u32 b, u32 c,
                       u32 half, u32 select, u32 abs, u32 neg, bool clamp) {
    code[pc_dw++] = 0xcc000000 | (op << 16) | dst | (abs << 8) | (select << 11) |
                      ((half & 4) << 12) | (u32(clamp) << 15);
    code[pc_dw++] = a | (b << 9) | (c << 18) | ((half & 3) << 27) | (neg << 29);
  };
  for (u32 dst = 4; dst <= 7; dst++) {
    code[pc_dw++] = 0x7e0002ff | (dst << 17);
    code[pc_dw++] = 0x12345678;
  }
  mix(0x21, 4, 8, 9, 10, 6, 6, 0, 4, false);
  mix(0x22, 5, 8, 9, 10, 6, 0, 2, 0, false);
  mix(0x21, 6, 8, 242, 240, 0, 0, 0, 0, true);
  mix(0x22, 6, 242, 9, 10, 6, 6, 0, 4, false);
  mix(0x22, 7, 8, 242, 240, 0, 0, 0, 0, true);
  code[pc_dw++] = 0xf0200f00; code[pc_dw++] = 0x00000400;
  code[pc_dw++] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
  const u64 address = reinterpret_cast<u64>(dest.data());
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0] = address >> 8;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 1] =
      ((address >> 40) & 0xff) | (77 << 20);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 3] = 0x80000fac;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8] = std::bit_cast<u32>(2.f);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 9] = 0x4200c400;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 10] = 0x3c00c000;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
  const std::array<u32, 4> expected = {0x12344500, 0x46005678, 0x40003c00, 0x3c005678};
  for (u32 i = 0; i < expected.size(); i++)
    EXPECT_EQ(dest[i], expected[i]);
}

TEST(RdnaComputeAlu, PermlaneSelectsRowsAndHonorsInactiveLanes) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static std::array<u32, 16384> dest{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  u32 pc_dw = 0;
  const auto permute = [&](u32 op, u32 dst, u32 flags) {
    code[pc_dw++] = 0xd4000000 | (op << 16) | (flags << 11) | dst;
    code[pc_dw++] = 256 | (8 << 9) | (9 << 18);
  };
  for (u32 dst = 4; dst <= 7; dst++) {
    code[pc_dw++] = 0x7e0002ff | (dst << 17);
    code[pc_dw++] = 0x12345678;
  }
  // Only even lanes execute; the reversed selectors always fetch odd lanes.
  code[pc_dw++] = 0xbefe03ff; code[pc_dw++] = 0x55555555;
  code[pc_dw++] = 0xbeff03ff; code[pc_dw++] = 0x55555555;
  permute(0x377, 4, 1);
  permute(0x378, 5, 1);
  permute(0x378, 6, 0);
  permute(0x378, 7, 2);
  code[pc_dw++] = 0xbefe03c1;
  code[pc_dw++] = 0xbeff03c1;
  code[pc_dw++] = 0xf0200f00; code[pc_dw++] = 0x00000400;
  code[pc_dw++] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 64;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
  const u64 address = reinterpret_cast<u64>(dest.data());
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0] = address >> 8;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 1] =
      ((address >> 40) & 0xff) | (75 << 20) | (3u << 30);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 2] = 15;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 3] = 0x80000fac;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8] = 0x89abcdef;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 9] = 0x01234567;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
  for (u32 lane = 0; lane < 64; lane++) {
    const u32 selected = (lane & ~15u) | (15 - (lane & 15));
    EXPECT_EQ(dest[lane * 4], (lane & 1) ? 0x12345678 : selected) << lane;
    EXPECT_EQ(dest[lane * 4 + 1], (lane & 1) ? 0x12345678 : (selected ^ 16)) << lane;
    EXPECT_EQ(dest[lane * 4 + 2], 0x12345678) << lane;
    EXPECT_EQ(dest[lane * 4 + 3], (lane & 1) ? 0x12345678 : 0u) << lane;
  }
}

TEST(RdnaComputeAlu, DppRowXmaskKeepsEachExchangeInsideItsRow) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static std::array<u32, 16384> dest{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  u32 pc_dw = 0;
  constexpr std::array<u32, 4> masks = {0, 4, 7, 15};
  for (u32 i = 0; i < masks.size(); i++) {
    code[pc_dw++] = 0x7e0002fa | ((4 + i) << 17);
    code[pc_dw++] = 0xff080000 | ((0x160 + masks[i]) << 8);
  }
  code[pc_dw++] = 0xf0200f00; code[pc_dw++] = 0x00000400;
  code[pc_dw++] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 64;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
  const u64 address = reinterpret_cast<u64>(dest.data());
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0] = address >> 8;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 1] =
      ((address >> 40) & 0xff) | (75 << 20) | (3u << 30);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 2] = 15;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 3] = 0x80000fac;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8] = 0x89abcdef;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 9] = 0x01234567;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
  for (u32 lane = 0; lane < 64; lane++)
    for (u32 i = 0; i < masks.size(); i++)
      EXPECT_EQ(dest[lane * 4 + i], lane ^ masks[i]) << lane << ":" << i;
}


TEST(RdnaComputeImage, GatherOffsetAndCompareUseFourDepthTaps) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static std::array<u32, 16384> source{};
  alignas(65536) static std::array<std::array<u32, 16384>, 10> outputs{};
  alignas(256) static std::array<u32, 4096> code{};
  for (u32 i = 0; i < 16; i++)
    source[(i / 4) * 64 + i % 4] = std::bit_cast<u32>(i / 16.f);
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(source.data()), sizeof(source));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(outputs.data()), sizeof(outputs));
  for (u32 func = 0; func < outputs.size(); func++) {
    auto& dest = outputs[func];
    u32 n = 0;
    const auto mov = [&](u32 vgpr, u32 value) {
      code[n++] = 0x7e0002ff | (vgpr << 17);
      code[n++] = value;
    };
    mov(3, 0x3f01);  // Offset (+1, -1) moves the footprint to (2, 0).
    const bool compare = func < 8;
    if (compare)
      mov(4, std::bit_cast<u32>(3.f / 16));
    mov(compare ? 5 : 4, std::bit_cast<u32>(0.4f));
    mov(compare ? 6 : 5, std::bit_cast<u32>(0.4f));
    code[n++] = 0xbe9803ff; code[n++] = func << 13;
    const u32 op = compare ? 0x5f : func == 8 ? 0x57 : 0x47;
    code[n++] = 0xf0000108 | (op << 18);
    code[n++] = (func == 9 ? 4 : 3) | (8 << 8) | (6 << 21);
    code[n++] = 0xf0200f00; code[n++] = 0x00020800;
    code[n++] = 0xbf810000;
    gpu::rdna::NextProgramGeneration();
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
    const u64 src = reinterpret_cast<u64>(source.data());
    const u64 dst = reinterpret_cast<u64>(dest.data());
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    regs[ud] = src >> 8;
    regs[ud + 1] = ((src >> 40) & 0xff) | (22 << 20) | (3u << 30);
    regs[ud + 2] = 3 << 14;
    regs[ud + 3] = 0x90000fac;
    regs[ud + 8] = dst >> 8;
    regs[ud + 9] = ((dst >> 40) & 0xff) | (77 << 20);
    regs[ud + 11] = 0x80000fac;
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    constexpr std::array<u32, 4> taps = {6, 7, 3, 2};
    for (u32 i = 0; i < taps.size(); i++) {
      const u32 depth = taps[i] + (func == 9 ? 3 : 0);
      const std::array<bool, 8> pass = {
          false, 3 < depth, 3 == depth, 3 <= depth,
          3 > depth, 3 != depth, 3 >= depth, true};
      const float expected = compare ? (pass[func] ? 1.f : 0.f) : depth / 16.f;
      EXPECT_FLOAT_EQ(std::bit_cast<float>(dest[i]), expected)
          << "compare " << func << " tap " << i;
    }
  }
}

TEST(RdnaComputeImageConversion, SrgbReadAndUnormMipWriteShareStorage) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static std::array<u8, 65536> image{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(layout, 8, 8, 8, 1, 4,
                                             0x11b, false, 4));
  ASSERT_LE(layout.size, image.size());
  std::array<u32, 64> pixels;
  pixels.fill(0xff408080);
  ASSERT_TRUE(gpu::gcn::RetileTextureMip32(pixels.data(), image.data(),
                                            layout, 0, 0));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(image.data()), image.size());
  const u32 shader[] = {
      0xf0000f08, 0x00000400,  // image_load from s[0:7]
      0xf0200f08, 0x00020400,  // image_store through s[8:15]
      0xbf810000};
  std::copy(std::begin(shader), std::end(shader), code.begin());
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 4;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 4;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
  const u64 address = reinterpret_cast<u64>(image.data());
  for (u32 view = 0; view < 2; view++) {
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0 + view * 8;
    regs[ud] = address >> 8;
    regs[ud + 1] = ((address >> 40) & 0xff) |
                     ((view ? 56 : 130) << 20) | (3u << 30);
    regs[ud + 2] = 1 | (7 << 14);
    regs[ud + 3] = 0x91b00fac | (view << 12) | (view << 16);
    regs[ud + 5] = 3 << 4;
  }
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
  std::array<u32, 64> base{};
  std::array<u32, 16> mip{};
  ASSERT_TRUE(gpu::gcn::DetileTextureMip32(image.data(), base.data(),
                                            layout, 0, 0));
  ASSERT_TRUE(gpu::gcn::DetileTextureMip32(image.data(), mip.data(),
                                            layout, 1, 0));
  EXPECT_EQ(base, pixels);
  for (u32 pixel : mip) {
    EXPECT_NEAR(pixel & 255, 55, 1);
    EXPECT_NEAR((pixel >> 8) & 255, 55, 1);
    EXPECT_NEAR((pixel >> 16) & 255, 13, 1);
    EXPECT_EQ(pixel >> 24, 255u);
  }
}

TEST(RdnaComputeImageConversion, SrgbLoadsAndFilteringDecodeRgbBeforeInterpolation) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static std::array<std::array<u32, 16384>, 3> inputs{}, outputs{};
  alignas(256) static std::array<u32, 4096> code{};
  for (u32 run = 0; run < 3; ++run) {
    auto& input = inputs[run];
    auto& output = outputs[run];
    input[0] = run == 2 ? 0x40000000 : 0x400a8080;
    input[1] = 0xc0ffffff;
    gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(input.data()), sizeof(input));
    gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(output.data()), sizeof(output));
    code.fill(0);
    u32 pc = 0;
    if (run) {
      code[pc++] = 0x7e0002ff; // x = .25 (texel 0) or .5 (between texels)
      code[pc++] = std::bit_cast<u32>(run == 1 ? .25f : .5f);
    }
    code[pc++] = 0x7e020280;
    code[pc++] = run ? 0xf09c0f00 : 0xf0000f00; // sample_lz / load
    code[pc++] = 0x00600400; // sampler s[12:15] (sample only)
    code[pc++] = 0xe0780000;
    code[pc++] = 0x80020400;
    code[pc++] = 0xbf810000;
    gpu::rdna::NextProgramGeneration();
    gpu::ps5::Regs regs;
    const u64 program = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = program >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = program >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    const u64 src = reinterpret_cast<u64>(input.data());
    regs[ud] = src >> 8;
    regs[ud + 1] = ((src >> 40) & 255) | (130u << 20) | (1u << 30);
    regs[ud + 3] = 0x80000fac; // two texels, 1D linear
    const u64 dst = reinterpret_cast<u64>(output.data());
    regs[ud + 8] = dst;
    regs[ud + 9] = (dst >> 32) & 0xffff;
    regs[ud + 10] = 16;
    regs[ud + 11] = 0x20000;
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    const std::array<float, 4> expected = run == 2
        ? std::array<float, 4>{.5f, .5f, .5f, 128.f / 255.f}
        : std::array<float, 4>{.2158605f, .2158605f, 10.f / (255.f * 12.92f), 64.f / 255.f};
    for (u32 i = 0; i < 4; ++i)
      EXPECT_NEAR(std::bit_cast<float>(output[i]), expected[i], 1e-6f)
          << "run=" << run << " channel=" << i;
  }
}


TEST(RdnaComputeImageConversion, DecoderRg16IntegerLoadsSignExtendBothComponents) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer)) GTEST_SKIP() << "Vulkan is required";
  alignas(65536) static std::array<std::array<u32, 16384>, 2> sources{}, outputs{};
  alignas(256) static std::array<u32, 16384> code{};
  const u32 program[] = {0x7e020280, 0xf0000300, 0x00000400,
                        0xe0740000, 0x80020400, 0xbf810000};
  std::copy(std::begin(program), std::end(program), code.begin());
  for (u32 run = 0; run < 2; ++run) {
    auto& source = sources[run]; auto& output = outputs[run];
    source[0] = 0x8001ffff;
    output.fill(0xa5a5a5a5);
    gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(source.data()), sizeof(source));
    gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(output.data()), sizeof(output));
    gpu::rdna::NextProgramGeneration();
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 12 << 1;
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    const u64 src = reinterpret_cast<u64>(source.data()), dst = reinterpret_cast<u64>(output.data());
    regs[ud] = src >> 8;
    regs[ud + 1] = ((src >> 40) & 255) | ((27 + run) << 20);
    regs[ud + 3] = 0x8000002c; // 1D RG integer image
    regs[ud + 8] = dst;
    regs[ud + 9] = (dst >> 32) & 0xffff;
    regs[ud + 10] = 8;
    regs[ud + 11] = 0x20000;
    const u32 launch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, launch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    EXPECT_EQ(output[0], run ? 0xffffffffu : 0xffffu);
    EXPECT_EQ(output[1], run ? 0xffff8001u : 0x8001u);
    EXPECT_EQ(output[2], 0xa5a5a5a5);
  }
}

TEST(RdnaComputeImageConversion, DecoderRg16IntegerStoresPackComponents) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer)) GTEST_SKIP() << "Vulkan is required";
  alignas(65536) static std::array<std::array<u32, 16384>, 4> sources{}, outputs{};
  alignas(256) static std::array<u32, 16384> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(sources.data()), sizeof(sources));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(outputs.data()), sizeof(outputs));
  for (u32 run = 0; run < 4; ++run) {
    const bool signed_format = run & 1;
    const bool green_only = run & 2;
    auto& source = sources[run];
    auto& output = outputs[run];
    source[0] = signed_format ? 0xffffffffu : 65535;
    source[1] = signed_format ? 0xffff8000u : 32768;
    source[2] = 123;
    source[3] = 456;
    source[4] = 0;
    source[5] = 32767;
    output.fill(0xa5a51234);
    const u32 program[] = {
        0x7e020280, 0xf0000300, 0x00000400,
        0xf0200000 | ((green_only ? 2u : 3u) << 8), 0x00020400, 0xbf810000};
    std::copy(std::begin(program), std::end(program), code.begin());
    gpu::rdna::NextProgramGeneration();
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 3;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
    const auto descriptor = [&](u32 first, const void* memory, u32 format) {
      const u64 address = reinterpret_cast<u64>(memory);
      regs[first] = address >> 8;
      regs[first + 1] = ((address >> 40) & 255) | (format << 20) | (2u << 30);
      regs[first + 3] = 0x8000002c;  // width 3, linear 1D RG
    };
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, source.data(), signed_format ? 63 : 62);
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, output.data(), signed_format ? 28 : 27);
    const u32 launch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, launch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    for (u32 x = 0; x < 3; ++x) {
      const u32 red = green_only ? 0x1234 : source[x * 2] & 0xffff;
      const u32 green = source[x * 2 + (green_only ? 0 : 1)] & 0xffff;
      EXPECT_EQ(output[x], red | (green << 16)) << "run " << run << " texel " << x;
    }
    EXPECT_EQ(output[3], 0xa5a51234);
  }
}

TEST(RdnaComputeImageConversion, DecoderPackedIntegersRoundTripWithoutNormalization) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer)) GTEST_SKIP() << "Vulkan is required";
  struct Case { u32 format, channels, bits; bool is_signed; };
  constexpr Case cases[] = {{5, 1, 8, false}, {18, 2, 8, false},
      {60, 4, 8, false}, {61, 4, 8, true},
      {69, 4, 16, false}, {70, 4, 16, true}};
  struct Storage { std::array<u32, 16384> source, packed, result; };
  alignas(65536) static std::array<Storage, 2 * std::size(cases)> storage{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(storage.data()), sizeof(storage));
  for (u32 run = 0; run < storage.size(); ++run) {
    const auto c = cases[run % std::size(cases)];
    const bool physical = run >= std::size(cases);
    auto& s = storage[run];
    const u32 mask = (1u << c.bits) - 1;
    const u32 values[] = {mask, 1u << (c.bits - 1), 0, 127, 1, 42, mask - 1,
                         17, 63, 128, 2, (1u << (c.bits - 1)) - 1};
    std::copy(std::begin(values), std::end(values), s.source.begin());
    s.packed.fill(0xa5a5a5a5);
    s.result.fill(0xa5a5a5a5);
    for (u32 pass = 0; pass < 2; ++pass) {
      const u32 program[] = {0x7e020280, 0xf0000f00, 0x00000400,
          0xf0200000 | ((pass ? 15u : (1u << c.channels) - 1) << 8),
          0x00020400, 0xbf810000};
      // A checked GLOBAL read gives the second set the same guest-address
      // capability as the native decoder. Address zero safely reads zero.
      const u32 prefix[] = {0x7e280280, 0x7e2a0280, 0xdc308000, 0x147d0014};
      u32 offset = 0;
      if (physical) {
        std::copy(std::begin(prefix), std::end(prefix), code.begin());
        offset = std::size(prefix);
      }
      std::copy(std::begin(program), std::end(program), code.begin() + offset);
      gpu::rdna::NextProgramGeneration();
      gpu::ps5::Regs regs;
      const u64 pc = reinterpret_cast<u64>(code.data());
      regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
      regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
      regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 3;
      regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
      regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
      regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
      const auto descriptor = [&](u32 first, const void* memory, u32 format) {
        const u64 address = reinterpret_cast<u64>(memory);
        regs[first] = address >> 8;
        regs[first + 1] = ((address >> 40) & 255) | (format << 20) | (2u << 30);
        regs[first + 3] = 0x80000fac; // width 3, linear 1D RGBA
      };
      const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
      descriptor(ud, pass ? s.packed.data() : s.source.data(), pass ? c.format : 76);
      descriptor(ud + 8, pass ? s.result.data() : s.packed.data(), pass ? 76 : c.format);
      const u32 launch[] = {1, 1, 1, 1};
      gpu::ps5::DispatchCompute(renderer, regs, launch, 4);
      ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    }
    for (u32 x = 0; x < 3; ++x) {
      for (u32 channel = 0; channel < c.channels; ++channel) {
        u32 expected = values[x * 4 + channel] & mask;
        if (c.is_signed && (expected & (1u << (c.bits - 1)))) expected |= ~mask;
        EXPECT_EQ(s.result[x * 4 + channel], expected)
            << "format " << c.format << " physical " << physical
            << " texel " << x << " channel " << channel;
      }
    }
    EXPECT_EQ(s.result[12], 0xa5a5a5a5);
    const auto* packed = reinterpret_cast<const u8*>(s.packed.data());
    EXPECT_EQ(packed[3 * c.channels * c.bits / 8], 0xa5);
  }
}

TEST(RdnaComputeImageConversion, LinearIntegerViewsShareWritesWithinOneDispatch) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer)) GTEST_SKIP() << "Vulkan is required";
  alignas(65536) static std::array<std::array<u32, 16384>, 2> memory{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(memory.data()), sizeof(memory));
  for (u32 reverse = 0; reverse < 2; ++reverse) {
    memory[0].fill(0xa5a5a5a5);
    memory[0][1] = 0xdeadbeef;
    memory[1].fill(0xa5a5a5a5);
    const u64 dst = reinterpret_cast<u64>(memory[1].data());
    const u32 program[] = {
        0x7e0002ff, reverse ? 5u : 1u, 0x7e020280,
        0x7e0802ff, reverse ? 0x66u : 0x12345678u,
        0xf0200100, 0x00000400, // store through s[0:7]
        0x7e0402ff, reverse ? 1u : 5u, 0x7e060280,
        0xf0000100, 0x00020502, // read the same word through s[8:15]
        0x7e2802ff, u32(dst), 0x7e2a02ff, u32(dst >> 32),
        0xdc708000, 0x007d0514, // GLOBAL store v5 through v[20:21]
        0xbf810000};
    std::copy(std::begin(program), std::end(program), code.begin());
    gpu::rdna::NextProgramGeneration();
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
    const auto descriptor = [&](u32 first, bool byte) {
      const u64 base = reinterpret_cast<u64>(memory[0].data());
      const u32 last = byte ? 7 : 1;
      regs[first] = base >> 8;
      regs[first + 1] = ((base >> 40) & 255) | ((byte ? 5u : 20u) << 20) | ((last & 3) << 30);
      regs[first + 2] = last >> 2;
      regs[first + 3] = 0x80000004;
    };
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, reverse);
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, !reverse);
    const u32 launch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, launch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    EXPECT_EQ(memory[1][0], reverse ? 0xdead66efu : 0x56u);
    EXPECT_EQ(memory[0][1], reverse ? 0xdead66efu : 0x12345678u);
    EXPECT_EQ(memory[0][0], 0xa5a5a5a5);
    EXPECT_EQ(memory[0][2], 0xa5a5a5a5);
  }
}

TEST(RdnaComputeImageConversion, LinearIntegerArraysRespectRowsLayersAndBounds) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer)) GTEST_SKIP() << "Vulkan is required";
  alignas(65536) static std::array<std::array<u32, 16384>, 2> memory{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(memory.data()), sizeof(memory));
  const u32 program[] = {0x7e280280, 0x7e2a0280, 0xdc308000, 0x147d0014,
      0xf0000128, 0x00000400, 0xf0200128, 0x00020400, 0xbf810000};
  std::copy(std::begin(program), std::end(program), code.begin());
  auto* source = reinterpret_cast<u8*>(memory[0].data());
  for (u32 layer = 0; layer < 2; ++layer)
    for (u32 y = 0; y < 3; ++y)
      for (u32 x = 0; x < 7; ++x)
        source[(layer * 3 + y) * 256 + x] = 16 + layer * 32 + y * 8 + x;
  for (u32 base_layer = 0; base_layer < 2; ++base_layer) {
    memory[1].fill(0xa5a5a5a5);
    std::array<u32, 16384> expected;
    expected.fill(0xa5a5a5a5);
    for (u32 layer = base_layer; layer < 2; ++layer)
      for (u32 y = 0; y < 3; ++y)
        for (u32 x = 0; x < 7; ++x)
          expected[(layer * 3 + y) * 64 + x] = 16 + layer * 32 + y * 8 + x;
    gpu::rdna::NextProgramGeneration();
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 8;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 4;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 3;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
    for (u32 image = 0; image < 2; ++image) {
      const u64 base = reinterpret_cast<u64>(memory[image].data());
      const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0 + image * 8;
      regs[ud] = base >> 8;
      regs[ud + 1] = ((base >> 40) & 255) | ((image ? 20u : 5u) << 20) | (2u << 30);
      regs[ud + 2] = 1 | (2u << 14); // 7 x 3
      regs[ud + 3] = 0xd0000004; // 2D array, R
      regs[ud + 4] = 1 | (base_layer << 16);
    }
    const u32 launch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, launch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    EXPECT_EQ(memory[1], expected) << "base layer " << base_layer;
  }
}

TEST(RdnaComputeImageConversion, AstroBc6CubeArrayStagingExceedsRawBufferLimit) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  // Exact dimensions of Astro's lighting environment: 192 faces, nine mips.
  // 16.5 MiB of guest BC6 expands to slightly more than 256 MiB of RGBA32F.
  alignas(65536) static std::array<u8, 0x1080000> input{};
  alignas(65536) static std::array<u32, 16384> output{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::gcn::TextureLayout32 blocks, pixels;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(blocks, 64, 64, 64, 192, 9,
                                           0x105, false, 16));
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(pixels, 256, 256, 256, 192, 9,
                                           8, false, 16));
  ASSERT_EQ(blocks.size, input.size());
  ASSERT_EQ(pixels.size, 0x100b0000u);
  std::array<u8, 16> encoded{};
  u32 bit = 0;
  const auto put = [&](u32 value, u32 count) {
    for (u32 i = 0; i < count; ++i, ++bit)
      encoded[bit / 8] |= ((value >> i) & 1) << (bit % 8);
  };
  put(3, 5);
  for (u32 endpoint = 0; endpoint < 2; ++endpoint) {
    put(0, 10); put(512, 10); put(1023, 10);
  }
  ASSERT_TRUE(gpu::gcn::RetileTextureMip32(encoded.data(), input.data(),
                                         blocks, 8, 191));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(input.data()), input.size());
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(output.data()), sizeof(output));
  const u32 program[] = {
      0x7e020280, 0x7e0402ff, 191, // y=0, face=191
      0xf0000f28, 0x00000400,     // image_load RGBA, 2D array
      0xe0780000, 0x80020400,     // buffer_store RGBA to s[8:11]
      0xbf810000};
  std::copy(std::begin(program), std::end(program), code.begin());
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 12 << 1;
  const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
  const u64 src = reinterpret_cast<u64>(input.data());
  regs[ud] = src >> 8;
  regs[ud + 1] = ((src >> 40) & 255) | 0xcb300000;
  regs[ud + 2] = 0x003fc03f;
  regs[ud + 3] = 0xb0588fac;  // base mip 8, last mip 8
  regs[ud + 4] = 191;
  regs[ud + 5] = 0x80;
  const u64 dst = reinterpret_cast<u64>(output.data());
  regs[ud + 8] = dst;
  regs[ud + 9] = (dst >> 32) & 0xffff;
  regs[ud + 10] = 16;
  regs[ud + 11] = 0x20000;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
  const float expected[] = {0, 1.5146484375f, 65504, 1};
  for (u32 i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(std::bit_cast<float>(output[i]), expected[i]);
}

TEST(RdnaComputeImageConversion, Bc1LoadsRespectTilingMipsLayersAndSrgb) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static std::array<std::array<u8, 131072>, 4> inputs{};
  alignas(65536) static std::array<u32, 16384> output{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(inputs.data()), sizeof(inputs));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(output.data()), sizeof(output));
  constexpr u16 colors[] = {0xf800, 0x07e0, 0x001f, 0x8410};
  for (u32 variant = 0; variant < 4; variant++) {
    auto& input = inputs[variant];
    const u32 tile = variant & 1 ? 9 : 0;
    const bool srgb = variant & 2;
    gpu::gcn::TextureLayout32 blocks;
    ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(blocks, 2, 2, 2, 2, 4,
                                               0x100 | tile, false, 8));
    ASSERT_LE(blocks.size, input.size());
    for (u32 mip = 0; mip < 4; mip++)
      for (u32 layer = 0; layer < 2; layer++) {
        std::array<u64, 4> encoded{};
        for (u32 b = 0; b < 4; b++)
          encoded[b] = colors[(mip + layer + b) % 4];
        ASSERT_TRUE(gpu::gcn::RetileTextureMip32(encoded.data(), input.data(),
                                                  blocks, mip, layer));
      }
    for (u32 mip = 0; mip < 4; mip++)
      for (u32 layer = 0; layer < 2; layer++) {
        const u32 xy = mip ? 0 : 5;
        const u32 shader[] = {
            0x7e0802ff, xy, 0x7e0a02ff, xy, 0x7e0c02ff, layer,
            0xf0000f28, 0x00000804,
            0xe0780000, 0x80020800,
            0xbf810000};
        std::copy(std::begin(shader), std::end(shader), code.begin());
        gpu::rdna::NextProgramGeneration();
        gpu::ps5::Regs regs;
        const u64 pc = reinterpret_cast<u64>(code.data());
        regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
        regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
        regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
        regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
        regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
        regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 12 << 1;
        const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
        const u64 src = reinterpret_cast<u64>(input.data());
        regs[ud] = src >> 8;
        regs[ud + 1] = ((src >> 40) & 255) | ((srgb ? 170 : 169) << 20) | (3u << 30);
        regs[ud + 2] = 1 | (7 << 14);
        regs[ud + 3] = 0xd0000fac | (tile << 20) | (mip << 12) | (mip << 16);
        regs[ud + 4] = 1;
        regs[ud + 5] = 3 << 4;
        const u64 dst = reinterpret_cast<u64>(output.data());
        regs[ud + 8] = dst;
        regs[ud + 9] = (dst >> 32) & 0xffff;
        regs[ud + 10] = 16;
        regs[ud + 11] = 0x20000;
        const u32 dispatch[] = {1, 1, 1, 1};
        gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
        ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
        const u16 color = colors[(mip + layer + (mip ? 0 : 3)) % 4];
        const u32 rgb[] = {((color >> 11) * 527 + 23) >> 6,
                           (((color >> 5) & 63) * 259 + 33) >> 6,
                           ((color & 31) * 527 + 23) >> 6};
        for (u32 c = 0; c < 4; c++) {
          float expected = c < 3 ? rgb[c] / 255.f : 1.f;
          if (srgb && c < 3)
            expected = expected <= .04045f ? expected / 12.92f
                : std::pow((expected + .055f) / 1.055f, 2.4f);
          EXPECT_NEAR(std::bit_cast<float>(output[c]), expected, 0.00001f)
              << "variant " << variant << " mip " << mip << " layer " << layer;
        }
      }
  }
}

TEST(RdnaComputeImageConversion, LargeShadowArrayReadsPastRawBufferLimit) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static std::array<u8, 320u * 1024 * 1024> input{};
  alignas(65536) static std::array<u32, 16384> output{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(layout, 1024, 1024, 1024, 80, 1,
                                             0x118, false, 4));
  ASSERT_EQ(layout.size, input.size());
  std::vector<u32> pixels(1024 * 1024, std::bit_cast<u32>(0.75f));
  ASSERT_TRUE(gpu::gcn::RetileTextureMip32(pixels.data(), input.data(), layout, 0, 79));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(input.data()), input.size());
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(output.data()), sizeof(output));
  const u32 shader[] = {0x7e020280, 0x7e0402ff, 79,
                         0xf0000128, 0x00000400,
                         0xe0700000, 0x80020400, 0xbf810000};
  std::copy(std::begin(shader), std::end(shader), code.begin());
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 12 << 1;
  const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
  const u64 src = reinterpret_cast<u64>(input.data());
  regs[ud] = src >> 8;
  regs[ud + 1] = ((src >> 40) & 255) | 0xc1600000;
  regs[ud + 2] = 0x00ffc0ff;
  regs[ud + 3] = 0xd1800924;
  regs[ud + 4] = 79;
  const u64 dst = reinterpret_cast<u64>(output.data());
  regs[ud + 8] = dst;
  regs[ud + 9] = (dst >> 32) & 0xffff;
  regs[ud + 10] = 4;
  regs[ud + 11] = 0x20000;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
  EXPECT_FLOAT_EQ(std::bit_cast<float>(output[0]), 0.75f);
}

TEST(RdnaComputeImageConversion, UniformDescriptorTablePreservesDynamicCubeSelection) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static std::array<u32, 16384> texture{};
  alignas(65536) static std::array<std::array<u32, 16384>, 3> tables{}, outputs{};
  alignas(256) static std::array<u32, 4096> code{};
  for (u32 face = 0; face < 6; face++)
    texture[face * 64] = std::bit_cast<u32>((face + 1) * .25f);
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(texture.data()), sizeof(texture));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(tables.data()), sizeof(tables));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(outputs.data()), sizeof(outputs));
  const u64 image = reinterpret_cast<u64>(texture.data());
  const std::array<u32, 8> descriptor = {
      u32(image >> 8), u32((image >> 40) & 255) | (22 << 20), 0,
      0xb0000fac, 5, 0, 0, 0};
  for (u32 variant = 0; variant < 3; variant++) {
    auto& table = tables[variant];
    auto& output = outputs[variant];
    std::copy(descriptor.begin(), descriptor.end(), table.begin());
    std::copy(descriptor.begin(), descriptor.end(), table.begin() + 8);
    if (variant == 1)
      table[8]++;  // Different entries cannot be folded to a single image.
    output[0] = output[1] = output[2] = 0xdeadbeef;
    const u32 shader[] = {
        0x7e0a0210,             // v5 = workgroup ID in s16
        0x7ed40505,             // s106 = readfirstlane(v5)
        variant == 2 ? 0x8f16846a : 0x8f16856a, // s22 = index << 4 or 5
        0xf42c0004, 0x2c000000, // s[0:7] = table at s[8:11] + s22
        0x7e1002ff, std::bit_cast<u32>(1.5f),
        0x7e1202ff, std::bit_cast<u32>(1.5f),
        0x7e140d05, 0x7e160280, // face = float(v5), LOD = 0
        0xf0900118, 0x00e00408, // sample the selected cube, sampler s[28:31]
        0xe0702000, 0x80030405, // output indexed by workgroup
        0xbf810000};
    std::copy(std::begin(shader), std::end(shader), code.begin());
    gpu::rdna::NextProgramGeneration();
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = (16 << 1) | (1 << 7);
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    const u64 src = reinterpret_cast<u64>(table.data());
    regs[ud + 8] = src;
    regs[ud + 9] = ((src >> 32) & 65535) | (32 << 16);
    regs[ud + 10] = 2;
    regs[ud + 11] = 0x5204;
    const u64 dst = reinterpret_cast<u64>(output.data());
    regs[ud + 12] = dst;
    regs[ud + 13] = ((dst >> 32) & 65535) | (4 << 16);
    regs[ud + 14] = 3;
    regs[ud + 15] = 0x14004;
    const u32 dispatch[] = {3, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    if (variant == 0) {
      EXPECT_FLOAT_EQ(std::bit_cast<float>(output[0]), .25f);
      EXPECT_FLOAT_EQ(std::bit_cast<float>(output[1]), .5f);
      EXPECT_EQ(output[2], 0u);
    } else {
      EXPECT_EQ(output[0], 0xdeadbeefu);
      EXPECT_EQ(output[1], 0xdeadbeefu);
      EXPECT_EQ(output[2], 0xdeadbeefu);
    }
  }
}
TEST(RdnaComputeImageConversion, PlainStoresValidateOnlyTheAccessedMip) {
  auto& renderer = gpu::rhi::DefaultRenderer();
  if (!gpu::rhi::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static std::array<std::array<u32, 2 * 1024 * 1024>, 5> images{};
  alignas(256) static std::array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(images.data()), sizeof(images));
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(layout, 512, 512, 512, 1, 9,
                                             0x109, false, 16));
  ASSERT_LE(layout.size, sizeof(images[0]));
  constexpr u32 levels[] = {0, 3, 8, 9, 0};
  constexpr u32 values[] = {0x01234567, 0x89abcdef, 0xfedcba98, 0x76543210};
  for (u32 variant = 0; variant < 5; variant++) {
    auto& image = images[variant];
    image.fill(0xdeadbeef);
    const u32 mip = levels[variant];
    const bool explicit_mip = variant == 4;
    const u32 shader[] = {
        0x7e020280, 0x7e040280,
        0x7e0802ff, values[0], 0x7e0a02ff, values[1],
        0x7e0c02ff, values[2], 0x7e0e02ff, values[3],
        explicit_mip ? 0xf0240f08u : 0xf0200f08u, 0x00000400,
        0xbf810000};
    std::copy(std::begin(shader), std::end(shader), code.begin());
    gpu::rdna::NextProgramGeneration();
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 8 << 1;
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    const u64 dst = reinterpret_cast<u64>(image.data());
    regs[ud] = dst >> 8;
    regs[ud + 1] = ((dst >> 40) & 255) | 0xc4b00000;
    regs[ud + 2] = 0x007fc07f;
    regs[ud + 3] = 0x90990fac | (mip << 12);
    regs[ud + 5] = 0x00700080;
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(gpu::rhi::FlushCsWrites(renderer));
    for (u32 level = 0; level < layout.mip_levels; level++) {
      std::vector<u32> pixels(layout.mips[level].width * layout.mips[level].height * 4);
      ASSERT_TRUE(gpu::gcn::DetileTextureMip32(image.data(), pixels.data(), layout, level, 0));
      for (u32 c = 0; c < 4; c++)
        EXPECT_EQ(pixels[c], level == mip && !explicit_mip ? values[c] : 0xdeadbeefu)
            << "variant " << variant << " mip " << level;
      EXPECT_EQ(pixels.back(), pixels.size() == 4 && level == mip && !explicit_mip
                                   ? values[3] : 0xdeadbeefu);
    }
  }
}
}  // namespace

// Performance counters normally owned by the command-processor composition.
namespace gpu::rhi {
u64 g_ns_dcb = 0, g_ns_dcb_lock = 0;
u32 g_submit_queue = 0, g_dcb_n = 0;
}
extern "C" bool prosperity_ps5_is_display_buffer(u64) { return false; }
