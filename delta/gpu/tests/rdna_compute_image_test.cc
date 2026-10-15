#include <cmath>
#include <cstring>

#include <gtest/gtest.h>

#include "base/algorithm.h"
#include "base/containers/array.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/bit_cast.h"
#include "gpu/gcn/gcn_detile.h"
#include "gpu/ps5/compute_dispatch.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/render/compute/compute.h"
#include "gpu/render/frame.h"
#include "gpu/render/image/tiling.h"

namespace {
bool FlushCsWritesLanded(gpu::render::Renderer& renderer) {
  return gpu::render::FlushCsWrites(renderer) &&
         gpu::render::SettleGuestWrites(0, ~0ull >> 1, "test");
}

TEST(Gfx10GpuTiling, MatchesCpuAcrossMipsLayersAndPadding) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  for (u32 mode :
       {0u, 1u, 2u, 5u, 6u, 9u, 10u, 17u, 18u, 21u, 22u, 24u, 25u, 26u, 27u}) {
    for (u32 elem : {1u, 2u, 4u, 8u, 16u}) {
      SCOPED_TRACE(testing::Message() << "mode=" << mode << " elem=" << elem);
      gpu::gcn::TextureLayout32 tiled, linear;
      const u32 stage_elem = base::Max(4u, elem);
      const u32 layers = mode == 27 && elem == 8 ? 72 : 3;
      ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(tiled, 263, 137, 271, layers,
                                                 6, 0x100 + mode, false, elem));
      ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(linear, 263, 137, 271, layers,
                                                 6, 8, false, stage_elem));
      base::Vector<u8> source(tiled.size), expected(linear.size, 0);
      for (size_t i = 0; i < source.size(); i++)
        source[i] = static_cast<u8>((i * 137) ^ (i >> 7) ^ (i >> 15));
      for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
        const auto& l = linear.mips[mip];
        for (u32 layer = 0; layer < tiled.layers; layer++) {
          const u64 offset =
              l.offset + u64(layer) * l.pitch * l.stored_height * stage_elem;
          base::Vector<u8> tight(u64(l.width) * l.height * elem);
          ASSERT_TRUE(gpu::gcn::DetileTextureMip32Pitched(
              source.data(), tight.data(), l.width * elem, tiled, mip, layer));
          for (u32 y = 0; y < l.height; y++)
            for (u32 x = 0; x < l.width; x++)
              std::memcpy(expected.data() + offset +
                              (u64(y) * l.pitch + x) * stage_elem,
                          tight.data() + (u64(y) * l.width + x) * elem, elem);
        }
      }
      base::Vector<u8> actual(linear.size, 0x5a);
      ASSERT_TRUE(gpu::render::ConvertGfx10Image(tiled, linear, source.data(),
                                                 actual.data(), true));
      ASSERT_EQ(actual, expected);
      for (size_t i = 0; i < actual.size(); i++)
        actual[i] ^= 0x97;
      base::Vector<u8> retiled(tiled.size, 0xa5), reference = retiled;
      for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
        const auto& l = linear.mips[mip];
        for (u32 layer = 0; layer < tiled.layers; layer++) {
          const u64 offset =
              l.offset + u64(layer) * l.pitch * l.stored_height * stage_elem;
          base::Vector<u8> tight(u64(l.width) * l.height * elem);
          for (u32 y = 0; y < l.height; y++)
            for (u32 x = 0; x < l.width; x++)
              std::memcpy(
                  tight.data() + (u64(y) * l.width + x) * elem,
                  actual.data() + offset + (u64(y) * l.pitch + x) * stage_elem,
                  elem);
          ASSERT_TRUE(gpu::gcn::RetileTextureMip32Pitched(
              tight.data(), l.width * elem, reference.data(), tiled, mip,
              layer));
        }
      }
      ASSERT_TRUE(gpu::render::ConvertGfx10Image(tiled, linear, actual.data(),
                                                 retiled.data(), false));
      ASSERT_EQ(retiled, reference);
    }
  }
}

TEST(Gfx10GpuTiling, PacksR11G11B10LikeTheHost) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  for (u32 mode : {27u, 0u}) {
    SCOPED_TRACE(testing::Message() << "mode=" << mode);
    gpu::gcn::TextureLayout32 tiled, linear;
    ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(tiled, 121, 67, 121, 2, 3,
                                               0x100 + mode, false, 4));
    ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(linear, 121, 67, 121, 2, 3, 8,
                                               false, 16));
    const float specials[] = {0.f,         -0.f,     1.f,      -1.f,
                              65000.f,     1e9f,     6.1e-5f,  3e-6f,
                              1e-9f,       0.5f,     0.49999f, 1.0078125f,
                              1.00390625f, 123.456f, 7e-45f,   1e-38f};
    base::Vector<u8> source(linear.size);
    for (size_t i = 0; i < source.size() / 4; i++) {
      u32 bits = static_cast<u32>(i * 2654435761u) ^ static_cast<u32>(i >> 3);
      if (i % 7 == 0) {
        float v = specials[(i / 7) % 16];
        std::memcpy(&bits, &v, 4);
      } else if (i % 13 == 0) {
        bits = i % 2 ? 0x7f800000u : 0x7fc00001u;
      }
      std::memcpy(source.data() + i * 4, &bits, 4);
    }
    base::Vector<u8> reference(tiled.size, 0xa5), actual = reference;
    for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
      const auto& l = linear.mips[mip];
      for (u32 layer = 0; layer < tiled.layers; layer++) {
        const u64 offset =
            l.offset + u64(layer) * l.pitch * l.stored_height * 16;
        base::Vector<u8> tight(u64(l.width) * l.height * 4);
        for (u32 y = 0; y < l.height; y++)
          for (u32 x = 0; x < l.width; x++) {
            const u32 packed = gpu::render::PackR11G11B10Texel(
                source.data() + offset + (u64(y) * l.pitch + x) * 16);
            std::memcpy(tight.data() + (u64(y) * l.width + x) * 4, &packed, 4);
          }
        ASSERT_TRUE(gpu::gcn::RetileTextureMip32Pitched(
            tight.data(), l.width * 4, reference.data(), tiled, mip, layer));
      }
    }
    ASSERT_TRUE(gpu::render::ConvertGfx10Image(tiled, linear, source.data(),
                                               actual.data(), false, true));
    ASSERT_EQ(actual, reference);
    base::Vector<u8> unpacked(linear.size, 0), expected(linear.size, 0);
    ASSERT_TRUE(gpu::render::ConvertGfx10Image(tiled, linear, actual.data(),
                                               unpacked.data(), true, true));
    for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
      const auto& l = linear.mips[mip];
      for (u32 layer = 0; layer < tiled.layers; layer++) {
        const u64 offset =
            l.offset + u64(layer) * l.pitch * l.stored_height * 16;
        base::Vector<u8> tight(u64(l.width) * l.height * 4);
        ASSERT_TRUE(gpu::gcn::DetileTextureMip32Pitched(
            actual.data(), tight.data(), l.width * 4, tiled, mip, layer));
        for (u32 y = 0; y < l.height; y++)
          for (u32 x = 0; x < l.width; x++) {
            u32 packed;
            std::memcpy(&packed, tight.data() + (u64(y) * l.width + x) * 4, 4);
            gpu::render::UnpackR11G11B10Texel(
                packed, expected.data() + offset + (u64(y) * l.pitch + x) * 16);
          }
      }
    }
    for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
      const auto& l = linear.mips[mip];
      for (u32 layer = 0; layer < tiled.layers; layer++)
        for (u32 y = 0; y < l.height; y++) {
          const u64 row = l.offset +
                          u64(layer) * l.pitch * l.stored_height * 16 +
                          u64(y) * l.pitch * 16;
          ASSERT_EQ(std::memcmp(unpacked.data() + row, expected.data() + row,
                                u64(l.width) * 16),
                    0)
              << "mip " << mip << " layer " << layer << " row " << y;
        }
    }
  }
}

class RdnaComputeImage : public testing::TestWithParam<u32> {};

TEST(RdnaComputeImageConversion, ReadsDescriptorChannelsAndConstants) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<base::Array<u32, 16384>, 8> inputs{},
      outputs{};
  alignas(256) static base::Array<u32, 4096> code{};
  for (u32 trial = 0; trial < 8; trial++) {
    if (trial == 4) {
      gpu::render::RecordComputeInFrame();
      gpu::render::BeginFrame(renderer);
      ASSERT_TRUE(gpu::render::CsBindsImages());
    }
    const bool integer = trial & 2;
    const bool sample = trial & 1;
    SCOPED_TRACE(testing::Message() << "trial=" << trial);
    inputs[trial][0] = 0x80402010;
    inputs[trial][1] = 0xa0603018;
    outputs[trial].fill(0xdeadbeef);
    gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(inputs[trial].data()),
                          sizeof(inputs[trial]));
    gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(outputs[trial].data()),
                          sizeof(outputs[trial]));
    const u32 program[] = {0x7e0002ff,
                           integer && sample ? 0x3f19999au : 0u,
                           0x7e020280,
                           sample ? 0xf09c0f08u : 0xf0000f08u,
                           sample ? 0x00600400u : 0x00000400u,
                           0xe0780000,
                           0x80020400,
                           0xbf810000};
    base::Copy(program, program + base::ArraySize(program), code.begin());
    gpu::rdna::NextProgramGeneration();
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    const u64 src = reinterpret_cast<u64>(inputs[trial].data());
    regs[ud] = src >> 8;
    regs[ud + 1] =
        ((src >> 40) & 255) | ((integer ? 60u : 56u) << 20) | (1u << 30);
    regs[ud + 3] = 0x90000226;
    const u64 dst = reinterpret_cast<u64>(outputs[trial].data());
    regs[ud + 8] = dst;
    regs[ud + 9] = (dst >> 32) & 0xffff;
    regs[ud + 10] = 16;
    regs[ud + 11] = 0x20000;
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    const base::Array<u32, 4> values = {integer && sample ? 96u : 64u,
                                        integer && sample ? 24u : 16u, 0, 1};
    for (u32 i = 0; i < values.size(); i++) {
      if (integer)
        EXPECT_EQ(outputs[trial][i], values[i]);
      else
        EXPECT_FLOAT_EQ(base::BitCast<float>(outputs[trial][i]),
                        i == 3 ? 1.f : values[i] / 255.f);
    }
  }
  gpu::render::EndFrame(renderer, 0);
}

TEST_P(RdnaComputeImage, Raw32LoadStorePreservesUnwrittenChannels) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";

  alignas(65536) static base::Array<base::Array<u32, 16384>, 6> sources{},
      dests{};
  const u32 format = GetParam();
  const u32 allocation = format < 65 ? format - 62 : format - 72;
  auto& source = sources[allocation];
  auto& dest = dests[allocation];
  alignas(256) static base::Array<u32, 4096> code{};
  constexpr base::Array<u32, 12> kValues = {
      0x3e800000, 0xc0000000, 0x40800000, 0x3f800000, 0x3f000000, 0xbf800000,
      0x41000000, 0x3f800000, 0x3f400000, 0xc0400000, 0x41800000, 0x3f800000};
  base::Copy(kValues.begin(), kValues.end(), source.begin());
  constexpr u32 kUntouched = 0x3f123456;
  const u32 channels = format < 65 ? 2 : 4;
  dest.fill(kUntouched);
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(source.data()), sizeof(source));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));

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
    regs[first + 2] = 0;
    regs[first + 3] = 0x80000fac;
  };
  descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, source.data());
  descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, dest.data());
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  for (u32 x = 0; x < 3; x++) {
    EXPECT_EQ(dest[x * channels], kValues[x * channels]) << "texel " << x;
    EXPECT_EQ(dest[x * channels + 1], kUntouched) << "texel " << x;
    if (channels == 4) {
      EXPECT_EQ(dest[x * 4 + 2], kValues[x * 4 + 1]) << "texel " << x;
      EXPECT_EQ(dest[x * 4 + 3], kUntouched) << "texel " << x;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(RgAndRgbaUintSintFloat,
                         RdnaComputeImage,
                         testing::Values(62u, 63u, 64u, 75u, 76u, 77u));

TEST(RdnaComputeImageConversion, R8UnormStoreWritesBackByteImage) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<u32, 16384> source{};
  alignas(65536) static base::Array<u8, 65536> dest{};
  alignas(256) static base::Array<u32, 4096> code{};
  source[0] = base::BitCast<u32>(0.f);
  source[1] = base::BitCast<u32>(1.f);
  source[2] = base::BitCast<u32>(64.f / 255.f);
  dest.fill(0xa5);
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(source.data()), sizeof(source));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  code[0] = 0x7e020280;
  code[1] = 0xf0000100;
  code[2] = 0x00000400;
  code[3] = 0xf0200100;
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
  const auto descriptor = [&](u32 first, const void* memory, u32 format) {
    const u64 address = reinterpret_cast<u64>(memory);
    regs[first] = address >> 8;
    regs[first + 1] = ((address >> 40) & 0xff) | (format << 20) | (2u << 30);
    regs[first + 3] = 0x80000fac;
  };
  descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, source.data(), 22);
  descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, dest.data(), 1);
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  EXPECT_EQ(dest[0], 0);
  EXPECT_EQ(dest[1], 255);
  EXPECT_EQ(dest[2], 64);
  EXPECT_EQ(dest[3], 0xa5);
}

TEST(RdnaComputeImageConversion, WritesMappedImagesOutsideGpuPools) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<u32, 16384> source{};
  alignas(65536) static base::Array<u8, 65536> dest{};
  alignas(65536) static const base::Array<u8, 65536> readonly{0xa5};
  alignas(256) static base::Array<u32, 4096> code{};
  source[0] = base::BitCast<u32>(1.f);
  dest.fill(0xa5);
  ASSERT_FALSE(gpu::ps5::IsGpuAddress(reinterpret_cast<u64>(dest.data())));
  ASSERT_FALSE(gpu::ps5::IsGpuAddress(reinterpret_cast<u64>(readonly.data())));
  code[0] = 0x7e020280;
  code[1] = 0xf0000100;
  code[2] = 0x00000400;
  code[3] = 0xf0200100;
  code[4] = 0x00020400;
  code[5] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
  const auto descriptor = [&](u32 first, const void* memory, u32 format) {
    const u64 address = reinterpret_cast<u64>(memory);
    regs[first] = address >> 8;
    regs[first + 1] = ((address >> 40) & 0xff) | (format << 20);
    regs[first + 3] = 0x80000fac;
  };
  descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, source.data(), 22);
  const u32 dispatch[] = {1, 1, 1, 1};
  for (const u8* output :
       {static_cast<const u8*>(dest.data()), readonly.data()}) {
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, output, 1);
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
  }
  EXPECT_EQ(dest[0], 255);
  EXPECT_EQ(dest[1], 0xa5);
  EXPECT_EQ(readonly[0], 0xa5);
}

class RdnaR16Unorm : public testing::TestWithParam<u32> {};

TEST(RdnaComputeImageConversion, LargeNarrowStoresPreserveUntouchedTexels) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<base::Array<u8, 1024 * 1024>, 4> memory{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(memory.data()), sizeof(memory));
  alignas(256) static base::Array<u32, 4096> code{};
  code[0] = 0x7e020280;
  code[1] = 0xf0000108;
  code[2] = 0x00000400;
  code[3] = 0xf0200108;
  code[4] = 0x00020400;
  code[5] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  constexpr u32 kWidth = 257, kHeight = 256;
  for (u32 elem : {1u, 2u}) {
    auto& source = memory[(elem - 1) * 2];
    auto& dest = memory[(elem - 1) * 2 + 1];
    const float values[] = {-1.f, 2.f, .5f};
    std::memcpy(source.data(), values, sizeof(values));
    dest.fill(0xa5);
    gpu::gcn::TextureLayout32 layout;
    ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(layout, kWidth, kHeight, kWidth,
                                               1, 1, 0x118, false, elem));
    ASSERT_LE(layout.size, dest.size());
    base::Vector<u8> tight(u64(kWidth) * kHeight * elem, 0xa5);
    const u32 expected_values[] = {0, elem == 1 ? 255u : 65535u,
                                   elem == 1 ? 128u : 32768u};
    for (u32 x = 0; x < 3; x++)
      std::memcpy(tight.data() + x * elem, &expected_values[x], elem);
    base::Vector<u8> expected(dest.begin(), dest.end());
    ASSERT_TRUE(gpu::gcn::RetileTextureMip32(tight.data(), expected.data(),
                                             layout, 0, 0));
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
                        (((kWidth - 1) & 3) << 30);
      regs[first + 2] = ((kWidth - 1) >> 2) | ((kHeight - 1) << 14);
      regs[first + 3] = 0x90000fac | (swizzle << 20);
    };
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, source.data(), 22, 0);
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, dest.data(),
               elem == 1 ? 1 : 7, 24);
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    EXPECT_EQ(base::Vector<u8>(dest.begin(), dest.end()), expected)
        << "elem=" << elem;
  }
}

TEST_P(RdnaR16Unorm, MipArrayLoadAndClampedStorePreserveAdjacentTexels) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<base::Array<u8, 131072>, 8> memory{};
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
  const base::Array<u16, 8> values = {0,     1,     32768, 65535,
                                      16384, 49151, 42,    60000};
  const base::Array<float, 8> input = {-1.f, 2.f, .5f,  .25f,
                                       0.f,  1.f, .75f, .125f};
  const base::Array<u16, 8> sentinel = {0xa5a5, 0xa5a5, 0xa5a5, 0xa5a5,
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
  alignas(256) static base::Array<u32, 4096> code{};
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
      regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] =
          mip ? base::Max(4u >> mip, 1u) : 3;
      regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = base::Max(2u >> mip, 1u);
      regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 2;
      regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
      const auto descriptor = [&](u32 first, const void* data, u32 fmt,
                                  u32 tile) {
        const u64 address = reinterpret_cast<u64>(data);
        regs[first] = address >> 8;
        regs[first + 1] = ((address >> 40) & 0xff) | (fmt << 20) | (3u << 30);
        regs[first + 2] = 1 << 14;
        regs[first + 3] = 0xd0020fac | (mip << 12) | (tile << 20);
        regs[first + 4] = 1;
        regs[first + 5] = 2 << 4;
      };
      descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, src.data(), sfmt, stile);
      descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, dst.data(), dfmt, dtile);
      const u32 dispatch[] = {1, 1, 1, 1};
      gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
      ASSERT_TRUE(FlushCsWritesLanded(renderer));
    };
    transfer(packed, 7, swizzle, loaded, 22, 0);
    transfer(floats, 22, 0, stored, 7, swizzle);
    for (u32 layer = 0; layer < 2; ++layer) {
      base::Array<float, 8> result{};
      base::Array<u16, 8> output{};
      ASSERT_TRUE(gpu::gcn::DetileTextureMip32(loaded.data(), result.data(),
                                               layout32, mip, layer));
      ASSERT_TRUE(gpu::gcn::DetileTextureMip32(stored.data(), output.data(),
                                               layout16, mip, layer));
      const base::Array<u16, 8> expected = {0, 65535, 32768, 0xa5a5,
                                            0, 65535, 49151, 0xa5a5};
      for (u32 i = 0; i < base::Max(4u >> mip, 1u) * base::Max(2u >> mip, 1u);
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

class RdnaBc6Image : public testing::TestWithParam<u32> {};

TEST_P(RdnaBc6Image, HdrLoadAndCubeSamplePreserveBlocksMipsAndLayers) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  const u32 param = GetParam();
  const bool is_signed = param & 1;
  const bool cube = param & 4;
  const u32 layer_count = cube ? 6 : 2;
  const u32 tile = param & 2 ? 5 : 0;
  alignas(65536) static base::Array<base::Array<u8, 131072>, 8> inputs{},
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
      base::Array<u8, 64> tight{};
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
  alignas(256) static base::Array<u32, 4096> code{};
  code[0] = 0xf0000f28;
  code[1] = 0x00000400;
  code[2] = 0xf0200f28;
  code[3] = 0x00020400;
  code[4] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  for (u32 mip = 0; mip < 4; ++mip) {
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    const u32 dim = 8u >> mip;
    if (cube) {
      const u32 scale = base::BitCast<u32>(1.f / dim);
      const u32 bias = base::BitCast<u32>(1.f + .5f / dim);
      const u32 sample_code[] = {0x7e100d00,
                                 0x7e120d01,
                                 0x7e140d02,
                                 0x101010ff,
                                 scale,
                                 0x061010ff,
                                 base::BitCast<u32>(1.f + 1.f / dim),
                                 0x101212ff,
                                 scale,
                                 0x061212ff,
                                 bias,
                                 0x7e160280,
                                 0xf0900f18,
                                 0x00800408,
                                 0xf0200f28,
                                 0x00020400,
                                 0xbf810000};
      base::Copy(sample_code, sample_code + base::ArraySize(sample_code),
                 code.begin());
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
      regs[first + 2] = 1 | (7 << 14);
      regs[first + 3] = 0xd0030fac | (mip << 12) | (sw << 20);
      regs[first + 4] = layer_count - 1;
      if (cube && first == gpu::ps5::mmCOMPUTE_USER_DATA_0) {
        regs[first + 3] = (regs[first + 3] & 0x0fffffff) | 0xb0000000;
        regs[first + 4] = 0;
      }
      regs[first + 5] = 3 << 4;
    };
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, input.data(),
               is_signed ? 180 : 179, tile);
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, output.data(), 77, 0);
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    for (u32 layer = 0; layer < layer_count; ++layer) {
      base::Array<float, 8 * 8 * 4> result{};
      ASSERT_TRUE(gpu::gcn::DetileTextureMip32(output.data(), result.data(),
                                               pixels, mip, layer));
      for (u32 y = 0; y < dim; ++y) {
        for (u32 x = 0; x < dim; ++x) {
          const u32 b = (y / 4) * blocks.mips[mip].width + x / 4;
          for (u32 c = 0; c < 4; ++c) {
            float value = c == 3 ? 1.f : expected[(mip + layer + b + c) % 3];
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
  EXPECT_EQ(input, original);
}

INSTANTIATE_TEST_SUITE_P(SignedUnsignedLinearTiled,
                         RdnaBc6Image,
                         testing::Values(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u));

TEST(RdnaComputeImageConversion, Rgba16UnormLoadAndMaskedClampedStore) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<u32, 16384> packed{}, floats{}, result{},
      output{};
  packed[0] = 0xffff0000;
  packed[2] = 0x80004000;
  packed[4] = 0x0001ffff;
  result.fill(0x12345678);
  output.fill(0x76543210);
  const base::Array<float, 12> values = {-1.f, 2.f, 0.f, 0.f, 0.5f, 0.25f,
                                         0.f,  0.f, 1.f, 0.f, 0.f,  0.f};
  for (u32 i = 0; i < values.size(); i++)
    floats[i] = base::BitCast<u32>(values[i]);
  alignas(256) static base::Array<u32, 4096> code{};
  code[0] = 0x7e020280;
  code[1] = 0xf0000f00;
  code[2] = 0x00000400;
  code[3] = 0xf0200500;
  code[4] = 0x00020400;
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
    return FlushCsWritesLanded(renderer);
  };
  ASSERT_TRUE(transfer(packed.data(), 65, result.data(), 77));
  for (u32 x = 0; x < 3; x++) {
    EXPECT_FLOAT_EQ(base::BitCast<float>(result[x * 4]),
                    float(packed[x * 2] & 0xffff) / 65535.f);
    EXPECT_FLOAT_EQ(base::BitCast<float>(result[x * 4 + 2]),
                    float(packed[x * 2] >> 16) / 65535.f);
    EXPECT_EQ(result[x * 4 + 1], 0x12345678u);
    EXPECT_EQ(result[x * 4 + 3], 0x12345678u);
  }
  ASSERT_TRUE(transfer(floats.data(), 77, output.data(), 65));
  const base::Array<u32, 6> expected = {0x76540000, 0x7654ffff, 0x76548000,
                                        0x76544000, 0x7654ffff, 0x76540000};
  for (u32 i = 0; i < expected.size(); i++)
    EXPECT_EQ(output[i], expected[i]) << "dword " << i;
}

TEST(RdnaComputeAlu, Scalar64CompareUsesHighDwordAndInlineSignExtension) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<u32, 16384> output{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(output.data()), sizeof(output));
  u32 pc_dw = 0;
  const auto compare = [&](u32 op, u32 a, u32 b, u32 dst) {
    code[pc_dw++] = 0xbf000000 | (op << 16) | (b << 8) | a;
    code[pc_dw++] = 0x85108081;
    code[pc_dw++] = 0x7e000210 | (dst << 17);
  };
  compare(0x12, 8, 10, 4);
  compare(0x13, 8, 10, 5);
  compare(0x12, 12, 193, 6);
  compare(0x13, 12, 193, 7);
  code[pc_dw++] = 0x7c041ef9;
  code[pc_dw++] = 0x86860080;
  code[pc_dw++] = 0xf0200f00;
  code[pc_dw++] = 0x00000400;
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
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  const base::Array<u32, 4> expected = {0, 1, 1, 0};
  for (u32 i = 0; i < 4; ++i)
    EXPECT_EQ(output[i], expected[i]);
}

TEST(RdnaComputeAlu, XadUsesXorBeforeWrappingAdd) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";

  alignas(65536) static base::Array<u32, 16384> dest{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
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
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  EXPECT_EQ(dest[0], 1u);
  EXPECT_EQ(dest[1], 0u);
  EXPECT_EQ(dest[2], 0xffffffffu);
}

TEST(RdnaComputeAlu, HalfPrecisionOpsSelectHalvesAndPackResults) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<u32, 16384> dest{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  const u32 shader[] = {
      0x7e0202ff, 0x40003c00, 0x7e0402ff, 0x42004400, 0x7e0602ff, 0x3800bc00,
      0x7e0c02ff, 0xabcd0000, 0x64080501, 0xd7513805, 0x040e0501, 0x740c04f9,
      0x05051401, 0xd7544008, 0x040e0501, 0x7e0e0280, 0xf0200100, 0x00000407,
      0x7e0e0281, 0xf0200100, 0x00000507, 0x7e0e0282, 0xf0200100, 0x00000607,
      0x7e0e0283, 0xf0200100, 0x00000807, 0xbf810000};
  base::Copy(shader, shader + base::ArraySize(shader), code.begin());
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 8 << 1;
  const u64 address = reinterpret_cast<u64>(dest.data());
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0] = address >> 8;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 1] =
      ((address >> 40) & 0xff) | (20 << 20) | (3u << 30);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 3] = 0x80000fac;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  EXPECT_EQ(dest[0], 0x00004500u);
  EXPECT_EQ(dest[1], 0x00003800u);
  EXPECT_EQ(dest[2], 0xabcd4000u);
  EXPECT_EQ(dest[3], 0x44000000u);
}

TEST(RdnaComputeAlu, WritebackInvalidatesOverlappingCachedBuffers) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static base::Array<u32, 16384> parent{}, value{}, output{};
  alignas(256) static base::Array<u32, 4096> copy_code{}, update_code{};
  const u32 copy[] = {0xe0002000, 0x80000400, 0xe0102000, 0x80010400,
                      0xbf810000};
  const u32 update[] = {0xe0002000, 0x80000400, 0x7e000288,
                        0xe0102000, 0x80010400, 0xbf810000};
  base::Copy(copy, copy + base::ArraySize(copy), copy_code.begin());
  base::Copy(update, update + base::ArraySize(update), update_code.begin());
  parent[8] = 0x11223344;
  value[0] = 0x55667788;
  output[0] = 0;
  for (const auto* memory : {parent.data(), value.data(), output.data()})
    gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(memory), sizeof(parent));
  gpu::rdna::NextProgramGeneration();
  gpu::render::RecordComputeInFrame();
  gpu::render::BeginFrame(renderer);
  const auto dispatch = [&](const u32* code, const u32* source, u32* dest,
                            u32 records) {
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code);
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 8 << 1;
    const auto descriptor = [&](u32 first, const u32* memory, u32 count) {
      const u64 address = reinterpret_cast<u64>(memory);
      regs[first] = address;
      regs[first + 1] = ((address >> 32) & 0xffff) | (4 << 16);
      regs[first + 2] = count;
      regs[first + 3] = (20 << 12) | 4;
    };
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    descriptor(ud, source, 1);
    descriptor(ud + 4, dest, records);
    const u32 groups[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, groups, 4);
    EXPECT_TRUE(FlushCsWritesLanded(renderer));
  };
  dispatch(copy_code.data(), parent.data() + 8, output.data(), 1);
  EXPECT_EQ(output[0], 0x11223344u);
  dispatch(update_code.data(), value.data(), parent.data(), 16);
  EXPECT_EQ(parent[8], 0x55667788u);
  dispatch(copy_code.data(), parent.data() + 8, output.data(), 1);
  EXPECT_EQ(output[0], 0x55667788u);
  gpu::render::EndFrame(renderer, 0);
}

TEST(RdnaComputeAlu, FormattedBuffersDecodeRdnaIntegerChannels) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static base::Array<base::Array<u32, 16384>, 3> sources{},
      packed{}, raw{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(sources.data()), sizeof(sources));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(packed.data()), sizeof(packed));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(raw.data()), sizeof(raw));
  const u32 shader[] = {0xe0042000, 0x80000400, 0xe0142000, 0x80010400,
                        0xe0742000, 0x80020400, 0xbf810000};
  base::Copy(shader, shader + base::ArraySize(shader), code.begin());
  for (u32 run = 0; run < 3; run++) {
    auto& source = sources[run];
    source[0] = 0xffff8001;
    source[1] = 0x7fff1234;
    source[2] = 0x80007ffe;
    packed[run].fill(0xa5a5a5a5);
    gpu::rdna::NextProgramGeneration();
    gpu::ps5::Regs regs;
    const u64 pc = reinterpret_cast<u64>(code.data());
    regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
    regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 3;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
    regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
    regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 12 << 1;
    const u32 format = run == 0 ? 12 : run == 1 ? 28 : 21;
    const u32 stride = run ? 4 : 2;
    const auto descriptor = [&](u32 first, const void* memory, u32 bytes,
                                u32 fmt) {
      const u64 address = reinterpret_cast<u64>(memory);
      regs[first] = address;
      regs[first + 1] = ((address >> 32) & 0xffff) | (bytes << 16);
      regs[first + 2] = 3;
      regs[first + 3] = (fmt << 12) | 0x2c;
    };
    const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
    descriptor(ud, source.data(), stride, format);
    descriptor(ud + 4, packed[run].data(), stride, format);
    descriptor(ud + 8, raw[run].data(), 8, 0);
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    for (u32 lane = 0; lane < 3; lane++) {
      for (u32 component = 0; component < 2; component++) {
        const u32 index = run ? lane * 2 + component : lane;
        const u32 bits = (source[index / 2] >> ((index & 1) * 16)) & 0xffff;
        const u32 expected =
            run == 2 ? (component ? 0 : source[lane])
            : !run && component
                ? 0
                : static_cast<u32>(static_cast<i32>(static_cast<i16>(bits)));
        EXPECT_EQ(raw[run][lane * 2 + component], expected)
            << run << ':' << lane;
      }
    }
    EXPECT_EQ(packed[run][0], source[0]);
    EXPECT_EQ(packed[run][1], run ? source[1] : 0xa5a51234u);
    EXPECT_EQ(packed[run][2], run ? source[2] : 0xa5a5a5a5u);
  }
}

TEST(RdnaComputeAlu, ReadlaneCrossesSubgroupsWithExecEmpty) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static base::Array<u32, 16384> dest{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  const u32 shader[] = {
      0xbefe0380,       0xbeff0380, 0xd760000c, 256 | (168 << 9), 0xd760000d,
      256 | (129 << 9), 0xbefe03c1, 0xbeff03c1, 0x7e08020c,       0x7e0a020d,
      0xf0200300,       0x00000400, 0xbf810000};
  base::Copy(shader, shader + base::ArraySize(shader), code.begin());
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 64;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 8 << 1;
  const u64 address = reinterpret_cast<u64>(dest.data());
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0] = address >> 8;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 1] =
      ((address >> 40) & 0xff) | (62 << 20) | (63u << 30);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 2] = 63 >> 2;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 3] = 0x80000fac;
  for (bool wave32 : {false, true}) {
    dest.fill(0);
    const u32 dispatch[] = {1, 1, 1, 1u | (u32(wave32) << 15)};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    for (u32 lane = 0; lane < 64; lane++) {
      const u32 wave_base = wave32 ? lane & ~31u : 0;
      EXPECT_EQ(dest[lane * 2], wave_base + (wave32 ? 8u : 40u)) << lane;
      EXPECT_EQ(dest[lane * 2 + 1], wave_base + 1) << lane;
    }
  }
}

TEST(RdnaComputeAlu, LockstepWaveBranchesOnFullExecMask) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<u32, 16384> dest{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  const u32 shader[] = {0x34020082, 0xd8340000, 0x00000001, 0xbf8cc07f,
                        0xd8d80000, 0x02000001, 0xbe8c03a8, 0xbe8d0381,
                        0x7e060280, 0x7d8606f9, 0x06868a0d, 0x7d86000c,
                        0x8dea0a6a, 0xbefe046a, 0xbf880003, 0x4a080481,
                        0xf0200100, 0x00000400, 0xbf810000};
  base::Copy(shader, shader + base::ArraySize(shader), code.begin());
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 64;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = (8 << 1) | (1 << 15);
  const u64 address = reinterpret_cast<u64>(dest.data());
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0] = address >> 8;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 1] =
      ((address >> 40) & 0xff) | (20 << 20) | (63u << 30);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 2] = 63 >> 2;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 3] = 0x80000fac;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  for (u32 x = 0; x < 64; x++)
    EXPECT_EQ(dest[x], x < 40 ? x + 1 : 0u) << "lane " << x;
}

TEST(RdnaComputeAlu, BitfieldMaskWrapsWidthAndOffset) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<u32, 16384> dest{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  u32 pc_dw = 0;
  for (u32 i = 0; i < 4; i++) {
    code[pc_dw++] = 0xd7630000 | (4 + i);
    code[pc_dw++] = (8 + i * 2) | ((9 + i * 2) << 9);
  }
  code[pc_dw++] = 0xf0200f00;
  code[pc_dw++] = 0x00000400;
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
  const base::Array<u32, 8> operands = {0, 0, 31, 0, 36, 40, 8, 28};
  for (u32 i = 0; i < operands.size(); i++)
    regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8 + i] = operands[i];
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  const base::Array<u32, 4> expected = {0, 0x7fffffff, 0xf00, 0xf0000000};
  for (u32 i = 0; i < expected.size(); i++)
    EXPECT_EQ(dest[i], expected[i]);
}

TEST(RdnaComputeAlu, FmaMixSelectsPrecisionHalvesAndSourceModifiers) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<u32, 16384> dest{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  u32 pc_dw = 0;
  const auto mix = [&](u32 dst, u32 a, u32 b, u32 c, u32 half, u32 select,
                       u32 abs, u32 neg, bool clamp) {
    code[pc_dw++] = 0xcc200000 | dst | (abs << 8) | (select << 11) |
                    ((half & 4) << 12) | (u32(clamp) << 15);
    code[pc_dw++] = a | (b << 9) | (c << 18) | ((half & 3) << 27) | (neg << 29);
  };
  mix(4, 8, 9, 10, 6, 6, 0, 4, false);
  mix(5, 8, 9, 10, 6, 0, 2, 0, false);
  mix(6, 8, 242, 240, 0, 0, 0, 0, true);
  mix(7, 242, 9, 10, 6, 6, 0, 4, false);
  code[pc_dw++] = 0xf0200f00;
  code[pc_dw++] = 0x00000400;
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
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8] = base::BitCast<u32>(2.f);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 9] = 0x4200c400;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 10] = 0x3c00c000;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  const base::Array<float, 4> expected = {5.f, 6.f, 1.f, 2.f};
  for (u32 i = 0; i < expected.size(); i++)
    EXPECT_FLOAT_EQ(base::BitCast<float>(dest[i]), expected[i]);
}

TEST(RdnaComputeAlu, FmaMixHalfResultsPreserveTheOtherHalf) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<u32, 16384> dest{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  u32 pc_dw = 0;
  const auto mix = [&](u32 op, u32 dst, u32 a, u32 b, u32 c, u32 half,
                       u32 select, u32 abs, u32 neg, bool clamp) {
    code[pc_dw++] = 0xcc000000 | (op << 16) | dst | (abs << 8) |
                    (select << 11) | ((half & 4) << 12) | (u32(clamp) << 15);
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
  code[pc_dw++] = 0xf0200f00;
  code[pc_dw++] = 0x00000400;
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
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8] = base::BitCast<u32>(2.f);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 9] = 0x4200c400;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 10] = 0x3c00c000;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  const base::Array<u32, 4> expected = {0x12344500, 0x46005678, 0x40003c00,
                                        0x3c005678};
  for (u32 i = 0; i < expected.size(); i++)
    EXPECT_EQ(dest[i], expected[i]);
}

TEST(RdnaComputeAlu, PermlaneSelectsRowsAndHonorsInactiveLanes) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<u32, 16384> dest{};
  alignas(256) static base::Array<u32, 4096> code{};
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
  code[pc_dw++] = 0xbefe03ff;
  code[pc_dw++] = 0x55555555;
  code[pc_dw++] = 0xbeff03ff;
  code[pc_dw++] = 0x55555555;
  permute(0x377, 4, 1);
  permute(0x378, 5, 1);
  permute(0x378, 6, 0);
  permute(0x378, 7, 2);
  code[pc_dw++] = 0xbefe03c1;
  code[pc_dw++] = 0xbeff03c1;
  code[pc_dw++] = 0xf0200f00;
  code[pc_dw++] = 0x00000400;
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
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  for (u32 lane = 0; lane < 64; lane++) {
    const u32 selected = (lane & ~15u) | (15 - (lane & 15));
    EXPECT_EQ(dest[lane * 4], (lane & 1) ? 0x12345678 : selected) << lane;
    EXPECT_EQ(dest[lane * 4 + 1], (lane & 1) ? 0x12345678 : (selected ^ 16))
        << lane;
    EXPECT_EQ(dest[lane * 4 + 2], 0x12345678) << lane;
    EXPECT_EQ(dest[lane * 4 + 3], (lane & 1) ? 0x12345678 : 0u) << lane;
  }
}

TEST(RdnaComputeAlu, DppRowXmaskKeepsEachExchangeInsideItsRow) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required for this integration test";
  alignas(65536) static base::Array<u32, 16384> dest{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  u32 pc_dw = 0;
  constexpr base::Array<u32, 4> kMasks = {0, 4, 7, 15};
  for (u32 i = 0; i < kMasks.size(); i++) {
    code[pc_dw++] = 0x7e0002fa | ((4 + i) << 17);
    code[pc_dw++] = 0xff080000 | ((0x160 + kMasks[i]) << 8);
  }
  code[pc_dw++] = 0xf0200f00;
  code[pc_dw++] = 0x00000400;
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
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  for (u32 lane = 0; lane < 64; lane++)
    for (u32 i = 0; i < kMasks.size(); i++)
      EXPECT_EQ(dest[lane * 4 + i], lane ^ kMasks[i]) << lane << ":" << i;
}

TEST(RdnaComputeAlu, DppInvalidSourcePreservesDestination) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static base::Array<u32, 16384> dest{};
  alignas(256) static base::Array<u32, 4096> code{};
  u32 n = 0;
  const auto mov = [&](u32 reg, u32 value) {
    code[n++] = 0x7e0002ff | (reg << 17);
    code[n++] = value;
  };
  mov(1, 7);
  for (u32 reg = 4; reg <= 7; reg++)
    mov(reg, 1000 + reg);
  const auto add_dpp = [&](u32 reg, u32 control, bool zero) {
    code[n++] = (0x25u << 25) | (reg << 17) | (1u << 9) | 250;
    code[n++] = 0xff000000 | (control << 8) | (u32(zero) << 19);
  };
  add_dpp(4, 0x111, false);
  add_dpp(5, 0x111, true);
  add_dpp(6, 0x101, false);
  code[n++] = (0x25u << 25) | (7u << 17) | (1u << 9) | 256;
  code[n++] = 0xe0782000;
  code[n++] = 0x80020400;
  code[n++] = 0xbf810000;
  gpu::rdna::NextProgramGeneration();
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(dest.data()), sizeof(dest));
  gpu::ps5::Regs regs;
  const u64 pc = reinterpret_cast<u64>(code.data());
  regs[gpu::ps5::mmCOMPUTE_PGM_LO] = pc >> 8;
  regs[gpu::ps5::mmCOMPUTE_PGM_HI] = pc >> 40;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_X] = 64;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Y] = 1;
  regs[gpu::ps5::mmCOMPUTE_NUM_THREAD_Z] = 1;
  regs[gpu::ps5::mmCOMPUTE_PGM_RSRC2] = 16 << 1;
  const u64 address = reinterpret_cast<u64>(dest.data());
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8] = address;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 9] = (address >> 32) | (16u << 16);
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 10] = 64;
  regs[gpu::ps5::mmCOMPUTE_USER_DATA_0 + 11] = 0x14004;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  for (u32 lane = 0; lane < 64; lane++) {
    EXPECT_EQ(dest[lane * 4], (lane & 15) ? lane + 6 : 1004) << lane;
    EXPECT_EQ(dest[lane * 4 + 1], (lane & 15) ? lane + 6 : 7) << lane;
    EXPECT_EQ(dest[lane * 4 + 2], (lane & 15) != 15 ? lane + 8 : 1006) << lane;
    EXPECT_EQ(dest[lane * 4 + 3], lane + 7) << lane;
  }
}

TEST(RdnaComputeImage, GatherOffsetAndCompareUseFourDepthTaps) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static base::Array<u32, 16384> source{};
  alignas(65536) static base::Array<base::Array<u32, 16384>, 10> outputs{};
  alignas(256) static base::Array<u32, 4096> code{};
  for (u32 i = 0; i < 16; i++)
    source[(i / 4) * 64 + i % 4] = base::BitCast<u32>(i / 16.f);
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(source.data()), sizeof(source));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(outputs.data()), sizeof(outputs));
  for (u32 func = 0; func < outputs.size(); func++) {
    auto& dest = outputs[func];
    u32 n = 0;
    const auto mov = [&](u32 vgpr, u32 value) {
      code[n++] = 0x7e0002ff | (vgpr << 17);
      code[n++] = value;
    };
    mov(3, 0x3f01);
    const bool compare = func < 8;
    if (compare)
      mov(4, base::BitCast<u32>(3.f / 16));
    mov(compare ? 5 : 4, base::BitCast<u32>(0.4f));
    mov(compare ? 6 : 5, base::BitCast<u32>(0.4f));
    code[n++] = 0xbe9803ff;
    code[n++] = func << 13;
    const u32 op = compare ? 0x5f : func == 8 ? 0x57 : 0x47;
    code[n++] = 0xf0000108 | (op << 18);
    code[n++] = (func == 9 ? 4 : 3) | (8 << 8) | (6 << 21);
    code[n++] = 0xf0200f00;
    code[n++] = 0x00020800;
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
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    constexpr base::Array<u32, 4> kTaps = {6, 7, 3, 2};
    for (u32 i = 0; i < kTaps.size(); i++) {
      const u32 depth = kTaps[i] + (func == 9 ? 3 : 0);
      const base::Array<bool, 8> pass = {false,      3 < depth, 3 == depth,
                                         3 <= depth, 3 > depth, 3 != depth,
                                         3 >= depth, true};
      const float expected = compare ? (pass[func] ? 1.f : 0.f) : depth / 16.f;
      EXPECT_FLOAT_EQ(base::BitCast<float>(dest[i]), expected)
          << "compare " << func << " tap " << i;
    }
  }
}

TEST(RdnaComputeImageConversion, SrgbReadAndUnormMipWriteShareStorage) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static base::Array<u8, 65536> image{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(
      gpu::gcn::BuildTextureLayout32(layout, 8, 8, 8, 1, 4, 0x11b, false, 4));
  ASSERT_LE(layout.size, image.size());
  base::Array<u32, 64> pixels;
  pixels.fill(0xff408080);
  ASSERT_TRUE(
      gpu::gcn::RetileTextureMip32(pixels.data(), image.data(), layout, 0, 0));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(image.data()), image.size());
  const u32 shader[] = {0xf0000f08, 0x00000400, 0xf0200f08, 0x00020400,
                        0xbf810000};
  base::Copy(shader, shader + base::ArraySize(shader), code.begin());
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
    regs[ud + 1] =
        ((address >> 40) & 0xff) | ((view ? 56 : 130) << 20) | (3u << 30);
    regs[ud + 2] = 1 | (7 << 14);
    regs[ud + 3] = 0x91b00fac | (view << 12) | (view << 16);
    regs[ud + 5] = 3 << 4;
  }
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  base::Array<u32, 64> base{};
  base::Array<u32, 16> mip{};
  ASSERT_TRUE(
      gpu::gcn::DetileTextureMip32(image.data(), base.data(), layout, 0, 0));
  ASSERT_TRUE(
      gpu::gcn::DetileTextureMip32(image.data(), mip.data(), layout, 1, 0));
  EXPECT_EQ(base, pixels);
  for (u32 pixel : mip) {
    EXPECT_NEAR(pixel & 255, 55, 1);
    EXPECT_NEAR((pixel >> 8) & 255, 55, 1);
    EXPECT_NEAR((pixel >> 16) & 255, 13, 1);
    EXPECT_EQ(pixel >> 24, 255u);
  }
}

TEST(RdnaComputeImageConversion,
     SrgbLoadsAndFilteringDecodeRgbBeforeInterpolation) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static base::Array<base::Array<u32, 16384>, 3> inputs{},
      outputs{};
  alignas(256) static base::Array<u32, 4096> code{};
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
      code[pc++] = 0x7e0002ff;
      code[pc++] = base::BitCast<u32>(run == 1 ? .25f : .5f);
    }
    code[pc++] = 0x7e020280;
    code[pc++] = run ? 0xf09c0f00 : 0xf0000f00;
    code[pc++] = 0x00600400;
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
    regs[ud + 3] = 0x80000fac;
    const u64 dst = reinterpret_cast<u64>(output.data());
    regs[ud + 8] = dst;
    regs[ud + 9] = (dst >> 32) & 0xffff;
    regs[ud + 10] = 16;
    regs[ud + 11] = 0x20000;
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    const base::Array<float, 4> expected =
        run == 2 ? base::Array<float, 4>{.5f, .5f, .5f, 128.f / 255.f}
                 : base::Array<float, 4>{.2158605f, .2158605f,
                                         10.f / (255.f * 12.92f), 64.f / 255.f};
    for (u32 i = 0; i < 4; ++i)
      EXPECT_NEAR(base::BitCast<float>(output[i]), expected[i], 1e-6f)
          << "run=" << run << " channel=" << i;
  }
}

TEST(RdnaComputeImageConversion, ExplicitGradientsSelectTheImageMip) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "Vulkan is required";
  alignas(65536) static base::Array<u8, 65536> input{};
  alignas(65536) static base::Array<base::Array<u32, 16384>, 6> outputs{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(
      gpu::gcn::BuildTextureLayout32(layout, 8, 8, 8, 1, 4, 0x109, false, 4));
  for (u32 mip = 0; mip < 4; ++mip) {
    base::Vector<u32> pixels((8u >> mip) * (8u >> mip),
                             0xff000000u | (32 + mip * 64));
    ASSERT_TRUE(gpu::gcn::RetileTextureMip32(pixels.data(), input.data(),
                                             layout, mip, 0));
  }
  const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
  for (u32 trial = 0; trial < 6; ++trial) {
    const u32 mip = trial % 3;
    if (trial == 3) {
      gpu::render::RecordComputeInFrame();
      gpu::render::BeginFrame(renderer);
      ASSERT_TRUE(gpu::render::CsBindsImages());
    }
    auto& output = outputs[trial];
    u32 pc = 0;
    const float step = float(1u << mip) / 8.f;
    const float address[] = {step, 0.f, 0.f, step, .5f, .5f};
    for (u32 i = 0; i < 6; ++i) {
      code[pc++] = 0x7e0002ff | ((4 + i) << 17);
      code[pc++] = base::BitCast<u32>(address[i]);
    }
    code[pc++] = 0xf0880f08;
    code[pc++] = 0x00601004;
    code[pc++] = 0xe0780000;
    code[pc++] = 0x80021000;
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
    const u64 src = reinterpret_cast<u64>(input.data());
    regs[ud] = src >> 8;
    regs[ud + 1] = ((src >> 40) & 255) | (56u << 20) | (3u << 30);
    regs[ud + 2] = 1 | (7 << 14);
    regs[ud + 3] = 0x90930fac;
    regs[ud + 5] = 3 << 4;
    const u64 dst = reinterpret_cast<u64>(output.data());
    regs[ud + 8] = dst;
    regs[ud + 9] = (dst >> 32) & 0xffff;
    regs[ud + 10] = 16;
    regs[ud + 11] = 0x20000;
    regs[ud + 13] = 3 << 20;
    regs[ud + 14] = 1 << 26;
    const u32 dispatch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    EXPECT_NEAR(base::BitCast<float>(output[0]), (32 + mip * 64) / 255.f,
                1e-6f);
    EXPECT_FLOAT_EQ(base::BitCast<float>(output[3]), 1.f);
  }
  gpu::render::EndFrame(renderer, 0);
}

TEST(RdnaComputeImageConversion,
     DecoderRg16IntegerLoadsSignExtendBothComponents) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "Vulkan is required";
  alignas(65536) static base::Array<base::Array<u32, 16384>, 2> sources{},
      outputs{};
  alignas(256) static base::Array<u32, 16384> code{};
  const u32 program[] = {0x7e020280, 0xf0000300, 0x00000400,
                         0xe0740000, 0x80020400, 0xbf810000};
  base::Copy(program, program + base::ArraySize(program), code.begin());
  for (u32 run = 0; run < 2; ++run) {
    auto& source = sources[run];
    auto& output = outputs[run];
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
    const u64 src = reinterpret_cast<u64>(source.data()),
              dst = reinterpret_cast<u64>(output.data());
    regs[ud] = src >> 8;
    regs[ud + 1] = ((src >> 40) & 255) | ((27 + run) << 20);
    regs[ud + 3] = 0x8000002c;
    regs[ud + 8] = dst;
    regs[ud + 9] = (dst >> 32) & 0xffff;
    regs[ud + 10] = 8;
    regs[ud + 11] = 0x20000;
    const u32 launch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, launch, 4);
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    EXPECT_EQ(output[0], run ? 0xffffffffu : 0xffffu);
    EXPECT_EQ(output[1], run ? 0xffff8001u : 0x8001u);
    EXPECT_EQ(output[2], 0xa5a5a5a5);
  }
}

TEST(RdnaComputeImageConversion, DecoderRg16IntegerStoresPackComponents) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "Vulkan is required";
  alignas(65536) static base::Array<base::Array<u32, 16384>, 4> sources{},
      outputs{};
  alignas(256) static base::Array<u32, 16384> code{};
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
        0x7e020280, 0xf0000300,
        0x00000400, 0xf0200000 | ((green_only ? 2u : 3u) << 8),
        0x00020400, 0xbf810000};
    base::Copy(program, program + base::ArraySize(program), code.begin());
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
      regs[first + 3] = 0x8000002c;
    };
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, source.data(),
               signed_format ? 63 : 62);
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, output.data(),
               signed_format ? 28 : 27);
    const u32 launch[] = {1, 1, 1, 1};
    gpu::ps5::DispatchCompute(renderer, regs, launch, 4);
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    for (u32 x = 0; x < 3; ++x) {
      const u32 red = green_only ? 0x1234 : source[x * 2] & 0xffff;
      const u32 green = source[x * 2 + (green_only ? 0 : 1)] & 0xffff;
      EXPECT_EQ(output[x], red | (green << 16))
          << "run " << run << " texel " << x;
    }
    EXPECT_EQ(output[3], 0xa5a51234);
  }
}

TEST(RdnaComputeImageConversion,
     DecoderPackedIntegersRoundTripWithoutNormalization) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "Vulkan is required";
  struct Case {
    u32 format, channels, bits;
    bool is_signed;
  };
  constexpr Case kCases[] = {{5, 1, 8, false},   {18, 2, 8, false},
                             {60, 4, 8, false},  {61, 4, 8, true},
                             {69, 4, 16, false}, {70, 4, 16, true}};
  struct Storage {
    base::Array<u32, 16384> source, packed, result;
  };
  alignas(65536) static base::Array<Storage, 2 * base::ArraySize(kCases)>
      storage{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(storage.data()), sizeof(storage));
  for (u32 run = 0; run < storage.size(); ++run) {
    const auto c = kCases[run % base::ArraySize(kCases)];
    const bool physical = run >= base::ArraySize(kCases);
    auto& s = storage[run];
    const u32 mask = (1u << c.bits) - 1;
    const u32 values[] = {
        mask, 1u << (c.bits - 1),      0, 127, 1, 42, mask - 1, 17, 63, 128,
        2,    (1u << (c.bits - 1)) - 1};
    base::Copy(values, values + base::ArraySize(values), s.source.begin());
    s.packed.fill(0xa5a5a5a5);
    s.result.fill(0xa5a5a5a5);
    for (u32 pass = 0; pass < 2; ++pass) {
      const u32 program[] = {
          0x7e020280, 0xf0000f00,
          0x00000400, 0xf0200000 | ((pass ? 15u : (1u << c.channels) - 1) << 8),
          0x00020400, 0xbf810000};
      const u32 prefix[] = {0x7e280280, 0x7e2a0280, 0xdc308000, 0x147d0014};
      u32 offset = 0;
      if (physical) {
        base::Copy(prefix, prefix + base::ArraySize(prefix), code.begin());
        offset = base::ArraySize(prefix);
      }
      base::Copy(program, program + base::ArraySize(program),
                 code.begin() + offset);
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
        regs[first + 3] = 0x80000fac;
      };
      const u32 ud = gpu::ps5::mmCOMPUTE_USER_DATA_0;
      descriptor(ud, pass ? s.packed.data() : s.source.data(),
                 pass ? c.format : 76);
      descriptor(ud + 8, pass ? s.result.data() : s.packed.data(),
                 pass ? 76 : c.format);
      const u32 launch[] = {1, 1, 1, 1};
      gpu::ps5::DispatchCompute(renderer, regs, launch, 4);
      ASSERT_TRUE(FlushCsWritesLanded(renderer));
    }
    for (u32 x = 0; x < 3; ++x) {
      for (u32 channel = 0; channel < c.channels; ++channel) {
        u32 expected = values[x * 4 + channel] & mask;
        if (c.is_signed && (expected & (1u << (c.bits - 1))))
          expected |= ~mask;
        EXPECT_EQ(s.result[x * 4 + channel], expected)
            << "format " << c.format << " physical " << physical << " texel "
            << x << " channel " << channel;
      }
    }
    EXPECT_EQ(s.result[12], 0xa5a5a5a5);
    const auto* packed = reinterpret_cast<const u8*>(s.packed.data());
    EXPECT_EQ(packed[3 * c.channels * c.bits / 8], 0xa5);
  }
}

TEST(RdnaComputeImageConversion,
     LinearIntegerViewsShareWritesWithinOneDispatch) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "Vulkan is required";
  alignas(65536) static base::Array<base::Array<u32, 16384>, 2> memory{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(memory.data()), sizeof(memory));
  for (u32 pass = 0; pass < 3; ++pass) {
    const u32 reverse = pass ? pass - 1 : 0;
    memory[0].fill(0xa5a5a5a5);
    memory[0][1] = 0xdeadbeef;
    memory[1].fill(0xa5a5a5a5);
    const u64 dst = reinterpret_cast<u64>(memory[1].data());
    const u32 program[] = {0x7e0002ff,
                           reverse ? 5u : 1u,
                           0x7e020280,
                           0x7e0802ff,
                           reverse ? 0x66u : 0x12345678u,
                           0xf0200100,
                           0x00000400,
                           0x7e0402ff,
                           reverse ? 1u : 5u,
                           0x7e060280,
                           0xf0000100,
                           0x00020502,
                           0x7e2802ff,
                           u32(dst),
                           0x7e2a02ff,
                           u32(dst >> 32),
                           0xdc708000,
                           0x007d0514,
                           0xbf810000};
    base::Copy(program, program + base::ArraySize(program), code.begin());
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
      regs[first + 1] =
          ((base >> 40) & 255) | ((byte ? 5u : 20u) << 20) | ((last & 3) << 30);
      regs[first + 2] = last >> 2;
      regs[first + 3] = 0x80000004;
    };
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0, reverse);
    descriptor(gpu::ps5::mmCOMPUTE_USER_DATA_0 + 8, !reverse);
    const u32 launch[] = {1, 1, 1, 1};
    for (auto& block : memory)
      gpu::render::CsNoteGuestWrites(u64(block.data()),
                                     u64(block.data()) + sizeof(block));
    gpu::ps5::DispatchCompute(renderer, regs, launch, 4);
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    if (!pass)
      continue;
    EXPECT_EQ(memory[1][0], reverse ? 0xdead66efu : 0x56u);
    EXPECT_EQ(memory[0][1], reverse ? 0xdead66efu : 0x12345678u);
    EXPECT_EQ(memory[0][0], 0xa5a5a5a5);
    EXPECT_EQ(memory[0][2], 0xa5a5a5a5);
  }
}

TEST(RdnaComputeImageConversion,
     LinearIntegerArraysRespectRowsLayersAndBounds) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "Vulkan is required";
  alignas(65536) static base::Array<base::Array<u32, 16384>, 2> memory{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(memory.data()), sizeof(memory));
  const u32 program[] = {0x7e280280, 0x7e2a0280, 0xdc308000,
                         0x147d0014, 0xf0000128, 0x00000400,
                         0xf0200128, 0x00020400, 0xbf810000};
  base::Copy(program, program + base::ArraySize(program), code.begin());
  auto* source = reinterpret_cast<u8*>(memory[0].data());
  for (u32 layer = 0; layer < 2; ++layer)
    for (u32 y = 0; y < 3; ++y)
      for (u32 x = 0; x < 7; ++x)
        source[(layer * 3 + y) * 256 + x] = 16 + layer * 32 + y * 8 + x;
  for (u32 base_layer = 0; base_layer < 2; ++base_layer) {
    memory[1].fill(0xa5a5a5a5);
    base::Array<u32, 16384> expected;
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
      regs[ud + 1] =
          ((base >> 40) & 255) | ((image ? 20u : 5u) << 20) | (2u << 30);
      regs[ud + 2] = 1 | (2u << 14);
      regs[ud + 3] = 0xd0000004;
      regs[ud + 4] = 1 | (base_layer << 16);
    }
    const u32 launch[] = {1, 1, 1, 1};
    for (auto& block : memory)
      gpu::render::CsNoteGuestWrites(u64(block.data()),
                                     u64(block.data()) + sizeof(block));
    gpu::ps5::DispatchCompute(renderer, regs, launch, 4);
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    EXPECT_EQ(memory[1], expected) << "base layer " << base_layer;
  }
}

TEST(RdnaComputeImageConversion,
     AstroBc6CubeArrayStagingExceedsRawBufferLimit) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static base::Array<u8, 0x1080000> input{};
  alignas(65536) static base::Array<u32, 16384> output{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::gcn::TextureLayout32 blocks, pixels;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(blocks, 64, 64, 64, 192, 9, 0x105,
                                             false, 16));
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(pixels, 256, 256, 256, 192, 9, 8,
                                             false, 16));
  ASSERT_EQ(blocks.size, input.size());
  ASSERT_EQ(pixels.size, 0x100b0000u);
  if (pixels.size > gpu::render::Device().caps().max_storage_buffer_range)
    GTEST_SKIP() << "The device cannot bind the expanded BC6 image";
  base::Array<u8, 16> encoded{};
  u32 bit = 0;
  const auto put = [&](u32 value, u32 count) {
    for (u32 i = 0; i < count; ++i, ++bit)
      encoded[bit / 8] |= ((value >> i) & 1) << (bit % 8);
  };
  put(3, 5);
  for (u32 endpoint = 0; endpoint < 2; ++endpoint) {
    put(0, 10);
    put(512, 10);
    put(1023, 10);
  }
  ASSERT_TRUE(gpu::gcn::RetileTextureMip32(encoded.data(), input.data(), blocks,
                                           8, 191));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(input.data()), input.size());
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(output.data()), sizeof(output));
  const u32 program[] = {0x7e020280, 0x7e0402ff, 191,        0xf0000f28,
                         0x00000400, 0xe0780000, 0x80020400, 0xbf810000};
  base::Copy(program, program + base::ArraySize(program), code.begin());
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
  regs[ud + 3] = 0xb0588fac;
  regs[ud + 4] = 191;
  regs[ud + 5] = 0x80;
  const u64 dst = reinterpret_cast<u64>(output.data());
  regs[ud + 8] = dst;
  regs[ud + 9] = (dst >> 32) & 0xffff;
  regs[ud + 10] = 16;
  regs[ud + 11] = 0x20000;
  const u32 dispatch[] = {1, 1, 1, 1};
  gpu::ps5::DispatchCompute(renderer, regs, dispatch, 4);
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  const float expected[] = {0, 1.5146484375f, 65504, 1};
  for (u32 i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(base::BitCast<float>(output[i]), expected[i]);
}

TEST(RdnaComputeImageConversion, Bc1LoadsRespectTilingMipsLayersAndSrgb) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static base::Array<base::Array<u8, 131072>, 4> inputs{};
  alignas(65536) static base::Array<u32, 16384> output{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(inputs.data()), sizeof(inputs));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(output.data()), sizeof(output));
  constexpr u16 kColors[] = {0xf800, 0x07e0, 0x001f, 0x8410};
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
        base::Array<u64, 4> encoded{};
        for (u32 b = 0; b < 4; b++)
          encoded[b] = kColors[(mip + layer + b) % 4];
        ASSERT_TRUE(gpu::gcn::RetileTextureMip32(encoded.data(), input.data(),
                                                 blocks, mip, layer));
      }
    for (u32 mip = 0; mip < 4; mip++)
      for (u32 layer = 0; layer < 2; layer++) {
        const u32 xy = mip ? 0 : 5;
        const u32 shader[] = {0x7e0802ff, xy,         0x7e0a02ff, xy,
                              0x7e0c02ff, layer,      0xf0000f28, 0x00000804,
                              0xe0780000, 0x80020800, 0xbf810000};
        base::Copy(shader, shader + base::ArraySize(shader), code.begin());
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
        regs[ud + 1] =
            ((src >> 40) & 255) | ((srgb ? 170 : 169) << 20) | (3u << 30);
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
        ASSERT_TRUE(FlushCsWritesLanded(renderer));
        const u16 color = kColors[(mip + layer + (mip ? 0 : 3)) % 4];
        const u32 rgb[] = {((color >> 11) * 527 + 23) >> 6,
                           (((color >> 5) & 63) * 259 + 33) >> 6,
                           ((color & 31) * 527 + 23) >> 6};
        for (u32 c = 0; c < 4; c++) {
          float expected = c < 3 ? rgb[c] / 255.f : 1.f;
          if (srgb && c < 3)
            expected = expected <= .04045f
                           ? expected / 12.92f
                           : std::pow((expected + .055f) / 1.055f, 2.4f);
          EXPECT_NEAR(base::BitCast<float>(output[c]), expected, 0.00001f)
              << "variant " << variant << " mip " << mip << " layer " << layer;
        }
      }
  }
}

TEST(RdnaComputeImageConversion, LargeShadowArrayReadsPastRawBufferLimit) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static base::Array<u8, 320u * 1024 * 1024> input{};
  alignas(65536) static base::Array<u32, 16384> output{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(layout, 1024, 1024, 1024, 80, 1,
                                             0x118, false, 4));
  ASSERT_EQ(layout.size, input.size());
  if (layout.size > gpu::render::Device().caps().max_storage_buffer_range)
    GTEST_SKIP() << "The device cannot bind the shadow array";
  base::Vector<u32> pixels(1024 * 1024, base::BitCast<u32>(0.75f));
  ASSERT_TRUE(
      gpu::gcn::RetileTextureMip32(pixels.data(), input.data(), layout, 0, 79));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(input.data()), input.size());
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(output.data()), sizeof(output));
  const u32 shader[] = {0x7e020280, 0x7e0402ff, 79,         0xf0000128,
                        0x00000400, 0xe0700000, 0x80020400, 0xbf810000};
  base::Copy(shader, shader + base::ArraySize(shader), code.begin());
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
  ASSERT_TRUE(FlushCsWritesLanded(renderer));
  EXPECT_FLOAT_EQ(base::BitCast<float>(output[0]), 0.75f);
}

TEST(RdnaComputeImageConversion,
     UniformDescriptorTablePreservesDynamicCubeSelection) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static base::Array<u32, 16384> texture{};
  alignas(65536) static base::Array<base::Array<u32, 16384>, 3> tables{},
      outputs{};
  alignas(256) static base::Array<u32, 4096> code{};
  for (u32 face = 0; face < 6; face++)
    texture[face * 64] = base::BitCast<u32>((face + 1) * .25f);
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(texture.data()), sizeof(texture));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(tables.data()), sizeof(tables));
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(outputs.data()), sizeof(outputs));
  const u64 image = reinterpret_cast<u64>(texture.data());
  const base::Array<u32, 8> descriptor = {u32(image >> 8),
                                          u32((image >> 40) & 255) | (22 << 20),
                                          0,
                                          0xb0000fac,
                                          5,
                                          0,
                                          0,
                                          0};
  for (u32 variant = 0; variant < 3; variant++) {
    auto& table = tables[variant];
    auto& output = outputs[variant];
    base::Copy(descriptor.begin(), descriptor.end(), table.begin());
    base::Copy(descriptor.begin(), descriptor.end(), table.begin() + 8);
    if (variant == 1)
      table[8]++;
    output[0] = output[1] = output[2] = 0xdeadbeef;
    const u32 shader[] = {0x7e0a0210,
                          0x7ed40505,
                          variant == 2 ? 0x8f16846a : 0x8f16856a,
                          0xf42c0004,
                          0x2c000000,
                          0x7e1002ff,
                          base::BitCast<u32>(1.5f),
                          0x7e1202ff,
                          base::BitCast<u32>(1.5f),
                          0x7e140d05,
                          0x7e160280,
                          0xf0900118,
                          0x00e00408,
                          0xe0702000,
                          0x80030405,
                          0xbf810000};
    base::Copy(shader, shader + base::ArraySize(shader), code.begin());
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
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    if (variant == 0) {
      EXPECT_FLOAT_EQ(base::BitCast<float>(output[0]), .25f);
      EXPECT_FLOAT_EQ(base::BitCast<float>(output[1]), .5f);
      EXPECT_EQ(output[2], 0u);
    } else {
      EXPECT_EQ(output[0], 0xdeadbeefu);
      EXPECT_EQ(output[1], 0xdeadbeefu);
      EXPECT_EQ(output[2], 0xdeadbeefu);
    }
  }
}
TEST(RdnaComputeImageConversion, PlainStoresValidateOnlyTheAccessedMip) {
  auto& renderer = gpu::render::DefaultRenderer();
  if (!gpu::render::Init(renderer))
    GTEST_SKIP() << "A Vulkan device is required";
  alignas(65536) static base::Array<base::Array<u32, 2 * 1024 * 1024>, 5>
      images{};
  alignas(256) static base::Array<u32, 4096> code{};
  gpu::ps5::NoteGpuPool(reinterpret_cast<u64>(images.data()), sizeof(images));
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(layout, 512, 512, 512, 1, 9, 0x109,
                                             false, 16));
  ASSERT_LE(layout.size, sizeof(images[0]));
  constexpr u32 kLevels[] = {0, 3, 8, 9, 0};
  constexpr u32 kValues[] = {0x01234567, 0x89abcdef, 0xfedcba98, 0x76543210};
  for (u32 variant = 0; variant < 5; variant++) {
    auto& image = images[variant];
    image.fill(0xdeadbeef);
    const u32 mip = kLevels[variant];
    const bool explicit_mip = variant == 4;
    const u32 shader[] = {0x7e020280,
                          0x7e040280,
                          0x7e0802ff,
                          kValues[0],
                          0x7e0a02ff,
                          kValues[1],
                          0x7e0c02ff,
                          kValues[2],
                          0x7e0e02ff,
                          kValues[3],
                          explicit_mip ? 0xf0240f08u : 0xf0200f08u,
                          0x00000400,
                          0xbf810000};
    base::Copy(shader, shader + base::ArraySize(shader), code.begin());
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
    ASSERT_TRUE(FlushCsWritesLanded(renderer));
    for (u32 level = 0; level < layout.mip_levels; level++) {
      base::Vector<u32> pixels(layout.mips[level].width *
                               layout.mips[level].height * 4);
      ASSERT_TRUE(gpu::gcn::DetileTextureMip32(image.data(), pixels.data(),
                                               layout, level, 0));
      for (u32 c = 0; c < 4; c++)
        EXPECT_EQ(pixels[c],
                  level == mip && !explicit_mip ? kValues[c] : 0xdeadbeefu)
            << "variant " << variant << " mip " << level;
      EXPECT_EQ(pixels.back(),
                pixels.size() == 4 && level == mip && !explicit_mip
                    ? kValues[3]
                    : 0xdeadbeefu);
    }
  }
}
}  // namespace

namespace gpu::render {
u32 g_submit_queue = 0, g_dcb_n = 0;
}  // namespace gpu::render
// NOLINTNEXTLINE(readability-identifier-naming): C-linkage bridge
extern "C" bool prosperity_ps5_is_display_buffer(u64) {
  return false;
}
