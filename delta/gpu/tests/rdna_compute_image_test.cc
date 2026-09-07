#include <array>
#include <bit>
#include <cstring>

#include <gtest/gtest.h>

#include "gpu/ps5/compute_dispatch.h"
#include "gpu/ps5/guest_address.h"
#include "gpu/ps5/rdna/rdna_decode.h"

namespace {
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
}  // namespace

// Performance counters normally owned by the command-processor composition.
namespace gpu::rhi {
u64 g_ns_dcb = 0, g_ns_dcb_lock = 0;
u32 g_submit_queue = 0, g_dcb_n = 0;
}
extern "C" bool prosperity_ps5_is_display_buffer(u64) { return false; }
