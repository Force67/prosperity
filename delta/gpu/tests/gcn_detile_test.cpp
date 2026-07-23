#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "gpu/ps4/gcn/gcn_detile.h"

namespace {

struct SplitMode {
  uint32_t tiling_index;
  uint32_t tile_split_bytes;
  uint32_t bank_height;
  uint32_t macro_aspect;
};

uint64_t ExpectedTiledOffset(const SplitMode &mode, uint32_t x, uint32_t y,
                             uint32_t layer, uint32_t pitch, uint32_t height) {
  constexpr uint32_t kNumPipes = 8;
  constexpr uint32_t kNumBanks = 16;

  const uint32_t pixel_index = (x & 1) | ((y & 1) << 1) |
                               (((x >> 1) & 1) << 2) | (((y >> 1) & 1) << 3) |
                               (((x >> 2) & 1) << 4) | (((y >> 2) & 1) << 5);
  uint32_t element_offset = pixel_index * sizeof(uint32_t);
  const uint32_t tile_split_slice = element_offset / mode.tile_split_bytes;
  element_offset %= mode.tile_split_bytes;

  const uint32_t slices_per_tile = 256 / mode.tile_split_bytes;
  const uint32_t macro_pitch = 8 * kNumPipes * mode.macro_aspect;
  const uint32_t macro_height =
      8 * mode.bank_height * kNumBanks / mode.macro_aspect;
  const uint32_t macro_tile_bytes = mode.tile_split_bytes * (macro_pitch / 8) *
                                    (macro_height / 8) /
                                    (kNumPipes * kNumBanks);
  const uint32_t macro_tiles_per_row = pitch / macro_pitch;
  const uint64_t macro_tile_offset =
      static_cast<uint64_t>((y / macro_height) * macro_tiles_per_row +
                            x / macro_pitch) *
      macro_tile_bytes;
  const uint32_t macro_tiles_per_slice =
      macro_tiles_per_row * (height / macro_height);
  const uint64_t slice_bytes =
      static_cast<uint64_t>(macro_tiles_per_slice) * macro_tile_bytes;
  const uint64_t slice_offset =
      slice_bytes * (tile_split_slice + slices_per_tile * layer);
  const uint32_t tile_row = (y / 8) % mode.bank_height;
  const uint32_t tile_offset = tile_row * mode.tile_split_bytes;
  const uint64_t total_offset =
      slice_offset + macro_tile_offset + tile_offset + element_offset;

  const uint32_t tile_x = x >> 3;
  const uint32_t tile_y = y >> 3;
  const uint32_t pipe = ((tile_x & 1) ^ (tile_y & 1) ^ ((tile_x >> 1) & 1)) |
                        ((((tile_x >> 1) & 1) ^ ((tile_y >> 1) & 1)) << 1) |
                        ((((tile_x >> 2) & 1) ^ ((tile_y >> 2) & 1)) << 2);

  const uint32_t bank_x = tile_x / kNumPipes;
  const uint32_t bank_y = tile_y / mode.bank_height;
  uint32_t bank =
      ((bank_x & 1) ^ ((bank_y >> 3) & 1)) |
      ((((bank_x >> 1) & 1) ^ ((bank_y >> 2) & 1) ^ ((bank_y >> 3) & 1)) << 1) |
      ((((bank_x >> 2) & 1) ^ ((bank_y >> 1) & 1)) << 2) |
      ((((bank_x >> 3) & 1) ^ (bank_y & 1)) << 3);
  bank ^= 7 * layer;
  bank ^= 9 * tile_split_slice;
  bank &= kNumBanks - 1;

  return (total_offset & 255) | (static_cast<uint64_t>(pipe) << 8) |
         (static_cast<uint64_t>(bank) << 11) | ((total_offset >> 8) << 15);
}

void VerifySplitMode(const SplitMode &mode) {
  constexpr uint32_t kLayers = 2;
  const uint32_t macro_pitch = 8 * 8 * mode.macro_aspect;
  const uint32_t macro_height = 8 * mode.bank_height * 16 / mode.macro_aspect;
  const uint32_t width = macro_pitch * 2;
  const uint32_t height = macro_height * 2;

  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(
      layout, width, height, width, kLayers, 1, mode.tiling_index, false));
  ASSERT_TRUE(layout.mips[0].macro_tiled);
  ASSERT_EQ(layout.mips[0].pitch, width);
  ASSERT_EQ(layout.mips[0].stored_height, height);

  std::vector<uint32_t> tiled(layout.size / sizeof(uint32_t));
  std::vector<bool> occupied(tiled.size());
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const uint64_t offset =
            ExpectedTiledOffset(mode, x, y, layer, width, height);
        ASSERT_EQ(offset % sizeof(uint32_t), 0u);
        ASSERT_LE(offset + sizeof(uint32_t), layout.size);
        const size_t index = offset / sizeof(uint32_t);
        ASSERT_FALSE(occupied[index]);
        occupied[index] = true;
        tiled[index] = 1 + (layer * height + y) * width + x;
      }
    }
  }
  ASSERT_EQ(static_cast<size_t>(
                std::count(occupied.begin(), occupied.end(), true)),
            occupied.size());

  std::vector<uint32_t> linear(static_cast<size_t>(width) * height);
  std::vector<uint32_t> expected(linear.size());
  std::vector<uint32_t> retiled(tiled.size());
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    ASSERT_TRUE(gpu::gcn::DetileTextureMip32(tiled.data(), linear.data(),
                                             layout, 0, layer));
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        expected[static_cast<size_t>(y) * width + x] =
            1 + (layer * height + y) * width + x;
      }
    }
    EXPECT_EQ(linear, expected);
    ASSERT_TRUE(gpu::gcn::RetileTextureMip32(expected.data(), retiled.data(),
                                             layout, 0, layer));
  }
  EXPECT_EQ(retiled, tiled);
}

void Verify16BitRoundTrip(uint32_t tiling_index) {
  constexpr uint32_t kWidth = 256;
  constexpr uint32_t kHeight = 128;
  constexpr uint32_t kLayers = 2;
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(
      layout, kWidth, kHeight, kWidth, kLayers, 1, tiling_index, false, 2));

  std::vector<uint8_t> tiled(layout.size);
  std::vector<uint16_t> source(static_cast<size_t>(kWidth) * kHeight);
  std::vector<uint16_t> result(source.size());
  for (uint32_t layer = 0; layer < kLayers; layer++) {
    for (size_t i = 0; i < source.size(); i++)
      source[i] = static_cast<uint16_t>(1 + i + layer * source.size());
    ASSERT_TRUE(gpu::gcn::RetileTextureMip32(source.data(), tiled.data(),
                                             layout, 0, layer));
  }
  for (uint32_t layer = 0; layer < kLayers; layer++) {
    for (size_t i = 0; i < source.size(); i++)
      source[i] = static_cast<uint16_t>(1 + i + layer * source.size());
    ASSERT_TRUE(gpu::gcn::DetileTextureMip32(tiled.data(), result.data(),
                                             layout, 0, layer));
    EXPECT_EQ(result, source);
  }
}

uint64_t HashBytes(const uint8_t *data, size_t size, uint64_t hash) {
  for (size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

TEST(GcnDetile, Depth64ByteSplitIsBijectiveAcrossArrayLayers) {
  VerifySplitMode({0, 64, 4, 4});
}

TEST(GcnDetile, Depth128ByteSplitIsBijectiveAcrossArrayLayers) {
  VerifySplitMode({1, 128, 2, 2});
}

TEST(GcnDetile, Depth64ByteSplitSupports16BitElements) {
  Verify16BitRoundTrip(0);
}

TEST(GcnDetile, Thin2DMacroSupports16BitElements) {
  Verify16BitRoundTrip(14);
}

TEST(GcnDetile, Thin2DMacroSupports64BitElements) {
  constexpr uint32_t kWidth = 256;
  constexpr uint32_t kHeight = 128;
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(
      layout, kWidth, kHeight, kWidth, 1, 1, 14, false, 8));
  std::vector<uint64_t> source(static_cast<size_t>(kWidth) * kHeight);
  for (size_t i = 0; i < source.size(); i++) source[i] = i + 1;
  std::vector<uint8_t> tiled(layout.size);
  std::vector<uint64_t> result(source.size());
  ASSERT_TRUE(gpu::gcn::RetileTextureMip32(source.data(), tiled.data(), layout,
                                           0, 0));
  ASSERT_TRUE(gpu::gcn::DetileTextureMip32(tiled.data(), result.data(), layout,
                                           0, 0));
  EXPECT_EQ(result, source);
}

TEST(GcnDetile, AllModesAndElementWidthsMatchReferenceDigest) {
  uint64_t hash = 1469598103934665603ull;
  for (uint32_t tiling = 0; tiling <= 31; ++tiling) {
    if (tiling > 26 && tiling != 31) continue;
    for (uint32_t elem : {2u, 4u, 8u, 16u}) {
      gpu::gcn::TextureLayout32 layout;
      ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(
          layout, 263, 137, 271, 3, 4, tiling, false, elem));
      std::vector<uint8_t> tiled(layout.size);
      for (size_t i = 0; i < tiled.size(); ++i)
        tiled[i] = static_cast<uint8_t>((i * 193u + i / 29u + tiling) & 255u);

      for (uint32_t mip = 0; mip < layout.mip_levels; ++mip) {
        const auto &level = layout.mips[mip];
        std::vector<uint8_t> linear(static_cast<size_t>(level.width) *
                                    level.height * elem);
        for (uint32_t layer = 0; layer < layout.layers; ++layer) {
          ASSERT_TRUE(gpu::gcn::DetileTextureMip32(
              tiled.data(), linear.data(), layout, mip, layer));
          hash = HashBytes(linear.data(), linear.size(), hash);
        }
      }

      std::fill(tiled.begin(), tiled.end(), 0xa5);
      for (uint32_t mip = 0; mip < layout.mip_levels; ++mip) {
        const auto &level = layout.mips[mip];
        std::vector<uint8_t> linear(static_cast<size_t>(level.width) *
                                    level.height * elem);
        for (uint32_t layer = 0; layer < layout.layers; ++layer) {
          for (size_t i = 0; i < linear.size(); ++i)
            linear[i] =
                static_cast<uint8_t>((i * 157u + layer * 17u + mip) & 255u);
          ASSERT_TRUE(gpu::gcn::RetileTextureMip32(
              linear.data(), tiled.data(), layout, mip, layer));
        }
      }
      hash = HashBytes(tiled.data(), tiled.size(), hash);
    }
  }
  EXPECT_EQ(hash, 0xe0e2ac1035064882ull);
}

TEST(GcnDetile, PitchedTransfersLeaveLinearPaddingUntouched) {
  constexpr uint32_t kWidth = 263;
  constexpr uint32_t kHeight = 137;
  constexpr uint32_t kElem = 8;
  constexpr size_t kRowBytes = kWidth * kElem + 40;
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(
      layout, kWidth, kHeight, 271, 1, 1, 14, false, kElem));

  std::vector<uint8_t> source(kRowBytes * kHeight, 0xcd);
  std::vector<uint8_t> result(kRowBytes * kHeight, 0xee);
  std::vector<uint8_t> tiled(layout.size, 0xa5);
  for (uint32_t y = 0; y < kHeight; ++y)
    for (uint32_t x = 0; x < kWidth * kElem; ++x)
      source[static_cast<size_t>(y) * kRowBytes + x] =
          static_cast<uint8_t>((y * 37u + x * 13u) & 255u);

  ASSERT_TRUE(gpu::gcn::RetileTextureMip32Pitched(
      source.data(), kRowBytes, tiled.data(), layout, 0, 0));
  ASSERT_TRUE(gpu::gcn::DetileTextureMip32Pitched(
      tiled.data(), result.data(), kRowBytes, layout, 0, 0));
  for (uint32_t y = 0; y < kHeight; ++y) {
    const size_t row = static_cast<size_t>(y) * kRowBytes;
    EXPECT_TRUE(std::equal(source.begin() + row,
                           source.begin() + row + kWidth * kElem,
                           result.begin() + row));
    EXPECT_TRUE(std::all_of(result.begin() + row + kWidth * kElem,
                            result.begin() + row + kRowBytes,
                            [](uint8_t value) { return value == 0xee; }));
  }
}

TEST(GcnDetile, NestedParallelRegionsRunInline) {
  std::atomic<uint32_t> work{0};
  gpu::gcn::DetileParallelRows(32, [&](uint32_t outer0, uint32_t outer1) {
    gpu::gcn::DetileParallelRows(32, [&](uint32_t inner0, uint32_t inner1) {
      work.fetch_add((outer1 - outer0) * (inner1 - inner0),
                     std::memory_order_relaxed);
    });
  });
  EXPECT_EQ(work.load(std::memory_order_relaxed), 32u * 32u);
}

}  // namespace
