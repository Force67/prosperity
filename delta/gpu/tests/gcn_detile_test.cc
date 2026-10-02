#include <algorithm>
#include <cstring>
#include <thread>
#include "base/arch.h"

#include <gtest/gtest.h>
#include "base/memory/unique_pointer.h"
#include "base/threading/thread.h"

#include "base/algorithm.h"
#include "base/atomic.h"
#include "base/containers/vector.h"
#include "gpu/gcn/gcn_detile.h"

namespace {

struct SplitMode {
  u32 tiling_index;
  u32 tile_split_bytes;
  u32 bank_height;
  u32 macro_aspect;
};

u64 ExpectedTiledOffset(const SplitMode& mode,
                        u32 x,
                        u32 y,
                        u32 layer,
                        u32 pitch,
                        u32 height) {
  constexpr u32 kNumPipes = 8;
  constexpr u32 kNumBanks = 16;

  const u32 pixel_index = (x & 1) | ((y & 1) << 1) | (((x >> 1) & 1) << 2) |
                          (((y >> 1) & 1) << 3) | (((x >> 2) & 1) << 4) |
                          (((y >> 2) & 1) << 5);
  u32 element_offset = pixel_index * sizeof(u32);
  const u32 tile_split_slice = element_offset / mode.tile_split_bytes;
  element_offset %= mode.tile_split_bytes;

  const u32 slices_per_tile = 256 / mode.tile_split_bytes;
  const u32 macro_pitch = 8 * kNumPipes * mode.macro_aspect;
  const u32 macro_height = 8 * mode.bank_height * kNumBanks / mode.macro_aspect;
  const u32 macro_tile_bytes = mode.tile_split_bytes * (macro_pitch / 8) *
                               (macro_height / 8) / (kNumPipes * kNumBanks);
  const u32 macro_tiles_per_row = pitch / macro_pitch;
  const u64 macro_tile_offset =
      static_cast<u64>((y / macro_height) * macro_tiles_per_row +
                       x / macro_pitch) *
      macro_tile_bytes;
  const u32 macro_tiles_per_slice =
      macro_tiles_per_row * (height / macro_height);
  const u64 slice_bytes =
      static_cast<u64>(macro_tiles_per_slice) * macro_tile_bytes;
  const u64 slice_offset =
      slice_bytes * (tile_split_slice + slices_per_tile * layer);
  const u32 tile_row = (y / 8) % mode.bank_height;
  const u32 tile_offset = tile_row * mode.tile_split_bytes;
  const u64 total_offset =
      slice_offset + macro_tile_offset + tile_offset + element_offset;

  const u32 tile_x = x >> 3;
  const u32 tile_y = y >> 3;
  const u32 pipe = ((tile_x & 1) ^ (tile_y & 1) ^ ((tile_x >> 1) & 1)) |
                   ((((tile_x >> 1) & 1) ^ ((tile_y >> 1) & 1)) << 1) |
                   ((((tile_x >> 2) & 1) ^ ((tile_y >> 2) & 1)) << 2);

  const u32 bank_x = tile_x / kNumPipes;
  const u32 bank_y = tile_y / mode.bank_height;
  u32 bank =
      ((bank_x & 1) ^ ((bank_y >> 3) & 1)) |
      ((((bank_x >> 1) & 1) ^ ((bank_y >> 2) & 1) ^ ((bank_y >> 3) & 1)) << 1) |
      ((((bank_x >> 2) & 1) ^ ((bank_y >> 1) & 1)) << 2) |
      ((((bank_x >> 3) & 1) ^ (bank_y & 1)) << 3);
  bank ^= 7 * layer;
  bank ^= 9 * tile_split_slice;
  bank &= kNumBanks - 1;

  return (total_offset & 255) | (static_cast<u64>(pipe) << 8) |
         (static_cast<u64>(bank) << 11) | ((total_offset >> 8) << 15);
}

void VerifySplitMode(const SplitMode& mode) {
  constexpr u32 kLayers = 2;
  const u32 macro_pitch = 8 * 8 * mode.macro_aspect;
  const u32 macro_height = 8 * mode.bank_height * 16 / mode.macro_aspect;
  const u32 width = macro_pitch * 2;
  const u32 height = macro_height * 2;

  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(
      layout, width, height, width, kLayers, 1, mode.tiling_index, false));
  ASSERT_TRUE(layout.mips[0].macro_tiled);
  ASSERT_EQ(layout.mips[0].pitch, width);
  ASSERT_EQ(layout.mips[0].stored_height, height);

  base::Vector<u32> tiled(layout.size / sizeof(u32));
  base::Vector<bool> occupied(tiled.size());
  for (u32 layer = 0; layer < kLayers; ++layer) {
    for (u32 y = 0; y < height; ++y) {
      for (u32 x = 0; x < width; ++x) {
        const u64 offset =
            ExpectedTiledOffset(mode, x, y, layer, width, height);
        ASSERT_EQ(offset % sizeof(u32), 0u);
        ASSERT_LE(offset + sizeof(u32), layout.size);
        const size_t index = offset / sizeof(u32);
        ASSERT_FALSE(occupied[index]);
        occupied[index] = true;
        tiled[index] = 1 + (layer * height + y) * width + x;
      }
    }
  }
  ASSERT_EQ(base::Count(occupied.begin(), occupied.end(), true),
            occupied.size());

  base::Vector<u32> linear(static_cast<size_t>(width) * height);
  base::Vector<u32> expected(linear.size());
  base::Vector<u32> retiled(tiled.size());
  for (u32 layer = 0; layer < kLayers; ++layer) {
    ASSERT_TRUE(gpu::gcn::DetileTextureMip32(tiled.data(), linear.data(),
                                             layout, 0, layer));
    for (u32 y = 0; y < height; ++y) {
      for (u32 x = 0; x < width; ++x) {
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

void Verify16BitRoundTrip(u32 tiling_index) {
  constexpr u32 kWidth = 256;
  constexpr u32 kHeight = 128;
  constexpr u32 kLayers = 2;
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(
      layout, kWidth, kHeight, kWidth, kLayers, 1, tiling_index, false, 2));

  base::Vector<u8> tiled(layout.size);
  base::Vector<u16> source(static_cast<size_t>(kWidth) * kHeight);
  base::Vector<u16> result(source.size());
  for (u32 layer = 0; layer < kLayers; layer++) {
    for (size_t i = 0; i < source.size(); i++)
      source[i] = static_cast<u16>(1 + i + layer * source.size());
    ASSERT_TRUE(gpu::gcn::RetileTextureMip32(source.data(), tiled.data(),
                                             layout, 0, layer));
  }
  for (u32 layer = 0; layer < kLayers; layer++) {
    for (size_t i = 0; i < source.size(); i++)
      source[i] = static_cast<u16>(1 + i + layer * source.size());
    ASSERT_TRUE(gpu::gcn::DetileTextureMip32(tiled.data(), result.data(),
                                             layout, 0, layer));
    EXPECT_EQ(result, source);
  }
}

u64 HashBytes(const u8* data, size_t size, u64 hash) {
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
  constexpr u32 kWidth = 256;
  constexpr u32 kHeight = 128;
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(layout, kWidth, kHeight, kWidth, 1,
                                             1, 14, false, 8));
  base::Vector<u64> source(static_cast<size_t>(kWidth) * kHeight);
  for (size_t i = 0; i < source.size(); i++)
    source[i] = i + 1;
  base::Vector<u8> tiled(layout.size);
  base::Vector<u64> result(source.size());
  ASSERT_TRUE(
      gpu::gcn::RetileTextureMip32(source.data(), tiled.data(), layout, 0, 0));
  ASSERT_TRUE(
      gpu::gcn::DetileTextureMip32(tiled.data(), result.data(), layout, 0, 0));
  EXPECT_EQ(result, source);
}

TEST(GcnDetile, AllModesAndElementWidthsMatchReferenceDigest) {
  u64 hash = 1469598103934665603ull;
  for (u32 tiling = 0; tiling <= 31; ++tiling) {
    if (tiling > 26 && tiling != 31)
      continue;
    for (u32 elem : {2u, 4u, 8u, 16u}) {
      gpu::gcn::TextureLayout32 layout;
      ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(layout, 263, 137, 271, 3, 4,
                                                 tiling, false, elem));
      base::Vector<u8> tiled(layout.size);
      for (size_t i = 0; i < tiled.size(); ++i)
        tiled[i] = static_cast<u8>((i * 193u + i / 29u + tiling) & 255u);

      for (u32 mip = 0; mip < layout.mip_levels; ++mip) {
        const auto& level = layout.mips[mip];
        base::Vector<u8> linear(static_cast<size_t>(level.width) *
                                level.height * elem);
        for (u32 layer = 0; layer < layout.layers; ++layer) {
          ASSERT_TRUE(gpu::gcn::DetileTextureMip32(tiled.data(), linear.data(),
                                                   layout, mip, layer));
          hash = HashBytes(linear.data(), linear.size(), hash);
        }
      }

      base::Fill(tiled.begin(), tiled.end(), 0xa5);
      for (u32 mip = 0; mip < layout.mip_levels; ++mip) {
        const auto& level = layout.mips[mip];
        base::Vector<u8> linear(static_cast<size_t>(level.width) *
                                level.height * elem);
        for (u32 layer = 0; layer < layout.layers; ++layer) {
          for (size_t i = 0; i < linear.size(); ++i)
            linear[i] = static_cast<u8>((i * 157u + layer * 17u + mip) & 255u);
          ASSERT_TRUE(gpu::gcn::RetileTextureMip32(linear.data(), tiled.data(),
                                                   layout, mip, layer));
        }
      }
      hash = HashBytes(tiled.data(), tiled.size(), hash);
    }
  }
  EXPECT_EQ(hash, 0xe0e2ac1035064882ull);
}

// Every address the separable terms produce agrees with the detiler, for each
// layout they are offered for; and the macro modes compute targets use get
// them.
TEST(GcnDetile, SeparableTermsMatchTheDetiler) {
  struct Shape {
    u32 width, height, pitch, layers, mips;
  };
  u32 macro_ok = 0;
  for (const Shape& shape :
       {Shape{1680, 948, 1680, 1, 1}, Shape{263, 137, 271, 3, 4}}) {
    for (u32 tiling = 0; tiling <= 26; ++tiling) {
      for (u32 elem : {4u, 8u, 16u}) {
        gpu::gcn::TextureLayout32 layout;
        if (gpu::gcn::TilingIsLinear(tiling) ||
            !gpu::gcn::BuildTextureLayout32(layout, shape.width, shape.height,
                                            shape.pitch, shape.layers,
                                            shape.mips, tiling, false, elem))
          continue;
        base::Vector<u8> tiled(layout.size);
        for (size_t i = 0; i < tiled.size(); ++i)
          tiled[i] = static_cast<u8>((i * 193u + i / 29u) & 255u);
        for (u32 mip = 0; mip < layout.mip_levels; ++mip) {
          base::Vector<u32> terms;
          u32 mask;
          u64 stride;
          if (!gpu::gcn::BuildSeparableAddressTable(layout, mip, terms, mask,
                                                    stride))
            continue;
          if (mip == 0 && layout.mips[0].macro_tiled)
            macro_ok++;
          const auto& level = layout.mips[mip];
          base::Vector<u8> linear(size_t(level.width) * level.height * elem);
          for (u32 layer = 0; layer < layout.layers; ++layer) {
            ASSERT_TRUE(gpu::gcn::DetileTextureMip32(
                tiled.data(), linear.data(), layout, mip, layer));
            const u32 zt = terms[level.width + level.height + layer];
            for (u32 y = 0; y < level.height; ++y)
              for (u32 x = 0; x < level.width; ++x) {
                const u32 xt = terms[x], yt = terms[level.width + y];
                const u64 at = level.offset + stride * layer + (xt & ~mask) +
                               (yt & ~mask) + ((xt ^ yt ^ zt) & mask);
                ASSERT_EQ(
                    std::memcmp(&tiled[at],
                                &linear[(size_t(y) * level.width + x) * elem],
                                elem),
                    0)
                    << "tiling " << tiling << " elem " << elem << " mip " << mip
                    << " layer " << layer << " at " << x << "," << y;
              }
          }
        }
      }
    }
  }
  EXPECT_GT(macro_ok, 0u);
}

TEST(GcnDetile, Gfx10SingleMipStartsAtTheBlockOrigin) {
  for (u32 mode : {5u, 9u, 25u, 27u}) {
    SCOPED_TRACE(mode);
    gpu::gcn::TextureLayout32 layout;
    ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(
        layout, 16, 16, 16, 1, 1, gpu::gcn::kGfx10TilingBase + mode, false, 4));
    EXPECT_EQ(layout.mips[0].mip_tail_x, 0u);
    EXPECT_EQ(layout.mips[0].mip_tail_y, 0u);
    base::Vector<u8> tiled(layout.size, 0);
    tiled[0] = 42;
    base::Vector<u8> linear(16 * 16 * 4, 0);
    ASSERT_TRUE(gpu::gcn::DetileTextureMip32(
        tiled.data(), linear.data(), layout, 0, 0));
    EXPECT_EQ(linear[0], 42);
  }
}

TEST(GcnDetile, PitchedTransfersLeaveLinearPaddingUntouched) {
  constexpr u32 kWidth = 263;
  constexpr u32 kHeight = 137;
  constexpr u32 kElem = 8;
  constexpr size_t kRowBytes = kWidth * kElem + 40;
  gpu::gcn::TextureLayout32 layout;
  ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(layout, kWidth, kHeight, 271, 1, 1,
                                             14, false, kElem));

  base::Vector<u8> source(kRowBytes * kHeight, 0xcd);
  base::Vector<u8> result(kRowBytes * kHeight, 0xee);
  base::Vector<u8> tiled(layout.size, 0xa5);
  for (u32 y = 0; y < kHeight; ++y)
    for (u32 x = 0; x < kWidth * kElem; ++x)
      source[static_cast<size_t>(y) * kRowBytes + x] =
          static_cast<u8>((y * 37u + x * 13u) & 255u);

  ASSERT_TRUE(gpu::gcn::RetileTextureMip32Pitched(source.data(), kRowBytes,
                                                  tiled.data(), layout, 0, 0));
  ASSERT_TRUE(gpu::gcn::DetileTextureMip32Pitched(tiled.data(), result.data(),
                                                  kRowBytes, layout, 0, 0));
  for (u32 y = 0; y < kHeight; ++y) {
    const size_t row = static_cast<size_t>(y) * kRowBytes;
    EXPECT_EQ(
        std::memcmp(source.data() + row, result.data() + row, kWidth * kElem),
        0);
    EXPECT_TRUE(base::AllOf(result.begin() + row + kWidth * kElem,
                            result.begin() + row + kRowBytes,
                            [](u8 value) { return value == 0xee; }));
  }
}

TEST(GcnDetile, ByteWritebackRoundTripsAcrossMipsLayersAndPaddedRows) {
  // Include Astro's 64KB_R_X byte-image layout (0x11b), as well as linear
  // and GCN tiled layouts. Each layer is written before any are read back.
  for (u32 tiling : {8u, 14u, 31u, 0x11bu}) {
    SCOPED_TRACE(tiling);
    gpu::gcn::TextureLayout32 layout;
    ASSERT_TRUE(gpu::gcn::BuildTextureLayout32(layout, 263, 137, 271, 2, 4,
                                               tiling, false, 1));
    base::Vector<u8> tiled(layout.size, 0xa5);
    for (u32 mip = 0; mip < layout.mip_levels; ++mip) {
      const auto& level = layout.mips[mip];
      const size_t row_bytes = level.width + 13;
      for (u32 layer = 0; layer < layout.layers; ++layer) {
        base::Vector<u8> source(row_bytes * level.height, 0xcd);
        for (u32 y = 0; y < level.height; ++y)
          for (u32 x = 0; x < level.width; ++x)
            source[y * row_bytes + x] =
                static_cast<u8>(x * 13 + y * 37 + layer * 53 + mip * 71);
        ASSERT_TRUE(gpu::gcn::RetileTextureMip32Pitched(
            source.data(), row_bytes, tiled.data(), layout, mip, layer));
      }
    }
    for (u32 mip = 0; mip < layout.mip_levels; ++mip) {
      const auto& level = layout.mips[mip];
      const size_t row_bytes = level.width + 13;
      for (u32 layer = 0; layer < layout.layers; ++layer) {
        base::Vector<u8> result(row_bytes * level.height, 0xee);
        base::Vector<u8> expected(result);
        for (u32 y = 0; y < level.height; ++y)
          for (u32 x = 0; x < level.width; ++x)
            expected[y * row_bytes + x] =
                static_cast<u8>(x * 13 + y * 37 + layer * 53 + mip * 71);
        ASSERT_TRUE(gpu::gcn::DetileTextureMip32Pitched(
            tiled.data(), result.data(), row_bytes, layout, mip, layer));
        EXPECT_EQ(result, expected) << "mip " << mip << " layer " << layer;
      }
    }
  }
}

TEST(GcnDetile, NestedParallelRegionsRunInline) {
  base::Atomic<u32> work{0};
  gpu::gcn::DetileParallelRows(32, [&](u32 outer0, u32 outer1) {
    gpu::gcn::DetileParallelRows(32, [&](u32 inner0, u32 inner1) {
      work.fetch_add((outer1 - outer0) * (inner1 - inner0),
                     base::memory_order_relaxed);
    });
  });
  EXPECT_EQ(work.load(base::memory_order_relaxed), 32u * 32u);
}

TEST(GcnDetile, ConcurrentRepeatedRegionsFinishEveryRowBeforeReturning) {
  base::Atomic<u32> failures{0};
  base::Vector<base::UniquePointer<base::Thread>> callers;
  for (u32 caller = 0; caller < 4; caller++) {
    callers.push_back(base::MakeUnique<base::Thread>(
        "caller",
        [&] {
          for (u32 iteration = 0; iteration < 500; iteration++) {
            base::Vector<base::Atomic<u32>> rows(128);
            gpu::gcn::DetileParallelRows(rows.size(), [&](u32 first, u32 last) {
              for (u32 row = first; row < last; row++)
                rows[row].fetch_add(1, base::memory_order_relaxed);
            });
            for (const auto& row : rows)
              if (row.load(base::memory_order_relaxed) != 1)
                failures.fetch_add(1, base::memory_order_relaxed);
          }
        },
        true));
  }
  for (auto& caller : callers)
    caller->Join();
  EXPECT_EQ(failures.load(), 0u);
}

}  // namespace

// gfx10 linear surfaces take the plain row terms, slices and mips included.
TEST(GcnDetile, SeparableTermsMatchTheDetilerForGfx10Linear) {
  const u32 tiling = gpu::gcn::kGfx10TilingBase;
  ASSERT_TRUE(gpu::gcn::TilingIsLinear(tiling));
  struct Shape {
    u32 width, height, pitch, layers, mips;
  };
  u32 checked = 0;
  for (const Shape& shape : {Shape{240, 135, 240, 64, 1},
                             Shape{263, 137, 271, 3, 4},
                             Shape{960, 540, 960, 1, 1}}) {
    for (u32 elem : {4u, 8u, 16u}) {
      gpu::gcn::TextureLayout32 layout;
      if (!gpu::gcn::BuildTextureLayout32(layout, shape.width, shape.height,
                                          shape.pitch, shape.layers,
                                          shape.mips, tiling, false, elem))
        continue;
      base::Vector<u8> tiled(layout.size);
      for (size_t i = 0; i < tiled.size(); ++i)
        tiled[i] = static_cast<u8>((i * 193u + i / 29u) & 255u);
      for (u32 mip = 0; mip < layout.mip_levels; ++mip) {
        base::Vector<u32> terms;
        u32 mask;
        u64 stride;
        ASSERT_TRUE(gpu::gcn::BuildSeparableAddressTable(layout, mip, terms,
                                                         mask, stride));
        const auto& level = layout.mips[mip];
        base::Vector<u8> linear(size_t(level.width) * level.height * elem);
        for (u32 layer = 0; layer < layout.layers; ++layer) {
          ASSERT_TRUE(gpu::gcn::DetileTextureMip32(
              tiled.data(), linear.data(), layout, mip, layer));
          const u32 zt = terms[level.width + level.height + layer];
          for (u32 y = 0; y < level.height; ++y)
            for (u32 x = 0; x < level.width; ++x) {
              const u32 xt = terms[x], yt = terms[level.width + y];
              const u64 at = level.offset + stride * layer + (xt & ~mask) +
                             (yt & ~mask) + ((xt ^ yt ^ zt) & mask);
              ASSERT_EQ(
                  std::memcmp(&tiled[at],
                              &linear[(size_t(y) * level.width + x) * elem],
                              elem),
                  0)
                  << "elem " << elem << " mip " << mip << " layer " << layer
                  << " at " << x << "," << y;
            }
          checked++;
        }
      }
    }
  }
  EXPECT_GT(checked, 0u);
}
