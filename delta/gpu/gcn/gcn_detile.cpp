/*
 * PS4Delta : GCN / Liverpool texture de-tiling (32bpp). See gcn_detile.h.
 *
 * Ported from shadPS4's tiling.comp / tiling.cpp (AMD AddrLib). Restricted to the
 * 32bpp, single-mip, single-slice, 1-sample case that Isaac's room/sprite
 * textures use. Two paths: 1D micro-tiled and 2D macro-tiled (the rest).
 */

#include "gcn_detile.h"

#include <cstring>

namespace gpu::gcn {
namespace {

enum MicroMode { MM_Display = 0, MM_Thin = 1, MM_Depth = 2, MM_Thick = 4 };

// 1D micro-tiled modes (everything tiled-but-not-macro): 1DThin1/1DThick.
inline bool isMicro1D(uint32_t idx) {
  return idx == 5 || idx == 9 || idx == 13 || idx == 19;
}

// Micro-tile mode (Display/Thin/Depth/Thick) for a given tilingIdx. Determines
// the in-tile pixel swizzle. Table per shadPS4 GetMicroTileMode().
inline MicroMode microModeOf(uint32_t idx) {
  if (idx <= 7) return MM_Depth;              // 0-7  depth
  if (idx >= 8 && idx <= 12) return MM_Display; // 8-12 display (8 linear handled earlier)
  if (idx >= 13 && idx <= 18) return MM_Thin;   // 13-18 thin
  return MM_Thick;                              // 19+   thick
}

// Pixel index within an 8x8 micro tile for 32bpp.
// Display: x0 x1 y0 x2 y1 y2   Thin/Depth: x0 y0 x1 y1 x2 y2 (Morton/Z)
inline uint32_t pixIdx32(uint32_t x, uint32_t y, MicroMode m) {
  uint32_t x0 = (x >> 0) & 1, x1 = (x >> 1) & 1, x2 = (x >> 2) & 1;
  uint32_t y0 = (y >> 0) & 1, y1 = (y >> 1) & 1, y2 = (y >> 2) & 1;
  if (m == MM_Display)
    return x0 | (x1 << 1) | (y0 << 2) | (x2 << 3) | (y1 << 4) | (y2 << 5);
  return x0 | (y0 << 1) | (x1 << 2) | (y1 << 3) | (x2 << 4) | (y2 << 5);
}

// --- Macro-tile params for the 32bpp / 1-sample / 2D-thin case (Liverpool). ---
// CalculateMacrotileMode on 32bpp thin: tile_bytes=256 -> mode index 2 ->
// num_banks=16, bank_width=1, bank_height=1, macro_aspect=2. num_pipes=8,
// pipe_config = P8_32x32_16x16. (Display/Thin both land here for colour art.)
constexpr uint32_t kNumPipes = 8;
constexpr uint32_t kNumBanks = 16;
constexpr uint32_t kBankWidth = 1;
constexpr uint32_t kBankHeight = 1;
constexpr uint32_t kMacroAspect = 2;
constexpr uint32_t kMicroTileBytes = 8 * 8 * 4;            // 256
constexpr uint32_t kMacroTilePitch = 8 * kBankWidth * kNumPipes * kMacroAspect;   // 128
constexpr uint32_t kMacroTileHeight = 8 * kBankHeight * kNumBanks / kMacroAspect; // 64

// ComputePipeFromCoord, P8_32x32_16x16.
inline uint32_t pipeFromCoord(uint32_t x, uint32_t y) {
  uint32_t x3 = (x >> 3) & 1, x4 = (x >> 4) & 1, x5 = (x >> 5) & 1;
  uint32_t y3 = (y >> 3) & 1, y4 = (y >> 4) & 1, y5 = (y >> 5) & 1;
  uint32_t p0 = x3 ^ y3 ^ x4;
  uint32_t p1 = x4 ^ y4;
  uint32_t p2 = x5 ^ y5;
  return p0 | (p1 << 1) | (p2 << 2);
}

// ComputeBankFromCoord, num_banks=16, bank_width=1, bank_height=1, num_pipes=8.
inline uint32_t bankFromCoord(uint32_t x, uint32_t y) {
  uint32_t tx = (x >> 3) / (kBankWidth * kNumPipes);  // (x/8)/8
  uint32_t ty = (y >> 3) / kBankHeight;               // (y/8)/1
  uint32_t x3 = (tx >> 0) & 1, x4 = (tx >> 1) & 1, x5 = (tx >> 2) & 1, x6 = (tx >> 3) & 1;
  uint32_t y3 = (ty >> 0) & 1, y4 = (ty >> 1) & 1, y5 = (ty >> 2) & 1, y6 = (ty >> 3) & 1;
  uint32_t b0 = x3 ^ y6;
  uint32_t b1 = x4 ^ y5 ^ y6;
  uint32_t b2 = x5 ^ y4;
  uint32_t b3 = x6 ^ y3;
  return (b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)) & (kNumBanks - 1);
}

// 1D micro-tiled byte offset (32bpp), slice/sample 0.
inline uint32_t addrMicro32(uint32_t x, uint32_t y, uint32_t pitch, MicroMode m) {
  uint32_t microTilesPerRow = pitch / 8;
  uint32_t microTileOffset = ((y >> 3) * microTilesPerRow + (x >> 3)) * kMicroTileBytes;
  return microTileOffset + pixIdx32(x, y, m) * 4;
}

// 2D macro-tiled byte offset (32bpp), slice/sample 0.
inline uint32_t addrMacro32(uint32_t x, uint32_t y, uint32_t pitch, MicroMode m) {
  uint32_t elementOffset = pixIdx32(x, y, m) * 4;  // < kMicroTileBytes, no tile-split at 32bpp

  uint32_t macroTileBytes = kMicroTileBytes * (kMacroTilePitch / 8) * (kMacroTileHeight / 8) /
                            (kNumPipes * kNumBanks);  // = 256
  uint32_t macroTilesPerRow = pitch / kMacroTilePitch;
  uint32_t mtx = x / kMacroTilePitch;
  uint32_t mty = y / kMacroTileHeight;
  uint32_t macroTileOffset = (mty * macroTilesPerRow + mtx) * macroTileBytes;

  // tile_row/tile_column are 0 for bank_width=bank_height=1.
  uint32_t totalOffset = macroTileOffset + elementOffset;

  uint32_t pipe = pipeFromCoord(x, y);
  uint32_t bank = bankFromCoord(x, y);
  const uint32_t pipeInterleaveBits = 8, numPipeBits = 3, numBankBits = 4;
  uint32_t interleaveOffset = totalOffset & ((1u << pipeInterleaveBits) - 1);
  uint32_t offset = totalOffset >> pipeInterleaveBits;

  uint32_t addr = interleaveOffset;
  addr |= pipe << pipeInterleaveBits;
  addr |= bank << (pipeInterleaveBits + numPipeBits);
  addr |= offset << (pipeInterleaveBits + numPipeBits + numBankBits);
  return addr;
}

}  // namespace

bool tilingIsLinear(uint32_t tilingIdx) {
  return tilingIdx == 8 || tilingIdx == 31;
}

bool detile32(const uint32_t *src, uint32_t *dst, uint32_t width, uint32_t height,
              uint32_t tilingIdx, uint32_t pitch) {
  if (tilingIsLinear(tilingIdx)) {
    if (pitch == width) {
      std::memcpy(dst, src, (size_t)width * height * 4);
    } else {
      for (uint32_t y = 0; y < height; y++)
        std::memcpy(dst + (size_t)y * width, src + (size_t)y * pitch, (size_t)width * 4);
    }
    return true;
  }

  const bool micro = isMicro1D(tilingIdx);
  const MicroMode mm = microModeOf(tilingIdx);
  // Align pitch to the addressing granularity so micro_tiles_per_row /
  // macro_tiles_per_row come out right even for odd widths.
  uint32_t apitch = pitch;
  if (micro) apitch = (apitch + 7) & ~7u;
  else apitch = (apitch + (kMacroTilePitch - 1)) & ~(kMacroTilePitch - 1);

  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
      uint32_t off = micro ? addrMicro32(x, y, apitch, mm)
                           : addrMacro32(x, y, apitch, mm);
      dst[(size_t)y * width + x] = src[off >> 2];
    }
  }
  return true;
}

}  // namespace gpu::gcn
