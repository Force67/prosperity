/*
 * PS4Delta : GCN / Liverpool texture de-tiling (32bpp). See gcn_detile.h.
 *
 * Faithful port of the AMD AddrLib (Liverpool) tiling/de-tiling swizzle,
 * restricted to the 32bpp / 1-sample / single-mip case that the
 * 2D sprite/RT-as-texture path needs.
 *
 * Unlike the previous version, every addressing parameter (array mode, micro
 * tile mode, pipe config, num_pipes/banks, bank_width/height, macro aspect,
 * tile split) is DERIVED from the surface's tiling_index via the standard
 * Liverpool GB_TILE_MODE + macro-tile-mode tables, rather than hardcoding one
 * config. That keeps Isaac's surfaces (Display2DThin / 1D micro, P8_32x32_16x16,
 * 16 banks) byte-identical to before while also covering the other tiled modes
 * a title like Doom64 can ask for (e.g. Thin* modes that use P8_32x32_8x16).
 *
 * Tables follow the AMD AddrLib GB_TILE_MODE register decode for the Liverpool
 * (base-PS4) GPU, as documented in the AMD GCN/Sea Islands ISA + register specs.
 * Console reported as base PS4 (non-Neo) so num_pipes is 8, the standard value
 * for these modes.
 */

#include "gcn_detile.h"

#include <algorithm>
#include <cstring>

namespace gpu::gcn {
namespace {

// Linux defines the CIK array/tiling enums and GB_TILE_MODE fields, but not the
// Liverpool table contents or the byte-address equations implemented below:
// https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_enum.h
// https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_sh_mask.h
// https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdgpu/gfx_v7_0.c

// --- GB_TILE_MODE decode (per tiling_index 0..31), Liverpool / base PS4. ----
// The AddrLib GB_TILE_MODE decode (array mode / micro tile mode / pipe config /
// sample split / tile split per index). We only need a compact form of each.

enum ArrayMode {
  AM_LinearGeneral = 0,
  AM_LinearAligned = 1,
  AM_1DThin1 = 2,
  AM_1DThick = 3,
  AM_2DThin1 = 4,
  AM_PrtThin1 = 5,
  AM_Prt2DThin1 = 6,
  AM_2DThick = 7,
  AM_2DXThick = 8,
  AM_PrtThick = 9,
  AM_Prt2DThick = 10,
  AM_Prt3DThin1 = 11,
  AM_3DThin1 = 12,
  AM_3DThick = 13,
  AM_3DXThick = 14,
  AM_Prt3DThick = 15,
};

enum MicroMode { MM_Display = 0, MM_Thin = 1, MM_Depth = 2, MM_Rotated = 3, MM_Thick = 4 };

// Two 8-pipe pipe-config equations are used by the colour/depth modes on
// Liverpool: P8_32x32_16x16 and P8_32x32_8x16. (P2 only for LinearGeneral.)
enum PipeConfig { PC_P2 = 0, PC_P8_32x32_8x16 = 10, PC_P8_32x32_16x16 = 12 };

ArrayMode arrayModeOf(uint32_t idx) {
  switch (idx) {
    case 5:  // Depth1DThin
    case 9:  // Display1DThin
    case 13: // Thin1DThin
      return AM_1DThin1;
    case 0: case 1: case 2: case 3: case 4: // Depth2DThin*
    case 10: // Display2DThin
    case 14: // Thin2DThin
      return AM_2DThin1;
    case 11: // DisplayThinPrt
    case 16: // ThinThinPrt
      return AM_PrtThin1;
    case 6:  // Depth2DThinPrt256
    case 7:  // Depth2DThinPrt1K
    case 12: // Display2DThinPrt
    case 17: // Thin2DThinPrt
      return AM_Prt2DThin1;
    case 15: // Thin3DThin
    case 18: // Thin3DThinPrt
      return AM_3DThin1;
    case 19: return AM_1DThick;   // Thick1DThick
    case 20: return AM_2DThick;   // Thick2DThick
    case 21: return AM_3DThick;   // Thick3DThick
    case 22: return AM_PrtThick;  // ThickThickPrt
    case 23: return AM_Prt2DThick;// Thick2DThickPrt
    case 24: return AM_Prt3DThick;// Thick3DThickPrt
    case 25: return AM_2DXThick;  // Thick2DXThick
    case 26: return AM_3DXThick;  // Thick3DXThick
    case 8:  return AM_LinearAligned;  // DisplayLinearAligned
    case 31: return AM_LinearGeneral;  // DisplayLinearGeneral
    default: return AM_2DThin1;         // unknown -> safest 2D thin
  }
}

MicroMode microModeOf(uint32_t idx) {
  switch (idx) {
    case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7:
      return MM_Depth;
    case 8: case 9: case 10: case 11: case 12: case 31:
      return MM_Display;
    case 13: case 14: case 15: case 16: case 17: case 18:
      return MM_Thin;
    default: // 19..26
      return MM_Thick;
  }
}

PipeConfig pipeConfigOf(uint32_t idx) {
  switch (idx) {
    // P8_32x32_8x16: DisplayThinPrt(11), Thin3DThin(15), ThinThinPrt(16),
    // Thick3DThick(21), ThickThickPrt(22), Thick3DThickPrt(24), Thick3DXThick(26)
    case 11: case 15: case 16: case 21: case 22: case 24: case 26:
      return PC_P8_32x32_8x16;
    case 31:
      return PC_P2;
    default:
      return PC_P8_32x32_16x16;
  }
}

// SAMPLE_SPLIT (sample-split count) per GB_TILE_MODE; 2 for Display2DThin and the
// 2D/3D thin colour modes, 1 elsewhere. Affects the 32bpp tile_split (-> 512).
uint32_t sampleSplitOf(uint32_t idx) {
  switch (idx) {
    case 10: case 11: case 12: // Display2DThin, DisplayThinPrt, Display2DThinPrt
    case 14: case 15: case 16: case 17: case 18: // Thin 2D/3D/Prt
      return 2;
    default:
      return 1;
  }
}

// TILE_SPLIT (hw field) per GB_TILE_MODE; only the depth modes use the large
// values. Colour modes use 64 (then color_tile_split overrides for 32bpp).
uint32_t tileSplitHwOf(uint32_t idx) {
  switch (idx) {
    case 1: return 128;             // Depth2DThin128
    case 2: case 6: return 256;     // Depth2DThin256 / Prt256
    case 3: return 512;             // Depth2DThin512
    case 4: case 7: return 1024;    // Depth2DThin1K / Prt1K
    default: return 64;
  }
}

// --- Macro-tile-mode params (AddrLib GetNumBanks/BankWidth/BankHeight/...). ---
// The macro tile mode index for 32bpp/1-sample is derived from tile_split, then
// (per AMD AddrLib) yields num_banks/bank_width/bank_height/macro_aspect.
struct MacroParams {
  uint32_t numBanks, bankWidth, bankHeight, macroAspect;
};

// AddrLib macro-tile-mode LUTs, indexed by MacroTileMode (0..15). For the
// non-Neo (base PS4) primary config we use the non-alt tables.
MacroParams macroParamsForMode(uint32_t m) {
  // {numBanks, bankWidth, bankHeight, macroAspect}
  static const MacroParams tbl[16] = {
      /*0  Mode_1x4_16     */ {16, 1, 4, 4},
      /*1  Mode_1x2_16     */ {16, 1, 2, 2},
      /*2  Mode_1x1_16     */ {16, 1, 1, 2},
      /*3  Mode_1x1_16_Dup */ {16, 1, 1, 2},
      /*4  Mode_1x1_8      */ {8,  1, 1, 1},
      /*5  Mode_1x1_4      */ {4,  1, 1, 1},
      /*6  Mode_1x1_2      */ {2,  1, 1, 1},
      /*7  Mode_1x1_2_Dup  */ {2,  1, 1, 1},
      /*8  Mode_1x8_16     */ {16, 1, 8, 4},
      /*9  Mode_1x4_16_Dup */ {16, 1, 4, 4},
      /*10 Mode_1x2_16_Dup */ {16, 1, 2, 2},
      /*11 Mode_1x1_16_Dup2*/ {16, 1, 1, 2},
      /*12 Mode_1x1_8_Dup  */ {8,  1, 1, 1},
      /*13 Mode_1x1_4_Dup  */ {4,  1, 1, 1},
      /*14 Mode_1x1_2_Dup2 */ {2,  1, 1, 1},
      /*15 Mode_1x1_2_Dup3 */ {2,  1, 1, 1},
  };
  return tbl[m & 15];
}

constexpr uint32_t kBpp = 32;
constexpr uint32_t kMicroW = 8, kMicroH = 8;
constexpr uint32_t kMicroTilePixels = kMicroW * kMicroH;            // 64
constexpr uint32_t kMicroTileBytes = kMicroTilePixels * kBpp / 8;   // 256
constexpr uint32_t kPipeInterleaveBits = 8;

uint32_t numPipesOf(PipeConfig pc) { return pc == PC_P2 ? 2u : 8u; }
uint32_t numPipeBitsOf(PipeConfig pc) { return pc == PC_P2 ? 1u : 3u; }

bool isPrt(ArrayMode am) {
  return am == AM_PrtThin1 || am == AM_PrtThick || am == AM_Prt2DThin1 ||
         am == AM_Prt2DThick || am == AM_Prt3DThin1 || am == AM_Prt3DThick;
}

// tile_split for 32bpp / 1-sample (CalculateTileSplit + min vs num_samples*bytes).
uint32_t tileSplitBytes(uint32_t idx, MicroMode mm) {
  const uint32_t tileBytes1x = kMicroTileBytes;  // 32bpp, thickness 1
  const uint32_t colorSplit = sampleSplitOf(idx) * tileBytes1x;
  const uint32_t cts = colorSplit < 256u ? 256u : colorSplit;
  uint32_t split = (mm == MM_Depth) ? tileSplitHwOf(idx) : cts;
  if (split > 1024u) split = 1024u;
  // min with num_samples(1) * tile_bytes_1x:
  if (split > tileBytes1x) split = tileBytes1x;  // 32bpp 1-sample => 256
  return split;
}

// CalculateMacrotileMode: mtm = log2(tile_split/64); +8 if PRT.
uint32_t macroTileModeIndex(uint32_t idx, MicroMode mm, ArrayMode am) {
  uint32_t split = tileSplitBytes(idx, mm);
  uint32_t q = split / 64u;
  uint32_t mtm = 0;
  while ((1u << (mtm + 1)) <= q) mtm++;  // bit_width(q)-1
  if (isPrt(am)) mtm += 8;
  return mtm;
}

// Pixel index within an 8x8 micro tile for 32bpp (matches tiling.comp).
// Display: x0 x1 y0 x2 y1 y2 ; Thin/Depth: x0 y0 x1 y1 x2 y2 (Morton/Z).
inline uint32_t pixIdx32(uint32_t x, uint32_t y, MicroMode m) {
  uint32_t x0 = x & 1, x1 = (x >> 1) & 1, x2 = (x >> 2) & 1;
  uint32_t y0 = y & 1, y1 = (y >> 1) & 1, y2 = (y >> 2) & 1;
  if (m == MM_Display)
    return x0 | (x1 << 1) | (y0 << 2) | (x2 << 3) | (y1 << 4) | (y2 << 5);
  // Thin / Depth / Thick all use the Morton order for the in-plane bits.
  return x0 | (y0 << 1) | (x1 << 2) | (y1 << 3) | (x2 << 4) | (y2 << 5);
}

// ComputePipeFromCoord (tiling.comp), 8-pipe configs.
inline uint32_t pipeFromCoord(uint32_t x, uint32_t y, uint32_t slice,
                              PipeConfig pc, ArrayMode am) {
  uint32_t tx = x >> 3, ty = y >> 3;
  uint32_t x3 = tx & 1, x4 = (tx >> 1) & 1, x5 = (tx >> 2) & 1;
  uint32_t y3 = ty & 1, y4 = (ty >> 1) & 1, y5 = (ty >> 2) & 1;
  uint32_t pipe;
  if (pc == PC_P2) pipe = x3 ^ y3;
  else if (pc == PC_P8_32x32_8x16) {
    uint32_t p0 = x4 ^ y3 ^ x5;
    uint32_t p1 = x3 ^ y4;
    uint32_t p2 = x5 ^ y5;
    pipe = p0 | (p1 << 1) | (p2 << 2);
  } else {
    // P8_32x32_16x16
    uint32_t p0 = x3 ^ y3 ^ x4;
    uint32_t p1 = x4 ^ y4;
    uint32_t p2 = x5 ^ y5;
    pipe = p0 | (p1 << 1) | (p2 << 2);
  }
  if (am == AM_3DThin1 || am == AM_3DThick || am == AM_3DXThick) {
    uint32_t rotation = std::max(1u, numPipesOf(pc) / 2 - 1) * slice;
    pipe ^= rotation & (numPipesOf(pc) - 1);
  }
  return pipe;
}

// ComputeBankFromCoord (tiling.comp), parameterized by num_banks/widths.
inline uint32_t bankFromCoord(uint32_t x, uint32_t y, uint32_t slice,
                              const MacroParams &mp, uint32_t numPipes,
                              ArrayMode am) {
  uint32_t tx = (x >> 3) / (mp.bankWidth * numPipes);
  uint32_t ty = (y >> 3) / mp.bankHeight;
  uint32_t x3 = tx & 1, x4 = (tx >> 1) & 1, x5 = (tx >> 2) & 1, x6 = (tx >> 3) & 1;
  uint32_t y3 = ty & 1, y4 = (ty >> 1) & 1, y5 = (ty >> 2) & 1, y6 = (ty >> 3) & 1;
  uint32_t bank = 0;
  switch (mp.numBanks) {
    case 16:
      bank = (x3 ^ y6) | ((x4 ^ y5 ^ y6) << 1) | ((x5 ^ y4) << 2) | ((x6 ^ y3) << 3);
      break;
    case 8:
      bank = (x3 ^ y5) | ((x4 ^ y4 ^ y5) << 1) | ((x5 ^ y3) << 2);
      break;
    case 4:
      bank = (x3 ^ y4) | ((x4 ^ y3) << 1);
      break;
    case 2:
      bank = (x3 ^ y3);
      break;
    default:
      bank = (x3 ^ y6) | ((x4 ^ y5 ^ y6) << 1) | ((x5 ^ y4) << 2) | ((x6 ^ y3) << 3);
      break;
  }
  uint32_t rotation = 0;
  if (am == AM_2DThin1 || am == AM_2DThick || am == AM_2DXThick)
    rotation = (mp.numBanks / 2 - 1) * slice;
  else if (am == AM_3DThin1 || am == AM_3DThick || am == AM_3DXThick)
    rotation = std::max(1u, numPipes / 2 - 1) * slice / numPipes;
  return (bank ^ rotation) & (mp.numBanks - 1);
}

// 1D micro-tiled byte offset (32bpp), slice/sample 0.
inline uint32_t addrMicro32(uint32_t x, uint32_t y, uint32_t pitch, MicroMode m) {
  uint32_t microTilesPerRow = pitch / kMicroW;
  uint32_t microTileOffset = ((y >> 3) * microTilesPerRow + (x >> 3)) * kMicroTileBytes;
  return microTileOffset + pixIdx32(x, y, m) * 4;
}

struct Macro2D {
  ArrayMode am;
  MicroMode mm;
  PipeConfig pc;
  MacroParams mp;
  uint32_t numPipes, numPipeBits, numBankBits;
  uint32_t macroPitch, macroHeight, macroTileBytes;
};

inline uint32_t numBankBitsOf(uint32_t numBanks) {
  uint32_t b = 0;
  while ((1u << (b + 1)) <= numBanks) b++;
  return b;  // bit_width(numBanks)-1
}

// 2D macro-tiled byte offset (32bpp), single slice/sample (tile_split inert at
// 32bpp 1-sample, slice/bank rotation zero for single slice).
inline uint32_t addrMacro32(uint32_t x, uint32_t y, uint32_t slice,
                            uint32_t pitch, const Macro2D &c) {
  uint32_t elementOffset = pixIdx32(x, y, c.mm) * 4;  // < kMicroTileBytes

  uint32_t macroTilesPerRow = pitch / c.macroPitch;
  uint32_t mtx = x / c.macroPitch;
  uint32_t mty = y / c.macroHeight;
  uint32_t macroTileOffset = (mty * macroTilesPerRow + mtx) * c.macroTileBytes;

  // tile_row/tile_column rotation within the macro tile (0 when bw==bh==1).
  uint32_t tileRow = (y >> 3) % c.mp.bankHeight;
  uint32_t tileCol = ((x >> 3) / c.numPipes) % c.mp.bankWidth;
  uint32_t tileOffset = (tileRow * c.mp.bankWidth + tileCol) * kMicroTileBytes;

  uint32_t totalOffset = macroTileOffset + elementOffset + tileOffset;

  uint32_t pipe = pipeFromCoord(x, y, slice, c.pc, c.am);
  uint32_t bank = bankFromCoord(x, y, slice, c.mp, c.numPipes, c.am);

  uint32_t interleaveOffset = totalOffset & ((1u << kPipeInterleaveBits) - 1);
  uint32_t offset = totalOffset >> kPipeInterleaveBits;

  uint32_t addr = interleaveOffset;
  addr |= pipe << kPipeInterleaveBits;
  addr |= bank << (kPipeInterleaveBits + c.numPipeBits);
  addr |= offset << (kPipeInterleaveBits + c.numPipeBits + c.numBankBits);
  return addr;
}

bool configureMacro2D(uint32_t tilingIdx, ArrayMode am, MicroMode mm, Macro2D &c) {
  c.am = am;
  c.mm = mm;
  c.pc = pipeConfigOf(tilingIdx);
  c.numPipes = numPipesOf(c.pc);
  c.numPipeBits = numPipeBitsOf(c.pc);
  uint32_t mtm = macroTileModeIndex(tilingIdx, mm, am);
  c.mp = macroParamsForMode(mtm);
  c.numBankBits = numBankBitsOf(c.mp.numBanks);
  c.macroPitch = kMicroW * c.mp.bankWidth * c.numPipes * c.mp.macroAspect;
  c.macroHeight = kMicroH * c.mp.bankHeight * c.mp.numBanks / c.mp.macroAspect;
  c.macroTileBytes = kMicroTileBytes * (c.macroPitch / kMicroW) *
                     (c.macroHeight / kMicroH) / (c.numPipes * c.mp.numBanks);
  return c.macroPitch && c.macroHeight;
}

}  // namespace

bool tilingIsLinear(uint32_t tilingIdx) {
  // Per the Liverpool GB_TILE_MODE table only DisplayLinearAligned(8) and
  // DisplayLinearGeneral(31) are genuinely linear (ArrayLinearAligned/General).
  // Keep the same set the previous code used so Isaac's linear surfaces and
  // small UI textures are still straight-copied.
  return tilingIdx == 8 || tilingIdx == 31;
}

bool detile32(const uint32_t *src, uint32_t *dst, uint32_t width, uint32_t height,
              uint32_t tilingIdx, uint32_t pitch, uint32_t slice) {
  if (tilingIsLinear(tilingIdx)) {
    if (pitch == width) {
      std::memcpy(dst, src, (size_t)width * height * 4);
    } else {
      for (uint32_t y = 0; y < height; y++)
        std::memcpy(dst + (size_t)y * width, src + (size_t)y * pitch, (size_t)width * 4);
    }
    return true;
  }

  const ArrayMode am = arrayModeOf(tilingIdx);
  const MicroMode mm = microModeOf(tilingIdx);
  if (mm == MM_Thick) return false;  // z-interleaved thick tiles are not modelled

  // 1D micro-tiled (no pipe/bank interleave).
  const bool micro1D = (am == AM_1DThin1 || am == AM_1DThick);

  if (micro1D) {
    uint32_t apitch = (pitch + (kMicroW - 1)) & ~(kMicroW - 1);
    for (uint32_t y = 0; y < height; y++)
      for (uint32_t x = 0; x < width; x++)
        dst[(size_t)y * width + x] = src[addrMicro32(x, y, apitch, mm) >> 2];
    return true;
  }

  // 2D macro-tiled: derive every param from the tile mode.
  Macro2D c{};
  if (!configureMacro2D(tilingIdx, am, mm, c)) return false;

  // Align pitch up to the macro-tile pitch so macro_tiles_per_row is right.
  uint32_t apitch = (pitch + (c.macroPitch - 1)) & ~(c.macroPitch - 1);

  for (uint32_t y = 0; y < height; y++)
    for (uint32_t x = 0; x < width; x++)
      dst[(size_t)y * width + x] = src[addrMacro32(x, y, slice, apitch, c) >> 2];
  return true;
}

uint64_t tiledSliceSize32(uint32_t width, uint32_t height, uint32_t tilingIdx,
                          uint32_t pitch) {
  uint32_t p = pitch ? pitch : width;
  if (tilingIsLinear(tilingIdx))
    return static_cast<uint64_t>(p) * height * 4;

  ArrayMode am = arrayModeOf(tilingIdx);
  if (microModeOf(tilingIdx) == MM_Thick) return 0;
  if (am == AM_1DThin1 || am == AM_1DThick) {
    uint32_t apitch = (p + kMicroW - 1) & ~(kMicroW - 1);
    uint32_t aheight = (height + kMicroH - 1) & ~(kMicroH - 1);
    return static_cast<uint64_t>(apitch) * aheight * 4;
  }

  Macro2D c{};
  if (!configureMacro2D(tilingIdx, am, microModeOf(tilingIdx), c))
    return static_cast<uint64_t>(p) * height * 4;
  uint32_t apitch = (p + c.macroPitch - 1) & ~(c.macroPitch - 1);
  uint32_t aheight = (height + c.macroHeight - 1) & ~(c.macroHeight - 1);
  return static_cast<uint64_t>(apitch) * aheight * 4;
}

}  // namespace gpu::gcn
