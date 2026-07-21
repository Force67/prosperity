/*
 * PS4Delta : GCN / Liverpool texture de-tiling (32bpp). See gcn_detile.h.
 *
 * Faithful port of the AMD AddrLib (Liverpool) tiling/de-tiling swizzle for
 * 32bpp, one-sample 2D and 2D-array textures, including mip chains and thick
 * microtile Z interleaving. XThick surfaces are degraded to Thick as AddrLib
 * requires when a 32bpp microtile exceeds Liverpool's 1 KiB row size.
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
  kAmLinearGeneral = 0,
  kAmLinearAligned = 1,
  kAm1DThin1 = 2,
  kAm1DThick = 3,
  kAm2DThin1 = 4,
  kAmPrtThin1 = 5,
  kAmPrt2DThin1 = 6,
  kAm2DThick = 7,
  kAm2DXThick = 8,
  kAmPrtThick = 9,
  kAmPrt2DThick = 10,
  kAmPrt3DThin1 = 11,
  kAm3DThin1 = 12,
  kAm3DThick = 13,
  kAm3DXThick = 14,
  kAmPrt3DThick = 15,
};

enum MicroMode { kMmDisplay = 0, kMmThin = 1, kMmDepth = 2, kMmRotated = 3, kMmThick = 4 };

// Two 8-pipe pipe-config equations are used by the colour/depth modes on
// Liverpool: P8_32x32_16x16 and P8_32x32_8x16. (P2 only for LinearGeneral.)
enum PipeConfig { kPcP2 = 0, kPcP8_32x32_8x16 = 10, kPcP8_32x32_16x16 = 12 };

ArrayMode ArrayModeOf(uint32_t idx) {
  switch (idx) {
    case 5:  // Depth1DThin
    case 9:  // Display1DThin
    case 13: // Thin1DThin
      return kAm1DThin1;
    case 0: case 1: case 2: case 3: case 4: // Depth2DThin*
    case 10: // Display2DThin
    case 14: // Thin2DThin
      return kAm2DThin1;
    case 11: // DisplayThinPrt
    case 16: // ThinThinPrt
      return kAmPrtThin1;
    case 6:  // Depth2DThinPrt256
    case 7:  // Depth2DThinPrt1K
    case 12: // Display2DThinPrt
    case 17: // Thin2DThinPrt
      return kAmPrt2DThin1;
    case 15: // Thin3DThin
      return kAm3DThin1;
    case 18: // Thin3DThinPrt
      return kAmPrt3DThin1;
    case 19: return kAm1DThick;   // Thick1DThick
    case 20: return kAm2DThick;   // Thick2DThick
    case 21: return kAm3DThick;   // Thick3DThick
    case 22: return kAmPrtThick;  // ThickThickPrt
    case 23: return kAmPrt2DThick;// Thick2DThickPrt
    case 24: return kAmPrt3DThick;// Thick3DThickPrt
    case 25: return kAm2DXThick;  // Thick2DXThick
    case 26: return kAm3DXThick;  // Thick3DXThick
    case 8:  return kAmLinearAligned;  // DisplayLinearAligned
    case 31: return kAmLinearGeneral;  // DisplayLinearGeneral
    default: return kAmLinearGeneral;   // reserved; rejected by ValidTileMode()
  }
}

MicroMode MicroModeOf(uint32_t idx) {
  switch (idx) {
    case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7:
      return kMmDepth;
    case 8: case 9: case 10: case 11: case 12: case 31:
      return kMmDisplay;
    case 13: case 14: case 15: case 16: case 17: case 18:
      return kMmThin;
    default: // 19..26
      return kMmThick;
  }
}

PipeConfig PipeConfigOf(uint32_t idx) {
  switch (idx) {
    // P8_32x32_8x16: DisplayThinPrt(11), Thin3DThin(15), ThinThinPrt(16),
    // Thick3DThick(21), ThickThickPrt(22), Thick3DThickPrt(24), Thick3DXThick(26)
    case 11: case 15: case 16: case 21: case 22: case 24: case 26:
      return kPcP8_32x32_8x16;
    case 31:
      return kPcP2;
    default:
      return kPcP8_32x32_16x16;
  }
}

// SAMPLE_SPLIT (sample-split count) per GB_TILE_MODE; 2 for Display2DThin and the
// 2D/3D thin colour modes, 1 elsewhere. Affects the 32bpp tile_split (-> 512).
uint32_t SampleSplitOf(uint32_t idx) {
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
uint32_t TileSplitHwOf(uint32_t idx) {
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
  uint32_t num_banks, bank_width, bank_height, macro_aspect;
};

// AddrLib macro-tile-mode LUTs, indexed by MacroTileMode (0..15). For the
// non-Neo (base PS4) primary config we use the non-alt tables.
MacroParams MacroParamsForMode(uint32_t m) {
  // {num_banks, bank_width, bank_height, macro_aspect}
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

constexpr uint32_t kMicroW = 8, kMicroH = 8;
constexpr uint32_t kMicroTilePixels = kMicroW * kMicroH;
constexpr uint32_t kPipeInterleaveBits = 8;

bool ValidTileMode(uint32_t idx) {
  return idx <= 26 || idx == 31;
}

uint32_t EffectiveTileMode(uint32_t idx) {
  if (idx == 25) return 20;  // 2D XThick -> 2D Thick at 32bpp
  if (idx == 26) return 21;  // 3D XThick -> 3D Thick at 32bpp
  return idx;
}

uint32_t TileThickness(ArrayMode am) {
  switch (am) {
    case kAm1DThick: case kAm2DThick: case kAmPrtThick:
    case kAmPrt2DThick: case kAm3DThick: case kAmPrt3DThick:
      return 4;
    case kAm2DXThick: case kAm3DXThick:
      return 8;
    default:
      return 1;
  }
}

bool IsMacroTiled(ArrayMode am) {
  return am != kAmLinearGeneral && am != kAmLinearAligned &&
         am != kAm1DThin1 && am != kAm1DThick;
}

uint32_t NumPipesOf(PipeConfig pc) { return pc == kPcP2 ? 2u : 8u; }
uint32_t NumPipeBitsOf(PipeConfig pc) { return pc == kPcP2 ? 1u : 3u; }

bool IsPrt(ArrayMode am) {
  return am == kAmPrtThin1 || am == kAmPrtThick || am == kAmPrt2DThin1 ||
         am == kAmPrt2DThick || am == kAmPrt3DThin1 || am == kAmPrt3DThick;
}

// Effective tile size used to select the macro-tile table. Thick microtiles may
// exceed the 1 KiB DRAM-row split, but unlike thin multisample tiles they are not
// split into virtual slices by the address equation.
uint32_t TileSizeBytes(uint32_t idx, MicroMode mm, uint32_t thickness,
                       uint32_t elem) {
  const uint32_t tile_bytes_1x = kMicroTilePixels * elem * thickness;
  const uint32_t color_split = SampleSplitOf(idx) * tile_bytes_1x;
  const uint32_t cts = color_split < 256u ? 256u : color_split;
  uint32_t split = (mm == kMmDepth) ? TileSplitHwOf(idx) : cts;
  if (split > 1024u) split = 1024u;
  if (split > tile_bytes_1x) split = tile_bytes_1x;
  return split;
}

// CalculateMacrotileMode: mtm = log2(tile_split/64); +8 if PRT.
uint32_t MacroTileModeIndex(uint32_t idx, MicroMode mm, ArrayMode am,
                             uint32_t thickness, uint32_t elem) {
  uint32_t split = TileSizeBytes(idx, mm, thickness, elem);
  uint32_t q = split / 64u;
  uint32_t mtm = 0;
  while ((1u << (mtm + 1)) <= q) mtm++;  // bit_width(q)-1
  if (IsPrt(am)) mtm += 8;
  return mtm;
}

// Element index within an 8x8x{1,4,8} microtile.
inline uint32_t PixIdx(uint32_t x, uint32_t y, uint32_t z, MicroMode m,
                       uint32_t thickness, uint32_t elem) {
  uint32_t x0 = x & 1, x1 = (x >> 1) & 1, x2 = (x >> 2) & 1;
  uint32_t y0 = y & 1, y1 = (y >> 1) & 1, y2 = (y >> 2) & 1;
  if (m == kMmDisplay) {
    if (elem == 2)
      return x0 | (x1 << 1) | (x2 << 2) | (y0 << 3) | (y1 << 4) |
             (y2 << 5);
    if (elem == 4)
      return x0 | (x1 << 1) | (y0 << 2) | (x2 << 3) | (y1 << 4) |
             (y2 << 5);
    if (elem == 8)
      return x0 | (y0 << 1) | (x1 << 2) | (x2 << 3) | (y1 << 4) |
             (y2 << 5);
    return y0 | (x0 << 1) | (x1 << 2) | (x2 << 3) | (y1 << 4) |
           (y2 << 5);
  }
  if (m != kMmThick)
    return x0 | (y0 << 1) | (x1 << 2) | (y1 << 3) | (x2 << 4) | (y2 << 5);
  uint32_t z0 = z & 1, z1 = (z >> 1) & 1;
  uint32_t index;
  if (elem <= 2)
    index = x0 | (y0 << 1) | (x1 << 2) | (y1 << 3) | (z0 << 4) |
            (z1 << 5) | (x2 << 6) | (y2 << 7);
  else if (elem == 4)
    index = x0 | (y0 << 1) | (x1 << 2) | (z0 << 3) | (y1 << 4) |
            (z1 << 5) | (x2 << 6) | (y2 << 7);
  else
    index = x0 | (y0 << 1) | (z0 << 2) | (x1 << 3) | (y1 << 4) |
            (z1 << 5) | (x2 << 6) | (y2 << 7);
  if (thickness == 8) index |= ((z >> 2) & 1) << 8;
  return index;
}

// ComputePipeFromCoord (tiling.comp), 8-pipe configs.
inline uint32_t PipeFromCoord(uint32_t x, uint32_t y, uint32_t slice,
                               PipeConfig pc, ArrayMode am, uint32_t thickness) {
  uint32_t tx = x >> 3, ty = y >> 3;
  uint32_t x3 = tx & 1, x4 = (tx >> 1) & 1, x5 = (tx >> 2) & 1;
  uint32_t y3 = ty & 1, y4 = (ty >> 1) & 1, y5 = (ty >> 2) & 1;
  uint32_t pipe;
  if (pc == kPcP2) pipe = x3 ^ y3;
  else if (pc == kPcP8_32x32_8x16) {
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
  if (am == kAm3DThin1 || am == kAm3DThick || am == kAm3DXThick) {
    uint32_t rotation = std::max(1u, NumPipesOf(pc) / 2 - 1) * (slice / thickness);
    pipe ^= rotation & (NumPipesOf(pc) - 1);
  }
  return pipe;
}

// ComputeBankFromCoord (tiling.comp), parameterized by num_banks/widths.
inline uint32_t BankFromCoord(uint32_t x, uint32_t y, uint32_t slice,
                               const MacroParams &mp, uint32_t num_pipes,
                               ArrayMode am, uint32_t thickness,
                               uint32_t tile_split_slice) {
  uint32_t tx = (x >> 3) / (mp.bank_width * num_pipes);
  uint32_t ty = (y >> 3) / mp.bank_height;
  uint32_t x3 = tx & 1, x4 = (tx >> 1) & 1, x5 = (tx >> 2) & 1, x6 = (tx >> 3) & 1;
  uint32_t y3 = ty & 1, y4 = (ty >> 1) & 1, y5 = (ty >> 2) & 1, y6 = (ty >> 3) & 1;
  uint32_t bank = 0;
  switch (mp.num_banks) {
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
  if (am == kAm2DThin1 || am == kAm2DThick || am == kAm2DXThick)
    rotation = (mp.num_banks / 2 - 1) * (slice / thickness);
  else if (am == kAm3DThin1 || am == kAm3DThick || am == kAm3DXThick)
    rotation = std::max(1u, num_pipes / 2 - 1) * (slice / thickness) / num_pipes;
  uint32_t tile_split_rotation = 0;
  if (am == kAm2DThin1 || am == kAm3DThin1 || am == kAmPrt2DThin1 ||
      am == kAmPrt3DThin1)
    tile_split_rotation = (mp.num_banks / 2 + 1) * tile_split_slice;
  return (bank ^ rotation ^ tile_split_rotation) & (mp.num_banks - 1);
}

// 1D micro-tiled byte offset, including thick slice groups. `elem` is the
// element size in bytes (4 = pixel, 8/16 = BCn block); the intra-tile swizzle
// is an element index, so it scales directly.
inline uint64_t AddrMicro32(uint32_t x, uint32_t y, uint32_t slice,
                            uint32_t pitch, uint32_t height, MicroMode m,
                            uint32_t thickness, uint32_t elem) {
  uint32_t micro_tiles_per_row = pitch / kMicroW;
  uint64_t group_bytes = static_cast<uint64_t>(pitch) * height * thickness * elem;
  uint64_t slice_offset = (slice / thickness) * group_bytes;
  uint64_t micro_tile_offset =
      static_cast<uint64_t>((y >> 3) * micro_tiles_per_row + (x >> 3)) *
      (kMicroTilePixels * elem) * thickness;
  return slice_offset + micro_tile_offset +
         static_cast<uint64_t>(PixIdx(x, y, slice, m, thickness, elem)) * elem;
}

struct Macro2D {
  ArrayMode am;
  MicroMode mm;
  PipeConfig pc;
  MacroParams mp;
  uint32_t num_pipes, num_pipe_bits, num_bank_bits;
  uint32_t thickness, elem_bytes, micro_tile_bytes, slices_per_tile;
  uint32_t macro_pitch, macro_height, macro_tile_bytes, base_align;
};

inline uint32_t NumBankBitsOf(uint32_t num_banks) {
  uint32_t b = 0;
  while ((1u << (b + 1)) <= num_banks) b++;
  return b;  // bit_width(num_banks)-1
}

// 2D macro-tiled byte offset, one sample.
inline uint64_t AddrMacro32(uint32_t x, uint32_t y, uint32_t slice,
                            uint32_t pitch, uint32_t height, const Macro2D &c) {
  uint32_t element_offset =
      PixIdx(x, y, slice, c.mm, c.thickness, c.elem_bytes) * c.elem_bytes;
  uint32_t tile_split_slice = 0;
  if (c.slices_per_tile > 1) {
    tile_split_slice = element_offset / c.micro_tile_bytes;
    element_offset %= c.micro_tile_bytes;
  }

  uint32_t macro_tiles_per_row = pitch / c.macro_pitch;
  uint32_t mtx = x / c.macro_pitch;
  uint32_t mty = y / c.macro_height;
  uint64_t macro_tile_offset =
      static_cast<uint64_t>(mty * macro_tiles_per_row + mtx) * c.macro_tile_bytes;
  uint32_t macro_tiles_per_slice = macro_tiles_per_row * (height / c.macro_height);
  uint64_t slice_bytes = static_cast<uint64_t>(macro_tiles_per_slice) * c.macro_tile_bytes;
  uint64_t slice_offset = slice_bytes *
      (tile_split_slice + c.slices_per_tile * (slice / c.thickness));

  // tile_row/tile_column rotation within the macro tile (0 when bw==bh==1).
  uint32_t tile_row = (y >> 3) % c.mp.bank_height;
  uint32_t tile_col = ((x >> 3) / c.num_pipes) % c.mp.bank_width;
  uint32_t tile_offset =
      (tile_row * c.mp.bank_width + tile_col) * c.micro_tile_bytes;

  uint64_t total_offset = slice_offset + macro_tile_offset + element_offset + tile_offset;

  uint32_t swizzle_x = x, swizzle_y = y;
  if (c.am == kAmPrtThin1 || c.am == kAmPrtThick) {
    swizzle_x %= c.macro_pitch;
    swizzle_y %= c.macro_height;
  }
  uint32_t pipe = PipeFromCoord(swizzle_x, swizzle_y, slice, c.pc, c.am, c.thickness);
  uint32_t bank = BankFromCoord(swizzle_x, swizzle_y, slice, c.mp, c.num_pipes,
                                c.am, c.thickness, tile_split_slice);

  uint64_t interleave_offset = total_offset & ((1u << kPipeInterleaveBits) - 1);
  uint64_t offset = total_offset >> kPipeInterleaveBits;

  uint64_t addr = interleave_offset;
  addr |= pipe << kPipeInterleaveBits;
  addr |= bank << (kPipeInterleaveBits + c.num_pipe_bits);
  addr |= offset << (kPipeInterleaveBits + c.num_pipe_bits + c.num_bank_bits);
  return addr;
}

bool ConfigureMacro2D(uint32_t tiling_idx, ArrayMode am, MicroMode mm,
                      uint32_t elem, Macro2D &c) {
  c.am = am;
  c.mm = mm;
  c.pc = PipeConfigOf(tiling_idx);
  c.num_pipes = NumPipesOf(c.pc);
  c.num_pipe_bits = NumPipeBitsOf(c.pc);
  c.thickness = TileThickness(am);
  c.elem_bytes = elem;
  const uint32_t full_micro_tile_bytes =
      kMicroTilePixels * elem * c.thickness;
  const uint32_t tile_split_bytes =
      TileSizeBytes(tiling_idx, mm, c.thickness, elem);
  c.micro_tile_bytes = full_micro_tile_bytes;
  c.slices_per_tile = 1;
  // AddrLib addresses each split portion as a virtual slice with its own bank
  // rotation.
  if (full_micro_tile_bytes > tile_split_bytes && c.thickness == 1) {
    c.micro_tile_bytes = tile_split_bytes;
    c.slices_per_tile = full_micro_tile_bytes / tile_split_bytes;
  }
  uint32_t mtm =
      MacroTileModeIndex(tiling_idx, mm, am, c.thickness, elem);
  c.mp = MacroParamsForMode(mtm);
  c.num_bank_bits = NumBankBitsOf(c.mp.num_banks);
  c.macro_pitch = kMicroW * c.mp.bank_width * c.num_pipes * c.mp.macro_aspect;
  c.macro_height = kMicroH * c.mp.bank_height * c.mp.num_banks / c.mp.macro_aspect;
  c.macro_tile_bytes = c.micro_tile_bytes * (c.macro_pitch / kMicroW) *
                      (c.macro_height / kMicroH) / (c.num_pipes * c.mp.num_banks);
  c.base_align = c.num_pipes * c.mp.bank_width * c.mp.num_banks * c.mp.bank_height *
                 TileSizeBytes(tiling_idx, mm, c.thickness, elem);
  return c.macro_pitch && c.macro_height;
}

uint32_t AlignUp(uint32_t value, uint32_t align) {
  return (value + align - 1) & ~(align - 1);
}

uint64_t AlignUp(uint64_t value, uint64_t align) {
  return (value + align - 1) & ~(align - 1);
}

uint32_t BitCeil(uint32_t value) {
  if (value <= 1) return 1;
  value--;
  value |= value >> 1; value |= value >> 2; value |= value >> 4;
  value |= value >> 8; value |= value >> 16;
  return value + 1;
}

}  // namespace

bool TilingIsLinear(uint32_t tiling_idx) {
  // Per the Liverpool GB_TILE_MODE table only DisplayLinearAligned(8) and
  // DisplayLinearGeneral(31) are genuinely linear (ArrayLinearAligned/General).
  // Keep the same set the previous code used so Isaac's linear surfaces and
  // small UI textures are still straight-copied.
  return tiling_idx == 8 || tiling_idx == 31;
}

bool BuildTextureLayout32(TextureLayout32 &out, uint32_t width, uint32_t height,
                          uint32_t pitch, uint32_t layers, uint32_t mip_levels,
                          uint32_t tiling_idx, bool pow2_pad,
                          uint32_t elem_bytes) {
  out = {};
  if (!width || !height || !layers || !mip_levels || mip_levels > out.mips.size() ||
      width > 16384 || height > 16384 || pitch > 16384 || layers > 8192 ||
      !ValidTileMode(tiling_idx))
    return false;
  if (elem_bytes != 2 && elem_bytes != 4 && elem_bytes != 8 &&
      elem_bytes != 16)
    return false;

  tiling_idx = EffectiveTileMode(tiling_idx);
  out.mip_levels = mip_levels;
  out.layers = layers;
  out.tiling_idx = tiling_idx;
  out.elem_bytes = elem_bytes;
  const ArrayMode am = ArrayModeOf(tiling_idx);
  const MicroMode mm = MicroModeOf(tiling_idx);
  const uint32_t thickness = TileThickness(am);
  Macro2D macro{};
  if (IsMacroTiled(am) &&
      !ConfigureMacro2D(tiling_idx, am, mm, elem_bytes, macro))
    return false;

  uint64_t end = 0;
  for (uint32_t mip = 0; mip < mip_levels; mip++) {
    TextureMipLayout32 &level = out.mips[mip];
    level.width = std::max(width >> mip, 1u);
    level.height = std::max(height >> mip, 1u);
    uint32_t raw_pitch = std::max(std::max(pitch >> mip, 1u), level.width);
    uint32_t storage_height = level.height;
    if (pow2_pad) {
      raw_pitch = BitCeil(raw_pitch);
      storage_height = BitCeil(storage_height);
    }
    level.thickness = thickness;

    uint64_t base_align = 1;
    if (tiling_idx == 31) {
      level.pitch = raw_pitch;
      level.stored_height = storage_height;
    } else if (tiling_idx == 8) {
      level.pitch = AlignUp(raw_pitch, 16);
      level.stored_height = storage_height;
      while ((static_cast<uint64_t>(level.pitch) * level.stored_height) % 64)
        level.pitch += 16;
      base_align = 256;
    } else {
      level.macro_tiled = IsMacroTiled(am) &&
          !(mip > 0 && (raw_pitch < macro.macro_pitch ||
                        storage_height < macro.macro_height));
      if (level.macro_tiled) {
        level.pitch = AlignUp(raw_pitch, macro.macro_pitch);
        level.stored_height = AlignUp(storage_height, macro.macro_height);
        base_align = macro.base_align;
      } else {
        level.pitch = AlignUp(raw_pitch, kMicroW);
        level.stored_height = AlignUp(storage_height, kMicroH);
        base_align = 256;
      }
    }

    const uint32_t stored_layers = AlignUp(layers, thickness);
    level.offset = AlignUp(end, base_align);
    level.size = static_cast<uint64_t>(level.pitch) * level.stored_height *
                 stored_layers * elem_bytes;
    if (!level.size || level.offset > UINT64_MAX - level.size) return false;
    end = level.offset + level.size;
  }
  out.size = end;
  return out.size != 0;
}

bool DetileTextureMip32(const void *src, void *dst,
                         const TextureLayout32 &layout, uint32_t mip,
                         uint32_t layer) {
  if (!src || !dst || mip >= layout.mip_levels || layer >= layout.layers)
    return false;
  const TextureMipLayout32 &level = layout.mips[mip];
  const uint32_t elem = layout.elem_bytes;
  const uint8_t *level_src = static_cast<const uint8_t *>(src) + level.offset;
  uint8_t *out = static_cast<uint8_t *>(dst);

  if (TilingIsLinear(layout.tiling_idx)) {
    const uint8_t *layer_src =
        level_src + static_cast<uint64_t>(layer) * level.pitch *
                        level.stored_height * elem;
    for (uint32_t y = 0; y < level.height; y++)
      std::memcpy(out + static_cast<size_t>(y) * level.width * elem,
                  layer_src + static_cast<size_t>(y) * level.pitch * elem,
                  static_cast<size_t>(level.width) * elem);
    return true;
  }

  const ArrayMode am = ArrayModeOf(layout.tiling_idx);
  const MicroMode mm = MicroModeOf(layout.tiling_idx);
  if (!level.macro_tiled) {
    for (uint32_t y = 0; y < level.height; y++)
      for (uint32_t x = 0; x < level.width; x++)
        std::memcpy(out + (static_cast<size_t>(y) * level.width + x) * elem,
                    level_src + AddrMicro32(x, y, layer, level.pitch,
                                            level.stored_height, mm,
                                            level.thickness, elem),
                    elem);
    return true;
  }

  Macro2D macro{};
  if (!ConfigureMacro2D(layout.tiling_idx, am, mm, elem, macro)) return false;
  for (uint32_t y = 0; y < level.height; y++)
    for (uint32_t x = 0; x < level.width; x++)
      std::memcpy(out + (static_cast<size_t>(y) * level.width + x) * elem,
                  level_src + AddrMacro32(x, y, layer, level.pitch,
                                           level.stored_height, macro),
                  elem);
  return true;
}

bool RetileTextureMip32(const void *src, void *dst,
                        const TextureLayout32 &layout, uint32_t mip,
                        uint32_t layer) {
  if (!src || !dst || mip >= layout.mip_levels || layer >= layout.layers)
    return false;
  const TextureMipLayout32 &level = layout.mips[mip];
  const uint32_t elem = layout.elem_bytes;
  const uint8_t *in = static_cast<const uint8_t *>(src);
  uint8_t *level_dst = static_cast<uint8_t *>(dst) + level.offset;

  if (TilingIsLinear(layout.tiling_idx)) {
    uint8_t *layer_dst =
        level_dst + static_cast<uint64_t>(layer) * level.pitch *
                        level.stored_height * elem;
    for (uint32_t y = 0; y < level.height; y++)
      std::memcpy(layer_dst + static_cast<size_t>(y) * level.pitch * elem,
                  in + static_cast<size_t>(y) * level.width * elem,
                  static_cast<size_t>(level.width) * elem);
    return true;
  }

  const ArrayMode am = ArrayModeOf(layout.tiling_idx);
  const MicroMode mm = MicroModeOf(layout.tiling_idx);
  if (!level.macro_tiled) {
    for (uint32_t y = 0; y < level.height; y++)
      for (uint32_t x = 0; x < level.width; x++)
        std::memcpy(level_dst + AddrMicro32(x, y, layer, level.pitch,
                                            level.stored_height, mm,
                                            level.thickness, elem),
                    in + (static_cast<size_t>(y) * level.width + x) * elem,
                    elem);
    return true;
  }

  Macro2D macro{};
  if (!ConfigureMacro2D(layout.tiling_idx, am, mm, elem, macro)) return false;
  for (uint32_t y = 0; y < level.height; y++)
    for (uint32_t x = 0; x < level.width; x++)
      std::memcpy(level_dst + AddrMacro32(x, y, layer, level.pitch,
                                          level.stored_height, macro),
                  in + (static_cast<size_t>(y) * level.width + x) * elem,
                  elem);
  return true;
}

}  // namespace gpu::gcn
