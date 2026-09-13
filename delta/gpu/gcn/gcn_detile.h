#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN / Liverpool (Sea Islands) texture de-tiling. PS4 textures live in guest
 * memory in a tiled layout (micro 8x8 tiles, optionally arranged into 2D macro
 * tiles with pipe/bank swizzle). Reading them linearly scrambles the image, so
 * before uploading we de-tile into a row-major linear buffer.
 *
 * Faithful (32bpp) implementation of the AMD AddrLib address swizzle
 * (video_core/host_shaders/tiling.comp + video_core/amdgpu/tiling.cpp).
 *
 * PS5 (gfx10.3 / RDNA2) surfaces are described by a 5-bit swizzle mode rather
 * than a GB_TILE_MODE index, so they enter the same API as
 * kGfx10TilingBase + sw_mode. Those ids take a separate address equation and
 * layout path; nothing in the Liverpool range can reach it.
 */

#include <array>
#include "base/arch.h"
#include <cstddef>
#include <functional>
#include <vector>

namespace gpu::gcn {

// Split the row range [0, rows) into chunks and invoke fn(y0, y1) on each
// across an internal persistent worker pool, joining before return. Falls back
// to a single fn(0, rows) call when the multithreaded path is disabled
// (DELTA_GPU_DETILE_MT=0) or the workload is too small to be worth splitting.
// DELTA_GPU_DETILE_THREADS sets the total lane count, including the caller.
// The pool is owned entirely by this module; callers see a plain blocking call
// and need no synchronization of their own. The detilers use it for 8-row
// microtile bands; the GPU staging paths reuse it for row-major
// format-convert loops. Only one row-parallel region runs at a time (calls are
// serialized), matching the single-threaded GPU pipeline.
void DetileParallelRows(u32 rows,
                        const std::function<void(u32, u32)>& fn);

// As above, but the caller states how much work the units carry (bytes or
// texels) instead of the row heuristic. A few dozen block rows of a 4K surface
// are megabytes of copying and worth splitting; the same count of scanlines is
// not, and only the caller can tell the two apart.
void DetileParallelWork(u32 units,
                        u64 work_items,
                        const std::function<void(u32, u32)>& fn);

// A PS5 gfx10.3 swizzle mode enters the tiling_idx parameter as
// kGfx10TilingBase + sw_mode, keeping it disjoint from the Liverpool indices
// (0..31) and from the older gfx10 "standard" ids the Minecraft path uses.
constexpr u32 kGfx10TilingBase = 0x100;

// True if tiling_idx denotes a linear surface (no de-tile needed): only
// DisplayLinearAligned(8) and DisplayLinearGeneral(31) are linear on Liverpool,
// and LINEAR(0) / LINEAR_GENERAL(31) in the gfx10 range.
bool TilingIsLinear(u32 tiling_idx);

// True if a layout can be built for tiling_idx at all. Callers use it to reject
// a surface before staging it; BuildTextureLayout32 refuses the same set.
bool TilingSupported(u32 tiling_idx);

struct TextureMipLayout32 {
  u64 offset = 0;  // byte offset of this complete mip level
  u64 size = 0;    // bytes occupied by all physical layers
  u32 width = 0;   // logical dimensions copied to Vulkan
  u32 height = 0;
  u32 pitch = 0;  // storage dimensions after tile-mode alignment
  u32 stored_height = 0;
  u32 thickness = 1;    // slices interleaved in each thick microtile
  bool macro_tiled = false;  // false for linear and mip-downgraded 1D tiling
  // gfx10 packs its small mips into one shared block; these place this level
  // inside it. Zero for level 0 and for every Liverpool surface.
  u32 mip_tail_x = 0;
  u32 mip_tail_y = 0;
};

struct TextureLayout32 {
  std::array<TextureMipLayout32, 16> mips{};
  u64 size = 0;
  u32 mip_levels = 0;
  u32 layers = 0;
  u32 tiling_idx = 0;
  u32 elem_bytes = 4;  // bytes per element (2/4 = pixel, 8/16 = BCn block)
  // gfx10 stores the whole mip chain per array slice, so a layer is one stride
  // apart at every level. Zero means the Liverpool layout (mip-major, layers
  // interleaved inside each level).
  u64 layer_stride = 0;
};

// Separable gfx10 byte-address terms, ordered x, y, then array layer.
// Block offsets add; the low bits selected by block_mask XOR together.
bool BuildGfx10AddressTable(const TextureLayout32& layout,
                            u32 mip,
                            std::vector<u32>& terms,
                            u32& block_mask);

// Copy only logical texels between matching gfx10 layouts, leaving padding
// untouched. Whole interior blocks use contiguous copies.
void CopyGfx10ImageContents(const TextureLayout32& layout,
                           const void* src,
                           void* dst);

// Compute the complete physical layout of a one-sample 2D/2D-array image whose
// elements are `elem_bytes` wide (2/4 = a pixel; 8/16 = a BCn block, with
// width/height/pitch given in blocks). Mips are stored mip-major; each mip
// contains all array layers. Later macro-tiled mips are downgraded to 1D
// microtiling when they no longer span a macro tile.
bool BuildTextureLayout32(TextureLayout32& out,
                          u32 width,
                          u32 height,
                          u32 pitch,
                          u32 layers,
                          u32 mip_levels,
                          u32 tiling_idx,
                          bool pow2_pad,
                          u32 elem_bytes = 4);

// De-tile one physical mip/layer into tightly packed row-major elements.
bool DetileTextureMip32(const void* src,
                        void* dst,
                        const TextureLayout32& layout,
                        u32 mip,
                        u32 layer);

// De-tile into rows separated by dst_row_bytes. This avoids an intermediate
// tightly packed buffer when the consumer already has a pitched linear layout.
bool DetileTextureMip32Pitched(const void* src,
                               void* dst,
                               size_t dst_row_bytes,
                               const TextureLayout32& layout,
                               u32 mip,
                               u32 layer);

// Tile one tightly packed row-major physical mip/layer back into guest layout.
bool RetileTextureMip32(const void* src,
                        void* dst,
                        const TextureLayout32& layout,
                        u32 mip,
                        u32 layer);

// Re-tile from rows separated by src_row_bytes.
bool RetileTextureMip32Pitched(const void* src,
                               size_t src_row_bytes,
                               void* dst,
                               const TextureLayout32& layout,
                               u32 mip,
                               u32 layer);

}  // namespace gpu::gcn
