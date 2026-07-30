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
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>

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
void DetileParallelRows(uint32_t rows,
                        const std::function<void(uint32_t, uint32_t)>& fn);

// True if tiling_idx denotes a linear surface (no de-tile needed): only
// DisplayLinearAligned(8) and DisplayLinearGeneral(31) are linear on Liverpool.
bool TilingIsLinear(uint32_t tiling_idx);

struct TextureMipLayout32 {
  uint64_t offset = 0;  // byte offset of this complete mip level
  uint64_t size = 0;    // bytes occupied by all physical layers
  uint32_t width = 0;   // logical dimensions copied to Vulkan
  uint32_t height = 0;
  uint32_t pitch = 0;  // storage dimensions after tile-mode alignment
  uint32_t stored_height = 0;
  uint32_t thickness = 1;    // slices interleaved in each thick microtile
  bool macro_tiled = false;  // false for linear and mip-downgraded 1D tiling
};

struct TextureLayout32 {
  std::array<TextureMipLayout32, 16> mips{};
  uint64_t size = 0;
  uint32_t mip_levels = 0;
  uint32_t layers = 0;
  uint32_t tiling_idx = 0;
  uint32_t elem_bytes = 4;  // bytes per element (2/4 = pixel, 8/16 = BCn block)
};

// Compute the complete physical layout of a one-sample 2D/2D-array image whose
// elements are `elem_bytes` wide (2/4 = a pixel; 8/16 = a BCn block, with
// width/height/pitch given in blocks). Mips are stored mip-major; each mip
// contains all array layers. Later macro-tiled mips are downgraded to 1D
// microtiling when they no longer span a macro tile.
bool BuildTextureLayout32(TextureLayout32& out,
                          uint32_t width,
                          uint32_t height,
                          uint32_t pitch,
                          uint32_t layers,
                          uint32_t mip_levels,
                          uint32_t tiling_idx,
                          bool pow2_pad,
                          uint32_t elem_bytes = 4);

// De-tile one physical mip/layer into tightly packed row-major elements.
bool DetileTextureMip32(const void* src,
                        void* dst,
                        const TextureLayout32& layout,
                        uint32_t mip,
                        uint32_t layer);

// De-tile into rows separated by dst_row_bytes. This avoids an intermediate
// tightly packed buffer when the consumer already has a pitched linear layout.
bool DetileTextureMip32Pitched(const void* src,
                               void* dst,
                               size_t dst_row_bytes,
                               const TextureLayout32& layout,
                               uint32_t mip,
                               uint32_t layer);

// Tile one tightly packed row-major physical mip/layer back into guest layout.
bool RetileTextureMip32(const void* src,
                        void* dst,
                        const TextureLayout32& layout,
                        uint32_t mip,
                        uint32_t layer);

// Re-tile from rows separated by src_row_bytes.
bool RetileTextureMip32Pitched(const void* src,
                               size_t src_row_bytes,
                               void* dst,
                               const TextureLayout32& layout,
                               uint32_t mip,
                               uint32_t layer);

}  // namespace gpu::gcn
