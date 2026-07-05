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

#include <cstdint>

namespace gpu::gcn {

// True if tilingIdx denotes a linear surface (no de-tile needed): only
// DisplayLinearAligned(8) and DisplayLinearGeneral(31) are linear on Liverpool.
bool tilingIsLinear(uint32_t tilingIdx);

// De-tile a 32bpp (RGBA8) surface from `src` (tiled, guest layout) into `dst`
// (row-major linear, stride = width*4). `pitch` is the tiled surface pitch in
// pixels (T#.pitch). Linear surfaces are copied straight through. Returns false
// if the mode is unsupported (caller should fall back to a linear copy).
bool detile32(const uint32_t *src, uint32_t *dst, uint32_t width, uint32_t height,
              uint32_t tilingIdx, uint32_t pitch);

}  // namespace gpu::gcn
