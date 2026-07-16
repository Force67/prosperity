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
#include <cstdint>

namespace gpu::gcn {

// True if tilingIdx denotes a linear surface (no de-tile needed): only
// DisplayLinearAligned(8) and DisplayLinearGeneral(31) are linear on Liverpool.
bool tilingIsLinear(uint32_t tilingIdx);

struct TextureMipLayout32 {
  uint64_t offset = 0;       // byte offset of this complete mip level
  uint64_t size = 0;         // bytes occupied by all physical layers
  uint32_t width = 0;        // logical dimensions copied to Vulkan
  uint32_t height = 0;
  uint32_t pitch = 0;        // storage dimensions after tile-mode alignment
  uint32_t storedHeight = 0;
  uint32_t thickness = 1;    // slices interleaved in each thick microtile
  bool macroTiled = false;   // false for linear and mip-downgraded 1D tiling
};

struct TextureLayout32 {
  std::array<TextureMipLayout32, 16> mips{};
  uint64_t size = 0;
  uint32_t mipLevels = 0;
  uint32_t layers = 0;
  uint32_t tilingIdx = 0;
};

// Compute the complete physical layout of a 32bpp, one-sample 2D/2D-array image.
// Mips are stored mip-major; each mip contains all array layers. Later macro-tiled
// mips are downgraded to 1D microtiling when they no longer span a macro tile.
bool buildTextureLayout32(TextureLayout32 &out, uint32_t width, uint32_t height,
                          uint32_t pitch, uint32_t layers, uint32_t mipLevels,
                          uint32_t tilingIdx, bool pow2Pad);

// De-tile one physical mip/layer into tightly packed row-major RGBA8 pixels.
bool detileTextureMip32(const uint32_t *src, uint32_t *dst,
                        const TextureLayout32 &layout, uint32_t mip,
                        uint32_t layer);

}  // namespace gpu::gcn
