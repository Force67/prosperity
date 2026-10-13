#pragma once

#include <cstddef>
#include "base/arch.h"
#include "base/containers/array.h"
#include "base/containers/vector.h"
#include "base/functional/function.h"

namespace gpu::gcn {

void DetileParallelRows(u32 rows, const base::Function<void(u32, u32)>& fn);

void DetileParallelWork(u32 units,
                        u64 work_items,
                        const base::Function<void(u32, u32)>& fn);

constexpr u32 kGfx10TilingBase = 0x100;

bool TilingIsLinear(u32 tiling_idx);

bool TilingSupported(u32 tiling_idx);

struct TextureMipLayout32 {
  u64 offset = 0;
  u64 size = 0;
  u32 width = 0;
  u32 height = 0;
  u32 pitch = 0;
  u32 stored_height = 0;
  u32 thickness = 1;
  bool macro_tiled = false;
  u32 mip_tail_x = 0;
  u32 mip_tail_y = 0;
};

struct TextureLayout32 {
  base::Array<TextureMipLayout32, 16> mips{};
  u64 size = 0;
  u32 mip_levels = 0;
  u32 layers = 0;
  u32 tiling_idx = 0;
  u32 elem_bytes = 4;
  u64 layer_stride = 0;
};

bool BuildGfx10AddressTable(const TextureLayout32& layout,
                            u32 mip,
                            base::Vector<u32>& terms,
                            u32& block_mask);

bool BuildSeparableAddressTable(const TextureLayout32& layout,
                                u32 mip,
                                base::Vector<u32>& terms,
                                u32& block_mask,
                                u64& slice_stride);

void CopyImageContents(const TextureLayout32& layout,
                       const void* src,
                       void* dst);

bool BuildTextureLayout32(TextureLayout32& out,
                          u32 width,
                          u32 height,
                          u32 pitch,
                          u32 layers,
                          u32 mip_levels,
                          u32 tiling_idx,
                          bool pow2_pad,
                          u32 elem_bytes = 4);

bool DetileTextureMip32(const void* src,
                        void* dst,
                        const TextureLayout32& layout,
                        u32 mip,
                        u32 layer);

bool DetileTextureMip32Pitched(const void* src,
                               void* dst,
                               size_t dst_row_bytes,
                               const TextureLayout32& layout,
                               u32 mip,
                               u32 layer);

bool RetileTextureMip32(const void* src,
                        void* dst,
                        const TextureLayout32& layout,
                        u32 mip,
                        u32 layer);

bool RetileTextureMip32Pitched(const void* src,
                               size_t src_row_bytes,
                               void* dst,
                               const TextureLayout32& layout,
                               u32 mip,
                               u32 layer);

}  // namespace gpu::gcn
