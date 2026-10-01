#pragma once

#include "gpu/gcn/gcn_detile.h"

namespace gpu::render {

// Synchronous gfx10 conversion: direct 32/64/128-bit texels or 8/16-bit texels
// expanded to dwords. Retiling preserves padding and truncates expanded values.
bool ConvertGfx10Image(const gcn::TextureLayout32& tiled,
                       const gcn::TextureLayout32& linear,
                       const void* src,
                       void* dst,
                       bool detile,
                       bool pack11 = false);
// The host's R11G11B10F packing of one RGBA32F texel.
u32 PackR11G11B10Texel(const u8* rgba32f);
void UnpackR11G11B10Texel(u32 packed, u8* rgba32f);

}  // namespace gpu::render
