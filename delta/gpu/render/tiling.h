#pragma once

#include "gpu/gcn/gcn_detile.h"

namespace gpu::render {

bool ConvertGfx10Image(const gcn::TextureLayout32& tiled,
                       const gcn::TextureLayout32& linear,
                       const void* src,
                       void* dst,
                       bool detile,
                       bool pack11 = false);
u32 PackR11G11B10Texel(const u8* rgba32f);
void UnpackR11G11B10Texel(u32 packed, u8* rgba32f);

}  // namespace gpu::render
