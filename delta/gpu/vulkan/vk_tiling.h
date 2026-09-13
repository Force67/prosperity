#pragma once

#include "gpu/gcn/gcn_detile.h"

namespace gpu::vk {

// Synchronous gfx10 conversion: direct 32/64/128-bit texels or 8/16-bit texels
// expanded to dwords. Retiling preserves padding and truncates expanded values.
bool ConvertGfx10Image(const gcn::TextureLayout32& tiled,
                       const gcn::TextureLayout32& linear,
                       const void* src,
                       void* dst,
                       bool detile);

}  // namespace gpu::vk
