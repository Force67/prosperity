/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "gpu/render/renderer.h"

namespace gpu::render {

// Writes 32-bit indices into the device-local ring before the render pass.
bool ConvertGuestIndices(Renderer& renderer,
                         u64 base,
                         u32 count,
                         u32 index_type,
                         bool quads,
                         u64 offset);

}  // namespace gpu::render
