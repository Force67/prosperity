/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "gpu/render/renderer.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

// Writes 32-bit indices into the device-local ring before the render pass.
// Indirect quad draws also get triangle arguments written into the ring.
bool ConvertGuestIndices(Renderer& renderer,
                         u64 base,
                         u32 count,
                         u32 index_type,
                         bool quads,
                         u64 offset,
                         rhi::Buffer** indirect_buffer = nullptr,
                         u64* indirect_offset = nullptr);

}  // namespace gpu::render
