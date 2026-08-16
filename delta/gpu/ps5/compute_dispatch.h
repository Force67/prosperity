#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * How a compute dispatch's registers and descriptors become one renderer
 * dispatch.
 *
 * A dispatch names nothing directly: the COMPUTE_* registers give a program
 * address and a workgroup shape, and every buffer and image it touches is a
 * descriptor the shader loads out of its user data, possibly through an SRT
 * chain in guest memory another dispatch wrote. Recovering those live ranges,
 * deciding which need staging (a tiled image, a format the GPU cannot alias)
 * and refusing the ones that do not add up happens here.
 *
 * Titles build real content this way -- Minecraft assembles its UI vertex
 * buffers with compute -- so a skipped dispatch leaves the draws that read them
 * fetching zeros. Getting one wrong is a black surface, not a crash, so a
 * refused dispatch is loud (DELTA_GPU_CSDUMP / DELTA_GPU_CSRES).
 */

#include "base/arch.h"

#include "gpu/ps5/agc_regs.h"
#include "gpu/rhi/renderer.h"

namespace gpu::ps5 {

// Execute one IT_DISPATCH_DIRECT. `body` is the packet body: workgroup counts
// in x, y, z followed by the dispatch initiator.
void DispatchCompute(rhi::Renderer& renderer,
                     const Regs& regs,
                     const u32* body,
                     u32 count);

}  // namespace gpu::ps5
