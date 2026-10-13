#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "base/arch.h"

#include "gpu/ps5/agc_regs.h"
#include "gpu/render/renderer.h"

namespace gpu::ps5 {

void DispatchCompute(render::Renderer& renderer,
                     const Regs& regs,
                     const u32* body,
                     u32 count,
                     u64 indirect_args = 0);

}  // namespace gpu::ps5
