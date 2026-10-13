#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"

#include "gpu/ps4/liverpool.h"
#include "gpu/render/renderer.h"

namespace gpu::ps4 {

void DispatchCompute(render::Renderer& renderer,
                     const Regs& regs,
                     const u32* body,
                     u32 count,
                     u64 indirect_args = 0);

void DispatchComputeIndirect(render::Renderer& renderer,
                             const Regs& regs,
                             const u32* body,
                             u32 count,
                             u64 argument_base);

void PrefetchComputeDispatch(const Regs& regs);

}  // namespace gpu::ps4
