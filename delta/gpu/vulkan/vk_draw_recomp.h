/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The recompiled-shader draw path: runs the game's own VS/PS, resolving its
// vertex streams, constant buffers, samplers and render targets out of the draw
// the command processor decoded.

#include "gpu/rhi/command.h"
#include "gpu/rhi/renderer.h"

namespace gpu::vk {

// False when the draw cannot be handled; the caller then falls back to the
// heuristic quad path.
bool DrawRecomp(rhi::Renderer& renderer, const rhi::DrawInfo& d);

// DELTA_GPU_SKIP_PS / DELTA_GPU_ONLY_PS: whether this draw is filtered out.
// Asked before either draw path, so a shader that never recompiled is dropped
// too instead of reaching the heuristic renderer.
bool ShaderFilterDrops(u64 ps_addr);

// Log the tally of why draws declined the recompiled path.
void ReportDeclines();

}  // namespace gpu::vk
