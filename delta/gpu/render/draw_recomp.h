/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The recompiled-shader draw path: runs the game's own VS/PS, resolving its
// vertex streams, constant buffers, samplers and render targets out of the draw
// the command processor decoded.

#include "gpu/render/command.h"
#include "gpu/render/renderer.h"

namespace gpu::render {

// False when the draw cannot be handled; it is then not drawn.
bool DrawRecomp(render::Renderer& renderer, const render::DrawInfo& d);

// DELTA_GPU_SKIP_PS / DELTA_GPU_ONLY_PS: whether this draw is filtered out.
// Asked before recompiling, so a shader that never recompiled is dropped too.
bool ShaderFilterDrops(u64 ps_addr);

// Release raw-buffer descriptor sets after their in-flight frames retire.
void ReleaseDrawBufferBindings();

// Log the tally of why draws declined the recompiled path.
void ReportDeclines();

}  // namespace gpu::render
