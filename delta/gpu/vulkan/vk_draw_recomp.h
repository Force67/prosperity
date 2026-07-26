/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// The recompiled-shader draw path: runs the game's own VS/PS, resolving its
// vertex streams, constant buffers, samplers and render targets out of the draw
// the command processor decoded.

#include "rhi/command.h"

namespace gpu::vk {

// False when the draw cannot be handled; the caller then falls back to the
// heuristic quad path.
bool drawRecomp(const rhi::DrawInfo &d);

// Log the tally of why draws declined the recompiled path.
void reportDeclines();

}  // namespace gpu::vk
