/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "gpu/render/command.h"
#include "gpu/render/renderer.h"

namespace gpu::render {

bool DrawRecomp(render::Renderer& renderer, const render::DrawInfo& d);

bool ShaderFilterDrops(u64 ps_addr);

void ReleaseDrawBufferBindings();

void ReportDeclines();

}  // namespace gpu::render
