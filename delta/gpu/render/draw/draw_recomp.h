/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "gpu/render/command.h"
#include "gpu/render/renderer.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

bool DrawRecomp(render::Renderer& renderer, const render::DrawInfo& d);

bool ShaderFilterDrops(u64 ps_addr);

void ReleaseDrawBufferBindings();

void ReportDeclines();

rhi::TextureView* BindlessTextureView(const DrawInfo::DrawTex& t,
                                      rhi::TextureState* state);

}  // namespace gpu::render
