#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "base/arch.h"

namespace gpu::rhi {
class BindGroup;
class BindGroupLayout;
class CommandList;
class Sampler;
class TextureView;
enum class TextureState : u8;
}  // namespace gpu::rhi

namespace gpu::render::bindless {

using TextureResolver = rhi::TextureView* (*)(u32 kind,
                                              const u32* tsharp,
                                              rhi::TextureState* state);
using SamplerResolver = rhi::Sampler* (*)(u32 kind, const u32* ssharp);

bool Supported();
bool Enabled();
void SetResolvers(TextureResolver texture, SamplerResolver sampler);
rhi::BindGroupLayout* Layout();
rhi::BindGroup* Group();

void FrameBegin(rhi::CommandList* list);
void FrameEnd(rhi::CommandList* list, u32 slot);
void FrameFinished(u32 slot);

}  // namespace gpu::render::bindless
