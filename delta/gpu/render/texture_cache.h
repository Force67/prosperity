/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

#include "gpu/render/command.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

constexpr u32 kMaxTex = 64;

struct TextureBindings {
  bool descriptors_ready = false;
  rhi::BindGroupLayout* layout = nullptr;
  rhi::Sampler* sampler = nullptr;
  rhi::Sampler* sampler_nearest = nullptr;

  rhi::Texture* white_img = nullptr;
  rhi::Texture* white_3d_img = nullptr;
  rhi::TextureView* white_view = nullptr;
  rhi::TextureView* white_uint_view = nullptr;
  rhi::TextureView* white_uint_array_view = nullptr;
  rhi::TextureView* white_uint_3d_view = nullptr;
  rhi::TextureView* white_array_view = nullptr;
  rhi::TextureView* white_3d_view = nullptr;
  rhi::TextureView* zero_view = nullptr;
  rhi::TextureView* zero_array_view = nullptr;
  rhi::TextureView* zero_3d_view = nullptr;
  rhi::Texture* depth_default_img = nullptr;
  rhi::TextureView* depth_default_view = nullptr;
  rhi::BindGroup* white_set = nullptr;
  rhi::BindGroup* white_array_set = nullptr;
  rhi::BindGroup* white_3d_set = nullptr;
  rhi::BindGroup* zero_set = nullptr;
  rhi::BindGroup* zero_array_set = nullptr;
  rhi::BindGroup* zero_3d_set = nullptr;
};

extern TextureBindings& g_tex;

bool CreateTextureDescriptors();

rhi::BindGroup* SampledTextureGroup(
    rhi::TextureView* view,
    rhi::Sampler* sampler,
    rhi::TextureState state = rhi::TextureState::kShaderRead);

rhi::BindGroup* GetTexture(u64 base,
                           u32 w,
                           u32 h,
                           u32 dfmt,
                           u32 nfmt,
                           u32 tiling = 8,
                           u32 pitch = 0,
                           u32 layers = 1,
                           u32 base_array = 0,
                           u32 view_layers = 1,
                           u32 mip_levels = 1,
                           u32 base_mip = 0,
                           u32 view_mips = 1,
                           u32 min_lod = 0,
                           bool pow2_pad = false,
                           const u32* sampler = nullptr,
                           bool sampler_valid = false,
                           bool arrayed = false,
                           bool force_lod_zero = false,
                           bool depth_compare = false,
                           u32 swizzle = 0,
                           u32 depth = 1,
                           bool is_3d = false);
bool GuestTextureUploadSupported(u32 dfmt, u32 nfmt);
rhi::TextureView* TexViewFor(const render::DrawInfo::DrawTex& t);
rhi::Sampler* SamplerForTex(const render::DrawInfo::DrawTex& t,
                            rhi::Format view_format);
rhi::TextureView* DefaultTexView(const render::DrawInfo::DrawTex& t,
                                 bool want_uint);
void BumpTextureEpoch();
u64 TextureEpoch();

rhi::BindGroup* GetMultiTexSet(const render::DrawInfo& d,
                               rhi::BindGroupLayout* set_layout,
                               u32 num_bindings,
                               rhi::TextureView* const* resolved_views,
                               const rhi::TextureState* resolved_layouts,
                               const rhi::Format* resolved_formats,
                               const u64* depth_src = nullptr);

void InvalidateTexRange(u64 base, u64 size);

void ReleaseRetiredTextures();
void ClearMultiTexCache();

}  // namespace gpu::render
