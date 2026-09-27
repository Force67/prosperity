/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Guest textures as Vulkan images: the descriptor infrastructure every sampled
// image binds through, and the image/view/sampler/descriptor-set caches keyed
// by the guest T#. A cached surface is revalidated against a content hash, so a
// buffer the guest rewrites is re-uploaded rather than served stale.

#include "base/arch.h"


#include "gpu/render/command.h"
#include "gpu/rhi/device.h"

namespace gpu::render {

// Sampler bindings a recompiled PS may consume in one draw. SOTTR's lighting pass
// declares 23 and a shader needing more is declined outright, so this is a rendering
// cliff, not a tuning knob; 24 sits far inside every driver's reported
// maxPerStageDescriptorSampledImages (Vulkan guarantees only 16).
constexpr u32 kMaxTex = 64;

// Descriptor infrastructure shared by every sampled image (guest textures,
// render targets sampled as textures, the 1x1 white fallback).
struct TextureBindings {
  bool descriptors_ready = false;
  rhi::BindGroupLayout* layout = nullptr;  // binding 0 = combined sampler
  rhi::Sampler* sampler = nullptr;  // default, for an unresolved guest S#
  // The same default for a binding whose view is an integer format, which
  // may not be filtered at all.
  rhi::Sampler* sampler_nearest = nullptr;

  // Multi-texture (a recomp PS sampling >1 texture, e.g. Doom64's 3D walls):
  // pools for the per-draw sets, plus a 1x1 white default for any binding we
  // could not resolve, so diffuse*lightmap with a missing map shows the
  // diffuse instead of going black.
  rhi::Texture* white_img = nullptr;
  // A binding a shader declares as Dim3D cannot be satisfied by a 2D view, so
  // the default has a 1x1x1 volume twin.
  rhi::Texture* white_3d_img = nullptr;
  rhi::TextureView* white_view = nullptr;
  // The same texel through an integer view, for a binding the module declared
  // as an integer sampler.
  rhi::TextureView* white_uint_view = nullptr;
  rhi::TextureView* white_uint_array_view = nullptr;
  rhi::TextureView* white_uint_3d_view = nullptr;
  rhi::TextureView* white_array_view = nullptr;
  rhi::TextureView* white_3d_view = nullptr;
  rhi::TextureView* zero_view = nullptr;
  rhi::TextureView* zero_array_view = nullptr;
  rhi::TextureView* zero_3d_view = nullptr;
  // image_sample_c compares against the sampled value, which is defined only
  // on depth-comparison formats; bound to a colour surface there is nothing
  // valid, so bind this 1x1 D32 image at the far plane (reads as "nothing
  // occludes").
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

// A one-binding group (g_tex.layout) sampling `view` through `sampler`.
rhi::BindGroup* SampledTextureGroup(
    rhi::TextureView* view,
    rhi::Sampler* sampler,
    rhi::TextureState state = rhi::TextureState::kShaderRead);

// Upload (or reuse) a guest texture; returns a group bound to it, or null.
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
// Anything that can change what a texture binding resolves to (compute
// writes and writebacks, fills, invalidations, eviction) calls this; it
// retires TexViewFor's per-binding memo.
void BumpTextureEpoch();
u64 TextureEpoch();

// N-sampler group (set 0) for a recomp PS. `num_bindings` is what the layout
// declares, which is not always what the draw resolved textures for.
rhi::BindGroup* GetMultiTexSet(const render::DrawInfo& d,
                               rhi::BindGroupLayout* set_layout,
                               u32 num_bindings,
                               rhi::TextureView* const* resolved_views,
                               const rhi::TextureState* resolved_layouts,
                               // Per binding: the format its view was created
                               // with, or rhi::Format::kUndefined where the T#
                               // describes it. An integer one may not filter.
                               const rhi::Format* resolved_formats,
                               // Per binding: the depth target it resolved to,
                               // or 0. A depth comparison is only defined on a
                               // view whose format supports it.
                               const u64* depth_src = nullptr);

// Drop cached textures overlapping a range a compute dispatch wrote.
void InvalidateTexRange(u64 base, u64 size);

// Destroy objects retired two frames ago; called once per BeginFrame.
void ReleaseRetiredTextures();
// Drop every cached multi-texture descriptor set (a view they name is going).
void ClearMultiTexCache();

}  // namespace gpu::render
