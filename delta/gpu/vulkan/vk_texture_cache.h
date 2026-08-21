/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Guest textures as Vulkan images: the descriptor infrastructure every sampled
// image binds through, and the image/view/sampler/descriptor-set caches keyed
// by the guest T#. A cached surface is revalidated against a content hash, so a
// buffer the guest rewrites is re-uploaded rather than served stale.

#include <vulkan/vulkan.h>
#include "base/arch.h"

#include <vector>

#include "gpu/rhi/command.h"
#include "gpu/vulkan/vk_memory.h"

namespace gpu::vk {

// Sampler bindings a recompiled PS may consume in one draw.
// A pixel shader's sampler bindings. Tomb Raider's lighting pass declares 23,
// and a shader that needs more than this is declined outright, so the limit is
// a rendering cliff rather than a tuning knob. 24 stays far inside what every
// desktop and mobile driver reports for maxPerStageDescriptorSampledImages,
// which is well above the 16 Vulkan merely guarantees.
constexpr u32 kMaxTex = 24;

// Descriptor infrastructure shared by every sampled image (guest textures,
// render targets sampled as textures, the 1x1 white fallback).
struct TextureBindings {
  bool descriptors_ready = false;
  VkDescriptorSetLayout ds_layout =
      VK_NULL_HANDLE;  // binding 0 = combined sampler
  VkDescriptorPool ds_pool = VK_NULL_HANDLE;
  std::vector<VkDescriptorPool> ds_pools;
  VkSampler sampler = VK_NULL_HANDLE;  // default, for an unresolved guest S#

  // Multi-texture (a recomp PS sampling >1 texture, e.g. Doom64's 3D walls): a
  // kMaxTex-binding set-0 layout and its pools, plus a 1x1 white default for
  // any binding we could not resolve -- so diffuse*lightmap with a missing map
  // shows the diffuse instead of going black.
  VkDescriptorSetLayout tex_array_layout = VK_NULL_HANDLE;
  VkDescriptorPool mtex_pool = VK_NULL_HANDLE;
  std::vector<VkDescriptorPool> mtex_pools;
  VkImage white_img = VK_NULL_HANDLE;
  ImageAllocation white_allocation;
  // A binding a shader declares as Dim3D cannot be satisfied by a 2D view, so
  // the default has a 1x1x1 volume twin.
  VkImage white_3d_img = VK_NULL_HANDLE;
  ImageAllocation white_3d_allocation;
  VkImageView white_view = VK_NULL_HANDLE;
  VkImageView white_array_view = VK_NULL_HANDLE;
  VkImageView white_3d_view = VK_NULL_HANDLE;
  VkImageView zero_view = VK_NULL_HANDLE;
  VkImageView zero_array_view = VK_NULL_HANDLE;
  VkImageView zero_3d_view = VK_NULL_HANDLE;
  // A shader that does image_sample_c compares against the sampled value, and
  // Vulkan only defines that on a format supporting depth comparison. When such
  // a binding resolves to a colour surface -- which every guest texture and
  // every colour target is -- there is nothing valid to bind, so bind this: a
  // 1x1 D32 image holding the far plane, which reads as "nothing occludes this"
  // rather than as undefined.
  VkImage depth_default_img = VK_NULL_HANDLE;
  ImageAllocation depth_default_allocation;
  VkImageView depth_default_view = VK_NULL_HANDLE;
  VkDescriptorSet white_set = VK_NULL_HANDLE;
  VkDescriptorSet white_array_set = VK_NULL_HANDLE;
  VkDescriptorSet white_3d_set = VK_NULL_HANDLE;
  VkDescriptorSet zero_set = VK_NULL_HANDLE;
  VkDescriptorSet zero_array_set = VK_NULL_HANDLE;
  VkDescriptorSet zero_3d_set = VK_NULL_HANDLE;
};

extern TextureBindings& g_tex;

bool CreateTextureDescriptors();

// A descriptor set from the pool chain, growing it when every pool is full.
VkDescriptorSet AllocateSamplerSet(VkDescriptorSetLayout layout,
                                   bool multi,
                                   VkDescriptorPool& owner);

// Upload (or reuse) a guest texture; returns a set bound to it, or null.
VkDescriptorSet GetTexture(u64 base,
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
VkImageView TexViewFor(const rhi::DrawInfo::DrawTex& t);

// N-sampler descriptor set (set 0) for a recomp PS. `num_bindings` is what the
// set layout declares, which is not always what the draw resolved textures for.
VkDescriptorSet GetMultiTexSet(const rhi::DrawInfo& d,
                               VkDescriptorSetLayout set_layout,
                               u32 num_bindings,
                               const VkImageView* resolved_views,
                               const VkImageLayout* resolved_layouts,
                               // Per binding: the depth target it
                               // resolved to, or 0. A depth
                               // comparison is only defined on a
                               // view whose format supports it.
                               const u64* depth_src = nullptr);

// Drop cached textures overlapping a range a compute dispatch wrote.
void InvalidateTexRange(u64 base, u64 size);

// Destroy objects retired two frames ago; called once per BeginFrame.
void ReleaseRetiredTextures();

}  // namespace gpu::vk
