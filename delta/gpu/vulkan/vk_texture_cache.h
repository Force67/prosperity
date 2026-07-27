/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Guest textures as Vulkan images: the descriptor infrastructure every sampled
// image binds through, and the image/view/sampler/descriptor-set caches keyed
// by the guest T#. A cached surface is revalidated against a content hash, so a
// buffer the guest rewrites is re-uploaded rather than served stale.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

#include "gpu/rhi/command.h"

namespace gpu::vk {

// Sampler bindings a recompiled PS may consume in one draw.
constexpr uint32_t kMaxTex = 16;

// Descriptor infrastructure shared by every sampled image (guest textures,
// render targets sampled as textures, the 1x1 white fallback).
struct TextureBindings {
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
  VkDeviceMemory white_mem = VK_NULL_HANDLE;
  VkImageView white_view = VK_NULL_HANDLE;
  VkImageView white_array_view = VK_NULL_HANDLE;
  VkDescriptorSet white_set = VK_NULL_HANDLE;
  VkDescriptorSet white_array_set = VK_NULL_HANDLE;
};

extern TextureBindings g_tex;

bool CreateTextureDescriptors();

// A descriptor set from the pool chain, growing it when every pool is full.
VkDescriptorSet AllocateSamplerSet(VkDescriptorSetLayout layout,
                                   bool multi,
                                   VkDescriptorPool& owner);

// Upload (or reuse) a guest texture; returns a set bound to it, or null.
VkDescriptorSet GetTexture(uint64_t base,
                           uint32_t w,
                           uint32_t h,
                           uint32_t dfmt,
                           uint32_t nfmt,
                           uint32_t tiling = 8,
                           uint32_t pitch = 0,
                           uint32_t layers = 1,
                           uint32_t base_array = 0,
                           uint32_t view_layers = 1,
                           uint32_t mip_levels = 1,
                           uint32_t base_mip = 0,
                           uint32_t view_mips = 1,
                           uint32_t min_lod = 0,
                           bool pow2_pad = false,
                           const uint32_t* sampler = nullptr,
                           bool sampler_valid = false,
                           bool arrayed = false,
                           bool force_lod_zero = false,
                           bool depth_compare = false,
                           uint32_t swizzle = 0);
bool GuestTextureUploadSupported(uint32_t dfmt, uint32_t nfmt);
VkImageView TexViewFor(const rhi::DrawInfo::DrawTex& t);

// N-sampler descriptor set (set 0, bindings 0..num_texs-1) for a recomp PS.
VkDescriptorSet GetMultiTexSet(const rhi::DrawInfo& d,
                               VkDescriptorSetLayout set_layout,
                               const VkImageView* resolved_views,
                               const VkImageLayout* resolved_layouts);

// Drop cached textures overlapping a range a compute dispatch wrote.
void InvalidateTexRange(uint64_t base, uint64_t size);

// Destroy objects retired two frames ago; called once per BeginFrame.
void ReleaseRetiredTextures();

}  // namespace gpu::vk
