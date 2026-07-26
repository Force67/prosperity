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

#include "rhi/command.h"

namespace gpu::vk {

// Sampler bindings a recompiled PS may consume in one draw.
constexpr uint32_t kMaxTex = 16;

// Descriptor infrastructure shared by every sampled image (guest textures,
// render targets sampled as textures, the 1x1 white fallback).
struct TextureBindings {
  VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;  // binding 0 = combined sampler
  VkDescriptorPool dsPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorPool> dsPools;
  VkSampler sampler = VK_NULL_HANDLE;  // default, for an unresolved guest S#

  // Multi-texture (a recomp PS sampling >1 texture, e.g. Doom64's 3D walls): a
  // kMaxTex-binding set-0 layout and its pools, plus a 1x1 white default for
  // any binding we could not resolve -- so diffuse*lightmap with a missing map
  // shows the diffuse instead of going black.
  VkDescriptorSetLayout texArrayLayout = VK_NULL_HANDLE;
  VkDescriptorPool mtexPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorPool> mtexPools;
  VkImage whiteImg = VK_NULL_HANDLE;
  VkDeviceMemory whiteMem = VK_NULL_HANDLE;
  VkImageView whiteView = VK_NULL_HANDLE;
  VkImageView whiteArrayView = VK_NULL_HANDLE;
  VkDescriptorSet whiteSet = VK_NULL_HANDLE;
  VkDescriptorSet whiteArraySet = VK_NULL_HANDLE;
};

extern TextureBindings g_tex;

bool createTextureDescriptors();

// A descriptor set from the pool chain, growing it when every pool is full.
VkDescriptorSet allocateSamplerSet(VkDescriptorSetLayout layout, bool multi,
                                   VkDescriptorPool &owner);

// Upload (or reuse) a guest texture; returns a set bound to it, or null.
VkDescriptorSet getTexture(uint64_t base, uint32_t w, uint32_t h, uint32_t dfmt,
                           uint32_t nfmt, uint32_t tiling = 8,
                           uint32_t pitch = 0, uint32_t layers = 1,
                           uint32_t base_array = 0, uint32_t view_layers = 1,
                           uint32_t mip_levels = 1, uint32_t base_mip = 0,
                           uint32_t view_mips = 1, uint32_t min_lod = 0,
                           bool pow2_pad = false,
                           const uint32_t *sampler = nullptr,
                           bool sampler_valid = false, bool arrayed = false,
                           bool force_lod_zero = false,
                           bool depth_compare = false, uint32_t swizzle = 0);
bool guestTextureUploadSupported(uint32_t dfmt, uint32_t nfmt);
VkImageView texViewFor(const rhi::DrawInfo::DrawTex &t);

// N-sampler descriptor set (set 0, bindings 0..nTexs-1) for a recomp PS.
VkDescriptorSet getMultiTexSet(const rhi::DrawInfo &d,
                               VkDescriptorSetLayout setLayout,
                               const VkImageView *resolvedViews,
                               const VkImageLayout *resolvedLayouts);

// Drop cached textures overlapping a range a compute dispatch wrote.
void invalidateTexRange(uint64_t base, uint64_t size);

// Destroy objects retired two frames ago; called once per beginFrame.
void releaseRetiredTextures();

}  // namespace gpu::vk
