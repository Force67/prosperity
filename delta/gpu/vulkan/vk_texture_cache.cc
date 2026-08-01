/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_texture_cache.h"

#include "gpu/gpu_check.h"
#include "gpu/guest_memory.h"
#include "gpu/gcn/gcn_detile.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/rhi/renderer.h"
#include "gpu/vulkan/vk_capture.h"
#include "gpu/vulkan/vk_debug.h"
#include "gpu/vulkan/vk_device.h"
#include "gpu/vulkan/vk_format.h"
#include "gpu/vulkan/vk_frame.h"
#include "gpu/vulkan/vk_hash.h"
#include "gpu/vulkan/vk_memory.h"
#include "gpu/vulkan/vk_perf.h"
#include "gpu/vulkan/vk_render_target.h"
#include "gpu/vulkan/vk_upload_ring.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utl/options.h>

namespace {
DELTA_OPTION(int, kForceTile, "DELTA_GPU_FORCETILE", -1);
DELTA_OPTION(uint32_t, kDumpMin, "DELTA_GPU_TEXDUMP_MIN", 128);
DELTA_OPTION(uint64_t, kTextureMb, "DELTA_GPU_TEX_MB", 1024);
DELTA_OPTION(bool, kDefaultSampler, "DELTA_GPU_DEFSAMPLER", false);
DELTA_OPTION(bool, kDumpUpload, "DELTA_GPU_TEXDUMP_UPLOAD", false);
DELTA_OPTION(bool, kForceWhite, "DELTA_GPU_FORCEWHITE", false);
DELTA_OPTION(bool, kNoDetile, "DELTA_GPU_NODETILE", false);
DELTA_OPTION(bool, kTexDump, "DELTA_GPU_TEXDUMP", false);
DELTA_OPTION(bool, kTexFail, "DELTA_GPU_TEXFAIL", false);
DELTA_OPTION(bool, kTexMiss, "DELTA_GPU_TEXMISS", false);
DELTA_OPTION(bool, kTexRaw, "DELTA_GPU_TEXRAW", false);
DELTA_OPTION(uint64_t, kTexWatch, "DELTA_GPU_TEXWATCH", 0);
}  // namespace

namespace gpu::vk {

using rhi::DrawInfo;

struct TexImageKey {
  uint64_t base = 0;
  uint32_t w = 0, h = 0, tiling = 8, pitch = 0, layers = 1;
  uint32_t depth = 1;
  uint32_t mip_levels = 1;
  VkFormat format = VK_FORMAT_UNDEFINED;
  bool pow2_pad = false;
  // A volume and a 2D surface at the same guest address are different VkImage
  // types, so is_3d has to separate them here.
  bool is_3d = false;
  bool operator==(const TexImageKey& o) const {
    return base == o.base && w == o.w && h == o.h && tiling == o.tiling &&
           pitch == o.pitch && layers == o.layers && depth == o.depth &&
           mip_levels == o.mip_levels && format == o.format &&
           pow2_pad == o.pow2_pad && is_3d == o.is_3d;
  }
};

struct TexImageKeyHash {
  size_t operator()(const TexImageKey& k) const {
    uint64_t h = 1469598103934665603ull;
    h = HashWord(h, k.base);
    h = HashWord(h, k.w);
    h = HashWord(h, k.h);
    h = HashWord(h, k.tiling);
    h = HashWord(h, k.pitch);
    h = HashWord(h, k.layers);
    h = HashWord(h, k.depth);
    h = HashWord(h, k.mip_levels);
    h = HashWord(h, k.pow2_pad);
    h = HashWord(h, k.is_3d);
    h = HashWord(h, k.format);
    return static_cast<size_t>(h);
  }
};

struct SamplerKey {
  uint32_t raw[4] = {};
  uint32_t image_min_lod = 0;
  bool valid = false;
  bool force_lod_zero = false;
  bool depth_compare = false;
  bool operator==(const SamplerKey& o) const {
    return valid == o.valid && image_min_lod == o.image_min_lod &&
           force_lod_zero == o.force_lod_zero &&
           depth_compare == o.depth_compare &&
           std::memcmp(raw, o.raw, sizeof(raw)) == 0;
  }
};

struct SamplerKeyHash {
  size_t operator()(const SamplerKey& k) const {
    uint64_t h = HashWord(1469598103934665603ull, k.valid);
    for (uint32_t word : k.raw)
      h = HashWord(h, word);
    h = HashWord(h, k.image_min_lod);
    h = HashWord(h, k.force_lod_zero);
    h = HashWord(h, k.depth_compare);
    return static_cast<size_t>(h);
  }
};

struct TexKey {
  TexImageKey image;
  uint32_t base_array = 0, view_layers = 1;
  uint32_t base_mip = 0, view_mips = 1;
  uint32_t swizzle = 0;  // packed T# DST_SEL (0 = identity)
  SamplerKey sampler;
  bool arrayed = false;
  bool operator==(const TexKey& o) const {
    return swizzle == o.swizzle && image == o.image &&
           base_array == o.base_array && view_layers == o.view_layers &&
           base_mip == o.base_mip && view_mips == o.view_mips &&
           sampler == o.sampler && arrayed == o.arrayed;
  }
};

struct TexKeyHash {
  size_t operator()(const TexKey& k) const {
    uint64_t h = TexImageKeyHash{}(k.image);
    h = HashWord(h, k.base_array);
    h = HashWord(h, k.view_layers);
    h = HashWord(h, k.base_mip);
    h = HashWord(h, k.view_mips);
    h = HashWord(h, k.swizzle);
    h = HashWord(h, SamplerKeyHash{}(k.sampler));
    return static_cast<size_t>(HashWord(h, k.arrayed));
  }
};

struct TexViewKey {
  TexImageKey image;
  uint32_t base_array = 0, view_layers = 1;
  uint32_t base_mip = 0, view_mips = 1;
  uint32_t swizzle = 0;  // packed T# DST_SEL_X/Y/Z/W
  bool arrayed = false;
  bool operator==(const TexViewKey& o) const {
    return swizzle == o.swizzle && image == o.image &&
           base_array == o.base_array && view_layers == o.view_layers &&
           base_mip == o.base_mip && view_mips == o.view_mips &&
           arrayed == o.arrayed;
  }
};

struct TexViewKeyHash {
  size_t operator()(const TexViewKey& k) const {
    uint64_t h = TexImageKeyHash{}(k.image);
    h = HashWord(h, k.base_array);
    h = HashWord(h, k.view_layers);
    h = HashWord(h, k.base_mip);
    h = HashWord(h, k.view_mips);
    h = HashWord(h, k.swizzle);
    return static_cast<size_t>(HashWord(h, k.arrayed));
  }
};

struct TexImageEntry {
  VkImage image = VK_NULL_HANDLE;
  ImageAllocation allocation;
  uint64_t footprint = 0;
  VkDeviceSize allocation_size = 0;
  uint64_t hash = 0;
  int last_checked_frame = -1;
  int last_used_frame = -1;
};

struct TexViewEntry {
  VkImageView view = VK_NULL_HANDLE;
};
struct TexEntry {
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
};

std::unordered_map<TexImageKey, TexImageEntry, TexImageKeyHash> g_tex_images;
std::unordered_map<TexViewKey, TexViewEntry, TexViewKeyHash> g_tex_views;
std::unordered_map<TexKey, TexEntry, TexKeyHash> g_tex_cache;
std::unordered_map<SamplerKey, VkSampler, SamplerKeyHash> g_sampler_cache;
constexpr uint32_t kTexturePageShift = 16;
std::unordered_map<uint64_t, std::vector<TexImageKey>> g_texture_pages;
uint64_t g_tex_image_bytes = 0;
std::vector<TexImageEntry> g_retired_tex_images;
std::vector<TexViewEntry> g_retired_tex_views;
std::vector<TexEntry> g_retired_tex_sets;

void RegisterTexturePages(const TexImageKey& key, uint64_t bytes) {
  const uint64_t end = key.base + bytes;
  for (uint64_t page = key.base >> kTexturePageShift;
       page <= (end - 1) >> kTexturePageShift; page++)
    g_texture_pages[page].push_back(key);
}

void UnregisterTexturePages(const TexImageKey& key, uint64_t bytes) {
  const uint64_t end = key.base + bytes;
  for (uint64_t page = key.base >> kTexturePageShift;
       page <= (end - 1) >> kTexturePageShift; page++) {
    auto found = g_texture_pages.find(page);
    if (found == g_texture_pages.end())
      continue;
    auto& keys = found->second;
    keys.erase(std::remove(keys.begin(), keys.end(), key), keys.end());
    if (keys.empty())
      g_texture_pages.erase(found);
  }
}
TexKey TextureKey(uint64_t base,
                  uint32_t w,
                  uint32_t h,
                  uint32_t dfmt,
                  uint32_t nfmt,
                  uint32_t tiling,
                  uint32_t pitch,
                  uint32_t layers,
                  uint32_t base_array,
                  uint32_t view_layers,
                  uint32_t mip_levels,
                  uint32_t base_mip,
                  uint32_t view_mips,
                  uint32_t min_lod,
                  bool pow2_pad,
                  const uint32_t* sampler,
                  bool sampler_valid,
                  bool arrayed,
                  bool force_lod_zero,
                  bool depth_compare,
                  uint32_t swizzle,
                  uint32_t depth,
                  bool is_3d) {
  if (is_3d)
    layers = 1;
  else
    depth = 1;
  if (layers && base_array < layers)
    view_layers = std::min(view_layers, layers - base_array);
  if (!arrayed)
    view_layers = 1;
  if (mip_levels && base_mip < mip_levels)
    view_mips = std::min(view_mips, mip_levels - base_mip);
  TexKey key;
  key.image = {base,   w,          h,
               tiling, pitch,      layers,
               depth,  mip_levels, GuestTextureFormat(dfmt, nfmt),
               pow2_pad, is_3d};
  key.base_array = base_array;
  key.view_layers = view_layers;
  key.base_mip = base_mip;
  key.view_mips = view_mips;
  key.swizzle = swizzle;
  key.sampler.valid = sampler_valid && sampler;
  if (key.sampler.valid)
    std::memcpy(key.sampler.raw, sampler, sizeof(key.sampler.raw));
  key.sampler.image_min_lod = min_lod;
  key.sampler.force_lod_zero = force_lod_zero;
  key.sampler.depth_compare = depth_compare;
  key.arrayed = arrayed;
  return key;
}

TexViewKey TextureViewKey(const TexKey& key) {
  return {key.image,     key.base_array, key.view_layers, key.base_mip,
          key.view_mips, key.swizzle,    key.arrayed};
}

uint32_t TextureTiling(uint32_t tiling) {
  return kForceTile >= 0 ? static_cast<uint32_t>(kForceTile.get()) : tiling;
}

bool UploadTexPixelsImmediate(VkImage img,
                              uint64_t base,
                              const gcn::TextureLayout32& layout,
                              uint32_t texel_w,
                              uint32_t texel_h,
                              bool is_3d = false);  // defined below
void ClearMultiTexCache();

bool CreateTextureDescriptors() {
  if (g_tex.descriptors_ready)
    return true;
  if (g_tex.ds_layout)
    return false;
  // descriptor set layout: binding 0 = combined image sampler (fragment).
  VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dl{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dl.bindingCount = 1;
  dl.pBindings = &b;
  VKOK(vkCreateDescriptorSetLayout(g_dev.device, &dl, nullptr,
                                   &g_tex.ds_layout));

  constexpr uint32_t kSinglePoolSets = 1024;
  constexpr uint32_t kMultiPoolSets = 512;
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                          kSinglePoolSets};
  VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dp.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  dp.maxSets = kSinglePoolSets;
  dp.poolSizeCount = 1;
  dp.pPoolSizes = &ps;
  VKOK(vkCreateDescriptorPool(g_dev.device, &dp, nullptr, &g_tex.ds_pool));
  g_tex.ds_pools.push_back(g_tex.ds_pool);

  VkSamplerCreateInfo sc{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sc.magFilter = sc.minFilter = VK_FILTER_LINEAR;
  sc.addressModeU = sc.addressModeV = sc.addressModeW =
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  VKOK(vkCreateSampler(g_dev.device, &sc, nullptr, &g_tex.sampler));

  // Multi-texture path: a 16-binding set-0 layout + a pool, used only by recomp
  // PS that sample >1 texture (single-texture draws keep the 1-binding
  // ds_layout/ds_pool).
  {
    VkDescriptorSetLayoutBinding mb[kMaxTex];
    for (uint32_t i = 0; i < kMaxTex; i++)
      mb[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
               VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo ml{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ml.bindingCount = kMaxTex;
    ml.pBindings = mb;
    VKOK(vkCreateDescriptorSetLayout(g_dev.device, &ml, nullptr,
                                     &g_tex.tex_array_layout));
    VkDescriptorPoolSize mps[2] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMultiPoolSets * kMaxTex},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMultiPoolSets * kMaxTex},
    };
    VkDescriptorPoolCreateInfo mp{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    mp.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    mp.maxSets = kMultiPoolSets;
    mp.poolSizeCount = 2;
    mp.pPoolSizes = mps;
    VKOK(vkCreateDescriptorPool(g_dev.device, &mp, nullptr, &g_tex.mtex_pool));
    g_tex.mtex_pools.push_back(g_tex.mtex_pool);

    // 1x1 white default texture (for unresolved sampler bindings).
    VkImageCreateInfo wi{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    wi.imageType = VK_IMAGE_TYPE_2D;
    wi.format = VK_FORMAT_R8G8B8A8_UNORM;
    wi.extent = {1, 1, 1};
    wi.mipLevels = 1;
    wi.arrayLayers = 1;
    wi.samples = VK_SAMPLE_COUNT_1_BIT;
    wi.tiling = VK_IMAGE_TILING_OPTIMAL;
    wi.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VKOK(vkCreateImage(g_dev.device, &wi, nullptr, &g_tex.white_img));
    if (!g_image_memory.Allocate(g_dev, g_tex.white_img,
                                 g_tex.white_allocation)) {
      vkDestroyImage(g_dev.device, g_tex.white_img, nullptr);
      g_tex.white_img = VK_NULL_HANDLE;
      return false;
    }
    uint32_t white = 0xFFFFFFFFu;
    gcn::TextureLayout32 white_layout;
    gcn::BuildTextureLayout32(white_layout, 1, 1, 1, 1, 1, 31, false);
    if (!UploadTexPixelsImmediate(g_tex.white_img,
                                  reinterpret_cast<uint64_t>(&white),
                                  white_layout, 1, 1)) {
      vkDestroyImage(g_dev.device, g_tex.white_img, nullptr);
      g_image_memory.Free(g_dev, g_tex.white_allocation);
      g_tex.white_img = VK_NULL_HANDLE;
      return false;
    }
    // Same default for a Dim3D binding, which cannot sample a 2D view.
    VkImageCreateInfo wi3 = wi;
    wi3.imageType = VK_IMAGE_TYPE_3D;
    VKOK(vkCreateImage(g_dev.device, &wi3, nullptr, &g_tex.white_3d_img));
    if (!g_image_memory.Allocate(g_dev, g_tex.white_3d_img,
                                 g_tex.white_3d_allocation)) {
      vkDestroyImage(g_dev.device, g_tex.white_3d_img, nullptr);
      g_tex.white_3d_img = VK_NULL_HANDLE;
      return false;
    }
    if (!UploadTexPixelsImmediate(g_tex.white_3d_img,
                                  reinterpret_cast<uint64_t>(&white),
                                  white_layout, 1, 1, true)) {
      vkDestroyImage(g_dev.device, g_tex.white_3d_img, nullptr);
      g_image_memory.Free(g_dev, g_tex.white_3d_allocation);
      g_tex.white_3d_img = VK_NULL_HANDLE;
      return false;
    }
    VkImageViewCreateInfo wv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    wv.image = g_tex.white_img;
    wv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    wv.format = VK_FORMAT_R8G8B8A8_UNORM;
    wv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VKOK(vkCreateImageView(g_dev.device, &wv, nullptr, &g_tex.white_view));
    wv.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    VKOK(
        vkCreateImageView(g_dev.device, &wv, nullptr, &g_tex.white_array_view));
    wv.components = {VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO,
                     VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO};
    wv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    VKOK(vkCreateImageView(g_dev.device, &wv, nullptr, &g_tex.zero_view));
    wv.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    VKOK(
        vkCreateImageView(g_dev.device, &wv, nullptr, &g_tex.zero_array_view));
    wv.image = g_tex.white_3d_img;
    wv.viewType = VK_IMAGE_VIEW_TYPE_3D;
    VKOK(vkCreateImageView(g_dev.device, &wv, nullptr, &g_tex.zero_3d_view));
    wv.components = {};
    VKOK(vkCreateImageView(g_dev.device, &wv, nullptr, &g_tex.white_3d_view));

    VkDescriptorSetLayout layouts[6] = {
        g_tex.ds_layout, g_tex.ds_layout, g_tex.ds_layout,
        g_tex.ds_layout, g_tex.ds_layout, g_tex.ds_layout};
    VkDescriptorSet sets[6];
    VkDescriptorSetAllocateInfo wa{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    wa.descriptorPool = g_tex.ds_pool;
    wa.descriptorSetCount = 6;
    wa.pSetLayouts = layouts;
    VKOK(vkAllocateDescriptorSets(g_dev.device, &wa, sets));
    g_tex.white_set = sets[0];
    g_tex.white_array_set = sets[1];
    g_tex.zero_set = sets[2];
    g_tex.zero_array_set = sets[3];
    g_tex.white_3d_set = sets[4];
    g_tex.zero_3d_set = sets[5];
    VkDescriptorImageInfo infos[6] = {
        {g_tex.sampler, g_tex.white_view,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {g_tex.sampler, g_tex.white_array_view,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {g_tex.sampler, g_tex.zero_view,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {g_tex.sampler, g_tex.zero_array_view,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {g_tex.sampler, g_tex.white_3d_view,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {g_tex.sampler, g_tex.zero_3d_view,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    VkWriteDescriptorSet writes[6];
    for (uint32_t i = 0; i < 6; i++) {
      writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      writes[i].dstSet = sets[i];
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[i].pImageInfo = &infos[i];
    }
    vkUpdateDescriptorSets(g_dev.device, 6, writes, 0, nullptr);
  }
  g_tex.descriptors_ready = true;
  return true;
}

VkDescriptorSet AllocateSamplerSet(VkDescriptorSetLayout layout,
                                   bool multi,
                                   VkDescriptorPool& owner) {
  auto& pools = multi ? g_tex.mtex_pools : g_tex.ds_pools;
  for (auto it = pools.rbegin(); it != pools.rend(); ++it) {
    const VkDescriptorPool pool = *it;
    VkDescriptorSet set;
    VkDescriptorSetAllocateInfo da{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    da.descriptorPool = pool;
    da.descriptorSetCount = 1;
    da.pSetLayouts = &layout;
    const VkResult result = vkAllocateDescriptorSets(g_dev.device, &da, &set);
    if (result == VK_SUCCESS) {
      owner = pool;
      return set;
    }
    if (result != VK_ERROR_OUT_OF_POOL_MEMORY &&
        result != VK_ERROR_FRAGMENTED_POOL)
      return VK_NULL_HANDLE;
  }

  const uint32_t set_capacity = multi ? 512u : 1024u;
  VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       set_capacity * (multi ? kMaxTex : 1u)},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, set_capacity * (multi ? kMaxTex : 1u)},
  };
  VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  ci.maxSets = set_capacity;
  ci.poolSizeCount = multi ? 2u : 1u;
  ci.pPoolSizes = sizes;
  VkDescriptorPool pool;
  if (vkCreateDescriptorPool(g_dev.device, &ci, nullptr, &pool) != VK_SUCCESS)
    return VK_NULL_HANDLE;
  pools.push_back(pool);
  VkDescriptorSet set;
  VkDescriptorSetAllocateInfo da{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  da.descriptorPool = pool;
  da.descriptorSetCount = 1;
  da.pSetLayouts = &layout;
  if (vkAllocateDescriptorSets(g_dev.device, &da, &set) != VK_SUCCESS) {
    pools.pop_back();
    vkDestroyDescriptorPool(g_dev.device, pool, nullptr);
    return VK_NULL_HANDLE;
  }
  owner = pool;
  return set;
}

VkSampler SamplerFor(const SamplerKey& key) {
  // DELTA_GPU_DEFSAMPLER: ignore every guest S# and use the default sampler,
  // to tell a mis-decoded sampler apart from a mis-bound image.
  if (kDefaultSampler)
    return g_tex.sampler;
  if (!key.valid && !key.force_lod_zero && !key.depth_compare)
    return g_tex.sampler;
  auto found = g_sampler_cache.find(key);
  if (found != g_sampler_cache.end())
    return found->second;
  if (g_sampler_cache.size() >= 4096)
    return g_tex.sampler;

  auto address_mode = [](uint32_t mode) {
    switch (mode & 7) {
      case 0:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
      case 1:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
      case 3:
      case 5:
      case 7:
        return g_dev.sampler_mirror_clamp
                   ? VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE
                   : VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
      case 4:
      case 6:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
      default:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
  };
  VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  ci.addressModeU = address_mode(key.raw[0]);
  ci.addressModeV = address_mode(key.raw[0] >> 3);
  ci.addressModeW = address_mode(key.raw[0] >> 6);
  uint32_t mag = (key.raw[2] >> 20) & 3;
  uint32_t min = (key.raw[2] >> 22) & 3;
  ci.magFilter = (mag & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  ci.minFilter = (min & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  uint32_t mip_filter = (key.raw[2] >> 26) & 3;
  ci.mipmapMode = mip_filter == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                  : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  if (mip_filter) {
    uint32_t min_lod = std::max(key.raw[1] & 0xFFF, key.image_min_lod);
    ci.minLod = static_cast<float>(min_lod) / 256.0f;
    ci.maxLod = static_cast<float>((key.raw[1] >> 12) & 0xFFF) / 256.0f;
    ci.maxLod = std::max(ci.minLod, ci.maxLod);
  }
  if (key.force_lod_zero)
    ci.minLod = ci.maxLod = 0.0f;
  int32_t bias = static_cast<int32_t>(key.raw[2] << 18) >> 18;
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(g_dev.phys, &props);
  ci.mipLodBias = std::clamp(static_cast<float>(bias) / 256.0f,
                             -props.limits.maxSamplerLodBias,
                             props.limits.maxSamplerLodBias);
  switch ((key.raw[3] >> 30) & 3) {
    case 1:
      ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
      break;
    case 2:
      ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
      break;
    default:
      ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
      break;
  }
  ci.compareEnable = key.depth_compare;
  switch ((key.raw[0] >> 12) & 7) {
    case 0:
      ci.compareOp = VK_COMPARE_OP_NEVER;
      break;
    case 1:
      ci.compareOp = VK_COMPARE_OP_LESS;
      break;
    case 2:
      ci.compareOp = VK_COMPARE_OP_EQUAL;
      break;
    case 3:
      ci.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
      break;
    case 4:
      ci.compareOp = VK_COMPARE_OP_GREATER;
      break;
    case 5:
      ci.compareOp = VK_COMPARE_OP_NOT_EQUAL;
      break;
    case 6:
      ci.compareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
      break;
    default:
      ci.compareOp = VK_COMPARE_OP_ALWAYS;
      break;
  }
  if (g_dev.sampler_anisotropy && (mag >= 2 || min >= 2)) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_dev.phys, &props);
    ci.anisotropyEnable = VK_TRUE;
    ci.maxAnisotropy =
        std::min(static_cast<float>(1u << ((key.raw[0] >> 9) & 7)),
                 props.limits.maxSamplerAnisotropy);
  }
  VkSampler sampler = VK_NULL_HANDLE;
  if (vkCreateSampler(g_dev.device, &ci, nullptr, &sampler) != VK_SUCCESS)
    return g_tex.sampler;
  g_sampler_cache.emplace(key, sampler);
  return sampler;
}

uint64_t TextureLinearBytes(const gcn::TextureLayout32& layout) {
  uint64_t bytes = 0;
  for (uint32_t mip = 0; mip < layout.mip_levels; mip++)
    bytes += static_cast<uint64_t>(layout.mips[mip].width) *
             layout.mips[mip].height * layout.layers * layout.elem_bytes;
  return bytes;
}

void PackTexPixels(uint8_t* linear,
                   VkDeviceSize buffer_offset,
                   uint64_t base,
                   const gcn::TextureLayout32& layout,
                   uint32_t texel_w,
                   uint32_t texel_h,
                   VkBufferImageCopy* copies,
                   bool is_3d) {
  const uint32_t elem = layout.elem_bytes;
  const uint8_t* src = reinterpret_cast<const uint8_t*>(base);
  uint64_t linear_offset = 0;
  for (uint32_t mip = 0; mip < layout.mip_levels; mip++) {
    const auto& level = layout.mips[mip];
    copies[mip].bufferOffset = buffer_offset + linear_offset;
    // A volume's slices are the layout's layers, but Vulkan takes them as one
    // copy of `depth` z-slices into a single-layer image.
    copies[mip].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0,
                                    is_3d ? 1u : layout.layers};
    copies[mip].imageExtent = {std::max(texel_w >> mip, 1u),
                               std::max(texel_h >> mip, 1u),
                               is_3d ? layout.layers : 1u};
    const uint64_t layer_bytes =
        static_cast<uint64_t>(level.width) * level.height * elem;
    for (uint32_t layer = 0; layer < layout.layers; layer++) {
      uint8_t* dst = linear + linear_offset + layer * layer_bytes;
      if (!kNoDetile) {
        gcn::DetileTextureMip32(src, dst, layout, mip, layer);
      } else {
        const uint8_t* level_src = src + level.offset +
                                   static_cast<uint64_t>(layer) * level.pitch *
                                       level.stored_height * elem;
        for (uint32_t y = 0; y < level.height; y++)
          std::memcpy(dst + static_cast<size_t>(y) * level.width * elem,
                      level_src + static_cast<size_t>(y) * level.pitch * elem,
                      static_cast<size_t>(level.width) * elem);
      }
    }
    linear_offset += layer_bytes * layout.layers;
  }
  // DELTA_GPU_TEXDUMP_UPLOAD: dump the post-detile pixels that Vulkan
  // receives. This differs from DELTA_GPU_TEXDUMP, which inspects raw guest
  // memory and is intentionally useful for spotting tiling rather than
  // validating the detiler.
  // DELTA_GPU_TEXDUMP_MIN: smallest edge to dump (default 128, i.e. content
  // atlases only). Lower it to reach the tiny nine-slice UI surfaces, which are
  // a handful of texels each and are best read as text.
  static int dump_small_count = 0;
  if (kDumpUpload && dump_small_count < 24 && elem == 4 && texel_w <= 32 &&
      texel_h <= 32 && texel_w >= kDumpMin && texel_h >= kDumpMin) {
    dump_small_count++;
    std::fprintf(stderr,
                 "[texsmall] base=%#lx %ux%u layers=%u tiling=%u pitch=%u "
                 "stored_h=%u raw:",
                 (unsigned long)base, texel_w, texel_h, layout.layers,
                 layout.tiling_idx, layout.mips[0].pitch,
                 layout.mips[0].stored_height);
    std::fprintf(stderr, "\n");
    char rp[256];
    std::snprintf(rp, sizeof(rp), "%s/rawblk_%02d_%ux%u.bin", DumpDir(),
                  dump_small_count - 1, texel_w, texel_h);
    if (FILE* rf = std::fopen(rp, "wb")) {
      std::fwrite(src, 1,
                  static_cast<size_t>(layout.mips[0].pitch) *
                      layout.mips[0].stored_height * elem,
                  rf);
      std::fclose(rf);
    }
    for (uint32_t y = 0; y < texel_h; y++) {
      std::fprintf(stderr, "[texsmall]  ");
      for (uint32_t x = 0; x < texel_w; x++) {
        const uint8_t* p = linear + (static_cast<uint64_t>(y) * texel_w + x) * 4;
        std::fprintf(stderr, "%02x%02x%02x%02x ", p[0], p[1], p[2], p[3]);
      }
      std::fprintf(stderr, "\n");
    }
  }
  static int dump_upload_count = 0;
  if (kDumpUpload && dump_upload_count < 32 && elem == 4 && texel_w >= 128 &&
      texel_h >= 128) {
    // Every layer, so an array/cube surface can be inspected face by face.
    for (uint32_t l = 1; l < layout.layers && l < 8; l++) {
      char lp[256];
      std::snprintf(lp, sizeof(lp), "%s/tex_layer%u_%02d_%#lx_%ux%u.ppm",
                    DumpDir(), l, dump_upload_count, (unsigned long)base,
                    texel_w, texel_h);
      if (FILE* lf = std::fopen(lp, "wb")) {
        std::fprintf(lf, "P6\n%u %u\n255\n", texel_w, texel_h);
        const uint8_t* lp8 =
            linear + static_cast<uint64_t>(l) * texel_w * texel_h * 4;
        for (uint64_t i = 0; i < static_cast<uint64_t>(texel_w) * texel_h; i++) {
          std::fputc(lp8[i * 4], lf);
          std::fputc(lp8[i * 4 + 1], lf);
          std::fputc(lp8[i * 4 + 2], lf);
        }
        std::fclose(lf);
      }
    }
    uint64_t rgb_nonzero = 0, alpha_nonzero = 0;
    const uint64_t pixels = static_cast<uint64_t>(texel_w) * texel_h;
    for (uint64_t i = 0; i < pixels; i++) {
      const uint8_t* p = linear + i * 4;
      if (p[0] || p[1] || p[2])
        rgb_nonzero++;
      if (p[3])
        alpha_nonzero++;
    }
    char path[256];
    std::snprintf(path, sizeof(path), "%s/tex_upload_%02d_%#lx_%ux%u.ppm",
                  DumpDir(), dump_upload_count, (unsigned long)base, texel_w,
                  texel_h);
    FILE* file = std::fopen(path, "wb");
    if (file) {
      std::fprintf(file, "P6\n%u %u\n255\n", texel_w, texel_h);
      for (uint64_t i = 0; i < pixels; i++) {
        const uint8_t* p = linear + i * 4;
        std::fputc(p[0], file);
        std::fputc(p[1], file);
        std::fputc(p[2], file);
      }
      std::fclose(file);
    }
    std::fprintf(stderr,
                 "[texupload] %d base=%#lx %ux%u rgb=%lu alpha=%lu/%lu -> %s\n",
                 dump_upload_count, (unsigned long)base, texel_w, texel_h,
                 (unsigned long)rgb_nonzero, (unsigned long)alpha_nonzero,
                 (unsigned long)pixels, path);
    dump_upload_count++;
  }
}

// One-time initialization upload used before frame recording starts.
bool UploadTexPixelsImmediate(VkImage img,
                              uint64_t base,
                              const gcn::TextureLayout32& layout,
                              uint32_t texel_w,
                              uint32_t texel_h,
                              bool is_3d) {
  const VkDeviceSize sz = TextureLinearBytes(layout);
  const uint32_t barrier_layers = is_3d ? 1 : layout.layers;
  VkBuffer stg = VK_NULL_HANDLE;
  VkDeviceMemory stg_mem = VK_NULL_HANDLE;
  VkCommandBuffer command = VK_NULL_HANDLE;
  void* map = nullptr;
  auto cleanup = [&] {
    if (map)
      vkUnmapMemory(g_dev.device, stg_mem);
    if (command)
      vkFreeCommandBuffers(g_dev.device, g_dev.pool, 1, &command);
    if (stg)
      vkDestroyBuffer(g_dev.device, stg, nullptr);
    if (stg_mem)
      vkFreeMemory(g_dev.device, stg_mem, nullptr);
  };
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = sz;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (vkCreateBuffer(g_dev.device, &bi, nullptr, &stg) != VK_SUCCESS)
    return false;
  VkMemoryRequirements br;
  vkGetBufferMemoryRequirements(g_dev.device, stg, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  ba.memoryTypeIndex = FindMemoryType(br.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (vkAllocateMemory(g_dev.device, &ba, nullptr, &stg_mem) != VK_SUCCESS ||
      vkBindBufferMemory(g_dev.device, stg, stg_mem, 0) != VK_SUCCESS ||
      vkMapMemory(g_dev.device, stg_mem, 0, sz, 0, &map) != VK_SUCCESS) {
    cleanup();
    return false;
  }

  VkBufferImageCopy copies[16]{};
  PackTexPixels(static_cast<uint8_t*>(map), 0, base, layout, texel_w, texel_h,
                copies, is_3d);
  vkUnmapMemory(g_dev.device, stg_mem);
  map = nullptr;

  VkCommandBufferAllocateInfo ca{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = g_dev.pool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(g_dev.device, &ca, &command) != VK_SUCCESS) {
    cleanup();
    return false;
  }
  VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(command, &cbi) != VK_SUCCESS) {
    cleanup();
    return false;
  }
  ImageBarrier(command, img, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, barrier_layers,
               layout.mip_levels);
  vkCmdCopyBufferToImage(command, stg, img,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         layout.mip_levels, copies);
  ImageBarrier(command, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               barrier_layers, layout.mip_levels);
  const VkResult end_result = vkEndCommandBuffer(command);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &command;
  uint64_t _t0 = NowNs();
  vkResetFences(g_dev.device, 1, &g_dev.fence);
  const VkResult up_submit =
      end_result == VK_SUCCESS ? vkQueueSubmit(g_dev.queue, 1, &si, g_dev.fence)
                               : end_result;
  const VkResult up_wait =
      up_submit == VK_SUCCESS
          ? vkWaitForFences(g_dev.device, 1, &g_dev.fence, VK_TRUE, UINT64_MAX)
          : up_submit;
  if (end_result != VK_SUCCESS || up_submit != VK_SUCCESS ||
      up_wait != VK_SUCCESS) {
    std::fprintf(stderr,
                 "[gpuvk] tex upload DEVICE FAULT: submit=%d wait=%d "
                 "base=%#lx %ux%u mips=%u layers=%u bytes=%llu\n",
                 (int)up_submit, (int)up_wait, (unsigned long)base,
                 layout.mips[0].width, layout.mips[0].height, layout.mip_levels,
                 layout.layers, (unsigned long long)sz);
    ReportDeviceFault(g_dev);
  }
  uint64_t _tex_dt = NowNs() - _t0;
  g_ns_tex_up += _tex_dt;
  g_fr_tex_up += _tex_dt;
  g_tex_ups++;
  cleanup();
  return end_result == VK_SUCCESS && up_submit == VK_SUCCESS &&
         up_wait == VK_SUCCESS;
}

bool RecordTexPixels(VkImage img,
                     VkImageLayout old_layout,
                     uint64_t base,
                     const gcn::TextureLayout32& layout,
                     uint32_t texel_w,
                     uint32_t texel_h,
                     bool is_3d) {
  const VkDeviceSize bytes = TextureLinearBytes(layout);
  const uint32_t barrier_layers = is_3d ? 1 : layout.layers;
  TextureUploadSlice upload;
  if (!AllocateTextureUpload(g_frame.slot_idx, bytes,
                             std::max<uint32_t>(16, layout.elem_bytes), upload))
    return false;
  const uint64_t start = NowNs();
  VkBufferImageCopy copies[16]{};
  PackTexPixels(upload.map, upload.offset, base, layout, texel_w, texel_h,
                copies, is_3d);
  EndRegion();
  const VkAccessFlags source_access =
      old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
          ? VK_ACCESS_SHADER_READ_BIT
          : 0;
  ImageBarrier(g_frame.cmd, img, old_layout,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, source_access,
               VK_ACCESS_TRANSFER_WRITE_BIT, barrier_layers,
               layout.mip_levels);
  vkCmdCopyBufferToImage(g_frame.cmd, upload.buffer, img,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         layout.mip_levels, copies);
  ImageBarrier(g_frame.cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               barrier_layers, layout.mip_levels);
  const uint64_t elapsed = NowNs() - start;
  g_ns_tex_up += elapsed;
  g_fr_tex_up += elapsed;
  g_tex_ups++;
  return true;
}

void RetireTextureImage(const TexImageKey& key) {
  bool retired = false;
  for (auto set = g_tex_cache.begin(); set != g_tex_cache.end();) {
    if (set->first.image == key) {
      g_retired_tex_sets.push_back(set->second);
      set = g_tex_cache.erase(set);
      retired = true;
    } else {
      ++set;
    }
  }
  for (auto view = g_tex_views.begin(); view != g_tex_views.end();) {
    if (view->first.image == key) {
      g_retired_tex_views.push_back(view->second);
      view = g_tex_views.erase(view);
      retired = true;
    } else {
      ++view;
    }
  }
  auto image = g_tex_images.find(key);
  if (image != g_tex_images.end()) {
    UnregisterTexturePages(key, image->second.footprint);
    GPU_BUGCHECK(g_tex_image_bytes >= image->second.allocation_size,
                 "texture budget underflow: %llu live < %llu retiring",
                 (unsigned long long)g_tex_image_bytes,
                 (unsigned long long)image->second.allocation_size);
    g_tex_image_bytes -= image->second.allocation_size;
    g_retired_tex_images.push_back(image->second);
    g_tex_images.erase(image);
    retired = true;
  }
  if (retired)
    ClearMultiTexCache();
}

bool EvictOldestTexture() {
  auto oldest = g_tex_images.end();
  for (auto it = g_tex_images.begin(); it != g_tex_images.end(); ++it) {
    if (it->second.last_used_frame == g_frame.num)
      continue;
    if (oldest == g_tex_images.end() ||
        it->second.last_used_frame < oldest->second.last_used_frame)
      oldest = it;
  }
  if (oldest == g_tex_images.end())
    return false;
  const TexImageKey key = oldest->first;
  RetireTextureImage(key);
  return true;
}

// Upload a guest texture (linear 32bpp RGBA) and return a descriptor set bound
// to it. Cached by guest base; re-uploaded when the guest pixels change (the
// room art is composed/loaded into the same buffer after the first sample, so a
// once-only cache would serve a stale black frame).
VkDescriptorSet GetTexture(uint64_t base,
                           uint32_t w,
                           uint32_t h,
                           uint32_t dfmt,
                           uint32_t nfmt,
                           uint32_t tiling,
                           uint32_t pitch,
                           uint32_t layers,
                           uint32_t base_array,
                           uint32_t view_layers,
                           uint32_t mip_levels,
                           uint32_t base_mip,
                           uint32_t view_mips,
                           uint32_t min_lod,
                           bool pow2_pad,
                           const uint32_t* sampler,
                           bool sampler_valid,
                           bool arrayed,
                           bool force_lod_zero,
                           bool depth_compare,
                           uint32_t swizzle,
                           uint32_t depth,
                           bool is_3d) {
  constexpr uint64_t kMaxTextureBytes = 256ull * 1024 * 1024;
  // DELTA_GPU_TEXWATCH=<base>: the head of one texture's guest memory, once per
  // frame it is bound. A surface the guest fills after our first upload reads
  // as its initial contents forever, and no other trace distinguishes that from
  // memory the guest never wrote.
  if (kTexWatch && base == (uint64_t)kTexWatch) {
    static int watched = -1;
    if (watched != g_frame.num && gpu::IsReadableRange(base, 32)) {
      watched = g_frame.num;
      const uint8_t* p = reinterpret_cast<const uint8_t*>(base);
      // Scan the whole surface, not just its head: a grading LUT legitimately
      // starts at black, so a zero prefix says nothing about whether the guest
      // filled it.
      const uint64_t span = uint64_t(w) * h * depth * 4;
      uint64_t first_nz = span;
      if (gpu::IsReadableRange(base, span))
        for (uint64_t i = 0; i < span; i++)
          if (p[i]) {
            first_nz = i;
            break;
          }
      std::fprintf(stderr, "[texwatch] f%d %#lx %ux%ux%u span=%lu first_nz=%ld:",
                   g_frame.num, (unsigned long)base, w, h, depth,
                   (unsigned long)span,
                   first_nz == span ? -1L : (long)first_nz);
      for (uint32_t i = 0; i < 32; i++)
        std::fprintf(stderr, " %02x", p[i]);
      std::fprintf(stderr, "\n");
    }
  }
  if (!w || !h || w > 8192 || h > 8192)
    return VK_NULL_HANDLE;
  VkFormat format = GuestTextureFormat(dfmt, nfmt);
  if (format == VK_FORMAT_UNDEFINED)
    return VK_NULL_HANDLE;
  if (is_3d) {
    // maxImageDimension3D (commonly 2048) applies to every axis of a volume,
    // well below the 2D limits checked above.
    static uint32_t max_3d = 0;
    if (!max_3d) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(g_dev.phys, &props);
      max_3d = props.limits.maxImageDimension3D;
    }
    if (!depth || depth > max_3d || w > max_3d || h > max_3d)
      return VK_NULL_HANDLE;
    layers = 1;
  } else {
    depth = 1;
  }
  if (!layers || base_array >= layers)
    return VK_NULL_HANDLE;
  view_layers = std::min(view_layers, layers - base_array);
  if (!arrayed)
    view_layers = 1;
  if (!view_layers)
    return VK_NULL_HANDLE;
  if (!mip_levels || base_mip >= mip_levels)
    return VK_NULL_HANDLE;
  view_mips = std::min(view_mips, mip_levels - base_mip);
  if (!view_mips)
    return VK_NULL_HANDLE;
  // Diagnostic override used to identify incorrectly described guest surfaces.
  tiling = TextureTiling(tiling);
  // Block-compressed formats lay out and detile in 4x4-block ("element")
  // space. Mip chains only halve cleanly in block space for power-of-two
  // texel dimensions; decline others (rare) rather than mis-address them.
  const bool bc = GuestFormatBlockCompressed(dfmt);
  const uint32_t elem_bytes = GuestFormatElemBytes(dfmt);
  if (bc && mip_levels > 1 && ((w & (w - 1)) || (h & (h - 1))))
    return VK_NULL_HANDLE;
  // BCn volume images are an optional Vulkan feature; decline rather than
  // create an image the driver need not support.
  if (bc && is_3d)
    return VK_NULL_HANDLE;
  const uint32_t lw = bc ? (w + 3) / 4 : w;
  const uint32_t lh = bc ? (h + 3) / 4 : h;
  const uint32_t lpitch =
      bc ? ((pitch ? pitch : w) + 3) / 4 : (pitch ? pitch : w);
  gcn::TextureLayout32 layout;
  // DELTA_GPU_TEXFAIL: why a guest texture declines to upload. Skyrim's font
  // atlas fell out here and every glyph then sampled the white default, which
  // fills each glyph quad solid instead of masking it.
  // The layout's layer axis is the volume's slice axis.
  if (!gcn::BuildTextureLayout32(layout, lw, lh, lpitch, is_3d ? depth : layers,
                                 mip_levels, tiling, pow2_pad, elem_bytes)) {
    if (kTexFail) {
      static int n = 0;
      if (n++ < 12)
        std::fprintf(
            stderr,
            "[texfail] layout %#lx %ux%u pitch=%u tiling=%u elem=%u mips=%u\n",
            (unsigned long)base, w, h, lpitch, tiling, elem_bytes, mip_levels);
    }
    return VK_NULL_HANDLE;
  }
  uint64_t footprint = layout.size;
  if (footprint > kMaxTextureBytes) {
    if (kTexFail) {
      static int n = 0;
      if (n++ < 12)
        std::fprintf(stderr,
                     "[texfail] range %#lx %ux%u footprint=%lu (max %lu)\n",
                     (unsigned long)base, w, h, (unsigned long)footprint,
                     (unsigned long)kMaxTextureBytes);
    }
    return VK_NULL_HANDLE;
  }
  TexKey key = TextureKey(base, w, h, dfmt, nfmt, tiling, pitch, layers,
                          base_array, view_layers, mip_levels, base_mip,
                          view_mips, min_lod, pow2_pad, sampler, sampler_valid,
                          arrayed, force_lod_zero, depth_compare, swizzle,
                          depth, is_3d);
  // DELTA_GPU_TEXRAW: write each large texture's raw tiled footprint, with its
  // layout in the name, so a swizzle can be worked out offline.
  if (kTexRaw && w >= 256 && h >= 128 && gpu::IsReadableRange(base, footprint)) {
    static int rawn = 0;
    static std::unordered_set<uint64_t> raw_seen;
    if (rawn < 12 && raw_seen.insert(static_cast<uint64_t>(w) << 32 | h).second) {
      char p[320];
      std::snprintf(p, sizeof(p), "%s/raw_%02d_%ux%u_pitch%u_sh%u_tile%u_e%u.bin",
                    DumpDir(), rawn++, w, h, layout.mips[0].pitch,
                    layout.mips[0].stored_height, tiling, elem_bytes);
      if (FILE* f = std::fopen(p, "wb")) {
        std::fwrite(reinterpret_cast<const void*>(base), 1, footprint, f);
        std::fclose(f);
        std::fprintf(stderr, "[texraw] %s (%llu bytes)\n", p,
                     (unsigned long long)footprint);
      }
    }
  }
  // Diagnostic (DELTA_GPU_TEXDUMP): in deep gameplay, dump the first few large
  // guest textures sampled, so a non-tutorial room's floor texture can be
  // inspected (is it loaded/brown or black/zero?). Counts non-zero pixels too.
  if (kTexDump && g_frame.num > 1200 && w >= 128 && h >= 128) {
    if (!gpu::IsReadableRange(base, footprint))
      return VK_NULL_HANDLE;
    static int tdn = 0;
    if (tdn < 16) {
      const uint32_t* px = reinterpret_cast<const uint32_t*>(base);
      uint64_t cnt = (uint64_t)w * h, nz = 0,
               step = cnt > 8192 ? cnt / 8192 : 1;
      for (uint64_t i = 0; i < cnt; i += step)
        if (px[i] & 0x00FFFFFFu)
          nz++;
      char p[256];
      std::snprintf(p, sizeof(p), "%s/tex_%02d_%#lx_%ux%u.ppm", DumpDir(), tdn,
                    (unsigned long)base, w, h);
      FILE* f = std::fopen(p, "wb");
      if (f) {
        std::fprintf(f, "P6\n%u %u\n255\n", w, h);
        const uint8_t* b = reinterpret_cast<const uint8_t*>(base);
        for (uint64_t i = 0; i < cnt; i++) {
          std::fputc(b[i * 4], f);
          std::fputc(b[i * 4 + 1], f);
          std::fputc(b[i * 4 + 2], f);
        }
        std::fclose(f);
      }
      std::fprintf(stderr, "[texdump] %d base=%#lx %ux%u nonzero=%lu/8192\n",
                   tdn, (unsigned long)base, w, h, (unsigned long)nz);
      tdn++;
    }
  }
  if (!rhi::FlushCsWritesRange(rhi::DefaultRenderer(), base, footprint))
    return VK_NULL_HANDLE;
  auto image_it = g_tex_images.find(key.image);
  uint64_t hsh = 0;
  if (image_it == g_tex_images.end() ||
      image_it->second.last_checked_frame != g_frame.num) {
    // Mapping probes are syscall-heavy. Perform one alongside the
    // once-per-frame content validation rather than on every draw that reuses
    // this image.
    if (!gpu::IsReadableRange(base, footprint))
      return VK_NULL_HANDLE;
    hsh = TexHash(base, footprint);
    if (image_it != g_tex_images.end() && image_it->second.hash != hsh) {
      if (!RecordTexPixels(image_it->second.image,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, base,
                           layout, w, h, is_3d))
        return VK_NULL_HANDLE;
      image_it->second.hash = hsh;
    }
  }
  if (image_it == g_tex_images.end()) {
    while (g_tex_images.size() >= 3000)
      if (!EvictOldestTexture())
        return VK_NULL_HANDLE;
    TexImageEntry image_entry;
    image_entry.footprint = footprint;
    image_entry.hash = hsh;
    image_entry.last_checked_frame = g_frame.num;
    image_entry.last_used_frame = g_frame.num;
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = is_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    ii.format = format;
    ii.extent = {w, h, is_3d ? depth : 1};
    ii.mipLevels = mip_levels;
    ii.arrayLayers = layers;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (vkCreateImage(g_dev.device, &ii, nullptr, &image_entry.image) !=
        VK_SUCCESS)
      return VK_NULL_HANDLE;
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(g_dev.device, image_entry.image, &mr);
    // Budget on LIVE image bytes. Retired images are only destroyed two
    // BeginFrames later (ReleaseRetiredTextures), so actual allocated memory
    // can transiently exceed the budget by up to two frames of retirements;
    // capping allocated bytes instead would make creation fail outright at
    // the budget edge, since eviction cannot free memory mid-frame.
    static const uint64_t kTextureBudget =
        std::max<uint64_t>(kTextureMb, 64) * 1024 * 1024;
    while (g_tex_image_bytes + mr.size > kTextureBudget)
      if (!EvictOldestTexture()) {
        vkDestroyImage(g_dev.device, image_entry.image, nullptr);
        return VK_NULL_HANDLE;
      }
    if (!g_image_memory.Allocate(g_dev, image_entry.image,
                                 image_entry.allocation)) {
      vkDestroyImage(g_dev.device, image_entry.image, nullptr);
      return VK_NULL_HANDLE;
    }
    if (!RecordTexPixels(image_entry.image, VK_IMAGE_LAYOUT_UNDEFINED, base,
                         layout, w, h, is_3d)) {
      vkDestroyImage(g_dev.device, image_entry.image, nullptr);
      g_image_memory.Free(g_dev, image_entry.allocation);
      return VK_NULL_HANDLE;
    }
    image_entry.allocation_size = mr.size;
    NameObject(VK_OBJECT_TYPE_IMAGE, (uint64_t)image_entry.image,
               "tex %#llx %ux%u mips=%u layers=%u", (unsigned long long)base, w,
               h, mip_levels, layers);
    image_it = g_tex_images.emplace(key.image, image_entry).first;
    g_tex_image_bytes += mr.size;
    RegisterTexturePages(key.image, footprint);
  } else {
    image_it->second.last_checked_frame = g_frame.num;
  }
  image_it->second.last_used_frame = g_frame.num;

  auto it = g_tex_cache.find(key);
  if (it != g_tex_cache.end())
    return it->second.set;
  if (g_tex_cache.size() >= 12000)
    return VK_NULL_HANDLE;
  TexViewKey view_key = TextureViewKey(key);
  auto view_it = g_tex_views.find(view_key);
  if (view_it == g_tex_views.end()) {
    if (g_tex_views.size() >= 12000)
      return VK_NULL_HANDLE;
    TexViewEntry view_entry;
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = image_it->second.image;
    vci.viewType = is_3d      ? VK_IMAGE_VIEW_TYPE_3D
                   : arrayed  ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                              : VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, base_mip, view_mips,
                            is_3d ? 0u : base_array,
                            (arrayed && !is_3d) ? view_layers : 1u};
    // T# DST_SEL: 0 = zero, 1 = one, 4..7 = R/G/B/A. A single-channel mask (a
    // font atlas) selects its coverage into the components the shader reads;
    // without this every glyph samples alpha = 1 and fills solid.
    vci.components = TextureComponents(view_key.swizzle);
    if (vkCreateImageView(g_dev.device, &vci, nullptr, &view_entry.view) !=
        VK_SUCCESS)
      return VK_NULL_HANDLE;
    view_it = g_tex_views.emplace(view_key, view_entry).first;
  }

  TexEntry e;
  e.set = AllocateSamplerSet(g_tex.ds_layout, false, e.pool);
  if (!e.set)
    return VK_NULL_HANDLE;
  VkDescriptorImageInfo dii{SamplerFor(key.sampler), view_it->second.view,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wr.dstSet = e.set;
  wr.descriptorCount = 1;
  wr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  wr.pImageInfo = &dii;
  vkUpdateDescriptorSets(g_dev.device, 1, &wr, 0, nullptr);

  g_tex_cache.emplace(key, e);
  return e.set;
}

// Image view for a guest texture (ensures it is cached/uploaded via
// GetTexture).
bool GuestTextureUploadSupported(uint32_t dfmt, uint32_t nfmt) {
  // The detiler preserves packed 32-bpp texels; Vulkan applies the descriptor's
  // numeric interpretation when sampling.
  return GuestTextureFormat(dfmt, nfmt) != VK_FORMAT_UNDEFINED;
}

VkImageView TexViewFor(const DrawInfo::DrawTex& t) {
  if (!t.base || !t.w || !t.h || !GuestTextureUploadSupported(t.dfmt, t.nfmt))
    return VK_NULL_HANDLE;
  if (GetTexture(t.base, t.w, t.h, t.dfmt, t.nfmt, t.tiling, t.pitch, t.layers,
                 t.base_array, t.view_layers, t.mip_levels, t.base_mip,
                 t.view_mips, t.min_lod, t.pow2_pad, t.sampler, t.sampler_valid,
                 t.arrayed, t.force_lod_zero, t.depth_compare, t.swizzle,
                 t.depth, t.is_3d) == VK_NULL_HANDLE)
    return VK_NULL_HANDLE;
  TexKey key = TextureKey(
      t.base, t.w, t.h, t.dfmt, t.nfmt, TextureTiling(t.tiling), t.pitch,
      t.layers, t.base_array, t.view_layers, t.mip_levels, t.base_mip,
      t.view_mips, t.min_lod, t.pow2_pad, t.sampler, t.sampler_valid, t.arrayed,
      t.force_lod_zero, t.depth_compare, t.swizzle, t.depth, t.is_3d);
  auto it = g_tex_views.find(TextureViewKey(key));
  return it != g_tex_views.end() ? it->second.view : VK_NULL_HANDLE;
}

// An N-sampler descriptor set (set 0, bindings 0..kMaxTex-1) for a recomp PS
// that samples >1 texture. Cached by the combination of the textures'
// (base,w,h); any binding we cannot resolve gets the 1x1 white default so a
// diffuse*lightmap shader with a missing map shows the diffuse instead of going
// black.
struct MultiTexSet {
  VkDescriptorSet set = VK_NULL_HANDLE;
  VkDescriptorPool pool = VK_NULL_HANDLE;
};

struct MultiTexKey {
  uint32_t num_texs = 0;
  TexKey tex[kMaxTex];
  VkImageView view[kMaxTex] = {};
  VkImageLayout layout[kMaxTex] = {};
  bool storage[kMaxTex] = {};
  bool operator==(const MultiTexKey& o) const {
    if (num_texs != o.num_texs)
      return false;
    for (uint32_t i = 0; i < num_texs; i++)
      if (!(tex[i] == o.tex[i]) || view[i] != o.view[i] ||
          layout[i] != o.layout[i] || storage[i] != o.storage[i])
        return false;
    return true;
  }
};

struct MultiTexKeyHash {
  size_t operator()(const MultiTexKey& k) const {
    uint64_t h = HashWord(1469598103934665603ull, k.num_texs);
    for (uint32_t i = 0; i < k.num_texs; i++) {
      h = HashWord(h, TexKeyHash{}(k.tex[i]));
      h = HashWord(h, std::hash<VkImageView>{}(k.view[i]));
      h = HashWord(h, k.layout[i]);
      h = HashWord(h, k.storage[i]);
    }
    return static_cast<size_t>(h);
  }
};

std::unordered_map<MultiTexKey, MultiTexSet, MultiTexKeyHash> g_mtex_cache;
std::vector<MultiTexSet> g_retired_mtex;
void ClearMultiTexCache() {
  for (const auto& [key, entry] : g_mtex_cache) {
    (void)key;
    if (entry.set)
      g_retired_mtex.push_back(entry);
  }
  g_mtex_cache.clear();
}

void ReleaseRetiredTextures() {
  // Two-stage aging for the pipelined frame: an object retired while frame N
  // recorded may still be referenced by BOTH in-flight command buffers (N-1
  // until N's EndFrame, N until N+1's EndFrame). Objects therefore rest one
  // extra BeginFrame in the `aged` generation before being destroyed -- by
  // then every command buffer that could reference them has been fence-waited.
  static std::vector<MultiTexSet> aged_mtex;
  static std::vector<TexEntry> aged_tex_sets;
  static std::vector<TexViewEntry> aged_tex_views;
  static std::vector<TexImageEntry> aged_tex_images;
  for (const MultiTexSet& entry : aged_mtex)
    vkFreeDescriptorSets(g_dev.device, entry.pool, 1, &entry.set);
  for (const TexEntry& e : aged_tex_sets)
    if (e.set)
      vkFreeDescriptorSets(g_dev.device, e.pool, 1, &e.set);
  for (const TexViewEntry& e : aged_tex_views)
    if (e.view)
      vkDestroyImageView(g_dev.device, e.view, nullptr);
  for (const TexImageEntry& e : aged_tex_images) {
    if (e.image)
      vkDestroyImage(g_dev.device, e.image, nullptr);
    ImageAllocation allocation = e.allocation;
    g_image_memory.Free(g_dev, allocation);
  }
  aged_mtex = std::move(g_retired_mtex);
  aged_tex_sets = std::move(g_retired_tex_sets);
  aged_tex_views = std::move(g_retired_tex_views);
  aged_tex_images = std::move(g_retired_tex_images);
  g_retired_mtex.clear();
  g_retired_tex_sets.clear();
  g_retired_tex_views.clear();
  g_retired_tex_images.clear();
}

VkDescriptorSet GetMultiTexSet(const DrawInfo& d,
                               VkDescriptorSetLayout set_layout,
                               const VkImageView* resolved_views,
                               const VkImageLayout* resolved_layouts) {
  MultiTexKey key;
  key.num_texs = std::min(d.num_texs, kMaxTex);
  for (uint32_t i = 0; i < key.num_texs; i++) {
    const auto& t = d.texs[i];
    key.tex[i] = TextureKey(
        t.base, t.w, t.h, t.dfmt, t.nfmt, TextureTiling(t.tiling), t.pitch,
        t.layers, t.base_array, t.view_layers, t.mip_levels, t.base_mip,
        t.view_mips, t.min_lod, t.pow2_pad, t.sampler, t.sampler_valid,
        t.arrayed, t.force_lod_zero, t.depth_compare, t.swizzle, t.depth,
        t.is_3d);
    key.view[i] = resolved_views[i];
    key.layout[i] = resolved_layouts[i];
    key.storage[i] = d.texs[i].storage;
  }
  auto ci = g_mtex_cache.find(key);
  if (ci != g_mtex_cache.end())
    return ci->second.set;
  if (g_mtex_cache.size() > 3500)
    return VK_NULL_HANDLE;
  // DELTA_GPU_FORCEWHITE: bind the 1x1 white default for every sampler
  // (diagnostic). Doom64's world textures are built by compute dispatches we
  // don't execute, so the atlases are all-zero and the alpha-blended world
  // samples transparent-black (= invisible). Forcing white makes the geometry
  // render opaque, proving the 3D transform/raster/depth path works and
  // isolating the blackness to the texture data.
  VkImageView views[kMaxTex];
  VkImageLayout layouts[kMaxTex];
  // DELTA_GPU_TEXMISS: report every sampler binding that falls back to the 1x1
  // white default (the source of "everything renders white" chains) with the
  // descriptor state that failed to resolve.
  static int tex_miss_logged = 0;
  for (uint32_t i = 0; i < key.num_texs; i++) {
    VkImageView v =
        (i < key.num_texs && !kForceWhite) ? resolved_views[i] : VK_NULL_HANDLE;
    bool arrayed = i < key.num_texs && d.texs[i].arrayed;
    bool is_3d = i < key.num_texs && d.texs[i].is_3d;
    if (kTexMiss && !v && i < key.num_texs && tex_miss_logged < 64) {
      tex_miss_logged++;
      const auto& t = d.texs[i];
      std::fprintf(
          stderr,
          "[texmiss] bind=%u base=%#lx %ux%u dfmt=%u nfmt=%u tiling=%u "
          "layers=%u mips=%u arrayed=%d\n",
          i, (unsigned long)t.base, t.w, t.h, t.dfmt, t.nfmt, t.tiling,
          t.layers, t.mip_levels, t.arrayed);
    }
    if (d.texs[i].storage && !v)
      return VK_NULL_HANDLE;
    VkImageView fallback =
        is_3d ? g_tex.white_3d_view
              : (arrayed ? g_tex.white_array_view : g_tex.white_view);
    views[i] = v ? v : fallback;
    layouts[i] =
        v ? resolved_layouts[i] : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  }
  MultiTexSet entry;
  entry.set = AllocateSamplerSet(set_layout, true, entry.pool);
  if (!entry.set)
    return VK_NULL_HANDLE;
  VkDescriptorImageInfo dii[kMaxTex];
  VkWriteDescriptorSet wr[kMaxTex];
  for (uint32_t i = 0; i < key.num_texs; i++) {
    SamplerKey sampler;
    if (i < key.num_texs) {
      sampler.valid = d.texs[i].sampler_valid;
      std::memcpy(sampler.raw, d.texs[i].sampler, sizeof(sampler.raw));
      sampler.image_min_lod = d.texs[i].min_lod;
      sampler.force_lod_zero = d.texs[i].force_lod_zero;
      sampler.depth_compare = d.texs[i].depth_compare;
    }
    dii[i] = {d.texs[i].storage ? VK_NULL_HANDLE : SamplerFor(sampler),
              views[i], layouts[i]};
    wr[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr[i].dstSet = entry.set;
    wr[i].dstBinding = i;
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = d.texs[i].storage
                               ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                               : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[i].pImageInfo = &dii[i];
  }
  vkUpdateDescriptorSets(g_dev.device, key.num_texs, wr, 0, nullptr);
  g_mtex_cache[key] = entry;
  return entry.set;
}

// Mark cached textures overlapping a written range for validation. Their next
// use refreshes the existing compatible image in place.
void InvalidateTexRange(uint64_t base, uint64_t size) {
  if (!size || base > UINT64_MAX - size)
    return;
  uint64_t end = base + size;
  std::vector<TexImageKey> overlap;
  for (uint64_t page = base >> kTexturePageShift;
       page <= (end - 1) >> kTexturePageShift; page++) {
    auto found = g_texture_pages.find(page);
    if (found == g_texture_pages.end())
      continue;
    for (const TexImageKey& key : found->second) {
      if (std::find(overlap.begin(), overlap.end(), key) == overlap.end())
        overlap.push_back(key);
    }
  }
  for (const TexImageKey& key : overlap) {
    auto found = g_tex_images.find(key);
    if (found != g_tex_images.end() && key.base < end &&
        base < key.base + found->second.footprint)
      found->second.last_checked_frame = -1;
  }
}

}  // namespace gpu::vk
