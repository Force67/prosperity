/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Per-frame host-visible upload rings. Every draw's vertices, indices and
// constant-buffer windows are copied into these mapped rings and bound from
// there; each frame slot owns one half, so the in-flight frame's data is never
// overwritten (see vk_frame).

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

#include "gpu/ps4/gcn/gcn_translate.h"

namespace gpu::vk {

struct TextureUploadBlock {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  uint8_t* map = nullptr;
  VkDeviceSize capacity = 0;
  VkDeviceSize offset = 0;
};

struct TextureUploadSlice {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  uint8_t* map = nullptr;
};

constexpr VkDeviceSize kVbRing = 16ull * 1024 * 1024;  // per-frame vertex ring
constexpr VkDeviceSize kIbRing =
    8ull * 1024 * 1024;  // per-frame index ring (32-bit)
constexpr VkDeviceSize kUboRing =
    64ull * 1024 * 1024;  // per-frame recomp cbuffer ring
constexpr uint32_t kCbufWindow = gpu::gcn::kCbufDwords * 4;
constexpr uint32_t kCbufBindings =
    gpu::gcn::kMaxCbufBindings;  // set-1 UBO bindings

struct UploadRings {
  // Vertex ring: interleaved pos+colour+uv for the heuristic path, the raw
  // guest vertex records for the recompiled path.
  VkBuffer vb = VK_NULL_HANDLE;
  VkDeviceMemory vb_mem = VK_NULL_HANDLE;
  uint8_t* vb_map = nullptr;
  VkDeviceSize vb_offset = 0, vb_end = kVbRing;

  // Index ring: 32-bit indices (16-bit guest indices are widened on upload).
  VkBuffer ib = VK_NULL_HANDLE;
  VkDeviceMemory ib_mem = VK_NULL_HANDLE;
  uint8_t* ib_map = nullptr;
  VkDeviceSize ib_offset = 0, ib_end = kIbRing;

  // Recomp cbuffer ring: per-draw VS/PS constant buffers live at set 1 bindings
  // 0..kCbufBindings-1, each addressed by a dynamic offset into this ring.
  // empty_layout fills set 0 for untextured recomp draws.
  VkBuffer ubo_buf = VK_NULL_HANDLE;
  VkDeviceMemory ubo_mem = VK_NULL_HANDLE;
  uint8_t* ubo_map = nullptr;
  VkDeviceSize ubo_offset = 0, ubo_end = kUboRing;
  uint32_t ubo_align = 256;
  VkDeviceSize ubo_stride = kCbufWindow;
  std::vector<uint32_t> ubo_written;
  VkDescriptorSetLayout ubo_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout empty_layout = VK_NULL_HANDLE;
  VkDescriptorPool ubo_pool = VK_NULL_HANDLE;
  VkDescriptorSet ubo_set = VK_NULL_HANDLE;

  // Texture uploads are recorded into the active frame command buffer. Each
  // frame slot owns its blocks so an in-flight transfer is never overwritten.
  std::vector<TextureUploadBlock> texture_uploads[2];
};

extern UploadRings& g_ring;

bool CreateUploadRings(const VkPhysicalDeviceProperties& props);
bool AllocateTextureUpload(uint32_t slot,
                           VkDeviceSize bytes,
                           VkDeviceSize alignment,
                           TextureUploadSlice& slice);
void ResetTextureUploads(uint32_t slot);

}  // namespace gpu::vk
