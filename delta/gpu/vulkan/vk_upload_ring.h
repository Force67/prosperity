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

#include "gcn/gcn_translate.h"

namespace gpu::vk {

constexpr VkDeviceSize kVbRing = 16ull * 1024 * 1024;   // per-frame vertex ring
constexpr VkDeviceSize kIbRing = 8ull * 1024 * 1024;    // per-frame index ring (32-bit)
constexpr VkDeviceSize kUboRing = 64ull * 1024 * 1024;  // per-frame recomp cbuffer ring
constexpr uint32_t kCbufWindow = gpu::gcn::kCbufDwords * 4;
constexpr uint32_t kCbufBindings = gpu::gcn::kMaxCbufBindings;  // set-1 UBO bindings

struct UploadRings {
  // Vertex ring: interleaved pos+colour+uv for the heuristic path, the raw
  // guest vertex records for the recompiled path.
  VkBuffer vb = VK_NULL_HANDLE;
  VkDeviceMemory vbMem = VK_NULL_HANDLE;
  uint8_t *vbMap = nullptr;
  VkDeviceSize vbOffset = 0, vbEnd = kVbRing;

  // Index ring: 32-bit indices (16-bit guest indices are widened on upload).
  VkBuffer ib = VK_NULL_HANDLE;
  VkDeviceMemory ibMem = VK_NULL_HANDLE;
  uint8_t *ibMap = nullptr;
  VkDeviceSize ibOffset = 0, ibEnd = kIbRing;

  // Recomp cbuffer ring: per-draw VS/PS constant buffers live at set 1 bindings
  // 0..kCbufBindings-1, each addressed by a dynamic offset into this ring.
  // emptyLayout fills set 0 for untextured recomp draws.
  VkBuffer uboBuf = VK_NULL_HANDLE;
  VkDeviceMemory uboMem = VK_NULL_HANDLE;
  uint8_t *uboMap = nullptr;
  VkDeviceSize uboOffset = 0, uboEnd = kUboRing;
  uint32_t uboAlign = 256;
  VkDescriptorSetLayout uboLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout emptyLayout = VK_NULL_HANDLE;
  VkDescriptorPool uboPool = VK_NULL_HANDLE;
  VkDescriptorSet uboSet = VK_NULL_HANDLE;
};

extern UploadRings g_ring;

bool createUploadRings(const VkPhysicalDeviceProperties &props);

}  // namespace gpu::vk
