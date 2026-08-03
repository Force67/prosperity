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

#include "gpu/gcn/gcn_translate.h"

namespace gpu::vk {

struct TextureUploadBlock {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  uint8_t* map = nullptr;
  VkDeviceSize capacity = 0;
  VkDeviceSize offset = 0;
  // Consecutive ResetTextureUploads calls with no allocation from this block;
  // long-idle blocks are destroyed at reset (see ResetTextureUploads).
  uint64_t idle_resets = 0;
};

struct TextureUploadSlice {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  uint8_t* map = nullptr;
};

constexpr VkDeviceSize kVbRing = 16ull * 1024 * 1024;  // per-frame vertex ring
// DELTA_GPU_VBRING_MB=<n>: override the vertex ring size (default kVbRing).
// Halved per frame slot like every other ring, so the usable per-frame budget
// is n/2 MB. Exists because the ring is a hard per-frame draw budget, not a
// cache: a draw whose vertices do not fit is declined outright (kRing) and
// simply never appears, which is invisible in a draw count. Read once.
VkDeviceSize VbRingBytes();
constexpr VkDeviceSize kIbRing =
    8ull * 1024 * 1024;  // per-frame index ring (32-bit)
constexpr VkDeviceSize kUboRing =
    64ull * 1024 * 1024;  // per-frame recomp cbuffer ring
constexpr uint32_t kCbufWindow = gpu::gcn::kCbufDwords * 4;
constexpr uint32_t kCbufBindings =
    gpu::gcn::kMaxCbufBindings;  // set-1 UBO bindings
// Raw-buffer ring: the windows a recompiled shader's hand-written MUBUF loads
// read from, at set-2 bindings 0..kRawBufBindings-1. Four dynamic storage
// buffers is the Vulkan floor (maxDescriptorSetStorageBuffersDynamic), so the
// recompiler's cap matches it exactly.
// The window is a compromise the shader knows about: a MUBUF index has no
// static bound, so a resource larger than this is staged only up to here and
// the recompiled shader clamps, reading the window's last dword past it.
constexpr VkDeviceSize kSboRing = 128ull * 1024 * 1024;
constexpr uint32_t kRawBufWindow = gpu::gcn::kGfxBufferDwords * 4;
constexpr uint32_t kRawBufBindings = gpu::gcn::kMaxGfxBuffers;

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
  bool zero_window_initialized = false;
  VkDescriptorSetLayout ubo_layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout empty_layout = VK_NULL_HANDLE;
  VkDescriptorPool ubo_pool = VK_NULL_HANDLE;
  VkDescriptorSet ubo_set = VK_NULL_HANDLE;

  // Raw-buffer ring: same shape as the cbuffer ring (fixed windows selected by
  // a dynamic offset), but storage buffers, because a MUBUF address is a
  // per-lane index rather than a uniform offset. Window 0 stays zero and is
  // what an unresolved binding points at.
  VkBuffer sbo_buf = VK_NULL_HANDLE;
  VkDeviceMemory sbo_mem = VK_NULL_HANDLE;
  uint8_t* sbo_map = nullptr;
  VkDeviceSize sbo_offset = 0, sbo_end = kSboRing;
  // Bindings actually created, = min(device dynamic-SSBO limit, kRawBufBindings).
  // The recompiler is told this via gcn::SetMaxGfxBuffers so it never plans a
  // binding the layout does not have.
  uint32_t sbo_count = gpu::gcn::kMinGfxBuffers;
  uint32_t sbo_align = 256;
  VkDeviceSize sbo_stride = kRawBufWindow;
  std::vector<uint32_t> sbo_written;
  VkDescriptorSetLayout sbo_layout = VK_NULL_HANDLE;
  VkDescriptorPool sbo_pool = VK_NULL_HANDLE;
  VkDescriptorSet sbo_set = VK_NULL_HANDLE;

  // Texture uploads are recorded into the active frame command buffer. Each
  // frame slot owns its blocks so an in-flight transfer is never overwritten.
  std::vector<TextureUploadBlock> texture_uploads[2];
};

extern UploadRings& g_ring;

bool CreateUploadRings(const VkPhysicalDeviceProperties& props);
// Allocate the raw-buffer ring + its descriptor set. Deferred to the first
// draw that needs one: the ring is large, and a title whose vertex fetches the
// vertex-input state already covers never binds set 2 at all. The set layout
// itself is created up front, because pipeline layouts name it.
bool EnsureRawBufferRing();
bool AllocateTextureUpload(uint32_t slot,
                           VkDeviceSize bytes,
                           VkDeviceSize alignment,
                           TextureUploadSlice& slice);
void ResetTextureUploads(uint32_t slot);

}  // namespace gpu::vk
