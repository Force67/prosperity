/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Render targets keyed by their guest address. One image serves both as a
// colour attachment and as a sampled texture, so render-to-texture and MRT fall
// out of the same lookup; an address -> image page table resolves a sampled
// address to every live image whose footprint overlaps it.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "rhi/command.h"

namespace gpu::vk {

// An image in the resource cache: a render target keyed by its guest base address,
// that also doubles as a sampleable texture (render-to-texture). This is the unit
// of the resource model -- RT-bind and texture-sample both resolve to
// the same Image via the address page table, so render-to-texture/MRT "just work".
struct RTarget {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;  // for sampling this RT as a texture
  // Sampling an image while it is also a color attachment requires Vulkan's
  // attachment-feedback-loop extension. Keep a lazy copy instead so the shader
  // reads the attachment contents as they existed before the draw.
  VkImage feedbackImage = VK_NULL_HANDLE;
  VkDeviceMemory feedbackMem = VK_NULL_HANDLE;
  VkImageView feedbackView = VK_NULL_HANDLE;
  VkDescriptorSet feedbackSet = VK_NULL_HANDLE;
  std::unordered_map<uint32_t, VkImageView> sampledViews;
  std::unordered_map<uint32_t, VkImageView> feedbackSampledViews;
  VkImageLayout feedbackLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  uint32_t w = 0, h = 0;
  VkFormat fmt = VK_FORMAT_B8G8R8A8_UNORM;  // identity: addr alone doesn't pin format
  bool isDepth = false;                      // depth/stencil attachment (MRT/depth task)
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  bool usedThisFrame = false;
  uint32_t draws = 0;     // draws into this RT this frame
  int lastFrame = -1000;  // frame number this RT was last rendered into
  bool clearPending = false;  // a fullscreen black clear was requested; applied lazily
                              // (as loadOp=CLEAR) only when later content redraws this RT
                              // in the same frame.
  VkClearColorValue clearValue{{0.0f, 0.0f, 0.0f, 0.0f}};
  bool everRendered = false;  // false until first real render (then loadOp can LOAD)
};

extern std::unordered_map<uint64_t, RTarget> g_rts;

// Depth/stencil attachment, keyed by its guest DB_Z_WRITE_BASE. Allocated on demand
// when a 3D draw binds a Z buffer. Internal format is always D32_SFLOAT (we never
// read depth back to guest memory, so only a valid depth format is needed).
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
struct DepthTarget {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  std::unordered_map<uint32_t, VkImageView> sampledViews;
  uint32_t w = 0, h = 0;
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  int lastFrame = -1000;
  bool usedThisFrame = false;
  bool clearPending = false;
  float clearValue = 1.0f;
};

extern std::unordered_map<uint64_t, DepthTarget> g_depths;

// Address -> image page table (the resource model's core). Maps a 64 KiB guest page
// to the RT bases whose memory footprint covers it, so a sampled address resolves to
// every overlapping live image in O(pages) instead of scanning the whole cache. A
// page can be touched by several overlapping/aliased RTs (double-buffer pairs, a
// pool of cycled scene buffers), so each page holds a list.
constexpr uint32_t kRtPageShift = 16;  // 64 KiB
extern std::unordered_map<uint64_t, std::vector<uint64_t>> g_rtPages;

uint64_t rtByteSizeWH(uint32_t w, uint32_t h, VkFormat fmt);
uint64_t rtByteSize(const RTarget &rt);

RTarget *getRT(uint64_t base, uint32_t w, uint32_t h, VkFormat fmt);
DepthTarget *getDepthRT(uint64_t base, uint32_t w, uint32_t h);

// Sampling an image while it is also a colour attachment needs Vulkan's
// attachment-feedback-loop extension; copy it instead and sample the copy.
VkDescriptorSet snapshotRT(RTarget &rt);

VkImageView sampledView(RTarget &rt, uint32_t swizzle, bool feedback = false);
VkImageView sampledView(DepthTarget &depth, uint32_t swizzle);

// Resolve a sampled guest address to the live image backing it (0 = none).
uint64_t resolveSampledRT(uint64_t addr, uint32_t w, uint32_t h);
uint64_t resolveSampledDepth(uint64_t addr, uint32_t w, uint32_t h);

// The open dynamic-rendering region, and which targets the frame has touched.
struct RenderRegion {
  uint64_t curRt = 0;        // primary RT (MRT0) of the open region (0 = none)
  uint64_t curMrt[8] = {0};  // all colour targets bound in the open region
  uint32_t curMrtCount = 0;
  uint64_t curDepth = 0;     // depth target bound in the open region (0 = none)
  uint64_t lastRt = 0;       // last RT rendered to (present fallback)
  uint64_t firstRt = 0;      // first colour RT created (diagnostic selector)
  uint64_t busiestRt = 0;
  uint32_t busiestRtDraws = 0;
  bool open = false;
};

extern RenderRegion g_region;

void beginRegion(const uint64_t *mrtBase, const uint32_t *mrtInfo,
                 uint32_t mrtCount, uint32_t w, uint32_t h,
                 uint64_t depthBase = 0, float depthClear = 1.0f);
void endRegion();
void setGuestViewport(const rhi::DrawInfo &d);

}  // namespace gpu::vk
