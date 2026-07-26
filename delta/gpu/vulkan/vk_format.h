/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Translation of the guest GPU's surface, vertex, blend and primitive encodings
// into their Vulkan equivalents, plus the conversion of a readback texel back
// to BGRA8. Pure tables: no device state, no caches.

#include <vulkan/vulkan.h>

#include <cstdint>

namespace gpu::vk {

// Presentation format, and the fallback for a colour target whose
// CB_COLORn_INFO encoding we do not map.
constexpr VkFormat kDefaultRtFormat = VK_FORMAT_B8G8R8A8_UNORM;

VkFormat guestTextureFormat(uint32_t dfmt, uint32_t nfmt);
bool guestFormatBlockCompressed(uint32_t dfmt);
uint32_t guestFormatElemBytes(uint32_t dfmt);
VkFormat colorTargetFormat(uint32_t info);
VkComponentMapping textureComponents(uint32_t swizzle);
uint32_t formatBytes(VkFormat fmt);

VkBlendFactor vkFactor(uint32_t f);
VkBlendOp vkBlendOp(uint32_t f);
VkPipelineColorBlendAttachmentState blendAttachment(uint32_t bc, bool en);

VkFormat vfmt(uint32_t dfmt, uint32_t nfmt);
uint32_t vfmtBytes(uint32_t dfmt);
VkPrimitiveTopology vkTopology(uint32_t prim);

void readbackPixelBgra(const uint8_t *src, VkFormat fmt, uint8_t *dst);

}  // namespace gpu::vk
