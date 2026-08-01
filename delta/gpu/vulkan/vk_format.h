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

VkFormat GuestTextureFormat(uint32_t dfmt, uint32_t nfmt);
bool GuestFormatBlockCompressed(uint32_t dfmt);
uint32_t GuestFormatElemBytes(uint32_t dfmt);
VkFormat ColorTargetFormat(uint32_t info);
VkClearColorValue ColorTargetClearValue(uint32_t info,
                                        uint32_t word0,
                                        uint32_t word1);
VkComponentMapping TextureComponents(uint32_t swizzle);
uint32_t FormatBytes(VkFormat fmt);

VkBlendFactor BlendFactor(uint32_t f);
VkBlendOp BlendOp(uint32_t f);
VkPipelineColorBlendAttachmentState BlendAttachment(uint32_t bc, bool en);

VkFormat VertexFormat(uint32_t dfmt, uint32_t nfmt);
uint32_t VertexFormatBytes(uint32_t dfmt);
VkPrimitiveTopology PrimitiveTopology(uint32_t prim);

void ReadbackPixelBgra(const uint8_t* src, VkFormat fmt, uint8_t* dst);

}  // namespace gpu::vk
