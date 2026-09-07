/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Translation of the guest GPU's surface, vertex, blend and primitive encodings
// into their Vulkan equivalents, plus the conversion of a readback texel back
// to BGRA8. Pure tables: no device state, no caches.

#include <vulkan/vulkan.h>
#include "base/arch.h"


namespace gpu::vk {

// Presentation format, and the fallback for a colour target whose
// CB_COLORn_INFO encoding we do not map.
constexpr VkFormat kDefaultRtFormat = VK_FORMAT_B8G8R8A8_UNORM;

VkFormat GuestTextureFormat(u32 dfmt, u32 nfmt);
bool GuestFormatBlockCompressed(u32 dfmt);
bool FormatBlockCompressed(VkFormat format);
u32 GuestFormatElemBytes(u32 dfmt);
VkFormat ColorTargetFormat(u32 info);

// A colour target whose texels are integers. The fragment shader must declare
// an integer output for one, blending is not allowed on one, and a sampler
// reading one may not filter -- three rules that all key off this.
bool IsIntegerColorFormat(VkFormat format);
VkClearColorValue ColorTargetClearValue(u32 info,
                                        u32 word0,
                                        u32 word1);
VkComponentMapping TextureComponents(u32 swizzle);
u32 FormatBytes(VkFormat fmt);

VkBlendFactor BlendFactor(u32 f);
VkBlendOp BlendOp(u32 f);
VkPipelineColorBlendAttachmentState BlendAttachment(u32 bc, bool en);
// shader_mask == 0 means the frontend supplies only the recompiler's export
// mask. The target's per-component write mask still applies independently.
VkColorComponentFlags ColorWriteMask(u32 target_mask, u32 shader_mask,
                                     u8 export_mask, u32 target);

VkFormat VertexFormat(u32 dfmt, u32 nfmt);
u32 VertexFormatBytes(u32 dfmt);
VkPrimitiveTopology PrimitiveTopology(u32 prim);

void ReadbackPixelBgra(const u8* src, VkFormat fmt, u8* dst);

}  // namespace gpu::vk
