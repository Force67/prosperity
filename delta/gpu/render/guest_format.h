/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Translation of the guest GPU's surface, vertex, blend and primitive encodings
// into their Vulkan equivalents, plus the conversion of a readback texel back
// to BGRA8. Pure tables: no device state, no caches.

#include "base/arch.h"
#include "gpu/rhi/types.h"

namespace gpu::render {

// Presentation format, and the fallback for a colour target whose
// CB_COLORn_INFO encoding we do not map.
constexpr rhi::Format kDefaultRtFormat = rhi::Format::kBGRA8Unorm;

rhi::Format GuestTextureFormat(u32 dfmt, u32 nfmt);
bool GuestFormatBlockCompressed(u32 dfmt);
bool FormatBlockCompressed(rhi::Format format);
u32 GuestFormatElemBytes(u32 dfmt);
rhi::Format ColorTargetFormat(u32 info);

// A colour target whose texels are integers. The fragment shader must declare
// an integer output for one, blending is not allowed on one, and a sampler
// reading one may not filter. Three rules that all key off this.
bool IsIntegerColorFormat(rhi::Format format);
rhi::ClearColor ColorTargetClearValue(u32 info, u32 word0, u32 word1);
// The same T# DST_SEL as a view swizzle.
void TextureSwizzle(u32 swizzle, rhi::Swizzle out[4]);
u32 FormatBytes(rhi::Format fmt);

rhi::BlendFactor BlendFactor(u32 f);
rhi::BlendOp BlendOp(u32 f);
rhi::BlendAttachment BlendAttachment(u32 bc, bool en);
// shader_mask == 0 means the frontend supplies only the recompiler's export
// mask. The target's per-component write mask still applies independently.
u8 ColorWriteMask(u32 target_mask, u32 shader_mask, u8 export_mask, u32 target);

rhi::Format VertexFormat(u32 dfmt, u32 nfmt);
u32 VertexFormatBytes(u32 dfmt);
rhi::Topology PrimitiveTopology(u32 prim);

void ReadbackPixelBgra(const u8* src, rhi::Format fmt, u8* dst);

}  // namespace gpu::render
