/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_rhi.h"

namespace gpu::vk {

namespace {

using rhi::Format;

constexpr VkFormat kVkFormats[] = {
    VK_FORMAT_UNDEFINED,
    VK_FORMAT_R8_UNORM,
    VK_FORMAT_R8_SNORM,
    VK_FORMAT_R8_UINT,
    VK_FORMAT_R8_SINT,
    VK_FORMAT_R8_USCALED,
    VK_FORMAT_R8_SSCALED,
    VK_FORMAT_R8G8_UNORM,
    VK_FORMAT_R8G8_SNORM,
    VK_FORMAT_R8G8_UINT,
    VK_FORMAT_R8G8_SINT,
    VK_FORMAT_R8G8_USCALED,
    VK_FORMAT_R8G8_SSCALED,
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R8G8B8A8_SNORM,
    VK_FORMAT_R8G8B8A8_UINT,
    VK_FORMAT_R8G8B8A8_SINT,
    VK_FORMAT_R8G8B8A8_USCALED,
    VK_FORMAT_R8G8B8A8_SSCALED,
    VK_FORMAT_R8G8B8A8_SRGB,
    VK_FORMAT_B8G8R8A8_UNORM,
    VK_FORMAT_B8G8R8A8_SRGB,
    VK_FORMAT_R16_UNORM,
    VK_FORMAT_R16_SNORM,
    VK_FORMAT_R16_UINT,
    VK_FORMAT_R16_SINT,
    VK_FORMAT_R16_USCALED,
    VK_FORMAT_R16_SSCALED,
    VK_FORMAT_R16_SFLOAT,
    VK_FORMAT_R16G16_UNORM,
    VK_FORMAT_R16G16_SNORM,
    VK_FORMAT_R16G16_UINT,
    VK_FORMAT_R16G16_SINT,
    VK_FORMAT_R16G16_USCALED,
    VK_FORMAT_R16G16_SSCALED,
    VK_FORMAT_R16G16_SFLOAT,
    VK_FORMAT_R16G16B16A16_UNORM,
    VK_FORMAT_R16G16B16A16_SNORM,
    VK_FORMAT_R16G16B16A16_UINT,
    VK_FORMAT_R16G16B16A16_SINT,
    VK_FORMAT_R16G16B16A16_USCALED,
    VK_FORMAT_R16G16B16A16_SSCALED,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_FORMAT_R32_UINT,
    VK_FORMAT_R32_SINT,
    VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R32G32_UINT,
    VK_FORMAT_R32G32_SINT,
    VK_FORMAT_R32G32_SFLOAT,
    VK_FORMAT_R32G32B32_SFLOAT,
    VK_FORMAT_R32G32B32A32_UINT,
    VK_FORMAT_R32G32B32A32_SINT,
    VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_B10G11R11_UFLOAT_PACK32,
    VK_FORMAT_A2B10G10R10_UNORM_PACK32,
    VK_FORMAT_A2R10G10B10_UNORM_PACK32,
    VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
    VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
    VK_FORMAT_BC2_UNORM_BLOCK,
    VK_FORMAT_BC2_SRGB_BLOCK,
    VK_FORMAT_BC3_UNORM_BLOCK,
    VK_FORMAT_BC3_SRGB_BLOCK,
    VK_FORMAT_BC4_UNORM_BLOCK,
    VK_FORMAT_BC4_SNORM_BLOCK,
    VK_FORMAT_BC5_UNORM_BLOCK,
    VK_FORMAT_BC5_SNORM_BLOCK,
    VK_FORMAT_BC6H_UFLOAT_BLOCK,
    VK_FORMAT_BC6H_SFLOAT_BLOCK,
    VK_FORMAT_BC7_UNORM_BLOCK,
    VK_FORMAT_BC7_SRGB_BLOCK,
    VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_D32_SFLOAT_S8_UINT,
    VK_FORMAT_S8_UINT,
};
static_assert(sizeof(kVkFormats) / sizeof(kVkFormats[0]) ==
                  static_cast<size_t>(Format::kCount),
              "every rhi::Format needs a VkFormat");

}  // namespace

VkFormat ToVkFormat(rhi::Format format) {
  const size_t i = static_cast<size_t>(format);
  return i < static_cast<size_t>(Format::kCount) ? kVkFormats[i]
                                                 : VK_FORMAT_UNDEFINED;
}

rhi::Format FromVkFormat(VkFormat format) {
  for (size_t i = 0; i < static_cast<size_t>(Format::kCount); ++i)
    if (kVkFormats[i] == format)
      return static_cast<Format>(i);
  return Format::kUndefined;
}

VkImageLayout ToVkLayout(rhi::TextureState state, u8 aspect) {
  using rhi::TextureState;
  const bool depth = aspect & rhi::kAspectDepth;
  const bool stencil = aspect & rhi::kAspectStencil;
  switch (state) {
    case TextureState::kUndefined:
      return VK_IMAGE_LAYOUT_UNDEFINED;
    case TextureState::kGeneral:
      return VK_IMAGE_LAYOUT_GENERAL;
    case TextureState::kColorTarget:
      return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case TextureState::kDepthTarget:
      if (depth && stencil)
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      return stencil ? VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL
                     : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    case TextureState::kDepthRead:
    case TextureState::kShaderRead:
      if (depth && stencil)
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
      if (depth)
        return VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
      if (stencil)
        return VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL;
      return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case TextureState::kCopySrc:
      return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case TextureState::kCopyDst:
      return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  }
  return VK_IMAGE_LAYOUT_GENERAL;
}

rhi::TextureState FromVkLayout(VkImageLayout layout) {
  using rhi::TextureState;
  switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
      return TextureState::kUndefined;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      return TextureState::kColorTarget;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL:
      return TextureState::kDepthTarget;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL:
      return TextureState::kDepthRead;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return TextureState::kShaderRead;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return TextureState::kCopySrc;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      return TextureState::kCopyDst;
    default:
      return TextureState::kGeneral;
  }
}

}  // namespace gpu::vk
