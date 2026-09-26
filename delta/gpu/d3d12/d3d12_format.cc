/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include <array>

#include "gpu/d3d12/d3d12_internal.h"

namespace gpu::d3d12::impl {

namespace {

using rhi::Format;

DxgiInfo Plain(DXGI_FORMAT f, DXGI_FORMAT typeless) {
  DxgiInfo i;
  i.format = f;
  i.typeless = typeless;
  i.srv = f;
  return i;
}

// A vertex-only format: DXGI has no texture equivalent.
DxgiInfo Scaled(DXGI_FORMAT integer, u8 kind) {
  DxgiInfo i;
  i.format = integer;
  i.scaled = kind;
  i.texture = false;
  return i;
}

DxgiInfo Build(Format f) {
  switch (f) {
#define P(rhi, dxgi, tl) \
  case Format::k##rhi:   \
    return Plain(DXGI_FORMAT_##dxgi, DXGI_FORMAT_##tl);
    P(R8Unorm, R8_UNORM, R8_TYPELESS)
    P(R8Snorm, R8_SNORM, R8_TYPELESS)
    P(R8Uint, R8_UINT, R8_TYPELESS)
    P(R8Sint, R8_SINT, R8_TYPELESS)
    P(RG8Unorm, R8G8_UNORM, R8G8_TYPELESS)
    P(RG8Snorm, R8G8_SNORM, R8G8_TYPELESS)
    P(RG8Uint, R8G8_UINT, R8G8_TYPELESS)
    P(RG8Sint, R8G8_SINT, R8G8_TYPELESS)
    P(RGBA8Unorm, R8G8B8A8_UNORM, R8G8B8A8_TYPELESS)
    P(RGBA8Snorm, R8G8B8A8_SNORM, R8G8B8A8_TYPELESS)
    P(RGBA8Uint, R8G8B8A8_UINT, R8G8B8A8_TYPELESS)
    P(RGBA8Sint, R8G8B8A8_SINT, R8G8B8A8_TYPELESS)
    P(RGBA8Srgb, R8G8B8A8_UNORM_SRGB, R8G8B8A8_TYPELESS)
    P(BGRA8Unorm, B8G8R8A8_UNORM, B8G8R8A8_TYPELESS)
    P(BGRA8Srgb, B8G8R8A8_UNORM_SRGB, B8G8R8A8_TYPELESS)
    P(R16Unorm, R16_UNORM, R16_TYPELESS)
    P(R16Snorm, R16_SNORM, R16_TYPELESS)
    P(R16Uint, R16_UINT, R16_TYPELESS)
    P(R16Sint, R16_SINT, R16_TYPELESS)
    P(R16Float, R16_FLOAT, R16_TYPELESS)
    P(RG16Unorm, R16G16_UNORM, R16G16_TYPELESS)
    P(RG16Snorm, R16G16_SNORM, R16G16_TYPELESS)
    P(RG16Uint, R16G16_UINT, R16G16_TYPELESS)
    P(RG16Sint, R16G16_SINT, R16G16_TYPELESS)
    P(RG16Float, R16G16_FLOAT, R16G16_TYPELESS)
    P(RGBA16Unorm, R16G16B16A16_UNORM, R16G16B16A16_TYPELESS)
    P(RGBA16Snorm, R16G16B16A16_SNORM, R16G16B16A16_TYPELESS)
    P(RGBA16Uint, R16G16B16A16_UINT, R16G16B16A16_TYPELESS)
    P(RGBA16Sint, R16G16B16A16_SINT, R16G16B16A16_TYPELESS)
    P(RGBA16Float, R16G16B16A16_FLOAT, R16G16B16A16_TYPELESS)
    P(R32Uint, R32_UINT, R32_TYPELESS)
    P(R32Sint, R32_SINT, R32_TYPELESS)
    P(R32Float, R32_FLOAT, R32_TYPELESS)
    P(RG32Uint, R32G32_UINT, R32G32_TYPELESS)
    P(RG32Sint, R32G32_SINT, R32G32_TYPELESS)
    P(RG32Float, R32G32_FLOAT, R32G32_TYPELESS)
    P(RGB32Float, R32G32B32_FLOAT, R32G32B32_TYPELESS)
    P(RGBA32Uint, R32G32B32A32_UINT, R32G32B32A32_TYPELESS)
    P(RGBA32Sint, R32G32B32A32_SINT, R32G32B32A32_TYPELESS)
    P(RGBA32Float, R32G32B32A32_FLOAT, R32G32B32A32_TYPELESS)
    P(B10G11R11Float, R11G11B10_FLOAT, R11G11B10_FLOAT)
    P(A2B10G10R10Unorm, R10G10B10A2_UNORM, R10G10B10A2_TYPELESS)
    P(BC1Unorm, BC1_UNORM, BC1_TYPELESS)
    P(BC1Srgb, BC1_UNORM_SRGB, BC1_TYPELESS)
    P(BC2Unorm, BC2_UNORM, BC2_TYPELESS)
    P(BC2Srgb, BC2_UNORM_SRGB, BC2_TYPELESS)
    P(BC3Unorm, BC3_UNORM, BC3_TYPELESS)
    P(BC3Srgb, BC3_UNORM_SRGB, BC3_TYPELESS)
    P(BC4Unorm, BC4_UNORM, BC4_TYPELESS)
    P(BC4Snorm, BC4_SNORM, BC4_TYPELESS)
    P(BC5Unorm, BC5_UNORM, BC5_TYPELESS)
    P(BC5Snorm, BC5_SNORM, BC5_TYPELESS)
    P(BC6HUfloat, BC6H_UF16, BC6H_TYPELESS)
    P(BC6HSfloat, BC6H_SF16, BC6H_TYPELESS)
    P(BC7Unorm, BC7_UNORM, BC7_TYPELESS)
    P(BC7Srgb, BC7_UNORM_SRGB, BC7_TYPELESS)
#undef P
    case Format::kR8Uscaled:
      return Scaled(DXGI_FORMAT_R8_UINT, 1);
    case Format::kR8Sscaled:
      return Scaled(DXGI_FORMAT_R8_SINT, 2);
    case Format::kRG8Uscaled:
      return Scaled(DXGI_FORMAT_R8G8_UINT, 1);
    case Format::kRG8Sscaled:
      return Scaled(DXGI_FORMAT_R8G8_SINT, 2);
    case Format::kRGBA8Uscaled:
      return Scaled(DXGI_FORMAT_R8G8B8A8_UINT, 1);
    case Format::kRGBA8Sscaled:
      return Scaled(DXGI_FORMAT_R8G8B8A8_SINT, 2);
    case Format::kR16Uscaled:
      return Scaled(DXGI_FORMAT_R16_UINT, 1);
    case Format::kR16Sscaled:
      return Scaled(DXGI_FORMAT_R16_SINT, 2);
    case Format::kRG16Uscaled:
      return Scaled(DXGI_FORMAT_R16G16_UINT, 1);
    case Format::kRG16Sscaled:
      return Scaled(DXGI_FORMAT_R16G16_SINT, 2);
    case Format::kRGBA16Uscaled:
      return Scaled(DXGI_FORMAT_R16G16B16A16_UINT, 1);
    case Format::kRGBA16Sscaled:
      return Scaled(DXGI_FORMAT_R16G16B16A16_SINT, 2);
    case Format::kA2R10G10B10Unorm: {
      DxgiInfo i = Scaled(DXGI_FORMAT_R10G10B10A2_UNORM, 0);
      i.swap_rb = true;
      return i;
    }
    case Format::kD32Float: {
      DxgiInfo i = Plain(DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_TYPELESS);
      i.srv = DXGI_FORMAT_R32_FLOAT;
      return i;
    }
    case Format::kD32FloatS8Uint: {
      DxgiInfo i = Plain(DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
                         DXGI_FORMAT_R32G8X24_TYPELESS);
      i.srv = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
      i.stencil_srv = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
      return i;
    }
    case Format::kS8Uint:  // DXGI has no stencil-only format
    case Format::kUndefined:
    case Format::kCount:
      break;
  }
  DxgiInfo none;
  none.texture = false;
  return none;
}

}  // namespace

const DxgiInfo& Dxgi(Format format) {
  static const auto table = [] {
    std::array<DxgiInfo, static_cast<size_t>(Format::kCount)> t;
    for (size_t i = 0; i < t.size(); i++)
      t[i] = Build(static_cast<Format>(i));
    return t;
  }();
  const size_t i = static_cast<size_t>(format);
  return table[i < table.size() ? i : 0];
}

DXGI_FORMAT VertexFormat(Format format) {
  return Dxgi(format).format;
}

}  // namespace gpu::d3d12::impl
