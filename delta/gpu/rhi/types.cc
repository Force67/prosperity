/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/rhi/types.h"

namespace gpu::rhi {

namespace {

struct FormatEntry {
  Format format;
  const char* name;
  FormatInfo info;
};

constexpr FormatInfo Plain(u8 bytes, u8 channels) {
  return {bytes, channels, false, false, false, false, false};
}
constexpr FormatInfo Int(u8 bytes, u8 channels) {
  return {bytes, channels, false, true, false, false, false};
}
constexpr FormatInfo Srgb(u8 bytes, u8 channels) {
  return {bytes, channels, false, false, false, false, true};
}
constexpr FormatInfo Block(u8 bytes, u8 channels, bool srgb = false) {
  return {bytes, channels, true, false, false, false, srgb};
}

#define F(name) Format::k##name, #name
constexpr FormatEntry kFormats[] = {
    {F(Undefined), {}},
    {F(R8Unorm), Plain(1, 1)},       {F(R8Snorm), Plain(1, 1)},
    {F(R8Uint), Int(1, 1)},          {F(R8Sint), Int(1, 1)},
    {F(R8Uscaled), Plain(1, 1)},     {F(R8Sscaled), Plain(1, 1)},
    {F(RG8Unorm), Plain(2, 2)},      {F(RG8Snorm), Plain(2, 2)},
    {F(RG8Uint), Int(2, 2)},         {F(RG8Sint), Int(2, 2)},
    {F(RG8Uscaled), Plain(2, 2)},    {F(RG8Sscaled), Plain(2, 2)},
    {F(RGBA8Unorm), Plain(4, 4)},    {F(RGBA8Snorm), Plain(4, 4)},
    {F(RGBA8Uint), Int(4, 4)},       {F(RGBA8Sint), Int(4, 4)},
    {F(RGBA8Uscaled), Plain(4, 4)},  {F(RGBA8Sscaled), Plain(4, 4)},
    {F(RGBA8Srgb), Srgb(4, 4)},
    {F(BGRA8Unorm), Plain(4, 4)},    {F(BGRA8Srgb), Srgb(4, 4)},
    {F(R16Unorm), Plain(2, 1)},      {F(R16Snorm), Plain(2, 1)},
    {F(R16Uint), Int(2, 1)},         {F(R16Sint), Int(2, 1)},
    {F(R16Uscaled), Plain(2, 1)},    {F(R16Sscaled), Plain(2, 1)},
    {F(R16Float), Plain(2, 1)},
    {F(RG16Unorm), Plain(4, 2)},     {F(RG16Snorm), Plain(4, 2)},
    {F(RG16Uint), Int(4, 2)},        {F(RG16Sint), Int(4, 2)},
    {F(RG16Uscaled), Plain(4, 2)},   {F(RG16Sscaled), Plain(4, 2)},
    {F(RG16Float), Plain(4, 2)},
    {F(RGBA16Unorm), Plain(8, 4)},   {F(RGBA16Snorm), Plain(8, 4)},
    {F(RGBA16Uint), Int(8, 4)},      {F(RGBA16Sint), Int(8, 4)},
    {F(RGBA16Uscaled), Plain(8, 4)}, {F(RGBA16Sscaled), Plain(8, 4)},
    {F(RGBA16Float), Plain(8, 4)},
    {F(R32Uint), Int(4, 1)},         {F(R32Sint), Int(4, 1)},
    {F(R32Float), Plain(4, 1)},
    {F(RG32Uint), Int(8, 2)},        {F(RG32Sint), Int(8, 2)},
    {F(RG32Float), Plain(8, 2)},
    {F(RGB32Float), Plain(12, 3)},
    {F(RGBA32Uint), Int(16, 4)},     {F(RGBA32Sint), Int(16, 4)},
    {F(RGBA32Float), Plain(16, 4)},
    {F(B10G11R11Float), Plain(4, 3)},
    {F(A2B10G10R10Unorm), Plain(4, 4)},
    {F(A2R10G10B10Unorm), Plain(4, 4)},
    {F(BC1Unorm), Block(8, 4)},      {F(BC1Srgb), Block(8, 4, true)},
    {F(BC2Unorm), Block(16, 4)},     {F(BC2Srgb), Block(16, 4, true)},
    {F(BC3Unorm), Block(16, 4)},     {F(BC3Srgb), Block(16, 4, true)},
    {F(BC4Unorm), Block(8, 1)},      {F(BC4Snorm), Block(8, 1)},
    {F(BC5Unorm), Block(16, 2)},     {F(BC5Snorm), Block(16, 2)},
    {F(BC6HUfloat), Block(16, 3)},   {F(BC6HSfloat), Block(16, 3)},
    {F(BC7Unorm), Block(16, 4)},     {F(BC7Srgb), Block(16, 4, true)},
    {F(D32Float), {4, 1, false, false, true, false, false}},
    {F(D32FloatS8Uint), {8, 2, false, false, true, true, false}},
    {F(S8Uint), {1, 1, false, true, false, true, false}},
};
#undef F

static_assert(sizeof(kFormats) / sizeof(kFormats[0]) ==
                  static_cast<size_t>(Format::kCount),
              "every Format needs an entry");

constexpr bool InOrder() {
  for (size_t i = 0; i < sizeof(kFormats) / sizeof(kFormats[0]); ++i)
    if (static_cast<size_t>(kFormats[i].format) != i)
      return false;
  return true;
}
static_assert(InOrder(), "kFormats must be in enum order");

const FormatEntry& Entry(Format format) {
  const size_t i = static_cast<size_t>(format);
  return kFormats[i < static_cast<size_t>(Format::kCount) ? i : 0];
}

}  // namespace

const char* BackendName(Backend backend) {
  switch (backend) {
    case Backend::kVulkan:
      return "vulkan";
    case Backend::kOpenGL:
      return "opengl";
    case Backend::kD3D12:
      return "d3d12";
  }
  return "?";
}

const FormatInfo& GetFormatInfo(Format format) {
  return Entry(format).info;
}

const char* FormatName(Format format) {
  return Entry(format).name;
}

}  // namespace gpu::rhi
