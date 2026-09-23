/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/rhi/types.h"

namespace gpu::rhi {
namespace {

struct FormatInfo {
  const char* name;
  u8 texel_bytes;  // bytes per texel, or per 4x4 block when compressed
  u8 block_dim;    // 1, or 4 for BC
  bool depth;
  bool stencil;
};

// Indexed by Format. One row per enumerator, in enumerator order; the static
// assert below is what keeps them in step.
constexpr FormatInfo kFormats[] = {
    {"Unknown", 0, 1, false, false},
    {"R8Unorm", 1, 1, false, false},
    {"R8Uint", 1, 1, false, false},
    {"RG8Unorm", 2, 1, false, false},
    {"RGBA8Unorm", 4, 1, false, false},
    {"RGBA8Srgb", 4, 1, false, false},
    {"BGRA8Unorm", 4, 1, false, false},
    {"BGRA8Srgb", 4, 1, false, false},
    {"R16Uint", 2, 1, false, false},
    {"R16Float", 2, 1, false, false},
    {"RG16Float", 4, 1, false, false},
    {"RGBA16Float", 8, 1, false, false},
    {"RGBA16Unorm", 8, 1, false, false},
    {"R32Uint", 4, 1, false, false},
    {"R32Float", 4, 1, false, false},
    {"RG32Float", 8, 1, false, false},
    {"RGB32Float", 12, 1, false, false},
    {"RGBA32Float", 16, 1, false, false},
    {"RGB10A2Unorm", 4, 1, false, false},
    {"RG11B10Float", 4, 1, false, false},
    {"D16Unorm", 2, 1, true, false},
    {"D32Float", 4, 1, true, false},
    {"D32FloatS8Uint", 8, 1, true, true},
    {"BC1Unorm", 8, 4, false, false},
    {"BC2Unorm", 16, 4, false, false},
    {"BC3Unorm", 16, 4, false, false},
    {"BC4Unorm", 8, 4, false, false},
    {"BC5Unorm", 16, 4, false, false},
    {"BC7Unorm", 16, 4, false, false},
};

static_assert(sizeof(kFormats) / sizeof(kFormats[0]) ==
                  static_cast<u32>(Format::kCount),
              "the format table and the Format enum disagree");

const FormatInfo& Info(Format f) {
  const u32 i = static_cast<u32>(f);
  return kFormats[i < static_cast<u32>(Format::kCount) ? i : 0];
}

}  // namespace

u32 FormatTexelBytes(Format f) {
  return Info(f).texel_bytes;
}

u32 FormatBlockDim(Format f) {
  return Info(f).block_dim;
}

bool FormatIsDepth(Format f) {
  return Info(f).depth;
}

bool FormatHasStencil(Format f) {
  return Info(f).stencil;
}

bool FormatIsCompressed(Format f) {
  return Info(f).block_dim > 1;
}

const char* FormatName(Format f) {
  return Info(f).name;
}

u64 FormatRowBytes(Format f, u32 width) {
  const FormatInfo& info = Info(f);
  const u64 blocks = (width + info.block_dim - 1) / info.block_dim;
  return blocks * info.texel_bytes;
}

}  // namespace gpu::rhi
