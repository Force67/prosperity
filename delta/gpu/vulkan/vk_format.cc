/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_format.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utl/options.h>

namespace {
// A colour target whose NUMBER_TYPE is UINT/SINT holds packed bits, not a
// colour. Mapping one to UNORM clamps every export into [0,1] and the target
// reads back black -- SotC lost two whole G-buffer planes that way. On by
// default; DELTA_GPU_INT_RT=0 restores the old UNORM mapping.
DELTA_OPTION(bool, kIntegerRt, "DELTA_GPU_INT_RT", true);
DELTA_OPTION(bool, kNoBlend, "DELTA_GPU_NOBLEND", false);
DELTA_OPTION(bool, kNoSwizzle, "DELTA_GPU_NOSWIZZLE", false);
}  // namespace

namespace gpu::vk {
namespace {

float HalfToFloat(uint16_t value) {
  uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
  uint32_t exponent = (value >> 10) & 0x1F;
  uint32_t mantissa = value & 0x3FF;
  uint32_t bits;
  if (!exponent) {
    if (!mantissa) {
      bits = sign;
    } else {
      int unbiased = -14;
      while (!(mantissa & 0x400)) {
        mantissa <<= 1;
        unbiased--;
      }
      bits = sign | static_cast<uint32_t>(unbiased + 127) << 23 |
             (mantissa & 0x3FF) << 13;
    }
  } else if (exponent == 0x1F) {
    bits = sign | 0x7F800000 | mantissa << 13;
  } else {
    bits = sign | (exponent + (127 - 15)) << 23 | mantissa << 13;
  }
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint8_t Unorm8(float value) {
  if (!std::isfinite(value))
    return 0;
  return static_cast<uint8_t>(
      std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

float PackedUfloat(uint32_t value, uint32_t mantissa_bits) {
  uint32_t mantissa_mask = (1u << mantissa_bits) - 1;
  uint32_t mantissa = value & mantissa_mask;
  uint32_t exponent = value >> mantissa_bits;
  if (!exponent)
    return std::ldexp(static_cast<float>(mantissa),
                      -14 - static_cast<int>(mantissa_bits));
  if (exponent == 0x1F)
    return mantissa ? NAN : INFINITY;
  return std::ldexp(1.0f + static_cast<float>(mantissa) / (1u << mantissa_bits),
                    static_cast<int>(exponent) - 15);
}

int32_t SignExtend(uint32_t value, uint32_t bits) {
  return static_cast<int32_t>(value << (32 - bits)) >> (32 - bits);
}

}  // namespace

VkFormat GuestTextureFormat(uint32_t dfmt, uint32_t nfmt) {
  // Narrow and float channel counts: RDNA2 titles sample single-channel masks
  // and packed HDR buffers that the PS4 titles never used.
  if (dfmt == 1 && nfmt == 0)
    return VK_FORMAT_R8_UNORM;
  if (dfmt == 1 && nfmt == 1)
    return VK_FORMAT_R8_SNORM;
  if (dfmt == 1 && nfmt == 4)
    return VK_FORMAT_R8_UINT;
  if (dfmt == 2 && nfmt == 0)
    return VK_FORMAT_R16_UNORM;
  if (dfmt == 2 && nfmt == 4)
    return VK_FORMAT_R16_UINT;
  if (dfmt == 2 && nfmt == 7)
    return VK_FORMAT_R16_SFLOAT;
  if (dfmt == 3 && nfmt == 0)
    return VK_FORMAT_R8G8_UNORM;
  if (dfmt == 3 && nfmt == 4)
    return VK_FORMAT_R8G8_UINT;
  if (dfmt == 4 && nfmt == 4)
    return VK_FORMAT_R32_UINT;
  if (dfmt == 4 && nfmt == 7)
    return VK_FORMAT_R32_SFLOAT;
  if (dfmt == 5 && nfmt == 0)
    return VK_FORMAT_R16G16_UNORM;
  if (dfmt == 5 && nfmt == 4)
    return VK_FORMAT_R16G16_UINT;
  if (dfmt == 8 && nfmt == 0)
    return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
  if (dfmt == 9 && nfmt == 0)
    return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
  if (dfmt == 10 && nfmt == 1)
    return VK_FORMAT_R8G8B8A8_SNORM;
  if (dfmt == 10 && nfmt == 4)
    return VK_FORMAT_R8G8B8A8_UINT;
  if (dfmt == 11 && nfmt == 4)
    return VK_FORMAT_R32G32_UINT;
  if (dfmt == 11 && nfmt == 7)
    return VK_FORMAT_R32G32_SFLOAT;
  if (dfmt == 12 && nfmt == 0)
    return VK_FORMAT_R16G16B16A16_UNORM;
  if (dfmt == 14 && nfmt == 7)
    return VK_FORMAT_R32G32B32A32_SFLOAT;
  if (dfmt == 5 && nfmt == 7)
    return VK_FORMAT_R16G16_SFLOAT;
  if (dfmt == 6 && nfmt == 7)
    return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
  if (dfmt == 12 && nfmt == 7)
    return VK_FORMAT_R16G16B16A16_SFLOAT;
  if (dfmt == 10 && nfmt == 0)
    return VK_FORMAT_R8G8B8A8_UNORM;
  if (dfmt == 10 && nfmt == 9)
    return VK_FORMAT_R8G8B8A8_SRGB;
  // Block-compressed (IMG_DATA_FORMAT_BC1..BC7 = 35..41); sampled natively.
  if (dfmt == 35)
    return nfmt == 9 ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK
                     : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
  if (dfmt == 36)
    return nfmt == 9 ? VK_FORMAT_BC2_SRGB_BLOCK : VK_FORMAT_BC2_UNORM_BLOCK;
  if (dfmt == 37)
    return nfmt == 9 ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK;
  if (dfmt == 38)
    return nfmt == 1 ? VK_FORMAT_BC4_SNORM_BLOCK : VK_FORMAT_BC4_UNORM_BLOCK;
  if (dfmt == 39)
    return nfmt == 1 ? VK_FORMAT_BC5_SNORM_BLOCK : VK_FORMAT_BC5_UNORM_BLOCK;
  if (dfmt == 41)
    return nfmt == 9 ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
  return VK_FORMAT_UNDEFINED;
}

// Block-compressed guest formats: 4x4-texel blocks of 8 or 16 bytes. The
// detiler and upload then work in block ("element") space.
bool GuestFormatBlockCompressed(uint32_t dfmt) {
  return dfmt >= 35 && dfmt <= 41;
}

uint32_t GuestFormatElemBytes(uint32_t dfmt) {
  switch (dfmt) {
    case 1:
      return 1;  // 8
    case 2:
    case 3:
      return 2;  // 16, 8_8
    case 11:
    case 12:
      return 8;  // 32_32, 16_16_16_16
    case 13:
      return 12;  // 32_32_32
    case 14:
      return 16;  // 32_32_32_32
    case 35:
    case 38:
      return 8;  // BC1/BC4 block
    default:
      return GuestFormatBlockCompressed(dfmt) ? 16  // BC2/3/5/6/7 block
                                              : 4;  // all 32-bit texel formats
  }
}

// CB_COLORn_INFO uses the GFX7 SurfaceFormat/SurfaceNumber encodings. Keep the
// established BGRA8 host path for 8_8_8_8 targets; floating-point effect
// buffers must retain their native precision or HDR/negative values clamp to
// black.
bool IsIntegerColorFormat(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT:
      return true;
    default:
      return false;
  }
}

VkFormat ColorTargetFormat(uint32_t info) {
  uint32_t dfmt = (info >> 2) & 0x1F;
  uint32_t nfmt = (info >> 8) & 0x7;
  if (nfmt == 7) {
    switch (dfmt) {
      case 2:
        return VK_FORMAT_R16_SFLOAT;
      case 4:
        return VK_FORMAT_R32_SFLOAT;
      case 5:
        return VK_FORMAT_R16G16_SFLOAT;
      case 6:
        return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
      case 11:
        return VK_FORMAT_R32G32_SFLOAT;
      case 12:
        return VK_FORMAT_R16G16B16A16_SFLOAT;
      case 13:
        return VK_FORMAT_R32G32B32_SFLOAT;
      case 14:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
      default:
        break;
    }
  }
  if (kIntegerRt && (nfmt == 4 || nfmt == 5)) {
    const bool sint = nfmt == 5;
    switch (dfmt) {
      case 1:
        return sint ? VK_FORMAT_R8_SINT : VK_FORMAT_R8_UINT;
      case 2:
        return sint ? VK_FORMAT_R16_SINT : VK_FORMAT_R16_UINT;
      case 3:
        return sint ? VK_FORMAT_R8G8_SINT : VK_FORMAT_R8G8_UINT;
      case 4:
        return sint ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_UINT;
      case 5:
        return sint ? VK_FORMAT_R16G16_SINT : VK_FORMAT_R16G16_UINT;
      case 10:
        return sint ? VK_FORMAT_R8G8B8A8_SINT : VK_FORMAT_R8G8B8A8_UINT;
      case 11:
        return sint ? VK_FORMAT_R32G32_SINT : VK_FORMAT_R32G32_UINT;
      case 12:
        return sint ? VK_FORMAT_R16G16B16A16_SINT : VK_FORMAT_R16G16B16A16_UINT;
      case 14:
        return sint ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_UINT;
      default:
        break;
    }
  }
  if (nfmt == 0) {
    switch (dfmt) {
      case 1:
        return VK_FORMAT_R8_UNORM;
      case 2:
        return VK_FORMAT_R16_UNORM;
      case 3:
        return VK_FORMAT_R8G8_UNORM;
      case 5:
        return VK_FORMAT_R16G16_UNORM;
      case 10:
        return VK_FORMAT_B8G8R8A8_UNORM;
      case 12:
        return VK_FORMAT_R16G16B16A16_UNORM;
      default:
        break;
    }
  }
  return kDefaultRtFormat;
}

// CB_COLORn_CLEAR_WORD0/1 hold the fast-clear colour already encoded in the
// target's own surface format, component 0 at bit 0. Vulkan takes a clear value
// in R,G,B,A component order and applies the format's channel order itself, so
// the guest components map straight across even though ColorTargetFormat picks
// a BGRA host format for 8_8_8_8.
VkClearColorValue ColorTargetClearValue(uint32_t info,
                                        uint32_t word0,
                                        uint32_t word1) {
  const uint32_t dfmt = (info >> 2) & 0x1F;
  const uint32_t nfmt = (info >> 8) & 0x7;
  const auto unmapped = [&](const char* what) {
    static int n = 0;
    if (n++ < 8)
      std::fprintf(stderr,
                   "[gpuvk] clear word: unmapped %s (info=%#x dfmt=%u nfmt=%u "
                   "words %08x %08x), clearing to opaque black\n",
                   what, info, dfmt, nfmt, word0, word1);
    return VkClearColorValue{{0.0f, 0.0f, 0.0f, 1.0f}};
  };
  uint32_t width[4] = {0, 0, 0, 0};
  switch (dfmt) {
    case 1:  // 8
      width[0] = 8;
      break;
    case 2:  // 16
      width[0] = 16;
      break;
    case 3:  // 8_8
      width[0] = width[1] = 8;
      break;
    case 4:  // 32
      width[0] = 32;
      break;
    case 5:  // 16_16
      width[0] = width[1] = 16;
      break;
    case 6:  // 10_11_11
      width[0] = width[1] = 11;
      width[2] = 10;
      break;
    case 9:  // 2_10_10_10
      width[0] = width[1] = width[2] = 10;
      width[3] = 2;
      break;
    case 10:  // 8_8_8_8
      width[0] = width[1] = width[2] = width[3] = 8;
      break;
    case 11:  // 32_32
      width[0] = width[1] = 32;
      break;
    case 12:  // 16_16_16_16
      width[0] = width[1] = width[2] = width[3] = 16;
      break;
    case 13:  // 32_32_32
    case 14:  // 32_32_32_32
      // The clear words are 64 bits, so only the low two components of a wider
      // texel are representable at all.
      width[0] = width[1] = 32;
      break;
    default:
      return unmapped("colour format");
  }
  const uint64_t packed =
      static_cast<uint64_t>(word0) | static_cast<uint64_t>(word1) << 32;
  VkClearColorValue out{};
  // Seed opaque: a format with fewer than four components never writes alpha,
  // and a transparent target is a hole in a deferred composite.
  if (nfmt == 4 || nfmt == 5)
    out.uint32[3] = 1;
  else
    out.float32[3] = 1.0f;
  uint32_t shift = 0;
  for (uint32_t i = 0; i < 4 && width[i]; i++) {
    const uint32_t bits = width[i];
    const uint64_t mask = bits == 32 ? 0xFFFFFFFFull : (1ull << bits) - 1;
    const uint32_t raw = static_cast<uint32_t>((packed >> shift) & mask);
    shift += bits;
    switch (nfmt) {
      case 0:  // UNORM
      case 6:  // SRGB: ColorTargetFormat gives it a UNORM host image, so the
               // encoded value has to pass through unconverted
        out.float32[i] = static_cast<float>(raw) / static_cast<float>(mask);
        break;
      case 1:  // SNORM
        out.float32[i] =
            std::max(static_cast<float>(SignExtend(raw, bits)) /
                         static_cast<float>((1u << (bits - 1)) - 1),
                     -1.0f);
        break;
      case 4:  // UINT
        out.uint32[i] = raw;
        break;
      case 5:  // SINT
        out.int32[i] = SignExtend(raw, bits);
        break;
      case 7:  // FLOAT
        if (bits == 32)
          std::memcpy(&out.float32[i], &raw, sizeof(raw));
        else if (bits == 16)
          out.float32[i] = HalfToFloat(static_cast<uint16_t>(raw));
        else if (bits == 11 || bits == 10)
          out.float32[i] = PackedUfloat(raw, bits - 5);
        else
          return unmapped("float width");
        break;
      default:
        return unmapped("number type");
    }
  }
  return out;
}

VkComponentMapping TextureComponents(uint32_t swizzle) {
  if (!swizzle || kNoSwizzle)
    return {};
  const auto comp = [](uint32_t sel) {
    switch (sel) {
      case 0:
        return VK_COMPONENT_SWIZZLE_ZERO;
      case 1:
        return VK_COMPONENT_SWIZZLE_ONE;
      case 4:
        return VK_COMPONENT_SWIZZLE_R;
      case 5:
        return VK_COMPONENT_SWIZZLE_G;
      case 6:
        return VK_COMPONENT_SWIZZLE_B;
      case 7:
        return VK_COMPONENT_SWIZZLE_A;
      default:
        return VK_COMPONENT_SWIZZLE_IDENTITY;
    }
  };
  return {comp(swizzle & 7), comp((swizzle >> 3) & 7), comp((swizzle >> 6) & 7),
          comp((swizzle >> 9) & 7)};
}

uint32_t FormatBytes(VkFormat fmt) {
  switch (fmt) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
      return 1;
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT:
      return 2;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT:
      return 8;
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT:
      return 16;
    case VK_FORMAT_R32G32B32_SFLOAT:
      return 12;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return 16;
    default:
      return 4;
  }
}

// GNM blend multiplier (CB_BLENDn_CONTROL factor field) -> Vulkan blend factor.
VkBlendFactor BlendFactor(uint32_t f) {
  switch (f) {
    case 0:
      return VK_BLEND_FACTOR_ZERO;
    case 1:
      return VK_BLEND_FACTOR_ONE;
    case 2:
      return VK_BLEND_FACTOR_SRC_COLOR;
    case 3:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 4:
      return VK_BLEND_FACTOR_SRC_ALPHA;
    case 5:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 6:
      return VK_BLEND_FACTOR_DST_ALPHA;
    case 7:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 8:
      return VK_BLEND_FACTOR_DST_COLOR;
    case 9:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 10:
      return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 11:
      return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 12:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 13:
      return VK_BLEND_FACTOR_SRC1_COLOR;
    case 14:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 15:
      return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 16:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 17:
      return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 18:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default:
      return VK_BLEND_FACTOR_ONE;
  }
}

// GNM blend function (combine fcn) -> Vulkan blend op.
VkBlendOp BlendOp(uint32_t f) {
  switch (f) {
    case 0:
      return VK_BLEND_OP_ADD;
    case 1:
      return VK_BLEND_OP_SUBTRACT;
    case 2:
      return VK_BLEND_OP_MIN;
    case 3:
      return VK_BLEND_OP_MAX;
    case 4:
      return VK_BLEND_OP_REVERSE_SUBTRACT;
    default:
      return VK_BLEND_OP_ADD;
  }
}

// Decode CB_BLEND0_CONTROL into a Vulkan colour-blend attachment. `en` is the
// per-target blend enable (bit 30). Falls back to a sensible src-alpha blend
// when the guest enables blend but the control word is zero (default state, not
// yet set).
VkPipelineColorBlendAttachmentState BlendAttachment(uint32_t bc, bool en) {
  VkPipelineColorBlendAttachmentState cba{};
  cba.colorWriteMask = 0xF;
  // DELTA_GPU_NOBLEND: force opaque (diagnostic) to test whether a draw
  // vanishes because its src-alpha blend multiplies by a zero texel alpha
  // (Doom64 3D walls).
  if (kNoBlend)
    en = false;
  if (!en) {
    cba.blendEnable = VK_FALSE;
    return cba;
  }
  cba.blendEnable = VK_TRUE;
  uint32_t cs = bc & 0x1F, cf = (bc >> 5) & 0x7, cd = (bc >> 8) & 0x1F;
  bool sep = (bc >> 29) & 1;
  uint32_t as = sep ? (bc >> 16) & 0x1F : cs;
  uint32_t af = sep ? (bc >> 21) & 0x7 : cf;
  uint32_t ad = sep ? (bc >> 24) & 0x1F : cd;
  cba.srcColorBlendFactor = BlendFactor(cs);
  cba.dstColorBlendFactor = BlendFactor(cd);
  cba.colorBlendOp = BlendOp(cf);
  cba.srcAlphaBlendFactor = BlendFactor(as);
  cba.dstAlphaBlendFactor = BlendFactor(ad);
  cba.alphaBlendOp = BlendOp(af);
  return cba;
}

// GCN data format -> Vulkan vertex format.
VkFormat VertexFormat(uint32_t dfmt, uint32_t nfmt) {
  // GCN number formats: 0 UNORM, 1 SNORM, 2 USCALED, 3 SSCALED, 4 UINT,
  // 5 SINT, 7 FLOAT. The scaled forms deliver the integer value as a float,
  // which is what Vulkan's *_SSCALED/_USCALED do.
  switch (dfmt) {
    case 1:  // 8
      switch (nfmt) {
        case 1:
          return VK_FORMAT_R8_SNORM;
        case 2:
          return VK_FORMAT_R8_USCALED;
        case 3:
          return VK_FORMAT_R8_SSCALED;
        case 4:
          return VK_FORMAT_R8_UINT;
        case 5:
          return VK_FORMAT_R8_SINT;
        default:
          return VK_FORMAT_R8_UNORM;
      }
    case 2:  // 16
      switch (nfmt) {
        case 1:
          return VK_FORMAT_R16_SNORM;
        case 2:
          return VK_FORMAT_R16_USCALED;
        case 3:
          return VK_FORMAT_R16_SSCALED;
        case 4:
          return VK_FORMAT_R16_UINT;
        case 5:
          return VK_FORMAT_R16_SINT;
        case 7:
          return VK_FORMAT_R16_SFLOAT;
        default:
          return VK_FORMAT_R16_UNORM;
      }
    case 3:  // 8_8
      switch (nfmt) {
        case 1:
          return VK_FORMAT_R8G8_SNORM;
        case 2:
          return VK_FORMAT_R8G8_USCALED;
        case 3:
          return VK_FORMAT_R8G8_SSCALED;
        case 4:
          return VK_FORMAT_R8G8_UINT;
        case 5:
          return VK_FORMAT_R8G8_SINT;
        default:
          return VK_FORMAT_R8G8_UNORM;
      }
    case 4:  // 32
      return nfmt == 4   ? VK_FORMAT_R32_UINT
             : nfmt == 5 ? VK_FORMAT_R32_SINT
                         : VK_FORMAT_R32_SFLOAT;
    case 5:  // 16_16
      switch (nfmt) {
        case 0:
          return VK_FORMAT_R16G16_UNORM;
        case 1:
          return VK_FORMAT_R16G16_SNORM;
        case 2:
          return VK_FORMAT_R16G16_USCALED;
        case 3:
          return VK_FORMAT_R16G16_SSCALED;
        case 4:
          return VK_FORMAT_R16G16_UINT;
        case 5:
          return VK_FORMAT_R16G16_SINT;
        default:
          return VK_FORMAT_R16G16_SFLOAT;
      }
    case 8:
      return VK_FORMAT_A2R10G10B10_UNORM_PACK32;  // 10_10_10_2
    case 9:
      return VK_FORMAT_A2B10G10R10_UNORM_PACK32;  // 2_10_10_10
    case 10:                                      // 8_8_8_8
      switch (nfmt) {
        case 1:
          return VK_FORMAT_R8G8B8A8_SNORM;
        case 2:
          return VK_FORMAT_R8G8B8A8_USCALED;
        case 3:
          return VK_FORMAT_R8G8B8A8_SSCALED;
        case 4:
          return VK_FORMAT_R8G8B8A8_UINT;
        case 5:
          return VK_FORMAT_R8G8B8A8_SINT;
        default:
          return VK_FORMAT_R8G8B8A8_UNORM;
      }
    case 11:
      return nfmt == 4   ? VK_FORMAT_R32G32_UINT
             : nfmt == 5 ? VK_FORMAT_R32G32_SINT
                         : VK_FORMAT_R32G32_SFLOAT;
    case 12:  // 16_16_16_16
      switch (nfmt) {
        case 0:
          return VK_FORMAT_R16G16B16A16_UNORM;
        case 1:
          return VK_FORMAT_R16G16B16A16_SNORM;
        case 2:
          return VK_FORMAT_R16G16B16A16_USCALED;
        case 3:
          return VK_FORMAT_R16G16B16A16_SSCALED;
        case 4:
          return VK_FORMAT_R16G16B16A16_UINT;
        case 5:
          return VK_FORMAT_R16G16B16A16_SINT;
        default:
          return VK_FORMAT_R16G16B16A16_SFLOAT;
      }
    case 13:
      return VK_FORMAT_R32G32B32_SFLOAT;
    case 14:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
    default:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
  }
}

// Byte size of one vertex element in the given GCN data format -- must match
// the VkFormat VertexFormat() selects. Used to size a stride-0 (constant)
// binding's upload, where there is no source stride to derive the record extent
// from.
uint32_t VertexFormatBytes(uint32_t dfmt) {
  switch (dfmt) {
    case 1:
      return 1;  // R8
    case 3:
      return 2;  // R8G8
    case 2:
      return 2;  // R16
    case 4:
      return 4;  // R32
    case 5:
      return 4;  // R16G16
    case 6:
      return 4;  // R11G11B10
    case 8:
    case 9:
      return 4;  // 10_10_10_2 / 2_10_10_10
    case 10:
      return 4;  // R8G8B8A8
    case 11:
      return 8;  // R32G32
    case 12:
      return 8;  // R16G16B16A16
    case 13:
      return 12;  // R32G32B32
    case 14:
      return 16;  // R32G32B32A32
    default:
      return 16;
  }
}

// VGT_PRIMITIVE_TYPE -> Vulkan topology. Unknown/2D types fall back to triangle
// list (the previous hardcoded topology), so the 2D path is unchanged.
VkPrimitiveTopology PrimitiveTopology(uint32_t prim) {
  switch (prim) {
    case 1:
      return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case 2:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case 3:
      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case 5:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case 6:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case 4:  // triangle list
    default:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  }
}

void ReadbackPixelBgra(const uint8_t* src, VkFormat fmt, uint8_t* dst) {
  float rgba[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  switch (fmt) {
    case VK_FORMAT_B8G8R8A8_UNORM:
      std::memcpy(dst, src, 4);
      return;
    case VK_FORMAT_R8_UNORM:
      rgba[0] = src[0] / 255.0f;
      break;
    case VK_FORMAT_R8G8_UNORM:
      rgba[0] = src[0] / 255.0f;
      rgba[1] = src[1] / 255.0f;
      break;
    case VK_FORMAT_R16_UNORM: {
      uint16_t v;
      std::memcpy(&v, src, sizeof(v));
      rgba[0] = v / 65535.0f;
      break;
    }
    case VK_FORMAT_R16_SFLOAT: {
      uint16_t v;
      std::memcpy(&v, src, sizeof(v));
      rgba[0] = HalfToFloat(v);
      break;
    }
    case VK_FORMAT_R16G16_UNORM: {
      uint16_t v[2];
      std::memcpy(v, src, sizeof(v));
      rgba[0] = v[0] / 65535.0f;
      rgba[1] = v[1] / 65535.0f;
      break;
    }
    case VK_FORMAT_R16G16_SFLOAT: {
      uint16_t v[2];
      std::memcpy(v, src, sizeof(v));
      rgba[0] = HalfToFloat(v[0]);
      rgba[1] = HalfToFloat(v[1]);
      break;
    }
    case VK_FORMAT_R16G16B16A16_UNORM: {
      uint16_t v[4];
      std::memcpy(v, src, sizeof(v));
      for (int i = 0; i < 4; i++)
        rgba[i] = v[i] / 65535.0f;
      break;
    }
    case VK_FORMAT_R16G16B16A16_SFLOAT: {
      uint16_t v[4];
      std::memcpy(v, src, sizeof(v));
      for (int i = 0; i < 4; i++)
        rgba[i] = HalfToFloat(v[i]);
      break;
    }
    case VK_FORMAT_R32_SFLOAT:
      std::memcpy(&rgba[0], src, sizeof(float));
      break;
    case VK_FORMAT_R32G32_SFLOAT:
      std::memcpy(rgba, src, sizeof(float) * 2);
      break;
    case VK_FORMAT_R32G32B32_SFLOAT:
      std::memcpy(rgba, src, sizeof(float) * 3);
      break;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      std::memcpy(rgba, src, sizeof(rgba));
      break;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: {
      uint32_t packed;
      std::memcpy(&packed, src, sizeof(packed));
      rgba[0] = PackedUfloat(packed & 0x7FF, 6);
      rgba[1] = PackedUfloat((packed >> 11) & 0x7FF, 6);
      rgba[2] = PackedUfloat(packed >> 22, 5);
      break;
    }
    default:
      std::memcpy(dst, src, 4);
      return;
  }
  dst[0] = Unorm8(rgba[2]);
  dst[1] = Unorm8(rgba[1]);
  dst[2] = Unorm8(rgba[0]);
  dst[3] = Unorm8(rgba[3]);
}

}  // namespace gpu::vk
