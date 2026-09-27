/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/guest_format.h"
#include "base/arch.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <base/logging.h>
#include <options/options.h>
#include <base/algorithm.h>
#include <base/math/value_bounds.h>

namespace {
// A colour target whose NUMBER_TYPE is UINT/SINT holds packed bits, not a
// colour. Mapping one to UNORM clamps every export into [0,1] and the target
// reads back black; SotC lost two whole G-buffer planes that way. On by
// default; DELTA_GPU_INT_RT=0 restores the old UNORM mapping.
DELTA_OPTION(bool, kIntegerRt, "DELTA_GPU_INT_RT", true);
DELTA_OPTION(bool, kNoBlend, "DELTA_GPU_NOBLEND", false);
DELTA_OPTION(bool, kNoSwizzle, "DELTA_GPU_NOSWIZZLE", false);
}  // namespace

namespace gpu::render {
namespace {

float HalfToFloat(u16 value) {
  u32 sign = static_cast<u32>(value & 0x8000) << 16;
  u32 exponent = (value >> 10) & 0x1F;
  u32 mantissa = value & 0x3FF;
  u32 bits;
  if (!exponent) {
    if (!mantissa) {
      bits = sign;
    } else {
      int unbiased = -14;
      while (!(mantissa & 0x400)) {
        mantissa <<= 1;
        unbiased--;
      }
      bits = sign | static_cast<u32>(unbiased + 127) << 23 |
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

u8 Unorm8(float value) {
  if (!std::isfinite(value))
    return 0;
  return static_cast<u8>(
      std::lround(base::Clamp(value, 0.0f, 1.0f) * 255.0f));
}

float PackedUfloat(u32 value, u32 mantissa_bits) {
  u32 mantissa_mask = (1u << mantissa_bits) - 1;
  u32 mantissa = value & mantissa_mask;
  u32 exponent = value >> mantissa_bits;
  if (!exponent)
    return std::ldexp(static_cast<float>(mantissa),
                      -14 - static_cast<int>(mantissa_bits));
  if (exponent == 0x1F)
    return mantissa ? NAN : INFINITY;
  return std::ldexp(1.0f + static_cast<float>(mantissa) / (1u << mantissa_bits),
                    static_cast<int>(exponent) - 15);
}

i32 SignExtend(u32 value, u32 bits) {
  return static_cast<i32>(value << (32 - bits)) >> (32 - bits);
}

}  // namespace

rhi::Format GuestTextureFormat(u32 dfmt, u32 nfmt) {
  // Narrow and float channel counts: RDNA2 titles sample single-channel masks
  // and packed HDR buffers that the PS4 titles never used.
  if (dfmt == 1 && nfmt == 0)
    return rhi::Format::kR8Unorm;
  if (dfmt == 1 && nfmt == 1)
    return rhi::Format::kR8Snorm;
  if (dfmt == 1 && nfmt == 4)
    return rhi::Format::kR8Uint;
  if (dfmt == 2 && nfmt == 0)
    return rhi::Format::kR16Unorm;
  if (dfmt == 2 && nfmt == 4)
    return rhi::Format::kR16Uint;
  if (dfmt == 2 && nfmt == 7)
    return rhi::Format::kR16Float;
  if (dfmt == 3 && nfmt == 0)
    return rhi::Format::kRG8Unorm;
  if (dfmt == 3 && nfmt == 4)
    return rhi::Format::kRG8Uint;
  if (dfmt == 4 && nfmt == 4)
    return rhi::Format::kR32Uint;
  if (dfmt == 4 && nfmt == 7)
    return rhi::Format::kR32Float;
  if (dfmt == 5 && nfmt == 0)
    return rhi::Format::kRG16Unorm;
  if (dfmt == 5 && nfmt == 4)
    return rhi::Format::kRG16Uint;
  if (dfmt == 8 && nfmt == 0)
    return rhi::Format::kA2R10G10B10Unorm;
  if (dfmt == 9 && nfmt == 0)
    return rhi::Format::kA2B10G10R10Unorm;
  if (dfmt == 10 && nfmt == 1)
    return rhi::Format::kRGBA8Snorm;
  if (dfmt == 10 && nfmt == 4)
    return rhi::Format::kRGBA8Uint;
  if (dfmt == 11 && nfmt == 4)
    return rhi::Format::kRG32Uint;
  if (dfmt == 11 && nfmt == 7)
    return rhi::Format::kRG32Float;
  if (dfmt == 12 && nfmt == 0)
    return rhi::Format::kRGBA16Unorm;
  if (dfmt == 14 && nfmt == 7)
    return rhi::Format::kRGBA32Float;
  if (dfmt == 5 && nfmt == 7)
    return rhi::Format::kRG16Float;
  if (dfmt == 6 && nfmt == 7)
    return rhi::Format::kB10G11R11Float;
  if (dfmt == 12 && nfmt == 7)
    return rhi::Format::kRGBA16Float;
  if (dfmt == 10 && nfmt == 0)
    return rhi::Format::kRGBA8Unorm;
  if (dfmt == 10 && nfmt == 9)
    return rhi::Format::kRGBA8Srgb;
  // Block-compressed (IMG_DATA_FORMAT_BC1..BC7 = 35..41); sampled natively.
  if (dfmt == 35)
    return nfmt == 9 ? rhi::Format::kBC1Srgb
                     : rhi::Format::kBC1Unorm;
  if (dfmt == 36)
    return nfmt == 9 ? rhi::Format::kBC2Srgb : rhi::Format::kBC2Unorm;
  if (dfmt == 37)
    return nfmt == 9 ? rhi::Format::kBC3Srgb : rhi::Format::kBC3Unorm;
  if (dfmt == 38)
    return nfmt == 1 ? rhi::Format::kBC4Snorm : rhi::Format::kBC4Unorm;
  if (dfmt == 39)
    return nfmt == 1 ? rhi::Format::kBC5Snorm : rhi::Format::kBC5Unorm;
  if (dfmt == 40)
    return nfmt == 1 ? rhi::Format::kBC6HSfloat : rhi::Format::kBC6HUfloat;
  if (dfmt == 41)
    return nfmt == 9 ? rhi::Format::kBC7Srgb : rhi::Format::kBC7Unorm;
  return rhi::Format::kUndefined;
}

// Block-compressed guest formats: 4x4-texel blocks of 8 or 16 bytes. The
// detiler and upload then work in block ("element") space.
bool GuestFormatBlockCompressed(u32 dfmt) {
  return dfmt >= 35 && dfmt <= 41;
}

// The host side of the same question: a block-compressed rhi::Format shares no
// view-compatibility class with any uncompressed one.
bool FormatBlockCompressed(rhi::Format format) {
  switch (format) {
    case rhi::Format::kBC1Unorm:
    case rhi::Format::kBC1Srgb:
    case rhi::Format::kBC2Unorm:
    case rhi::Format::kBC2Srgb:
    case rhi::Format::kBC3Unorm:
    case rhi::Format::kBC3Srgb:
    case rhi::Format::kBC4Unorm:
    case rhi::Format::kBC4Snorm:
    case rhi::Format::kBC5Unorm:
    case rhi::Format::kBC5Snorm:
    case rhi::Format::kBC6HUfloat:
    case rhi::Format::kBC6HSfloat:
    case rhi::Format::kBC7Unorm:
    case rhi::Format::kBC7Srgb:
      return true;
    default:
      return false;
  }
}

u32 GuestFormatElemBytes(u32 dfmt) {
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
bool IsIntegerColorFormat(rhi::Format format) {
  switch (format) {
    case rhi::Format::kR8Uint:
    case rhi::Format::kR8Sint:
    case rhi::Format::kR16Uint:
    case rhi::Format::kR16Sint:
    case rhi::Format::kRG8Uint:
    case rhi::Format::kRG8Sint:
    case rhi::Format::kR32Uint:
    case rhi::Format::kR32Sint:
    case rhi::Format::kRG16Uint:
    case rhi::Format::kRG16Sint:
    case rhi::Format::kRGBA8Uint:
    case rhi::Format::kRGBA8Sint:
    case rhi::Format::kRG32Uint:
    case rhi::Format::kRG32Sint:
    case rhi::Format::kRGBA16Uint:
    case rhi::Format::kRGBA16Sint:
    case rhi::Format::kRGBA32Uint:
    case rhi::Format::kRGBA32Sint:
      return true;
    default:
      return false;
  }
}

rhi::Format ColorTargetFormat(u32 info) {
  u32 dfmt = (info >> 2) & 0x1F;
  u32 nfmt = (info >> 8) & 0x7;
  if (nfmt == 7) {
    switch (dfmt) {
      case 2:
        return rhi::Format::kR16Float;
      case 4:
        return rhi::Format::kR32Float;
      case 5:
        return rhi::Format::kRG16Float;
      case 6:
        return rhi::Format::kB10G11R11Float;
      case 11:
        return rhi::Format::kRG32Float;
      case 12:
        return rhi::Format::kRGBA16Float;
      case 13:
        return rhi::Format::kRGB32Float;
      case 14:
        return rhi::Format::kRGBA32Float;
      default:
        break;
    }
  }
  if (kIntegerRt && (nfmt == 4 || nfmt == 5)) {
    const bool sint = nfmt == 5;
    switch (dfmt) {
      case 1:
        return sint ? rhi::Format::kR8Sint : rhi::Format::kR8Uint;
      case 2:
        return sint ? rhi::Format::kR16Sint : rhi::Format::kR16Uint;
      case 3:
        return sint ? rhi::Format::kRG8Sint : rhi::Format::kRG8Uint;
      case 4:
        return sint ? rhi::Format::kR32Sint : rhi::Format::kR32Uint;
      case 5:
        return sint ? rhi::Format::kRG16Sint : rhi::Format::kRG16Uint;
      case 10:
        return sint ? rhi::Format::kRGBA8Sint : rhi::Format::kRGBA8Uint;
      case 11:
        return sint ? rhi::Format::kRG32Sint : rhi::Format::kRG32Uint;
      case 12:
        return sint ? rhi::Format::kRGBA16Sint : rhi::Format::kRGBA16Uint;
      case 14:
        return sint ? rhi::Format::kRGBA32Sint : rhi::Format::kRGBA32Uint;
      default:
        break;
    }
  }
  // An SRGB target encodes what the shader exports; left UNORM it stores the
  // linear value, and a pass reading it through its SRGB T# decodes it again
  // (GTA:SA's base colour plane, which darkened every lit material).
  if (nfmt == 6 && dfmt == 10)
    return ((info >> 11) & 3) == 1 ? rhi::Format::kBGRA8Srgb
                                   : rhi::Format::kRGBA8Srgb;
  if (nfmt == 0) {
    switch (dfmt) {
      case 1:
        return rhi::Format::kR8Unorm;
      case 2:
        return rhi::Format::kR16Unorm;
      case 3:
        return rhi::Format::kRG8Unorm;
      case 5:
        return rhi::Format::kRG16Unorm;
      case 10:
        // COMP_SWAP (bits 12:11) names the order the hardware writes the four
        // components in: STD puts red in the first byte, ALT puts blue there.
        // Answering BGRA for both is invisible on screen (the present path
        // swaps back), but a pass that SAMPLES the target aliases the image in
        // the T#'s own format, and there an STD target comes back with red and
        // blue exchanged (Dead Cells' scene composite renders purple).
        return ((info >> 11) & 3) == 1 ? rhi::Format::kBGRA8Unorm
                                       : rhi::Format::kRGBA8Unorm;
      // The same host formats the sampled path picks for these T#s. Left to
      // the default, a 2_10_10_10 target was BGRA8, and a pass sampling it
      // read 8-bit texels as 10-bit fields (GTA:SA's grading LUT, which tints
      // every frame through the tonemapper).
      case 8:
        return rhi::Format::kA2R10G10B10Unorm;
      case 9:
        return rhi::Format::kA2B10G10R10Unorm;
      case 12:
        return rhi::Format::kRGBA16Unorm;
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
rhi::ClearColor ColorTargetClearValue(u32 info, u32 word0, u32 word1) {
  const u32 dfmt = (info >> 2) & 0x1F;
  const u32 nfmt = (info >> 8) & 0x7;
  const auto unmapped = [&](const char* what) {
    static int n = 0;
    if (n++ < 8)
      BASE_LOGI("gpuvk", "clear word: unmapped {} (info={:#x} dfmt={} nfmt={} "
                         "words {:08x} {:08x}), clearing to opaque black",
                what, info, dfmt, nfmt, word0, word1);
    return rhi::ClearColor{{0.0f, 0.0f, 0.0f, 1.0f}};
  };
  u32 width[4] = {0, 0, 0, 0};
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
  const u64 packed =
      static_cast<u64>(word0) | static_cast<u64>(word1) << 32;
  rhi::ClearColor out{};
  // Seed opaque: a format with fewer than four components never writes alpha,
  // and a transparent target is a hole in a deferred composite.
  if (nfmt == 4 || nfmt == 5)
    out.u[3] = 1;
  else
    out.f[3] = 1.0f;
  u32 shift = 0;
  for (u32 i = 0; i < 4 && width[i]; i++) {
    const u32 bits = width[i];
    const u64 mask = bits == 32 ? 0xFFFFFFFFull : (1ull << bits) - 1;
    const u32 raw = static_cast<u32>((packed >> shift) & mask);
    shift += bits;
    switch (nfmt) {
      case 0:  // UNORM
      case 6:  // SRGB: ColorTargetFormat gives it a UNORM host image, so the
               // encoded value has to pass through unconverted
        out.f[i] = static_cast<float>(raw) / static_cast<float>(mask);
        break;
      case 1:  // SNORM
        out.f[i] =
            base::Max(static_cast<float>(SignExtend(raw, bits)) /
                         static_cast<float>((1u << (bits - 1)) - 1),
                     -1.0f);
        break;
      case 4:  // UINT
        out.u[i] = raw;
        break;
      case 5:  // SINT
        out.i[i] = SignExtend(raw, bits);
        break;
      case 7:  // FLOAT
        if (bits == 32)
          std::memcpy(&out.f[i], &raw, sizeof(raw));
        else if (bits == 16)
          out.f[i] = HalfToFloat(static_cast<u16>(raw));
        else if (bits == 11 || bits == 10)
          out.f[i] = PackedUfloat(raw, bits - 5);
        else
          return unmapped("float width");
        break;
      default:
        return unmapped("number type");
    }
  }
  return out;
}


void TextureSwizzle(u32 swizzle, rhi::Swizzle out[4]) {
  for (u32 i = 0; i < 4; i++) {
    out[i] = rhi::Swizzle::kIdentity;
    if (!swizzle || kNoSwizzle)
      continue;
    switch ((swizzle >> (3 * i)) & 7) {
      case 0:
        out[i] = rhi::Swizzle::kZero;
        break;
      case 1:
        out[i] = rhi::Swizzle::kOne;
        break;
      case 4:
        out[i] = rhi::Swizzle::kR;
        break;
      case 5:
        out[i] = rhi::Swizzle::kG;
        break;
      case 6:
        out[i] = rhi::Swizzle::kB;
        break;
      case 7:
        out[i] = rhi::Swizzle::kA;
        break;
      default:
        break;
    }
  }
}

u32 FormatBytes(rhi::Format fmt) {
  switch (fmt) {
    case rhi::Format::kR8Unorm:
    case rhi::Format::kR8Uint:
    case rhi::Format::kR8Sint:
      return 1;
    case rhi::Format::kR16Unorm:
    case rhi::Format::kR16Float:
    case rhi::Format::kRG8Unorm:
    case rhi::Format::kR16Uint:
    case rhi::Format::kR16Sint:
    case rhi::Format::kRG8Uint:
    case rhi::Format::kRG8Sint:
      return 2;
    case rhi::Format::kRGBA16Float:
    case rhi::Format::kRGBA16Unorm:
    case rhi::Format::kRG32Float:
    case rhi::Format::kRGBA16Uint:
    case rhi::Format::kRGBA16Sint:
    case rhi::Format::kRG32Uint:
    case rhi::Format::kRG32Sint:
      return 8;
    case rhi::Format::kRGBA32Uint:
    case rhi::Format::kRGBA32Sint:
      return 16;
    case rhi::Format::kRGB32Float:
      return 12;
    case rhi::Format::kRGBA32Float:
      return 16;
    default:
      return 4;
  }
}

// GNM blend multiplier (CB_BLENDn_CONTROL factor field) -> Vulkan blend factor.
rhi::BlendFactor BlendFactor(u32 f) {
  switch (f) {
    case 0:
      return rhi::BlendFactor::kZero;
    case 1:
      return rhi::BlendFactor::kOne;
    case 2:
      return rhi::BlendFactor::kSrcColor;
    case 3:
      return rhi::BlendFactor::kOneMinusSrcColor;
    case 4:
      return rhi::BlendFactor::kSrcAlpha;
    case 5:
      return rhi::BlendFactor::kOneMinusSrcAlpha;
    // 6 upwards is neither the D3D order nor SRC_ALPHA_SATURATE-first: the
    // hardware enum (V_028780_BLEND_*) runs DST_ALPHA, DST_COLOR,
    // SRC_ALPHA_SATURATE, the two BOTH_* forms, the constants and only then
    // the dual-source factors at 0x0f..0x12. That last range is what pins it:
    // KytyPS5's IsDualSourceBlendFactor tests exactly 0x0f..0x12. Reading
    // 0x0d (CONSTANT_COLOR) as SRC1_COLOR gave a shader with no Index-1
    // output an undefined second source, which comes out as a black colour
    // term (Astro Bot's whole composite chain).
    case 6:
      return rhi::BlendFactor::kDstAlpha;
    case 7:
      return rhi::BlendFactor::kOneMinusDstAlpha;
    case 8:
      return rhi::BlendFactor::kDstColor;
    case 9:
      return rhi::BlendFactor::kOneMinusDstColor;
    case 10:
      return rhi::BlendFactor::kSrcAlphaSaturate;
    // BOTH_SRC_ALPHA / BOTH_INV_SRC_ALPHA set the colour and alpha factors
    // together on hardware; Vulkan has no such factor, and the pair they
    // stand for is (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) per channel.
    case 11:
      return rhi::BlendFactor::kSrcAlpha;
    case 12:
      return rhi::BlendFactor::kOneMinusSrcAlpha;
    case 13:
      return rhi::BlendFactor::kConstantColor;
    case 14:
      return rhi::BlendFactor::kOneMinusConstantColor;
    case 15:
      return rhi::BlendFactor::kSrc1Color;
    case 16:
      return rhi::BlendFactor::kOneMinusSrc1Color;
    case 17:
      return rhi::BlendFactor::kSrc1Alpha;
    case 18:
      return rhi::BlendFactor::kOneMinusSrc1Alpha;
    case 19:
      return rhi::BlendFactor::kConstantAlpha;
    case 20:
      return rhi::BlendFactor::kOneMinusConstantAlpha;
    default:
      return rhi::BlendFactor::kOne;
  }
}

// GNM blend function (combine fcn) -> blend op.
rhi::BlendOp BlendOp(u32 f) {
  switch (f) {
    case 0:
      return rhi::BlendOp::kAdd;
    case 1:
      return rhi::BlendOp::kSubtract;
    case 2:
      return rhi::BlendOp::kMin;
    case 3:
      return rhi::BlendOp::kMax;
    case 4:
      return rhi::BlendOp::kReverseSubtract;
    default:
      return rhi::BlendOp::kAdd;
  }
}

u8 ColorWriteMask(u32 target_mask, u32 shader_mask,
                                     u8 export_mask, u32 target) {
  if (target >= 8 || !(export_mask & (1u << target)))
    return 0;
  u32 mask = (target_mask >> (4 * target)) & 0xF;
  if (shader_mask)
    mask &= (shader_mask >> (4 * target)) & 0xF;
  return static_cast<u8>(mask);
}

// Decode CB_BLEND0_CONTROL into a colour-blend attachment. `en` is the
// per-target blend enable (bit 30). Falls back to a sensible src-alpha blend
// when the guest enables blend but the control word is zero (default state, not
// yet set).
rhi::BlendAttachment BlendAttachment(u32 bc, bool en) {
  rhi::BlendAttachment cba;
  // DELTA_GPU_NOBLEND: force opaque (diagnostic) to test whether a draw
  // vanishes because its src-alpha blend multiplies by a zero texel alpha
  // (Doom64 3D walls).
  if (kNoBlend)
    en = false;
  if (!en)
    return cba;
  cba.enable = true;
  u32 cs = bc & 0x1F, cf = (bc >> 5) & 0x7, cd = (bc >> 8) & 0x1F;
  bool sep = (bc >> 29) & 1;
  u32 as = sep ? (bc >> 16) & 0x1F : cs;
  u32 af = sep ? (bc >> 21) & 0x7 : cf;
  u32 ad = sep ? (bc >> 24) & 0x1F : cd;
  cba.src_color = BlendFactor(cs);
  cba.dst_color = BlendFactor(cd);
  cba.color_op = BlendOp(cf);
  cba.src_alpha = BlendFactor(as);
  cba.dst_alpha = BlendFactor(ad);
  cba.alpha_op = BlendOp(af);
  return cba;
}

// GCN data format -> Vulkan vertex format.
rhi::Format VertexFormat(u32 dfmt, u32 nfmt) {
  // The recompiled vertex shader declares every attribute as a float vector,
  // so an integer attribute format is a type mismatch and the attribute reads
  // undefined (VUID-VkGraphicsPipelineCreateInfo-Input-08733). The SCALED
  // forms deliver the same integer VALUE as a float, which is what a shader
  // that fetches an integer attribute and converts it ends up with. 32-bit
  // integers have no SCALED form but need none: the module BITCASTS every
  // attribute component into its VGPR (what the hardware does for an integer
  // number format), and a 32-bit SFLOAT attribute delivers those same 32 bits
  // to a float input, so the SFLOAT twin is both bit-exact and correctly
  // typed.
  // GCN number formats: 0 UNORM, 1 SNORM, 2 USCALED, 3 SSCALED, 4 UINT,
  // 5 SINT, 7 FLOAT. The scaled forms deliver the integer value as a float,
  // which is what Vulkan's *_SSCALED/_USCALED do.
  switch (dfmt) {
    case 1:  // 8
      switch (nfmt) {
        case 1:
          return rhi::Format::kR8Snorm;
        case 2:
          return rhi::Format::kR8Uscaled;
        case 3:
          return rhi::Format::kR8Sscaled;
        case 4:
          return rhi::Format::kR8Uscaled;
        case 5:
          return rhi::Format::kR8Sscaled;
        default:
          return rhi::Format::kR8Unorm;
      }
    case 2:  // 16
      switch (nfmt) {
        case 1:
          return rhi::Format::kR16Snorm;
        case 2:
          return rhi::Format::kR16Uscaled;
        case 3:
          return rhi::Format::kR16Sscaled;
        case 4:
          return rhi::Format::kR16Uscaled;
        case 5:
          return rhi::Format::kR16Sscaled;
        case 7:
          return rhi::Format::kR16Float;
        default:
          return rhi::Format::kR16Unorm;
      }
    case 3:  // 8_8
      switch (nfmt) {
        case 1:
          return rhi::Format::kRG8Snorm;
        case 2:
          return rhi::Format::kRG8Uscaled;
        case 3:
          return rhi::Format::kRG8Sscaled;
        case 4:
          return rhi::Format::kRG8Uscaled;
        case 5:
          return rhi::Format::kRG8Sscaled;
        default:
          return rhi::Format::kRG8Unorm;
      }
    case 4:  // 32
      return rhi::Format::kR32Float;
    case 5:  // 16_16
      switch (nfmt) {
        case 0:
          return rhi::Format::kRG16Unorm;
        case 1:
          return rhi::Format::kRG16Snorm;
        case 2:
          return rhi::Format::kRG16Uscaled;
        case 3:
          return rhi::Format::kRG16Sscaled;
        case 4:
          return rhi::Format::kRG16Uscaled;
        case 5:
          return rhi::Format::kRG16Sscaled;
        default:
          return rhi::Format::kRG16Float;
      }
    case 8:
      return rhi::Format::kA2R10G10B10Unorm;  // 10_10_10_2
    case 9:
      return rhi::Format::kA2B10G10R10Unorm;  // 2_10_10_10
    case 10:                                      // 8_8_8_8
      switch (nfmt) {
        case 1:
          return rhi::Format::kRGBA8Snorm;
        case 2:
          return rhi::Format::kRGBA8Uscaled;
        case 3:
          return rhi::Format::kRGBA8Sscaled;
        case 4:
          return rhi::Format::kRGBA8Uscaled;
        case 5:
          return rhi::Format::kRGBA8Sscaled;
        default:
          return rhi::Format::kRGBA8Unorm;
      }
    case 11:
      return rhi::Format::kRG32Float;
    case 12:  // 16_16_16_16
      switch (nfmt) {
        case 0:
          return rhi::Format::kRGBA16Unorm;
        case 1:
          return rhi::Format::kRGBA16Snorm;
        case 2:
          return rhi::Format::kRGBA16Uscaled;
        case 3:
          return rhi::Format::kRGBA16Sscaled;
        case 4:
          return rhi::Format::kRGBA16Uscaled;
        case 5:
          return rhi::Format::kRGBA16Sscaled;
        default:
          return rhi::Format::kRGBA16Float;
      }
    case 13:
      return rhi::Format::kRGB32Float;
    case 14:
      return rhi::Format::kRGBA32Float;
    default:
      return rhi::Format::kRGBA32Float;
  }
}

// Byte size of one vertex element in the given GCN data format; must match
// the rhi::Format VertexFormat() selects. Used to size a stride-0 (constant)
// binding's upload, where there is no source stride to derive the record extent
// from.
u32 VertexFormatBytes(u32 dfmt) {
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
rhi::Topology PrimitiveTopology(u32 prim) {
  switch (prim) {
    case 1:
      return rhi::Topology::kPointList;
    case 2:
      return rhi::Topology::kLineList;
    case 3:
      return rhi::Topology::kLineStrip;
    case 5:
      return rhi::Topology::kTriangleFan;
    case 6:
      return rhi::Topology::kTriangleStrip;
    case 4:  // triangle list
    default:
      return rhi::Topology::kTriangleList;
  }
}

void ReadbackPixelBgra(const u8* src, rhi::Format fmt, u8* dst) {
  float rgba[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  switch (fmt) {
    case rhi::Format::kBGRA8Unorm:
      std::memcpy(dst, src, 4);
      return;
    case rhi::Format::kRGBA8Unorm:
    case rhi::Format::kRGBA8Srgb:
      dst[0] = src[2];
      dst[1] = src[1];
      dst[2] = src[0];
      dst[3] = src[3];
      return;
    case rhi::Format::kR8Unorm:
      rgba[0] = src[0] / 255.0f;
      break;
    case rhi::Format::kRG8Unorm:
      rgba[0] = src[0] / 255.0f;
      rgba[1] = src[1] / 255.0f;
      break;
    case rhi::Format::kR16Unorm: {
      u16 v;
      std::memcpy(&v, src, sizeof(v));
      rgba[0] = v / 65535.0f;
      break;
    }
    case rhi::Format::kR16Float: {
      u16 v;
      std::memcpy(&v, src, sizeof(v));
      rgba[0] = HalfToFloat(v);
      break;
    }
    case rhi::Format::kRG16Unorm: {
      u16 v[2];
      std::memcpy(v, src, sizeof(v));
      rgba[0] = v[0] / 65535.0f;
      rgba[1] = v[1] / 65535.0f;
      break;
    }
    case rhi::Format::kRG16Float: {
      u16 v[2];
      std::memcpy(v, src, sizeof(v));
      rgba[0] = HalfToFloat(v[0]);
      rgba[1] = HalfToFloat(v[1]);
      break;
    }
    case rhi::Format::kRGBA16Unorm: {
      u16 v[4];
      std::memcpy(v, src, sizeof(v));
      for (int i = 0; i < 4; i++)
        rgba[i] = v[i] / 65535.0f;
      break;
    }
    case rhi::Format::kRGBA16Float: {
      u16 v[4];
      std::memcpy(v, src, sizeof(v));
      for (int i = 0; i < 4; i++)
        rgba[i] = HalfToFloat(v[i]);
      break;
    }
    case rhi::Format::kR32Float:
      std::memcpy(&rgba[0], src, sizeof(float));
      break;
    case rhi::Format::kRG32Float:
      std::memcpy(rgba, src, sizeof(float) * 2);
      break;
    case rhi::Format::kRGB32Float:
      std::memcpy(rgba, src, sizeof(float) * 3);
      break;
    case rhi::Format::kRGBA32Float:
      std::memcpy(rgba, src, sizeof(rgba));
      break;
    case rhi::Format::kA2B10G10R10Unorm: {
      u32 packed;
      std::memcpy(&packed, src, sizeof(packed));
      rgba[0] = (packed & 0x3FF) / 1023.0f;
      rgba[1] = ((packed >> 10) & 0x3FF) / 1023.0f;
      rgba[2] = ((packed >> 20) & 0x3FF) / 1023.0f;
      rgba[3] = (packed >> 30) / 3.0f;
      break;
    }
    case rhi::Format::kA2R10G10B10Unorm: {
      u32 packed;
      std::memcpy(&packed, src, sizeof(packed));
      rgba[2] = (packed & 0x3FF) / 1023.0f;
      rgba[1] = ((packed >> 10) & 0x3FF) / 1023.0f;
      rgba[0] = ((packed >> 20) & 0x3FF) / 1023.0f;
      rgba[3] = (packed >> 30) / 3.0f;
      break;
    }
    case rhi::Format::kB10G11R11Float: {
      u32 packed;
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

}  // namespace gpu::render
