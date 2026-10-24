/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/image/guest_format.h"
#include "base/arch.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "base/algorithm.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "gpu/gcn/gcn_resource.h"
#include "options/options.h"

namespace {
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
  return static_cast<u8>(std::lround(base::Clamp(value, 0.0f, 1.0f) * 255.0f));
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
  if ((dfmt == 6 || dfmt == 7) && nfmt == 7)
    return rhi::Format::kB10G11R11Float;
  if (dfmt == 12 && nfmt == 7)
    return rhi::Format::kRGBA16Float;
  if (dfmt == 10 && nfmt == 0)
    return rhi::Format::kRGBA8Unorm;
  if (dfmt == 10 && nfmt == 9)
    return rhi::Format::kRGBA8Srgb;
  if (dfmt == 35)
    return nfmt == 9 ? rhi::Format::kBC1Srgb : rhi::Format::kBC1Unorm;
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

bool GuestFormatBlockCompressed(u32 dfmt) {
  return dfmt >= 35 && dfmt <= 41;
}

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
      return 1;
    case 2:
    case 3:
      return 2;
    case 11:
    case 12:
      return 8;
    case 13:
      return 12;
    case 14:
      return 16;
    case 35:
    case 38:
      return 8;
    default:
      return GuestFormatBlockCompressed(dfmt) ? 16 : 4;
  }
}

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
      case 7:
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
        return ((info >> 11) & 3) == 1 ? rhi::Format::kBGRA8Unorm
                                       : rhi::Format::kRGBA8Unorm;
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

rhi::ClearColor ColorTargetClearValue(u32 info, u32 word0, u32 word1) {
  const u32 dfmt = (info >> 2) & 0x1F;
  const u32 nfmt = (info >> 8) & 0x7;
  const auto unmapped = [&](const char* what) {
    static int n = 0;
    if (n++ < 8)
      BASE_LOGI("gpuvk",
                "clear word: unmapped {} (info={:#x} dfmt={} nfmt={} "
                "words {:08x} {:08x}), clearing to opaque black",
                what, info, dfmt, nfmt, word0, word1);
    return rhi::ClearColor{{0.0f, 0.0f, 0.0f, 1.0f}};
  };
  u32 width[4] = {0, 0, 0, 0};
  switch (dfmt) {
    case 1:
      width[0] = 8;
      break;
    case 2:
      width[0] = 16;
      break;
    case 3:
      width[0] = width[1] = 8;
      break;
    case 4:
      width[0] = 32;
      break;
    case 5:
      width[0] = width[1] = 16;
      break;
    case 6:
      width[0] = width[1] = 11;
      width[2] = 10;
      break;
    case 7:
      width[0] = 10;
      width[1] = width[2] = 11;
      break;
    case 9:
      width[0] = width[1] = width[2] = 10;
      width[3] = 2;
      break;
    case 10:
      width[0] = width[1] = width[2] = width[3] = 8;
      break;
    case 11:
      width[0] = width[1] = 32;
      break;
    case 12:
      width[0] = width[1] = width[2] = width[3] = 16;
      break;
    case 13:
    case 14:
      width[0] = width[1] = 32;
      break;
    default:
      return unmapped("colour format");
  }
  const u64 packed = static_cast<u64>(word0) | static_cast<u64>(word1) << 32;
  rhi::ClearColor out{};
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
      case 0:
      case 6:
        out.f[i] = static_cast<float>(raw) / static_cast<float>(mask);
        break;
      case 1:
        out.f[i] = base::Max(static_cast<float>(SignExtend(raw, bits)) /
                                 static_cast<float>((1u << (bits - 1)) - 1),
                             -1.0f);
        break;
      case 4:
        out.u[i] = raw;
        break;
      case 5:
        out.i[i] = SignExtend(raw, bits);
        break;
      case 7:
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

u8 ColorWriteMask(u32 target_mask,
                  u32 shader_mask,
                  u8 export_mask,
                  u32 target) {
  if (target >= 8 || !(export_mask & (1u << target)))
    return 0;
  u32 mask = (target_mask >> (4 * target)) & 0xF;
  if (shader_mask)
    mask &= (shader_mask >> (4 * target)) & 0xF;
  return static_cast<u8>(mask);
}

rhi::BlendAttachment BlendAttachment(u32 bc, bool en) {
  rhi::BlendAttachment cba;
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

rhi::Format VertexFormat(u32 dfmt, u32 nfmt) {
  switch (dfmt) {
    case 1:
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
    case 2:
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
    case 3:
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
    case 4:
      return rhi::Format::kR32Float;
    case 5:
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
      return rhi::Format::kA2R10G10B10Unorm;
    case 9:
      return gcn::PackedAttrFormat(dfmt, nfmt) ? rhi::Format::kR32Uint
                                               : rhi::Format::kA2B10G10R10Unorm;
    case 10:
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
    case 12:
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

u32 VertexFormatBytes(u32 dfmt) {
  switch (dfmt) {
    case 1:
      return 1;
    case 3:
      return 2;
    case 2:
      return 2;
    case 4:
      return 4;
    case 5:
      return 4;
    case 6:
      return 4;
    case 8:
    case 9:
      return 4;
    case 10:
      return 4;
    case 11:
      return 8;
    case 12:
      return 8;
    case 13:
      return 12;
    case 14:
      return 16;
    default:
      return 16;
  }
}

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
    case 4:
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
