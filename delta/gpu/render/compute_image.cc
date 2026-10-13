/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/compute_internal.h"

#include <cstdio>
#include <cstring>
#include <limits>

#define BCDECDEF static inline
#define BCDEC_IMPLEMENTATION
#include <bcdec.h>

#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/bit_cast.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/render/capture.h"
#include "gpu/render/tiling.h"
#include "guest/session.h"

namespace gpu::render::cs {

bool BuildCsImageLayouts(const ComputeInfo::Res& res,
                         gcn::TextureLayout32& tiled,
                         gcn::TextureLayout32& linear) {
  const u32 stage_tiling = res.tiling_idx == 31 ? 31 : 8;
  const bool compressed = res.dfmt == 35 || res.dfmt == 40;
  return res.image_staging &&
         gcn::BuildTextureLayout32(
             tiled, compressed ? (res.width + 3) / 4 : res.width,
             compressed ? (res.height + 3) / 4 : res.height,
             compressed ? (res.pitch + 3) / 4 : res.pitch, res.layers,
             res.mip_levels, res.tiling_idx, res.pow2_pad, res.elem_bytes) &&
         gcn::BuildTextureLayout32(linear, res.width, res.height, res.pitch,
                                   res.layers, res.mip_levels, stage_tiling,
                                   res.pow2_pad, res.stage_elem_bytes) &&
         tiled.size == res.guest_size && linear.size == res.size;
}

float UnpackUnsignedFloat(u32 value, u32 mantissa_bits) {
  const u32 mantissa_mask = (1u << mantissa_bits) - 1;
  const u32 mantissa = value & mantissa_mask;
  const u32 exponent = (value >> mantissa_bits) & 0x1F;
  if (!exponent)
    return static_cast<float>(mantissa) *
           base::BitCast<float>((127u - 14u - mantissa_bits) << 23);
  if (exponent == 0x1F)
    return mantissa ? std::numeric_limits<float>::quiet_NaN()
                    : std::numeric_limits<float>::infinity();
  return base::BitCast<float>((exponent + 112) << 23 |
                              mantissa << (23 - mantissa_bits));
}

u32 PackUnsignedFloat(float value, u32 mantissa_bits) {
  const u32 bits = base::BitCast<u32>(value);
  const u32 inf = 0x1Fu << mantissa_bits;
  const u32 exp32 = (bits >> 23) & 0xFF;
  const u32 mant23 = bits & 0x7FFFFF;
  if (exp32 == 0xFF)
    return mant23 ? inf | 1u : (bits >> 31 ? 0 : inf);
  if (bits >> 31)
    return 0;
  const u32 drop = 23 - mantissa_bits;
  if (exp32 > 112) {
    const u32 packed =
        (((exp32 - 112) << 23 | mant23) + (1u << (drop - 1))) >> drop;
    return base::Min(packed, inf);
  }
  const u32 shift = drop + 113 - (exp32 ? exp32 : 1);
  const u32 m = exp32 ? mant23 | 0x800000 : mant23;
  return shift < 32 ? (m + (1u << (shift - 1))) >> shift : 0;
}

void UnpackR11G11B10(u32 packed, u8* dst) {
  const float value[4] = {
      UnpackUnsignedFloat(packed, 6),
      UnpackUnsignedFloat(packed >> 11, 6),
      UnpackUnsignedFloat(packed >> 22, 5),
      1.f,
  };
  std::memcpy(dst, value, sizeof(value));
}

u32 PackR11G11B10(const u8* src) {
  float value[4];
  std::memcpy(value, src, sizeof(value));
  return PackUnsignedFloat(value[0], 6) |
         (PackUnsignedFloat(value[1], 6) << 11) |
         (PackUnsignedFloat(value[2], 5) << 22);
}

}  // namespace gpu::render::cs

namespace gpu::render {

u32 PackR11G11B10Texel(const u8* rgba32f) {
  return PackR11G11B10(rgba32f);
}

void UnpackR11G11B10Texel(u32 packed, u8* rgba32f) {
  UnpackR11G11B10(packed, rgba32f);
}

}  // namespace gpu::render
