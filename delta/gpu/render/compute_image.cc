/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

// CPU staging of image resources: detile into the linear layout a dispatch
// binds, and retile its writes back.

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

// Bit arithmetic rather than frexp/lround: this packs every texel of a
// float-staged R11G11B10 target on each writeback. Rounds to nearest, ties
// away, as lround did.
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
  // Denormal in the target: the implicit bit joins the shifted mantissa.
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

bool StageCsImage(const ComputeInfo::Res& res, void* dst) {
  gcn::TextureLayout32 tiled, linear;
  if (!BuildCsImageLayouts(res, tiled, linear))
    return false;
  const bool bc6 = res.dfmt == 40;
  const bool bc1 = res.dfmt == 35;
  const bool direct = !bc6 && !bc1 && res.elem_bytes == res.stage_elem_bytes;
  // Every path but the block decodes writes each staged texel whole, so a
  // tightly packed layout needs no clearing first (the widened 8/16-bit planes
  // of UC2's video decoder spent more time in the memset than in the copy).
  bool fills_complete_layout = !bc6 && !bc1;
  u64 filled_bytes = 0;
  for (u32 mip = 0; fills_complete_layout && mip < linear.mip_levels; ++mip) {
    const auto& level = linear.mips[mip];
    const u64 logical_bytes = static_cast<u64>(level.width) * level.height *
                              linear.layers * res.stage_elem_bytes;
    fills_complete_layout =
        level.offset == filled_bytes && level.pitch == level.width &&
        level.stored_height == level.height && level.size == logical_bytes;
    filled_bytes = level.offset + level.size;
  }
  fills_complete_layout = fills_complete_layout && filled_bytes == res.size;
  if (!fills_complete_layout)
    std::memset(dst, 0, res.size);
  base::Vector<u8> tight;
  if (!direct)
    tight.resize(static_cast<size_t>(tiled.mips[0].width) *
                 tiled.mips[0].height * res.elem_bytes);
  // DELTA_GPU_DETILEDUMP=<base>: write the de-tiled level-0 bytes of that guest
  // surface to <dumpdir>/detiled.bin once, so the swizzle can be checked
  // against an offline decode of the same texture.
  static bool detile_dumped = false;
  for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& src_level = tiled.mips[mip];
    const auto& dst_level = linear.mips[mip];
    for (u32 layer = 0; layer < tiled.layers; layer++) {
      u8* level_dst = static_cast<u8*>(dst) + dst_level.offset +
                      static_cast<u64>(layer) * dst_level.pitch *
                          dst_level.stored_height * res.stage_elem_bytes;
      if (direct) {
        if (!gcn::DetileTextureMip32Pitched(
                reinterpret_cast<const void*>(res.base), level_dst,
                static_cast<size_t>(dst_level.pitch) * res.stage_elem_bytes,
                tiled, mip, layer))
          return false;
        if (kDetileDump && res.base == kDetileDump && !mip && !layer &&
            !detile_dumped) {
          detile_dumped = true;
          char p[256];
          std::snprintf(p, sizeof p, "%s/detiled.bin", DumpDir());
          if (FILE* f = std::fopen(p, "wb")) {
            std::fwrite(level_dst, 1,
                        static_cast<size_t>(dst_level.pitch) *
                            dst_level.stored_height * res.stage_elem_bytes,
                        f);
            std::fclose(f);
            BASE_LOGI("detiledump", "{:#x} pitch={} h={} elem={}",
                      (unsigned long)res.base, dst_level.pitch,
                      dst_level.stored_height, res.stage_elem_bytes);
          }
        }
        continue;
      }
      if (!gcn::DetileTextureMip32(reinterpret_cast<const void*>(res.base),
                                   tight.data(), tiled, mip, layer))
        return false;
      if (bc1) {
        gcn::DetileParallelRows(src_level.height, [&](u32 y0, u32 y1) {
          for (u32 by = y0; by < y1; ++by)
            for (u32 bx = 0; bx < src_level.width; ++bx) {
              u8 rgba[4][4][4];
              const u8* block =
                  tight.data() +
                  (static_cast<size_t>(by) * src_level.width + bx) * 8;
              bcdec_bc1(block, rgba, 16);
              for (u32 y = 0; y < 4 && by * 4 + y < dst_level.height; ++y) {
                const size_t offset =
                    (static_cast<size_t>(by * 4 + y) * dst_level.pitch +
                     bx * 4) *
                    4;
                std::memcpy(level_dst + offset, rgba[y],
                            base::Min(4u, dst_level.width - bx * 4) * 4);
              }
            }
        });
        continue;
      }
      if (bc6) {
        gcn::DetileParallelRows(src_level.height, [&](u32 y0, u32 y1) {
          for (u32 by = y0; by < y1; ++by) {
            for (u32 bx = 0; bx < src_level.width; ++bx) {
              float rgb[4][4][3];
              const u8* block =
                  tight.data() +
                  (static_cast<size_t>(by) * src_level.width + bx) * 16;
              bcdec_bc6h_float(block, rgb, 12, res.nfmt == 1);
              for (u32 y = 0; y < 4 && by * 4 + y < dst_level.height; ++y) {
                for (u32 x = 0; x < 4 && bx * 4 + x < dst_level.width; ++x) {
                  const float rgba[4] = {rgb[y][x][0], rgb[y][x][1],
                                         rgb[y][x][2], 1.f};
                  const size_t offset =
                      (static_cast<size_t>(by * 4 + y) * dst_level.pitch +
                       bx * 4 + x) *
                      16;
                  std::memcpy(level_dst + offset, rgba, sizeof(rgba));
                }
              }
            }
          }
        });
        continue;
      }
      gcn::DetileParallelRows(src_level.height, [&](u32 y0, u32 y1) {
        for (u32 y = y0; y < y1; y++) {
          u8* dst_row = level_dst + static_cast<size_t>(y) * dst_level.pitch *
                                        res.stage_elem_bytes;
          const u8* src_row = tight.data() + static_cast<size_t>(y) *
                                                 src_level.width *
                                                 res.elem_bytes;
          if (res.dfmt == 6) {
            for (u32 x = 0; x < src_level.width; x++) {
              u32 packed;
              std::memcpy(&packed, src_row + static_cast<size_t>(x) * 4, 4);
              UnpackR11G11B10(packed, dst_row + static_cast<size_t>(x) * 16);
            }
          } else if (res.elem_bytes == 1) {
            for (u32 x = 0; x < src_level.width; x++) {
              const u32 expanded = src_row[x];
              std::memcpy(dst_row + static_cast<size_t>(x) * 4, &expanded, 4);
            }
          } else {
            for (u32 x = 0; x < src_level.width; x++) {
              u16 value;
              std::memcpy(&value, src_row + static_cast<size_t>(x) * 2, 2);
              const u32 expanded = value;
              std::memcpy(dst_row + static_cast<size_t>(x) * 4, &expanded, 4);
            }
          }
        }
      });
    }
  }
  return true;
}

bool WritebackCsImage(const ComputeInfo::Res& res, const void* src) {
  if (res.dfmt == 35 || res.dfmt == 40)
    return false;  // Compressed textures are sampled, never image-stored.
  gcn::TextureLayout32 tiled, linear;
  // A bail here leaves the destination holding whatever it held before the
  // dispatch, which for a first upload is zeros, indistinguishable from a
  // dispatch that never ran unless it says so (DELTA_GPU_CS_WB_AUDIT).
  if (!BuildCsImageLayouts(res, tiled, linear)) {
    if (kCsWbAudit) {
      static int n = 0;
      if (n++ < 16) {
        gcn::TextureLayout32 t2, l2;
        const u32 stage_tiling = res.tiling_idx == 31 ? 31 : 8;
        const bool tok = gcn::BuildTextureLayout32(
            t2, res.width, res.height, res.pitch, res.layers, res.mip_levels,
            res.tiling_idx, res.pow2_pad, res.elem_bytes);
        const bool lok = gcn::BuildTextureLayout32(
            l2, res.width, res.height, res.pitch, res.layers, res.mip_levels,
            stage_tiling, res.pow2_pad, res.stage_elem_bytes);
        BASE_LOGI("cswb",
                  "image layout REJECT base={:#x} {}x{} layers={} mips={} "
                  "tiling={} elem={}/{} tiledOk={}({} vs guest {}) "
                  "linearOk={}({} vs size {})",
                  (unsigned long long)res.base, res.width, res.height,
                  res.layers, res.mip_levels, res.tiling_idx, res.elem_bytes,
                  res.stage_elem_bytes, (int)tok,
                  (unsigned long long)(tok ? t2.size : 0),
                  (unsigned long long)res.guest_size, (int)lok,
                  (unsigned long long)(lok ? l2.size : 0),
                  (unsigned long long)res.size);
      }
    }
    return false;
  }
  const bool direct = res.elem_bytes == res.stage_elem_bytes;
  base::Vector<u8> tight;
  if (!direct)
    tight.resize(static_cast<size_t>(res.width) * res.height * res.elem_bytes);
  for (u32 mip = 0; mip < tiled.mip_levels; mip++) {
    const auto& dst_level = tiled.mips[mip];
    const auto& src_level = linear.mips[mip];
    for (u32 layer = 0; layer < tiled.layers; layer++) {
      const u8* level_src = static_cast<const u8*>(src) + src_level.offset +
                            static_cast<u64>(layer) * src_level.pitch *
                                src_level.stored_height * res.stage_elem_bytes;
      if (direct) {
        if (!gcn::RetileTextureMip32Pitched(
                level_src,
                static_cast<size_t>(src_level.pitch) * res.stage_elem_bytes,
                reinterpret_cast<void*>(res.base), tiled, mip, layer)) {
          if (kCsWbAudit) {
            static int n = 0;
            if (n++ < 16)
              BASE_LOGI("cswb",
                        "image retile REJECT base={:#x} mip={} "
                        "layer={} tiling={}",
                        (unsigned long long)res.base, mip, layer,
                        res.tiling_idx);
          }
          return false;
        }
        continue;
      }
      gcn::DetileParallelRows(dst_level.height, [&](u32 y0, u32 y1) {
        for (u32 y = y0; y < y1; y++) {
          u8* dst_row = tight.data() + static_cast<size_t>(y) *
                                           dst_level.width * res.elem_bytes;
          const u8* src_row = level_src + static_cast<size_t>(y) *
                                              src_level.pitch *
                                              res.stage_elem_bytes;
          if (res.dfmt == 6) {
            for (u32 x = 0; x < dst_level.width; x++) {
              const u32 packed =
                  PackR11G11B10(src_row + static_cast<size_t>(x) * 16);
              std::memcpy(dst_row + static_cast<size_t>(x) * 4, &packed, 4);
            }
          } else if (res.elem_bytes == 1) {
            for (u32 x = 0; x < dst_level.width; x++) {
              u32 expanded;
              std::memcpy(&expanded, src_row + static_cast<size_t>(x) * 4, 4);
              dst_row[x] = static_cast<u8>(expanded);
            }
          } else {
            for (u32 x = 0; x < dst_level.width; x++) {
              u32 expanded;
              std::memcpy(&expanded, src_row + static_cast<size_t>(x) * 4, 4);
              const u16 value = static_cast<u16>(expanded);
              std::memcpy(dst_row + static_cast<size_t>(x) * 2, &value, 2);
            }
          }
        }
      });
      if (!gcn::RetileTextureMip32(tight.data(),
                                   reinterpret_cast<void*>(res.base), tiled,
                                   mip, layer))
        return false;
    }
  }
  return true;
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
