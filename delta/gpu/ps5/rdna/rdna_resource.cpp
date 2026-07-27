/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) resource decode + per-draw texture tracking. See
 * rdna_resource.h.
 */

#include "gpu/ps5/rdna/rdna_resource.h"

#include "gpu/guest_memory.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <utility>

namespace gpu::rdna {
namespace {

using gpu::gcn::Enc;
using gpu::gcn::Inst;
using gpu::gcn::MimgBindingPlan;
using gpu::gcn::Program;
using gpu::gcn::TImage;

bool InGuest(uint64_t a) {
  return a >= 0x10000ull && a < 0x1000000000000ull;
}

bool GuestRange(uint64_t address, uint64_t bytes) {
  return bytes && InGuest(address) && bytes <= 0x1000000000000ull - address &&
         gpu::IsReadableRange(address, bytes);
}

bool MimgArrayed(uint32_t dim) {
  return dim == 5;
}

// gfx10.3 T#s carry a 9-bit unified format enum (word1 [28:20]). Values 1..77
// are the buffer format table (GPU Shader Core ISA spec 4.x "Buffer Format
// Conversions"); 130+ are image-only (SRGB, packed 16-bit, BCn). Map the
// sampled subset onto the GCN (dfmt,nfmt) pairs the shared renderer's
// GuestTextureFormat() understands; unmapped formats yield (0,0) so the upload
// declines (white fallback) instead of misreading texels.
void Gfx10ImgFormat(uint32_t gfmt, uint32_t& dfmt, uint32_t& nfmt) {
  switch (gfmt) {
    case 1:
      dfmt = 1;
      nfmt = 0;
      break;  // 8_UNORM
    case 2:
      dfmt = 1;
      nfmt = 1;
      break;  // 8_SNORM
    case 5:
      dfmt = 1;
      nfmt = 4;
      break;  // 8_UINT
    case 6:
      dfmt = 1;
      nfmt = 5;
      break;  // 8_SINT
    case 7:
      dfmt = 2;
      nfmt = 0;
      break;  // 16_UNORM
    case 11:
      dfmt = 2;
      nfmt = 4;
      break;  // 16_UINT
    case 13:
      dfmt = 2;
      nfmt = 7;
      break;  // 16_FLOAT
    case 14:
      dfmt = 3;
      nfmt = 0;
      break;  // 8_8_UNORM
    case 18:
      dfmt = 3;
      nfmt = 4;
      break;  // 8_8_UINT
    case 20:
      dfmt = 4;
      nfmt = 4;
      break;  // 32_UINT
    case 21:
      dfmt = 4;
      nfmt = 5;
      break;  // 32_SINT
    case 22:
      dfmt = 4;
      nfmt = 7;
      break;  // 32_FLOAT (also depth-resolve key)
    case 23:
      dfmt = 5;
      nfmt = 0;
      break;  // 16_16_UNORM
    case 27:
      dfmt = 5;
      nfmt = 4;
      break;  // 16_16_UINT
    case 29:
      dfmt = 5;
      nfmt = 7;
      break;  // 16_16_FLOAT
    case 36:
      dfmt = 6;
      nfmt = 7;
      break;  // 11_11_10_FLOAT
    case 44:
      dfmt = 9;
      nfmt = 0;
      break;  // 2_10_10_10_UNORM
    case 50:
      dfmt = 8;
      nfmt = 0;
      break;  // 10_10_10_2_UNORM
    case 56:
      dfmt = 10;
      nfmt = 0;
      break;  // 8_8_8_8_UNORM
    case 62:
      dfmt = 11;
      nfmt = 4;
      break;  // 32_32_UINT
    case 64:
      dfmt = 11;
      nfmt = 7;
      break;  // 32_32_FLOAT
    case 71:
      dfmt = 12;
      nfmt = 7;
      break;  // 16_16_16_16_FLOAT
    case 74:
      dfmt = 13;
      nfmt = 7;
      break;  // 32_32_32_FLOAT
    case 77:
      dfmt = 14;
      nfmt = 7;
      break;  // 32_32_32_32_FLOAT
    case 57:
      dfmt = 10;
      nfmt = 1;
      break;  // 8_8_8_8_SNORM
    case 60:
      dfmt = 10;
      nfmt = 4;
      break;  // 8_8_8_8_UINT
    case 130:
      dfmt = 10;
      nfmt = 9;
      break;  // 8_8_8_8_SRGB
    case 169:
      dfmt = 35;
      nfmt = 0;
      break;  // BC1
    case 170:
      dfmt = 35;
      nfmt = 9;
      break;
    case 171:
      dfmt = 36;
      nfmt = 0;
      break;  // BC2
    case 172:
      dfmt = 36;
      nfmt = 9;
      break;
    case 173:
      dfmt = 37;
      nfmt = 0;
      break;  // BC3
    case 174:
      dfmt = 37;
      nfmt = 9;
      break;
    case 175:
      dfmt = 38;
      nfmt = 0;
      break;  // BC4
    case 176:
      dfmt = 38;
      nfmt = 1;
      break;
    case 177:
      dfmt = 39;
      nfmt = 0;
      break;  // BC5
    case 178:
      dfmt = 39;
      nfmt = 1;
      break;
    case 181:
      dfmt = 41;
      nfmt = 0;
      break;  // BC7
    case 182:
      dfmt = 41;
      nfmt = 9;
      break;
    default:
      dfmt = 0;
      nfmt = 0;
      break;
  }
}

struct ScalarEval {
  static constexpr uint32_t kRegs = 136;
  uint32_t sgpr[kRegs] = {};
  bool known[kRegs] = {};
  bool scc = false;
  bool scc_known = false;

  ScalarEval(const uint32_t* user_data,
             uint32_t user_sgprs,
             uint32_t user_sgpr_base) {
    const uint32_t count = std::min(user_sgprs, 32u);
    for (uint32_t i = 0; i < count && user_sgpr_base + i < kRegs; i++) {
      sgpr[user_sgpr_base + i] = user_data[i];
      known[user_sgpr_base + i] = true;
    }
    if (user_sgpr_base == 8) {
      sgpr[3] = 0x0101;
      known[3] = true;
    }
  }

  bool AllKnown(uint32_t s, uint32_t n) const {
    if (s + n > kRegs)
      return false;
    for (uint32_t i = 0; i < n; i++)
      if (!known[s + i])
        return false;
    return true;
  }
  uint64_t Ptr(uint32_t s) const {
    return sgpr[s] | (static_cast<uint64_t>(sgpr[s + 1] & 0xFFFF) << 32);
  }
  void Set(uint32_t s, uint32_t value) {
    if (s < kRegs) {
      sgpr[s] = value;
      known[s] = true;
    }
  }
  void Clear(uint32_t s) {
    if (s < kRegs)
      known[s] = false;
  }

  bool Source(uint32_t field, uint32_t literal, uint32_t& value) const {
    if (field == 125) {
      value = 0;
      return true;
    }
    if (field <= 127) {
      if (!known[field])
        return false;
      value = sgpr[field];
      return true;
    }
    if (field == 128)
      value = 0;
    else if (field >= 129 && field <= 192)
      value = field - 128;
    else if (field >= 193 && field <= 208)
      value = static_cast<uint32_t>(-static_cast<int32_t>(field - 192));
    else if (field == 240)
      value = 0x3f000000u;
    else if (field == 241)
      value = 0xbf000000u;
    else if (field == 242)
      value = 0x3f800000u;
    else if (field == 243)
      value = 0xbf800000u;
    else if (field == 244)
      value = 0x40000000u;
    else if (field == 245)
      value = 0xc0000000u;
    else if (field == 246)
      value = 0x40800000u;
    else if (field == 247)
      value = 0xc0800000u;
    else if (field == 255)
      value = literal;
    else
      return false;
    return true;
  }
  bool SourceHi(uint32_t field, uint32_t& value) const {
    if (field <= 126)
      return Source(field + 1, 0, value);
    value = 0;
    return true;
  }

  void Step(const Inst& inst) {
    if (inst.enc == Enc::kSop1) {
      const uint32_t sdst = (inst.raw[0] >> 16) & 0x7F,
                     ssrc = inst.raw[0] & 0xFF;
      if (inst.opcode == 0x03) {
        uint32_t value;
        if (Source(ssrc, inst.literal, value))
          Set(sdst, value);
        else
          Clear(sdst);
      } else if (inst.opcode == 0x04) {
        uint32_t lo, hi;
        if (Source(ssrc, inst.literal, lo) && SourceHi(ssrc, hi)) {
          Set(sdst, lo);
          Set(sdst + 1, hi);
        } else {
          Clear(sdst);
          Clear(sdst + 1);
        }
      } else if (inst.opcode != 0x20) {
        Clear(sdst);
        if (inst.opcode == 0x06 || inst.opcode == 0x08 || inst.opcode == 0x0A ||
            inst.opcode == 0x21 || (inst.opcode >= 0x24 && inst.opcode <= 0x2B))
          Clear(sdst + 1);
      }
      return;
    }
    if (inst.enc == Enc::kSop2) {
      const uint32_t sdst = (inst.raw[0] >> 16) & 0x7F;
      if (inst.opcode == 0x0A || inst.opcode == 0x0B) {
        if (!scc_known) {
          Clear(sdst);
          if (inst.opcode == 0x0B)
            Clear(sdst + 1);
          return;
        }
        const uint32_t source =
            scc ? inst.raw[0] & 0xFF : (inst.raw[0] >> 8) & 0xFF;
        uint32_t value;
        if (Source(source, inst.literal, value))
          Set(sdst, value);
        else
          Clear(sdst);
        if (inst.opcode == 0x0B) {
          if (SourceHi(source, value))
            Set(sdst + 1, value);
          else
            Clear(sdst + 1);
        }
        return;
      }
      uint32_t a, b;
      if (!Source(inst.raw[0] & 0xFF, inst.literal, a) ||
          !Source((inst.raw[0] >> 8) & 0xFF, inst.literal, b)) {
        Clear(sdst);
        if (DecodeScalarWrite(inst).count == 2)
          Clear(sdst + 1);
        scc_known = false;
        return;
      }
      uint32_t value;
      switch (inst.opcode) {
        case 0x00:
        case 0x02:
          value = a + b;
          break;
        case 0x01:
        case 0x03:
          value = a - b;
          break;
        case 0x0e:
          value = a & b;
          break;
        case 0x10:
          value = a | b;
          break;
        case 0x12:
          value = a ^ b;
          break;
        case 0x14:
          value = a & ~b;
          break;
        case 0x16:
          value = a | ~b;
          break;
        case 0x18:
          value = ~(a & b);
          break;
        case 0x1a:
          value = ~(a | b);
          break;
        case 0x1c:
          value = ~(a ^ b);
          break;
        case 0x1e:
          value = a << (b & 31);
          break;
        case 0x20:
          value = a >> (b & 31);
          break;
        case 0x22:
          value = static_cast<uint32_t>(static_cast<int32_t>(a) >> (b & 31));
          break;
        case 0x26:
          value = a * b;
          break;
        case 0x27: {
          const uint32_t offset = b & 31;
          const uint32_t width = (b >> 16) & 0x7F;
          value = width >= 32 - offset
                      ? a >> offset
                      : (a >> offset) & ((uint32_t{1} << width) - 1);
          break;
        }
        case 0x2f:
        case 0x30:
        case 0x31:
        case 0x32:
          value = (a << (inst.opcode - 0x2e)) + b;
          break;
        default:
          Clear(sdst);
          if (inst.opcode == 0x0B || inst.opcode == 0x29 ||
              (inst.opcode >= 0x0F && inst.opcode <= 0x23 && (inst.opcode & 1)))
            Clear(sdst + 1);
          scc_known = false;
          return;
      }
      Set(sdst, value);
      if (inst.opcode <= 0x03 || (inst.opcode >= 0x2f && inst.opcode <= 0x32))
        scc_known = false;
      switch (inst.opcode) {
        case 0x0e:
        case 0x10:
        case 0x12:
        case 0x14:
        case 0x16:
        case 0x18:
        case 0x1a:
        case 0x1c:
        case 0x1e:
        case 0x20:
        case 0x22:
        case 0x27:
          scc = value != 0;
          scc_known = true;
          break;
      }
      return;
    }
    if (inst.enc == Enc::kSopk) {
      const uint32_t sdst = (inst.raw[0] >> 16) & 0x7F;
      const uint32_t imm = inst.raw[0] & 0xFFFF;
      const uint32_t simm = static_cast<uint32_t>(
          static_cast<int32_t>(static_cast<int16_t>(imm)));
      if (inst.opcode == 0x00) {
        Set(sdst, simm);
      } else if (inst.opcode == 0x02) {
        if (!scc_known)
          Clear(sdst);
        else if (scc)
          Set(sdst, simm);
      } else if (inst.opcode >= 0x03 && inst.opcode <= 0x0E) {
        if (!known[sdst]) {
          scc_known = false;
          return;
        }
        const int32_t a = static_cast<int32_t>(sgpr[sdst]);
        const int32_t b = static_cast<int32_t>(simm);
        switch (inst.opcode) {
          case 0x03:
            scc = a == b;
            break;
          case 0x04:
            scc = a != b;
            break;
          case 0x05:
            scc = a > b;
            break;
          case 0x06:
            scc = a >= b;
            break;
          case 0x07:
            scc = a < b;
            break;
          case 0x08:
            scc = a <= b;
            break;
          case 0x09:
            scc = sgpr[sdst] == imm;
            break;
          case 0x0A:
            scc = sgpr[sdst] != imm;
            break;
          case 0x0B:
            scc = sgpr[sdst] > imm;
            break;
          case 0x0C:
            scc = sgpr[sdst] >= imm;
            break;
          case 0x0D:
            scc = sgpr[sdst] < imm;
            break;
          case 0x0E:
            scc = sgpr[sdst] <= imm;
            break;
        }
        scc_known = true;
      } else if (inst.opcode == 0x0F || inst.opcode == 0x10) {
        if (known[sdst])
          Set(sdst,
              inst.opcode == 0x0F ? sgpr[sdst] + simm : sgpr[sdst] * simm);
        if (inst.opcode == 0x0F)
          scc_known = false;
      } else if (inst.opcode == 0x12) {
        Set(sdst, 0);
      }
      return;
    }
    if (inst.enc == Enc::kSopc) {
      uint32_t a, b;
      if (!Source(inst.raw[0] & 0xFF, inst.literal, a) ||
          !Source((inst.raw[0] >> 8) & 0xFF, inst.literal, b)) {
        scc_known = false;
        return;
      }
      switch (inst.opcode) {
        case 0x00:
        case 0x06:
          scc = a == b;
          break;
        case 0x01:
        case 0x07:
          scc = a != b;
          break;
        case 0x02:
          scc = static_cast<int32_t>(a) > static_cast<int32_t>(b);
          break;
        case 0x03:
          scc = static_cast<int32_t>(a) >= static_cast<int32_t>(b);
          break;
        case 0x04:
          scc = static_cast<int32_t>(a) < static_cast<int32_t>(b);
          break;
        case 0x05:
          scc = static_cast<int32_t>(a) <= static_cast<int32_t>(b);
          break;
        case 0x08:
          scc = a > b;
          break;
        case 0x09:
          scc = a >= b;
          break;
        case 0x0a:
          scc = a < b;
          break;
        case 0x0b:
          scc = a <= b;
          break;
        case 0x0c:
          scc = ((a >> (b & 31)) & 1) == 0;
          break;
        case 0x0d:
          scc = ((a >> (b & 31)) & 1) != 0;
          break;
        default:
          scc_known = false;
          return;
      }
      scc_known = true;
      return;
    }
    if (inst.enc != Enc::kSmrd)
      return;
    const Smem smem = DecodeSmem(inst);
    const uint32_t dwords = SmemLoadCount(smem.op);
    if (!dwords)
      return;
    const bool buffer = smem.op >= 0x08;
    const bool base_known = AllKnown(smem.sbase, buffer ? 4 : 2);
    const uint64_t base = base_known ? Ptr(smem.sbase) : 0;
    uint32_t soffset = 0;
    const bool offset_known = Source(smem.soffset, 0, soffset);
    const int64_t immediate = buffer
                                  ? static_cast<int64_t>(inst.raw[1] & 0xFFFFF)
                                  : static_cast<int64_t>(smem.offset);
    const int64_t byte_offset =
        static_cast<int64_t>(soffset & ~3u) + (immediate & ~int64_t{3});
    for (uint32_t i = 0; i < dwords; i++)
      Clear(smem.sdst + i);
    if (!base_known || !offset_known || byte_offset < 0 ||
        static_cast<uint64_t>(byte_offset) > UINT64_MAX - base)
      return;
    const uint64_t address = base + static_cast<uint64_t>(byte_offset);
    if (!GuestRange(address, static_cast<uint64_t>(dwords) * 4))
      return;
    const auto* src = reinterpret_cast<const uint32_t*>(address);
    for (uint32_t i = 0; i < dwords; i++)
      Set(smem.sdst + i, src[i]);
  }
};

}  // namespace

MimgBindingPlan RdnaPlanMimg(const Program& program) {
  MimgBindingPlan plan;
  struct BindingKey {
    uint32_t srsrc;
    uint32_t ssamp;
    uint32_t flags;
    uint32_t versions[12];
  };
  std::vector<BindingKey> keys;
  uint32_t versions[136] = {};
  uint32_t generation = 1;
  for (const Inst& inst : program) {
    if (inst.enc == Enc::kMimg) {
      const uint32_t w0 = inst.raw[0], w1 = inst.raw[1], op = inst.opcode;
      const uint32_t dim = (w0 >> 3) & 0x7;
      const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
      const bool sampling = op >= 0x20;
      const bool storage = op == 0x08 || op == 0x09;
      const uint32_t ssamp = sampling ? ((w1 >> 21) & 0x1F) * 4 : 0xFFu;
      BindingKey key{
          .srsrc = srsrc,
          .ssamp = ssamp,
          .flags = static_cast<uint32_t>(MimgArrayed(dim)) |
                   ((op == 0x28 || op == 0x2f ? 1u : 0u) << 1) |
                   ((op == 0x47 ? 1u : 0u) << 2) |
                   (static_cast<uint32_t>(storage) << 3),
      };
      for (uint32_t i = 0; i < 8; i++)
        key.versions[i] = versions[srsrc + i];
      if (sampling)
        for (uint32_t i = 0; i < 4; i++)
          key.versions[8 + i] = versions[ssamp + i];

      uint32_t binding = static_cast<uint32_t>(keys.size());
      for (uint32_t i = 0; i < keys.size(); i++)
        if (std::memcmp(&keys[i], &key, sizeof(key)) == 0) {
          binding = i;
          break;
        }
      if (binding == keys.size()) {
        keys.push_back(key);
        plan.binding_srsrc.push_back(srsrc);
        plan.binding_storage.push_back(storage);
      }
      plan.binding_by_pc[inst.pc] = binding;
    }

    const ScalarWrite write = DecodeScalarWrite(inst);
    for (uint32_t i = 0; i < write.count && write.first + i < 136; i++)
      versions[write.first + i] = generation;
    generation++;
  }
  return plan;
}

TImage DecodeTImage(const uint32_t* d) {
  TImage t;
  const uint64_t base_units = d[0] | (static_cast<uint64_t>(d[1] & 0xFF) << 32);
  t.base = base_units << 8;
  t.min_lod = (d[1] >> 8) & 0xFFF;
  t.width = (((d[1] >> 30) & 0x3) | ((d[2] & 0xFFF) << 2)) + 1;
  t.height = ((d[2] >> 14) & 0x3FFF) + 1;
  t.base_mip = (d[3] >> 12) & 0xF;
  const uint32_t last_level = (d[3] >> 16) & 0xF;
  const uint32_t sw_mode = (d[3] >> 20) & 0x1F;
  t.type = (d[3] >> 28) & 0xF;
  for (int i = 0; i < 4; i++)
    t.dst_sel[i] = (d[3] >> (i * 3)) & 0x7;
  const uint32_t depth = d[4] & 0xFFFF;
  t.base_array = (d[4] >> 16) & 0xFFFF;
  const uint32_t max_mip = (d[5] >> 4) & 0xF;

  t.pitch = t.width;
  t.arrayed = t.type == 12 || t.type == 13;              // 1D/2D array
  const bool volumetric = t.type == 10 || t.type == 11;  // 3D / cube
  t.layers = (t.arrayed || volumetric) ? depth + 1 : 1;
  t.view_layers =
      t.arrayed ? std::max<uint32_t>(depth + 1 - t.base_array, 1) : 1;
  t.mip_levels = max_mip + 1;
  t.view_mips = std::max<uint32_t>(last_level + 1 - t.base_mip, 1);
  const bool valid_array = !t.arrayed || t.base_array <= depth;
  const uint32_t gfmt = (d[1] >> 20) & 0x1FF;
  Gfx10ImgFormat(gfmt, t.dfmt, t.nfmt);
  // gfx10 swizzle mode 0 = SW_LINEAR -> the renderer's linear index. The
  // "standard" modes (256 B / 4 KiB / 64 KiB, ids 1/5/9) map onto the gfx10
  // detiler's own id range. Everything else (Z/D/R, the _X pipe-XOR and _T
  // variants) has no detiler yet, so it is shifted past the valid range:
  // BuildTextureLayout32 rejects it and the draw gets the white fallback
  // instead of scrambled texels.
  // DELTA_GPU_SWCENSUS: which gfx10 swizzle modes this title's textures use --
  // the detiler only covers linear and the three "standard" modes, and anything
  // else is rejected into the white fallback (flat-coloured quads).
  if (std::getenv("DELTA_GPU_SWCENSUS")) {
    static uint32_t seen[64] = {};
    if (sw_mode < 64 && seen[sw_mode]++ == 0)
      std::fprintf(stderr, "[swcensus] sw_mode=%u first seen (%ux%u gfmt=%u)\n",
                   sw_mode, t.width, t.height, gfmt);
    static uint32_t seen_sel[4096] = {};
    const uint32_t packed = (t.dst_sel[0]) | (t.dst_sel[1] << 3) |
                            (t.dst_sel[2] << 6) | (t.dst_sel[3] << 9);
    if (packed < 4096 && seen_sel[packed]++ == 0)
      std::fprintf(
          stderr,
          "[swcensus] dst_sel = %u,%u,%u,%u (word3=%08x) on %ux%u gfmt=%u\n",
          t.dst_sel[0], t.dst_sel[1], t.dst_sel[2], t.dst_sel[3], d[3], t.width,
          t.height, gfmt);
  }
  switch (sw_mode) {
    case 0:
      t.tiling_idx = 8;
      break;
    case 1:
      t.tiling_idx = 0x50;
      break;
    case 5:
      t.tiling_idx = 0x51;
      break;
    case 9:
      t.tiling_idx = 0x52;
      break;
    default:
      t.tiling_idx = 0x40 + sw_mode;
      break;
  }
  if (sw_mode == 0 && t.dfmt && t.dfmt < 35) {
    // gfx10 linear surfaces align each row to 256 bytes.
    const uint32_t eb = t.dfmt == 12 ? 8 : 4;
    const uint32_t pa = 256 / eb;
    t.pitch = (t.width + pa - 1) & ~(pa - 1);
  }
  static const bool trace = std::getenv("DELTA_AGC_TRACE") != nullptr;
  if (trace) {
    static uint32_t seen[32], n_seen = 0;
    const uint32_t key = (gfmt << 8) | sw_mode;
    bool is_new = true;
    for (uint32_t i = 0; i < n_seen; i++)
      if (seen[i] == key) {
        is_new = false;
        break;
      }
    if (is_new && n_seen < 32) {
      seen[n_seen++] = key;
      std::fprintf(stderr,
                   "[agc] T# base=%#lx %ux%u gfmt=%u sw=%u type=%u mips=%u -> "
                   "dfmt=%u nfmt=%u tiling=%u pitch=%u\n",
                   (unsigned long)t.base, t.width, t.height, gfmt, sw_mode,
                   t.type, t.mip_levels, t.dfmt, t.nfmt, t.tiling_idx, t.pitch);
    }
  }
  // gfx10 mip chains pack their small levels into a shared "mip tail" block
  // whose layout the detiler does not model; only level 0 is addressed
  // correctly, so sample that one rather than reading a wrong offset.
  uint32_t max_levels = 1;
  for (uint32_t extent = std::max(t.width, t.height); extent > 1; extent >>= 1)
    max_levels++;
  const bool valid_mips = t.base_mip <= last_level && last_level < max_levels &&
                          t.base_mip + t.view_mips <= t.mip_levels;
  if (t.tiling_idx >= 0x50 && t.tiling_idx < 0x53 && t.mip_levels > 1) {
    t.mip_levels = 1;
    t.view_mips = 1;
    t.base_mip = 0;
    t.min_lod = 0;
    t.force_lod_zero = true;
  }
  t.valid = InGuest(t.base) && t.dfmt && t.width <= 16384 &&
            t.height <= 16384 && t.layers <= 16384 && valid_array && valid_mips;
  return t;
}

std::vector<TImage> TrackTextures(const uint32_t* ps_code,
                                  const uint32_t* pud,
                                  uint32_t user_sgprs) {
  std::vector<TImage> out;
  if (!ps_code || !pud || !InGuest(reinterpret_cast<uint64_t>(ps_code)))
    return out;
  const Program prog = DecodeShader(ps_code, 4096);
  const MimgBindingPlan plan = RdnaPlanMimg(prog);
  out.resize(plan.binding_srsrc.size());
  std::vector<bool> filled(out.size(), false);
  ScalarEval eval(pud, user_sgprs, 0);

  for (const Inst& in : prog) {
    eval.Step(in);
    if (in.enc != Enc::kMimg)
      continue;
    auto it = plan.binding_by_pc.find(in.pc);
    if (it == plan.binding_by_pc.end() || filled[it->second])
      continue;
    const uint32_t b = it->second;
    const uint32_t w0 = in.raw[0], w1 = in.raw[1], op = in.opcode;
    // DELTA_GPU_TEXRESOLVE: which SGPR quad each sampler's T# comes from, and
    // whether it resolved. A binding that reads back base 0 is either a null
    // descriptor or a chain we failed to walk.
    static const bool tr_resolve =
        std::getenv("DELTA_GPU_TEXRESOLVE") != nullptr;
    const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
    if (tr_resolve && !eval.AllKnown(srsrc, 8)) {
      std::fprintf(stderr,
                   "[texres]   mimg pc=%04x w0=%08x w1=%08x op=%#x nsa=%u "
                   "srsrc_field=%u ssamp_field=%u vaddr=%u vdata=%u\n",
                   in.pc, w0, w1, op, (w0 >> 1) & 3, (w1 >> 16) & 0x1F,
                   (w1 >> 21) & 0x1F, w1 & 0xFF, (w1 >> 8) & 0xFF);
      // An "inline" descriptor that user data never programmed has to come from
      // somewhere else in the shader: list every scalar write to its registers.
      for (const Inst& w : prog) {
        uint32_t d0 = 0xFFFF, cnt = 1;
        if (w.enc == Enc::kSop1) {
          d0 = (w.raw[0] >> 16) & 0x7F;
          cnt = w.opcode == 0x04 ? 2 : 1;
        } else if (w.enc == Enc::kSop2)
          d0 = (w.raw[0] >> 16) & 0x7F;
        else if (w.enc == Enc::kSmrd && w.opcode <= 0x0C) {
          d0 = (w.raw[0] >> 6) & 0x7F;
          cnt = 8;
        }
        if (d0 == 0xFFFF)
          continue;
        if (d0 + cnt <= srsrc || d0 >= srsrc + 8)
          continue;
        std::fprintf(
            stderr,
            "[texres]   writer pc=%04x enc=%d op=%#x -> s%u..%u (%08x %08x)\n",
            w.pc, (int)w.enc, w.opcode, d0, d0 + cnt - 1, w.raw[0], w.raw[1]);
      }
    }
    if (tr_resolve)
      std::fprintf(stderr, "[texres] binding=%u srsrc=s%u known=%d\n", b, srsrc,
                   eval.AllKnown(srsrc, 8));
    if (eval.AllKnown(srsrc, 8)) {
      out[b] = DecodeTImage(&eval.sgpr[srsrc]);
      out[b].arrayed = MimgArrayed((w0 >> 3) & 0x7);
      out[b].depth_compare = op == 0x28 || op == 0x2f;
      out[b].force_lod_zero = op == 0x47;
      out[b].storage = op == 0x08 || op == 0x09;
      const uint32_t ssamp = ((w1 >> 21) & 0x1F) * 4;
      if (op >= 0x20 && eval.AllKnown(ssamp, 4)) {
        std::memcpy(out[b].sampler, &eval.sgpr[ssamp], sizeof(out[b].sampler));
        out[b].sampler_valid = true;
      }
    }
    filled[b] = true;
  }
  return out;
}

std::unordered_map<uint32_t, BufferResource> ResolveBuffers(
    const uint32_t* code,
    const uint32_t* user_data,
    uint32_t user_sgprs,
    uint32_t user_sgpr_base) {
  std::unordered_map<uint32_t, BufferResource> out;
  if (!code || !user_data || !InGuest(reinterpret_cast<uint64_t>(code)))
    return out;
  ScalarEval eval(user_data, user_sgprs, user_sgpr_base);
  for (const Inst& inst : DecodeShader(code, 4096)) {
    if (inst.enc == Enc::kSmrd && SmemLoadCount(inst.opcode)) {
      const Smem smem = DecodeSmem(inst);
      const bool buffer = smem.op >= 0x08;
      if (eval.AllKnown(smem.sbase, buffer ? 4 : 2)) {
        BufferResource resource;
        resource.base = eval.Ptr(smem.sbase);
        out.emplace(inst.pc, resource);
      }
    } else if ((inst.enc == Enc::kMubuf || inst.enc == Enc::kMtbuf) &&
               inst.opcode <= 0x03) {
      const uint32_t srsrc = ((inst.raw[1] >> 16) & 0x1F) * 4;
      if (eval.AllKnown(srsrc, 4)) {
        BufferResource resource;
        resource.base = eval.Ptr(srsrc);
        std::memcpy(resource.descriptor, &eval.sgpr[srsrc],
                    sizeof(resource.descriptor));
        resource.descriptor_valid = true;
        out.emplace(inst.pc, resource);
      }
    }
    eval.Step(inst);
  }
  return out;
}

}  // namespace gpu::rdna
