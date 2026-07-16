/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN resource tracking. See gcn_resource.h.
 */

#include "gcn_resource.h"
#include "gcn_decode.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace gpu::gcn {
namespace {
const bool g_trace = std::getenv("DELTA_GPU_TRACE") != nullptr;

bool guestRange(uint64_t address, uint64_t size) {
  constexpr uint64_t lo = 0x1000000000ull, hi = 0x20000000000ull;
  return size && address >= lo && address < hi && size <= hi - address;
}
}

// Linux GFX 7.2 V#/T# descriptor fields and format enums:
// https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_sh_mask.h
// https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/asic_reg/gca/gfx_7_2_enum.h
VBuffer decodeVBuffer(const uint32_t *p) {
  // GCN V# (buffer resource descriptor), 4 dwords:
  //  [0]      base_address[31:0]
  //  [1]      base_address[47:32] in [15:0]; stride[13:0] in [29:16]
  //  [2]      num_records
  //  [3]      dst_sel/nfmt/dfmt/...: nfmt[14:12], dfmt[18:15]
  VBuffer v;
  v.base = (static_cast<uint64_t>(p[1] & 0xFFFF) << 32) | p[0];
  v.stride = (p[1] >> 16) & 0x3FFF;
  v.numRecords = p[2];
  v.dfmt = (p[3] >> 15) & 0xF;
  v.nfmt = (p[3] >> 12) & 0x7;
  return v;
}

TImage decodeTImage(const uint32_t *p) {
  // GCN T# (image resource), 8 dwords:
  //  [0] base[39:8] (base = [0] << 8 with high bits from [1])
  //  [1] base_hi[5:0]; mtype_l2[7:6]; min_lod[19:8]; formats
  //  [2] width[13:0]; height[27:14]
  //  [3] base_level[15:12]; last_level[19:16]; tiling_index[24:20];
  //      pow2_pad[25]; type[31:28]
  //  [4] depth[12:0]; pitch[26:13]
  //  [5] base_array[12:0]; last_array[25:13]
  TImage t;
  t.base = ((static_cast<uint64_t>(p[1] & 0x3F) << 32) | p[0]) << 8;
  t.minLod = (p[1] >> 8) & 0xFFF;
  t.dfmt = (p[1] >> 20) & 0x3F;
  t.nfmt = (p[1] >> 26) & 0xF;
  t.width = ((p[2] & 0x3FFF)) + 1;
  t.height = ((p[2] >> 14) & 0x3FFF) + 1;
  t.baseMip = (p[3] >> 12) & 0xF;
  uint32_t lastMip = (p[3] >> 16) & 0xF;
  t.mipLevels = lastMip + 1;
  t.viewMips = lastMip >= t.baseMip ? lastMip - t.baseMip + 1 : 0;
  t.tilingIdx = (p[3] >> 20) & 0x1F;
  t.pow2Pad = (p[3] >> 25) & 1;
  t.type = p[3] >> 28;
  t.pitch = ((p[4] >> 13) & 0x3FFF) + 1;
  if (t.pitch < t.width) t.pitch = t.width;  // fall back to width if unset
  if (t.type == 13) {  // SQ_RSRC_IMG_2D_ARRAY
    t.layers = (p[4] & 0x1FFF) + 1;
    if (t.pow2Pad) {
      uint32_t v = t.layers - 1;
      v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
      t.layers = v + 1;
    }
    t.baseArray = p[5] & 0x1FFF;
    t.viewLayers = 0;
    uint32_t lastArray = (p[5] >> 13) & 0x1FFF;
    if (t.baseArray < t.layers && lastArray >= t.baseArray)
      t.viewLayers = std::min(lastArray, t.layers - 1) - t.baseArray + 1;
  }
  bool supportedType = t.type == 9 || t.type == 13;
  bool validView = t.type != 13 || (t.baseArray < t.layers && t.viewLayers > 0);
  uint32_t maxLevels = 1;
  for (uint32_t extent = std::max(t.width, t.height); extent > 1; extent >>= 1)
    maxLevels++;
  bool validMips = t.viewMips && t.mipLevels <= maxLevels;
  t.valid = guestRange(t.base, 1) && supportedType &&
                t.width <= 8192 && t.height <= 8192 && t.layers <= 8192 &&
                validView && validMips;
  return t;
}

// SMRD operand fields (GFX6/7).
struct Smrd {
  uint32_t op, sdst, sbase, offset;
  bool imm;
};
Smrd decodeSmrd(uint32_t w) {
  Smrd s;
  s.op = (w >> 22) & 0x1F;
  s.sdst = (w >> 15) & 0x7F;
  s.sbase = (w >> 9) & 0x3F;  // SGPR pair index; actual base SGPR = sbase*2
  s.imm = (w >> 8) & 1;
  s.offset = w & 0xFF;
  return s;
}

std::vector<VBuffer> trackVertexBuffers(const uint32_t *fetchCode,
                                        uint32_t maxDwords,
                                        const uint32_t *vsUserData) {
  std::vector<VBuffer> result;
  if (!fetchCode || !vsUserData)
    return result;
  auto insts = decode(fetchCode, maxDwords);

  // The fetch shader loads each attribute's V# with an s_load_dwordx4 whose
  // SBASE is a user-SGPR pair holding the vertex-buffer-table pointer, at byte
  // offset (offset*4 for imm). Recover the table pointer from the user data and
  // read the V# there.
  for (const auto &in : insts) {
    if (in.enc != Enc::smrd)
      continue;
    Smrd s = decodeSmrd(in.raw[0]);
    if (s.op != 0x02)  // s_load_dwordx4 (a 4-dword V#)
      continue;
    uint32_t baseSgpr = s.sbase * 2;  // user_data index of the table pointer
    if (baseSgpr + 1 >= 16)
      continue;
    uint64_t table = (static_cast<uint64_t>(vsUserData[baseSgpr + 1] & 0xFFFF) << 32) |
                     vsUserData[baseSgpr];
    if (table < 0x1000000000ull || table >= 0x20000000000ull)
      continue;
    uint32_t byteOff = s.imm ? s.offset * 4 : 0;
    auto *vptr = reinterpret_cast<const uint32_t *>(table + byteOff);
    VBuffer v = decodeVBuffer(vptr);
    if (v.base >= 0x1000000000ull && v.base < 0x20000000000ull && v.stride &&
        v.stride <= 256 && v.numRecords && v.numRecords <= 0x100000) {
      if (g_trace)
        std::fprintf(stderr,
                     "[gcnres] VB sbase=sgpr%u table=%#lx off=%u -> base=%#lx "
                     "stride=%u nrec=%u dfmt=%u nfmt=%u\n",
                     baseSgpr, (unsigned long)table, byteOff,
                     (unsigned long)v.base, v.stride, v.numRecords, v.dfmt, v.nfmt);
      result.push_back(v);
    }
  }
  return result;
}

std::vector<TImage> trackTextures(const uint32_t *psCode, uint32_t maxDwords,
                                  const uint32_t *psUserData) {
  std::vector<TImage> result;
  if (!psCode || !psUserData)
    return result;
  auto insts = decode(psCode, maxDwords);

  struct Loaded { uint32_t sgpr, dwords; uint64_t address; };
  std::vector<Loaded> sloads;
  auto findLoad = [&](uint32_t sgpr, uint32_t dwords, uint64_t &address) {
    for (auto it = sloads.rbegin(); it != sloads.rend(); ++it) {
      if (sgpr >= it->sgpr && sgpr + dwords <= it->sgpr + it->dwords) {
        address = it->address + static_cast<uint64_t>(sgpr - it->sgpr) * 4;
        return guestRange(address, static_cast<uint64_t>(dwords) * 4);
      }
    }
    return false;
  };

  for (const auto &in : insts) {
    // Track descriptor-table loads as they execute so SGPR reuse resolves each
    // MIMG against the descriptor that was live at that instruction.
    if (in.enc == Enc::smrd) {
      Smrd s = decodeSmrd(in.raw[0]);
      if ((s.op == 0x02 || s.op == 0x03 || s.op == 0x04) && s.imm) {
        uint32_t baseSgpr = s.sbase * 2;
        uint32_t dwords = 1u << s.op;  // x4, x8, or x16
        if (baseSgpr + 1 < 16) {
          uint64_t table = (static_cast<uint64_t>(psUserData[baseSgpr + 1] & 0xFFFF) << 32) |
                           psUserData[baseSgpr];
          uint64_t byteOff = static_cast<uint64_t>(s.offset) * 4;
          if (guestRange(table, byteOff + static_cast<uint64_t>(dwords) * 4)) {
            uint64_t address = table + byteOff;
            sloads.erase(std::remove_if(sloads.begin(), sloads.end(), [&](const Loaded &ld) {
              return s.sdst < ld.sgpr + ld.dwords && ld.sgpr < s.sdst + dwords;
            }), sloads.end());
            sloads.push_back({s.sdst, dwords, address});
            if (g_trace)
              std::fprintf(stderr,
                           "[gcnres] s_load_dwordx%u sgpr%u <- table=%#lx off=%u -> %#lx\n",
                           dwords, s.sdst, (unsigned long)table, s.offset,
                           (unsigned long)address);
          }
        }
      }
      continue;
    }
    // MIMG image_sample: the T# / S# are referenced by SGPR indices (SRSRC /
    // SSAMP). A PS either passes them inline in its user-data SGPRs (Isaac does
    // no s_load) or loads them from the resource table via s_load_dwordx8
    // (3D shaders that sample several maps). Resolve inline first, else from the
    // s_load map, and preserve every MIMG binding. Invalid entries retain DA so
    // the renderer can bind a type-compatible fallback without compacting slots.
    if (in.enc != Enc::mimg)
      continue;
    uint32_t word1 = in.raw[1];
    uint32_t srsrc = ((word1 >> 16) & 0x1F) * 4;  // T# base SGPR
    uint64_t imageAddress = 0;
    bool loaded = findLoad(srsrc, 8, imageAddress);
    bool inline_ = !loaded && srsrc + 8 <= 16;
    TImage t;
    if (inline_)
      t = decodeTImage(&psUserData[srsrc]);
    else if (loaded)
      t = decodeTImage(reinterpret_cast<const uint32_t *>(imageAddress));
    uint32_t op = (in.raw[0] >> 18) & 0x7F;
    if (op >= 0x20) {
      uint32_t ssamp = ((word1 >> 21) & 0x1F) * 4;
      uint64_t samplerAddress = 0;
      bool loadedSampler = findLoad(ssamp, 4, samplerAddress);
      bool inlineSampler = !loadedSampler && ssamp + 4 <= 16;
      const uint32_t *sampler = loadedSampler
          ? reinterpret_cast<const uint32_t *>(samplerAddress)
          : inlineSampler ? &psUserData[ssamp] : nullptr;
      if (sampler) {
        std::memcpy(t.sampler, sampler, sizeof(t.sampler));
        t.samplerValid = true;
      }
    }
    t.arrayed = (in.raw[0] & 0x4000) != 0;  // MIMG DA
    if (t.valid) {
      // Empirical tiling census (DELTA_GPU_TILEHIST): tally tilingIdx of every
      // sampled texture so we can confirm which modes are linear (8/31) vs 1D
      // micro / 2D macro tiled, and dump it periodically.
      static const bool tileHist = std::getenv("DELTA_GPU_TILEHIST") != nullptr;
      if (tileHist) {
        static uint32_t hist[32] = {0};
        static uint64_t n = 0, pitchNe = 0;
        hist[t.tilingIdx & 31]++;
        if (t.pitch != t.width) pitchNe++;
        if ((++n % 4000) == 0) {
          std::fprintf(stderr, "[tilehist] n=%lu pitch!=width=%lu:", (unsigned long)n,
                       (unsigned long)pitchNe);
          for (int i = 0; i < 32; i++) if (hist[i])
            std::fprintf(stderr, " idx%d=%u", i, hist[i]);
          std::fprintf(stderr, "\n");
        }
      }
      if (g_trace)
        std::fprintf(stderr, "[gcnres] T# (%s sgpr%u) base=%#lx %ux%u pitch=%u "
                     "dfmt=%u nfmt=%u tiling=%u\n", inline_ ? "inline" : "s_load",
                     srsrc, (unsigned long)t.base, t.width, t.height, t.pitch,
                     t.dfmt, t.nfmt, t.tilingIdx);
    }
    result.push_back(t);
  }
  return result;
}

}  // namespace gpu::gcn
