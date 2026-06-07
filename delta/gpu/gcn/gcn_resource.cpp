/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN resource tracking. See gcn_resource.h.
 */

#include "gcn_resource.h"
#include "gcn_decode.h"

#include <cstdio>
#include <cstdlib>

namespace gpu::gcn {
namespace {
const bool g_trace = std::getenv("DELTA_GPU_TRACE") != nullptr;
}

VBuffer decodeVBuffer(const uint32_t *p) {
  // GCN V# (buffer resource descriptor), 4 dwords:
  //  [0]      base_address[31:0]
  //  [1] 0:43 base_address[47:32] in [11:0]; stride[13:0] in [29:16]
  //  [2]      num_records
  //  [3]      dst_sel/nfmt/dfmt/...: dfmt[18:15], nfmt[21:19]
  VBuffer v;
  v.base = (static_cast<uint64_t>(p[1] & 0xFFF) << 32) | p[0];
  v.stride = (p[1] >> 16) & 0x3FFF;
  v.numRecords = p[2];
  v.dfmt = (p[3] >> 15) & 0xF;
  v.nfmt = (p[3] >> 19) & 0x7;
  return v;
}

TImage decodeTImage(const uint32_t *p) {
  // GCN T# (image resource), 8 dwords:
  //  [0] base[39:8] (base = [0] << 8 with hi bits from [1])
  //  [1] base_hi[7:0]; min_lod[19:8]; dfmt[25:20]; nfmt[29:26]
  //  [2] width[13:0]; height[27:14]
  //  [3] ...; tiling_index[24:20]
  //  [4] depth[12:0]; pitch[26:13]
  TImage t;
  t.base = ((static_cast<uint64_t>(p[1] & 0xFF) << 32) | p[0]) << 8;
  t.dfmt = (p[1] >> 20) & 0x3F;
  t.nfmt = (p[1] >> 26) & 0xF;
  t.width = ((p[2] & 0x3FFF)) + 1;
  t.height = ((p[2] >> 14) & 0x3FFF) + 1;
  t.tilingIdx = (p[3] >> 20) & 0x1F;
  t.pitch = ((p[4] >> 13) & 0x3FFF) + 1;
  if (t.pitch < t.width) t.pitch = t.width;  // fall back to width if unset
  t.valid = t.base >= 0x1000000000ull && t.base < 0x20000000000ull &&
            t.width > 1 && t.width <= 8192 && t.height > 1 && t.height <= 8192;
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

  // First pass: track s_load_dwordx8 (SMRD op 3) that load a T# from the
  // extended-user-data resource table. The base pointer sits in the user-data
  // SGPR pair `sbase*2`; the T# is `off` dwords into that table. We record the
  // destination SGPR so a later MIMG referencing it can resolve the T# the
  // table holds. Isaac passes its single T# inline and emits no such load, so
  // this map stays empty for it (the inline path below is unchanged).
  struct Loaded { uint64_t taddr; };
  std::vector<std::pair<uint32_t, Loaded>> sloads;  // dstSgpr -> resolved T# address
  auto findLoad = [&](uint32_t sgpr) -> const Loaded * {
    for (auto it = sloads.rbegin(); it != sloads.rend(); ++it)
      if (it->first == sgpr) return &it->second;
    return nullptr;
  };
  for (const auto &in : insts) {
    if (in.enc != Enc::smrd)
      continue;
    Smrd s = decodeSmrd(in.raw[0]);
    if (s.op != 0x03)  // s_load_dwordx8 (an 8-dword T#)
      continue;
    uint32_t baseSgpr = s.sbase * 2;  // user_data index of the table pointer
    if (baseSgpr + 1 >= 16)
      continue;
    uint64_t table = (static_cast<uint64_t>(psUserData[baseSgpr + 1] & 0xFFFF) << 32) |
                     psUserData[baseSgpr];
    if (table < 0x1000000000ull || table >= 0x20000000000ull)
      continue;
    uint64_t taddr = table + (s.imm ? static_cast<uint64_t>(s.offset) * 4 : 0);
    sloads.push_back({s.sdst, {taddr}});
    if (g_trace)
      std::fprintf(stderr, "[gcnres] s_load_dwordx8 sgpr%u <- table=%#lx off=%u -> T# @%#lx\n",
                   s.sdst, (unsigned long)table, s.offset, (unsigned long)taddr);
  }

  for (const auto &in : insts) {
    // MIMG image_sample: the T# / S# are referenced by SGPR indices (SRSRC /
    // SSAMP). A PS either passes them inline in its user-data SGPRs (Isaac does
    // no s_load) or loads them from the resource table via s_load_dwordx8
    // (3D shaders that sample several maps). Resolve inline first, else from the
    // s_load map, and return every valid texture in MIMG (= binding) order.
    if (in.enc != Enc::mimg)
      continue;
    uint32_t word1 = in.raw[1];
    uint32_t srsrc = ((word1 >> 16) & 0x1F) * 4;  // T# base SGPR
    bool inline_ = srsrc + 8 <= 16;
    const Loaded *ld = inline_ ? nullptr : findLoad(srsrc);
    if (!inline_ && !ld)
      continue;
    TImage t = inline_ ? decodeTImage(&psUserData[srsrc])
                       : decodeTImage(reinterpret_cast<const uint32_t *>(ld->taddr));
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
      result.push_back(t);
    }
  }
  return result;
}

}  // namespace gpu::gcn
