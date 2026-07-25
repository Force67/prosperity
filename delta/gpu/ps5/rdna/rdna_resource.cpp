/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) resource decode + per-draw texture tracking. See rdna_resource.h.
 */

#include "rdna_resource.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>

namespace gpu::rdna {
namespace {

using gpu::gcn::Enc;
using gpu::gcn::Inst;
using gpu::gcn::MimgBindingPlan;
using gpu::gcn::Program;
using gpu::gcn::TImage;

bool inGuest(uint64_t a) { return a >= 0x10000ull && a < 0x1000000000000ull; }

// inGuest()'s bound spans essentially the whole 48-bit VA, so a T#/S# root
// pointer recovered by reinterpreting raw SGPR bits can pass it while still
// pointing at memory the guest never mapped -- an inactive/uninitialized
// descriptor slot does exactly this for some early draws and segfaults the
// host. Probe with mincore (same technique as kern/crash.cpp) before reading.
bool mapped(uint64_t va, uint64_t bytes) {
  const long pg = sysconf(_SC_PAGESIZE);
  const uint64_t start = va & ~static_cast<uint64_t>(pg - 1);
  const uint64_t end = (va + bytes + pg - 1) & ~static_cast<uint64_t>(pg - 1);
  for (uint64_t p = start; p < end; p += static_cast<uint64_t>(pg)) {
    unsigned char vec = 0;
    if (mincore(reinterpret_cast<void *>(p), 1, &vec) != 0) return false;
  }
  return true;
}

int32_t signExt21(uint32_t v) { return static_cast<int32_t>(v << 11) >> 11; }

// gfx10.3 T#s carry a 9-bit unified format enum (word1 [28:20]). Values 1..77
// are the buffer format table (GPU Shader Core ISA spec 4.x "Buffer Format
// Conversions"); 130+ are image-only (SRGB, packed 16-bit, BCn). Map the
// sampled subset onto the GCN (dfmt,nfmt) pairs the shared renderer's
// guestTextureFormat() understands; unmapped formats yield (0,0) so the upload
// declines (white fallback) instead of misreading texels.
void gfx10ImgFormat(uint32_t gfmt, uint32_t &dfmt, uint32_t &nfmt) {
  switch (gfmt) {
    case 1:   dfmt = 1;  nfmt = 0; break;  // 8_UNORM
    case 2:   dfmt = 1;  nfmt = 1; break;  // 8_SNORM
    case 5:   dfmt = 1;  nfmt = 4; break;  // 8_UINT
    case 6:   dfmt = 1;  nfmt = 5; break;  // 8_SINT
    case 7:   dfmt = 2;  nfmt = 0; break;  // 16_UNORM
    case 11:  dfmt = 2;  nfmt = 4; break;  // 16_UINT
    case 13:  dfmt = 2;  nfmt = 7; break;  // 16_FLOAT
    case 14:  dfmt = 3;  nfmt = 0; break;  // 8_8_UNORM
    case 18:  dfmt = 3;  nfmt = 4; break;  // 8_8_UINT
    case 20:  dfmt = 4;  nfmt = 4; break;  // 32_UINT
    case 21:  dfmt = 4;  nfmt = 5; break;  // 32_SINT
    case 22:  dfmt = 4;  nfmt = 7; break;  // 32_FLOAT (also depth-resolve key)
    case 23:  dfmt = 5;  nfmt = 0; break;  // 16_16_UNORM
    case 27:  dfmt = 5;  nfmt = 4; break;  // 16_16_UINT
    case 29:  dfmt = 5;  nfmt = 7; break;  // 16_16_FLOAT
    case 36:  dfmt = 6;  nfmt = 7; break;  // 11_11_10_FLOAT
    case 44:  dfmt = 9;  nfmt = 0; break;  // 2_10_10_10_UNORM
    case 50:  dfmt = 8;  nfmt = 0; break;  // 10_10_10_2_UNORM
    case 56:  dfmt = 10; nfmt = 0; break;  // 8_8_8_8_UNORM
    case 62:  dfmt = 11; nfmt = 4; break;  // 32_32_UINT
    case 64:  dfmt = 11; nfmt = 7; break;  // 32_32_FLOAT
    case 71:  dfmt = 12; nfmt = 7; break;  // 16_16_16_16_FLOAT
    case 74:  dfmt = 13; nfmt = 7; break;  // 32_32_32_FLOAT
    case 77:  dfmt = 14; nfmt = 7; break;  // 32_32_32_32_FLOAT
    case 57:  dfmt = 10; nfmt = 1; break;  // 8_8_8_8_SNORM
    case 60:  dfmt = 10; nfmt = 4; break;  // 8_8_8_8_UINT
    case 130: dfmt = 10; nfmt = 9; break;  // 8_8_8_8_SRGB
    case 169: dfmt = 35; nfmt = 0; break;  // BC1
    case 170: dfmt = 35; nfmt = 9; break;
    case 171: dfmt = 36; nfmt = 0; break;  // BC2
    case 172: dfmt = 36; nfmt = 9; break;
    case 173: dfmt = 37; nfmt = 0; break;  // BC3
    case 174: dfmt = 37; nfmt = 9; break;
    case 175: dfmt = 38; nfmt = 0; break;  // BC4
    case 176: dfmt = 38; nfmt = 1; break;
    case 177: dfmt = 39; nfmt = 0; break;  // BC5
    case 178: dfmt = 39; nfmt = 1; break;
    case 181: dfmt = 41; nfmt = 0; break;  // BC7
    case 182: dfmt = 41; nfmt = 9; break;
    default:  dfmt = 0;  nfmt = 0; break;
  }
}

// Copy the `n`-dword descriptor at SGPR `sgpr` into `dst`. It is either inline in
// user data or one s_load from a user-data table pointer (`loads[sgpr]` gives the
// root pointer SGPR + byte offset).
bool resolveDesc(uint32_t sgpr, uint32_t n, const uint32_t* pud,
                 const std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>>& loads,
                 uint32_t* dst) {
  auto it = loads.find(sgpr);
  if (it != loads.end()) {
    const uint32_t sbase = it->second.first, off = it->second.second;
    // The user-data block is 32 dwords, not 16: the final grading pass reaches
    // its textures through a pointer pair at s[28:29], and bounding this at 16
    // dropped those samplers to the 1x1 white default. The dereference is
    // self-validating (the address must land in guest memory), unlike reading
    // the user-data image directly.
    if (sbase + 1 >= 32) return false;
    const uint64_t ptr = pud[sbase] | (static_cast<uint64_t>(pud[sbase + 1] & 0xFFFF) << 32);
    const uint64_t addr = ptr + off;
    if (!inGuest(addr) || !mapped(addr, n * 4)) return false;
    std::memcpy(dst, reinterpret_cast<const void*>(addr), n * 4);
    return true;
  }
  if (sgpr + n <= 16) {
    std::memcpy(dst, &pud[sgpr], n * 4);
    return true;
  }
  return false;
}

}  // namespace

MimgBindingPlan RdnaPlanMimg(const Program& program) {
  MimgBindingPlan plan;
  struct Load { uint32_t sgpr, dwords, index; };
  std::vector<Load> loads;
  auto coveringLoad = [&](uint32_t sgpr, uint32_t dwords) -> uint32_t {
    for (auto it = loads.rbegin(); it != loads.rend(); ++it)
      if (sgpr >= it->sgpr && sgpr + dwords <= it->sgpr + it->dwords)
        return it->index;
    return 0xFFFF;  // inline user data (no covering load)
  };
  std::unordered_map<uint64_t, uint32_t> binding_of;
  uint32_t idx = 0;
  for (const Inst& inst : program) {
    const uint32_t i = idx++;
    if (inst.enc == Enc::kSmrd) {
      if (inst.opcode <= 0x04) {
        const uint32_t sdst = (inst.raw[0] >> 6) & 0x7F;
        const uint32_t dwords = SmemLoadCount(inst.opcode);
        loads.erase(std::remove_if(loads.begin(), loads.end(), [&](const Load& ld) {
          return sdst < ld.sgpr + ld.dwords && ld.sgpr < sdst + dwords;
        }), loads.end());
        loads.push_back({sdst, dwords, i});
      }
      continue;
    }
    if (inst.enc != Enc::kMimg) continue;
    const uint32_t w0 = inst.raw[0], w1 = inst.raw[1];
    const uint32_t op = (w0 >> 18) & 0x7F;
    const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
    const bool sampling = op >= 0x20;
    const bool storage = op == 0x08 || op == 0x09;
    const uint32_t ssamp = sampling ? ((w1 >> 21) & 0x1F) * 4 : 0xFFu;
    const uint32_t flags = ((w0 >> 14) & 1) |
                           ((op == 0x28 || op == 0x2f ? 1u : 0u) << 1) |
                           ((op == 0x47 ? 1u : 0u) << 2) |
                           (static_cast<uint32_t>(storage) << 3);
    const uint64_t key =
        static_cast<uint64_t>(srsrc) | (static_cast<uint64_t>(ssamp) << 8) |
        (static_cast<uint64_t>(coveringLoad(srsrc, 8)) << 16) |
        (static_cast<uint64_t>(sampling ? coveringLoad(ssamp, 4) : 0xFFFEu) << 32) |
        (static_cast<uint64_t>(flags) << 48);
    auto [it, inserted] = binding_of.emplace(
        key, static_cast<uint32_t>(plan.binding_srsrc.size()));
    if (inserted) {
      plan.binding_srsrc.push_back(srsrc);
      plan.binding_storage.push_back(storage);
    }
    plan.binding_by_pc[inst.pc] = it->second;
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
  const uint32_t depth = d[4] & 0xFFFF;
  t.base_array = (d[4] >> 16) & 0xFFFF;
  const uint32_t max_mip = (d[5] >> 4) & 0xF;

  t.pitch = t.width;
  t.arrayed = t.type == 12 || t.type == 13;  // 1D/2D array
  const bool volumetric = t.type == 10 || t.type == 11;  // 3D / cube
  t.layers = (t.arrayed || volumetric) ? depth + 1 : 1;
  t.view_layers = t.arrayed ? std::max<uint32_t>(depth + 1 - t.base_array, 1) : 1;
  t.mip_levels = max_mip + 1;
  t.view_mips = std::max<uint32_t>(last_level + 1 - t.base_mip, 1);
  const uint32_t gfmt = (d[1] >> 20) & 0x1FF;
  gfx10ImgFormat(gfmt, t.dfmt, t.nfmt);
  // gfx10 swizzle mode 0 = SW_LINEAR -> the renderer's linear index. The
  // "standard" modes (256 B / 4 KiB / 64 KiB, ids 1/5/9) map onto the gfx10
  // detiler's own id range. Everything else (Z/D/R, the _X pipe-XOR and _T
  // variants) has no detiler yet, so it is shifted past the valid range:
  // BuildTextureLayout32 rejects it and the draw gets the white fallback
  // instead of scrambled texels.
  switch (sw_mode) {
    case 0:  t.tiling_idx = 8; break;
    case 1:  t.tiling_idx = 0x50; break;
    case 5:  t.tiling_idx = 0x51; break;
    case 9:  t.tiling_idx = 0x52; break;
    default: t.tiling_idx = 0x40 + sw_mode; break;
  }
  if (sw_mode == 0 && t.dfmt && t.dfmt < 35) {
    // gfx10 linear surfaces align each row to 256 bytes.
    const uint32_t eb = t.dfmt == 12 ? 8 : 4;
    const uint32_t pa = 256 / eb;
    t.pitch = (t.width + pa - 1) & ~(pa - 1);
  }
  static const bool trace = std::getenv("DELTA_AGC_TRACE") != nullptr;
  if (trace) {
    static uint32_t seen[32], nSeen = 0;
    const uint32_t key = (gfmt << 8) | sw_mode;
    bool isNew = true;
    for (uint32_t i = 0; i < nSeen; i++)
      if (seen[i] == key) { isNew = false; break; }
    if (isNew && nSeen < 32) {
      seen[nSeen++] = key;
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
  if (t.tiling_idx >= 0x50 && t.tiling_idx < 0x53 && t.mip_levels > 1) {
    t.mip_levels = 1;
    t.view_mips = 1;
    t.base_mip = 0;
    t.min_lod = 0;
    t.force_lod_zero = true;
  }
  t.valid = t.base != 0;
  return t;
}

std::vector<TImage> TrackTextures(const uint32_t* ps_code, const uint32_t* pud) {
  std::vector<TImage> out;
  if (!ps_code || !pud || !inGuest(reinterpret_cast<uint64_t>(ps_code))) return out;
  const Program prog = DecodeShader(ps_code, 4096);
  const MimgBindingPlan plan = RdnaPlanMimg(prog);
  out.resize(plan.binding_srsrc.size());
  std::vector<bool> filled(out.size(), false);

  std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> loads;  // sdst->{root,off}
  for (const Inst& in : prog) {
    if (in.enc == Enc::kSmrd && in.opcode <= 0x04) {
      const uint32_t sdst = (in.raw[0] >> 6) & 0x7F;
      const uint32_t sbase = (in.raw[0] & 0x3F) * 2;
      const int32_t off = signExt21(in.raw[1] & 0x1FFFFF);
      loads[sdst] = {sbase, off < 0 ? 0u : static_cast<uint32_t>(off)};
      continue;
    }
    if (in.enc != Enc::kMimg) continue;
    auto it = plan.binding_by_pc.find(in.pc);
    if (it == plan.binding_by_pc.end() || filled[it->second]) continue;
    const uint32_t b = it->second;
    const uint32_t w0 = in.raw[0], w1 = in.raw[1], op = (w0 >> 18) & 0x7F;
    uint32_t desc[8];
    if (resolveDesc(((w1 >> 16) & 0x1F) * 4, 8, pud, loads, desc)) {
      out[b] = DecodeTImage(desc);
      out[b].arrayed = (w0 >> 14) & 1;
      out[b].depth_compare = op == 0x28 || op == 0x2f;
      out[b].force_lod_zero = op == 0x47;
      out[b].storage = op == 0x08 || op == 0x09;
      if (op >= 0x20 && resolveDesc(((w1 >> 21) & 0x1F) * 4, 4, pud, loads, desc)) {
        std::memcpy(out[b].sampler, desc, sizeof(out[b].sampler));
        out[b].sampler_valid = true;
      }
    }
    filled[b] = true;
  }
  return out;
}

}  // namespace gpu::rdna
