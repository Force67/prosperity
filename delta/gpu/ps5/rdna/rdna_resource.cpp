/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) resource decode + per-draw texture tracking. See rdna_resource.h.
 */

#include "rdna_resource.h"

#include <algorithm>
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

bool inGuest(uint64_t a) { return a >= 0x10000ull && a < 0x1000000000000ull; }

int32_t signExt21(uint32_t v) { return static_cast<int32_t>(v << 11) >> 11; }

// Copy the `n`-dword descriptor at SGPR `sgpr` into `dst`. It is either inline in
// user data or one s_load from a user-data table pointer (`loads[sgpr]` gives the
// root pointer SGPR + byte offset).
bool resolveDesc(uint32_t sgpr, uint32_t n, const uint32_t* pud,
                 const std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>>& loads,
                 uint32_t* dst) {
  auto it = loads.find(sgpr);
  if (it != loads.end()) {
    const uint32_t sbase = it->second.first, off = it->second.second;
    if (sbase + 1 >= 16) return false;
    const uint64_t ptr = pud[sbase] | (static_cast<uint64_t>(pud[sbase + 1] & 0xFFFF) << 32);
    const uint64_t addr = ptr + off;
    if (!inGuest(addr)) return false;
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
  t.tiling_idx = sw_mode == 0 ? 8 : sw_mode;  // gfx10 detile TODO for tiled modes
  // TODO(ps5): map the gfx10.3 9-bit unified format ((d[1]>>20)&0x1FF) to a
  // VkFormat; defaulting to RGBA8_UNORM covers the common 2D/UI sampler.
  t.dfmt = 10;  // FMT_8_8_8_8
  t.nfmt = 0;   // UNORM
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
