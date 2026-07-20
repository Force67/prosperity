/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN resource tracking. See gcn_resource.h.
 */

#include "gcn_resource.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace gpu::gcn {
namespace {

const bool g_trace = std::getenv("DELTA_GPU_TRACE") != nullptr;

constexpr uint64_t kGuestLo = 0x1000000000ull;
constexpr uint64_t kGuestHi = 0x20000000000ull;

bool GuestRange(uint64_t address, uint64_t size) {
  return size && address >= kGuestLo && address < kGuestHi &&
         size <= kGuestHi - address;
}

// SMRD operand fields (GFX7).
struct Smrd {
  uint32_t op;
  uint32_t sdst;
  uint32_t sbase;  // SGPR pair index; actual base SGPR = sbase * 2
  uint32_t offset;
  bool imm;
};

Smrd DecodeSmrd(uint32_t w) {
  return {
      .op = (w >> 22) & 0x1F,
      .sdst = (w >> 15) & 0x7F,
      .sbase = (w >> 9) & 0x3F,
      .offset = w & 0xFF,
      .imm = ((w >> 8) & 1) != 0,
  };
}

uint64_t UserDataPointer(const uint32_t* user_data, uint32_t sgpr) {
  return (static_cast<uint64_t>(user_data[sgpr + 1] & 0xFFFF) << 32) |
         user_data[sgpr];
}

uint32_t NextPow2(uint32_t v) {
  v -= 1;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  return v + 1;
}

// Symbolic identity of the descriptor pair an MIMG instruction references:
// the T#/S# SGPR indices plus WHICH s_load last wrote each of them (0xFFFF =
// inline user data), plus the access-type bits that select a distinct Vulkan
// binding (arrayed / depth-compare / gather-lz). Packs into one dword-pair key.
uint64_t MimgDescriptorKey(uint32_t srsrc, uint32_t ssamp, uint32_t load_rsrc,
                           uint32_t load_samp, uint32_t flags) {
  return (static_cast<uint64_t>(srsrc) << 0) |
         (static_cast<uint64_t>(ssamp) << 8) |
         (static_cast<uint64_t>(load_rsrc & 0xFFFF) << 16) |
         (static_cast<uint64_t>(load_samp & 0xFFFF) << 32) |
         (static_cast<uint64_t>(flags) << 48);
}

// Concrete evaluation of a graphics stage's scalar register file. Seed s0..s15
// from the live user data, then execute the scalar loads / moves in program
// order, actually reading guest memory. Because a load's base pointer may be a
// value a PRIOR load wrote (not just direct user data), stepping the program
// resolves arbitrarily nested extended-user-data / SRT descriptor chains:
// user-data SGPR -> s_load a pointer -> s_load the descriptor through that
// pointer -> ... Callers step every instruction and read the descriptor SGPRs
// (T#/S#/V#) at the instruction that consumes them (MIMG / s_buffer_load), so
// SGPR reuse resolves each consumer against the state live at its own point.
//
// Register file: 128 SGPRs; a T# SBASE (5 bits * 4) reaches s124, +8 = s132, so
// size for the descriptor tail. `known` marks which dwords hold a resolved
// value (seeded user data, or a value read from guest memory).
struct ScalarEval {
  static constexpr uint32_t kRegs = 136;
  uint32_t sgpr[kRegs] = {};
  bool known[kRegs] = {};
  bool trace = false;

  explicit ScalarEval(const uint32_t* user_data) {
    for (uint32_t i = 0; i < 16; i++) {
      sgpr[i] = user_data[i];
      known[i] = true;
    }
    static const bool eud_trace = std::getenv("DELTA_GPU_EUDTRACE") != nullptr;
    trace = eud_trace;
  }

  uint64_t Ptr(uint32_t s) const {  // 48-bit descriptor-table pointer pair
    return (static_cast<uint64_t>(sgpr[s + 1] & 0xFFFF) << 32) | sgpr[s];
  }
  bool AllKnown(uint32_t s, uint32_t n) const {
    if (s + n > kRegs) return false;
    for (uint32_t i = 0; i < n; i++)
      if (!known[s + i]) return false;
    return true;
  }
  void Set(uint32_t s, uint32_t v) {
    if (s < kRegs) { sgpr[s] = v; known[s] = true; }
  }
  void Clear(uint32_t s) {
    if (s < kRegs) known[s] = false;
  }

  // Advance the register file across one instruction. Only scalar moves and
  // pointer-relative scalar loads (the descriptor-chain ops) mutate it; every
  // other encoding leaves it unchanged. s_buffer_load (op >= 0x08) reads a
  // cbuffer through a V# and is a consumer, not a pointer op, so it is ignored
  // here.
  void Step(const Inst& inst) {
    if (inst.enc == Enc::kSop1) {
      const uint32_t w = inst.raw[0];
      const uint32_t sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF;
      if (inst.opcode == 0x03) {  // s_mov_b32: stage a pointer via a scalar move
        if (ssrc0 < kRegs && known[ssrc0]) Set(sdst, sgpr[ssrc0]);
        else Clear(sdst);
      } else if (inst.opcode == 0x04) {  // s_mov_b64
        if (ssrc0 + 1 < kRegs && known[ssrc0] && known[ssrc0 + 1]) {
          Set(sdst, sgpr[ssrc0]);
          Set(sdst + 1, sgpr[ssrc0 + 1]);
        } else { Clear(sdst); Clear(sdst + 1); }
      }
      return;
    }
    if (inst.enc != Enc::kSmrd) return;
    const Smrd s = DecodeSmrd(inst.raw[0]);
    if (s.op > 0x04) return;  // s_load_dword..dwordx16 only
    const uint32_t dwords = 1u << s.op;
    const uint32_t base = s.sbase * 2;
    // A load rewrites its destination SGPRs even if it cannot be resolved;
    // invalidate first so a later consumer does not read stale values.
    for (uint32_t i = 0; i < dwords; i++) Clear(s.sdst + i);
    if (!AllKnown(base, 2)) return;
    const uint64_t table = Ptr(base);
    uint64_t byte_off = 0;
    if (s.imm) {
      byte_off = static_cast<uint64_t>(s.offset) * 4;  // dword offset field
    } else if (s.offset < kRegs && known[s.offset]) {
      byte_off = sgpr[s.offset];  // SOFFSET SGPR carries a byte offset
    } else {
      return;
    }
    if (!GuestRange(table, byte_off + static_cast<uint64_t>(dwords) * 4)) return;
    const uint64_t address = table + byte_off;
    const uint32_t* src = reinterpret_cast<const uint32_t*>(address);
    for (uint32_t i = 0; i < dwords; i++) Set(s.sdst + i, src[i]);
    if (trace)
      std::fprintf(stderr, "[eud] s_load x%u s%u <- [s%u=%#lx + %#lx] = %#lx\n",
                   dwords, s.sdst, base, static_cast<unsigned long>(table),
                   static_cast<unsigned long>(byte_off),
                   static_cast<unsigned long>(address));
  }
};

// Per-program analysis reused across draws: the MIMG binding plan plus the
// subset of instructions the scalar walk actually consumes (descriptor-chain
// s_movs, SMRD loads, MIMG uses). TrackTextures/ResolveCbuffers run once per
// draw on shaders that are mostly VALU code, so stepping only this subset --
// and planning bindings once instead of per draw -- removes the bulk of the
// per-draw analysis cost. Keyed by the Program object; the cached shared_ptr
// pins the object so the pointer cannot be reused while the entry lives. A
// shader rewrite yields a new Program from CachedProgram -> a new entry.
struct ScalarPassInfo {
  MimgBindingPlan plan;
  std::vector<Inst> insts;  // program-order subset relevant to ScalarEval users
};

const ScalarPassInfo& CachedScalarInfo(
    const std::shared_ptr<const Program>& program) {
  struct Entry {
    std::shared_ptr<const Program> pin;
    ScalarPassInfo info;
  };
  static std::unordered_map<const Program*, Entry> cache;
  auto it = cache.find(program.get());
  if (it != cache.end()) return it->second.info;
  if (cache.size() > 512) cache.clear();  // unbounded-growth backstop
  Entry e;
  e.pin = program;
  e.info.plan = PlanMimgBindings(*program);
  for (const Inst& inst : *program) {
    const bool scalar_move =
        inst.enc == Enc::kSop1 && (inst.opcode == 0x03 || inst.opcode == 0x04);
    if (scalar_move || inst.enc == Enc::kSmrd || inst.enc == Enc::kMimg)
      e.info.insts.push_back(inst);
  }
  return cache.emplace(program.get(), std::move(e)).first->second.info;
}

}  // namespace

MimgBindingPlan PlanMimgBindings(const Program& program) {
  MimgBindingPlan plan;
  // Track, per SGPR range, the index of the last SMRD instruction covering it.
  struct Load {
    uint32_t sgpr, dwords, index;
  };
  std::vector<Load> loads;
  const auto covering_load = [&](uint32_t sgpr, uint32_t dwords) -> uint32_t {
    for (auto it = loads.rbegin(); it != loads.rend(); ++it)
      if (sgpr >= it->sgpr && sgpr + dwords <= it->sgpr + it->dwords)
        return it->index;
    return 0xFFFF;  // inline user data (no covering load)
  };

  std::unordered_map<uint64_t, uint32_t> binding_of;
  uint32_t inst_index = 0;
  for (const Inst& inst : program) {
    const uint32_t idx = inst_index++;
    if (inst.enc == Enc::kSmrd) {
      const Smrd s = DecodeSmrd(inst.raw[0]);
      if (s.op <= 0x04) {  // s_load_dword..x16 can rewrite descriptor SGPRs
        const uint32_t dwords = 1u << s.op;
        loads.erase(std::remove_if(loads.begin(), loads.end(),
                                   [&](const Load& ld) {
                                     return s.sdst < ld.sgpr + ld.dwords &&
                                            ld.sgpr < s.sdst + dwords;
                                   }),
                    loads.end());
        loads.push_back({s.sdst, dwords, idx});
      }
      continue;
    }
    if (inst.enc != Enc::kMimg) continue;
    const uint32_t w0 = inst.raw[0], w1 = inst.raw[1];
    const uint32_t op = (w0 >> 18) & 0x7F;
    const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
    const bool sampling = op >= 0x20;
    const uint32_t ssamp = sampling ? ((w1 >> 21) & 0x1F) * 4 : 0xFF;
    const uint32_t flags = (((w0 >> 14) & 1) << 0) |               // DA
                           (((op == 0x28 || op == 0x2f) ? 1 : 0) << 1) |  // dref
                           ((op == 0x47 ? 1 : 0) << 2);            // gather4_lz
    const uint64_t key = MimgDescriptorKey(
        srsrc, ssamp, covering_load(srsrc, 8),
        sampling ? covering_load(ssamp, 4) : 0xFFFE, flags);
    const auto [it, inserted] =
        binding_of.emplace(key, static_cast<uint32_t>(plan.binding_srsrc.size()));
    if (inserted) plan.binding_srsrc.push_back(srsrc);
    plan.binding_by_pc[inst.pc] = it->second;
  }
  return plan;
}

VBuffer DecodeVBuffer(const uint32_t* p) {
  // GCN V# (buffer resource descriptor), 4 dwords:
  //  [0]  base_address[31:0]
  //  [1]  base_address[47:32] in [15:0]; stride[13:0] in [29:16]
  //  [2]  num_records
  //  [3]  dst_sel/nfmt/dfmt/...: nfmt[14:12], dfmt[18:15]
  return {
      .base = (static_cast<uint64_t>(p[1] & 0xFFFF) << 32) | p[0],
      .stride = (p[1] >> 16) & 0x3FFF,
      .num_records = p[2],
      .dfmt = (p[3] >> 15) & 0xF,
      .nfmt = (p[3] >> 12) & 0x7,
  };
}

TImage DecodeTImage(const uint32_t* p) {
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
  t.min_lod = (p[1] >> 8) & 0xFFF;
  t.dfmt = (p[1] >> 20) & 0x3F;
  t.nfmt = (p[1] >> 26) & 0xF;
  t.width = (p[2] & 0x3FFF) + 1;
  t.height = ((p[2] >> 14) & 0x3FFF) + 1;
  t.base_mip = (p[3] >> 12) & 0xF;
  const uint32_t last_mip = (p[3] >> 16) & 0xF;
  t.mip_levels = last_mip + 1;
  t.view_mips = last_mip >= t.base_mip ? last_mip - t.base_mip + 1 : 0;
  t.tiling_idx = (p[3] >> 20) & 0x1F;
  t.pow2_pad = ((p[3] >> 25) & 1) != 0;
  t.type = p[3] >> 28;
  t.pitch = ((p[4] >> 13) & 0x3FFF) + 1;
  if (t.pitch < t.width) t.pitch = t.width;  // fall back to width if unset
  if (t.type == 13) {  // SQ_RSRC_IMG_2D_ARRAY
    t.layers = (p[4] & 0x1FFF) + 1;
    if (t.pow2_pad) t.layers = NextPow2(t.layers);
    t.base_array = p[5] & 0x1FFF;
    t.view_layers = 0;
    const uint32_t last_array = (p[5] >> 13) & 0x1FFF;
    if (t.base_array < t.layers && last_array >= t.base_array)
      t.view_layers = std::min(last_array, t.layers - 1) - t.base_array + 1;
  }

  const bool supported_type = t.type == 9 || t.type == 13;
  const bool valid_view =
      t.type != 13 || (t.base_array < t.layers && t.view_layers > 0);
  uint32_t max_levels = 1;
  for (uint32_t extent = std::max(t.width, t.height); extent > 1; extent >>= 1)
    max_levels++;
  const bool valid_mips = t.view_mips && t.mip_levels <= max_levels;
  t.valid = GuestRange(t.base, 1) && supported_type && t.width <= 8192 &&
            t.height <= 8192 && t.layers <= 8192 && valid_view && valid_mips;
  return t;
}

std::vector<VBuffer> TrackVertexBuffers(const Program& fetch_program,
                                        const uint32_t* vs_user_data) {
  std::vector<VBuffer> result;
  if (!vs_user_data) return result;

  // The fetch shader loads each attribute's V# with an s_load_dwordx4 whose
  // SBASE is a user-SGPR pair holding the vertex-buffer-table pointer, at byte
  // offset (offset*4 for imm). Recover the table pointer from the user data
  // and read the V# there.
  for (const Inst& inst : fetch_program) {
    if (inst.enc != Enc::kSmrd) continue;
    const Smrd s = DecodeSmrd(inst.raw[0]);
    if (s.op != 0x02) continue;  // s_load_dwordx4 (a 4-dword V#)
    const uint32_t base_sgpr = s.sbase * 2;  // user_data index of the table ptr
    if (base_sgpr + 1 >= 16) continue;
    const uint64_t table = UserDataPointer(vs_user_data, base_sgpr);
    if (!GuestRange(table, 16)) continue;
    const uint32_t byte_off = s.imm ? s.offset * 4 : 0;
    const VBuffer v =
        DecodeVBuffer(reinterpret_cast<const uint32_t*>(table + byte_off));
    if (v.base >= kGuestLo && v.base < kGuestHi && v.stride &&
        v.stride <= 256 && v.num_records && v.num_records <= 0x100000) {
      if (g_trace)
        std::fprintf(stderr,
                     "[gcnres] VB sbase=sgpr%u table=%#lx off=%u -> base=%#lx "
                     "stride=%u nrec=%u dfmt=%u nfmt=%u\n",
                     base_sgpr, static_cast<unsigned long>(table), byte_off,
                     static_cast<unsigned long>(v.base), v.stride,
                     v.num_records, v.dfmt, v.nfmt);
      result.push_back(v);
    }
  }
  return result;
}

std::vector<TImage> TrackTextures(
    const std::shared_ptr<const Program>& ps_program,
    const uint32_t* ps_user_data) {
  std::vector<TImage> result;
  if (!ps_program || !ps_user_data) return result;

  // Bindings come from the shared plan (one per unique descriptor identity),
  // so this list pairs 1:1 with the recompiled shader's set-0 samplers.
  const ScalarPassInfo& cached = CachedScalarInfo(ps_program);
  const MimgBindingPlan& plan = cached.plan;

  // Step the scalar register file across the program; at each MIMG read the
  // live T#/S# straight out of the resolved SGPRs. Inline user data, a single
  // indirect load, and nested EUD chains all land here identically.
  ScalarEval eval(ps_user_data);

  for (const Inst& inst : cached.insts) {
    eval.Step(inst);
    if (inst.enc != Enc::kMimg) continue;
    const auto plan_it = plan.binding_by_pc.find(inst.pc);
    if (plan_it == plan.binding_by_pc.end()) continue;  // unreachable
    const uint32_t binding = plan_it->second;
    const uint32_t word1 = inst.raw[1];
    const uint32_t srsrc = ((word1 >> 16) & 0x1F) * 4;  // T# base SGPR
    const uint32_t op = (inst.raw[0] >> 18) & 0x7F;

    // Resolve the sampler for sampling ops (used both for new bindings and to
    // backfill a binding first seen through a non-sampling op like resinfo).
    uint32_t sampler[4] = {};
    bool sampler_ok = false;
    if (op >= 0x20) {
      const uint32_t ssamp = ((word1 >> 21) & 0x1F) * 4;
      if (eval.AllKnown(ssamp, 4)) {
        std::memcpy(sampler, &eval.sgpr[ssamp], sizeof(sampler));
        sampler_ok = true;
      }
    }

    if (binding < result.size()) {  // repeat use of an existing binding
      TImage& entry = result[binding];
      if (sampler_ok && !entry.sampler_valid) {
        std::memcpy(entry.sampler, sampler, sizeof(entry.sampler));
        entry.sampler_valid = true;
      }
      continue;
    }

    TImage t;
    const bool image_ok = eval.AllKnown(srsrc, 8);
    if (image_ok) t = DecodeTImage(&eval.sgpr[srsrc]);
    if (eval.trace)
      std::fprintf(stderr,
                   "[eud] MIMG pc=%#x bind=%u srsrc=s%u known=%d base=%#lx "
                   "%ux%u valid=%d\n",
                   inst.pc, binding, srsrc, image_ok,
                   static_cast<unsigned long>(t.base), t.width, t.height,
                   t.valid);
    if (sampler_ok) {
      std::memcpy(t.sampler, sampler, sizeof(t.sampler));
      t.sampler_valid = true;
    }
    t.arrayed = (inst.raw[0] & 0x4000) != 0;  // MIMG DA
    t.force_lod_zero = op == 0x47;            // IMAGE_GATHER4_LZ
    t.depth_compare = op == 0x28 || op == 0x2f;
    if (t.valid) {
      // Empirical tiling census (DELTA_GPU_TILEHIST): tally tiling_idx of
      // every sampled texture to confirm which modes are linear vs tiled.
      static const bool tile_hist = std::getenv("DELTA_GPU_TILEHIST") != nullptr;
      if (tile_hist) {
        static uint32_t hist[32] = {0};
        static uint64_t n = 0, pitch_ne = 0;
        hist[t.tiling_idx & 31]++;
        if (t.pitch != t.width) pitch_ne++;
        if ((++n % 4000) == 0) {
          std::fprintf(stderr, "[tilehist] n=%lu pitch!=width=%lu:",
                       static_cast<unsigned long>(n),
                       static_cast<unsigned long>(pitch_ne));
          for (int i = 0; i < 32; i++)
            if (hist[i]) std::fprintf(stderr, " idx%d=%u", i, hist[i]);
          std::fprintf(stderr, "\n");
        }
      }
      if (g_trace)
        std::fprintf(stderr,
                     "[gcnres] T# (sgpr%u) base=%#lx %ux%u pitch=%u "
                     "dfmt=%u nfmt=%u tiling=%u\n",
                     srsrc, static_cast<unsigned long>(t.base), t.width,
                     t.height, t.pitch, t.dfmt, t.nfmt, t.tiling_idx);
    }
    result.push_back(t);
  }
  return result;
}

std::unordered_map<uint32_t, VBuffer> ResolveCbuffers(
    const std::shared_ptr<const Program>& program, const uint32_t* user_data) {
  std::unordered_map<uint32_t, VBuffer> result;
  if (!program || !user_data) return result;

  // Mirror TrackTextures: step the scalar register file, and at each
  // s_buffer_load read the live 4-dword V# out of the resolved SGPRs. FOX
  // passes cbuffer descriptors through extended user data too (s_load the V#
  // through an EUD pointer, then s_buffer_load through it), so reading the V#
  // straight from user data yields base=0. The recompiler assigns one binding
  // per base SGPR (PlanCbufs), so key by base SGPR and keep the first
  // resolvable V# seen for it.
  ScalarEval eval(user_data);
  for (const Inst& inst : CachedScalarInfo(program).insts) {
    eval.Step(inst);
    if (inst.enc != Enc::kSmrd) continue;
    const Smrd s = DecodeSmrd(inst.raw[0]);
    if (s.op < 0x08 || s.op > 0x0c) continue;  // s_buffer_load_dword{,x2..x16}
    const uint32_t base = s.sbase * 2;          // 4-dword V# base SGPR
    if (result.count(base) || !eval.AllKnown(base, 4)) continue;
    const VBuffer v = DecodeVBuffer(&eval.sgpr[base]);
    result.emplace(base, v);
    if (eval.trace)
      std::fprintf(stderr,
                   "[eud] cbuf s%u -> base=%#lx stride=%u nrec=%u\n", base,
                   static_cast<unsigned long>(v.base), v.stride, v.num_records);
  }
  return result;
}

}  // namespace gpu::gcn
