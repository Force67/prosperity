/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * RDNA2 (gfx10.3) -> SPIR-V translator. Reuses the shared gpu::gcn SPIR-V
 * backend (the register-file model in Translator, the scalar/vector ALU
 * emitters, exports, and constant-buffer plumbing); only the RDNA2-specific
 * per-instruction field decode + opcode remap and the SMEM constant-buffer path
 * live here. Control-flow lowering is a self-contained copy of the GFX7
 * while/switch state machine (it dispatches through the RDNA2 emitter), so the
 * PS4 backend is untouched.
 */

#include "gpu/ps5/rdna/rdna_translate.h"

#ifndef DELTA_HAVE_SPIRV_BACKEND
namespace gpu::rdna {
gpu::gcn::Recompiled Recompile(const uint32_t*,
                               const uint32_t*,
                               const uint32_t*,
                               const uint32_t*,
                               uint32_t,
                               bool,
                               uint32_t,
                               uint32_t) {
  return {};
}
}  // namespace gpu::rdna
#else

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gpu/guest_memory.h"
#include "gpu/ps4/gcn/spirv/spv_post.h"
#include "gpu/ps4/gcn/spirv/translator.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/rdna/rdna_resource.h"

namespace gpu::rdna {
namespace {

using gpu::gcn::Enc;
using gpu::gcn::Id;
using gpu::gcn::Inst;
using gpu::gcn::kMaxCbufBindings;
using gpu::gcn::Program;
using gpu::gcn::Recompiled;
using gpu::gcn::ShaderCbuf;
using gpu::gcn::StageContext;
using gpu::gcn::Translator;

// Bring-up debug: log decoded exports + recovered vertex fetches so the PS5
// pipeline can be verified against the guest shader. Honors either the shader
// trace knob (DELTA_GPU_SHTRACE) or the AGC command-stream trace
// (DELTA_AGC_TRACE).
bool ShDbg() {
  static const bool on = std::getenv("DELTA_GPU_SHTRACE") != nullptr ||
                         std::getenv("DELTA_AGC_TRACE") != nullptr;
  return on;
}

// A buffer_load_format is a real PER-VERTEX fetch when it is IDXEN and its
// srsrc V# is TABLE-CHAINED (loaded from a user-data descriptor table,
// `chained`) -- regardless of which VGPR indexes it. NGG streams index
// per-vertex data by v0 or a computed register (the composite VS uses v3, the
// sprite VS's uv/param streams use v4); the Vulkan vertex-input stage supplies
// per-vertex data by the draw index either way. A NON-chained IDXEN load reads
// an inline user-data descriptor (e.g. a 2D VS's ortho matrix): that is a
// CONSTANT bound as a UBO and read in the shader body, NOT lifted to a vertex
// input.
bool BufLoadIsVertexFetch(const Inst& in, bool chained) {
  const bool idxen = (in.raw[0] >> 13) & 1;
  return idxen && chained;
}

// Which entry of a user-data descriptor TABLE each buffer_load reads.
//
// A V# is 4 dwords, so one s_load_dwordx4 per table entry; the entry is
// selected by that s_load's SGPR soffset, computed at runtime from an index
// table the game uploads (`s_lshl_b32 soff, sN, 4` + `s_and_b32 soff, soff,
// 0x1f0`). The compiler emits those s_loads in entry order, but SCHEDULES the
// buffer_loads that consume them in a different order -- so the entry must be
// taken from the s_load, not from the position of the load. Returns, per
// buffer_load pc, {table root SGPR pair, entry index}; absent means the V# is
// inline in user data at srsrc.
std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t> >
MapTableChainedLoads(const Program& insts) {
  std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t> > out;
  int32_t root[128];
  uint32_t slot[128] = {};
  for (int i = 0; i < 128; i++)
    root[i] = -1;
  std::unordered_map<uint32_t, uint32_t> next_slot;  // per table root
  for (const Inst& in : insts) {
    if (in.enc == Enc::kSop1 && in.opcode == 0x20)
      break;  // s_setpc_b64 (return)
    if (in.enc == Enc::kSopp && in.opcode == 1)
      break;  // s_endpgm
    if (in.enc == Enc::kSmrd &&
        in.opcode <= 0x04) {  // s_load_dword{,x2,x4,x8,x16}
      const uint32_t sdst = (in.raw[0] >> 6) & 0x7F;
      const uint32_t sbase = (in.raw[0] & 0x3F) * 2;
      const uint32_t nreg = in.opcode == 0   ? 1
                            : in.opcode == 1 ? 2
                            : in.opcode == 2 ? 4
                            : in.opcode == 3 ? 8
                                             : 16;
      const uint32_t s = in.opcode == 2 ? next_slot[sbase]++ : 0;
      for (uint32_t k = 0; k < nreg && sdst + k < 128; k++) {
        root[sdst + k] = static_cast<int32_t>(sbase);
        slot[sdst + k] = s;
      }
      continue;
    }
    if ((in.enc != Enc::kMubuf && in.enc != Enc::kMtbuf) || in.opcode > 0x03)
      continue;
    const uint32_t srsrc = ((in.raw[1] >> 16) & 0x1F) * 4;
    if (srsrc < 128 && root[srsrc] >= 0)
      out[in.pc] = {static_cast<uint32_t>(root[srsrc]), slot[srsrc]};
  }
  return out;
}

const char* EncName(Enc e) {
  switch (e) {
    case Enc::kSop1:
      return "sop1";
    case Enc::kSop2:
      return "sop2";
    case Enc::kSopk:
      return "sopk";
    case Enc::kSopc:
      return "sopc";
    case Enc::kSopp:
      return "sopp";
    case Enc::kSmrd:
      return "smem";
    case Enc::kVop1:
      return "vop1";
    case Enc::kVop2:
      return "vop2";
    case Enc::kVop3:
      return "vop3";
    case Enc::kVop3p:
      return "vop3p";
    case Enc::kVopc:
      return "vopc";
    case Enc::kVintrp:
      return "vintrp";
    case Enc::kDs:
      return "ds";
    case Enc::kMubuf:
      return "mubuf";
    case Enc::kMtbuf:
      return "mtbuf";
    case Enc::kMimg:
      return "mimg";
    case Enc::kExp:
      return "exp";
    case Enc::kFlat:
      return "flat";
    default:
      return "UNKNOWN";
  }
}

// Per-instruction decode trace: pc / decoded length / encoding / opcode / raw
// dword(s). A length that lands the next pc mid-instruction shows up as a
// garbage "UNKNOWN" op on the following line (a decoder desync).
void DumpProgram(const Program& prog, const char* tag) {
  std::fprintf(stderr, "[gcnspv] === %s decode: %zu insts ===\n", tag,
               prog.size());
  for (const Inst& in : prog) {
    std::fprintf(stderr, "[gcnspv]   pc=%04x len=%u %-6s op=%#05x  %08x", in.pc,
                 in.size, EncName(in.enc), in.opcode, in.raw[0]);
    if (in.size >= 2)
      std::fprintf(stderr, " %08x", in.raw[1]);
    if (in.has_literal)
      std::fprintf(stderr, " lit=%08x", in.literal);
    std::fprintf(stderr, "\n");
  }
}

// RDNA2 renumbered v_cndmask_b32 from GFX7's VOP2 0x00 to 0x01 (0x01 is
// v_readlane on GFX7); map it back to the shared emitter's VCC select.
uint32_t RemapVop2(uint32_t op) {
  return op == 0x01 ? 0x00 : op;
}

bool RdnaSharedVop2(uint32_t op) {
  return (op >= 0x03 && op <= 0x05) || (op >= 0x08 && op <= 0x0c) ||
         (op >= 0x0f && op <= 0x14) || op == 0x16 || op == 0x18 ||
         (op >= 0x1a && op <= 0x1d) || op == 0x2f;
}

bool UnsupportedRdnaScalarSource(uint32_t field) {
  return (field >= 108 && field <= 123) || (field >= 209 && field <= 239) ||
         field == 249 || field == 250 || field == 254;
}

bool UnsupportedRdnaValuSource(uint32_t field) {
  if (field >= 256)
    return false;
  return (field >= 108 && field <= 123) || (field >= 209 && field <= 239) ||
         field == 249 || field == 250 || field == 254;
}

bool UnsupportedRdnaSdwaSource(uint32_t field) {
  return UnsupportedRdnaValuSource(field) || field == 255;
}

bool UsesUnsupportedRdnaSource(const Inst& inst) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  switch (inst.enc) {
    case Enc::kSop1:
      return UnsupportedRdnaScalarSource(w & 0xff);
    case Enc::kSop2:
    case Enc::kSopc:
      return UnsupportedRdnaScalarSource(w & 0xff) ||
             UnsupportedRdnaScalarSource((w >> 8) & 0xff);
    case Enc::kSmrd:
      return UnsupportedRdnaScalarSource((w1 >> 25) & 0x7f);
    case Enc::kVop1: {
      if (inst.extension == gpu::gcn::InstExtension::kSdwa) {
        const uint32_t src0 = (w1 & 0xff) + (((w1 >> 23) & 1) ? 0 : 256);
        return UnsupportedRdnaSdwaSource(src0);
      }
      return UnsupportedRdnaValuSource(w & 0x1ff);
    }
    case Enc::kVop2:
    case Enc::kVopc: {
      if (inst.extension == gpu::gcn::InstExtension::kSdwa) {
        const uint32_t src0 = (w1 & 0xff) + (((w1 >> 23) & 1) ? 0 : 256);
        const uint32_t src1 = ((w >> 9) & 0xff) + (((w1 >> 31) & 1) ? 0 : 256);
        return UnsupportedRdnaSdwaSource(src0) ||
               UnsupportedRdnaSdwaSource(src1);
      }
      return UnsupportedRdnaValuSource(w & 0x1ff);
    }
    case Enc::kVop3:
    case Enc::kVop3p: {
      const uint32_t count = Vop3SourceCount(inst.enc, inst.opcode);
      for (uint32_t source = 0; source < count; source++)
        if (UnsupportedRdnaValuSource((w1 >> (source * 9)) & 0x1ff))
          return true;
      return false;
    }
    default:
      return false;
  }
}

Id RdnaF16Bits(Translator& t, uint32_t field, uint32_t literal) {
  static constexpr uint16_t kInline[] = {0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000,
                                         0xc000, 0x4400, 0xc400, 0x3118};
  if (field >= 240 && field <= 248)
    return t.U32(kInline[field - 240]);
  return t.SrcRaw(field, literal);
}

// RDNA2 VOP3 integer operations whose gfx10 opcodes or destination layouts do
// not match the shared GFX7 emitter.
bool RdnaVop3HasSdst(uint32_t op) {
  return op == 0x128 || op == 0x129 || op == 0x12A || op == 0x30F ||
         op == 0x310 || op == 0x319 || op == 0x16D || op == 0x16E ||
         op == 0x176 || op == 0x177;
}

bool RdnaEmitVop3Int(Translator& t,
                     uint32_t op,
                     uint32_t vdst,
                     uint32_t sdst,
                     Id s0,
                     Id s1,
                     Id s2) {
  const auto carry = [&](spv::Op operation, Id a, Id b, Id carry_in = 0) {
    Id pair = t.m.Emit(operation, t.PairType(), {a, b});
    Id value = t.m.CompositeExtract(t.t_u, pair, 0);
    Id flag = t.m.CompositeExtract(t.t_u, pair, 1);
    if (carry_in) {
      pair =
          t.m.Emit(operation, t.PairType(), {value, t.And(carry_in, t.U32(1))});
      value = t.m.CompositeExtract(t.t_u, pair, 0);
      flag = t.Or(flag, t.m.CompositeExtract(t.t_u, pair, 1));
    }
    t.SetVg(vdst, value);
    t.SetSdst(sdst, 0, t.And(flag, t.Exec()));
  };
  switch (op) {
    case 0x30F:
      carry(spv::Op::OpIAddCarry, s0, s1);
      return true;  // v_add_co_u32
    case 0x310:
      carry(spv::Op::OpISubBorrow, s0, s1);
      return true;  // v_sub_co_u32
    case 0x319:
      carry(spv::Op::OpISubBorrow, s1, s0);
      return true;  // v_subrev_co_u32
    case 0x128:
      carry(spv::Op::OpIAddCarry, s0, s1, s2);
      return true;  // v_add_co_ci_u32
    case 0x129:
      carry(spv::Op::OpISubBorrow, s0, s1, s2);
      return true;  // v_sub_co_ci_u32
    case 0x12A:
      carry(spv::Op::OpISubBorrow, s1, s0, s2);
      return true;  // v_subrev_co_ci_u32
    case 0x125:
    case 0x37F:
      t.SetVg(vdst, t.Add(s0, s1));
      return true;  // v_add_nc_{u,i}32
    case 0x126:
    case 0x376:
      t.SetVg(vdst, t.Sub(s0, s1));
      return true;  // v_sub_nc_{u,i}32
    case 0x127:
      t.SetVg(vdst, t.Sub(s1, s0));
      return true;  // v_subrev_nc_u32
    case 0x11E:
      t.SetVg(vdst, t.Not(t.Xor(s0, s1)));
      return true;  // VOP3-form v_xnor_b32
    case 0x346:
      t.SetVg(vdst, t.Add(t.Shl(s0, s1), s2));
      return true;  // v_lshl_add_u32
    case 0x347:
      t.SetVg(vdst, t.Shl(t.Add(s0, s1), s2));
      return true;  // v_add_lshl_u32
    case 0x36D:
      t.SetVg(vdst, t.Add(t.Add(s0, s1), s2));
      return true;  // v_add3_u32
    case 0x36F:
      t.SetVg(vdst, t.Or(t.Shl(s0, s1), s2));
      return true;  // v_lshl_or_b32
    case 0x371:
      t.SetVg(vdst, t.Or(t.And(s0, s1), s2));
      return true;  // v_and_or_b32
    case 0x372:
      t.SetVg(vdst, t.Or(t.Or(s0, s1), s2));
      return true;  // v_or3_b32
    default:
      return false;
  }
}

// VOP3P packed f16: componentwise on the unpacked f32x2. op_sel/neg/clamp and
// packed-integer ops are not modelled (fall back).
bool RdnaEmitVop3p(Translator& t,
                   uint32_t op,
                   uint32_t vdst,
                   uint32_t s0,
                   uint32_t s1,
                   uint32_t s2,
                   uint32_t lit) {
  auto unpack = [&](uint32_t f) {
    return t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                       {RdnaF16Bits(t, f, lit)});
  };
  auto pack = [&](Id v2) {
    return t.m.ExtInst(t.t_u, GLSLstd450PackHalf2x16, {v2});
  };
  const Id a = unpack(s0), b = unpack(s1);
  switch (op) {
    case 0x0F:
      t.SetVg(vdst, pack(t.m.Emit(spv::Op::OpFAdd, t.t_v2, {a, b})));
      return true;  // v_pk_add_f16
    case 0x10:
      t.SetVg(vdst, pack(t.m.Emit(spv::Op::OpFMul, t.t_v2, {a, b})));
      return true;  // v_pk_mul_f16
    case 0x11:
      t.SetVg(vdst, pack(t.m.ExtInst(t.t_v2, GLSLstd450FMin, {a, b})));
      return true;  // v_pk_min_f16
    case 0x12:
      t.SetVg(vdst, pack(t.m.ExtInst(t.t_v2, GLSLstd450FMax, {a, b})));
      return true;  // v_pk_max_f16
    case 0x0E: {    // v_pk_fma_f16
      t.SetVg(vdst,
              pack(t.m.ExtInst(t.t_v2, GLSLstd450Fma, {a, b, unpack(s2)})));
      return true;
    }
    default:
      return false;
  }
}

// ---- SMEM (constant buffers) ------------------------------------------------
// Walk the s_load pointer chain feeding a descriptor base SGPR back to a
// user-data root. `loads` maps an s_load's destination SGPR to its {source base
// SGPR, byte offset}. Fills chain_off[] (root-first resolution order) and
// returns the root user-data SGPR; *len is the number of dereferences (0 ==
// already a user-data descriptor). Bounded to 3 levels.
struct CbufDef {
  uint32_t source, offset, count;
};

bool Overlaps(uint32_t first,
              uint32_t count,
              uint32_t other_first,
              uint32_t other_count) {
  return first < other_first + other_count && other_first < first + count;
}

void InvalidateCbufDefs(std::unordered_map<uint32_t, CbufDef>& loads,
                        ScalarWrite write) {
  if (!write.count)
    return;
  for (auto it = loads.begin(); it != loads.end();) {
    if (Overlaps(it->first, it->second.count, write.first, write.count))
      it = loads.erase(it);
    else
      ++it;
  }
}

bool UsedAsBaseBeforeOverwrite(const Program& program,
                               uint32_t index,
                               uint32_t sdst,
                               uint32_t count) {
  for (uint32_t i = index + 1; i < program.size(); i++) {
    const Inst& inst = program[i];
    if (inst.enc == Enc::kSmrd && SmemLoadCount(inst.opcode) &&
        DecodeSmem(inst).sbase == sdst)
      return true;
    const ScalarWrite write = DecodeScalarWrite(inst);
    if (Overlaps(sdst, count, write.first, write.count))
      return false;
  }
  return false;
}

std::unordered_map<uint32_t, uint64_t> BufferVersionKeys(
    const Program& program) {
  std::unordered_map<uint32_t, uint64_t> out;
  uint32_t versions[136] = {};
  uint32_t generation = 1;
  for (const Inst& inst : program) {
    if (inst.enc == Enc::kSmrd && SmemLoadCount(inst.opcode)) {
      const Smem smem = DecodeSmem(inst);
      const uint32_t dwords = smem.op >= 0x08 ? 4 : 2;
      uint64_t key = smem.sbase | (static_cast<uint64_t>(dwords == 4) << 7);
      for (uint32_t i = 0; i < dwords; i++)
        key |= static_cast<uint64_t>(versions[smem.sbase + i]) << (8 + i * 13);
      out.emplace(inst.pc, key);
    }
    const ScalarWrite write = DecodeScalarWrite(inst);
    for (uint32_t i = 0; i < write.count && write.first + i < 136; i++)
      versions[write.first + i] = generation;
    generation++;
  }
  return out;
}

uint32_t TraceCbufChain(uint32_t sbase,
                        const std::unordered_map<uint32_t, CbufDef>& loads,
                        uint32_t chain_off[3],
                        uint32_t* len) {
  uint32_t cur = sbase, n = 0, tmp[3] = {};
  while (n < 3) {
    auto it = loads.find(cur);
    if (it == loads.end())
      break;
    tmp[n++] = it->second.offset;
    cur = it->second.source;
  }
  for (uint32_t i = 0; i < n; i++)
    chain_off[i] = tmp[n - 1 - i];  // reverse
  *len = n;
  return cur;
}

// Plan the set-1 UBO bindings a stage's SMEM loads reference. A leaf read is an
// s_buffer_load* (op 0x08-0x0C, V# in the sbase quad) or an s_load* (op
// 0x00-0x04, pointer in the sbase pair) whose result is used as data -- not as
// another SMEM's descriptor base. When the base SGPR was itself s_load'd (a
// runtime pointer chain, e.g. a 2D VS that loads its transform's V# from a root
// descriptor table), the chain back to the user-data root is recorded so the
// renderer can walk it.
bool RdnaPlanCbufs(const Program& program,
                   uint32_t first_binding,
                   std::vector<ShaderCbuf>& cbufs,
                   std::unordered_map<uint32_t, uint32_t>& bindings,
                   std::unordered_map<uint32_t, uint32_t>& by_pc) {
  // Walk in program order, growing the def map as s_loads appear, so each
  // SMEM's base traces through the defs live AT that instruction. Shaders
  // reuse SGPRs (the sprite VS s_buffer_loads its transform from the s[8:11]
  // user-data V#, then s_loads the vertex V# INTO s[8:11]); a whole-program
  // last-write map would misroute the transform to the vertex chain.
  std::unordered_map<uint32_t, CbufDef> loads;
  std::unordered_map<uint64_t, uint32_t> binding_by_producer;
  const auto version_keys = BufferVersionKeys(program);
  uint32_t inst_index = 0;
  for (const Inst& inst : program) {
    const uint32_t producer = inst_index++;
    if (inst.enc != Enc::kSmrd) {
      InvalidateCbufDefs(loads, DecodeScalarWrite(inst));
      continue;
    }
    const Smem smem = DecodeSmem(inst);
    const uint32_t op = smem.op;
    const bool sbufload = op >= 0x08 && op <= 0x0C, sload = op <= 0x04;
    if (!sbufload && !sload)
      continue;
    const uint32_t sbase = smem.sbase, sdst = smem.sdst;
    const uint32_t load_count = SmemLoadCount(op);
    const int32_t off =
        sbufload ? static_cast<int32_t>(inst.raw[1] & 0xFFFFF) : smem.offset;
    if (sload && smem.soffset == 125 && off < 0)
      return false;
    if (sload &&
        UsedAsBaseBeforeOverwrite(program, producer, sdst, load_count)) {
      InvalidateCbufDefs(loads, {sdst, load_count});
      loads[sdst] = {sbase, static_cast<uint32_t>(off), load_count};
      continue;
    }
    const uint32_t hi =
        smem.soffset != 125
            ? gpu::gcn::kCbufDwords
            : static_cast<uint32_t>(off < 0 ? 0 : off) / 4 + SmemLoadCount(op);
    if (smem.soffset != 125 || hi > gpu::gcn::kCbufDwords)
      return false;

    uint32_t chain_off[3] = {}, chain_len = 0;
    const uint32_t root = TraceCbufChain(sbase, loads, chain_off, &chain_len);
    const uint64_t key = version_keys.at(inst.pc);

    auto it = binding_by_producer.find(key);
    if (it == binding_by_producer.end()) {
      const uint32_t binding =
          first_binding + static_cast<uint32_t>(cbufs.size());
      if (binding >= kMaxCbufBindings)
        return false;
      it = binding_by_producer.emplace(key, binding).first;
      bindings.emplace(sbase, binding);
      ShaderCbuf cb;
      cb.binding = binding;
      cb.ud_sgpr = root;
      cb.num_dwords = hi;
      cb.chain_len = chain_len;
      for (uint32_t i = 0; i < 3; i++)
        cb.chain_off[i] = chain_off[i];
      cb.use_pc = inst.pc;
      cbufs.push_back(cb);
    } else {
      for (ShaderCbuf& cb : cbufs)
        if (cb.binding == it->second && hi > cb.num_dwords)
          cb.num_dwords = hi;
    }
    by_pc[inst.pc] = it->second;
    InvalidateCbufDefs(loads, {sdst, load_count});
  }
  return true;
}

// Plan set-1 UBO bindings for CONSTANT buffer_load_format ops (a load whose
// index is not the vertex index reads a uniform, e.g. the 2D ortho matrix).
// Each distinct srsrc V# gets one binding; the renderer resolves the live V#
// from user data at draw time, exactly like the SMEM cbufs
// (decodeVBuffer(&vud[srsrc])).
void RdnaPlanBufLoadCbufs(const Program& program,
                          uint32_t first_binding,
                          std::vector<ShaderCbuf>& cbufs,
                          std::unordered_map<uint32_t, uint32_t>& bindings,
                          std::unordered_map<uint32_t, uint32_t>& by_pc) {
  // Mirror ParseFetchInsts' walk: srsrc SGPRs written by an s_load hold V#s
  // from the user-data descriptor TABLE (entry index from
  // MapTableChainedLoads). A constant load through such a V# becomes a chained
  // cbuf (root = the s_load's sbase pair, table offset = entry * 16 bytes); the
  // renderer derefs the table pointer at draw time. srsrc-keyed bindings
  // collide when the shader reuses an SGPR quad (s[8:11] = MVP V# then vertex
  // V#), so constant loads are bound per-instruction (by_pc) instead.
  const auto chained_loads = MapTableChainedLoads(program);
  for (const Inst& inst : program) {
    if (inst.enc == Enc::kSop1 && inst.opcode == 0x20)
      break;  // s_setpc_b64
    if (inst.enc == Enc::kSopp && inst.opcode == 1)
      break;  // s_endpgm
    if ((inst.enc != Enc::kMubuf && inst.enc != Enc::kMtbuf) ||
        inst.opcode > 0x03)
      continue;
    const uint32_t srsrc = ((inst.raw[1] >> 16) & 0x1F) * 4;
    const auto chain = chained_loads.find(inst.pc);
    const bool chained = chain != chained_loads.end();
    if (BufLoadIsVertexFetch(inst, chained))
      continue;  // real fetch -> vertex input
    const uint32_t binding =
        first_binding + static_cast<uint32_t>(cbufs.size());
    if (chained) {
      if (binding >= kMaxCbufBindings)
        return;
      by_pc[inst.pc] = binding;
      ShaderCbuf cb;
      cb.binding = binding;
      cb.ud_sgpr = chain->second.first;
      cb.num_dwords = 16;
      cb.chain_len = 1;
      cb.chain_off[0] = chain->second.second * 16;
      cb.use_pc = inst.pc;
      cbufs.push_back(cb);
      continue;
    }
    if (bindings.count(srsrc))
      continue;
    if (binding >= kMaxCbufBindings)
      return;
    bindings[srsrc] = binding;
    ShaderCbuf cb;
    cb.binding = binding;
    cb.ud_sgpr = srsrc;
    cb.num_dwords = 16;
    cb.use_pc = inst.pc;
    cbufs.push_back(cb);
  }
}

// Find `s_mov exec, sN` movs whose source SGPR is never written before the mov:
// sN then holds SPI launch state we do not model (the PS ABI saves the initial
// coverage mask in the first post-user-data SGPR and reloads EXEC from it).
// Emitting the mov would read our zero-initialised register file and turn EXEC
// off, making the CFG path skip every export (the PS then kills all fragments).
// Those movs are dropped so EXEC keeps its all-on seed.
std::unordered_set<uint32_t> LaunchExecMovPcs(const Program& program) {
  std::unordered_set<uint32_t> skip, written;
  for (const Inst& in : program) {
    uint32_t d0 = 0xFFFF, n = 1;
    switch (in.enc) {
      case Enc::kSop1: {
        const uint32_t sdst = (in.raw[0] >> 16) & 0x7F;
        const uint32_t src = in.raw[0] & 0xFF;
        const bool b64 = in.opcode == 0x04;
        if ((in.opcode == 0x03 || b64) && sdst == 126 && src <= 105 &&
            !written.count(src) && (!b64 || !written.count(src + 1))) {
          skip.insert(in.pc);
          if (ShDbg())
            std::fprintf(stderr,
                         "[gcnspv] drop launch-state exec mov @pc=%04x (s%u)\n",
                         in.pc, src);
        }
        d0 = sdst;
        n = b64 ? 2 : 1;
        break;
      }
      case Enc::kSop2:
      case Enc::kSopk:
        d0 = (in.raw[0] >> 16) & 0x7F;
        break;
      case Enc::kSmrd:
        if (in.opcode <= 0x0C) {
          d0 = (in.raw[0] >> 6) & 0x7F;
          n = SmemLoadCount(in.opcode);
        }
        break;
      default:
        break;
    }
    if (d0 <= 105)
      for (uint32_t k = 0; k < n; k++)
        written.insert(d0 + k);
  }
  return skip;
}

void RdnaEmitSmem(Translator& t, const Inst& inst, StageContext& sc) {
  const Smem smem = DecodeSmem(inst);
  const uint32_t op = smem.op;
  // s_load* (op 0x00-0x04, a pointer in the sbase pair) and s_buffer_load* (op
  // 0x08-0x0C, a V# in the sbase quad) read `off` bytes into sdst.. from the
  // UBO the renderer bound for this sbase. A 2D VS reads its transform matrix
  // this way.
  if (op <= 0x04 || (op >= 0x08 && op <= 0x0C)) {
    auto pc_it = sc.smem_cbuf_by_pc.find(inst.pc);
    auto base_it = sc.cbuf_bind.find(smem.sbase);
    if (pc_it == sc.smem_cbuf_by_pc.end() && base_it == sc.cbuf_bind.end()) {
      // An s_load whose result feeds a later SMEM is a pointer-chain link: the
      // renderer resolves that descriptor host-side, so it has no UBO and emits
      // nothing. An unplanned s_buffer_load is a real gap.
      if (op >= 0x08)
        gpu::gcn::WarnUnsupported("smem.cbuf-unplanned", op, inst.raw[0],
                                  inst.raw[1]);
      return;
    }
    const uint32_t binding =
        pc_it != sc.smem_cbuf_by_pc.end() ? pc_it->second : base_it->second;
    const uint32_t immediate = op >= 0x08
                                   ? inst.raw[1] & 0xFFFFC
                                   : static_cast<uint32_t>(smem.offset) & ~3u;
    const Id soffset =
        smem.soffset == 125 ? t.U32(0) : t.SrcRaw(smem.soffset, 0);
    const Id byte_offset = t.Add(t.And(soffset, t.U32(~3u)), t.U32(immediate));
    const Id dword0 = t.Shr(byte_offset, t.U32(2));
    const uint32_t n = SmemLoadCount(op);
    for (uint32_t k = 0; k < n; k++)
      t.SetSdst(smem.sdst, k, t.CbufDwordId(binding, t.Add(dword0, t.U32(k))));
    return;
  }
  gpu::gcn::WarnUnsupported("smem.rdna", op, inst.raw[0], inst.raw[1]);
}

// Address of the shader being translated, for shader-specific debug output.
thread_local uint64_t g_ps_addr = 0;

// ---- exports ----------------------------------------------------------------
// gfx10.3 export targets: MRT0..7 = 0..7, MRTZ = 8, NULL = 9, POS0..4 = 12..16,
// PRIM = 20 (NGG connectivity), PARAM0..31 = 32..63.
void EmitExport(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t en = w & 0xF, target = (w >> 4) & 0x3F, compr = (w >> 10) & 1;
  const uint32_t v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF,
                         (w1 >> 24) & 0xFF};
  if (ShDbg())
    std::fprintf(
        stderr, "[gcnspv] %s-exp target=%u en=%#x compr=%u done=%u vsrc=%08x\n",
        sc.is_ps ? "ps" : "vs", target, en, compr, (w >> 11) & 1, w1);
  if (!en)
    return;  // architecturally null export
  if (sc.is_ps) {
    // DELTA_GPU_EXPTRACE: the EN mask of each colour export. A component the
    // shader does not export gets a default here, and defaulting alpha to 1
    // would make every SRC_ALPHA blend opaque.
    if (std::getenv("DELTA_GPU_EXPTRACE") && target <= 7) {
      static uint32_t seen[256] = {};
      const uint32_t k = ((target & 7) << 5) | ((en & 0xF) << 1) | compr;
      if (seen[k]++ == 0)
        std::fprintf(stderr, "[exp] ps %#lx mrt%u en=%#x compr=%u\n",
                     (unsigned long)g_ps_addr, target, en, compr);
    }
    if (target <= 7) {  // MRT0..7
      sc.wrote_color = true;
      Id col;
      if (compr) {
        // A compressed export packs two components per VGPR, and EN selects
        // PAIRS: bit 0 covers components 0-1 in v[0], bit 2 covers 2-3 in v[1].
        // Reading v[1] when the shader never wrote it produced a garbage alpha,
        // which with SRC_ALPHA blending draws every blended pass fully opaque.
        Id c[4];
        if (en & 0x1) {
          const Id c01 =
              t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.Vg(v[0])});
          c[0] = t.m.CompositeExtract(t.t_f, c01, 0);
          c[1] = t.m.CompositeExtract(t.t_f, c01, 1);
        } else {
          c[0] = c[1] = t.F32(0.f);
        }
        if (en & 0x4) {
          const Id c23 =
              t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.Vg(v[1])});
          c[2] = t.m.CompositeExtract(t.t_f, c23, 0);
          c[3] = t.m.CompositeExtract(t.t_f, c23, 1);
        } else {
          c[2] = t.F32(0.f);
          c[3] = t.F32(1.f);
        }
        col = t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]});
      } else {
        Id c[4];
        for (int i = 0; i < 4; i++)
          c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(i == 3 ? 1.f : 0.f);
        col = t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]});
      }
      // DELTA_GPU_FORCECOLOR: override the PS color output with opaque red, so
      // a rendered draw is unmistakably visible regardless of
      // texture/vertex-color math. Combined with FORCEQUAD this isolates
      // rasterization/scanout from the PS color path. Replace non-finite colour
      // components with zero and clamp to the half range. A NaN written into an
      // HDR target poisons every pass that reads it: Skyrim's 2x2 exposure
      // buffer ends up all-ones (NaN), the tonemap then has no usable exposure,
      // and the frame swings between near-black and flat green. The NaNs come
      // from our own approximations, not from the title. DELTA_GPU_ALLOWNAN
      // keeps whatever the shader produced.
      static const bool no_nan = std::getenv("DELTA_GPU_ALLOWNAN") == nullptr;
      if (no_nan) {
        const Id v = col;
        Id comps[4];
        for (int i = 0; i < 4; i++) {
          const Id c = t.m.CompositeExtract(t.t_f, v, i);
          const Id finite = t.m.Emit(spv::Op::OpFOrdEqual, t.t_bool, {c, c});
          const Id clamped = t.m.ExtInst(t.t_f, GLSLstd450FClamp,
                                         {c, t.F32(-65504.f), t.F32(65504.f)});
          comps[i] =
              t.m.Emit(spv::Op::OpSelect, t.t_f, {finite, clamped, t.F32(0.f)});
        }
        col = t.m.CompositeConstruct(t.t_v4,
                                     {comps[0], comps[1], comps[2], comps[3]});
      }
      // DELTA_GPU_DEBUGALPHA: show the alpha the blend will use as greyscale.
      if (std::getenv("DELTA_GPU_DEBUGALPHA")) {
        const Id a = t.m.CompositeExtract(t.t_f, col, 3);
        col = t.m.CompositeConstruct(t.t_v4, {a, a, a, t.F32(1.f)});
      }
      static const bool force_col =
          std::getenv("DELTA_GPU_FORCECOLOR") != nullptr;
      if (force_col)
        col = t.m.CompositeConstruct(
            t.t_v4, {t.F32(1.f), t.F32(0.f), t.F32(0.f), t.F32(1.f)});
      t.m.Store(gpu::gcn::PsColorOut(t, sc, target), col);
      if (sc.color_written_var)
        t.m.Store(sc.color_written_var, t.U32(1));
      return;
    }
    if (target == 8 && (en & 1)) {  // MRTZ: depth export
      const Id depth =
          compr
              ? t.m.CompositeExtract(
                    t.t_f,
                    t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {t.Vg(v[0])}),
                    0)
              : t.VgF(v[0]);
      t.m.Store(gpu::gcn::PsDepthOut(t, sc), depth);
      return;
    }
    if (target != 9)
      gpu::gcn::WarnUnsupported("exp.ps-target", target, w, w1);
    return;
  }
  if (target == 12) {  // POS0 -> gl_Position
    if (sc.dbg_pos_in && sc.dbg_pos_cbuf >= 0) {
      Id in[4];
      const Id val = t.m.Load(sc.dbg_pos_comps == 2   ? t.t_v2
                              : sc.dbg_pos_comps == 3 ? t.t_v3
                                                      : t.t_v4,
                              sc.dbg_pos_in);
      for (uint32_t i = 0; i < 4; i++)
        in[i] = i < sc.dbg_pos_comps ? t.m.CompositeExtract(t.t_f, val, i)
                                     : t.F32(i == 3 ? 1.f : 0.f);
      if (sc.dbg_pos_world >= 0) {
        Id w[4];
        for (uint32_t r = 0; r < 3; r++) {
          w[r] = t.F32(0.f);
          for (uint32_t c = 0; c < 4; c++) {
            const Id m = t.m.Bitcast(
                t.t_f, t.CbufDword(static_cast<uint32_t>(sc.dbg_pos_world),
                                   r * 4 + c));
            w[r] = t.FAdd(w[r], t.FMul(m, in[c]));
          }
        }
        w[3] = t.F32(1.f);
        for (int i = 0; i < 4; i++)
          in[i] = w[i];
      }
      Id row[4];
      for (uint32_t r = 0; r < 4; r++) {
        row[r] = t.F32(0.f);
        for (uint32_t c = 0; c < 4; c++) {
          const Id m = t.m.Bitcast(
              t.t_f, t.CbufDword(static_cast<uint32_t>(sc.dbg_pos_cbuf),
                                 sc.dbg_pos_dword + r * 4 + c));
          row[r] = t.FAdd(row[r], t.FMul(m, in[c]));
        }
      }
      t.m.Store(sc.pos_out, t.m.CompositeConstruct(
                                t.t_v4, {row[0], row[1], row[2], row[3]}));
      return;
    }
    Id c[4];
    for (int i = 0; i < 4; i++)
      c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(i == 3 ? 1.f : 0.f);
    t.m.Store(sc.pos_out,
              t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]}));
  } else if (target >= 32 && target <= 63) {  // PARAM0..31
    const uint32_t p = target - 32;
    if (p + 1 > sc.max_param)
      sc.max_param = p + 1;
    const Id out_var = gpu::gcn::VsParamOut(t, sc, p);
    Id c[4];
    for (int i = 0; i < 4; i++)
      c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(0.f);
    t.m.Store(out_var,
              t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]}));
  } else if (target != 9) {
    gpu::gcn::WarnUnsupported("exp.vs-target", target, w, w1);
  }
}

// SDWA (src0 field == 249) puts the real operands in the extension dword. DPP
// extensions are rejected before reaching this partial SDWA implementation.
struct SdwaMod {
  uint32_t src0 = 0, src1 = 0;          // resolved operand fields
  uint32_t src0_sel = 6, src1_sel = 6;  // 0-3 byte, 4-5 word, 6 whole dword
  bool src0_sext = false, src1_sext = false;
  bool src0_neg = false, src0_abs = false;
  bool src1_neg = false, src1_abs = false;
  uint32_t dst_sel = 6;
  bool clamp = false;
  uint32_t omod = 0;
};

// S0/S1 (bits 23/31) choose the SCALAR file for that operand. Reading them as
// VGPRs instead is silent corruption: a 3D vertex shader multiplies its
// transform out of SGPRs, so the matrix comes back as zeros and every position
// collapses. DPP has no scalar-source form (its src0 is always a VGPR).
SdwaMod DecodeSdwa(const Inst& inst, uint32_t vsrc1, bool dpp) {
  const uint32_t m = inst.raw[1];
  SdwaMod s;
  s.src0 = (m & 0xFF) + ((!dpp && ((m >> 23) & 1)) ? 0u : 256u);
  s.src1 = vsrc1 + 256u;
  if (dpp)
    return s;
  s.dst_sel = (m >> 8) & 7;
  s.clamp = (m >> 13) & 1;
  s.omod = (m >> 14) & 3;
  s.src0_sel = (m >> 16) & 7;
  s.src0_sext = (m >> 19) & 1;
  s.src0_neg = (m >> 20) & 1;
  s.src0_abs = (m >> 21) & 1;
  s.src1_sel = (m >> 24) & 7;
  s.src1_sext = (m >> 27) & 1;
  s.src1_neg = (m >> 28) & 1;
  s.src1_abs = (m >> 29) & 1;
  if ((m >> 31) & 1)
    s.src1 = vsrc1;
  return s;
}

// Apply one operand's SDWA sub-dword selection to its raw 32-bit value.
Id SdwaSelect(Translator& t, Id raw, uint32_t sel, bool sext) {
  if (sel >= 6)
    return raw;
  const uint32_t bits = sel < 4 ? 8u : 16u;
  const uint32_t off = sel < 4 ? sel * 8u : (sel - 4) * 16u;
  const Id shifted = t.Shr(raw, t.U32(off));
  if (!sext)
    return t.And(shifted, t.U32(bits == 8 ? 0xFFu : 0xFFFFu));
  return t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_u,
                  {shifted, t.U32(0), t.U32(bits)});
}

void ResolveValuSrc0(const Inst& inst,
                     uint32_t src0,
                     uint32_t& field,
                     uint32_t& literal) {
  if (inst.extension == gpu::gcn::InstExtension::kSdwa) {
    field = DecodeSdwa(inst, 0, false).src0;
    literal = 0;
  } else {
    field = src0;
    literal = inst.literal;
  }
}

// ---- per-instruction dispatch ----------------------------------------------
// Decodes RDNA2 field layouts and calls the shared GFX7 emitters (which take
// pre-decoded operands + a GFX7-canonical opcode). The scalar and VOP1/2/C
// field layouts are identical to GFX7; VOP3 differs only in the opcode-field
// width and the CLAMP bit position (bit 15, not 11); SMEM replaces SMRD.
void RdnaEmitInst(Translator& t, const Inst& inst, StageContext& sc) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  if (inst.extension == gpu::gcn::InstExtension::kDpp ||
      inst.extension == gpu::gcn::InstExtension::kDpp8 ||
      inst.extension == gpu::gcn::InstExtension::kDpp8Fi) {
    gpu::gcn::WarnUnsupported("dpp.rdna", inst.opcode, w, w1);
    return;
  }
  if (UsesUnsupportedRdnaSource(inst)) {
    gpu::gcn::WarnUnsupported("source.special.rdna", inst.opcode, w, w1);
    return;
  }
  switch (inst.enc) {
    case Enc::kSop1:
      if (sc.skip_launch_movs.count(inst.pc))
        break;
      if (inst.opcode == 0x20 || inst.opcode == 0x21) {
        gpu::gcn::WarnUnsupported("sop1.control-flow.rdna", inst.opcode, w, w1);
        break;
      }
      gpu::gcn::EmitSop1(t, inst);
      break;
    case Enc::kSop2:
      gpu::gcn::EmitSop2(t, inst);
      break;
    case Enc::kSopc:
      gpu::gcn::EmitSopc(t, inst);
      break;
    case Enc::kSopk:
      if (inst.opcode >= 0x17 && inst.opcode <= 0x1A)
        break;  // per-counter waits; translated memory operations are ordered
      if (inst.opcode == 0x10) {
        const uint32_t sdst = (w >> 16) & 0x7f;
        const uint32_t simm = static_cast<uint32_t>(
            static_cast<int32_t>(static_cast<int16_t>(w & 0xffff)));
        t.SetSdst(sdst, 0, t.Mul(t.Sdst(sdst), t.U32(simm)));
        break;  // RDNA s_mulk_i32 does not modify SCC.
      }
      gpu::gcn::EmitSopk(t, inst);
      break;
    case Enc::kSopp:
      if (inst.opcode == 0x0A && sc.is_cs) {  // s_barrier
        t.m.EmitVoid(spv::Op::OpControlBarrier,
                     {t.U32(2), t.U32(2), t.U32(0x108)});
      } else if (inst.opcode != 0x00 && inst.opcode != 0x01 &&
                 inst.opcode != 0x02 &&
                 !(inst.opcode >= 0x04 && inst.opcode <= 0x09) &&
                 inst.opcode != 0x0C && inst.opcode != 0x0E &&
                 inst.opcode != 0x0F && inst.opcode != 0x1E &&
                 inst.opcode != 0x1F && inst.opcode != 0x20 &&
                 inst.opcode != 0x21 && inst.opcode != 0x23) {
        gpu::gcn::WarnUnsupported("sopp.rdna", inst.opcode, w, w1);
      }
      break;  // branches are emitted by the CFG; waits/hints are synchronous
    case Enc::kSmrd:
      RdnaEmitSmem(t, inst, sc);
      break;
    case Enc::kVop1: {
      const uint32_t op = inst.opcode, vdst = (w >> 17) & 0xFF;
      const uint32_t raw0 = w & 0x1FF;
      uint32_t src0, lit;
      ResolveValuSrc0(inst, raw0, src0, lit);
      if (op == 0x02) {
        gpu::gcn::WarnUnsupported("v_readfirstlane_b32", op, w, w1);
        break;
      }
      if (raw0 == 249) {  // SDWA: apply the source's sub-dword selection
        const SdwaMod sd = DecodeSdwa(inst, 0, false);
        if (sd.dst_sel != 6 || sd.src0_sel > 6 || sd.src0_neg || sd.src0_abs ||
            sd.clamp || sd.omod)
          gpu::gcn::WarnUnsupported("vop1.sdwa-mod", op, w, w1);
        gpu::gcn::EmitVop1(
            t, op, vdst,
            t.m.Bitcast(t.t_f, SdwaSelect(t, t.SrcRaw(src0, 0), sd.src0_sel,
                                          sd.src0_sext)));
        break;
      }
      if (op == 0x0b) {
        gpu::gcn::EmitVop1(t, op, vdst,
                           t.m.Bitcast(t.t_f, RdnaF16Bits(t, src0, lit)));
      } else {
        gpu::gcn::EmitVop1(t, op, vdst, t.SrcF(src0, lit));
      }
      break;
    }
    case Enc::kVop2: {
      const uint32_t op = inst.opcode, vdst = (w >> 17) & 0xFF;
      const uint32_t vsrc1 = (w >> 9) & 0xFF;
      const uint32_t raw0 = w & 0x1FF;
      uint32_t src0, lit;
      ResolveValuSrc0(inst, raw0, src0, lit);
      if (raw0 == 249 && op >= 0x2b && op <= 0x2d) {
        gpu::gcn::WarnUnsupported("vop2.fma-sdwa", op, w, w1);
        break;
      }
      uint32_t src1 = 256 + vsrc1;
      Id sdwa0 = 0, sdwa1 = 0;
      if (raw0 == 249) {
        const SdwaMod sd = DecodeSdwa(inst, vsrc1, false);
        if (sd.dst_sel != 6 || sd.src0_sel > 6 || sd.src1_sel > 6 ||
            sd.src0_neg || sd.src0_abs || sd.src1_neg || sd.src1_abs ||
            sd.clamp || sd.omod)
          gpu::gcn::WarnUnsupported("vop2.sdwa-mod", op, w, w1);
        src1 = sd.src1;
        sdwa0 = SdwaSelect(t, t.SrcRaw(sd.src0, 0), sd.src0_sel, sd.src0_sext);
        sdwa1 = SdwaSelect(t, t.SrcRaw(sd.src1, 0), sd.src1_sel, sd.src1_sext);
      }
      const Id s0u = sdwa0 ? sdwa0 : t.SrcRaw(src0, lit);
      const Id s1u = sdwa1 ? sdwa1 : t.SrcRaw(src1, lit);
      // RDNA2-only VOP2 numbers the shared GFX7 emitter would misinterpret: the
      // no-carry integer add/sub forms must NOT write VCC (a later v_cndmask
      // reads it), and v_xnor_b32 sits where GFX7 has v_bfm_b32.
      switch (op) {
        case 0x02:  // v_dot2c_f32_f16, not GFX7 v_writelane
        case 0x0D:  // v_dot4c_i32_i8, not GFX7 v_min_legacy_f32
          gpu::gcn::WarnUnsupported("vop2.rdna", op, w, w1);
          break;
        case 0x1E:
          t.SetVg(vdst, t.Not(t.Xor(s0u, s1u)));
          break;  // v_xnor_b32
        case 0x25:
          t.SetVg(vdst, t.Add(s0u, s1u));
          break;  // v_add_nc_u32
        case 0x26:
          t.SetVg(vdst, t.Sub(s0u, s1u));
          break;  // v_sub_nc_u32
        case 0x27:
          t.SetVg(vdst, t.Sub(s1u, s0u));
          break;  // v_subrev_nc_u32
        case 0x28:
        case 0x29:
        case 0x2A: {
          const spv::Op operation =
              op == 0x28 ? spv::Op::OpIAddCarry : spv::Op::OpISubBorrow;
          const Id a = op == 0x2A ? s1u : s0u;
          const Id b = op == 0x2A ? s0u : s1u;
          Id pair = t.m.Emit(operation, t.PairType(), {a, b});
          Id value = t.m.CompositeExtract(t.t_u, pair, 0);
          Id flag = t.m.CompositeExtract(t.t_u, pair, 1);
          pair = t.m.Emit(operation, t.PairType(),
                          {value, t.And(t.Sg(106), t.U32(1))});
          value = t.m.CompositeExtract(t.t_u, pair, 0);
          flag = t.Or(flag, t.m.CompositeExtract(t.t_u, pair, 1));
          t.SetVg(vdst, value);
          t.SetSg(106, t.And(flag, t.Exec()));
          break;
        }
        case 0x2B:
          t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450Fma,
                                     {t.m.Bitcast(t.t_f, s0u),
                                      t.m.Bitcast(t.t_f, s1u), t.VgF(vdst)}));
          break;
        case 0x2C:
          t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450Fma,
                                     {t.m.Bitcast(t.t_f, s0u),
                                      t.m.Bitcast(t.t_f, t.U32(lit)),
                                      t.m.Bitcast(t.t_f, s1u)}));
          break;
        case 0x2D:
          t.SetVgF(vdst, t.m.ExtInst(
                             t.t_f, GLSLstd450Fma,
                             {t.m.Bitcast(t.t_f, s0u), t.m.Bitcast(t.t_f, s1u),
                              t.m.Bitcast(t.t_f, t.U32(lit))}));
          break;
        default:
          if (op == 0x01 || RdnaSharedVop2(op)) {
            gpu::gcn::EmitVop2(
                t, RemapVop2(op), vdst,
                sdwa0 ? t.m.Bitcast(t.t_f, sdwa0) : t.SrcF(src0, lit),
                sdwa1 ? t.m.Bitcast(t.t_f, sdwa1) : t.SrcF(src1, lit), lit);
          } else {
            gpu::gcn::WarnUnsupported("vop2.opcode.rdna", op, w, w1);
          }
          break;
      }
      break;
    }
    case Enc::kVop3p: {
      const uint32_t op = inst.opcode, vdst = w & 0xFF;
      // Componentwise packed operations select each source's low half for the
      // low result and high half for the high result.
      if ((w & 0x0000FF00u) != 0x00004000u ||
          (w1 & 0xF8000000u) != 0x18000000u) {
        gpu::gcn::WarnUnsupported("vop3p.modifier", op, w, w1);
        break;
      }
      const uint32_t p0 = w1 & 0x1FF, p1 = (w1 >> 9) & 0x1FF;
      const uint32_t p2 = (w1 >> 18) & 0x1FF;
      if (!RdnaEmitVop3p(t, op, vdst, p0, p1, p2, inst.literal))
        gpu::gcn::WarnUnsupported("vop3p", op, w, w1);
      break;
    }
    case Enc::kVop3: {
      const uint32_t rdna_op = inst.opcode;
      uint32_t op = rdna_op;
      const uint32_t vdst = w & 0xFF;
      // VOP3-form v_cndmask is the VOP2 alias 0x101 on RDNA2 (0x01 renumber);
      // GFX7's explicit-S2 cndmask VOP3 op is 0x100.
      if (op == 0x101)
        op = 0x100;
      const uint32_t s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF;
      const uint32_t s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
      const bool vop3b = RdnaVop3HasSdst(op);
      const uint32_t sdst = vop3b ? ((w >> 8) & 0x7F) : 106;
      const uint32_t abs = vop3b ? 0 : ((w >> 8) & 7);
      const uint32_t op_sel = vop3b ? 0 : ((w >> 11) & 0xF);
      const bool clamp = (w >> 15) & 1;
      const uint32_t omod = (w1 >> 27) & 3;
      if (op == 0x102 || op == 0x10D) {
        gpu::gcn::WarnUnsupported("vop3.dot.rdna", op, w, w1);
        break;
      }
      if (op_sel)
        gpu::gcn::WarnUnsupported("vop3.op-sel", op, w, w1);
      if (RdnaEmitVop3Int(t, op, vdst, sdst, t.SrcRaw(s0, inst.literal),
                          t.SrcRaw(s1, inst.literal),
                          t.SrcRaw(s2, inst.literal))) {
        if (neg || abs || clamp || omod || op_sel)
          gpu::gcn::WarnUnsupported("vop3.integer-modifier", op, w, w1);
        break;
      }
      if (op == 0x12B) {
        if (clamp || omod || op_sel || (neg & 4) || (abs & 4)) {
          gpu::gcn::WarnUnsupported("vop3.fmac-modifier", op, w, w1);
          break;
        }
        t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450Fma,
                                   {t.SrcF(s0, inst.literal, neg & 1, abs & 1),
                                    t.SrcF(s1, inst.literal, neg & 2, abs & 2),
                                    t.VgF(vdst)}));
        break;
      }
      if (rdna_op >= 0x100 && rdna_op < 0x140 && rdna_op != 0x101 &&
          !RdnaSharedVop2(rdna_op - 0x100)) {
        gpu::gcn::WarnUnsupported("vop3.alias.rdna", rdna_op, w, w1);
        break;
      }
      if (op == 0x141 || (op >= 0x161 && op <= 0x163) || op == 0x16b) {
        gpu::gcn::WarnUnsupported("vop3.opcode.rdna", op, w, w1);
        break;
      }
      const bool supported_compare = op <= 0x1F || (op >= 0x80 && op <= 0x87) ||
                                     (op >= 0x90 && op <= 0x97) ||
                                     (op >= 0xC0 && op <= 0xC7) ||
                                     (op >= 0xD0 && op <= 0xD7);
      if (op < 0x100 && omod) {
        gpu::gcn::WarnUnsupported("vopc.omod", op, w, w1);
        break;
      }
      if (op < 0x100 && !supported_compare) {
        gpu::gcn::WarnUnsupported("vop3.compare.rdna", op, w, w1);
        break;
      }
      if (op < 0x100 && (op & 0x10)) {
        gpu::gcn::EmitVopc(t, op, t.SrcF(s0, inst.literal, neg & 1, abs & 1),
                           t.SrcF(s1, inst.literal, neg & 2, abs & 2),
                           t.SrcRaw(s0, inst.literal),
                           t.SrcRaw(s1, inst.literal), 126);
        break;
      }
      if (op == 0x16f) {
        if (clamp || omod || op_sel) {
          gpu::gcn::WarnUnsupported("vop3.div-fmas-modifier", op, w, w1);
          break;
        }
        const Id fma =
            t.m.ExtInst(t.t_f, GLSLstd450Fma,
                        {t.SrcF(s0, inst.literal, neg & 1, abs & 1),
                         t.SrcF(s1, inst.literal, neg & 2, abs & 2),
                         t.SrcF(s2, inst.literal, neg & 4, abs & 4)});
        t.SetVgF(vdst, t.SelectF(t.IsNonZero(t.Sg(106)),
                                 t.FMul(fma, t.F32(4294967296.0f)), fma));
        break;
      }
      Id source0 = t.SrcF(s0, inst.literal, neg & 1, abs & 1);
      if (op == 0x18b) {
        source0 = t.m.CompositeExtract(
            t.t_f,
            t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                        {RdnaF16Bits(t, s0, inst.literal)}),
            0);
        if (abs & 1)
          source0 = t.Ext1(GLSLstd450FAbs, source0);
        if (neg & 1)
          source0 = t.FNeg(source0);
      }
      gpu::gcn::EmitVop3(
          t, op, vdst, source0, t.SrcRawHi(s0, inst.literal, op == 0x163),
          t.SrcF(s1, inst.literal, neg & 2, abs & 2),
          t.SrcF(s2, inst.literal, neg & 4, abs & 4),
          t.SrcRawHi(s2, inst.literal, op == 0x177), sdst, clamp, omod);
      break;
    }
    case Enc::kVopc: {
      const uint32_t op = inst.opcode;
      uint32_t vsrc1 = (w >> 9) & 0xFF;
      const uint32_t raw0 = w & 0x1FF;
      uint32_t src0, lit;
      ResolveValuSrc0(inst, raw0, src0, lit);
      if (inst.extension == gpu::gcn::InstExtension::kSdwa) {
        gpu::gcn::WarnUnsupported("vopc.sdwa", op, w, w1);
        break;
      }
      uint32_t vopc_src1 = 256 + vsrc1;
      if (raw0 == 249)
        vopc_src1 = DecodeSdwa(inst, vsrc1, false).src1;
      // RDNA2 f16 compares sit at 0xC8-0xCF, which the GFX7-numbered EmitVopc
      // would read as u32 integer compares; run the float predicate (op-0xC8)
      // on the low-half f16 operands instead. u32 compares stay at 0xC0-0xC7.
      if (op >= 0xC8 && op <= 0xCF) {
        auto f16 = [&](uint32_t f) {
          return t.m.CompositeExtract(
              t.t_f,
              t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16,
                          {RdnaF16Bits(t, f, lit)}),
              0);
        };
        gpu::gcn::EmitVopc(t, op - 0xC8, f16(src0), f16(vopc_src1),
                           t.SrcRaw(src0, lit), t.SrcRaw(vopc_src1, lit));
        break;
      }
      const bool supported = op <= 0x1F || (op >= 0x80 && op <= 0x87) ||
                             (op >= 0x90 && op <= 0x97) ||
                             (op >= 0xC0 && op <= 0xC7) ||
                             (op >= 0xD0 && op <= 0xD7);
      if (!supported) {
        gpu::gcn::WarnUnsupported("vopc.rdna", op, w, w1);
        break;
      }
      gpu::gcn::EmitVopc(t, op, t.SrcF(src0, lit), t.SrcF(vopc_src1, lit),
                         t.SrcRaw(src0, lit), t.SrcRaw(vopc_src1, lit),
                         (op & 0x10) ? 126 : 106);
      break;
    }
    case Enc::kVintrp: {
      if (!sc.is_ps) {
        gpu::gcn::WarnUnsupported("vintrp.stage", inst.opcode, w, w1);
        break;
      }
      const uint32_t chan = (w >> 8) & 3, attr = (w >> 10) & 0x3F;
      const uint32_t op = (w >> 16) & 3, vdst = (w >> 18) & 0xFF;
      if (op == 1 || (op == 2 && (w & 0xFF) == 2)) {
        const Id var = gpu::gcn::PsInputVar(t, sc, attr);
        const Id p_in_f = t.m.TypePointer(spv::StorageClass::Input, t.t_f);
        t.SetVgF(vdst,
                 t.m.Load(t.t_f, t.m.AccessChain(p_in_f, var, {t.U32(chan)})));
      } else if (op != 0) {
        gpu::gcn::WarnUnsupported("vintrp.rdna", op, w, w1);
      }
      break;
    }
    case Enc::kExp:
      EmitExport(t, inst, sc);
      break;
    case Enc::kMtbuf:
    case Enc::kMubuf: {
      if (inst.opcode >
          0x03) {  // stores / raw loads: not used by Isaac's VS/PS
        gpu::gcn::WarnUnsupported(
            inst.enc == Enc::kMtbuf ? "mtbuf.rdna" : "mubuf.rdna", inst.opcode,
            w, w1);
        break;
      }
      const bool tfe = (w1 >> 23) & 1;
      const bool lds = inst.enc == Enc::kMubuf && ((w >> 16) & 1);
      const uint32_t soffset = (w1 >> 24) & 0xff;
      if (tfe || lds || (soffset != 125 && soffset != 128)) {
        gpu::gcn::WarnUnsupported("buffer.control.rdna", inst.opcode, w, w1);
        break;
      }
      // A per-vertex fetch was lifted to a Location vertex input; its pc has a
      // seed. Re-seed its destination VGPRs from that input HERE (where the
      // real buffer_load_format runs), overwriting any value the merged-wave
      // index math clobbered them with (e.g. v0/v3/v4, reused as the fetch
      // index). A CONSTANT load (e.g. the 2D ortho matrix) has no seed and
      // reads num_comps dwords from the bound UBO at the computed byte offset
      // into the destination VGPRs.
      if (auto sit = sc.vfetch_seed.find(inst.pc);
          sit != sc.vfetch_seed.end()) {
        const auto& vs = sit->second;
        const Id comp_ty = vs.num_comps == 1   ? t.t_f
                           : vs.num_comps == 2 ? t.t_v2
                           : vs.num_comps == 3 ? t.t_v3
                                               : t.t_v4;
        const Id val = t.m.Load(comp_ty, vs.in_var);
        for (uint32_t c = 0; c < vs.num_comps; c++)
          t.SetVgF(vs.dest_vgpr + c, vs.num_comps == 1
                                         ? val
                                         : t.m.CompositeExtract(t.t_f, val, c));
        break;
      }
      if (inst.enc == Enc::kMubuf) {
        gpu::gcn::WarnUnsupported("mubuf.format-conversion.rdna", inst.opcode,
                                  w, w1);
        break;
      }
      const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
      uint32_t binding;
      auto pit = sc.mubuf_cbuf_by_pc.find(inst.pc);
      if (pit != sc.mubuf_cbuf_by_pc.end()) {
        binding = pit->second;
      } else {
        auto it = sc.cbuf_bind.find(srsrc);
        if (it == sc.cbuf_bind.end()) {
          gpu::gcn::WarnUnsupported("mubuf.cbuf-unplanned", inst.opcode, w, w1);
          break;
        }
        binding = it->second;
      }
      const uint32_t nc = (inst.opcode & 3) + 1;
      // A typed constant load delivers the buffer's dwords unchanged only when
      // its format is 32-bit per channel; narrower ones would need unpacking.
      if (inst.enc == Enc::kMtbuf) {
        const uint32_t fmt = (w >> 19) & 0x7F;
        if (!(fmt >= 20 && (fmt <= 22 || (fmt >= 62 && fmt <= 77))))
          gpu::gcn::WarnUnsupported("mtbuf.fmt", fmt, w, w1);
      }
      const uint32_t inst_offset = w & 0xFFF, vdata = (w1 >> 8) & 0xFF;
      const bool idxen = (w >> 13) & 1, offen = (w >> 12) & 1;
      if (idxen) {
        gpu::gcn::WarnUnsupported("mtbuf.descriptor-stride.rdna", inst.opcode,
                                  w, w1);
        break;
      }
      const uint32_t vaddr = w1 & 0xFF;
      // byte offset = inst_offset + index*stride + voffset. The V# stride is
      // not available in the shader (the descriptor SGPRs are not seeded), so
      // an indexed uniform row-select assumes tight packing (stride == nc*4);
      // an unindexed load uses the immediate offset directly.
      Id byte_off = t.U32(inst_offset);
      if (idxen)
        byte_off = t.Add(byte_off, t.Mul(t.Vg(vaddr), t.U32(nc * 4)));
      if (offen)
        byte_off = t.Add(byte_off, t.Vg(vaddr + (idxen ? 1u : 0u)));
      const Id dword0 = t.Shr(byte_off, t.U32(2));
      for (uint32_t k = 0; k < nc; k++)
        t.SetVg(vdata + k, t.CbufDwordId(binding, t.Add(dword0, t.U32(k))));
      break;
    }
    case Enc::kMimg: {
      if (inst.opcode == 0x09 || inst.opcode == 0x21 || inst.opcode == 0x25) {
        gpu::gcn::WarnUnsupported("mimg.rdna-modifier", inst.opcode, w, w1);
        break;
      }
      const bool a16 = (w1 >> 30) & 1, d16 = (w1 >> 31) & 1;
      const bool tfe = (w >> 16) & 1, lwe = (w >> 17) & 1;
      const bool unrm = (w >> 12) & 1;
      if (a16 || d16 || tfe || lwe || unrm) {
        gpu::gcn::WarnUnsupported("mimg.control.rdna", inst.opcode, w, w1);
        break;
      }
      Inst lowered = inst;
      const uint32_t dim = (w >> 3) & 0x7;
      if (dim != 1 && dim != 5) {
        gpu::gcn::WarnUnsupported("mimg.dim", dim, w, w1);
        break;
      }
      if (((w >> 15) & 1) && dim == 5) {
        gpu::gcn::WarnUnsupported("mimg.r128-array.rdna", inst.opcode, w, w1);
        break;
      }
      lowered.raw[0] = (w & ~0x4000u) | (dim == 5 ? 0x4000u : 0u);
      // NSA names every address component explicitly. Load them before emitting
      // the image operation so aliases cannot overwrite a later source and no
      // architectural VGPRs are used as scratch storage.
      const uint32_t nsa = (w >> 1) & 0x3;
      if (nsa) {
        std::array<Id, 13> address{};
        address[0] = t.Vg(w1 & 0xFF);
        for (uint32_t d = 0; d < nsa; d++)
          for (uint32_t c = 0; c < 4; c++)
            address[1 + d * 4 + c] = t.Vg((inst.raw[2 + d] >> (c * 8)) & 0xFF);
        gpu::gcn::EmitMimg(t, lowered, sc, address.data());
        break;
      }
      gpu::gcn::EmitMimg(t, lowered, sc);
      break;
    }
    case Enc::kFlat:
      gpu::gcn::WarnUnsupported("flat.rdna", inst.opcode, w, w1);
      break;
    default:
      gpu::gcn::WarnUnsupported("rdna", inst.opcode, w, w1);
      break;
  }
}

// ---- control flow (self-contained copy of the GFX7 while/switch lowering) ---
// 0=none, 1=uncond, 2=scc0, 3=scc1, 4=vccz, 5=vccnz, 6=execz, 7=execnz,
// 8=endpgm.
int BranchKind(const Inst& inst) {
  if (inst.enc != Enc::kSopp)
    return 0;
  switch (inst.opcode) {
    case 0x01:
    case 0x1B:  // s_endpgm_saved
    case 0x1E:  // s_endpgm_ordered_ps_done
    case 0x1F:  // s_code_end
      return 8;
    case 0x02:
      return 1;
    case 0x04:
      return 2;
    case 0x05:
      return 3;
    case 0x06:
      return 4;
    case 0x07:
      return 5;
    case 0x08:
      return 6;
    case 0x09:
      return 7;
    default:
      return 0;
  }
}

bool HasControlFlow(const Program& program) {
  return std::any_of(program.begin(), program.end(), [](const Inst& inst) {
    const int k = BranchKind(inst);
    return k >= 1 && k <= 7;
  });
}

Id BranchTaken(Translator& t, int kind) {
  switch (kind) {
    case 2:
      return t.IsZero(t.Scc());
    case 3:
      return t.IsNonZero(t.Scc());
    case 4:
      return t.IsZero(t.Sg(106));
    case 5:
      return t.IsNonZero(t.Sg(106));
    case 6:
      return t.IsZero(t.Exec());
    case 7:
      return t.IsNonZero(t.Exec());
    default:
      return t.m.ConstBool(false);
  }
}

std::vector<uint32_t> BlockStarts(const Program& program, uint32_t max_pc) {
  std::vector<uint32_t> leaders{0};
  for (const Inst& inst : program) {
    const int k = BranchKind(inst);
    if (k == 0)
      continue;
    leaders.push_back(inst.pc + inst.size);
    if (k >= 1 && k <= 7) {
      const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
      leaders.push_back(static_cast<uint32_t>(static_cast<int32_t>(inst.pc) +
                                              static_cast<int32_t>(inst.size) +
                                              simm));
    }
  }
  std::sort(leaders.begin(), leaders.end());
  leaders.erase(std::unique(leaders.begin(), leaders.end()), leaders.end());
  std::vector<uint32_t> starts;
  for (uint32_t l : leaders)
    if (l < max_pc)
      starts.push_back(l);
  return starts;
}

void EmitCfg(Translator& t, const Program& program, StageContext& sc) {
  const uint32_t max_pc =
      program.empty() ? 0 : program.back().pc + program.back().size;
  const std::vector<uint32_t> starts = BlockStarts(program, max_pc);
  const uint32_t num_blocks = static_cast<uint32_t>(starts.size());
  const uint32_t kExit = num_blocks;
  const auto block_of = [&](uint32_t pc) -> uint32_t {
    if (pc >= max_pc)
      return kExit;
    uint32_t b = 0;
    for (uint32_t i = 0; i < num_blocks; i++)
      if (starts[i] <= pc)
        b = i;
      else
        break;
    return b;
  };

  const Id header = t.m.NewBlock(), dispatch = t.m.NewBlock();
  const Id merge_sel = t.m.NewBlock();
  const Id cont = t.m.NewBlock(), merge = t.m.NewBlock();
  const Id exit_blk = t.m.NewBlock();
  std::vector<Id> case_labels(num_blocks);
  for (Id& l : case_labels)
    l = t.m.NewBlock();

  t.SetState(0);
  t.m.Branch(header);
  t.m.OpenBlock(header);
  t.m.LoopMerge(merge, cont);
  t.m.Branch(dispatch);
  t.m.OpenBlock(dispatch);
  const Id state = t.State();
  t.m.SelectionMerge(merge_sel);
  std::vector<std::pair<uint32_t, Id> > cases;
  for (uint32_t i = 0; i < num_blocks; i++)
    cases.push_back({i, case_labels[i]});
  t.m.Switch(state, exit_blk, cases);

  for (uint32_t bi = 0; bi < num_blocks; bi++) {
    t.m.OpenBlock(case_labels[bi]);
    const uint32_t blk_start = starts[bi];
    const uint32_t blk_end = (bi + 1 < num_blocks) ? starts[bi + 1] : max_pc;
    bool terminated = false;
    for (const Inst& inst : program) {
      if (inst.pc < blk_start || inst.pc >= blk_end)
        continue;
      const int k = BranchKind(inst);
      if (k == 0) {
        RdnaEmitInst(t, inst, sc);
        continue;
      }
      const uint32_t fall = (bi + 1 < num_blocks) ? bi + 1 : kExit;
      if (k == 8) {
        t.SetState(kExit);
      } else if (k == 1) {
        const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
        t.SetState(block_of(
            static_cast<uint32_t>(static_cast<int32_t>(inst.pc) +
                                  static_cast<int32_t>(inst.size) + simm)));
      } else {
        const int32_t simm = static_cast<int16_t>(inst.raw[0] & 0xFFFF);
        const uint32_t target = block_of(
            static_cast<uint32_t>(static_cast<int32_t>(inst.pc) +
                                  static_cast<int32_t>(inst.size) + simm));
        t.SetStateId(t.SelectB(BranchTaken(t, k), t.U32(target), t.U32(fall)));
      }
      terminated = true;
      break;
    }
    if (!terminated)
      t.SetState((bi + 1 < num_blocks) ? bi + 1 : kExit);
    t.m.Branch(merge_sel);
  }
  t.m.OpenBlock(exit_blk);
  t.m.Branch(merge);
  t.m.OpenBlock(merge_sel);
  t.m.Branch(cont);
  t.m.OpenBlock(cont);
  t.m.Branch(header);
  t.m.OpenBlock(merge);
}

bool ForceCfg() {
  static const bool force = std::getenv("DELTA_GPU_SPIRV_CFG") != nullptr;
  return force;
}

void EmitBody(Translator& t, const Program& program, StageContext& sc) {
  t.SeedExec();
  t.predicate_vector = true;
  if (ForceCfg() || HasControlFlow(program)) {
    EmitCfg(t, program, sc);
    return;
  }
  for (const Inst& inst : program) {
    if (inst.enc == Enc::kSopp && (inst.opcode == 0x01 || inst.opcode == 0x1B ||
                                   inst.opcode == 0x1E || inst.opcode == 0x1F))
      break;
    RdnaEmitInst(t, inst, sc);
  }
}

// ---- vertex fetch -----------------------------------------------------------
// Best-effort parse of a Gnm/AGC fetch sub-shader (RDNA2 encoding):
// s_load_dwordx4 of a V# table + buffer_load_format into destination VGPRs, one
// per attribute. PS5 NGG shaders often fetch inline instead; then no attributes
// are recovered and the VS seeds from VertexIndex/InstanceIndex (procedural
// path).
struct FetchAttr {
  uint32_t semantic, num_comps, dest_vgpr, table_sgpr, dword_off;
  uint32_t pc =
      ~0u;  // inline fetch MUBUF pc (~0 = standalone fetch sub-shader)
  uint32_t inst_format = 0;  // typed (MTBUF) fetch format; 0 = take the V#'s
};

// Scan an instruction stream for the s_load_dwordx4(V# table) +
// buffer_load_format fetch pattern. Works on both a stand-alone fetch
// sub-shader and the head of an NGG vertex program that fetches inline (PS5 AGC
// frequently inlines the fetch), so buffer_load_format never reaches
// RdnaEmitInst as an unsupported op.
std::vector<FetchAttr> ParseFetchInsts(const Program& insts) {
  std::vector<FetchAttr> out;
  uint32_t sem = 0;  // dense vertex-input location (per-vertex fetches only)
  const auto chained_loads = MapTableChainedLoads(insts);
  for (const Inst& in : insts) {
    if (in.enc == Enc::kSop1 && in.opcode == 0x20)
      break;  // s_setpc_b64 (return)
    if (in.enc == Enc::kSopp && in.opcode == 1)
      break;  // s_endpgm
    // buffer_load_format_* (untyped, format from the V#) and its typed twin
    // tbuffer_load_format_* (MTBUF, format in the instruction). Skyrim's world
    // geometry fetches positions with the typed form.
    const bool typed = in.enc == Enc::kMtbuf;
    if ((in.enc != Enc::kMubuf && !typed) || in.opcode > 0x03)
      continue;
    const uint32_t vdata = (in.raw[1] >> 8) & 0xFF;
    const uint32_t srsrc = ((in.raw[1] >> 16) & 0x1F) * 4;
    const uint32_t nc = (in.opcode & 3) + 1;
    const bool idxen = (in.raw[0] >> 13) & 1, offen = (in.raw[0] >> 12) & 1;
    const uint32_t vaddr = in.raw[1] & 0xFF, soffset = (in.raw[1] >> 24) & 0xFF;
    // If srsrc's V# was loaded via s_load from a user_data pointer, this fetch
    // reads a TABLE of V#s at that pointer: table_sgpr = the s_load's sbase and
    // this attribute is that s_load's 16-byte (4-dword) entry. Otherwise the V#
    // is inline in user data at srsrc (table_sgpr = srsrc, dword_off = 0).
    const auto chain = chained_loads.find(in.pc);
    const bool chained = chain != chained_loads.end();
    const uint32_t table_sgpr = chained ? chain->second.first : srsrc;
    const uint32_t dword_off = chained ? chain->second.second * 4 : 0;
    const bool vtx = BufLoadIsVertexFetch(in, chained);
    if (ShDbg())
      std::fprintf(
          stderr,
          "[gcnspv] buf_load nc=%u vdst=v%u srsrc=s%u idxen=%u offen=%u "
          "vaddr=v%u soffset=s%u -> %s (table_sgpr=s%u doff=%u)\n",
          nc, vdata, srsrc, idxen, offen, vaddr, soffset,
          vtx ? "vertex-attr" : "const-ubo", table_sgpr, dword_off);
    // Only a genuine per-vertex fetch becomes a vertex input. A constant load
    // is left for the UBO path (RdnaPlanBufLoadCbufs assigns the same table
    // slots).
    if (vtx) {
      out.push_back({sem, nc, vdata, table_sgpr, dword_off, in.pc,
                     typed ? ((in.raw[0] >> 19) & 0x7F) : 0u});
      sem++;
    }
  }
  return out;
}

std::vector<FetchAttr> ParseFetch(uint64_t fetch_addr) {
  constexpr uint64_t kMaxFetchBytes = 256 * sizeof(uint32_t);
  if (!gpu::gcn::InGuest(fetch_addr) ||
      !gpu::IsReadableRange(fetch_addr, kMaxFetchBytes))
    return {};
  const auto* code = reinterpret_cast<const uint32_t*>(fetch_addr);
  std::vector<FetchAttr> attrs = ParseFetchInsts(Decode(code, 256));
  for (FetchAttr& attr : attrs)
    attr.pc = ~0u;
  return attrs;
}

// Address of the VS being translated, for shader-specific debug knobs.
thread_local uint64_t g_vs_addr = 0;

// ---- VS / PS drivers --------------------------------------------------------
Id DeclareUserData(Translator& t) {
  const Id words = t.m.TypeArray(t.t_u, 32);
  t.m.Decorate(words, spv::Decoration::ArrayStride, {4});
  const Id block = t.m.TypeStruct({words});
  t.m.Decorate(block, spv::Decoration::Block);
  t.m.MemberDecorate(block, 0, spv::Decoration::Offset, {0});
  return t.m.Variable(t.m.TypePointer(spv::StorageClass::PushConstant, block),
                      spv::StorageClass::PushConstant);
}

void SeedUserData(Translator& t,
                  Id user_data,
                  uint32_t sgpr_base,
                  uint32_t count) {
  const Id p_u = t.m.TypePointer(spv::StorageClass::PushConstant, t.t_u);
  for (uint32_t i = 0; i < std::min(count, 32u); i++)
    t.SetSg(
        sgpr_base + i,
        t.m.Load(t.t_u, t.m.AccessChain(p_u, user_data, {t.U32(0), t.U32(i)})));
}

bool TranslateVs(const Program& program,
                 const uint32_t* vs_user_data,
                 const std::unordered_set<uint32_t>& flat_attrs,
                 Recompiled& r,
                 Translator& t,
                 bool gl_clip_space,
                 uint32_t user_sgprs) {
  if (ShDbg())
    DumpProgram(program, "vs");
  const uint64_t fetch =
      user_sgprs >= 2
          ? (static_cast<uint64_t>(vs_user_data[1] & 0xFFFF) << 32) |
                vs_user_data[0]
          : 0;
  // Prefer a stand-alone fetch sub-shader; otherwise recover the fetch that the
  // NGG vertex program does inline (buffer_load_format in its own body). Either
  // way each attribute becomes a Location vertex input (RdnaEmitInst then
  // treats the inline buffer_load_format as a no-op) and the renderer binds the
  // real vertex buffers from r.attrs.
  std::vector<FetchAttr> attrs = ParseFetch(fetch);
  if (attrs.empty())
    attrs = ParseFetchInsts(program);
  if (ShDbg())
    for (const FetchAttr& a : attrs)
      std::fprintf(stderr, "[gcnspv] vs attr loc=%u nc=%u vgpr=%u pc=%#x\n",
                   a.semantic, a.num_comps, a.dest_vgpr, a.pc);
  t.InitTypes();

  std::vector<Id> iface;
  const Id pos_out =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_v4),
                   spv::StorageClass::Output);
  t.m.Decorate(pos_out, spv::Decoration::BuiltIn,
               {static_cast<uint32_t>(spv::BuiltIn::Position)});
  iface.push_back(pos_out);

  const Id user_data = DeclareUserData(t);
  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  SeedUserData(t, user_data, 8, user_sgprs);

  // NGG merged-wave prologue: the VS derives its EXEC/lane bookkeeping from
  // merged_wave_info in s3 (verts-in-wave [7:0], prims [15:8]); model a
  // 1-vert/1-prim wave so that math yields a live lane instead of zeros.
  t.SetSg(3, t.U32(0x0101));

  std::unordered_map<uint32_t, StageContext::VfetchSeed> vfetch_seed;
  Id vertex_index = 0;
  if (attrs.empty()) {  // procedural VS: seed the ABI VGPRs from Vulkan
                        // built-ins
    const Id p_in_u = t.m.TypePointer(spv::StorageClass::Input, t.t_u);
    vertex_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
    const Id instance_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
    t.m.Decorate(vertex_index, spv::Decoration::BuiltIn,
                 {static_cast<uint32_t>(spv::BuiltIn::VertexIndex)});
    t.m.Decorate(instance_index, spv::Decoration::BuiltIn,
                 {static_cast<uint32_t>(spv::BuiltIn::InstanceIndex)});
    iface.push_back(vertex_index);
    iface.push_back(instance_index);
    t.SetVg(0, t.m.Load(t.t_u, vertex_index));
    const Id instance = t.m.Load(t.t_u, instance_index);
    t.SetVg(1, instance);
    t.SetVg(3, instance);
  }

  Id first_attr_var = 0;
  uint32_t first_attr_comps = 0;
  for (const FetchAttr& a : attrs) {
    const Id comp_ty = a.num_comps == 1   ? t.t_f
                       : a.num_comps == 2 ? t.t_v2
                       : a.num_comps == 3 ? t.t_v3
                                          : t.t_v4;
    const Id in_var =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, comp_ty),
                     spv::StorageClass::Input);
    t.m.Decorate(in_var, spv::Decoration::Location, {a.semantic});
    t.m.Name(in_var, "v_attr" + std::to_string(a.semantic));
    iface.push_back(in_var);
    const Id val = t.m.Load(comp_ty, in_var);
    for (uint32_t c = 0; c < a.num_comps; c++) {
      const Id comp =
          a.num_comps == 1 ? val : t.m.CompositeExtract(t.t_f, val, c);
      t.SetVgF(a.dest_vgpr + c, comp);
    }
    // An inline fetch is a no-op in the body, but the merged-wave index math
    // can clobber its destination VGPRs (e.g. v0) between here and the
    // transform, so re-seed them at the fetch's pc where the real
    // buffer_load_format would run.
    if (a.pc != ~0u)
      vfetch_seed[a.pc] = {in_var, a.dest_vgpr, a.num_comps};
    if (!first_attr_var) {
      first_attr_var = in_var;
      first_attr_comps = a.num_comps;
    }
    r.attrs.push_back({a.semantic, a.num_comps, a.table_sgpr, a.dword_off,
                       false, a.inst_format, a.pc});
  }

  StageContext sc;
  sc.r = &r;
  sc.iface = &iface;
  sc.main_fn = main_fn;
  sc.pos_out = pos_out;
  sc.flat_attrs = &flat_attrs;
  sc.vfetch_seed = std::move(vfetch_seed);
  sc.skip_launch_movs = LaunchExecMovPcs(program);
  if (!RdnaPlanCbufs(program, 0, r.vs_cbufs, sc.cbuf_bind, sc.smem_cbuf_by_pc))
    return false;
  // Constant buffer_load descriptors (e.g. the ortho matrix a procedural 2D VS
  // reads) become additional set-1 UBOs after the SMEM cbufs.
  RdnaPlanBufLoadCbufs(program, 0, r.vs_cbufs, sc.cbuf_bind,
                       sc.mubuf_cbuf_by_pc);
  if (ShDbg())
    std::fprintf(stderr, "[gcnspv] vs planned %zu cbufs\n", r.vs_cbufs.size());
  // DELTA_GPU_DBGPOS=<vs address>[:<dword offset>]: recompute this one shader's
  // position export from its position input and a 4x4 transform in its cbuffer
  // (address 0 = every shader; the offset picks which matrix in the window).
  static const uint64_t dbg_pos_vs = [] {
    const char* e = std::getenv("DELTA_GPU_DBGPOS");
    return e ? std::strtoull(e, nullptr, 0) : ~0ull;
  }();
  static const uint32_t dbg_pos_off = [] {
    const char* e = std::getenv("DELTA_GPU_DBGPOS");
    const char* c = e ? std::strchr(e, ':') : nullptr;
    return c ? (uint32_t)std::strtoul(c + 1, nullptr, 0) : 0u;
  }();
  // ...:<world binding> applies a 4x3 world matrix from that binding first, so
  // the probe can reproduce a world-then-view-projection chain.
  static const int dbg_pos_world = [] {
    const char* e = std::getenv("DELTA_GPU_DBGPOS");
    const char* c = e ? std::strchr(e, ':') : nullptr;
    const char* c2 = c ? std::strchr(c + 1, ':') : nullptr;
    return c2 ? std::atoi(c2 + 1) : -1;
  }();
  if (dbg_pos_vs == 0 || dbg_pos_vs == g_vs_addr) {
    sc.dbg_pos_in = first_attr_var;
    sc.dbg_pos_comps = first_attr_comps;
    for (const auto& cb : r.vs_cbufs)
      if (cb.num_dwords >= 16) {
        sc.dbg_pos_cbuf = static_cast<int>(cb.binding);
        break;
      }
  }

  EmitBody(t, program, sc);
  r.num_params = sc.max_param;

  // DELTA_GPU_FORCEQUAD: ignore the VS's computed position and emit a
  // full-screen quad straight from gl_VertexIndex (0..3 -> the four NDC
  // corners). If this renders (with the PS forced white) the raster/PS pipeline
  // is sound and the real position math is the culprit (e.g. an unbound MVP ->
  // degenerate); if it still draws nothing the problem is downstream
  // (index/vertex_count/render target).
  static const bool force_quad = std::getenv("DELTA_GPU_FORCEQUAD") != nullptr;
  if (force_quad) {
    // A fetch-path VS never created a VertexIndex input, and its v0 is
    // clobbered by the vertex fetch -- bind a real VertexIndex here so the quad
    // is valid either way.
    Id vidx_var = vertex_index;
    if (!vidx_var) {
      vidx_var = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_u),
                              spv::StorageClass::Input);
      t.m.Decorate(vidx_var, spv::Decoration::BuiltIn,
                   {static_cast<uint32_t>(spv::BuiltIn::VertexIndex)});
      iface.push_back(vidx_var);
    }
    const Id vidx = t.m.Load(t.t_u, vidx_var);
    const Id fx =
        t.SelectF(t.IsNonZero(t.And(vidx, t.U32(1))), t.F32(1.f), t.F32(-1.f));
    const Id fy =
        t.SelectF(t.IsNonZero(t.And(vidx, t.U32(2))), t.F32(1.f), t.F32(-1.f));
    t.m.Store(pos_out,
              t.m.CompositeConstruct(t.t_v4, {fx, fy, t.F32(0.f), t.F32(1.f)}));
  }

  // DELTA_GPU_RAWPOS=<scale>: bypass the shader's transform and emit the raw
  // location-0 vertex input scaled into NDC. Separates "the vertex inputs never
  // reach the shader" from "the transform math is wrong": FORCEQUAD proves the
  // raster path but uses no vertex data at all, this uses the real attribute.
  static const float raw_pos = [] {
    const char* e = std::getenv("DELTA_GPU_RAWPOS");
    return e ? static_cast<float>(std::atof(e)) : 0.f;
  }();
  if (raw_pos != 0.f && first_attr_var && first_attr_comps >= 2) {
    const Id comp_ty = first_attr_comps == 2   ? t.t_v2
                       : first_attr_comps == 3 ? t.t_v3
                                               : t.t_v4;
    const Id val = t.m.Load(comp_ty, first_attr_var);
    t.m.Store(pos_out,
              t.m.CompositeConstruct(
                  t.t_v4,
                  {t.FMul(t.m.CompositeExtract(t.t_f, val, 0), t.F32(raw_pos)),
                   t.FMul(t.m.CompositeExtract(t.t_f, val, 1), t.F32(raw_pos)),
                   t.F32(0.f), t.F32(1.f)}));
  }

  // DELTA_GPU_POSPROBE: squash the shader's own clip position into the viewport
  // (x/(1+|x|) after the perspective divide). Any FINITE position then covers
  // pixels, so a still-black target means w is zero/NaN rather than the
  // geometry merely landing off-screen.
  static const bool pos_probe = std::getenv("DELTA_GPU_POSPROBE") != nullptr;
  if (pos_probe) {
    const Id p_out_f0 = t.m.TypePointer(spv::StorageClass::Output, t.t_f);
    const Id px =
        t.m.Load(t.t_f, t.m.AccessChain(p_out_f0, pos_out, {t.U32(0)}));
    const Id py =
        t.m.Load(t.t_f, t.m.AccessChain(p_out_f0, pos_out, {t.U32(1)}));
    const Id pw =
        t.m.Load(t.t_f, t.m.AccessChain(p_out_f0, pos_out, {t.U32(3)}));
    auto squash = [&](Id v) {
      const Id q = t.m.Emit(spv::Op::OpFDiv, t.t_f, {v, pw});
      const Id a = t.m.ExtInst(t.t_f, GLSLstd450FAbs, {q});
      return t.m.Emit(spv::Op::OpFDiv, t.t_f,
                      {q, t.m.Emit(spv::Op::OpFAdd, t.t_f, {t.F32(1.f), a})});
    };
    t.m.Store(pos_out,
              t.m.CompositeConstruct(
                  t.t_v4, {squash(px), squash(py), t.F32(0.f), t.F32(1.f)}));
  }

  // Clip-space convention, from PA_CL_CLIP_CNTL.DX_CLIP_SPACE_DEF. In DX mode
  // the shader already exports z in [0,w], exactly what Vulkan wants; remapping
  // it there squeezes depth into the far half AND lets geometry behind the near
  // plane survive clipping (Skyrim drew its world from inside itself). In GL
  // mode z spans [-w,w] and has to be remapped or everything is clipped away
  // (Isaac renders nothing without it).
  if (gl_clip_space) {
    const Id p_out_f = t.m.TypePointer(spv::StorageClass::Output, t.t_f);
    const Id z_ptr = t.m.AccessChain(p_out_f, pos_out, {t.U32(2)});
    const Id w_ptr = t.m.AccessChain(p_out_f, pos_out, {t.U32(3)});
    const Id z = t.m.Load(t.t_f, z_ptr), wv = t.m.Load(t.t_f, w_ptr);
    t.m.Store(z_ptr, t.FMul(t.FAdd(z, wv), t.F32(0.5f)));
  }

  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Vertex, main_fn, "main", iface);
  return true;
}

// SPI_PS_INPUT_ENA lays the PS's system input VGPRs into v0.. in ascending bit
// order (each set bit consumes the width below). The v_interp i/j barycentrics
// come from the first pair, which the Vulkan-interpolated Location inputs
// model, so those VGPRs need no value. But a shader that reads screen position
// directly (a 2D clip such as frag.x*a + frag.y*b < c) reads the POS_{X,Y,Z,W}
// VGPRs; unseeded they stay zero, collapsing the clip to "0 < c" and discarding
// every fragment. Seed those from gl_FragCoord (and FRONT_FACE from
// gl_FrontFacing).
void SeedPsInputVgprs(Translator& t, uint32_t ena, std::vector<Id>& iface) {
  if (!ena)
    return;
  static const uint8_t width[16] = {2, 2, 2, 3, 2, 2, 2, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1};
  uint32_t vg[16] = {}, next = 0;
  for (uint32_t b = 0; b < 16; b++)
    if (ena & (1u << b)) {
      vg[b] = next;
      next += width[b];
    }

  if ((ena >> 8) & 0xF) {  // POS_X..W at bits 8..11 -> gl_FragCoord.xyzw
    const Id fc =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_v4),
                     spv::StorageClass::Input);
    t.m.Decorate(fc, spv::Decoration::BuiltIn,
                 {static_cast<uint32_t>(spv::BuiltIn::FragCoord)});
    iface.push_back(fc);
    const Id v = t.m.Load(t.t_v4, fc);
    for (uint32_t c = 0; c < 4; c++)
      if (ena & (1u << (8 + c)))
        t.SetVgF(vg[8 + c], t.m.CompositeExtract(t.t_f, v, c));
  }
  if (ena & (1u << 12)) {  // FRONT_FACE -> 0xffffffff front / 0 back
    const Id ff =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_bool),
                     spv::StorageClass::Input);
    t.m.Decorate(ff, spv::Decoration::BuiltIn,
                 {static_cast<uint32_t>(spv::BuiltIn::FrontFacing)});
    iface.push_back(ff);
    t.SetVg(vg[12],
            t.SelectB(t.m.Load(t.t_bool, ff), t.U32(0xFFFFFFFFu), t.U32(0)));
  }
}

bool TranslatePs(const Program& program,
                 const std::unordered_set<uint32_t>& flat_attrs,
                 uint32_t ps_input_ena,
                 Recompiled& r,
                 Translator& t,
                 uint32_t user_sgprs) {
  if (ShDbg())
    DumpProgram(program, "ps");
  std::vector<Id> iface;
  StageContext sc;
  sc.is_ps = true;
  sc.r = &r;
  sc.iface = &iface;
  sc.flat_attrs = &flat_attrs;
  sc.skip_launch_movs = LaunchExecMovPcs(program);
  if (!RdnaPlanCbufs(program, static_cast<uint32_t>(r.vs_cbufs.size()),
                     r.ps_cbufs, sc.cbuf_bind, sc.smem_cbuf_by_pc))
    return false;
  const gpu::gcn::MimgBindingPlan mimg_plan = RdnaPlanMimg(program);
  if (mimg_plan.binding_srsrc.size() > StageContext::kMaxPsSamplers)
    return false;
  sc.mimg_plan = &mimg_plan;  // borrowed by EmitBody
  for (uint32_t i = 0; i < mimg_plan.binding_srsrc.size(); i++)
    r.ps_texs.push_back(
        {i, mimg_plan.binding_srsrc[i], mimg_plan.binding_storage[i]});

  const Id user_data = DeclareUserData(t);
  sc.main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  SeedUserData(t, user_data, 0, user_sgprs);
  SeedPsInputVgprs(t, ps_input_ena, iface);

  const bool has_color_export =
      std::any_of(program.begin(), program.end(), [](const Inst& inst) {
        return inst.enc == Enc::kExp && ((inst.raw[0] >> 4) & 0x3F) <= 7 &&
               (inst.raw[0] & 0xF);
      });

  const bool cfg = ForceCfg() || HasControlFlow(program);
  if (cfg && has_color_export) {
    t.m.Store(gpu::gcn::PsColorOut(t, sc, 0),
              t.m.ConstComposite(
                  t.t_v4, {t.F32(0.f), t.F32(0.f), t.F32(0.f), t.F32(0.f)}));
    sc.color_written_var = t.m.Variable(t.p_priv_u, spv::StorageClass::Private,
                                        t.m.ConstNull(t.t_u));
  }
  // DELTA_GPU_FORCECOLOR + DELTA_GPU_NOKILL: paint every rasterized fragment
  // green at entry, before any exec-mask discard can suppress the export. This
  // separates "draw produced no fragments" from "all fragments were discarded"
  // (FORCECOLOR alone only recolors fragments that reach the export).
  static const bool paint_raster =
      std::getenv("DELTA_GPU_FORCECOLOR") && std::getenv("DELTA_GPU_NOKILL");
  if (paint_raster)
    t.m.Store(gpu::gcn::PsColorOut(t, sc, 0),
              t.m.ConstComposite(
                  t.t_v4, {t.F32(0.f), t.F32(1.f), t.F32(0.f), t.F32(1.f)}));
  EmitBody(t, program, sc);

  // The straight-line alpha kill: v_cmpx_* compares and clears EXEC, and the
  // export then applies to no lane. Nothing consulted EXEC, so those fragments
  // were written anyway -- Skyrim's menu quads painted their transparent area
  // over the whole screen. Gate the fragment on EXEC only for shaders that
  // actually contain a cmpx: our EXEC model is one bit, not a lane mask, and
  // applying it everywhere discards everything in a shader that merely moves
  // EXEC around (Isaac's frame goes black). DELTA_GPU_NOKILL disables it.
  bool kills_lanes = false;
  for (const Inst& in : program) {
    // v_cmpx_* compares and writes EXEC.
    if ((in.enc == Enc::kVopc && (in.opcode & 0x10)) ||
        (in.enc == Enc::kVop3 && in.opcode <= 0xFF && (in.opcode & 0x10)))
      kills_lanes = true;
    // A scalar EXEC write is NOT a usable signal: every shader with control
    // flow moves EXEC around, and our model is a single bit, so gating on it
    // discards Isaac's entire frame. Only the explicit compare-and-kill counts.
  }
  if (sc.wrote_color && kills_lanes && !std::getenv("DELTA_GPU_NOKILL")) {
    const Id live = t.IsNonZero(t.Exec());
    const Id kill_blk = t.m.NewBlock(), after_kill = t.m.NewBlock();
    t.m.SelectionMerge(after_kill);
    t.m.BranchConditional(live, after_kill, kill_blk);
    t.m.OpenBlock(kill_blk);
    t.m.Kill();
    t.m.OpenBlock(after_kill);
  }

  if (!sc.wrote_color) {
    // No color export at all: opaque white fallback (matches the GFX7 path).
    t.m.Store(gpu::gcn::PsColorOut(t, sc, 0),
              t.m.ConstComposite(
                  t.t_v4, {t.F32(1.f), t.F32(1.f), t.F32(1.f), t.F32(1.f)}));
  } else if (sc.color_written_var) {
    static const bool no_kill = std::getenv("DELTA_GPU_NOKILL") != nullptr;
    if (!no_kill) {
      const Id wrote = t.IsNonZero(t.m.Load(t.t_u, sc.color_written_var));
      const Id kill_blk = t.m.NewBlock(), after_kill = t.m.NewBlock();
      t.m.SelectionMerge(after_kill);
      t.m.BranchConditional(wrote, after_kill, kill_blk);
      t.m.OpenBlock(kill_blk);
      t.m.Kill();
      t.m.OpenBlock(after_kill);
    }
  }

  // DELTA_GPU_PSUV: replace MRT0 with the interpolated location-0 varying, to
  // see the coordinate field a pass actually receives (a pure horizontal ramp
  // means the vertical component never made it across the VS->PS link).
  if (std::getenv("DELTA_GPU_PSUV")) {
    const Id in0 = gpu::gcn::PsInputVar(t, sc, 0);
    const Id p_in_f = t.m.TypePointer(spv::StorageClass::Input, t.t_f);
    const Id u = t.m.Load(t.t_f, t.m.AccessChain(p_in_f, in0, {t.U32(0)}));
    const Id v = t.m.Load(t.t_f, t.m.AccessChain(p_in_f, in0, {t.U32(1)}));
    t.m.Store(gpu::gcn::PsColorOut(t, sc, 0),
              t.m.CompositeConstruct(t.t_v4, {u, v, t.F32(0.f), t.F32(1.f)}));
  }

  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Fragment, sc.main_fn, "main", iface);
  t.m.ExecMode(sc.main_fn, spv::ExecutionMode::OriginUpperLeft);
  return true;
}

bool TranslateDepthOnlyPs(Translator& t) {
  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  t.m.ReturnVoid();
  t.m.EndFunction();
  t.m.EntryPoint(spv::ExecutionModel::Fragment, main_fn, "main", {});
  t.m.ExecMode(main_fn, spv::ExecutionMode::OriginUpperLeft);
  return true;
}

}  // namespace

Recompiled Recompile(const uint32_t* vs_code,
                     const uint32_t* ps_code,
                     const uint32_t* vs_user_data,
                     const uint32_t* ps_user_data,
                     uint32_t ps_input_ena,
                     bool gl_clip_space,
                     uint32_t vs_user_sgprs,
                     uint32_t ps_user_sgprs) {
  Recompiled r;
  if (!vs_code || !vs_user_data || !ps_user_data)
    return r;

  constexpr uint64_t kMaxShaderBytes = 4096 * sizeof(uint32_t);
  const uint64_t vs_address = reinterpret_cast<uintptr_t>(vs_code);
  const uint64_t ps_address = reinterpret_cast<uintptr_t>(ps_code);
  if (!gpu::IsReadableRange(vs_address, kMaxShaderBytes) ||
      (ps_code && !gpu::IsReadableRange(ps_address, kMaxShaderBytes)))
    return r;

  const Program vs_program = ReachableProgram(DecodeShader(vs_code, 4096));
  const Program ps_program =
      ps_code ? ReachableProgram(DecodeShader(ps_code, 4096)) : Program{};

  // V_INTERP_MOV P0 reads a per-primitive (flat) parameter; represent those
  // locations as flat varyings in both stages.
  std::unordered_set<uint32_t> flat_attrs;
  for (const Inst& inst : ps_program)
    if (inst.enc == Enc::kVintrp && inst.opcode == 2 &&
        (inst.raw[0] & 0xFF) == 2)
      flat_attrs.insert((inst.raw[0] >> 10) & 0x3F);

  Translator tv;
  tv.rdna_sources = true;
  gpu::gcn::ResetUnsupported();
  g_vs_addr = reinterpret_cast<uintptr_t>(vs_code);
  if (!TranslateVs(vs_program, vs_user_data, flat_attrs, r, tv, gl_clip_space,
                   vs_user_sgprs) ||
      gpu::gcn::HadUnsupported())
    return r;
  Translator tp;
  tp.rdna_sources = true;
  g_ps_addr = reinterpret_cast<uintptr_t>(ps_code);
  tp.InitTypes();
  gpu::gcn::ResetUnsupported();
  if (ps_code ? !TranslatePs(ps_program, flat_attrs, ps_input_ena, r, tp,
                             ps_user_sgprs)
              : !TranslateDepthOnlyPs(tp))
    return r;
  if (gpu::gcn::HadUnsupported())
    return r;

  const std::vector<uint32_t> vs = tv.m.Assemble();
  const std::vector<uint32_t> ps = tp.m.Assemble();
  // gfx10.3 RECTLIST draws reach us as three corners of a rectangle; Vulkan has
  // no such topology, so carry the same expansion stage the GFX7 path uses.
  const std::vector<uint32_t> gs =
      gpu::gcn::EmitRectListGeometry(r.num_params, flat_attrs);
  std::string err;
  if (!gpu::gcn::spirv::Validate(vs, &err)) {
    if (gpu::gcn::TraceEnabled())
      std::fprintf(stderr, "[rdna] VS invalid: %s\n", err.c_str());
    return r;
  }
  if (!gpu::gcn::spirv::Validate(ps, &err)) {
    if (gpu::gcn::TraceEnabled())
      std::fprintf(stderr, "[rdna] PS invalid: %s\n", err.c_str());
    return r;
  }
  // DELTA_GPU_SPVDUMP=<dir>: write each recompiled stage's SPIR-V so it can be
  // read with spirv-dis (the only way to see what the position math became).
  if (const char* dir = std::getenv("DELTA_GPU_SPVDUMP")) {
    char path[512];
    std::snprintf(path, sizeof(path), "%s/vs_%lx.spv", dir,
                  (unsigned long)reinterpret_cast<uintptr_t>(vs_code));
    if (FILE* f = std::fopen(path, "wb")) {
      std::fwrite(vs.data(), 4, vs.size(), f);
      std::fclose(f);
    }
    std::snprintf(path, sizeof(path), "%s/ps_%lx.spv", dir,
                  (unsigned long)reinterpret_cast<uintptr_t>(ps_code));
    if (FILE* f = std::fopen(path, "wb")) {
      std::fwrite(ps.data(), 4, ps.size(), f);
      std::fclose(f);
    }
  }
  // SPIRV-Tools' def-use rewrite takes minutes on some large Skyrim modules.
  // These binaries were validated above, so submit them without
  // post-processing.
  r.vs_spirv = vs;
  r.fs_spirv = ps;
  std::string gs_err;
  if (!gs.empty() && gpu::gcn::spirv::Validate(gs, &gs_err))
    r.gs_spirv = gs;
  else if (gpu::gcn::TraceEnabled())
    std::fprintf(stderr, "[rdna] RECTLIST GS invalid: %s\n", gs_err.c_str());
  r.ok = !r.vs_spirv.empty() && !r.fs_spirv.empty();
  return r;
}

}  // namespace gpu::rdna

#endif  // DELTA_HAVE_SPIRV_BACKEND
