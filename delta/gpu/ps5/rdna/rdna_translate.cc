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
#include "base/arch.h"

#ifndef DELTA_HAVE_SPIRV_BACKEND
namespace gpu::rdna {
gpu::gcn::Recompiled Recompile(const u32*,
                               const u32*,
                               const u32*,
                               const u32*,
                               u32,
                               bool,
                               u32,
                               u32) {
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
#include "gpu/gcn/gcn_audit.h"
#include "gpu/gcn/spirv/spv_post.h"
#include "gpu/gcn/spirv/translator.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/rdna/rdna_emit.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kExpTrace, "DELTA_GPU_EXPTRACE", false);
DELTA_OPTION(bool, kAllowNan, "DELTA_GPU_ALLOWNAN", false);
DELTA_OPTION(const char*, kDbgPos, "DELTA_GPU_DBGPOS", nullptr);
DELTA_OPTION(float, kRawPos, "DELTA_GPU_RAWPOS", 0.f);
DELTA_OPTION(bool, kDrawCensus, "DELTA_GPU_DRAWCENSUS", false);
DELTA_OPTION(const char*, kSpvDump, "DELTA_GPU_SPVDUMP", nullptr);
DELTA_OPTION(bool, kAgcTrace, "DELTA_AGC_TRACE", false);
DELTA_OPTION(bool, kGpuDebugalpha, "DELTA_GPU_DEBUGALPHA", false);
DELTA_OPTION(bool, kGpuForcecolor, "DELTA_GPU_FORCECOLOR", false);
DELTA_OPTION(bool, kGpuForcequad, "DELTA_GPU_FORCEQUAD", false);
DELTA_OPTION(bool, kGpuNokill, "DELTA_GPU_NOKILL", false);
DELTA_OPTION(bool, kGpuPosprobe, "DELTA_GPU_POSPROBE", false);
DELTA_OPTION(bool, kGpuPsuv, "DELTA_GPU_PSUV", false);
DELTA_OPTION(bool, kGpuShtrace, "DELTA_GPU_SHTRACE", false);
DELTA_OPTION(bool, kGpuSpirvCfg, "DELTA_GPU_SPIRV_CFG", false);
}  // namespace

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
  static const bool on = kGpuShtrace ||
                         kAgcTrace;
  return on;
}

// MUBUF opcode numbering. gfx10.3 REVERTED to the gfx6/gfx7 assignment, so the
// GFX7 ranges are correct here and must not be "corrected" to gfx8/gfx9's
// (verified against LLVM BUFInstructions.td, where every raw form is defined by
// MUBUF_Real_AllAddr_gfx6_gfx7_gfx10, and gfx8/gfx9 alone use the shifted
// MUBUF_Real_AllAddr_vi table):
//   0x00-0x03 buffer_load_format_x/xy/xyz/xyzw
//   0x04-0x07 buffer_store_format_*
//   0x08-0x0b buffer_load_ubyte/sbyte/ushort/sshort
//   0x0c-0x0f buffer_load_dword/x2/x4/x3   (x4 before x3, as on gfx6/gfx7)
//   0x18/0x1a buffer_store_byte/short   0x1c-0x1f store_dword/x2/x4/x3
//   0x30-0x60 atomics                   0x80-0x87 the d16 format forms
// The d16 forms only fit because the gfx10 opcode field is 8 bits ([25:18], vs
// GFX7's 7); rdna_decode's Classify reads all 8, so they never alias 0x00-0x07.
bool RdnaMubufStore(u32 op) {
  return (op >= 0x04 && op <= 0x07) || (op >= 0x84 && op <= 0x87) ||
         (op >= 0x18 && op <= 0x1f);
}

// MTBUF: 0x00-0x03 load, 0x04-0x07 store, 0x08-0x0f the d16 pair (opcode bit 3
// comes from word1[21], which rdna_decode folds in).
bool RdnaMtbufStore(u32 op) {
  return (op >= 0x04 && op <= 0x07) || (op >= 0x0c && op <= 0x0f);
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
std::unordered_map<u32, std::pair<u32, u32> >
MapTableChainedLoads(const Program& insts) {
  std::unordered_map<u32, std::pair<u32, u32> > out;
  i32 root[128];
  u32 slot[128] = {};
  for (int i = 0; i < 128; i++)
    root[i] = -1;
  std::unordered_map<u32, u32> next_slot;  // per table root
  for (const Inst& in : insts) {
    if (in.enc == Enc::kSop1 && in.opcode == 0x20)
      break;  // s_setpc_b64 (return)
    if (in.enc == Enc::kSopp && in.opcode == 1)
      break;  // s_endpgm
    if (in.enc == Enc::kSmrd &&
        in.opcode <= 0x04) {  // s_load_dword{,x2,x4,x8,x16}
      const u32 sdst = (in.raw[0] >> 6) & 0x7F;
      const u32 sbase = (in.raw[0] & 0x3F) * 2;
      const u32 nreg = in.opcode == 0   ? 1
                            : in.opcode == 1 ? 2
                            : in.opcode == 2 ? 4
                            : in.opcode == 3 ? 8
                                             : 16;
      const u32 s = in.opcode == 2 ? next_slot[sbase]++ : 0;
      for (u32 k = 0; k < nreg && sdst + k < 128; k++) {
        root[sdst + k] = static_cast<i32>(sbase);
        slot[sdst + k] = s;
      }
      continue;
    }
    if ((in.enc != Enc::kMubuf && in.enc != Enc::kMtbuf) || in.opcode > 0x03)
      continue;
    const u32 srsrc = ((in.raw[1] >> 16) & 0x1F) * 4;
    if (srsrc < 128 && root[srsrc] >= 0)
      out[in.pc] = {static_cast<u32>(root[srsrc]), slot[srsrc]};
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
  BASE_LOGI("gcnspv", "=== {} decode: {} insts ===", tag, prog.size());
  for (const Inst& in : prog) {
    base::String line;
    base::FormatTo(line, "  pc={:04x} len={} {:<6} op={:#05x}  {:08x}", in.pc,
                   in.size, EncName(in.enc), in.opcode, in.raw[0]);
    if (in.size >= 2)
      base::FormatTo(line, " {:08x}", in.raw[1]);
    if (in.has_literal)
      base::FormatTo(line, " lit={:08x}", in.literal);
    BASE_LOGI("gcnspv", "{}", line.c_str());
  }
}

// RDNA2 renumbered v_cndmask_b32 from GFX7's VOP2 0x00 to 0x01 (0x01 is
// v_readlane on GFX7); map it back to the shared emitter's VCC select.
u32 RemapVop2(u32 op) {
  return op == 0x01 ? 0x00 : op;
}

bool RdnaSharedVop2(u32 op) {
  return (op >= 0x03 && op <= 0x05) || (op >= 0x08 && op <= 0x0c) ||
         (op >= 0x0f && op <= 0x14) || op == 0x16 || op == 0x18 ||
         (op >= 0x1a && op <= 0x1d) || op == 0x2f;
}

bool UnsupportedRdnaScalarSource(u32 field) {
  return (field >= 108 && field <= 123) || (field >= 209 && field <= 239) ||
         field == 249 || field == 250 || field == 254;
}

bool UnsupportedRdnaValuSource(u32 field) {
  if (field >= 256)
    return false;
  return (field >= 108 && field <= 123) || (field >= 209 && field <= 239) ||
         field == 249 || field == 250 || field == 254;
}

bool UnsupportedRdnaSdwaSource(u32 field) {
  return UnsupportedRdnaValuSource(field) || field == 255;
}

bool UsesUnsupportedRdnaSource(const Inst& inst) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
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
        const u32 src0 = (w1 & 0xff) + (((w1 >> 23) & 1) ? 0 : 256);
        return UnsupportedRdnaSdwaSource(src0);
      }
      return UnsupportedRdnaValuSource(w & 0x1ff);
    }
    case Enc::kVop2:
    case Enc::kVopc: {
      if (inst.extension == gpu::gcn::InstExtension::kSdwa) {
        const u32 src0 = (w1 & 0xff) + (((w1 >> 23) & 1) ? 0 : 256);
        const u32 src1 = ((w >> 9) & 0xff) + (((w1 >> 31) & 1) ? 0 : 256);
        return UnsupportedRdnaSdwaSource(src0) ||
               UnsupportedRdnaSdwaSource(src1);
      }
      return UnsupportedRdnaValuSource(w & 0x1ff);
    }
    case Enc::kVop3:
    case Enc::kVop3p: {
      const u32 count = Vop3SourceCount(inst.enc, inst.opcode);
      for (u32 source = 0; source < count; source++)
        if (UnsupportedRdnaValuSource((w1 >> (source * 9)) & 0x1ff))
          return true;
      return false;
    }
    default:
      return false;
  }
}

Id RdnaF16Bits(Translator& t, u32 field, u32 literal) {
  static constexpr u16 kInline[] = {0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000,
                                         0xc000, 0x4400, 0xc400, 0x3118};
  if (field >= 240 && field <= 248)
    return t.U32(kInline[field - 240]);
  return t.SrcRaw(field, literal);
}

// RDNA2 VOP3 integer operations whose gfx10 opcodes or destination layouts do
// not match the shared GFX7 emitter.
bool RdnaVop3HasSdst(u32 op) {
  return op == 0x128 || op == 0x129 || op == 0x12A || op == 0x30F ||
         op == 0x310 || op == 0x319 || op == 0x16D || op == 0x16E ||
         op == 0x176 || op == 0x177;
}

bool RdnaEmitVop3Int(Translator& t,
                     u32 op,
                     u32 vdst,
                     u32 sdst,
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
                   u32 op,
                   u32 vdst,
                   u32 s0,
                   u32 s1,
                   u32 s2,
                   u32 lit) {
  auto unpack = [&](u32 f) {
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

// SOP2 slots gfx10 added above the GFX7 numbering the shared emitter speaks.
// 0x00-0x2c are identical on both (LLVM SOP2_Real_gfx6_gfx7_gfx10); 0x2b
// (s_cbranch_g_fork) and 0x2d are gfx6/gfx7-only, and 0x2e-0x36 are new. None of
// these writes SCC. s_pack_* reach the shared emitter too, but it gates them on
// the PS4's Neo ISA flag, which an RDNA2 Inst never carries.
bool RdnaEmitSop2(Translator& t, const Inst& inst) {
  const u32 w = inst.raw[0], op = inst.opcode;
  if (op < 0x32 || op > 0x36)
    return false;
  const u32 sdst = (w >> 16) & 0x7F;
  const Id a = t.SrcRaw(w & 0xFF, inst.literal);
  const Id b = t.SrcRaw((w >> 8) & 0xFF, inst.literal);
  const auto mul_hi = [&](spv::Op wide) {
    return t.m.CompositeExtract(t.t_u,
                                t.m.Emit(wide, t.PairType(), {a, b}), 1);
  };
  switch (op) {
    case 0x32:  // s_pack_ll_b32_b16: {b[15:0], a[15:0]}
      t.SetSdst(sdst, 0,
                t.Or(t.And(a, t.U32(0xFFFF)),
                     t.Shl(t.And(b, t.U32(0xFFFF)), t.U32(16))));
      return true;
    case 0x33:  // s_pack_lh_b32_b16: {b[31:16], a[15:0]}
      t.SetSdst(sdst, 0,
                t.Or(t.And(a, t.U32(0xFFFF)), t.And(b, t.U32(0xFFFF0000u))));
      return true;
    case 0x34:  // s_pack_hh_b32_b16: {b[31:16], a[31:16]}
      t.SetSdst(sdst, 0,
                t.Or(t.Shr(a, t.U32(16)), t.And(b, t.U32(0xFFFF0000u))));
      return true;
    case 0x35:
      t.SetSdst(sdst, 0, mul_hi(spv::Op::OpUMulExtended));
      return true;  // s_mul_hi_u32
    case 0x36:
      t.SetSdst(sdst, 0, mul_hi(spv::Op::OpSMulExtended));
      return true;  // s_mul_hi_i32
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
  u32 source, offset, count;
};

bool Overlaps(u32 first,
              u32 count,
              u32 other_first,
              u32 other_count) {
  return first < other_first + other_count && other_first < first + count;
}

void InvalidateCbufDefs(std::unordered_map<u32, CbufDef>& loads,
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
                               u32 index,
                               u32 sdst,
                               u32 count) {
  for (u32 i = index + 1; i < program.size(); i++) {
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

std::unordered_map<u32, u64> BufferVersionKeys(
    const Program& program) {
  std::unordered_map<u32, u64> out;
  u32 versions[136] = {};
  u32 generation = 1;
  for (const Inst& inst : program) {
    if (inst.enc == Enc::kSmrd && SmemLoadCount(inst.opcode)) {
      const Smem smem = DecodeSmem(inst);
      const u32 dwords = smem.op >= 0x08 ? 4 : 2;
      u64 key = smem.sbase | (static_cast<u64>(dwords == 4) << 7);
      for (u32 i = 0; i < dwords; i++)
        key |= static_cast<u64>(versions[smem.sbase + i]) << (8 + i * 13);
      out.emplace(inst.pc, key);
    }
    const ScalarWrite write = DecodeScalarWrite(inst);
    for (u32 i = 0; i < write.count && write.first + i < 136; i++)
      versions[write.first + i] = generation;
    generation++;
  }
  return out;
}

u32 TraceCbufChain(u32 sbase,
                        const std::unordered_map<u32, CbufDef>& loads,
                        u32 chain_off[3],
                        u32* len) {
  u32 cur = sbase, n = 0, tmp[3] = {};
  while (n < 3) {
    auto it = loads.find(cur);
    if (it == loads.end())
      break;
    tmp[n++] = it->second.offset;
    cur = it->second.source;
  }
  for (u32 i = 0; i < n; i++)
    chain_off[i] = tmp[n - 1 - i];  // reverse
  *len = n;
  return cur;
}

// SGPRs a VMEM/MIMG op reads as its descriptor (srsrc V#/T#, ssamp S#). An
// s_load writing one of these fetches a DESCRIPTOR, not constant data, so the
// cbuf planner must leave it alone: ParseFetchInsts/RdnaPlanBufLoadCbufs and
// RdnaPlanMimg resolve those at draw time from user data instead.
static std::unordered_set<u32> VmemDescriptorSgprs(const Program& program) {
  std::unordered_set<u32> regs;
  for (const Inst& inst : program) {
    const bool buf = inst.enc == Enc::kMubuf || inst.enc == Enc::kMtbuf;
    const bool img = inst.enc == Enc::kMimg;
    if (!buf && !img)
      continue;
    const u32 srsrc = ((inst.raw[1] >> 16) & 0x1F) * 4;
    for (u32 i = 0; i < 8; i++)
      regs.insert(srsrc + i);
    if (img && inst.opcode >= 0x20) {
      const u32 ssamp = ((inst.raw[1] >> 21) & 0x1F) * 4;
      for (u32 i = 0; i < 4; i++)
        regs.insert(ssamp + i);
    }
  }
  return regs;
}

// Plan the set-1 UBO bindings a stage's SMEM loads reference. A leaf read is an
// s_buffer_load* (op 0x08-0x0C, V# in the sbase quad) or an s_load* (op
// 0x00-0x04, pointer in the sbase pair) whose result is used as data -- not as
// another SMEM's descriptor base. When the base SGPR was itself s_load'd (a
// runtime pointer chain, e.g. a 2D VS that loads its transform's V# from a root
// descriptor table), the chain back to the user-data root is recorded so the
// renderer can walk it.
bool RdnaPlanCbufs(const Program& program,
                   u32 first_binding,
                   std::vector<ShaderCbuf>& cbufs,
                   std::unordered_map<u32, u32>& bindings,
                   std::unordered_map<u32, u32>& by_pc) {
  // Walk in program order, growing the def map as s_loads appear, so each
  // SMEM's base traces through the defs live AT that instruction. Shaders
  // reuse SGPRs (the sprite VS s_buffer_loads its transform from the s[8:11]
  // user-data V#, then s_loads the vertex V# INTO s[8:11]); a whole-program
  // last-write map would misroute the transform to the vertex chain.
  std::unordered_map<u32, CbufDef> loads;
  std::unordered_map<u64, u32> binding_by_producer;
  const auto version_keys = BufferVersionKeys(program);
  const auto descriptor_sgprs = VmemDescriptorSgprs(program);
  u32 inst_index = 0;
  for (const Inst& inst : program) {
    const u32 producer = inst_index++;
    if (inst.enc != Enc::kSmrd) {
      InvalidateCbufDefs(loads, DecodeScalarWrite(inst));
      continue;
    }
    const Smem smem = DecodeSmem(inst);
    const u32 op = smem.op;
    const bool sbufload = op >= 0x08 && op <= 0x0C, sload = op <= 0x04;
    if (!sbufload && !sload)
      continue;
    const u32 sbase = smem.sbase, sdst = smem.sdst;
    const u32 load_count = SmemLoadCount(op);
    const i32 off =
        sbufload ? static_cast<i32>(inst.raw[1] & 0xFFFFF) : smem.offset;
    // A descriptor fetch (the V# a vertex fetch or texture op then reads). Its
    // offset is often a runtime table index -- Minecraft's NGG VS computes one
    // into vcc_hi -- which no cbuf binding can express, and treating it as a
    // constant buffer failed the whole shader over a load the cbuf path never
    // needed to see.
    if (sload && descriptor_sgprs.count(sdst)) {
      InvalidateCbufDefs(loads, {sdst, load_count});
      continue;
    }
    if (sload && smem.soffset == 125 && off < 0) {
      if (ShDbg())
        BASE_LOGI("gcnspv", "cbuf plan reject pc={:#x} sload negative off={}",
                  inst.pc, off);
      return false;
    }
    if (sload &&
        UsedAsBaseBeforeOverwrite(program, producer, sdst, load_count)) {
      InvalidateCbufDefs(loads, {sdst, load_count});
      loads[sdst] = {sbase, static_cast<u32>(off), load_count};
      continue;
    }
    const u32 hi =
        smem.soffset != 125
            ? gpu::gcn::kCbufDwords
            : static_cast<u32>(off < 0 ? 0 : off) / 4 + SmemLoadCount(op);
    if (smem.soffset != 125 || hi > gpu::gcn::kCbufDwords) {
      if (ShDbg())
        BASE_LOGI("gcnspv",
                  "cbuf plan reject pc={:#x} op={:#x} soffset={} off={} "
                  "hi={} raw={:08x} {:08x} sbase={} sdst={}",
                  inst.pc, op, smem.soffset, off, hi, inst.raw[0],
                  inst.raw[1], smem.sbase, smem.sdst);
      return false;
    }

    u32 chain_off[3] = {}, chain_len = 0;
    const u32 root = TraceCbufChain(sbase, loads, chain_off, &chain_len);
    const u64 key = version_keys.at(inst.pc);

    auto it = binding_by_producer.find(key);
    if (it == binding_by_producer.end()) {
      const u32 binding =
          first_binding + static_cast<u32>(cbufs.size());
      if (binding >= kMaxCbufBindings) {
        if (ShDbg())
          BASE_LOGI("gcnspv", "cbuf plan reject pc={:#x} out of "
                              "bindings ({})",
                    inst.pc, binding);
        return false;
      }
      it = binding_by_producer.emplace(key, binding).first;
      bindings.emplace(sbase, binding);
      ShaderCbuf cb;
      cb.binding = binding;
      cb.ud_sgpr = root;
      cb.num_dwords = hi;
      cb.chain_len = chain_len;
      for (u32 i = 0; i < 3; i++)
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
// Raw MUBUF loads (buffer_load_dword{,x2,x3,x4} and the sub-dword forms) the
// shader indexes itself, plus every buffer_load_format the vertex-input path did
// not lift (`claimed`). Each distinct descriptor SGPR quad becomes one set-2
// storage buffer, which the command processor resolves per draw -- a format load
// belongs here rather than in a 64-byte UBO because its index is per-lane and
// reaches the whole resource. The shared PlanGfxBuffers cannot be reused: its
// descriptor-reload versioning reads the SMEM sdst with GCN field positions.
void RdnaPlanGfxBuffers(const Program& program,
                        u32 first_binding,
                        const std::unordered_set<u32>* claimed,
                        std::vector<gpu::gcn::ShaderBuffer>& buffers,
                        std::unordered_map<u32, u32>& bindings) {
  std::unordered_map<u32, u32> by_srsrc;
  for (const Inst& inst : program) {
    const bool raw = inst.opcode >= 0x08 && inst.opcode <= 0x0f;
    const bool format = inst.opcode <= 0x03;
    if (inst.enc != Enc::kMubuf || (!raw && !format))
      continue;
    if (claimed && claimed->count(inst.pc))
      continue;
    const u32 srsrc = ((inst.raw[1] >> 16) & 0x1F) * 4;
    const auto found = by_srsrc.find(srsrc);
    if (found != by_srsrc.end()) {
      bindings[inst.pc] = found->second;
      continue;
    }
    const u32 binding =
        first_binding + static_cast<u32>(buffers.size());
    if (binding >= gpu::gcn::MaxGfxBuffers()) {
      gpu::gcn::WarnUnsupported("mubuf.binding-count", binding + 1);
      continue;
    }
    by_srsrc[srsrc] = binding;
    bindings[inst.pc] = binding;
    buffers.push_back({binding, srsrc, inst.pc});
  }
}

void RdnaPlanBufLoadCbufs(const Program& program,
                          u32 first_binding,
                          std::vector<ShaderCbuf>& cbufs,
                          std::unordered_map<u32, u32>& bindings,
                          std::unordered_map<u32, u32>& by_pc) {
  // Mirror ParseFetchInsts' walk: srsrc SGPRs written by an s_load hold V#s
  // from the user-data descriptor TABLE (entry index from
  // MapTableChainedLoads). A constant load through such a V# becomes a chained
  // cbuf (root = the s_load's sbase pair, table offset = entry * 16 bytes); the
  // renderer derefs the table pointer at draw time. srsrc-keyed bindings
  // collide when the shader reuses an SGPR quad (s[8:11] = MVP V# then vertex
  // V#), so constant loads are bound per-instruction (by_pc) instead.
  // MUBUF format loads are NOT bound here: their format lives in the V#, so they
  // go through RdnaPlanGfxBuffers/RdnaEmitBufFormatLoad instead.
  const auto chained_loads = MapTableChainedLoads(program);
  for (const Inst& inst : program) {
    if (inst.enc == Enc::kSop1 && inst.opcode == 0x20)
      break;  // s_setpc_b64
    if (inst.enc == Enc::kSopp && inst.opcode == 1)
      break;  // s_endpgm
    if (inst.enc != Enc::kMtbuf || inst.opcode > 0x03)
      continue;
    const u32 srsrc = ((inst.raw[1] >> 16) & 0x1F) * 4;
    const auto chain = chained_loads.find(inst.pc);
    const bool chained = chain != chained_loads.end();
    if (BufLoadIsVertexFetch(inst, chained))
      continue;  // real fetch -> vertex input
    const u32 binding =
        first_binding + static_cast<u32>(cbufs.size());
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
std::unordered_set<u32> LaunchExecMovPcs(const Program& program) {
  std::unordered_set<u32> skip, written;
  for (const Inst& in : program) {
    u32 d0 = 0xFFFF, n = 1;
    switch (in.enc) {
      case Enc::kSop1: {
        const u32 sdst = (in.raw[0] >> 16) & 0x7F;
        const u32 src = in.raw[0] & 0xFF;
        const bool b64 = in.opcode == 0x04;
        if ((in.opcode == 0x03 || b64) && sdst == 126 && src <= 105 &&
            !written.count(src) && (!b64 || !written.count(src + 1))) {
          skip.insert(in.pc);
          if (ShDbg())
            BASE_LOGI("gcnspv", "drop launch-state exec mov @pc={:04x} (s{})",
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
      for (u32 k = 0; k < n; k++)
        written.insert(d0 + k);
  }
  return skip;
}

void RdnaEmitSmem(Translator& t, const Inst& inst, StageContext& sc) {
  const Smem smem = DecodeSmem(inst);
  const u32 op = smem.op;
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
    const u32 binding =
        pc_it != sc.smem_cbuf_by_pc.end() ? pc_it->second : base_it->second;
    const u32 immediate = op >= 0x08
                                   ? inst.raw[1] & 0xFFFFC
                                   : static_cast<u32>(smem.offset) & ~3u;
    const Id soffset =
        smem.soffset == 125 ? t.U32(0) : t.SrcRaw(smem.soffset, 0);
    const Id byte_offset = t.Add(t.And(soffset, t.U32(~3u)), t.U32(immediate));
    const Id dword0 = t.Shr(byte_offset, t.U32(2));
    const u32 n = SmemLoadCount(op);
    for (u32 k = 0; k < n; k++)
      t.SetSdst(smem.sdst, k, t.CbufDwordId(binding, t.Add(dword0, t.U32(k))));
    return;
  }
  gpu::gcn::WarnUnsupported("smem.rdna", op, inst.raw[0], inst.raw[1]);
}

// Address of the shader being translated, for shader-specific debug output.
thread_local u64 g_ps_addr = 0;

// ---- MUBUF buffer_load_format ----------------------------------------------
// buffer_load_format converts through the FORMAT in the V#, which the
// instruction does not carry. The V#s the stage's buffer ops read, resolved by
// replaying the stage's scalar code against the live user data -- the same
// resolution the command processor performs when it binds those buffers. The
// format is then baked into the module, so a later draw binding a
// differently-formatted V# to the same shader would be wrong; that is the
// assumption the lifted vertex-fetch path already makes about r.attrs.
thread_local std::unordered_map<u32, BufferResource> g_stage_bufs;

// One dropped-store report per translated stage.
thread_local bool g_warned_store = false;

// One channel layout of the gfx10.3 unified buffer format (V# word3[18:12]).
// The enum runs one layout at a time in the order UNorm, SNorm, UScaled,
// SScaled, UInt, SInt [, Float]. Names are most-significant channel first, so
// channel x takes the LAST number's bits at bit offset 0.
enum class BufNum : u8 {
  kUnorm,
  kSnorm,
  kUscaled,
  kSscaled,
  kUint,
  kSint,
  kFloat
};

struct BufFormat {
  u32 comps = 0;
  u32 bits[4] = {};
  BufNum num = BufNum::kUnorm;
};

bool DecodeBufFormat(u32 gfmt, BufFormat& out) {
  struct Run {
    u8 first, count, comps, bits[4];
  };
  static constexpr Run kRuns[] = {
      {1, 6, 1, {8}},
      {7, 7, 1, {16}},
      {14, 6, 2, {8, 8}},
      {20, 3, 1, {32}},
      {23, 7, 2, {16, 16}},
      {30, 7, 3, {10, 11, 11}},      // 11_11_10
      {37, 7, 3, {11, 11, 10}},      // 10_11_11
      {44, 6, 4, {10, 10, 10, 2}},   // 2_10_10_10
      {50, 6, 4, {2, 10, 10, 10}},   // 10_10_10_2
      {56, 6, 4, {8, 8, 8, 8}},
      {62, 3, 2, {32, 32}},
      {65, 7, 4, {16, 16, 16, 16}},
      {72, 3, 3, {32, 32, 32}},
      {75, 3, 4, {32, 32, 32, 32}},
  };
  for (const Run& r : kRuns) {
    if (gfmt < r.first || gfmt >= u32{r.first} + r.count)
      continue;
    const u32 i = gfmt - r.first;
    out.comps = r.comps;
    for (u32 c = 0; c < 4; c++)
      out.bits[c] = r.bits[c];
    // A 3-wide run is UInt, SInt, Float; a wider one appends Float last.
    out.num = r.count == 3 ? (i == 0   ? BufNum::kUint
                              : i == 1 ? BufNum::kSint
                                       : BufNum::kFloat)
              : i == 6     ? BufNum::kFloat
                           : static_cast<BufNum>(i);
    return true;
  }
  return false;
}

// One channel's stored bits -> the dword the hardware leaves in the VGPR.
Id ConvertBufChannel(Translator& t, Id raw, u32 bits, BufNum num) {
  const bool is_signed = num == BufNum::kSnorm || num == BufNum::kSscaled ||
                         num == BufNum::kSint;
  Id value = raw;
  if (is_signed && bits < 32)
    value = t.m.Bitcast(t.t_u,
                        t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                                 {t.m.Bitcast(t.t_i, raw), t.U32(0),
                                  t.U32(bits)}));
  const auto as_float = [&](Id f) { return t.m.Bitcast(t.t_u, f); };
  const auto to_f = [&](bool sgn) {
    return sgn ? t.m.Emit(spv::Op::OpConvertSToF, t.t_f,
                          {t.m.Bitcast(t.t_i, value)})
               : t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {value});
  };
  switch (num) {
    case BufNum::kUint:
    case BufNum::kSint:
      return value;
    case BufNum::kFloat:
      if (bits == 32)
        return value;
      return as_float(t.m.CompositeExtract(
          t.t_f,
          t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {value}), 0));
    case BufNum::kUscaled:
      return as_float(to_f(false));
    case BufNum::kSscaled:
      return as_float(to_f(true));
    case BufNum::kUnorm:
      return as_float(t.FMul(
          to_f(false), t.F32(1.f / static_cast<float>((1ull << bits) - 1))));
    case BufNum::kSnorm:
      // The most negative code maps below -1 and the hardware clamps it.
      return as_float(t.Ext2(
          GLSLstd450FMax,
          t.FMul(to_f(true),
                 t.F32(1.f / static_cast<float>((1ull << (bits - 1)) - 1))),
          t.F32(-1.f)));
  }
  return value;
}

// buffer_load_format_x/xy/xyz/xyzw against the set-2 window the planner bound.
// Returns false when the V# (and with it the format, stride and any scalar
// byte offset) could not be resolved, leaving the caller to decline.
bool RdnaEmitBufFormatLoad(Translator& t,
                           const Inst& inst,
                           StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const auto bind = sc.gfx_buf_bind.find(inst.pc);
  const auto res = g_stage_bufs.find(inst.pc);
  if (bind == sc.gfx_buf_bind.end() || res == g_stage_bufs.end() ||
      !res->second.descriptor_valid)
    return false;
  BufFormat fmt;
  if (!DecodeBufFormat((res->second.descriptor[3] >> 12) & 0x7F, fmt))
    return false;
  // 11-bit and 10-bit packed floats need their own exponent/mantissa unpack.
  if (fmt.num == BufNum::kFloat && fmt.bits[0] != 16 && fmt.bits[0] != 32)
    return false;

  const u32 soffset_field = (w1 >> 24) & 0xFF;
  u32 soffset = 0;
  if (soffset_field != 125 && soffset_field != 128) {
    if (!res->second.soffset_valid)
      return false;
    soffset = res->second.soffset;
  }
  const u32 stride = (res->second.descriptor[1] >> 16) & 0x3FFF;
  const u32 vdata = (w1 >> 8) & 0xFF, vaddr = w1 & 0xFF;
  const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
  Id byte_off = t.U32((w & 0xFFF) + soffset);
  u32 va = vaddr;
  if (idxen)
    byte_off = t.Add(byte_off, t.Mul(t.Vg(va++), t.U32(stride)));
  if (offen)
    byte_off = t.Add(byte_off, t.Vg(va));

  const Id var = t.EnsureGfxBuffer(bind->second);
  const Id p_u = t.m.TypePointer(spv::StorageClass::StorageBuffer, t.t_u);
  const auto dword = [&](Id index) {
    return t.m.Load(
        t.t_u,
        t.m.AccessChain(
            p_u, var,
            {t.U32(0),
             t.UMin(index, t.U32(gpu::gcn::kGfxBufferDwords - 1))}));
  };
  const Id first_bit = t.Shl(byte_off, t.U32(3));
  const u32 requested = (inst.opcode & 3) + 1;
  u32 channel_bit = 0;
  for (u32 c = 0; c < requested; c++) {
    if (c >= fmt.comps) {
      // A channel the format does not store reads 0, and alpha reads one.
      const bool integer =
          fmt.num == BufNum::kUint || fmt.num == BufNum::kSint;
      t.SetVg(vdata + c,
              t.U32(c != 3 ? 0u : (integer ? 1u : 0x3F800000u)));
      continue;
    }
    const u32 bits = fmt.bits[c];
    const Id bit = t.Add(first_bit, t.U32(channel_bit));
    const Id word = dword(t.Shr(bit, t.U32(5)));
    const Id raw =
        bits == 32 ? word
                   : t.m.Emit(spv::Op::OpBitFieldUExtract, t.t_u,
                              {word, t.And(bit, t.U32(31)), t.U32(bits)});
    t.SetVg(vdata + c, ConvertBufChannel(t, raw, bits, fmt.num));
    channel_bit += bits;
  }
  return true;
}

// ---- exports ----------------------------------------------------------------
// gfx10.3 export targets: MRT0..7 = 0..7, MRTZ = 8, NULL = 9, POS0..4 = 12..16,
// PRIM = 20 (NGG connectivity), PARAM0..31 = 32..63.
void EmitExport(Translator& t, const Inst& inst, StageContext& sc) {
  const u32 w = inst.raw[0], w1 = inst.raw[1];
  const u32 en = w & 0xF, target = (w >> 4) & 0x3F, compr = (w >> 10) & 1;
  const u32 v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF,
                         (w1 >> 24) & 0xFF};
  if (ShDbg())
    BASE_LOGI("gcnspv",
              "{}-exp target={} en={:#x} compr={} done={} vsrc={:08x}",
              sc.is_ps ? "ps" : "vs", target, en, compr, (w >> 11) & 1, w1);
  if (!en)
    return;  // architecturally null export
  if (sc.is_ps) {
    // DELTA_GPU_EXPTRACE: the EN mask of each colour export. A component the
    // shader does not export gets a default here, and defaulting alpha to 1
    // would make every SRC_ALPHA blend opaque.
    if (kExpTrace && target <= 7) {
      static u32 seen[256] = {};
      const u32 k = ((target & 7) << 5) | ((en & 0xF) << 1) | compr;
      if (seen[k]++ == 0)
        BASE_LOGI("exp", "ps {:#x} mrt{} en={:#x} compr={}",
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
      if (!kAllowNan) {
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
      if (kGpuDebugalpha) {
        const Id a = t.m.CompositeExtract(t.t_f, col, 3);
        col = t.m.CompositeConstruct(t.t_v4, {a, a, a, t.F32(1.f)});
      }
      if (kGpuForcecolor)
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
      for (u32 i = 0; i < 4; i++)
        in[i] = i < sc.dbg_pos_comps ? t.m.CompositeExtract(t.t_f, val, i)
                                     : t.F32(i == 3 ? 1.f : 0.f);
      if (sc.dbg_pos_world >= 0) {
        Id w[4];
        for (u32 r = 0; r < 3; r++) {
          w[r] = t.F32(0.f);
          for (u32 c = 0; c < 4; c++) {
            const Id m = t.m.Bitcast(
                t.t_f, t.CbufDword(static_cast<u32>(sc.dbg_pos_world),
                                   r * 4 + c));
            w[r] = t.FAdd(w[r], t.FMul(m, in[c]));
          }
        }
        w[3] = t.F32(1.f);
        for (int i = 0; i < 4; i++)
          in[i] = w[i];
      }
      Id row[4];
      for (u32 r = 0; r < 4; r++) {
        row[r] = t.F32(0.f);
        for (u32 c = 0; c < 4; c++) {
          const Id m = t.m.Bitcast(
              t.t_f, t.CbufDword(static_cast<u32>(sc.dbg_pos_cbuf),
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
    const u32 p = target - 32;
    if (p + 1 > sc.max_param)
      sc.max_param = p + 1;
    const Id out_var = gpu::gcn::VsParamOut(t, sc, p);
    Id c[4];
    for (int i = 0; i < 4; i++)
      c[i] = (en & (1 << i)) ? t.VgF(v[i]) : t.F32(0.f);
    t.m.Store(out_var,
              t.m.CompositeConstruct(t.t_v4, {c[0], c[1], c[2], c[3]}));
  } else if (target != 9 && target != 20) {
    // 20 = the NGG primitive export (connectivity + edge flags). The renderer
    // draws from the index buffer the command stream binds, so the topology the
    // shader would emit here is already known; only its position/param exports
    // matter.
    gpu::gcn::WarnUnsupported("exp.vs-target", target, w, w1);
  }
}

// SDWA (src0 field == 249) puts the real operands in the extension dword. DPP
// extensions are rejected before reaching this partial SDWA implementation.
struct SdwaMod {
  u32 src0 = 0, src1 = 0;          // resolved operand fields
  u32 src0_sel = 6, src1_sel = 6;  // 0-3 byte, 4-5 word, 6 whole dword
  bool src0_sext = false, src1_sext = false;
  bool src0_neg = false, src0_abs = false;
  bool src1_neg = false, src1_abs = false;
  u32 dst_sel = 6;
  bool clamp = false;
  u32 omod = 0;
};

// S0/S1 (bits 23/31) choose the SCALAR file for that operand. Reading them as
// VGPRs instead is silent corruption: a 3D vertex shader multiplies its
// transform out of SGPRs, so the matrix comes back as zeros and every position
// collapses. DPP has no scalar-source form (its src0 is always a VGPR).
SdwaMod DecodeSdwa(const Inst& inst, u32 vsrc1, bool dpp) {
  const u32 m = inst.raw[1];
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

// SDWA's per-operand NEG/ABS act on the operand as a float, so they apply after
// the sub-dword select and before the op sees it. Rejecting them outright cost
// every shader that scales by a negated or absolute value.
Id SdwaFloatMod(Translator& t, Id raw, bool neg, bool abs) {
  if (!neg && !abs)
    return raw;
  Id f = t.m.Bitcast(t.t_f, raw);
  if (abs)
    f = t.m.ExtInst(t.t_f, GLSLstd450FAbs, {f});
  if (neg)
    f = t.m.Emit(spv::Op::OpFNegate, t.t_f, {f});
  return t.m.Bitcast(t.t_u, f);
}

// Apply one operand's SDWA sub-dword selection to its raw 32-bit value.
Id SdwaSelect(Translator& t, Id raw, u32 sel, bool sext) {
  if (sel >= 6)
    return raw;
  const u32 bits = sel < 4 ? 8u : 16u;
  const u32 off = sel < 4 ? sel * 8u : (sel - 4) * 16u;
  const Id shifted = t.Shr(raw, t.U32(off));
  if (!sext)
    return t.And(shifted, t.U32(bits == 8 ? 0xFFu : 0xFFFFu));
  return t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_u,
                  {shifted, t.U32(0), t.U32(bits)});
}

void ResolveValuSrc0(const Inst& inst,
                     u32 src0,
                     u32& field,
                     u32& literal) {
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
  const u32 w = inst.raw[0], w1 = inst.raw[1];
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
  // Compute reaches guest memory through the shared CS resource model (set-0
  // storage buffers), not through the graphics cbuf/vertex-fetch bindings.
  if (sc.is_cs && EmitCsMemory(t, inst, sc))
    return;
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
      if (!RdnaEmitSop2(t, inst))
        gpu::gcn::EmitSop2(t, inst);
      break;
    case Enc::kSopc:
      gpu::gcn::EmitSopc(t, inst);
      break;
    case Enc::kSopk:
      if (inst.opcode >= 0x17 && inst.opcode <= 0x1A)
        break;  // per-counter waits; translated memory operations are ordered
      if (inst.opcode == 0x10) {
        const u32 sdst = (w >> 16) & 0x7f;
        const u32 simm = static_cast<u32>(
            static_cast<i32>(static_cast<i16>(w & 0xffff)));
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
                 inst.opcode != 0x0F && inst.opcode != 0x10 &&
                 inst.opcode != 0x1E &&
                 inst.opcode != 0x1F && inst.opcode != 0x20 &&
                 inst.opcode != 0x21 && inst.opcode != 0x23) {
        gpu::gcn::WarnUnsupported("sopp.rdna", inst.opcode, w, w1);
      }
      // Branches are emitted by the CFG; waits/hints are synchronous. 0x10 is
      // s_sendmsg, whose only use in an NGG stage is GS_ALLOC_REQ (reserving
      // vertex/primitive slots in the hardware's own export space).
      break;
    case Enc::kSmrd:
      RdnaEmitSmem(t, inst, sc);
      break;
    case Enc::kVop1: {
      const u32 op = inst.opcode, vdst = (w >> 17) & 0xFF;
      const u32 raw0 = w & 0x1FF;
      u32 src0, lit;
      ResolveValuSrc0(inst, raw0, src0, lit);
      if (raw0 == 249) {  // SDWA: apply the source's sub-dword selection
        const SdwaMod sd = DecodeSdwa(inst, 0, false);
        if (sd.dst_sel != 6 || sd.src0_sel > 6)
          gpu::gcn::WarnUnsupported("vop1.sdwa-mod", op, w, w1);
        gpu::gcn::EmitVop1(
            t, op, vdst,
            t.m.Bitcast(t.t_f,
                        SdwaFloatMod(t,
                                     SdwaSelect(t, t.SrcRaw(src0, 0),
                                                sd.src0_sel, sd.src0_sext),
                                     sd.src0_neg, sd.src0_abs)));
        if (sd.omod) {
          const float k = sd.omod == 1 ? 2.f : sd.omod == 2 ? 4.f : 0.5f;
          t.SetVgF(vdst, t.FMul(t.VgF(vdst), t.F32(k)));
        }
        if (sd.clamp)
          t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450FClamp,
                                     {t.VgF(vdst), t.F32(0.f), t.F32(1.f)}));
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
      const u32 op = inst.opcode, vdst = (w >> 17) & 0xFF;
      const u32 vsrc1 = (w >> 9) & 0xFF;
      const u32 raw0 = w & 0x1FF;
      u32 src0, lit;
      ResolveValuSrc0(inst, raw0, src0, lit);
      if (raw0 == 249 && op >= 0x2b && op <= 0x2d) {
        gpu::gcn::WarnUnsupported("vop2.fma-sdwa", op, w, w1);
        break;
      }
      u32 src1 = 256 + vsrc1;
      Id sdwa0 = 0, sdwa1 = 0;
      bool sdwa_clamp = false;
      u32 sdwa_omod = 0;
      if (raw0 == 249) {
        const SdwaMod sd = DecodeSdwa(inst, vsrc1, false);
        if (sd.dst_sel != 6 || sd.src0_sel > 6 || sd.src1_sel > 6)
          gpu::gcn::WarnUnsupported("vop2.sdwa-mod", op, w, w1);
        // CLAMP and OMOD are applied to the RESULT, so they need no operand
        // rewriting -- the shared emitter writes vdst and we scale/saturate it
        // afterwards. OMOD is a real gfx10 SDWA field: `v_mul_f32_sdwa ... mul:2`
        // is what Minecraft's world shaders use, and rejecting it dropped the
        // whole shader.
        sdwa_clamp = sd.clamp;
        sdwa_omod = sd.omod;
        src1 = sd.src1;
        sdwa0 = SdwaFloatMod(
            t, SdwaSelect(t, t.SrcRaw(sd.src0, 0), sd.src0_sel, sd.src0_sext),
            sd.src0_neg, sd.src0_abs);
        sdwa1 = SdwaFloatMod(
            t, SdwaSelect(t, t.SrcRaw(sd.src1, 0), sd.src1_sel, sd.src1_sext),
            sd.src1_neg, sd.src1_abs);
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
        case 0x1F:
          // v_mac_f32 (D = S0*S1 + D). This slot is gfx1010 numbering, which
          // gfx1030 dropped -- the PS5's ISA keeps some RDNA1 assignments, so
          // decode it the way llvm-mc -mcpu=gfx1010 does, not gfx1030.
          t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450Fma,
                                     {t.m.Bitcast(t.t_f, s0u),
                                      t.m.Bitcast(t.t_f, s1u), t.VgF(vdst)}));
          break;
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
        case 0x20:  // v_madmk_f32 (gfx10 v_fmamk_f32): D = S0 * K + S1
        case 0x2C:
          t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450Fma,
                                     {t.m.Bitcast(t.t_f, s0u),
                                      t.m.Bitcast(t.t_f, t.U32(lit)),
                                      t.m.Bitcast(t.t_f, s1u)}));
          break;
        case 0x21:  // v_madak_f32 (gfx10 v_fmaak_f32): D = S0 * S1 + K
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
      // OMOD scales the result (1 = x2, 2 = x4, 3 = /2) and runs before CLAMP.
      if (sdwa_omod) {
        const float k = sdwa_omod == 1 ? 2.f : sdwa_omod == 2 ? 4.f : 0.5f;
        t.SetVgF(vdst, t.FMul(t.VgF(vdst), t.F32(k)));
      }
      if (sdwa_clamp)
        t.SetVgF(vdst, t.m.ExtInst(t.t_f, GLSLstd450FClamp,
                                   {t.VgF(vdst), t.F32(0.f), t.F32(1.f)}));
      break;
    }
    case Enc::kVop3p: {
      const u32 op = inst.opcode, vdst = w & 0xFF;
      // Componentwise packed operations select each source's low half for the
      // low result and high half for the high result.
      if ((w & 0x0000FF00u) != 0x00004000u ||
          (w1 & 0xF8000000u) != 0x18000000u) {
        gpu::gcn::WarnUnsupported("vop3p.modifier", op, w, w1);
        break;
      }
      const u32 p0 = w1 & 0x1FF, p1 = (w1 >> 9) & 0x1FF;
      const u32 p2 = (w1 >> 18) & 0x1FF;
      if (!RdnaEmitVop3p(t, op, vdst, p0, p1, p2, inst.literal))
        gpu::gcn::WarnUnsupported("vop3p", op, w, w1);
      break;
    }
    case Enc::kVop3: {
      const u32 rdna_op = inst.opcode;
      u32 op = rdna_op;
      const u32 vdst = w & 0xFF;
      // VOP3-form v_cndmask is the VOP2 alias 0x101 on RDNA2 (0x01 renumber);
      // GFX7's explicit-S2 cndmask VOP3 op is 0x100.
      if (op == 0x101)
        op = 0x100;
      const u32 s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF;
      const u32 s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
      const bool vop3b = RdnaVop3HasSdst(op);
      const u32 sdst = vop3b ? ((w >> 8) & 0x7F) : 106;
      const u32 abs = vop3b ? 0 : ((w >> 8) & 7);
      const u32 op_sel = vop3b ? 0 : ((w >> 11) & 0xF);
      const bool clamp = (w >> 15) & 1;
      const u32 omod = (w1 >> 27) & 3;
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
      if (rdna_op == 0x11F) {
        // VOP3 form of v_mac_f32 (D = S0*S1 + D), the alias of VOP2 0x1f.
        Id r = t.m.ExtInst(t.t_f, GLSLstd450Fma,
                           {t.SrcF(s0, inst.literal, neg & 1, abs & 1),
                            t.SrcF(s1, inst.literal, neg & 2, abs & 2),
                            t.VgF(vdst)});
        if (clamp)
          r = t.m.ExtInst(t.t_f, GLSLstd450FClamp, {r, t.F32(0.f), t.F32(1.f)});
        t.SetVgF(vdst, r);
        break;
      }
      if (rdna_op >= 0x100 && rdna_op < 0x140 && rdna_op != 0x101 &&
          !RdnaSharedVop2(rdna_op - 0x100)) {
        gpu::gcn::WarnUnsupported("vop3.alias.rdna", rdna_op, w, w1);
        break;
      }
      if (op == 0x141) {
        // v_mad_f32 (D = S0*S1 + S2), gfx1010 numbering -- see the VOP2 0x1f
        // note. Minecraft's shaders use it for the classic *2-1 remap.
        Id r = t.m.ExtInst(t.t_f, GLSLstd450Fma,
                           {t.SrcF(s0, inst.literal, neg & 1, abs & 1),
                            t.SrcF(s1, inst.literal, neg & 2, abs & 2),
                            t.SrcF(s2, inst.literal, neg & 4, abs & 4)});
        if (clamp)
          r = t.m.ExtInst(t.t_f, GLSLstd450FClamp, {r, t.F32(0.f), t.F32(1.f)});
        t.SetVgF(vdst, r);
        break;
      }
      if ((op >= 0x161 && op <= 0x163) || op == 0x16b) {
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
      const u32 op = inst.opcode;
      u32 vsrc1 = (w >> 9) & 0xFF;
      const u32 raw0 = w & 0x1FF;
      u32 src0, lit;
      ResolveValuSrc0(inst, raw0, src0, lit);
      if (inst.extension == gpu::gcn::InstExtension::kSdwa) {
        const SdwaMod sd = DecodeSdwa(inst, vsrc1, false);
        if (sd.src0_sel > 6 || sd.src1_sel > 6) {
          gpu::gcn::WarnUnsupported("vopc.sdwa-mod", op, w, w1);
          break;
        }
        // A compare has no VGPR destination, so SDWA reuses those bits (which
        // carry DST_SEL/CLAMP/OMOD elsewhere) for the scalar one: SD picks
        // SDST over VCC.
        const u32 dst = ((w1 >> 15) & 1) ? ((w1 >> 8) & 0x7F) : 106u;
        const Id a = SdwaFloatMod(
            t, SdwaSelect(t, t.SrcRaw(sd.src0, 0), sd.src0_sel, sd.src0_sext),
            sd.src0_neg, sd.src0_abs);
        const Id b = SdwaFloatMod(
            t, SdwaSelect(t, t.SrcRaw(sd.src1, 0), sd.src1_sel, sd.src1_sext),
            sd.src1_neg, sd.src1_abs);
        gpu::gcn::EmitVopc(t, op, t.m.Bitcast(t.t_f, a), t.m.Bitcast(t.t_f, b),
                           a, b, dst);
        break;
      }
      u32 vopc_src1 = 256 + vsrc1;
      if (raw0 == 249)
        vopc_src1 = DecodeSdwa(inst, vsrc1, false).src1;
      // RDNA2 f16 compares sit at 0xC8-0xCF, which the GFX7-numbered EmitVopc
      // would read as u32 integer compares; run the float predicate (op-0xC8)
      // on the low-half f16 operands instead. u32 compares stay at 0xC0-0xC7.
      if (op >= 0xC8 && op <= 0xCF) {
        auto f16 = [&](u32 f) {
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
      const u32 chan = (w >> 8) & 3, attr = (w >> 10) & 0x3F;
      const u32 op = (w >> 16) & 3, vdst = (w >> 18) & 0xFF;
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
      // Raw loads the shader indexes itself go to the set-2 storage buffer the
      // planner assigned; the MUBUF field layout is the same as GCN's.
      if (inst.enc == Enc::kMubuf && inst.opcode >= 0x08 &&
          inst.opcode <= 0x0f) {
        gpu::gcn::EmitGfxMubuf(t, inst, sc);
        break;
      }
      // A store has nowhere to land: the set-2 window is a per-draw staging copy
      // that is never read back, and draws sharing a resource share one window.
      // Drop it and keep going -- a rejected shader drops the whole draw, while
      // a draw missing a side-effect write still rasterizes its geometry.
      if (inst.enc == Enc::kMubuf ? RdnaMubufStore(inst.opcode)
                                  : RdnaMtbufStore(inst.opcode)) {
        gpu::gcn::AuditNote(
            inst.enc == Enc::kMtbuf ? "mtbuf.store.rdna" : "mubuf.store.rdna",
            inst.opcode);
        if (ShDbg() && !g_warned_store) {
          g_warned_store = true;
          BASE_LOGI("gcnspv", "dropped buffer store op={:#x} @pc={:04x}",
                    inst.opcode, inst.pc);
        }
        break;
      }
      if (inst.opcode > 0x03) {  // atomics and the d16 forms
        gpu::gcn::WarnUnsupported(
            inst.enc == Enc::kMtbuf ? "mtbuf.rdna" : "mubuf.rdna", inst.opcode,
            w, w1);
        break;
      }
      const bool tfe = (w1 >> 23) & 1;
      const bool lds = inst.enc == Enc::kMubuf && ((w >> 16) & 1);
      const u32 soffset = (w1 >> 24) & 0xff;
      if (tfe || lds) {
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
        for (u32 c = 0; c < vs.num_comps; c++)
          t.SetVgF(vs.dest_vgpr + c, vs.num_comps == 1
                                         ? val
                                         : t.m.CompositeExtract(t.t_f, val, c));
        break;
      }
      // An untyped format load converts through the V#'s format, which only the
      // resolved descriptor knows; it reads the set-2 window (its index is
      // per-lane) rather than a UBO.
      if (inst.enc == Enc::kMubuf) {
        if (!RdnaEmitBufFormatLoad(t, inst, sc))
          gpu::gcn::WarnUnsupported("mubuf.format-conversion.rdna",
                                    inst.opcode, w, w1);
        break;
      }
      // Past the fetch path the load really is emitted, so a scalar byte offset
      // would move the read and is not expressible against a bound UBO. Sony's
      // compiler parks a scratch SGPR (vcc_hi) in this field even when the
      // offset is zero, so only a fetch -- replaced above by its vertex input --
      // can ignore it.
      if (soffset != 125 && soffset != 128) {
        gpu::gcn::WarnUnsupported("buffer.control.rdna", inst.opcode, w, w1);
        break;
      }
      const u32 srsrc = ((w1 >> 16) & 0x1F) * 4;
      u32 binding;
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
      const u32 nc = (inst.opcode & 3) + 1;
      // A typed constant load delivers the buffer's dwords unchanged only when
      // its format is 32-bit per channel; narrower ones would need unpacking.
      if (inst.enc == Enc::kMtbuf) {
        const u32 fmt = (w >> 19) & 0x7F;
        if (!(fmt >= 20 && (fmt <= 22 || (fmt >= 62 && fmt <= 77))))
          gpu::gcn::WarnUnsupported("mtbuf.fmt", fmt, w, w1);
      }
      const u32 inst_offset = w & 0xFFF, vdata = (w1 >> 8) & 0xFF;
      const bool idxen = (w >> 13) & 1, offen = (w >> 12) & 1;
      if (idxen) {
        gpu::gcn::WarnUnsupported("mtbuf.descriptor-stride.rdna", inst.opcode,
                                  w, w1);
        break;
      }
      const u32 vaddr = w1 & 0xFF;
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
      for (u32 k = 0; k < nc; k++)
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
      const u32 dim = (w >> 3) & 0x7;
      // dim 3 (cube) arrives with the face already selected, so it is the
      // same (s, t, layer) address the 2D-array path takes.
      const bool arrayed_dim = dim == 3 || dim == 5;
      if (dim != 1 && !arrayed_dim) {
        gpu::gcn::WarnUnsupported("mimg.dim", dim, w, w1);
        break;
      }
      if (((w >> 15) & 1) && arrayed_dim) {
        gpu::gcn::WarnUnsupported("mimg.r128-array.rdna", inst.opcode, w, w1);
        break;
      }
      lowered.raw[0] = (w & ~0x4000u) | (arrayed_dim ? 0x4000u : 0u);
      // NSA names every address component explicitly. Load them before emitting
      // the image operation so aliases cannot overwrite a later source and no
      // architectural VGPRs are used as scratch storage.
      const u32 nsa = (w >> 1) & 0x3;
      // A cube sample's s/t reach the hardware biased by +1.5 (the compiler
      // emits v_madak_f32 s, sc, rcp(|ma|), 1.5); gfx10's cube addressing
      // consumes that range. The 2D-array lowering wants a plain [0,1], so
      // take the bias back off. Without this every face samples its clamped
      // edge texel and comes out one flat colour.
      const bool cube = dim == 3;
      if (nsa || cube) {
        std::array<Id, 13> address{};
        const u32 vaddr = w1 & 0xFF;
        if (nsa) {
          address[0] = t.Vg(vaddr);
          for (u32 d = 0; d < nsa; d++)
            for (u32 c = 0; c < 4; c++)
              address[1 + d * 4 + c] = t.Vg((inst.raw[2 + d] >> (c * 8)) & 0xFF);
        } else {
          for (u32 c = 0; c < address.size(); c++)
            address[c] = t.Vg(vaddr + c);
        }
        if (cube)
          for (u32 c = 0; c < 2; c++)
            address[c] = t.m.Bitcast(
                t.t_u, t.FSub(t.m.Bitcast(t.t_f, address[c]), t.F32(1.0f)));
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

std::vector<u32> BlockStarts(const Program& program, u32 max_pc) {
  std::vector<u32> leaders{0};
  for (const Inst& inst : program) {
    const int k = BranchKind(inst);
    if (k == 0)
      continue;
    leaders.push_back(inst.pc + inst.size);
    if (k >= 1 && k <= 7) {
      const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
      leaders.push_back(static_cast<u32>(static_cast<i32>(inst.pc) +
                                              static_cast<i32>(inst.size) +
                                              simm));
    }
  }
  std::sort(leaders.begin(), leaders.end());
  leaders.erase(std::unique(leaders.begin(), leaders.end()), leaders.end());
  std::vector<u32> starts;
  for (u32 l : leaders)
    if (l < max_pc)
      starts.push_back(l);
  return starts;
}

bool ForceCfg() {
  return kGpuSpirvCfg;
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
  u32 semantic, num_comps, dest_vgpr, table_sgpr, dword_off;
  u32 pc =
      ~0u;  // inline fetch MUBUF pc (~0 = standalone fetch sub-shader)
  u32 inst_format = 0;  // typed (MTBUF) fetch format; 0 = take the V#'s
  u32 inst_offset = 0;  // byte offset immediate: this attr's field offset
};

// Scan an instruction stream for the s_load_dwordx4(V# table) +
// buffer_load_format fetch pattern. Works on both a stand-alone fetch
// sub-shader and the head of an NGG vertex program that fetches inline (PS5 AGC
// frequently inlines the fetch), so buffer_load_format never reaches
// RdnaEmitInst as an unsupported op.
std::vector<FetchAttr> ParseFetchInsts(const Program& insts) {
  std::vector<FetchAttr> out;
  u32 sem = 0;  // dense vertex-input location (per-vertex fetches only)
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
    const u32 vdata = (in.raw[1] >> 8) & 0xFF;
    const u32 srsrc = ((in.raw[1] >> 16) & 0x1F) * 4;
    const u32 nc = (in.opcode & 3) + 1;
    const bool idxen = (in.raw[0] >> 13) & 1, offen = (in.raw[0] >> 12) & 1;
    const u32 inst_offset = in.raw[0] & 0xFFF;
    const u32 vaddr = in.raw[1] & 0xFF, soffset = (in.raw[1] >> 24) & 0xFF;
    // If srsrc's V# was loaded via s_load from a user_data pointer, this fetch
    // reads a TABLE of V#s at that pointer: table_sgpr = the s_load's sbase and
    // this attribute is that s_load's 16-byte (4-dword) entry. Otherwise the V#
    // is inline in user data at srsrc (table_sgpr = srsrc, dword_off = 0).
    const auto chain = chained_loads.find(in.pc);
    const bool chained = chain != chained_loads.end();
    const u32 table_sgpr = chained ? chain->second.first : srsrc;
    const u32 dword_off = chained ? chain->second.second * 4 : 0;
    const bool vtx = BufLoadIsVertexFetch(in, chained);
    if (ShDbg())
      BASE_LOGI("gcnspv",
                "buf_load nc={} vdst=v{} srsrc=s{} idxen={} offen={} "
                "vaddr=v{} soffset=s{} ioff={} -> {} (table_sgpr=s{} doff={})",
                nc, vdata, srsrc, idxen, offen, vaddr, soffset, inst_offset,
                vtx ? "vertex-attr" : "const-ubo", table_sgpr, dword_off);
    // Only a genuine per-vertex fetch becomes a vertex input. A constant load
    // is left for the UBO path (RdnaPlanBufLoadCbufs assigns the same table
    // slots).
    if (vtx) {
      out.push_back({sem, nc, vdata, table_sgpr, dword_off, in.pc,
                     typed ? ((in.raw[0] >> 19) & 0x7F) : 0u, inst_offset});
      sem++;
    }
  }
  return out;
}

std::vector<FetchAttr> ParseFetch(u64 fetch_addr) {
  constexpr u64 kMaxFetchBytes = 256 * sizeof(u32);
  if (!gpu::gcn::InGuest(fetch_addr) ||
      !gpu::IsReadableRange(fetch_addr, kMaxFetchBytes))
    return {};
  const auto* code = reinterpret_cast<const u32*>(fetch_addr);
  std::vector<FetchAttr> attrs = ParseFetchInsts(Decode(code, 256));
  for (FetchAttr& attr : attrs)
    attr.pc = ~0u;
  return attrs;
}

// Address of the VS being translated, for shader-specific debug knobs.
thread_local u64 g_vs_addr = 0;

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
                  u32 sgpr_base,
                  u32 count) {
  const Id p_u = t.m.TypePointer(spv::StorageClass::PushConstant, t.t_u);
  for (u32 i = 0; i < std::min(count, 32u); i++)
    t.SetSg(
        sgpr_base + i,
        t.m.Load(t.t_u, t.m.AccessChain(p_u, user_data, {t.U32(0), t.U32(i)})));
}

bool TranslateVs(const Program& program,
                 const u32* vs_user_data,
                 const std::unordered_set<u32>& flat_attrs,
                 Recompiled& r,
                 Translator& t,
                 bool gl_clip_space,
                 u32 user_sgprs) {
  if (ShDbg())
    DumpProgram(program, "vs");
  const u64 fetch =
      user_sgprs >= 2
          ? (static_cast<u64>(vs_user_data[1] & 0xFFFF) << 32) |
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
      BASE_LOGI("gcnspv", "vs attr loc={} nc={} vgpr={} pc={:#x}",
                a.semantic, a.num_comps, a.dest_vgpr, a.pc);
  t.InitTypes();

  std::vector<Id> iface;
  const Id pos_out =
      t.m.Variable(t.m.TypePointer(spv::StorageClass::Output, t.t_v4),
                   spv::StorageClass::Output);
  t.m.Decorate(pos_out, spv::Decoration::BuiltIn,
               {static_cast<u32>(spv::BuiltIn::Position)});
  iface.push_back(pos_out);

  const Id user_data = DeclareUserData(t);
  const Id main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  SeedUserData(t, user_data, 8, user_sgprs);

  // NGG merged-wave prologue: the VS derives its EXEC/lane bookkeeping from
  // merged_wave_info in s3 (verts-in-wave [7:0], prims [15:8]); model a
  // 1-vert/1-prim wave so that math yields a live lane instead of zeros.
  t.SetSg(3, t.U32(0x0101));

  std::unordered_map<u32, StageContext::VfetchSeed> vfetch_seed;
  Id vertex_index = 0;
  if (attrs.empty()) {  // procedural VS: seed the ABI VGPRs from Vulkan
                        // built-ins
    const Id p_in_u = t.m.TypePointer(spv::StorageClass::Input, t.t_u);
    vertex_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
    const Id instance_index = t.m.Variable(p_in_u, spv::StorageClass::Input);
    t.m.Decorate(vertex_index, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::VertexIndex)});
    t.m.Decorate(instance_index, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::InstanceIndex)});
    iface.push_back(vertex_index);
    iface.push_back(instance_index);
    const Id vertex = t.m.Load(t.t_u, vertex_index);
    t.SetVg(0, vertex);
    const Id instance = t.m.Load(t.t_u, instance_index);
    t.SetVg(1, instance);
    t.SetVg(3, instance);
    // An NGG merged shader takes its vertex index in v5, not v0: v0..v4 carry
    // the GS half's packed vertex offsets, primitive id and invocation id, all
    // of which a shader with no attributes overwrites before reading. Demon's
    // Souls builds every fullscreen pass out of v5 alone, so leaving it zero
    // collapsed each quad to a point and nothing rasterised.
    t.SetVg(5, vertex);
  }

  Id first_attr_var = 0;
  u32 first_attr_comps = 0;
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
    for (u32 c = 0; c < a.num_comps; c++) {
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
                       false, a.inst_format, a.pc, a.inst_offset});
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
  // A fetch already lifted to a vertex input needs no buffer of its own.
  std::unordered_set<u32> lifted;
  for (const FetchAttr& a : attrs)
    if (a.pc != ~0u)
      lifted.insert(a.pc);
  RdnaPlanGfxBuffers(program, 0, &lifted, r.vs_bufs, sc.gfx_buf_bind);
  if (ShDbg())
    BASE_LOGI("gcnspv", "vs planned {} cbufs", r.vs_cbufs.size());
  // DELTA_GPU_DBGPOS=<vs address>[:<dword offset>]: recompute this one shader's
  // position export from its position input and a 4x4 transform in its cbuffer
  // (address 0 = every shader; the offset picks which matrix in the window).
  static const u64 dbg_pos_vs =
      kDbgPos ? std::strtoull(kDbgPos, nullptr, 0) : ~0ull;
  static const u32 dbg_pos_off = [] {
    const char* c = kDbgPos ? std::strchr(kDbgPos, ':') : nullptr;
    return c ? (u32)std::strtoul(c + 1, nullptr, 0) : 0u;
  }();
  // ...:<world binding> applies a 4x3 world matrix from that binding first, so
  // the probe can reproduce a world-then-view-projection chain.
  static const int dbg_pos_world = [] {
    const char* c = kDbgPos ? std::strchr(kDbgPos, ':') : nullptr;
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
  if (kGpuForcequad) {
    // A fetch-path VS never created a VertexIndex input, and its v0 is
    // clobbered by the vertex fetch -- bind a real VertexIndex here so the quad
    // is valid either way.
    Id vidx_var = vertex_index;
    if (!vidx_var) {
      vidx_var = t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_u),
                              spv::StorageClass::Input);
      t.m.Decorate(vidx_var, spv::Decoration::BuiltIn,
                   {static_cast<u32>(spv::BuiltIn::VertexIndex)});
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
  if (kRawPos != 0.f && first_attr_var && first_attr_comps >= 2) {
    const Id comp_ty = first_attr_comps == 2   ? t.t_v2
                       : first_attr_comps == 3 ? t.t_v3
                                               : t.t_v4;
    const Id val = t.m.Load(comp_ty, first_attr_var);
    t.m.Store(pos_out,
              t.m.CompositeConstruct(
                  t.t_v4,
                  {t.FMul(t.m.CompositeExtract(t.t_f, val, 0), t.F32(kRawPos)),
                   t.FMul(t.m.CompositeExtract(t.t_f, val, 1), t.F32(kRawPos)),
                   t.F32(0.f), t.F32(1.f)}));
  }

  // DELTA_GPU_POSPROBE: squash the shader's own clip position into the viewport
  // (x/(1+|x|) after the perspective divide). Any FINITE position then covers
  // pixels, so a still-black target means w is zero/NaN rather than the
  // geometry merely landing off-screen.
  if (kGpuPosprobe) {
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

bool TranslatePs(const Program& program,
                 const std::unordered_set<u32>& flat_attrs,
                 u32 ps_input_ena,
                 Recompiled& r,
                 Translator& t,
                 u32 user_sgprs) {
  if (ShDbg())
    DumpProgram(program, "ps");
  std::vector<Id> iface;
  StageContext sc;
  sc.is_ps = true;
  sc.r = &r;
  sc.iface = &iface;
  sc.flat_attrs = &flat_attrs;
  sc.skip_launch_movs = LaunchExecMovPcs(program);
  if (!RdnaPlanCbufs(program, static_cast<u32>(r.vs_cbufs.size()),
                     r.ps_cbufs, sc.cbuf_bind, sc.smem_cbuf_by_pc))
    return false;
  RdnaPlanGfxBuffers(program, static_cast<u32>(r.vs_bufs.size()),
                     nullptr, r.ps_bufs, sc.gfx_buf_bind);
  const gpu::gcn::MimgBindingPlan mimg_plan = RdnaPlanMimg(program);
  if (mimg_plan.binding_srsrc.size() > StageContext::kMaxPsSamplers)
    return false;
  sc.mimg_plan = &mimg_plan;  // borrowed by EmitBody
  for (u32 i = 0; i < mimg_plan.binding_srsrc.size(); i++)
    r.ps_texs.push_back(
        {i, mimg_plan.binding_srsrc[i], mimg_plan.binding_storage[i]});

  const Id user_data = DeclareUserData(t);
  sc.main_fn = t.m.BeginFunction(t.t_void, t.t_fn);
  SeedUserData(t, user_data, 0, user_sgprs);
  gpu::gcn::SeedPsInputVgprs(t, ps_input_ena, iface);

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
  if (kGpuForcecolor && kGpuNokill)
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
  if (sc.wrote_color && kills_lanes && !kGpuNokill) {
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
    if (!kGpuNokill) {
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
  if (kGpuPsuv) {
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

// Shared with the compute stage (rdna_compute.cc), which lowers the same
// branches through the same per-instruction dispatch.
void EmitCfg(Translator& t, const Program& program, StageContext& sc) {
  const u32 max_pc =
      program.empty() ? 0 : program.back().pc + program.back().size;
  const std::vector<u32> starts = BlockStarts(program, max_pc);
  const u32 num_blocks = static_cast<u32>(starts.size());
  const u32 kExit = num_blocks;
  const auto block_of = [&](u32 pc) -> u32 {
    if (pc >= max_pc)
      return kExit;
    u32 b = 0;
    for (u32 i = 0; i < num_blocks; i++)
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
  std::vector<std::pair<u32, Id> > cases;
  for (u32 i = 0; i < num_blocks; i++)
    cases.push_back({i, case_labels[i]});
  t.m.Switch(state, exit_blk, cases);

  for (u32 bi = 0; bi < num_blocks; bi++) {
    t.m.OpenBlock(case_labels[bi]);
    const u32 blk_start = starts[bi];
    const u32 blk_end = (bi + 1 < num_blocks) ? starts[bi + 1] : max_pc;
    bool terminated = false;
    for (const Inst& inst : program) {
      if (inst.pc < blk_start || inst.pc >= blk_end)
        continue;
      const int k = BranchKind(inst);
      if (k == 0) {
        RdnaEmitInst(t, inst, sc);
        continue;
      }
      const u32 fall = (bi + 1 < num_blocks) ? bi + 1 : kExit;
      if (k == 8) {
        t.SetState(kExit);
      } else if (k == 1) {
        const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
        t.SetState(block_of(
            static_cast<u32>(static_cast<i32>(inst.pc) +
                                  static_cast<i32>(inst.size) + simm)));
      } else {
        const i32 simm = static_cast<i16>(inst.raw[0] & 0xFFFF);
        const u32 target = block_of(
            static_cast<u32>(static_cast<i32>(inst.pc) +
                                  static_cast<i32>(inst.size) + simm));
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

Recompiled Recompile(const u32* vs_code,
                     const u32* ps_code,
                     const u32* vs_user_data,
                     const u32* ps_user_data,
                     u32 ps_input_ena,
                     bool gl_clip_space,
                     u32 vs_user_sgprs,
                     u32 ps_user_sgprs) {
  Recompiled r;
  if (!vs_code || !vs_user_data || !ps_user_data)
    return r;

  constexpr u64 kMaxShaderBytes = 4096 * sizeof(u32);
  const u64 vs_address = reinterpret_cast<uintptr_t>(vs_code);
  const u64 ps_address = reinterpret_cast<uintptr_t>(ps_code);
  if (!gpu::IsReadableRange(vs_address, kMaxShaderBytes) ||
      (ps_code && !gpu::IsReadableRange(ps_address, kMaxShaderBytes)))
    return r;

  const Program vs_program = ReachableProgram(DecodeShader(vs_code, 4096));
  const Program ps_program =
      ps_code ? ReachableProgram(DecodeShader(ps_code, 4096)) : Program{};

  // V_INTERP_MOV P0 reads a per-primitive (flat) parameter; represent those
  // locations as flat varyings in both stages.
  std::unordered_set<u32> flat_attrs;
  for (const Inst& inst : ps_program)
    if (inst.enc == Enc::kVintrp && inst.opcode == 2 &&
        (inst.raw[0] & 0xFF) == 2)
      flat_attrs.insert((inst.raw[0] >> 10) & 0x3F);

  Translator tv;
  tv.rdna_sources = true;
  gpu::gcn::ResetUnsupported();
  g_vs_addr = reinterpret_cast<uintptr_t>(vs_code);
  // The merged NGG stage is launched with its user data at s8 (kUdBase in the
  // command processor, and what SeedUserData assumes here); a PS gets it at s0.
  g_stage_bufs = ResolveBuffers(vs_code, vs_user_data, vs_user_sgprs, 8);
  g_warned_store = false;
  if (!TranslateVs(vs_program, vs_user_data, flat_attrs, r, tv, gl_clip_space,
                   vs_user_sgprs) ||
      gpu::gcn::HadUnsupported()) {
    if (ShDbg() || kDrawCensus)
      BASE_LOGI("gcnspv", "vs {:#x} rejected: {}",
                (unsigned long)reinterpret_cast<uintptr_t>(vs_code),
                gpu::gcn::UnsupportedOps().c_str());
    return r;
  }
  Translator tp;
  tp.rdna_sources = true;
  g_ps_addr = reinterpret_cast<uintptr_t>(ps_code);
  g_stage_bufs = ResolveBuffers(ps_code, ps_user_data, ps_user_sgprs, 0);
  g_warned_store = false;
  tp.InitTypes();
  gpu::gcn::ResetUnsupported();
  if (ps_code ? !TranslatePs(ps_program, flat_attrs, ps_input_ena, r, tp,
                             ps_user_sgprs)
              : !TranslateDepthOnlyPs(tp))
    return r;
  if (gpu::gcn::HadUnsupported()) {
    if (ShDbg() || kDrawCensus)
      BASE_LOGI("gcnspv", "ps {:#x} rejected: {}",
                (unsigned long)reinterpret_cast<uintptr_t>(ps_code),
                gpu::gcn::UnsupportedOps().c_str());
    return r;
  }

  const std::vector<u32> vs = tv.m.Assemble();
  const std::vector<u32> ps = tp.m.Assemble();
  // gfx10.3 RECTLIST draws reach us as three corners of a rectangle; Vulkan has
  // no such topology, so carry the same expansion stage the GFX7 path uses.
  const std::vector<u32> gs =
      gpu::gcn::EmitRectListGeometry(r.num_params, flat_attrs);
  std::string err;
  if (!gpu::gcn::spirv::Validate(vs, &err)) {
    if (gpu::gcn::TraceEnabled())
      BASE_LOGI("rdna", "VS invalid: {}", err.c_str());
    return r;
  }
  if (!gpu::gcn::spirv::Validate(ps, &err)) {
    if (gpu::gcn::TraceEnabled())
      BASE_LOGI("rdna", "PS invalid: {}", err.c_str());
    return r;
  }
  // DELTA_GPU_SPVDUMP=<dir>: write each recompiled stage's SPIR-V so it can be
  // read with spirv-dis (the only way to see what the position math became).
  if (const char* dir = kSpvDump) {
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
    BASE_LOGI("rdna", "RECTLIST GS invalid: {}", gs_err.c_str());
  r.ok = !r.vs_spirv.empty() && !r.fs_spirv.empty();
  return r;
}

}  // namespace gpu::rdna

#endif  // DELTA_HAVE_SPIRV_BACKEND
