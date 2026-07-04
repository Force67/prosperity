/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN -> SPIR-V translator. See gcn_spirv.h. Mirrors gcn_translate.cpp's
 * straight-line VS/PS handling, emitting SPIR-V via spv_emit instead of GLSL.
 */

#include "gcn_spirv.h"

#ifndef DELTA_HAVE_SPIRV_BACKEND
// Backend disabled at build time (SPIRV-Tools/Headers unavailable): the GLSL path
// is the only recompiler. recompileSpirv always declines so callers fall back.
namespace gpu::gcn {
bool recompileSpirv(const uint32_t *, const uint32_t *, const uint32_t *,
                    const uint32_t *, Recompiled &) { return false; }
}  // namespace gpu::gcn
#else

#include "../gcn_decode.h"
#include "spv_emit.h"
#include "spv_post.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <spirv/unified1/GLSL.std.450.h>

namespace gpu::gcn {
namespace {

namespace S = ::gpu::gcn::spirv;
using Id = S::Id;

bool inGuest(uint64_t a) { return a >= 0x1000000000ull && a < 0x20000000000ull; }
const bool g_dbg = std::getenv("DELTA_GPU_SHTRACE") != nullptr;

// Loud, deduplicated report of an instruction the translator does not implement,
// so it falls back to an approximation. Logged once per distinct (encoding,opcode)
// to stderr: silent wrong codegen is never acceptable, but a per-frame flood is
// useless, so we dedup. `enc`/`op` are the disassembly-level identity.
void warnUnsup(const char *enc, uint32_t op, uint32_t w0 = 0, uint32_t w1 = 0) {
  static std::unordered_set<uint64_t> seen;
  uint64_t key = (uint64_t)std::hash<std::string_view>{}(enc) ^ ((uint64_t)op << 40);
  if (seen.size() > 512 || !seen.insert(key).second) return;
  std::fprintf(stderr, "[gcnspv] UNSUPPORTED %s op=%#x (w0=%#x w1=%#x) -> approximated\n",
               enc, op, w0, w1);
}

// Fetch-shader attribute (mirrors gcn_translate.cpp's parseFetch).
struct FetchAttr { uint32_t semantic, numComps, destVgpr, tableSgpr, dwordOff; };
std::vector<FetchAttr> parseFetch(uint64_t fetchAddr) {
  std::vector<FetchAttr> out;
  if (!inGuest(fetchAddr)) return out;
  auto *code = reinterpret_cast<const uint32_t *>(fetchAddr);
  auto insts = decode(code, 256);
  struct Load { uint32_t tableSgpr, dwordOff; };
  std::unordered_map<uint32_t, Load> loads;
  uint32_t sem = 0;
  for (auto &in : insts) {
    uint32_t w = in.raw[0];
    if (in.enc == Enc::sop1 && (in.opcode == 0x20 || in.opcode == 0x21)) break;
    if (in.enc == Enc::sopp && in.opcode == 1) break;
    if (in.enc == Enc::smrd && in.opcode == 0x02) {
      uint32_t sdst = (w >> 15) & 0x7F, sbase = (w >> 9) & 0x3F, off = w & 0xFF;
      loads[sdst] = {sbase * 2u, off};
    } else if (in.enc == Enc::mubuf || in.enc == Enc::mtbuf) {
      uint32_t w1 = in.raw[1];
      uint32_t vdata = (w1 >> 8) & 0xFF;
      uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
      uint32_t nc = (in.opcode & 3) + 1;
      auto it = loads.find(srsrc);
      uint32_t tbl = it != loads.end() ? it->second.tableSgpr : 0;
      uint32_t doff = it != loads.end() ? it->second.dwordOff : 0;
      out.push_back({sem, nc, vdata, tbl, doff});
      sem++;
    }
  }
  return out;
}

// Translator context: the SPIR-V module + the register-file modelling helpers.
struct Tr {
  S::Module m;
  Id tVoid, tFn, tU, tI, tF, tBool, tV2, tV3, tV4;
  Id pPrivU, sgpr, vgpr;
  Id sccVar = 0;    // scalar condition code (s_cmp / scalar ALU -> s_cbranch_scc*)
  Id stateVar = 0;  // CFG block index for the while-switch dispatch
  Id pcVar = 0;          // cbuffer block (uniform buffer, set 1 binding 0)
  bool havePc = false;

  void initTypes() {
    tVoid = m.typeVoid();
    tFn = m.typeFunction(tVoid);
    tU = m.typeInt(32, false);
    tI = m.typeInt(32, true);
    tF = m.typeFloat(32);
    tBool = m.typeBool();
    tV2 = m.typeVec(tF, 2);
    tV3 = m.typeVec(tF, 3);
    tV4 = m.typeVec(tF, 4);
    pPrivU = m.typePointer(spv::StorageClass::Private, tU);
    Id arrSg = m.typeArray(tU, 128), arrVg = m.typeArray(tU, 256);
    sgpr = m.variable(m.typePointer(spv::StorageClass::Private, arrSg),
                      spv::StorageClass::Private, m.constNull(arrSg));
    vgpr = m.variable(m.typePointer(spv::StorageClass::Private, arrVg),
                      spv::StorageClass::Private, m.constNull(arrVg));
    m.name(sgpr, "sgpr");
    m.name(vgpr, "vgpr");
    sccVar = m.variable(pPrivU, spv::StorageClass::Private, m.constNull(tU));
    stateVar = m.variable(pPrivU, spv::StorageClass::Private, m.constNull(tU));
    m.name(sccVar, "scc");
    m.name(stateVar, "state");
  }

  // Seed EXEC all-active (sgpr[126]=1) at the start of the function body. In our
  // per-invocation (scalar-lane) model EXEC is a single "this lane active" bit, so
  // execz/execnz and s_and_saveexec behave for the common vectorised-if pattern.
  // Must be called after beginFunction (emits an OpStore).
  void seedExec() { stSg(126, m.constU32(1)); }

  // SCC / EXEC / CFG-state helpers.
  Id ldScc() { return m.load(tU, sccVar); }
  void stScc(Id v) { m.store(sccVar, v); }
  void stSccBool(Id b) { stScc(m.emit(spv::Op::OpSelect, tU, {b, m.constU32(1), m.constU32(0)})); }
  Id ldExec() { return ldSg(126); }
  Id ldState() { return m.load(tU, stateVar); }
  void stState(uint32_t s) { m.store(stateVar, m.constU32(s)); }
  void stStateId(Id s) { m.store(stateVar, s); }
  Id isNonZero(Id u) { return m.emit(spv::Op::OpINotEqual, tBool, {u, m.constU32(0)}); }
  Id isZero(Id u) { return m.emit(spv::Op::OpIEqual, tBool, {u, m.constU32(0)}); }

  // Declare the cbuffer as a uniform buffer: CB { uvec4 data[64]; } at descriptor
  // set 1, binding 0. Push constants can't hold it (the VS reads matrices past byte
  // 256, e.g. Undertale at cbuffer dwords 48 and 64); a UBO covers the full 1 KiB
  // window. Textures stay at set 0, so the texture path is unchanged.
  void ensurePc() {
    if (havePc) return;
    havePc = true;
    Id tUV4 = m.typeVec(tU, 4);
    Id arr = m.typeArray(tUV4, 64);
    m.decorate(arr, spv::Decoration::ArrayStride, {16});
    Id st = m.typeStruct({arr});
    m.decorate(st, spv::Decoration::Block);
    m.memberDecorate(st, 0, spv::Decoration::Offset, {0});
    pcVar = m.variable(m.typePointer(spv::StorageClass::Uniform, st),
                       spv::StorageClass::Uniform);
    m.decorate(pcVar, spv::Decoration::DescriptorSet, {1});
    m.decorate(pcVar, spv::Decoration::Binding, {0});
  }
  // Read cbuffer dword k (== uvec4 data[k>>2][k&3]) as a uint Id. Clamp the uvec4
  // index into the 64-element (1 KiB) window so an out-of-range constant index can't
  // produce an invalid SPIR-V access chain.
  Id pcDword(uint32_t k) {
    ensurePc();
    uint32_t v4 = (k >> 2) & 63;
    Id pU = m.typePointer(spv::StorageClass::Uniform, tU);
    Id ch = m.accessChain(pU, pcVar,
                          {m.constU32(0), m.constU32(v4), m.constU32(k & 3)});
    return m.load(tU, ch);
  }

  Id ptrSg(uint32_t i) { return m.accessChain(pPrivU, sgpr, {m.constU32(i)}); }
  Id ptrVg(uint32_t i) { return m.accessChain(pPrivU, vgpr, {m.constU32(i)}); }
  Id ldSg(uint32_t i) { return m.load(tU, ptrSg(i)); }
  Id ldVg(uint32_t i) { return m.load(tU, ptrVg(i)); }
  void stSg(uint32_t i, Id v) { m.store(ptrSg(i), v); }
  void stVg(uint32_t i, Id v) { m.store(ptrVg(i), v); }
  Id ldVgF(uint32_t i) { return m.bitcast(tF, ldVg(i)); }
  void stVgF(uint32_t i, Id vF) { stVg(i, m.bitcast(tU, vF)); }

  Id fconst(float v) { return m.constF32(v); }
  Id ext1(uint32_t op, Id a) { return m.extInst(tF, op, {a}); }
  Id ext2(uint32_t op, Id a, Id b) { return m.extInst(tF, op, {a, b}); }
  Id fmul(Id a, Id b) { return m.emit(spv::Op::OpFMul, tF, {a, b}); }
  Id fadd(Id a, Id b) { return m.emit(spv::Op::OpFAdd, tF, {a, b}); }
  Id fsub(Id a, Id b) { return m.emit(spv::Op::OpFSub, tF, {a, b}); }
  Id fdiv(Id a, Id b) { return m.emit(spv::Op::OpFDiv, tF, {a, b}); }
  Id fneg(Id a) { return m.emit(spv::Op::OpFNegate, tF, {a}); }

  // raw uint of a source operand field (mirrors srcRaw).
  Id srcRaw(uint32_t field, uint32_t literal) {
    if (field <= 127) return ldSg(field);
    if (field == 128) return m.constU32(0);
    if (field >= 129 && field <= 192) return m.constU32(field - 128);
    if (field >= 193 && field <= 208) return m.constU32((uint32_t)(-(int)(field - 192)));
    switch (field) {
      case 240: return m.constU32(0x3f000000u); case 241: return m.constU32(0xbf000000u);
      case 242: return m.constU32(0x3f800000u); case 243: return m.constU32(0xbf800000u);
      case 244: return m.constU32(0x40000000u); case 245: return m.constU32(0xc0000000u);
      case 246: return m.constU32(0x40800000u); case 247: return m.constU32(0xc0800000u);
    }
    if (field == 255) return m.constU32(literal);
    if (field >= 256) return ldVg(field - 256);
    return m.constU32(0);
  }
  // float source with neg/abs modifiers (mirrors srcF).
  Id srcF(uint32_t field, uint32_t literal, bool neg = false, bool abs = false) {
    Id f = m.bitcast(tF, srcRaw(field, literal));
    if (abs) f = ext1(GLSLstd450FAbs, f);
    if (neg) f = fneg(f);
    return f;
  }
};

// ---- VOP emitters (mirror gcn_translate.cpp Emit::vop*) ---------------------
void emitVop1(Tr &t, uint32_t op, uint32_t vdst, Id s0);
void emitVop2(Tr &t, uint32_t op, uint32_t vdst, Id s0, Id s1, uint32_t literal = 0);

// GFX6/7 (Liverpool / Sea Islands) VOP1 numbering (AMD CI ISA). An earlier table
// mis-numbered the transcendentals (rcp/rsq/sqrt/sin/cos), silently turning them
// into mov/sqrt/etc.; the numbers below are the real ones the recompiler decodes.
void emitVop1(Tr &t, uint32_t op, uint32_t vdst, Id s0) {
  auto setF = [&](Id f) { t.stVgF(vdst, f); };
  auto setU = [&](Id u) { t.stVg(vdst, u); };
  Id u0 = t.m.bitcast(t.tU, s0);
  switch (op) {
    case 0x01: setU(u0); break;                                          // v_mov_b32
    case 0x05: setF(t.m.emit(spv::Op::OpConvertSToF, t.tF, {t.m.bitcast(t.tI, u0)})); break;  // cvt_f32_i32
    case 0x06: setF(t.m.emit(spv::Op::OpConvertUToF, t.tF, {u0})); break;  // cvt_f32_u32
    case 0x07: setU(t.m.emit(spv::Op::OpConvertFToU, t.tU, {s0})); break;  // cvt_u32_f32
    case 0x08: setU(t.m.bitcast(t.tU, t.m.emit(spv::Op::OpConvertFToS, t.tI, {s0}))); break;  // cvt_i32_f32
    case 0x0c: setU(t.m.bitcast(t.tU, t.m.emit(spv::Op::OpConvertFToS, t.tI,
                  {t.ext1(GLSLstd450Floor, t.fadd(s0, t.fconst(0.5f)))}))); break;  // cvt_rpi_i32_f32
    case 0x0d: setU(t.m.bitcast(t.tU, t.m.emit(spv::Op::OpConvertFToS, t.tI,
                  {t.ext1(GLSLstd450Floor, s0)}))); break;               // cvt_flr_i32_f32
    case 0x11: case 0x12: case 0x13: case 0x14: {                        // cvt_f32_ubyte0..3
      Id b = t.m.emit(spv::Op::OpBitwiseAnd, t.tU,
                      {t.m.emit(spv::Op::OpShiftRightLogical, t.tU,
                                {u0, t.m.constU32((op - 0x11) * 8)}), t.m.constU32(0xFF)});
      setF(t.m.emit(spv::Op::OpConvertUToF, t.tF, {b}));
      break;
    }
    case 0x20: setF(t.ext1(GLSLstd450Fract, s0)); break;                 // v_fract_f32
    case 0x21: setF(t.ext1(GLSLstd450Trunc, s0)); break;                 // v_trunc_f32
    case 0x22: setF(t.ext1(GLSLstd450Ceil, s0)); break;                  // v_ceil_f32
    case 0x23: setF(t.ext1(GLSLstd450RoundEven, s0)); break;             // v_rndne_f32
    case 0x24: setF(t.ext1(GLSLstd450Floor, s0)); break;                 // v_floor_f32
    case 0x25: setF(t.ext1(GLSLstd450Exp2, s0)); break;                  // v_exp_f32
    case 0x26: case 0x27: setF(t.ext1(GLSLstd450Log2, s0)); break;       // v_log[_clamp]_f32
    case 0x28: case 0x29: case 0x2a: case 0x2b:
      setF(t.fdiv(t.fconst(1.0f), s0)); break;                           // v_rcp[_clamp/legacy/iflag]_f32
    case 0x2c: case 0x2d: case 0x2e:
      setF(t.ext1(GLSLstd450InverseSqrt, s0)); break;                    // v_rsq[_clamp/legacy]_f32
    case 0x33: setF(t.ext1(GLSLstd450Sqrt, s0)); break;                  // v_sqrt_f32
    // GCN trig takes the argument in revolutions (1.0 == 2*pi), so scale to radians.
    case 0x35: setF(t.ext1(GLSLstd450Sin, t.fmul(s0, t.fconst(6.28318530718f)))); break;  // v_sin_f32
    case 0x36: setF(t.ext1(GLSLstd450Cos, t.fmul(s0, t.fconst(6.28318530718f)))); break;  // v_cos_f32
    case 0x37: setU(t.m.emit(spv::Op::OpNot, t.tU, {u0})); break;        // v_not_b32
    default: warnUnsup("vop1", op); setU(u0); break;                     // mov fallback
  }
}

// GFX6/7 VOP2 numbering (AMD CI ISA). The min/max and bitwise ops were previously
// at the wrong opcodes (0x0a/0x0b and 0x25-0x27, which are actually integer mul-hi
// and integer add/sub); the numbers below are the real ones.
void emitVop2(Tr &t, uint32_t op, uint32_t vdst, Id s0, Id s1, uint32_t literal) {
  auto setF = [&](Id f) { t.stVgF(vdst, f); };
  auto setU = [&](Id u) { t.stVg(vdst, u); };
  Id u0 = t.m.bitcast(t.tU, s0), u1 = t.m.bitcast(t.tU, s1);
  Id i0 = t.m.bitcast(t.tI, s0), i1 = t.m.bitcast(t.tI, s1);
  auto sh = [&](Id x) { return t.m.emit(spv::Op::OpBitwiseAnd, t.tU, {x, t.m.constU32(31)}); };
  switch (op) {
    case 0x00: {  // v_cndmask_b32: VCC ? s1 : s0 (VCC stored as raw 1u/0u by VOPC)
      Id cond = t.m.emit(spv::Op::OpINotEqual, t.tBool, {t.ldSg(106), t.m.constU32(0)});
      setF(t.m.emit(spv::Op::OpSelect, t.tF, {cond, s1, s0}));
      break;
    }
    case 0x03: setF(t.fadd(s0, s1)); break;                             // v_add_f32
    case 0x04: setF(t.fsub(s0, s1)); break;                             // v_sub_f32
    case 0x05: setF(t.fsub(s1, s0)); break;                             // v_subrev_f32
    case 0x06: setF(t.fadd(t.fmul(s0, s1), t.ldVgF(vdst))); break;      // v_mac_legacy_f32
    case 0x07: case 0x08: setF(t.fmul(s0, s1)); break;                  // v_mul[_legacy]_f32
    case 0x09: case 0x0b: setU(t.m.emit(spv::Op::OpIMul, t.tU, {u0, u1})); break;  // v_mul_i32/u32_i24/u24 (low 32)
    case 0x0d: case 0x0f: setF(t.ext2(GLSLstd450FMin, s0, s1)); break;  // v_min[_legacy]_f32
    case 0x0e: case 0x10: setF(t.ext2(GLSLstd450FMax, s0, s1)); break;  // v_max[_legacy]_f32
    case 0x11: setU(t.m.bitcast(t.tU, t.m.extInst(t.tI, GLSLstd450SMin, {i0, i1}))); break;  // v_min_i32
    case 0x12: setU(t.m.bitcast(t.tU, t.m.extInst(t.tI, GLSLstd450SMax, {i0, i1}))); break;  // v_max_i32
    case 0x13: setU(t.m.extInst(t.tU, GLSLstd450UMin, {u0, u1})); break;  // v_min_u32
    case 0x14: setU(t.m.extInst(t.tU, GLSLstd450UMax, {u0, u1})); break;  // v_max_u32
    case 0x15: setU(t.m.emit(spv::Op::OpShiftRightLogical, t.tU, {u0, sh(u1)})); break;   // v_lshr_b32
    case 0x16: setU(t.m.emit(spv::Op::OpShiftRightLogical, t.tU, {u1, sh(u0)})); break;   // v_lshrrev_b32
    case 0x17: setU(t.m.bitcast(t.tU, t.m.emit(spv::Op::OpShiftRightArithmetic, t.tI, {i0, sh(u1)}))); break;  // v_ashr_i32
    case 0x18: setU(t.m.bitcast(t.tU, t.m.emit(spv::Op::OpShiftRightArithmetic, t.tI, {i1, sh(u0)}))); break;  // v_ashrrev_i32
    case 0x19: setU(t.m.emit(spv::Op::OpShiftLeftLogical, t.tU, {u0, sh(u1)})); break;    // v_lshl_b32
    case 0x1a: setU(t.m.emit(spv::Op::OpShiftLeftLogical, t.tU, {u1, sh(u0)})); break;    // v_lshlrev_b32
    case 0x1b: setU(t.m.emit(spv::Op::OpBitwiseAnd, t.tU, {u0, u1})); break;  // v_and_b32
    case 0x1c: setU(t.m.emit(spv::Op::OpBitwiseOr, t.tU, {u0, u1})); break;   // v_or_b32
    case 0x1d: setU(t.m.emit(spv::Op::OpBitwiseXor, t.tU, {u0, u1})); break;  // v_xor_b32
    case 0x1e: {  // v_bfm_b32: mask of s0[4:0] bits at offset s1[4:0]
      Id ones = t.m.emit(spv::Op::OpISub, t.tU,
                  {t.m.emit(spv::Op::OpShiftLeftLogical, t.tU, {t.m.constU32(1), sh(u0)}), t.m.constU32(1)});
      setU(t.m.emit(spv::Op::OpShiftLeftLogical, t.tU, {ones, sh(u1)}));
      break;
    }
    case 0x1f: setF(t.fadd(t.fmul(s0, s1), t.ldVgF(vdst))); break;      // v_mac_f32
    case 0x20: setF(t.fadd(t.fmul(s0, t.m.bitcast(t.tF, t.m.constU32(literal))), s1)); break;  // v_madmk_f32: s0*K+s1
    case 0x21: setF(t.fadd(t.fmul(s0, s1), t.m.bitcast(t.tF, t.m.constU32(literal)))); break;  // v_madak_f32: s0*s1+K
    case 0x25: case 0x28: setU(t.m.emit(spv::Op::OpIAdd, t.tU, {u0, u1})); break;  // v_add_i32 / v_addc_u32
    case 0x26: setU(t.m.emit(spv::Op::OpISub, t.tU, {u0, u1})); break;  // v_sub_i32
    case 0x27: setU(t.m.emit(spv::Op::OpISub, t.tU, {u1, u0})); break;  // v_subrev_i32
    case 0x2f: setU(t.m.extInst(t.tU, GLSLstd450PackHalf2x16,
                  {t.m.compositeConstruct(t.tV2, {s0, s1})})); break;    // v_cvt_pkrtz_f16_f32
    default: warnUnsup("vop2", op); setF(t.fmul(s0, s1)); break;
  }
}

// VOPC: vector compare -> a mask register as raw 1u/0u. The low nibble selects the
// predicate (1=lt 2=eq 3=le 4=gt 5=ne 6=ge) for all of f32 (op 0x00-0x1f, incl. the
// cmpx EXEC-writing variants which we treat the same), i32 (0x80-0x9f) and u32
// (0xc0-0xdf). s0f/s1f are the float operands, s0u/s1u the raw uint operands. `dst`
// is the destination SGPR: 106 (VCC) for the VOPC encoding, or the VOP3 vdst when a
// compare is emitted in VOP3 form (writing an explicit SGPR pair).
void emitVopc(Tr &t, uint32_t op, Id s0f, Id s1f, Id s0u, Id s1u, uint32_t dst = 106) {
  uint32_t lo = op & 0xF;
  Id cond = 0;
  Id si0 = t.m.bitcast(t.tI, s0u), si1 = t.m.bitcast(t.tI, s1u);
  auto F = [&](spv::Op o) { return t.m.emit(o, t.tBool, {s0f, s1f}); };
  auto I = [&](spv::Op o) { return t.m.emit(o, t.tBool, {si0, si1}); };
  auto U = [&](spv::Op o) { return t.m.emit(o, t.tBool, {s0u, s1u}); };
  if (op <= 0x3F) {  // f32 / f64-as-f32 / cmpx
    switch (lo) {
      case 1: cond = F(spv::Op::OpFOrdLessThan); break;
      case 2: cond = F(spv::Op::OpFOrdEqual); break;
      case 3: cond = F(spv::Op::OpFOrdLessThanEqual); break;
      case 4: cond = F(spv::Op::OpFOrdGreaterThan); break;
      case 5: cond = F(spv::Op::OpFUnordNotEqual); break;
      case 6: cond = F(spv::Op::OpFOrdGreaterThanEqual); break;
    }
  } else if (op >= 0x80 && op <= 0xBF) {  // i32
    switch (lo) {
      case 1: cond = I(spv::Op::OpSLessThan); break;
      case 2: cond = I(spv::Op::OpIEqual); break;
      case 3: cond = I(spv::Op::OpSLessThanEqual); break;
      case 4: cond = I(spv::Op::OpSGreaterThan); break;
      case 5: cond = I(spv::Op::OpINotEqual); break;
      case 6: cond = I(spv::Op::OpSGreaterThanEqual); break;
    }
  } else if (op >= 0xC0) {  // u32
    switch (lo) {
      case 1: cond = U(spv::Op::OpULessThan); break;
      case 2: cond = U(spv::Op::OpIEqual); break;
      case 3: cond = U(spv::Op::OpULessThanEqual); break;
      case 4: cond = U(spv::Op::OpUGreaterThan); break;
      case 5: cond = U(spv::Op::OpINotEqual); break;
      case 6: cond = U(spv::Op::OpUGreaterThanEqual); break;
    }
  }
  if (cond)
    t.stSg(dst, t.m.emit(spv::Op::OpSelect, t.tU, {cond, t.m.constU32(1), t.m.constU32(0)}));
}

void emitVop3(Tr &t, uint32_t op, uint32_t vdst, Id s0, Id s1, Id s2) {
  // VOP3 reflects the VOPC (0x000-0x0FF), VOP2 (0x100-0x13F) and VOP1 (0x180-0x1FF)
  // encodings; only 0x140-0x17F are VOP3-exclusive.
  Id u0 = t.m.bitcast(t.tU, s0), u1 = t.m.bitcast(t.tU, s1), u2 = t.m.bitcast(t.tU, s2);
  if (op < 0x100) {  // VOPC compare in VOP3 form: writes the mask to sgpr[vdst]
    emitVopc(t, op, s0, s1, u0, u1, vdst); return;
  }
  if (op >= 0x100 && op < 0x140) { emitVop2(t, op - 0x100, vdst, s0, s1); return; }
  if (op >= 0x180 && op < 0x200) { emitVop1(t, op - 0x180, vdst, s0); return; }
  auto setF = [&](Id f) { t.stVgF(vdst, f); };
  auto setU = [&](Id u) { t.stVg(vdst, u); };
  auto mulHi = [&](spv::Op mulOp) {  // high 32 bits of a 64-bit product
    Id st = t.m.typeStruct({t.tU, t.tU});
    return t.m.compositeExtract(t.tU, t.m.emit(mulOp, st, {u0, u1}), 1);
  };
  switch (op) {
    case 0x140: case 0x141: case 0x14b:                                 // v_mad[_legacy]_f32 / v_fma_f32
      setF(t.fadd(t.fmul(s0, s1), s2)); break;
    case 0x142: case 0x143:                                             // v_mad_i32/u32_i24/u24 (low 32)
      setU(t.m.emit(spv::Op::OpIAdd, t.tU, {t.m.emit(spv::Op::OpIMul, t.tU, {u0, u1}), u2})); break;
    case 0x148:                                                         // v_bfe_u32
      setU(t.m.emit(spv::Op::OpBitFieldUExtract, t.tU,
                    {u0, t.m.emit(spv::Op::OpBitwiseAnd, t.tU, {u1, t.m.constU32(31)}),
                     t.m.emit(spv::Op::OpBitwiseAnd, t.tU, {u2, t.m.constU32(31)})})); break;
    case 0x149:                                                         // v_bfe_i32
      setU(t.m.bitcast(t.tU, t.m.emit(spv::Op::OpBitFieldSExtract, t.tI,
                    {t.m.bitcast(t.tI, u0), t.m.emit(spv::Op::OpBitwiseAnd, t.tU, {u1, t.m.constU32(31)}),
                     t.m.emit(spv::Op::OpBitwiseAnd, t.tU, {u2, t.m.constU32(31)})}))); break;
    case 0x14a:                                                         // v_bfi_b32: (s0&s1)|(~s0&s2)
      setU(t.m.emit(spv::Op::OpBitwiseOr, t.tU,
                    {t.m.emit(spv::Op::OpBitwiseAnd, t.tU, {u0, u1}),
                     t.m.emit(spv::Op::OpBitwiseAnd, t.tU, {t.m.emit(spv::Op::OpNot, t.tU, {u0}), u2})})); break;
    case 0x169: case 0x16b: setU(t.m.emit(spv::Op::OpIMul, t.tU, {u0, u1})); break;  // v_mul_lo_u32/i32
    case 0x16a: setU(mulHi(spv::Op::OpUMulExtended)); break;             // v_mul_hi_u32
    case 0x16c: setU(mulHi(spv::Op::OpSMulExtended)); break;             // v_mul_hi_i32
    case 0x151: setF(t.ext2(GLSLstd450FMin, t.ext2(GLSLstd450FMin, s0, s1), s2)); break;  // v_min3_f32
    case 0x154: setF(t.ext2(GLSLstd450FMax, t.ext2(GLSLstd450FMax, s0, s1), s2)); break;  // v_max3_f32
    case 0x157: {  // v_med3_f32 = clamp(s2, min(s0,s1), max(s0,s1))
      Id lo = t.ext2(GLSLstd450FMin, s0, s1), hi = t.ext2(GLSLstd450FMax, s0, s1);
      setF(t.m.extInst(t.tF, GLSLstd450FClamp, {s2, lo, hi}));
      break;
    }
    default: warnUnsup("vop3", op); setF(s0); break;
  }
}

// ---- control flow: shared per-instruction emit + CFG while-switch -----------
// Per-stage declarations carried into the shared emitter. The straight-line
// (single-basic-block) path stays inline in translateVs/Ps (proven, unchanged);
// this shared path is used only when a shader has branches (or DELTA_GPU_SPIRV_CFG
// forces it for testing). Arbitrary control flow is lowered to a while/switch
// state machine over basic blocks (handles reducible and irreducible CFGs).
struct StageCtx {
  bool isPs = false;
  Recompiled *r = nullptr;
  std::vector<Id> *iface = nullptr;
  Id posOut = 0;                               // VS
  std::unordered_map<uint32_t, Id> paramOuts;  // VS
  uint32_t maxParam = 0;                        // VS
  bool haveCbuf = false;                        // VS
  Id colorOut = 0;                             // PS
  Id sampImgTy = 0, pSampImg = 0, imgTy = 0;   // PS
  std::unordered_map<uint32_t, Id> inVars;     // PS
  uint32_t maxIn = 0;                           // PS
  bool wroteColor = false;                      // PS (compile-time: shader has an exp)
  Id colorWrittenVar = 0;  // PS (runtime per-lane: this fragment reached a color exp)
};

Id psInputVar(Tr &t, StageCtx &sc, uint32_t attr) {
  auto it = sc.inVars.find(attr);
  if (it != sc.inVars.end()) return it->second;
  Id v = t.m.variable(t.m.typePointer(spv::StorageClass::Input, t.tV4), spv::StorageClass::Input);
  t.m.decorate(v, spv::Decoration::Location, {attr});
  sc.iface->push_back(v);
  sc.inVars[attr] = v;
  if (attr + 1 > sc.maxIn) sc.maxIn = attr + 1;
  return v;
}
Id vsParamOut(Tr &t, StageCtx &sc, uint32_t p) {
  auto it = sc.paramOuts.find(p);
  if (it != sc.paramOuts.end()) return it->second;
  Id v = t.m.variable(t.m.typePointer(spv::StorageClass::Output, t.tV4), spv::StorageClass::Output);
  t.m.decorate(v, spv::Decoration::Location, {p});
  sc.iface->push_back(v);
  sc.paramOuts[p] = v;
  return v;
}

// Emit a MIMG image op. Declares the texture as a combined sampler at set 0 /
// binding = the MIMG order, records it in psTexs, and samples/fetches it. Coords
// are treated as 2D (x,y in the first two address VGPRs). Handles the sample
// variants (implicit LOD, explicit LOD, LOD-zero) and the integer image_load; the
// sampler resource (S#) uses the renderer's default sampler.
void emitMimg(Tr &t, uint32_t op, uint32_t w0, uint32_t w1, Id imgTy, Id sampImgTy,
              Id pSampImg, Recompiled &r) {
  uint32_t dmask = (w0 >> 8) & 0xF, vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4, bind = (uint32_t)r.psTexs.size();
  r.psTexs.push_back({bind, srsrc});
  Id texVar = t.m.variable(pSampImg, spv::StorageClass::UniformConstant);
  t.m.decorate(texVar, spv::Decoration::DescriptorSet, {0});
  t.m.decorate(texVar, spv::Decoration::Binding, {bind});
  Id si = t.m.load(sampImgTy, texVar);
  Id uv = t.m.compositeConstruct(t.tV2, {t.ldVgF(vaddr), t.ldVgF(vaddr + 1)});
  uint32_t lodOp = (uint32_t)spv::ImageOperandsMask::Lod;
  bool known = op == 0x00 || op == 0x01 || op == 0x20 || op == 0x21 ||
               op == 0x24 || op == 0x25 || op == 0x27;
  if (!known) warnUnsup("mimg", op, w0, w1);
  Id texel;
  if (op == 0x00 || op == 0x01) {  // image_load[_mip]: integer fetch, no filtering
    Id ic = t.m.compositeConstruct(t.m.typeVec(t.tI, 2),
              {t.m.bitcast(t.tI, t.ldVg(vaddr)), t.m.bitcast(t.tI, t.ldVg(vaddr + 1))});
    Id img = t.m.emit(spv::Op::OpImage, imgTy, {si});
    texel = t.m.emit(spv::Op::OpImageFetch, t.tV4, {img, ic, lodOp, t.m.constU32(0)});
  } else if (op == 0x24) {  // image_sample_l: explicit LOD in the coord+2 VGPR
    texel = t.m.emit(spv::Op::OpImageSampleExplicitLod, t.tV4,
                     {si, uv, lodOp, t.ldVgF(vaddr + 2)});
  } else if (op == 0x27 || op == 0x2f) {  // image_sample_lz / _c_lz: forced LOD 0
    texel = t.m.emit(spv::Op::OpImageSampleExplicitLod, t.tV4,
                     {si, uv, lodOp, t.fconst(0.0f)});
  } else {  // image_sample / _cl / _b (bias/derivs/compare ignored): implicit LOD
    texel = t.m.emit(spv::Op::OpImageSampleImplicitLod, t.tV4, {si, uv});
  }
  uint32_t comp = 0;
  for (int i = 0; i < 4; i++)
    if (dmask & (1 << i)) t.stVgF(vdata + comp++, t.m.compositeExtract(t.tF, texel, i));
}

// Emit one non-terminator instruction (branches are handled by the CFG driver).
void emitInst(Tr &t, const Inst &in, StageCtx &sc) {
  uint32_t w = in.raw[0], w1 = in.raw[1];
  switch (in.enc) {
    case Enc::sop1: {
      uint32_t op = in.opcode, sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF;
      if (op == 0x03) t.stSg(sdst, t.srcRaw(ssrc0, in.literal));  // s_mov_b32
      else if (op == 0x04) {                                      // s_mov_b64
        t.stSg(sdst, t.srcRaw(ssrc0, in.literal));
        if (ssrc0 <= 103) t.stSg(sdst + 1, t.ldSg(ssrc0 + 1));
      } else if (op >= 0x24 && op <= 0x27) {  // s_{and,or,xor,andn2}_saveexec_b64
        Id oldExec = t.ldExec(), src = t.srcRaw(ssrc0, in.literal), ne;
        if (op == 0x24) ne = t.m.emit(spv::Op::OpBitwiseAnd, t.tU, {oldExec, src});
        else if (op == 0x25) ne = t.m.emit(spv::Op::OpBitwiseOr, t.tU, {oldExec, src});
        else if (op == 0x26) ne = t.m.emit(spv::Op::OpBitwiseXor, t.tU, {oldExec, src});
        else ne = t.m.emit(spv::Op::OpBitwiseAnd, t.tU,
                           {oldExec, t.m.emit(spv::Op::OpNot, t.tU, {src})});
        t.stSg(sdst, oldExec);
        t.stSg(126, ne);
        t.stSccBool(t.isNonZero(ne));
      }
      break;
    }
    case Enc::sop2: {
      uint32_t op = in.opcode, sdst = (w >> 16) & 0x7F, s0f = w & 0xFF, s1f = (w >> 8) & 0xFF;
      Id a = t.srcRaw(s0f, in.literal), b = t.srcRaw(s1f, in.literal), r = 0;
      bool scc = false;
      auto shamt = [&] { return t.m.emit(spv::Op::OpBitwiseAnd, t.tU, {b, t.m.constU32(31)}); };
      switch (op) {
        case 0x00: case 0x02: case 0x04: r = t.m.emit(spv::Op::OpIAdd, t.tU, {a, b}); break;
        case 0x01: case 0x03: case 0x05: r = t.m.emit(spv::Op::OpISub, t.tU, {a, b}); break;
        case 0x0a: case 0x0b: r = t.m.emit(spv::Op::OpSelect, t.tU, {t.isNonZero(t.ldScc()), a, b}); break;
        case 0x0e: case 0x0f: r = t.m.emit(spv::Op::OpBitwiseAnd, t.tU, {a, b}); scc = true; break;
        case 0x10: case 0x11: r = t.m.emit(spv::Op::OpBitwiseOr, t.tU, {a, b}); scc = true; break;
        case 0x12: case 0x13: r = t.m.emit(spv::Op::OpBitwiseXor, t.tU, {a, b}); scc = true; break;
        case 0x14: case 0x15: r = t.m.emit(spv::Op::OpBitwiseAnd, t.tU,
                              {a, t.m.emit(spv::Op::OpNot, t.tU, {b})}); scc = true; break;
        case 0x1e: case 0x1f: r = t.m.emit(spv::Op::OpShiftLeftLogical, t.tU, {a, shamt()}); scc = true; break;
        case 0x20: case 0x21: r = t.m.emit(spv::Op::OpShiftRightLogical, t.tU, {a, shamt()}); scc = true; break;
        case 0x22: case 0x23: r = t.m.emit(spv::Op::OpShiftRightArithmetic, t.tU,
                              {t.m.bitcast(t.tI, a), shamt()}); scc = true; break;
        case 0x24: r = t.m.emit(spv::Op::OpIMul, t.tU, {a, b}); break;  // s_mul_i32
        default: r = a; break;
      }
      if (r) { t.stSg(sdst, r); if (scc) t.stSccBool(t.isNonZero(r)); }
      break;
    }
    case Enc::sopc: {  // s_cmp_* -> SCC
      uint32_t op = in.opcode, s0f = w & 0xFF, s1f = (w >> 8) & 0xFF;
      Id a = t.srcRaw(s0f, in.literal), b = t.srcRaw(s1f, in.literal);
      Id ai = t.m.bitcast(t.tI, a), bi = t.m.bitcast(t.tI, b), c = 0;
      switch (op) {
        case 0x00: c = t.m.emit(spv::Op::OpIEqual, t.tBool, {a, b}); break;
        case 0x01: c = t.m.emit(spv::Op::OpINotEqual, t.tBool, {a, b}); break;
        case 0x02: c = t.m.emit(spv::Op::OpSGreaterThan, t.tBool, {ai, bi}); break;
        case 0x03: c = t.m.emit(spv::Op::OpSGreaterThanEqual, t.tBool, {ai, bi}); break;
        case 0x04: c = t.m.emit(spv::Op::OpSLessThan, t.tBool, {ai, bi}); break;
        case 0x05: c = t.m.emit(spv::Op::OpSLessThanEqual, t.tBool, {ai, bi}); break;
        case 0x06: c = t.m.emit(spv::Op::OpIEqual, t.tBool, {a, b}); break;
        case 0x07: c = t.m.emit(spv::Op::OpINotEqual, t.tBool, {a, b}); break;
        case 0x08: c = t.m.emit(spv::Op::OpUGreaterThan, t.tBool, {a, b}); break;
        case 0x09: c = t.m.emit(spv::Op::OpUGreaterThanEqual, t.tBool, {a, b}); break;
        case 0x0a: c = t.m.emit(spv::Op::OpULessThan, t.tBool, {a, b}); break;
        case 0x0b: c = t.m.emit(spv::Op::OpULessThanEqual, t.tBool, {a, b}); break;
        default: break;
      }
      if (c) t.stSccBool(c);
      break;
    }
    case Enc::smrd: {
      if (sc.isPs) break;  // VS cbuffer only (PS has no push range in our layout)
      uint32_t op = in.opcode, sdst = (w >> 15) & 0x7F, sbase = (w >> 9) & 0x3F;
      bool imm = (w >> 8) & 1; uint32_t off = w & 0xFF;
      if (op >= 0x08) {
        uint32_t n = op == 0x08 ? 1 : op == 0x09 ? 2 : op == 0x0a ? 4 : op == 0x0b ? 8 : 16;
        if (!sc.haveCbuf) { sc.haveCbuf = true;
          sc.r->vsCbufs.push_back({(uint32_t)sc.r->vsCbufs.size(), sbase * 2u, 16}); }
        uint32_t doff = imm ? off : 0;
        for (uint32_t i = 0; i < n; i++) t.stSg(sdst + i, t.pcDword(doff + i));
      }
      break;
    }
    case Enc::vop2: {
      uint32_t op = in.opcode, vdst = (w >> 17) & 0xFF, vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
      emitVop2(t, op, vdst, t.srcF(src0, in.literal), t.srcF(256 + vsrc1, in.literal), in.literal);
      break;
    }
    case Enc::vop1: {
      uint32_t op = in.opcode, vdst = (w >> 17) & 0xFF, src0 = w & 0x1FF;
      emitVop1(t, op, vdst, t.srcF(src0, in.literal));
      break;
    }
    case Enc::vop3: {
      uint32_t op = in.opcode, vdst = w & 0xFF, abs = (w >> 8) & 7;
      uint32_t s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF, s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
      emitVop3(t, op, vdst, t.srcF(s0, in.literal, neg & 1, abs & 1),
               t.srcF(s1, in.literal, neg & 2, abs & 2), t.srcF(s2, in.literal, neg & 4, abs & 4));
      break;
    }
    case Enc::vopc: {
      uint32_t op = in.opcode, vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
      emitVopc(t, op, t.srcF(src0, in.literal), t.srcF(256 + vsrc1, in.literal),
               t.srcRaw(src0, in.literal), t.srcRaw(256 + vsrc1, in.literal));
      break;
    }
    case Enc::vintrp: {
      if (!sc.isPs) break;
      uint32_t chan = (w >> 8) & 3, attr = (w >> 10) & 0x3F, op = (w >> 16) & 3, vdst = (w >> 18) & 0xFF;
      if (op == 1) {
        Id v = psInputVar(t, sc, attr);
        Id pInF = t.m.typePointer(spv::StorageClass::Input, t.tF);
        t.stVgF(vdst, t.m.load(t.tF, t.m.accessChain(pInF, v, {t.m.constU32(chan)})));
      }
      break;
    }
    case Enc::mimg: {
      if (!sc.isPs) break;
      emitMimg(t, in.opcode, w, w1, sc.imgTy, sc.sampImgTy, sc.pSampImg, *sc.r);
      break;
    }
    case Enc::exp: {
      uint32_t en = w & 0xF, target = (w >> 4) & 0x3F, compr = (w >> 10) & 1;
      uint32_t v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF, (w1 >> 24) & 0xFF};
      if (sc.isPs) {
        if (target <= 7) {
          sc.wroteColor = true;
          Id col;
          if (compr) {
            Id c01 = t.m.extInst(t.tV2, GLSLstd450UnpackHalf2x16, {t.ldVg(v[0])});
            Id c23 = t.m.extInst(t.tV2, GLSLstd450UnpackHalf2x16, {t.ldVg(v[1])});
            col = t.m.vectorShuffle(t.tV4, c01, c23, {0, 1, 2, 3});
          } else {
            Id c[4];
            for (int i = 0; i < 4; i++) c[i] = (en & (1 << i)) ? t.ldVgF(v[i]) : t.fconst(i == 3 ? 1.f : 0.f);
            col = t.m.compositeConstruct(t.tV4, {c[0], c[1], c[2], c[3]});
          }
          t.m.store(sc.colorOut, col);
          // Mark this fragment as having reached a color export, so the discard idiom
          // (control flow that branches over the exp) can be lowered to OpKill.
          if (sc.colorWrittenVar) t.m.store(sc.colorWrittenVar, t.m.constU32(1));
        }
      } else {
        if (target == 12) {  // POS0
          Id c[4];
          for (int i = 0; i < 4; i++) c[i] = (en & (1 << i)) ? t.ldVgF(v[i]) : t.fconst(i == 3 ? 1.f : 0.f);
          t.m.store(sc.posOut, t.m.compositeConstruct(t.tV4, {c[0], c[1], c[2], c[3]}));
        } else if (target >= 32 && target <= 63) {
          uint32_t p = target - 32; if (p + 1 > sc.maxParam) sc.maxParam = p + 1;
          Id outVar = vsParamOut(t, sc, p);
          Id c[4];
          for (int i = 0; i < 4; i++) c[i] = (en & (1 << i)) ? t.ldVgF(v[i]) : t.fconst(0.f);
          t.m.store(outVar, t.m.compositeConstruct(t.tV4, {c[0], c[1], c[2], c[3]}));
        }
      }
      break;
    }
    default: break;
  }
}

// Branch classification. 0=none, 1=uncond, 2=scc0, 3=scc1, 4=vccz, 5=vccnz,
// 6=execz, 7=execnz, 8=endpgm.
int branchKind(const Inst &in) {
  if (in.enc != Enc::sopp) return 0;
  switch (in.opcode) {
    case 0x01: return 8; case 0x02: return 1; case 0x04: return 2; case 0x05: return 3;
    case 0x06: return 4; case 0x07: return 5; case 0x08: return 6; case 0x09: return 7;
    default: return 0;
  }
}
bool hasControlFlow(const std::vector<Inst> &insts) {
  for (auto &in : insts) { int k = branchKind(in); if (k >= 1 && k <= 7) return true; }
  return false;
}
// "Take the branch" condition for a conditional branch kind.
Id branchTaken(Tr &t, int kind) {
  switch (kind) {
    case 2: return t.isZero(t.ldScc());
    case 3: return t.isNonZero(t.ldScc());
    case 4: return t.isZero(t.ldSg(106));
    case 5: return t.isNonZero(t.ldSg(106));
    case 6: return t.isZero(t.ldExec());
    case 7: return t.isNonZero(t.ldExec());
    default: return t.m.constBool(false);
  }
}

void emitCFG(Tr &t, std::vector<Inst> &insts, StageCtx &sc) {
  uint32_t maxPc = insts.empty() ? 0 : insts.back().pc + insts.back().size;
  // Basic-block leaders: entry, every branch target, the instruction after a branch.
  std::vector<uint32_t> leaders{0};
  for (auto &in : insts) {
    int k = branchKind(in);
    if (k == 0) continue;
    leaders.push_back(in.pc + in.size);  // fall-through
    if (k >= 1 && k <= 7) {              // has a PC-relative target
      int32_t simm = (int16_t)(in.raw[0] & 0xFFFF);
      leaders.push_back((uint32_t)((int32_t)in.pc + (int32_t)in.size + simm));
    }
  }
  std::sort(leaders.begin(), leaders.end());
  leaders.erase(std::unique(leaders.begin(), leaders.end()), leaders.end());
  // Drop out-of-range leaders (targets past the decoded program -> EXIT).
  std::vector<uint32_t> starts;
  for (uint32_t l : leaders) if (l < maxPc) starts.push_back(l);
  uint32_t nB = (uint32_t)starts.size(), EXIT = nB;
  auto blockOf = [&](uint32_t pc) -> uint32_t {
    if (pc >= maxPc) return EXIT;
    uint32_t b = 0;
    for (uint32_t i = 0; i < nB; i++) if (starts[i] <= pc) b = i; else break;
    return b;
  };

  Id header = t.m.newBlock(), dispatch = t.m.newBlock(), mergeSel = t.m.newBlock();
  Id cont = t.m.newBlock(), merge = t.m.newBlock(), exitBlk = t.m.newBlock();
  std::vector<Id> caseLbl(nB);
  for (auto &l : caseLbl) l = t.m.newBlock();

  t.stState(0);
  t.m.branch(header);
  t.m.openBlock(header);
  t.m.loopMerge(merge, cont);
  t.m.branch(dispatch);
  t.m.openBlock(dispatch);
  Id s = t.ldState();
  t.m.selectionMerge(mergeSel);
  std::vector<std::pair<uint32_t, Id>> cases;
  for (uint32_t i = 0; i < nB; i++) cases.push_back({i, caseLbl[i]});
  t.m.switchInst(s, exitBlk, cases);  // default (incl. EXIT state) -> exit the loop

  for (uint32_t bi = 0; bi < nB; bi++) {
    t.m.openBlock(caseLbl[bi]);
    uint32_t blkStart = starts[bi], blkEnd = (bi + 1 < nB) ? starts[bi + 1] : maxPc;
    bool terminated = false;
    for (auto &in : insts) {
      if (in.pc < blkStart || in.pc >= blkEnd) continue;
      int k = branchKind(in);
      if (k == 0) { emitInst(t, in, sc); continue; }
      // terminator
      uint32_t fall = (bi + 1 < nB) ? bi + 1 : EXIT;
      if (k == 8) {  // endpgm
        t.stState(EXIT);
      } else if (k == 1) {  // unconditional
        int32_t simm = (int16_t)(in.raw[0] & 0xFFFF);
        t.stState(blockOf((uint32_t)((int32_t)in.pc + (int32_t)in.size + simm)));
      } else {  // conditional
        int32_t simm = (int16_t)(in.raw[0] & 0xFFFF);
        uint32_t tb = blockOf((uint32_t)((int32_t)in.pc + (int32_t)in.size + simm));
        Id sel = t.m.emit(spv::Op::OpSelect, t.tU,
                          {branchTaken(t, k), t.m.constU32(tb), t.m.constU32(fall)});
        t.stStateId(sel);
      }
      terminated = true;
      break;
    }
    if (!terminated) t.stState((bi + 1 < nB) ? bi + 1 : EXIT);  // fall through
    t.m.branch(mergeSel);
  }
  t.m.openBlock(exitBlk);
  t.m.branch(merge);
  t.m.openBlock(mergeSel);
  t.m.branch(cont);
  t.m.openBlock(cont);
  t.m.branch(header);
  t.m.openBlock(merge);  // left open; caller emits the stage epilogue + return here
}

// ---- VS ---------------------------------------------------------------------
bool translateVs(const uint32_t *vsCode, const uint32_t *vsUserData, Recompiled &r,
                 Tr &t) {
  uint64_t fetch = (static_cast<uint64_t>(vsUserData[1] & 0xFFFF) << 32) | vsUserData[0];
  auto attrs = parseFetch(fetch);
  if (attrs.empty()) return false;
  t.initTypes();

  // Inputs + position/param outputs.
  std::vector<Id> iface;
  Id posOut = t.m.variable(t.m.typePointer(spv::StorageClass::Output, t.tV4),
                           spv::StorageClass::Output);
  t.m.decorate(posOut, spv::Decoration::BuiltIn, {(uint32_t)spv::BuiltIn::Position});
  iface.push_back(posOut);

  Id main = t.m.beginFunction(t.tVoid, t.tFn);

  // Seed destination VGPRs from the vertex attributes.
  for (auto &a : attrs) {
    Id compTy = a.numComps == 1 ? t.tF : a.numComps == 2 ? t.tV2
                : a.numComps == 3 ? t.tV3 : t.tV4;
    Id pIn = t.m.typePointer(spv::StorageClass::Input, compTy);
    Id inVar = t.m.variable(pIn, spv::StorageClass::Input);
    t.m.decorate(inVar, spv::Decoration::Location, {a.semantic});
    iface.push_back(inVar);
    Id val = t.m.load(compTy, inVar);
    // DELTA_GPU_VSFLIPZ: negate the z of the position attribute (semantic 0, >=3
    // comps). Doom64's world verts are view-space with +z forward, but the VS's
    // GL projection (clip.w=-z) expects -z, so everything lands behind the camera
    // (w<0) and is clipped -> black level. Flipping z at the source re-projects it
    // in front. Gated (default off) so Isaac/2D titles are unaffected.
    static const bool flipZ = std::getenv("DELTA_GPU_VSFLIPZ") != nullptr;
    for (uint32_t c = 0; c < a.numComps; c++) {
      Id comp = a.numComps == 1 ? val : t.m.compositeExtract(t.tF, val, c);
      if (flipZ && a.semantic == 0 && c == 2 && a.numComps >= 3) comp = t.fneg(comp);
      t.stVgF(a.destVgpr + c, comp);
    }
    r.attrs.push_back({a.semantic, a.numComps, a.tableSgpr, a.dwordOff});
  }

  auto insts = decode(vsCode, 4096);
  uint32_t maxParam = 0;
  std::unordered_map<uint32_t, Id> paramOuts;  // param index -> Output var
  bool haveCbuf = false;

  // Branchy shaders take the CFG (while-switch) path so their control flow (the GCN
  // alpha-test/discard idiom, conditional shading) is honoured; single-basic-block
  // shaders keep the proven straight-line loop below. DELTA_GPU_SPIRV_CFG forces the
  // CFG path even for single-BB shaders (machinery test).
  static const bool forceCfg = std::getenv("DELTA_GPU_SPIRV_CFG") != nullptr;
  if (forceCfg || hasControlFlow(insts)) {
    StageCtx sc; sc.isPs = false; sc.r = &r; sc.iface = &iface; sc.posOut = posOut;
    t.seedExec();
    emitCFG(t, insts, sc);
    r.numParams = sc.maxParam;
    Id pOutF = t.m.typePointer(spv::StorageClass::Output, t.tF);
    Id zPtr = t.m.accessChain(pOutF, posOut, {t.m.constU32(2)});
    Id wPtr = t.m.accessChain(pOutF, posOut, {t.m.constU32(3)});
    Id z = t.m.load(t.tF, zPtr), wv = t.m.load(t.tF, wPtr);
    t.m.store(zPtr, t.fmul(t.fadd(z, wv), t.fconst(0.5f)));
    t.m.returnVoid();
    t.m.endFunction();
    t.m.entryPoint(spv::ExecutionModel::Vertex, main, "main", iface);
    return true;
  }

  for (auto &in : insts) {
    uint32_t w = in.raw[0], w1 = in.raw[1];
    switch (in.enc) {
      case Enc::sop1: {
        uint32_t op = in.opcode, sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF;
        if (op == 0x03) t.stSg(sdst, t.srcRaw(ssrc0, in.literal));
        else if (op == 0x04) {
          t.stSg(sdst, t.srcRaw(ssrc0, in.literal));
          if (ssrc0 <= 103) t.stSg(sdst + 1, t.ldSg(ssrc0 + 1));
        }
        break;
      }
      case Enc::smrd: {
        uint32_t op = in.opcode, sdst = (w >> 15) & 0x7F, sbase = (w >> 9) & 0x3F;
        bool imm = (w >> 8) & 1; uint32_t off = w & 0xFF;
        if (op >= 0x08) {  // s_buffer_load_dword* -> push-constant cbuffer reads
          uint32_t n = op == 0x08 ? 1 : op == 0x09 ? 2 : op == 0x0a ? 4 : op == 0x0b ? 8 : 16;
          if (!haveCbuf) { haveCbuf = true;
            r.vsCbufs.push_back({(uint32_t)r.vsCbufs.size(), sbase * 2u, 16}); }
          uint32_t doff = imm ? off : 0;
          for (uint32_t i = 0; i < n; i++) t.stSg(sdst + i, t.pcDword(doff + i));
        }
        break;
      }
      case Enc::vop2: {
        uint32_t op = in.opcode, vdst = (w >> 17) & 0xFF, vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
        emitVop2(t, op, vdst, t.srcF(src0, in.literal), t.srcF(256 + vsrc1, in.literal), in.literal);
        break;
      }
      case Enc::vop1: {
        uint32_t op = in.opcode, vdst = (w >> 17) & 0xFF, src0 = w & 0x1FF;
        emitVop1(t, op, vdst, t.srcF(src0, in.literal));
        break;
      }
      case Enc::vop3: {
        uint32_t op = in.opcode, vdst = w & 0xFF, abs = (w >> 8) & 7;
        uint32_t s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF, s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
        emitVop3(t, op, vdst, t.srcF(s0, in.literal, neg & 1, abs & 1),
                 t.srcF(s1, in.literal, neg & 2, abs & 2),
                 t.srcF(s2, in.literal, neg & 4, abs & 4));
        break;
      }
      case Enc::vopc: {
        uint32_t op = in.opcode, vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
        emitVopc(t, op, t.srcF(src0, in.literal), t.srcF(256 + vsrc1, in.literal),
                 t.srcRaw(src0, in.literal), t.srcRaw(256 + vsrc1, in.literal));
        break;
      }
      case Enc::exp: {
        uint32_t en = w & 0xF, target = (w >> 4) & 0x3F;
        uint32_t v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF, (w1 >> 24) & 0xFF};
        if (target == 12) {  // POS0 -> gl_Position
          Id comps[4];
          for (int i = 0; i < 4; i++)
            comps[i] = (en & (1 << i)) ? t.ldVgF(v[i]) : t.fconst(i == 3 ? 1.0f : 0.0f);
          Id pos = t.m.compositeConstruct(t.tV4, {comps[0], comps[1], comps[2], comps[3]});
          t.m.store(posOut, pos);
        } else if (target >= 32 && target <= 63) {  // PARAM
          uint32_t p = target - 32; if (p + 1 > maxParam) maxParam = p + 1;
          Id outVar;
          auto pit = paramOuts.find(p);
          if (pit == paramOuts.end()) {
            outVar = t.m.variable(t.m.typePointer(spv::StorageClass::Output, t.tV4),
                                  spv::StorageClass::Output);
            t.m.decorate(outVar, spv::Decoration::Location, {p});
            iface.push_back(outVar);
            paramOuts[p] = outVar;
          } else outVar = pit->second;
          Id comps[4];
          for (int i = 0; i < 4; i++)
            comps[i] = (en & (1 << i)) ? t.ldVgF(v[i]) : t.fconst(0.0f);
          t.m.store(outVar, t.m.compositeConstruct(t.tV4, {comps[0], comps[1], comps[2], comps[3]}));
        }
        break;
      }
      default: break;
    }
    if (in.enc == Enc::sopp && in.opcode == 1) break;
  }
  r.numParams = maxParam;

  // GL clip space (z in [-w,w]) -> Vulkan (z in [0,w]): z = (z + w) * 0.5.
  Id pOutF = t.m.typePointer(spv::StorageClass::Output, t.tF);
  Id zPtr = t.m.accessChain(pOutF, posOut, {t.m.constU32(2)});
  Id wPtr = t.m.accessChain(pOutF, posOut, {t.m.constU32(3)});
  Id z = t.m.load(t.tF, zPtr), wv = t.m.load(t.tF, wPtr);
  t.m.store(zPtr, t.fmul(t.fadd(z, wv), t.fconst(0.5f)));

  t.m.returnVoid();
  t.m.endFunction();
  t.m.entryPoint(spv::ExecutionModel::Vertex, main, "main", iface);
  return true;
}

// ---- PS ---------------------------------------------------------------------
bool translatePs(const uint32_t *psCode, Recompiled &r, Tr &t) {
  auto insts = decode(psCode, 4096);
  std::vector<Id> iface;
  // Color outputs are declared lazily per MRT target (location == target index), so a
  // shader exporting to MRT0..7 produces a multi-attachment fragment output. Most 2D
  // titles only export MRT0 -> a single location-0 output (identical to a single RT).
  Id colorOuts[8] = {0};
  Id pV4Out = t.m.typePointer(spv::StorageClass::Output, t.tV4);
  auto colorOutVar = [&](uint32_t target) -> Id {
    if (colorOuts[target]) return colorOuts[target];
    Id v = t.m.variable(pV4Out, spv::StorageClass::Output);
    t.m.decorate(v, spv::Decoration::Location, {target});
    iface.push_back(v);
    colorOuts[target] = v;
    r.psMrtMask |= (uint8_t)(1u << target);
    return v;
  };

  // Sampled-image type for any texture.
  Id imgTy = t.m.typeImage(t.tF, spv::Dim::Dim2D, 0, 0, 0, 1, spv::ImageFormat::Unknown);
  Id sampImgTy = t.m.typeSampledImage(imgTy);
  Id pSampImg = t.m.typePointer(spv::StorageClass::UniformConstant, sampImgTy);

  // PS inputs (interpolants) are declared lazily as they are read.
  std::unordered_map<uint32_t, Id> inVars;  // attr index -> Input vec4 var
  uint32_t maxIn = 0;
  bool wroteColor = false;

  Id main = t.m.beginFunction(t.tVoid, t.tFn);
  auto inputVar = [&](uint32_t attr) -> Id {
    auto it = inVars.find(attr);
    if (it != inVars.end()) return it->second;
    Id v = t.m.variable(t.m.typePointer(spv::StorageClass::Input, t.tV4),
                        spv::StorageClass::Input);
    t.m.decorate(v, spv::Decoration::Location, {attr});
    iface.push_back(v);
    inVars[attr] = v;
    if (attr + 1 > maxIn) maxIn = attr + 1;
    return v;
  };

  static const bool forceCfgPs = std::getenv("DELTA_GPU_SPIRV_CFG") != nullptr;
  if (forceCfgPs || hasControlFlow(insts)) {  // branchy PS: honour control flow / discard
    Id co = colorOutVar(0);
    // Default the color to transparent so a fragment that never reaches an export
    // leaves a defined value (not garbage) even if the discard lowering is bypassed.
    t.m.store(co, t.m.constComposite(t.tV4,
              {t.fconst(0.f), t.fconst(0.f), t.fconst(0.f), t.fconst(0.f)}));
    StageCtx sc; sc.isPs = true; sc.r = &r; sc.iface = &iface; sc.colorOut = co;
    sc.sampImgTy = sampImgTy; sc.pSampImg = pSampImg; sc.imgTy = imgTy;
    sc.colorWrittenVar = t.m.variable(t.pPrivU, spv::StorageClass::Private, t.m.constNull(t.tU));
    t.seedExec();
    emitCFG(t, insts, sc);
    if (!sc.wroteColor) {
      // Shader has no export at all: opaque white fallback (matches the single-BB path).
      t.m.store(co, t.m.constComposite(t.tV4,
                {t.fconst(1.f), t.fconst(1.f), t.fconst(1.f), t.fconst(1.f)}));
      t.m.returnVoid();
    } else {
      // GCN alpha-test/kill idiom: control flow branches over the color export for
      // failing fragments (e.g. s_cmp + s_cbranch_scc0 -> s_endpgm). Discard those
      // (OpKill) instead of leaving the output undefined.
      // DELTA_GPU_NOKILL: skip the discard (always keep the fragment) -- a diagnostic
      // for "everything renders black": if the whole RT is being OpKill'd (a bad
      // sampled alpha / inverted test makes every fragment fail), disabling kill
      // makes the content appear (with wrong transparency), proving the discard path.
      static const bool noKill = std::getenv("DELTA_GPU_NOKILL") != nullptr;
      if (!noKill) {
        Id wrote = t.isNonZero(t.m.load(t.tU, sc.colorWrittenVar));
        Id killBlk = t.m.newBlock(), afterKill = t.m.newBlock();
        t.m.selectionMerge(afterKill);
        t.m.branchConditional(wrote, afterKill, killBlk);
        t.m.openBlock(killBlk);
        t.m.kill();
        t.m.openBlock(afterKill);
      }
      t.m.returnVoid();
    }
    t.m.endFunction();
    t.m.entryPoint(spv::ExecutionModel::Fragment, main, "main", iface);
    t.m.execMode(main, spv::ExecutionMode::OriginUpperLeft);
    return true;
  }

  for (auto &in : insts) {
    uint32_t w = in.raw[0], w1 = in.raw[1];
    switch (in.enc) {
      case Enc::vintrp: {
        uint32_t chan = (w >> 8) & 3, attr = (w >> 10) & 0x3F, op = (w >> 16) & 3, vdst = (w >> 18) & 0xFF;
        if (op == 1) {  // p2: read the interpolated input component
          Id v = inputVar(attr);
          Id pInF = t.m.typePointer(spv::StorageClass::Input, t.tF);
          Id comp = t.m.load(t.tF, t.m.accessChain(pInF, v, {t.m.constU32(chan)}));
          t.stVgF(vdst, comp);
        }
        break;
      }
      case Enc::vop2: {
        uint32_t op = in.opcode, vdst = (w >> 17) & 0xFF, vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
        emitVop2(t, op, vdst, t.srcF(src0, in.literal), t.srcF(256 + vsrc1, in.literal), in.literal);
        break;
      }
      case Enc::vop1: {
        uint32_t op = in.opcode, vdst = (w >> 17) & 0xFF, src0 = w & 0x1FF;
        emitVop1(t, op, vdst, t.srcF(src0, in.literal));
        break;
      }
      case Enc::vop3: {
        uint32_t op = in.opcode, vdst = w & 0xFF, abs = (w >> 8) & 7;
        uint32_t s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF, s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
        emitVop3(t, op, vdst, t.srcF(s0, in.literal, neg & 1, abs & 1),
                 t.srcF(s1, in.literal, neg & 2, abs & 2),
                 t.srcF(s2, in.literal, neg & 4, abs & 4));
        break;
      }
      case Enc::vopc: {
        uint32_t op = in.opcode, vsrc1 = (w >> 9) & 0xFF, src0 = w & 0x1FF;
        emitVopc(t, op, t.srcF(src0, in.literal), t.srcF(256 + vsrc1, in.literal),
                 t.srcRaw(src0, in.literal), t.srcRaw(256 + vsrc1, in.literal));
        break;
      }
      case Enc::mimg: {
        emitMimg(t, in.opcode, w, w1, imgTy, sampImgTy, pSampImg, r);
        break;
      }
      case Enc::exp: {
        uint32_t en = w & 0xF, target = (w >> 4) & 0x3F, compr = (w >> 10) & 1;
        uint32_t v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF, (w1 >> 24) & 0xFF};
        if (target <= 7) {  // MRT0..7
          wroteColor = true;
          Id col;
          if (compr) {
            Id c01 = t.m.extInst(t.tV2, GLSLstd450UnpackHalf2x16, {t.ldVg(v[0])});
            Id c23 = t.m.extInst(t.tV2, GLSLstd450UnpackHalf2x16, {t.ldVg(v[1])});
            col = t.m.vectorShuffle(t.tV4, c01, c23, {0, 1, 2, 3});
          } else {
            Id comps[4];
            for (int i = 0; i < 4; i++)
              comps[i] = (en & (1 << i)) ? t.ldVgF(v[i]) : t.fconst(i == 3 ? 1.0f : 0.0f);
            col = t.m.compositeConstruct(t.tV4, {comps[0], comps[1], comps[2], comps[3]});
          }
          t.m.store(colorOutVar(target), col);
        }
        break;
      }
      default: break;
    }
    if (in.enc == Enc::sopp && in.opcode == 1) break;
  }
  if (!wroteColor)
    t.m.store(colorOutVar(0), t.m.constComposite(t.tV4,
              {t.fconst(1.f), t.fconst(1.f), t.fconst(1.f), t.fconst(1.f)}));

  t.m.returnVoid();
  t.m.endFunction();
  t.m.entryPoint(spv::ExecutionModel::Fragment, main, "main", iface);
  t.m.execMode(main, spv::ExecutionMode::OriginUpperLeft);
  return true;
}

}  // namespace

bool recompileSpirv(const uint32_t *vsCode, const uint32_t *psCode,
                    const uint32_t *vsUserData, const uint32_t *psUserData,
                    Recompiled &r) {
  if (!vsCode || !psCode || !vsUserData || !psUserData) return false;
  // One-shot disassembly (DELTA_GPU_SHDIS): for the first branchy shader, list each
  // instruction's encoding + opcode so we can see which ops the CFG path must handle.
  if (std::getenv("DELTA_GPU_SHDIS")) {
    static int dn = 0;
    auto dump = [&](const char *tag, const uint32_t *code) {
      auto ins = decode(code, 4096);
      if (!hasControlFlow(ins) || dn >= 2) return;
      dn++;
      static const char *encName[] = {"unk","sop1","sop2","sopk","sopc","sopp","smrd",
        "vop1","vop2","vop3","vopc","vintrp","ds","mubuf","mtbuf","mimg","exp"};
      std::fprintf(stderr, "[shdis] %s branchy, %zu insts:\n", tag, ins.size());
      for (auto &in : ins)
        std::fprintf(stderr, "[shdis]  pc=%u %s op=%#x w0=%#x w1=%#x\n", in.pc,
                     encName[(int)in.enc <= 16 ? (int)in.enc : 0], in.opcode,
                     in.raw[0], in.raw[1]);
    };
    dump("VS", vsCode);
    dump("PS", psCode);
  }
  // VS and PS are separate SPIR-V modules (separate Tr/Module each).
  Tr tv;
  if (!translateVs(vsCode, vsUserData, r, tv)) return false;
  Tr tp;
  tp.initTypes();
  if (!translatePs(psCode, r, tp)) return false;

  auto vs = tv.m.assemble();
  auto ps = tp.m.assemble();
  std::string err;
  if (!spirv::validate(vs, &err)) {
    if (g_dbg) std::fprintf(stderr, "[gcnspv] VS invalid: %s\n", err.c_str());
    return false;
  }
  if (!spirv::validate(ps, &err)) {
    if (g_dbg) std::fprintf(stderr, "[gcnspv] PS invalid: %s\n", err.c_str());
    return false;
  }
  // DELTA_GPU_SPIRV_NOOPT: skip the optimize pass (use the naive memory-backed
  // register SPIR-V). Diagnostic: isolates an emission bug from a spirv-opt
  // mis-promotion (the naive form keeps the register file in memory, always correct).
  static const bool noOpt = std::getenv("DELTA_GPU_SPIRV_NOOPT") != nullptr;
  r.vsSpirv = noOpt ? vs : spirv::optimize(vs);
  r.fsSpirv = noOpt ? ps : spirv::optimize(ps);
  r.ok = !r.vsSpirv.empty() && !r.fsSpirv.empty();
  // Tally (DELTA_GPU_SPIRV): how many shaders the direct SPIR-V backend accepted vs
  // had to decline (-> GLSL fallback), and how many used the CFG path. Confirms the
  // backend is actually in use rather than silently falling back.
  static const bool tally = std::getenv("DELTA_GPU_SPIRV") != nullptr;
  if (tally) {
    static int okN = 0, cfN = 0; static int logged = 0;
    if (r.ok) okN++;
    if (hasControlFlow(decode(vsCode, 4096)) || hasControlFlow(decode(psCode, 4096))) cfN++;
    if (logged < 12) { logged++;
      std::fprintf(stderr, "[gcnspv] recompiled ok=%d (cfg-shaders=%d) this=%s\n", okN, cfN,
                   r.ok ? "spirv" : "FALLBACK"); }
  }
  return r.ok;
}

}  // namespace gpu::gcn

#endif  // DELTA_HAVE_SPIRV_BACKEND
