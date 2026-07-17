/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN -> SPIR-V translator. See gcn_spirv.h. Emits SPIR-V directly via spv_emit
 * (modelling the GCN register file as Private variables) and relies on the
 * SPIRV-Tools optimize pass to clean the naive output up.
 */

#include "gcn_spirv.h"

#ifndef DELTA_HAVE_SPIRV_BACKEND
// Backend disabled at build time (SPIRV-Tools/Headers unavailable). There is no
// other recompiler: every recompile declines and the affected draws/dispatches
// are skipped.
namespace gpu::gcn {
bool recompileSpirv(const uint32_t *, const uint32_t *, const uint32_t *,
                    const uint32_t *, Recompiled &) { return false; }
bool recompileComputeSpirv(const uint32_t *, uint32_t, uint32_t, uint32_t, uint32_t,
                           uint32_t, RecompiledCs &) { return false; }
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

// Decode a shader bounded by its real code length (from the OrbShdr footer) so an
// early-out s_endpgm no longer truncates the stream. Falls back to the legacy
// stop-at-first-endpgm scan only when no footer is found (e.g. a driver-generated
// sub-shader without one), which never over-reads into the footer/padding.
std::vector<Inst> decodeShader(const uint32_t *code, uint32_t cap) {
  uint32_t len = codeLength(code, cap);
  if (len && len <= cap)
    return decode(code, len, /*stopAtEndpgm=*/false);
  return decode(code, cap, /*stopAtEndpgm=*/true);
}

// Vertex attribute recovered from the Gnm fetch shader: an s_load_dwordx4 of the
// V# (from the vertex-buffer table a user SGPR points at) + a buffer_load_format
// into the destination VGPRs, one per attribute in semantic order.
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
  Id pcType = 0;    // shared CB { uvec4 data[64]; } type
  std::unordered_map<uint32_t, Id> pcVars;  // binding -> cbuffer UBO variable
  Id imgTypes[4] = {};      // sampled 2D / 2D-array, color / depth image types
  Id sampledTypes[4] = {};  // corresponding combined image-sampler types
  Id sampledPtrs[4] = {};   // UniformConstant pointers to sampledTypes
  bool imageQuery = false;

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
  Id ieq(Id a, Id b) { return m.emit(spv::Op::OpIEqual, tBool, {a, b}); }
  Id ult(Id a, Id b) { return m.emit(spv::Op::OpULessThan, tBool, {a, b}); }
  Id ule(Id a, Id b) { return m.emit(spv::Op::OpULessThanEqual, tBool, {a, b}); }
  Id uge(Id a, Id b) { return m.emit(spv::Op::OpUGreaterThanEqual, tBool, {a, b}); }
  Id land(Id a, Id b) { return m.emit(spv::Op::OpLogicalAnd, tBool, {a, b}); }

  // Declare a cbuffer as CB { uvec4 data[64]; } at set 1. Push constants cannot
  // cover the 1 KiB windows used by graphics shaders; separate bindings preserve
  // the distinct V# resources selected by each s_buffer_load.
  Id ensurePc(uint32_t binding) {
    auto it = pcVars.find(binding);
    if (it != pcVars.end()) return it->second;
    if (!pcType) {
      Id arr = m.typeArray(m.typeVec(tU, 4), 64);
      m.decorate(arr, spv::Decoration::ArrayStride, {16});
      pcType = m.typeStruct({arr});
      m.decorate(pcType, spv::Decoration::Block);
      m.memberDecorate(pcType, 0, spv::Decoration::Offset, {0});
    }
    Id v = m.variable(m.typePointer(spv::StorageClass::Uniform, pcType),
                      spv::StorageClass::Uniform);
    m.decorate(v, spv::Decoration::DescriptorSet, {1});
    m.decorate(v, spv::Decoration::Binding, {binding});
    pcVars[binding] = v;
    return v;
  }
  // Read cbuffer dword k (== uvec4 data[k>>2][k&3]) as a uint Id. Clamp the uvec4
  // index into the 64-element (1 KiB) window so an out-of-range constant index can't
  // produce an invalid SPIR-V access chain.
  Id pcDword(uint32_t binding, uint32_t k) {
    Id pcVar = ensurePc(binding);
    uint32_t v4 = (k >> 2) & 63;
    Id pU = m.typePointer(spv::StorageClass::Uniform, tU);
    Id ch = m.accessChain(pU, pcVar,
                          {m.constU32(0), m.constU32(v4), m.constU32(k & 3)});
    return m.load(tU, ch);
  }
  Id pcDwordId(uint32_t binding, Id k) {
    Id pcVar = ensurePc(binding);
    Id v4 = umin(shr(k, u32(2)), u32(63));
    Id pU = m.typePointer(spv::StorageClass::Uniform, tU);
    Id ch = m.accessChain(pU, pcVar, {u32(0), v4, band(k, u32(3))});
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

  // Integer (uint-domain) helpers. Operands/results are raw uint Ids; signed ops
  // bitcast through tI where the semantics require it. Shifts mask the amount to
  // [4:0] as GCN does.
  Id u32(uint32_t v) { return m.constU32(v); }
  Id iadd(Id a, Id b) { return m.emit(spv::Op::OpIAdd, tU, {a, b}); }
  Id isub(Id a, Id b) { return m.emit(spv::Op::OpISub, tU, {a, b}); }
  Id imul(Id a, Id b) { return m.emit(spv::Op::OpIMul, tU, {a, b}); }
  Id band(Id a, Id b) { return m.emit(spv::Op::OpBitwiseAnd, tU, {a, b}); }
  Id bor(Id a, Id b) { return m.emit(spv::Op::OpBitwiseOr, tU, {a, b}); }
  Id bxor(Id a, Id b) { return m.emit(spv::Op::OpBitwiseXor, tU, {a, b}); }
  Id bnot(Id a) { return m.emit(spv::Op::OpNot, tU, {a}); }
  Id shl(Id a, Id s) { return m.emit(spv::Op::OpShiftLeftLogical, tU, {a, band(s, u32(31))}); }
  Id shr(Id a, Id s) { return m.emit(spv::Op::OpShiftRightLogical, tU, {a, band(s, u32(31))}); }
  Id sar(Id a, Id s) {
    return m.bitcast(tU, m.emit(spv::Op::OpShiftRightArithmetic, tI,
                                {m.bitcast(tI, a), band(s, u32(31))}));
  }
  Id smin(Id a, Id b) { return m.bitcast(tU, m.extInst(tI, GLSLstd450SMin, {m.bitcast(tI, a), m.bitcast(tI, b)})); }
  Id smax(Id a, Id b) { return m.bitcast(tU, m.extInst(tI, GLSLstd450SMax, {m.bitcast(tI, a), m.bitcast(tI, b)})); }
  Id umin(Id a, Id b) { return m.extInst(tU, GLSLstd450UMin, {a, b}); }
  Id umax(Id a, Id b) { return m.extInst(tU, GLSLstd450UMax, {a, b}); }
  Id iselNZ(Id cond, Id a, Id b) {  // cond(uint!=0) ? a : b
    return m.emit(spv::Op::OpSelect, tU, {isNonZero(cond), a, b});
  }
  Id iselB(Id b, Id a, Id c) { return m.emit(spv::Op::OpSelect, tU, {b, a, c}); }
  Id popcnt(Id a) { return m.emit(spv::Op::OpBitCount, tU, {a}); }
  Id bitrev(Id a) { return m.emit(spv::Op::OpBitReverse, tU, {a}); }
  Id sext24(Id a) {  // sign-extend the low 24 bits
    return m.bitcast(tU, m.emit(spv::Op::OpBitFieldSExtract, tI, {m.bitcast(tI, a), u32(0), u32(24)}));
  }
  Id pairU() { return m.typeStruct({tU, tU}); }  // {result, carry/hi} for extended ops

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
  // High dword of a 64-bit VOP3 source. Register operands use the adjacent
  // SGPR/VGPR; inline and literal operands are extended from their low dword.
  Id srcRawHi(uint32_t field, uint32_t literal, bool signExtend) {
    if (field <= 126) return ldSg(field + 1);
    if (field >= 256 && field <= 510) return ldVg(field - 255);
    Id lo = srcRaw(field, literal);
    return signExtend ? sar(lo, u32(31)) : u32(0);
  }
  // float source with neg/abs modifiers (mirrors srcF).
  Id srcF(uint32_t field, uint32_t literal, bool neg = false, bool abs = false) {
    Id f = m.bitcast(tF, srcRaw(field, literal));
    if (abs) f = ext1(GLSLstd450FAbs, f);
    if (neg) f = fneg(f);
    return f;
  }
};

struct CarryResult {
  Id value;
  Id flag;
};

CarryResult addCarry(Tr &t, Id a, Id b, Id carry = 0) {
  Id p = t.m.emit(spv::Op::OpIAddCarry, t.pairU(), {a, b});
  Id value = t.m.compositeExtract(t.tU, p, 0);
  Id flag = t.m.compositeExtract(t.tU, p, 1);
  if (carry) {
    Id q = t.m.emit(spv::Op::OpIAddCarry, t.pairU(), {value, t.band(carry, t.u32(1))});
    value = t.m.compositeExtract(t.tU, q, 0);
    flag = t.bor(flag, t.m.compositeExtract(t.tU, q, 1));
  }
  return {value, flag};
}

CarryResult subBorrow(Tr &t, Id a, Id b, Id borrow = 0) {
  Id p = t.m.emit(spv::Op::OpISubBorrow, t.pairU(), {a, b});
  Id value = t.m.compositeExtract(t.tU, p, 0);
  Id flag = t.m.compositeExtract(t.tU, p, 1);
  if (borrow) {
    Id q = t.m.emit(spv::Op::OpISubBorrow, t.pairU(), {value, t.band(borrow, t.u32(1))});
    value = t.m.compositeExtract(t.tU, q, 0);
    flag = t.bor(flag, t.m.compositeExtract(t.tU, q, 1));
  }
  return {value, flag};
}

// ---- VOP emitters -----------------------------------------------------------
// Shared by all stages (VS/PS/CS): integer ops bitcast through tU/tI as needed,
// so passing float-typed sources is lossless. Opcode numbering is the GFX7
// (Sea Islands / Liverpool) ISA:
// https://docs.amd.com/v/u/en-US/sea-islands-instruction-set-architecture_0
void emitVop1(Tr &t, uint32_t op, uint32_t vdst, Id s0, bool clamp = false);
void emitVop2(Tr &t, uint32_t op, uint32_t vdst, Id s0, Id s1,
              uint32_t literal = 0, bool clamp = false);

void emitVop1(Tr &t, uint32_t op, uint32_t vdst, Id s0, bool clamp) {
  auto setF = [&](Id f) {
    if (clamp) f = t.m.extInst(t.tF, GLSLstd450FClamp,
                                {f, t.fconst(0.0f), t.fconst(1.0f)});
    t.stVgF(vdst, f);
  };
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
    // f16<->f32 (V_CVT_F16_F32 = 0x0a, V_CVT_F32_F16 = 0x0b). cvt_f16_f32 packs the
    // half into the low half-word (high half zero); cvt_f32_f16 reads it back.
    case 0x0a: setU(t.m.extInst(t.tU, GLSLstd450PackHalf2x16,
                  {t.m.compositeConstruct(t.tV2, {s0, t.fconst(0.f)})})); break;  // cvt_f16_f32
    case 0x0b: setF(t.m.compositeExtract(t.tF,
                  t.m.extInst(t.tV2, GLSLstd450UnpackHalf2x16, {u0}), 0)); break;  // cvt_f32_f16
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
    case 0x38: setU(t.bitrev(u0)); break;                                // v_bfrev_b32
    case 0x39: {  // v_ffbh_u32: count leading zeros; -1 if src==0
      Id msb = t.m.extInst(t.tU, GLSLstd450FindUMsb, {u0});
      setU(t.iselB(t.isZero(u0), t.u32(0xFFFFFFFFu), t.isub(t.u32(31), msb)));
      break;
    }
    case 0x3a: setU(t.m.extInst(t.tU, GLSLstd450FindILsb, {u0})); break;  // v_ffbl_b32
    case 0x3b: {  // v_ffbh_i32: leading-sign-bit count; -1 if src is 0 or -1
      Id smsb = t.m.bitcast(t.tU, t.m.extInst(t.tI, GLSLstd450FindSMsb, {t.m.bitcast(t.tI, u0)}));
      setU(t.iselB(t.m.emit(spv::Op::OpIEqual, t.tBool, {smsb, t.u32(0xFFFFFFFFu)}),
                   t.u32(0xFFFFFFFFu), t.isub(t.u32(31), smsb)));
      break;
    }
    default: warnUnsup("vop1", op); setU(u0); break;                     // mov fallback
  }
}

void emitVop2(Tr &t, uint32_t op, uint32_t vdst, Id s0, Id s1,
              uint32_t literal, bool clamp) {
  auto setF = [&](Id f) {
    if (clamp) f = t.m.extInst(t.tF, GLSLstd450FClamp,
                                {f, t.fconst(0.0f), t.fconst(1.0f)});
    t.stVgF(vdst, f);
  };
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
    // v_readlane / v_writelane: pick/write a specific wave lane. Our SPIR-V is
    // per-invocation (one lane), so there is no other lane to reach: the value is
    // just s0 (the common uniform-broadcast use is exact).
    case 0x01: case 0x02: setU(u0); break;
    case 0x03: setF(t.fadd(s0, s1)); break;                             // v_add_f32
    case 0x04: setF(t.fsub(s0, s1)); break;                             // v_sub_f32
    case 0x05: setF(t.fsub(s1, s0)); break;                             // v_subrev_f32
    case 0x06: setF(t.fadd(t.fmul(s0, s1), t.ldVgF(vdst))); break;      // v_mac_legacy_f32
    case 0x07: case 0x08: setF(t.fmul(s0, s1)); break;                  // v_mul[_legacy]_f32
    case 0x09: setU(t.imul(t.sext24(u0), t.sext24(u1))); break;         // v_mul_i32_i24
    case 0x0b: setU(t.imul(t.band(u0, t.u32(0xFFFFFF)), t.band(u1, t.u32(0xFFFFFF)))); break;  // v_mul_u32_u24
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
    case 0x22: setU(t.iadd(t.popcnt(u0), u1)); break;                   // v_bcnt_u32_b32
    // v_mbcnt_lo/hi_u32_b32 compute this lane's index within the wave. Our SPIR-V
    // is already per-invocation (one lane), so there are no prior lanes to count:
    // the running accumulator s1 passes through unchanged.
    case 0x23: case 0x24: setU(u1); break;
    case 0x25: {  // v_add_i32: carry-out -> VCC
      CarryResult r = addCarry(t, u0, u1);
      setU(r.value); t.stSg(106, r.flag); break;
    }
    case 0x26: {  // v_sub_i32: borrow-out -> VCC
      CarryResult r = subBorrow(t, u0, u1);
      setU(r.value); t.stSg(106, r.flag); break;
    }
    case 0x27: {  // v_subrev_i32: borrow-out -> VCC
      CarryResult r = subBorrow(t, u1, u0);
      setU(r.value); t.stSg(106, r.flag); break;
    }
    case 0x28: {  // v_addc_u32: s0 + s1 + VCC, carry-out -> VCC
      CarryResult r = addCarry(t, u0, u1, t.ldSg(106));
      setU(r.value); t.stSg(106, r.flag); break;
    }
    case 0x29: {  // v_subb_u32: s0 - s1 - VCC, borrow-out -> VCC
      CarryResult r = subBorrow(t, u0, u1, t.ldSg(106));
      setU(r.value); t.stSg(106, r.flag); break;
    }
    case 0x2a: {  // v_subbrev_u32: s1 - s0 - VCC, borrow-out -> VCC
      CarryResult r = subBorrow(t, u1, u0, t.ldSg(106));
      setU(r.value); t.stSg(106, r.flag); break;
    }
    case 0x2b: setF(t.m.extInst(t.tF, GLSLstd450Ldexp, {s0, t.m.bitcast(t.tI, u1)})); break;  // v_ldexp_f32
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

bool isVop3b(uint32_t op) {
  return (op >= 0x125 && op <= 0x12a) || op == 0x16d || op == 0x16e ||
         op == 0x176 || op == 0x177;
}

void emitVop3(Tr &t, uint32_t op, uint32_t vdst, Id s0, Id s1, Id s2, Id s2hi,
              uint32_t sdst, bool clamp) {
  // VOP3 reflects the VOPC (0x000-0x0FF), VOP2 (0x100-0x13F) and VOP1 (0x180-0x1FF)
  // encodings; only 0x140-0x17F are VOP3-exclusive.
  Id u0 = t.m.bitcast(t.tU, s0), u1 = t.m.bitcast(t.tU, s1), u2 = t.m.bitcast(t.tU, s2);
  auto setF = [&](Id f) {
    if (clamp) f = t.m.extInst(t.tF, GLSLstd450FClamp,
                                {f, t.fconst(0.0f), t.fconst(1.0f)});
    t.stVgF(vdst, f);
  };
  auto setU = [&](Id u) { t.stVg(vdst, u); };
  if (op < 0x100) {  // VOPC compare in VOP3 form: writes the mask to sgpr[vdst]
    emitVopc(t, op, s0, s1, u0, u1, vdst); return;
  }
  if (op == 0x100) {  // VOP3 cndmask uses explicit S2 instead of implicit VCC
    Id cond = t.isNonZero(t.band(u2, t.u32(1)));
    setU(t.m.emit(spv::Op::OpSelect, t.tU, {cond, u1, u0}));
    return;
  }
  if (op >= 0x125 && op <= 0x12a) {  // VOP3B integer add/sub + explicit SDST
    CarryResult r;
    if (op == 0x125) r = addCarry(t, u0, u1);
    else if (op == 0x126) r = subBorrow(t, u0, u1);
    else if (op == 0x127) r = subBorrow(t, u1, u0);
    else if (op == 0x128) r = addCarry(t, u0, u1, u2);
    else if (op == 0x129) r = subBorrow(t, u0, u1, u2);
    else r = subBorrow(t, u1, u0, u2);
    setU(r.value);
    t.stSg(sdst, r.flag);
    return;
  }
  if (op >= 0x100 && op < 0x140) {
    emitVop2(t, op - 0x100, vdst, s0, s1, 0, clamp);
    return;
  }
  if (op >= 0x180 && op < 0x200) {
    emitVop1(t, op - 0x180, vdst, s0, clamp);
    return;
  }
  auto mulHi = [&](spv::Op mulOp) {  // high 32 bits of a 64-bit product
    Id st = t.m.typeStruct({t.tU, t.tU});
    return t.m.compositeExtract(t.tU, t.m.emit(mulOp, st, {u0, u1}), 1);
  };
  // Median of 3 (no GLSL medN): max(min(a,b), min(max(a,b), c)).
  auto med3 = [&](Id (Tr::*mn)(Id, Id), Id (Tr::*mx)(Id, Id)) {
    return (t.*mx)((t.*mn)(u0, u1), (t.*mn)((t.*mx)(u0, u1), u2));
  };
  switch (op) {
    case 0x140: case 0x141: case 0x14b:                                 // v_mad[_legacy]_f32 / v_fma_f32
      setF(t.fadd(t.fmul(s0, s1), s2)); break;
    case 0x142:                                                         // v_mad_i32_i24
      setU(t.iadd(t.imul(t.sext24(u0), t.sext24(u1)), u2)); break;
    case 0x143:                                                         // v_mad_u32_u24
      setU(t.iadd(t.imul(t.band(u0, t.u32(0xFFFFFF)), t.band(u1, t.u32(0xFFFFFF))), u2)); break;
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
    case 0x144: case 0x145: case 0x146: case 0x147: {  // v_cube{id,sc,tc,ma}_f32
      // GFX7 cube-coordinate preparation, with the ISA's Z > Y > X tie priority.
      // https://docs.amd.com/v/u/en-US/sea-islands-instruction-set-architecture_0
      Id ax = t.ext1(GLSLstd450FAbs, s0);
      Id ay = t.ext1(GLSLstd450FAbs, s1);
      Id az = t.ext1(GLSLstd450FAbs, s2);
      Id zGeX = t.m.emit(spv::Op::OpFOrdGreaterThanEqual, t.tBool, {az, ax});
      Id zGeY = t.m.emit(spv::Op::OpFOrdGreaterThanEqual, t.tBool, {az, ay});
      Id zMajor = t.m.emit(spv::Op::OpLogicalAnd, t.tBool, {zGeX, zGeY});
      Id yGeX = t.m.emit(spv::Op::OpFOrdGreaterThanEqual, t.tBool, {ay, ax});
      Id notZ = t.m.emit(spv::Op::OpLogicalNot, t.tBool, {zMajor});
      Id yMajor = t.m.emit(spv::Op::OpLogicalAnd, t.tBool, {notZ, yGeX});
      Id xNeg = t.m.emit(spv::Op::OpFOrdLessThan, t.tBool, {s0, t.fconst(0.0f)});
      Id yNeg = t.m.emit(spv::Op::OpFOrdLessThan, t.tBool, {s1, t.fconst(0.0f)});
      Id zNeg = t.m.emit(spv::Op::OpFOrdLessThan, t.tBool, {s2, t.fconst(0.0f)});
      auto fsel = [&](Id cond, Id a, Id b) {
        return t.m.emit(spv::Op::OpSelect, t.tF, {cond, a, b});
      };

      Id result;
      if (op == 0x144) {  // face: +X,-X,+Y,-Y,+Z,-Z => 0..5
        Id xFace = fsel(xNeg, t.fconst(1.0f), t.fconst(0.0f));
        Id yFace = fsel(yNeg, t.fconst(3.0f), t.fconst(2.0f));
        Id zFace = fsel(zNeg, t.fconst(5.0f), t.fconst(4.0f));
        result = fsel(zMajor, zFace, fsel(yMajor, yFace, xFace));
      } else if (op == 0x145) {  // horizontal face coordinate
        Id xSc = fsel(xNeg, s2, t.fneg(s2));
        Id zSc = fsel(zNeg, t.fneg(s0), s0);
        result = fsel(zMajor, zSc, fsel(yMajor, s0, xSc));
      } else if (op == 0x146) {  // vertical face coordinate
        Id yTc = fsel(yNeg, t.fneg(s2), s2);
        result = fsel(yMajor, yTc, t.fneg(s1));
      } else {  // signed twice-major-axis value used for normalization
        Id major = fsel(zMajor, s2, fsel(yMajor, s1, s0));
        result = t.fmul(t.fconst(2.0f), major);
      }
      setF(result);
      break;
    }
    case 0x14d: {  // v_lerp_u8: per-byte (a + b + (c&1)) >> 1
      Id r = t.u32(0);
      for (int b = 0; b < 4; b++) {
        Id shb = t.u32((uint32_t)b * 8), mask = t.u32(0xFF);
        Id a = t.band(t.shr(u0, shb), mask), bb = t.band(t.shr(u1, shb), mask);
        Id cc = t.band(t.shr(u2, shb), t.u32(1));
        Id avg = t.shr(t.iadd(t.iadd(a, bb), cc), t.u32(1));
        r = t.bor(r, t.shl(avg, shb));
      }
      setU(r);
      break;
    }
    case 0x14e: {  // v_alignbit_b32: ({S0,S1} >> S2[4:0])[31:0] (S0 hi, S1 lo)
      Id shf = t.band(u2, t.u32(31));
      Id lo = t.shr(u1, shf);
      Id hi = t.iselB(t.isZero(shf), t.u32(0), t.shl(u0, t.isub(t.u32(32), shf)));
      setU(t.bor(lo, hi));
      break;
    }
    case 0x14f: {  // v_alignbyte_b32: byte-granular funnel shift
      Id shf = t.imul(t.band(u2, t.u32(3)), t.u32(8));
      Id lo = t.shr(u1, shf);
      Id hi = t.iselB(t.isZero(shf), t.u32(0), t.shl(u0, t.isub(t.u32(32), shf)));
      setU(t.bor(lo, hi));
      break;
    }
    case 0x169: case 0x16b: setU(t.m.emit(spv::Op::OpIMul, t.tU, {u0, u1})); break;  // v_mul_lo_u32/i32
    case 0x16a: setU(mulHi(spv::Op::OpUMulExtended)); break;             // v_mul_hi_u32
    case 0x16c: setU(mulHi(spv::Op::OpSMulExtended)); break;             // v_mul_hi_i32
    case 0x151: setF(t.ext2(GLSLstd450FMin, t.ext2(GLSLstd450FMin, s0, s1), s2)); break;  // v_min3_f32
    case 0x152: setU(t.smin(t.smin(u0, u1), u2)); break;               // v_min3_i32
    case 0x153: setU(t.umin(t.umin(u0, u1), u2)); break;               // v_min3_u32
    case 0x154: setF(t.ext2(GLSLstd450FMax, t.ext2(GLSLstd450FMax, s0, s1), s2)); break;  // v_max3_f32
    case 0x155: setU(t.smax(t.smax(u0, u1), u2)); break;               // v_max3_i32
    case 0x156: setU(t.umax(t.umax(u0, u1), u2)); break;               // v_max3_u32
    case 0x157: {  // v_med3_f32 = clamp(s2, min(s0,s1), max(s0,s1))
      Id lo = t.ext2(GLSLstd450FMin, s0, s1), hi = t.ext2(GLSLstd450FMax, s0, s1);
      setF(t.m.extInst(t.tF, GLSLstd450FClamp, {s2, lo, hi}));
      break;
    }
    case 0x158: setU(med3(&Tr::smin, &Tr::smax)); break;               // v_med3_i32
    case 0x159: setU(med3(&Tr::umin, &Tr::umax)); break;               // v_med3_u32
    case 0x15d: setU(t.iadd(t.isub(t.umax(u0, u1), t.umin(u0, u1)), u2)); break;  // v_sad_u32: |s0-s1|+s2
    // IEEE divide sequence (div_scale -> rcp -> div_fmas -> div_fixup). We short it
    // to an exact divide at the fixup (S2/S1); div_scale is an identity passthrough
    // and div_fmas an FMA feeding the estimate the fixup ignores.
    case 0x15f: setF(t.fdiv(s2, s1)); break;                          // v_div_fixup_f32 (S2/S1)
    case 0x16d: setF(s0); t.stSg(sdst, t.u32(0)); break;              // v_div_scale_f32: identity
    case 0x16f: setF(t.fadd(t.fmul(s0, s1), s2)); break;             // v_div_fmas_f32: FMA
    case 0x176: case 0x177: {  // v_mad_u64_u32 / v_mad_i64_i32
      bool sgn = (op == 0x177);
      Id prod = t.m.emit(sgn ? spv::Op::OpSMulExtended : spv::Op::OpUMulExtended,
                          t.pairU(), {u0, u1});
      Id plo = t.m.compositeExtract(t.tU, prod, 0), phi = t.m.compositeExtract(t.tU, prod, 1);
      CarryResult lo = addCarry(t, plo, u2);
      CarryResult hi = addCarry(t, phi, s2hi, lo.flag);
      setU(lo.value);
      if (vdst + 1 < 256) t.stVg(vdst + 1, hi.value);
      Id bit64 = hi.flag;
      if (sgn)
        bit64 = t.bxor(bit64, t.bxor(t.bxor(t.shr(phi, t.u32(31)),
                                              t.shr(s2hi, t.u32(31))),
                                       t.shr(hi.value, t.u32(31))));
      t.stSg(sdst, t.band(bit64, t.u32(1)));
      break;
    }
    default: warnUnsup("vop3", op); setF(s0); break;
  }
}

// ---- shared per-instruction emit + CFG while-switch -------------------------
// Per-stage state carried into the shared emitter (emitInst). Single-basic-block
// shaders emit the instruction stream straight-line; a shader with branches (or
// DELTA_GPU_SPIRV_CFG forcing it) goes through emitCFG, which lowers arbitrary
// control flow to a while/switch state machine over basic blocks (handles
// reducible and irreducible CFGs).
struct StageCtx {
  bool isPs = false;
  Recompiled *r = nullptr;
  std::vector<Id> *iface = nullptr;
  Id posOut = 0;                               // VS
  std::unordered_map<uint32_t, Id> paramOuts;  // VS
  uint32_t maxParam = 0;                        // VS
  Id colorOuts[8] = {};                        // PS (location == MRT target)
  std::unordered_map<uint32_t, Id> inVars;     // PS
  bool wroteColor = false;                      // PS (compile-time: shader has an exp)
  Id colorWrittenVar = 0;  // PS (runtime per-lane: this fragment reached a color exp)
  std::unordered_map<uint32_t, uint32_t> cbufBind;  // descriptor SGPR -> set-1 binding
  const std::unordered_set<uint32_t> *flatAttrs = nullptr;  // V_INTERP_MOV P0 locations

  // Compute (isCs): storage buffers modelling the guest memory the CS reads/writes.
  // csBind maps a descriptor SGPR (V#/T# base) to its storage-buffer binding; csSsbo
  // maps a binding to the emitted OpVariable. csUnsupported is set if an op the compute
  // backend can't handle is reached (caller declines the recompile).
  bool isCs = false;
  std::unordered_map<uint32_t, uint32_t> csBind;  // baseSgpr -> binding
  std::vector<Id> csSsbo;                          // binding -> storage-buffer var
  bool csUnsupported = false;
};

uint32_t smrdLoadCount(uint32_t op) {
  if (op < 0x08 || op > 0x0c) return 0;
  return op == 0x08 ? 1 : op == 0x09 ? 2 : op == 0x0a ? 4 :
         op == 0x0b ? 8 : 16;
}

void emitCbufSmrd(Tr &t, const Inst &in,
                   const std::unordered_map<uint32_t, uint32_t> &bindings) {
  uint32_t w = in.raw[0], n = smrdLoadCount(in.opcode);
  uint32_t sdst = (w >> 15) & 0x7F, baseSgpr = ((w >> 9) & 0x3F) * 2;
  bool imm = (w >> 8) & 1;
  auto it = bindings.find(baseSgpr);
  if (!n || it == bindings.end()) return;
  uint32_t off = w & 0xFF;
  if (imm) {
    for (uint32_t i = 0; i < n; i++)
      t.stSg(sdst + i, t.pcDword(it->second, off + i));
  } else {
    Id base = t.shr(t.srcRaw(off, in.literal), t.u32(2));
    for (uint32_t i = 0; i < n; i++)
      t.stSg(sdst + i, t.pcDwordId(it->second, t.iadd(base, t.u32(i))));
  }
}

bool planCbufs(const std::vector<Inst> &insts, uint32_t firstBinding,
               std::vector<ShaderCbuf> &cbufs,
               std::unordered_map<uint32_t, uint32_t> &bindings) {
  for (const auto &in : insts) {
    if (in.enc != Enc::smrd) continue;
    uint32_t n = smrdLoadCount(in.opcode);
    if (!n) continue;
    uint32_t w = in.raw[0], baseSgpr = ((w >> 9) & 0x3F) * 2;
    if (baseSgpr + 3 >= 16) continue;
    auto [it, inserted] = bindings.emplace(baseSgpr, firstBinding + cbufs.size());
    if (inserted) {
      if (it->second >= 8) return false;
      cbufs.push_back({it->second, baseSgpr, 0});
    }
    uint32_t end = ((w >> 8) & 1) ? (w & 0xFF) + n : 256;
    for (auto &cb : cbufs)
      if (cb.binding == it->second) cb.numDwords = std::max(cb.numDwords, end);
  }
  return true;
}

Id psInputVar(Tr &t, StageCtx &sc, uint32_t attr) {
  auto it = sc.inVars.find(attr);
  if (it != sc.inVars.end()) return it->second;
  Id v = t.m.variable(t.m.typePointer(spv::StorageClass::Input, t.tV4), spv::StorageClass::Input);
  t.m.decorate(v, spv::Decoration::Location, {attr});
  if (sc.flatAttrs && sc.flatAttrs->count(attr))
    t.m.decorate(v, spv::Decoration::Flat);
  sc.iface->push_back(v);
  sc.inVars[attr] = v;
  return v;
}
Id vsParamOut(Tr &t, StageCtx &sc, uint32_t p) {
  auto it = sc.paramOuts.find(p);
  if (it != sc.paramOuts.end()) return it->second;
  Id v = t.m.variable(t.m.typePointer(spv::StorageClass::Output, t.tV4), spv::StorageClass::Output);
  t.m.decorate(v, spv::Decoration::Location, {p});
  if (sc.flatAttrs && sc.flatAttrs->count(p))
    t.m.decorate(v, spv::Decoration::Flat);
  sc.iface->push_back(v);
  sc.paramOuts[p] = v;
  return v;
}
// Lazily declare the PS color output for an MRT target (location == target) and
// record it in psMrtMask so the renderer masks the unwritten attachments.
Id psColorOut(Tr &t, StageCtx &sc, uint32_t target) {
  if (sc.colorOuts[target]) return sc.colorOuts[target];
  Id v = t.m.variable(t.m.typePointer(spv::StorageClass::Output, t.tV4),
                      spv::StorageClass::Output);
  t.m.decorate(v, spv::Decoration::Location, {target});
  sc.iface->push_back(v);
  sc.colorOuts[target] = v;
  sc.r->psMrtMask |= (uint8_t)(1u << target);
  return v;
}

// Emit a MIMG image op. Declares the texture as a combined sampler at set 0 /
// binding = the MIMG order, records it in psTexs, and samples/fetches it. MIMG DA
// selects a 2D-array resource and adds the layer coordinate after x/y. The sampler
// resource (S#) uses the renderer's default sampler.
void emitMimg(Tr &t, uint32_t op, uint32_t w0, uint32_t w1, Recompiled &r) {
  uint32_t dmask = (w0 >> 8) & 0xF, vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4, bind = (uint32_t)r.psTexs.size();
  bool arrayed = (w0 & 0x4000) != 0;
  r.psTexs.push_back({bind, srsrc});
  bool dref = op == 0x28 || op == 0x2f;
  bool offset = op == 0x37;
  bool gather = op == 0x47;
  uint32_t typeIdx = (arrayed ? 1u : 0u) | (dref ? 2u : 0u);
  if (!t.imgTypes[typeIdx]) {
    t.imgTypes[typeIdx] = t.m.typeImage(t.tF, spv::Dim::Dim2D, dref ? 1 : 0,
                                        arrayed ? 1 : 0, 0, 1,
                                        spv::ImageFormat::Unknown);
    t.sampledTypes[typeIdx] = t.m.typeSampledImage(t.imgTypes[typeIdx]);
    t.sampledPtrs[typeIdx] = t.m.typePointer(spv::StorageClass::UniformConstant,
                                             t.sampledTypes[typeIdx]);
  }
  Id imgTy = t.imgTypes[typeIdx], sampImgTy = t.sampledTypes[typeIdx];
  Id pSampImg = t.sampledPtrs[typeIdx];
  Id texVar = t.m.variable(pSampImg, spv::StorageClass::UniformConstant);
  t.m.decorate(texVar, spv::Decoration::DescriptorSet, {0});
  t.m.decorate(texVar, spv::Decoration::Binding, {bind});
  Id si = t.m.load(sampImgTy, texVar);
  uint32_t bodyAddr = vaddr + (offset ? 1u : 0u) + (dref ? 1u : 0u);
  Id x = t.ldVgF(bodyAddr), y = t.ldVgF(bodyAddr + 1);
  if (offset) {
    // GFX7 packs signed six-bit X/Y texel offsets before the address body.
    // Vulkan 1.1 does not permit a dynamic Offset operand on OpImageSample*, so
    // apply the normalized-coordinate equivalent using the level-zero image size.
    if (!t.imageQuery) {
      t.m.capability(spv::Capability::ImageQuery);
      t.imageQuery = true;
    }
    Id packed = t.m.bitcast(t.tI, t.ldVg(vaddr));
    Id ox = t.m.emit(spv::Op::OpBitFieldSExtract, t.tI,
                     {packed, t.m.constU32(0), t.m.constU32(6)});
    Id oy = t.m.emit(spv::Op::OpBitFieldSExtract, t.tI,
                     {packed, t.m.constU32(8), t.m.constU32(6)});
    Id image = t.m.emit(spv::Op::OpImage, imgTy, {si});
    Id sizeTy = t.m.typeVec(t.tU, arrayed ? 3 : 2);
    Id size = t.m.emit(spv::Op::OpImageQuerySizeLod, sizeTy,
                       {image, t.m.constU32(0)});
    Id sx = t.m.emit(spv::Op::OpConvertUToF, t.tF,
                     {t.m.compositeExtract(t.tU, size, 0)});
    Id sy = t.m.emit(spv::Op::OpConvertUToF, t.tF,
                     {t.m.compositeExtract(t.tU, size, 1)});
    Id ofx = t.m.emit(spv::Op::OpConvertSToF, t.tF, {ox});
    Id ofy = t.m.emit(spv::Op::OpConvertSToF, t.tF, {oy});
    x = t.fadd(x, t.fdiv(ofx, sx));
    y = t.fadd(y, t.fdiv(ofy, sy));
  }
  Id uv = arrayed
      ? t.m.compositeConstruct(t.m.typeVec(t.tF, 3),
          {x, y, t.ldVgF(bodyAddr + 2)})
      : t.m.compositeConstruct(t.tV2, {x, y});
  uint32_t lodOp = (uint32_t)spv::ImageOperandsMask::Lod;
  bool known = op == 0x00 || op == 0x01 || op == 0x20 || op == 0x21 ||
                op == 0x24 || op == 0x25 || op == 0x27 || op == 0x28 ||
                op == 0x2f || op == 0x37 || op == 0x47;
  if (!known) warnUnsup("mimg", op, w0, w1);
  Id texel;
  if (op == 0x00 || op == 0x01) {  // image_load[_mip]: integer fetch, no filtering
    Id ix = t.m.bitcast(t.tI, t.ldVg(vaddr));
    Id iy = t.m.bitcast(t.tI, t.ldVg(vaddr + 1));
    Id ic = arrayed
        ? t.m.compositeConstruct(t.m.typeVec(t.tI, 3),
            {ix, iy, t.m.bitcast(t.tI, t.ldVg(vaddr + 2))})
        : t.m.compositeConstruct(t.m.typeVec(t.tI, 2), {ix, iy});
    Id img = t.m.emit(spv::Op::OpImage, imgTy, {si});
    Id lod = t.m.constU32(0);
    if (op == 0x01) {
      if (!t.imageQuery) {
        t.m.capability(spv::Capability::ImageQuery);
        t.imageQuery = true;
      }
      Id levels = t.m.emit(spv::Op::OpImageQueryLevels, t.tU, {img});
      lod = t.umin(t.ldVg(vaddr + (arrayed ? 3 : 2)),
                   t.isub(levels, t.u32(1)));
    }
    texel = t.m.emit(spv::Op::OpImageFetch, t.tV4, {img, ic, lodOp, lod});
  } else if (op == 0x24) {  // image_sample_l: explicit LOD in the coord+2 VGPR
    texel = t.m.emit(spv::Op::OpImageSampleExplicitLod, t.tV4,
                      {si, uv, lodOp, t.ldVgF(vaddr + (arrayed ? 3 : 2))});
  } else if (op == 0x28) {  // image_sample_c: z-compare precedes the address body
    texel = t.m.emit(spv::Op::OpImageSampleDrefImplicitLod, t.tF,
                     {si, uv, t.ldVgF(vaddr)});
  } else if (op == 0x2f) {  // image_sample_c_lz: PCF forced to level zero
    texel = t.m.emit(spv::Op::OpImageSampleDrefExplicitLod, t.tF,
                     {si, uv, t.ldVgF(vaddr), lodOp, t.fconst(0.0f)});
  } else if (op == 0x27 || op == 0x37) {  // image_sample_lz[_o]: forced LOD 0
    texel = t.m.emit(spv::Op::OpImageSampleExplicitLod, t.tV4,
                      {si, uv, lodOp, t.fconst(0.0f)});
  } else if (gather) {  // DMASK chooses the channel; gather always returns four texels
    uint32_t component = 0;
    while (component < 3 && !(dmask & (1u << component))) component++;
    texel = t.m.emit(spv::Op::OpImageGather, t.tV4,
                     {si, uv, t.m.constU32(component)});
  } else {  // image_sample / _cl / _b (bias/derivs/compare ignored): implicit LOD
    texel = t.m.emit(spv::Op::OpImageSampleImplicitLod, t.tV4, {si, uv});
  }
  if (dref) {
    if (dmask) t.stVgF(vdata, texel);
    return;
  }
  if (gather) {
    for (int i = 0; i < 4; i++)
      t.stVgF(vdata + i, t.m.compositeExtract(t.tF, texel, i));
    return;
  }
  uint32_t comp = 0;
  for (int i = 0; i < 4; i++)
    if (dmask & (1 << i)) t.stVgF(vdata + comp++, t.m.compositeExtract(t.tF, texel, i));
}

// ---- compute memory ops (guest memory modelled as storage buffers) ----------
// Each bound resource is a `uint data[]` storage buffer that aliases the guest range
// [descriptor.base, base+size); an access is by dword index relative to that base.
Id csSsboPtr(Tr &t, StageCtx &sc, uint32_t binding, Id dwordIdx) {
  Id pU = t.m.typePointer(spv::StorageClass::StorageBuffer, t.tU);
  return t.m.accessChain(pU, sc.csSsbo[binding], {t.m.constU32(0), dwordIdx});
}
Id csSsboLoad(Tr &t, StageCtx &sc, uint32_t binding, Id dwordIdx) {
  return t.m.load(t.tU, csSsboPtr(t, sc, binding, dwordIdx));
}
void csSsboStore(Tr &t, StageCtx &sc, uint32_t binding, Id dwordIdx, Id val) {
  t.m.store(csSsboPtr(t, sc, binding, dwordIdx), val);
}
int csBindingFor(StageCtx &sc, uint32_t baseSgpr) {
  auto it = sc.csBind.find(baseSgpr);
  return it != sc.csBind.end() ? (int)it->second : -1;
}
Id csMax1(Tr &t, Id value) { return t.umax(value, t.u32(1)); }

// Next power of two >= max(value, 1) (bit-smearing form).
Id csBitCeil(Tr &t, Id value) {
  Id v = t.isub(csMax1(t, value), t.u32(1));
  for (uint32_t s : {1u, 2u, 4u, 8u, 16u}) v = t.bor(v, t.shr(v, t.u32(s)));
  return t.iadd(v, t.u32(1));
}

Id csLinearMipPitch(Tr &t, Id basePitch, Id height, Id mip, Id linearGeneral,
                    Id pow2Pad) {
  Id raw = csMax1(t, t.shr(basePitch, mip));
  raw = t.iselB(pow2Pad, csBitCeil(t, raw), raw);
  Id aligned = t.band(t.iadd(raw, t.u32(15)), t.u32(~15u));
  for (uint32_t i = 0; i < 3; i++) {
    Id ok = t.isZero(t.band(t.imul(aligned, height), t.u32(63)));
    aligned = t.iselB(ok, aligned, t.iadd(aligned, t.u32(16)));
  }
  return t.iselB(linearGeneral, raw, aligned);
}

// SMRD s_load / s_buffer_load: read scalar constants from the bound buffer.
void emitCsSmrd(Tr &t, const Inst &in, StageCtx &sc) {
  uint32_t w = in.raw[0], op = in.opcode, sdst = (w >> 15) & 0x7F, sbase = (w >> 9) & 0x3F;
  bool imm = (w >> 8) & 1; uint32_t off = w & 0xFF, baseSgpr = sbase * 2;
  int b = csBindingFor(sc, baseSgpr);
  if (b < 0) { sc.csUnsupported = true; return; }
  uint32_t n = op < 0x08 ? (1u << op)
             : op == 0x08 ? 1 : op == 0x09 ? 2 : op == 0x0a ? 4 : op == 0x0b ? 8 : 16;
  // Immediate offset is a dword index; a register offset is a byte offset.
  Id dwOff = imm ? t.u32(off) : t.shr(t.srcRaw(off, in.literal), t.u32(2));
  for (uint32_t i = 0; i < n; i++)
    t.stSg(sdst + i, csSsboLoad(t, sc, (uint32_t)b, t.iadd(dwOff, t.u32(i))));
}

// MUBUF buffer_load: read a source element into a VGPR as raw bytes/dwords (no
// format conversion): copy shaders load bytes and the image store packs them, so
// the round trip is identity.
void emitCsMubuf(Tr &t, const Inst &in, StageCtx &sc) {
  uint32_t w = in.raw[0], w1 = in.raw[1];
  uint32_t op = (w >> 18) & 0x7F, instOff = w & 0xFFF;
  bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1;
  uint32_t vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4, soffField = (w1 >> 24) & 0xFF;
  if (op >= 0x18) { sc.csUnsupported = true; return; }  // buffer store: not handled
  int b = csBindingFor(sc, srsrc);
  if (b < 0) { sc.csUnsupported = true; return; }
  Id byteOff = t.iadd(t.srcRaw(soffField, in.literal), t.u32(instOff));
  if (idxen) {
    Id stride = t.band(t.shr(t.ldSg(srsrc + 1), t.u32(16)), t.u32(0x3FFF));
    byteOff = t.iadd(byteOff, t.imul(t.ldVg(vaddr), stride));
  } else if (offen) {
    byteOff = t.iadd(byteOff, t.ldVg(vaddr));
  }
  Id raw = csSsboLoad(t, sc, (uint32_t)b, t.shr(byteOff, t.u32(2)));
  bool fourByte = (op == 0x0c || op == 0x03);  // LOAD_DWORD / FORMAT_XYZW
  if (fourByte)
    t.stVg(vdata, raw);
  else
    t.stVg(vdata, t.band(t.shr(raw, t.shl(t.band(byteOff, t.u32(3)), t.u32(3))),
                         t.u32(0xFF)));
}

// MIMG image_load/store[_mip] for staged linear RGBA8 images. Storage is mip-major;
// each level contains all physical array layers and explicit LOD is view-relative.
void emitCsMimg(Tr &t, const Inst &in, StageCtx &sc) {
  uint32_t w = in.raw[0], w1 = in.raw[1];
  uint32_t op = (w >> 18) & 0x7F, dmask = (w >> 8) & 0xF;
  uint32_t vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF, srsrc = ((w1 >> 16) & 0x1F) * 4;
  bool mipOp = op == 0x01 || op == 0x09;
  bool store = op == 0x08 || op == 0x09;
  bool load = op == 0x00 || op == 0x01;
  if (!store && !load) { sc.csUnsupported = true; return; }  // sample/atomic: not handled
  int b = csBindingFor(sc, srsrc);
  if (b < 0) { sc.csUnsupported = true; return; }
  const bool da = (w & 0x4000) != 0;
  // T# field extraction (see decodeTImage for the dword layout).
  auto field = [&](uint32_t dword, uint32_t shift, uint32_t mask) {
    return t.band(t.shr(t.ldSg(srsrc + dword), t.u32(shift)), t.u32(mask));
  };
  Id x = t.ldVg(vaddr), y = t.ldVg(vaddr + 1);
  Id baseWidth = t.iadd(field(2, 0, 0x3FFF), t.u32(1));
  Id baseHeight = t.iadd(field(2, 14, 0x3FFF), t.u32(1));
  Id basePitch = t.iadd(field(4, 13, 0x3FFF), t.u32(1));
  Id baseMip = field(3, 12, 0xF);
  Id lastMip = field(3, 16, 0xF);
  Id isUnorm = t.isZero(field(1, 26, 0xF));  // nfmt 0 = UNORM (else UINT passthrough)
  Id safeLastMip = t.umax(baseMip, lastMip);
  Id requestedMip = mipOp ? t.ldVg(vaddr + (da ? 3 : 2)) : t.u32(0);
  Id viewMip = t.umin(requestedMip, t.isub(safeLastMip, baseMip));
  Id physicalMip = t.iadd(baseMip, viewMip);
  Id width = csMax1(t, t.shr(baseWidth, physicalMip));
  Id height = csMax1(t, t.shr(baseHeight, physicalMip));
  Id linearGeneral = t.ieq(field(3, 20, 0x1F), t.u32(31));  // tiling_index 31
  Id pow2Pad = t.isNonZero(field(3, 25, 1));
  Id storedHeight = t.iselB(pow2Pad, csBitCeil(t, height), height);
  Id pitch = csLinearMipPitch(t, basePitch, storedHeight, physicalMip,
                              linearGeneral, pow2Pad);
  Id baseArray = field(5, 0, 0x1FFF);
  Id lastArray = field(5, 13, 0x1FFF);
  Id viewLayer = da ? t.ldVg(vaddr + 2) : t.u32(0);
  Id physicalLayer = t.iadd(baseArray, viewLayer);
  Id isArray = t.ieq(field(3, 28, 0xF), t.u32(13));  // SQ_RSRC_IMG_2D_ARRAY
  Id descriptorLayers = t.iadd(field(4, 0, 0x1FFF), t.u32(1));
  descriptorLayers = t.iselB(pow2Pad, csBitCeil(t, descriptorLayers), descriptorLayers);
  Id layers = t.iselB(isArray, descriptorLayers, t.u32(1));
  Id arrayOk = t.land(t.ult(baseArray, layers), t.uge(lastArray, baseArray));
  arrayOk = t.land(arrayOk, t.ule(viewLayer, t.isub(lastArray, baseArray)));
  arrayOk = t.land(arrayOk, t.ult(physicalLayer, layers));
  Id layerOk = t.m.emit(spv::Op::OpSelect, t.tBool,
                         {isArray, arrayOk, t.m.constBool(true)});
  Id valid = t.land(t.ult(x, width), t.ult(y, height));
  valid = t.land(valid, layerOk);
  valid = t.land(valid, t.uge(lastMip, baseMip));
  if (load) {
    uint32_t comp = 0;
    for (int i = 0; i < 4; i++)
      if (dmask & (1 << i)) t.stVg(vdata + comp++, t.u32(0));
  }
  Id accessBlk = t.m.newBlock(), mergeBlk = t.m.newBlock();
  t.m.selectionMerge(mergeBlk);
  t.m.branchConditional(valid, accessBlk, mergeBlk);
  t.m.openBlock(accessBlk);
  Id layer = t.iselB(isArray, physicalLayer, t.u32(0));
  Id mipOff = t.u32(0);
  for (uint32_t mip = 0; mip < 16; mip++) {
    Id level = t.u32(mip);
    Id levelHeight = csMax1(t, t.shr(baseHeight, level));
    Id levelStoredHeight = t.iselB(pow2Pad, csBitCeil(t, levelHeight), levelHeight);
    Id levelPitch = csLinearMipPitch(t, basePitch, levelStoredHeight, level,
                                     linearGeneral, pow2Pad);
    Id levelSize = t.imul(t.imul(levelPitch, levelStoredHeight), layers);
    mipOff = t.iadd(mipOff, t.iselB(t.ult(level, physicalMip), levelSize, t.u32(0)));
  }
  Id layerOff = t.imul(layer, t.imul(pitch, storedHeight));
  Id dwordIdx = t.iadd(mipOff, t.iadd(layerOff, t.iadd(t.imul(y, pitch), x)));
  if (load) {
    Id raw = csSsboLoad(t, sc, (uint32_t)b, dwordIdx);
    uint32_t comp = 0;
    for (int i = 0; i < 4; i++) {
      if (!(dmask & (1 << i))) continue;
      Id byte = t.band(t.shr(raw, t.u32(i * 8u)), t.u32(0xFF));
      Id normalized = t.fmul(t.m.emit(spv::Op::OpConvertUToF, t.tF, {byte}),
                             t.fconst(1.0f / 255.0f));
      t.stVg(vdata + comp++,
             t.iselB(isUnorm, t.m.bitcast(t.tU, normalized), byte));
    }
  } else {
    auto storeByte = [&](uint32_t reg) {
      Id value = t.ldVg(reg);
      Id normalized = t.m.extInst(t.tF, GLSLstd450FClamp,
                                   {t.m.bitcast(t.tF, value), t.fconst(0.0f), t.fconst(1.0f)});
      normalized = t.m.extInst(t.tF, GLSLstd450RoundEven,
                               {t.fmul(normalized, t.fconst(255.0f))});
      Id unorm = t.m.emit(spv::Op::OpConvertFToU, t.tU, {normalized});
      return t.band(t.iselB(isUnorm, unorm, value), t.u32(0xFF));
    };
    Id packed;
    if (dmask == 0xF) {
      packed = storeByte(vdata);
      packed = t.bor(packed, t.shl(storeByte(vdata + 1), t.u32(8)));
      packed = t.bor(packed, t.shl(storeByte(vdata + 2), t.u32(16)));
      packed = t.bor(packed, t.shl(storeByte(vdata + 3), t.u32(24)));
    } else {
      packed = csSsboLoad(t, sc, (uint32_t)b, dwordIdx);  // read-modify-write
      uint32_t comp = 0;
      for (int i = 0; i < 4; i++) {
        if (!(dmask & (1 << i))) continue;
        Id keep = t.band(packed, t.u32(~(0xFFu << (i * 8))));
        packed = t.bor(keep, t.shl(storeByte(vdata + comp++), t.u32(i * 8u)));
      }
    }
    csSsboStore(t, sc, (uint32_t)b, dwordIdx, packed);
  }
  t.m.branch(mergeBlk);
  t.m.openBlock(mergeBlk);
}

// Emit one non-terminator instruction (branches are handled by the CFG driver).
void emitInst(Tr &t, const Inst &in, StageCtx &sc) {
  uint32_t w = in.raw[0], w1 = in.raw[1];
  switch (in.enc) {
    case Enc::sop1: {
      uint32_t op = in.opcode, sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF;
      Id a = t.srcRaw(ssrc0, in.literal);
      Id ahi = t.srcRawHi(ssrc0, in.literal, false);
      switch (op) {
        case 0x03: t.stSg(sdst, a); break;                         // s_mov_b32
        case 0x04:                                                 // s_mov_b64
          t.stSg(sdst, a);
          t.stSg(sdst + 1, ahi);
          break;
        case 0x05:                                                 // s_cmov_b32: SCC ? src : dst
          t.stSg(sdst, t.iselNZ(t.ldScc(), a, t.ldSg(sdst)));
          break;
        case 0x06:                                                 // s_cmov_b64
          t.stSg(sdst, t.iselNZ(t.ldScc(), a, t.ldSg(sdst)));
          t.stSg(sdst + 1, t.iselNZ(t.ldScc(), ahi, t.ldSg(sdst + 1)));
          break;
        case 0x07: { Id r = t.bnot(a); t.stSg(sdst, r); t.stSccBool(t.isNonZero(r)); break; }  // s_not_b32
        case 0x08: {                                               // s_not_b64
          Id lo = t.bnot(a), hi = t.bnot(ahi);
          t.stSg(sdst, lo); t.stSg(sdst + 1, hi);
          t.stSccBool(t.isNonZero(t.bor(lo, hi)));
          break;
        }
        // s_wqm (whole quad mode): a lane's bit is set if any lane in its quad is.
        // Single-lane model -> identity; SCC = (result != 0).
        case 0x09: t.stSg(sdst, a); t.stSccBool(t.isNonZero(a)); break;  // s_wqm_b32
        case 0x0a:                                                       // s_wqm_b64
          t.stSg(sdst, a); t.stSg(sdst + 1, ahi);
          t.stSccBool(t.isNonZero(t.bor(a, ahi)));
          break;
        case 0x0b: t.stSg(sdst, t.bitrev(a)); break;               // s_brev_b32
        case 0x0d: { Id r = t.isub(t.u32(32), t.popcnt(a)); t.stSg(sdst, r); t.stSccBool(t.isNonZero(r)); break; }  // s_bcnt0_i32_b32
        case 0x0e: { Id r = t.isub(t.u32(64), t.iadd(t.popcnt(a), t.popcnt(ahi)));
                     t.stSg(sdst, r); t.stSccBool(t.isNonZero(r)); break; }  // s_bcnt0_i32_b64
        case 0x0f: { Id r = t.popcnt(a); t.stSg(sdst, r); t.stSccBool(t.isNonZero(r)); break; }  // s_bcnt1_i32_b32
        case 0x10: { Id r = t.iadd(t.popcnt(a), t.popcnt(ahi)); t.stSg(sdst, r); t.stSccBool(t.isNonZero(r)); break; }  // s_bcnt1_i32_b64
        case 0x11: t.stSg(sdst, t.m.extInst(t.tU, GLSLstd450FindILsb, {t.bnot(a)})); break;  // s_ff0_i32_b32
        case 0x13: t.stSg(sdst, t.m.extInst(t.tU, GLSLstd450FindILsb, {a})); break;          // s_ff1_i32_b32
        case 0x15: {  // s_flbit_i32_b32: leading-zero count from the MSB; -1 if src==0
          Id msb = t.m.extInst(t.tU, GLSLstd450FindUMsb, {a});
          t.stSg(sdst, t.iselB(t.isZero(a), t.u32(0xFFFFFFFFu), t.isub(t.u32(31), msb)));
          break;
        }
        case 0x17: {  // s_flbit_i32: leading-sign-bit count; -1 if src is 0 or -1
          Id smsb = t.m.bitcast(t.tU, t.m.extInst(t.tI, GLSLstd450FindSMsb, {t.m.bitcast(t.tI, a)}));
          t.stSg(sdst, t.iselB(t.m.emit(spv::Op::OpIEqual, t.tBool, {smsb, t.u32(0xFFFFFFFFu)}),
                               t.u32(0xFFFFFFFFu), t.isub(t.u32(31), smsb)));
          break;
        }
        // The main VS uses these to call/return from its fetch shader. Vertex
        // attributes are decoded from that fetch program and supplied as Vulkan
        // inputs, so no runtime jump remains in the translated shader.
        case 0x20: case 0x21: break;  // s_setpc_b64 / s_swappc_b64
        case 0x24: case 0x25: case 0x26: case 0x27: {  // s_{and,or,xor,andn2}_saveexec_b64
          Id oldExec = t.ldExec(), src = a, ne;
          if (op == 0x24) ne = t.band(oldExec, src);
          else if (op == 0x25) ne = t.bor(oldExec, src);
          else if (op == 0x26) ne = t.bxor(oldExec, src);
          else ne = t.band(oldExec, t.bnot(src));
          t.stSg(sdst, oldExec);
          t.stSg(sdst + 1, t.u32(0));
          t.stSg(126, ne);
          t.stSccBool(t.isNonZero(ne));
          break;
        }
        case 0x34: { Id r = t.m.bitcast(t.tU, t.m.extInst(t.tI, GLSLstd450SAbs, {t.m.bitcast(t.tI, a)}));
                     t.stSg(sdst, r); t.stSccBool(t.isNonZero(r)); break; }  // s_abs_i32
        default: warnUnsup("sop1", op); break;
      }
      break;
    }
    case Enc::sop2: {
      uint32_t op = in.opcode, sdst = (w >> 16) & 0x7F, s0f = w & 0xFF, s1f = (w >> 8) & 0xFF;
      Id a = t.srcRaw(s0f, in.literal), b = t.srcRaw(s1f, in.literal), r = 0, rhi = 0;
      Id ahi = t.srcRawHi(s0f, in.literal, op == 0x23);
      Id bhi = t.srcRawHi(s1f, in.literal, false);
      bool scc = false, wideScc = false;
      // Signed-overflow bit for add: (a^r) & (b^r), sign bit.
      auto sovf = [&](Id x, Id y, Id res) {
        return t.isNonZero(t.band(t.band(t.bxor(x, res), t.bxor(y, res)), t.u32(0x80000000u)));
      };
      auto shift64 = [&](uint32_t kind) {
        Id n = t.band(b, t.u32(63)), nlo = t.band(n, t.u32(31));
        Id ge32 = t.m.emit(spv::Op::OpUGreaterThanEqual, t.tBool, {n, t.u32(32)});
        Id zero = t.isZero(n);
        Id inv = t.band(t.isub(t.u32(32), nlo), t.u32(31));
        if (kind == 0) {  // logical left
          Id cross = t.iselB(zero, t.u32(0), t.shr(a, inv));
          Id hiSmall = t.bor(t.shl(ahi, nlo), cross);
          r = t.iselB(ge32, t.u32(0), t.shl(a, nlo));
          rhi = t.iselB(ge32, t.shl(a, nlo), hiSmall);
        } else {
          Id cross = t.iselB(zero, t.u32(0), t.shl(ahi, inv));
          Id loSmall = t.bor(t.shr(a, nlo), cross);
          Id hiSmall = kind == 1 ? t.shr(ahi, nlo) : t.sar(ahi, nlo);
          r = t.iselB(ge32, kind == 1 ? t.shr(ahi, nlo) : t.sar(ahi, nlo), loSmall);
          rhi = t.iselB(ge32, kind == 1 ? t.u32(0) : t.sar(ahi, t.u32(31)), hiSmall);
        }
        scc = true;
        wideScc = true;
      };
      switch (op) {
        case 0x00: {  // s_add_u32: SCC = unsigned carry-out
          Id p = t.m.emit(spv::Op::OpIAddCarry, t.pairU(), {a, b});
          r = t.m.compositeExtract(t.tU, p, 0);
          t.stSccBool(t.isNonZero(t.m.compositeExtract(t.tU, p, 1)));
          break;
        }
        case 0x01: {  // s_sub_u32: SCC = unsigned borrow
          Id p = t.m.emit(spv::Op::OpISubBorrow, t.pairU(), {a, b});
          r = t.m.compositeExtract(t.tU, p, 0);
          t.stSccBool(t.isNonZero(t.m.compositeExtract(t.tU, p, 1)));
          break;
        }
        case 0x02: r = t.iadd(a, b); t.stSccBool(sovf(a, b, r)); break;    // s_add_i32
        case 0x03:  // s_sub_i32: overflow = (a^b) & (a^r), sign bit
          r = t.isub(a, b);
          t.stSccBool(t.isNonZero(t.band(t.band(t.bxor(a, b), t.bxor(a, r)), t.u32(0x80000000u))));
          break;
        case 0x04: {  // s_addc_u32: a + b + SCC, SCC = carry-out
          Id p = t.m.emit(spv::Op::OpIAddCarry, t.pairU(), {a, b});
          Id p2 = t.m.emit(spv::Op::OpIAddCarry, t.pairU(),
                           {t.m.compositeExtract(t.tU, p, 0), t.band(t.ldScc(), t.u32(1))});
          r = t.m.compositeExtract(t.tU, p2, 0);
          t.stSccBool(t.isNonZero(t.bor(t.m.compositeExtract(t.tU, p, 1),
                                        t.m.compositeExtract(t.tU, p2, 1))));
          break;
        }
        case 0x05: {  // s_subb_u32: a - b - SCC, SCC = borrow-out
          Id p = t.m.emit(spv::Op::OpISubBorrow, t.pairU(), {a, b});
          Id p2 = t.m.emit(spv::Op::OpISubBorrow, t.pairU(),
                           {t.m.compositeExtract(t.tU, p, 0), t.band(t.ldScc(), t.u32(1))});
          r = t.m.compositeExtract(t.tU, p2, 0);
          t.stSccBool(t.isNonZero(t.bor(t.m.compositeExtract(t.tU, p, 1),
                                        t.m.compositeExtract(t.tU, p2, 1))));
          break;
        }
        case 0x06: r = t.smin(a, b); t.stSccBool(t.m.emit(spv::Op::OpSLessThan, t.tBool,
                        {t.m.bitcast(t.tI, a), t.m.bitcast(t.tI, b)})); break;  // s_min_i32
        case 0x07: r = t.umin(a, b); t.stSccBool(t.m.emit(spv::Op::OpULessThan, t.tBool, {a, b})); break;  // s_min_u32
        case 0x08: r = t.smax(a, b); t.stSccBool(t.m.emit(spv::Op::OpSGreaterThan, t.tBool,
                        {t.m.bitcast(t.tI, a), t.m.bitcast(t.tI, b)})); break;  // s_max_i32
        case 0x09: r = t.umax(a, b); t.stSccBool(t.m.emit(spv::Op::OpUGreaterThan, t.tBool, {a, b})); break;  // s_max_u32
        case 0x0a: r = t.iselNZ(t.ldScc(), a, b); break;                   // s_cselect_b32
        case 0x0b:                                                        // s_cselect_b64
          r = t.iselNZ(t.ldScc(), a, b);
          rhi = t.iselNZ(t.ldScc(), ahi, bhi);
          break;
        case 0x0e: r = t.band(a, b); scc = true; break;                    // s_and_b32
        case 0x0f: r = t.band(a, b); rhi = t.band(ahi, bhi); scc = wideScc = true; break;
        case 0x10: r = t.bor(a, b); scc = true; break;                     // s_or_b32
        case 0x11: r = t.bor(a, b); rhi = t.bor(ahi, bhi); scc = wideScc = true; break;
        case 0x12: r = t.bxor(a, b); scc = true; break;                    // s_xor_b32
        case 0x13: r = t.bxor(a, b); rhi = t.bxor(ahi, bhi); scc = wideScc = true; break;
        case 0x14: r = t.band(a, t.bnot(b)); scc = true; break;            // s_andn2_b32
        case 0x15: r = t.band(a, t.bnot(b)); rhi = t.band(ahi, t.bnot(bhi)); scc = wideScc = true; break;
        case 0x16: r = t.bor(a, t.bnot(b)); scc = true; break;             // s_orn2_b32
        case 0x17: r = t.bor(a, t.bnot(b)); rhi = t.bor(ahi, t.bnot(bhi)); scc = wideScc = true; break;
        case 0x18: r = t.bnot(t.band(a, b)); scc = true; break;            // s_nand_b32
        case 0x19: r = t.bnot(t.band(a, b)); rhi = t.bnot(t.band(ahi, bhi)); scc = wideScc = true; break;
        case 0x1a: r = t.bnot(t.bor(a, b)); scc = true; break;             // s_nor_b32
        case 0x1b: r = t.bnot(t.bor(a, b)); rhi = t.bnot(t.bor(ahi, bhi)); scc = wideScc = true; break;
        case 0x1c: r = t.bnot(t.bxor(a, b)); scc = true; break;            // s_xnor_b32
        case 0x1d: r = t.bnot(t.bxor(a, b)); rhi = t.bnot(t.bxor(ahi, bhi)); scc = wideScc = true; break;
        case 0x1e: r = t.shl(a, b); scc = true; break;                     // s_lshl_b32
        case 0x1f: shift64(0); break;                                      // s_lshl_b64
        case 0x20: r = t.shr(a, b); scc = true; break;                     // s_lshr_b32
        case 0x21: shift64(1); break;                                      // s_lshr_b64
        case 0x22: r = t.sar(a, b); scc = true; break;                     // s_ashr_i32
        case 0x23: shift64(2); break;                                      // s_ashr_i64
        case 0x24: {  // s_bfm_b32: mask = ((1<<width)-1) << offset
          Id width = t.band(a, t.u32(31)), off = t.band(b, t.u32(31));
          r = t.shl(t.isub(t.shl(t.u32(1), width), t.u32(1)), off);
          break;
        }
        case 0x26: r = t.imul(a, b); break;                               // s_mul_i32
        case 0x27: {  // s_bfe_u32: offset=b[4:0], width=b[22:16]
          Id off = t.band(b, t.u32(31)), width = t.band(t.shr(b, t.u32(16)), t.u32(0x7F));
          r = t.m.emit(spv::Op::OpBitFieldUExtract, t.tU, {a, off, width});
          scc = true;
          break;
        }
        case 0x28: {  // s_bfe_i32: signed
          Id off = t.band(b, t.u32(31)), width = t.band(t.shr(b, t.u32(16)), t.u32(0x7F));
          r = t.m.bitcast(t.tU, t.m.emit(spv::Op::OpBitFieldSExtract, t.tI,
                                         {t.m.bitcast(t.tI, a), off, width}));
          scc = true;
          break;
        }
        case 0x2c: {  // s_absdiff_i32: |a - b|
          r = t.m.bitcast(t.tU, t.m.extInst(t.tI, GLSLstd450SAbs, {t.m.bitcast(t.tI, t.isub(a, b))}));
          scc = true;
          break;
        }
        default: warnUnsup("sop2", op); r = a; break;
      }
      if (r) {
        t.stSg(sdst, r);
        if (rhi) t.stSg(sdst + 1, rhi);
        if (scc) t.stSccBool(t.isNonZero(wideScc ? t.bor(r, rhi) : r));
      }
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
      if (sc.isCs) { emitCsSmrd(t, in, sc); break; }
      emitCbufSmrd(t, in, sc.cbufBind);
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
      uint32_t op = in.opcode, vdst = w & 0xFF;
      bool vop3b = isVop3b(op);
      uint32_t sdst = vop3b ? ((w >> 8) & 0x7F) : 106;
      uint32_t abs = vop3b ? 0 : ((w >> 8) & 7);
      bool clamp = !vop3b && ((w >> 11) & 1);
      uint32_t s0 = w1 & 0x1FF, s1 = (w1 >> 9) & 0x1FF, s2 = (w1 >> 18) & 0x1FF, neg = (w1 >> 29) & 7;
      emitVop3(t, op, vdst, t.srcF(s0, in.literal, neg & 1, abs & 1),
                 t.srcF(s1, in.literal, neg & 2, abs & 2),
                 t.srcF(s2, in.literal, neg & 4, abs & 4),
                 t.srcRawHi(s2, in.literal, op == 0x177), sdst, clamp);
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
      if (op == 1 || (op == 2 && (w & 0xFF) == 2)) {
        // Vulkan provides the completed interpolation directly. P2 therefore reads
        // the final value, while MOV reads the selected parameter input instead of
        // leaving the destination at its zero-initialized value.
        Id v = psInputVar(t, sc, attr);
        Id pInF = t.m.typePointer(spv::StorageClass::Input, t.tF);
        t.stVgF(vdst, t.m.load(t.tF, t.m.accessChain(pInF, v, {t.m.constU32(chan)})));
      }
      break;
    }
    case Enc::mubuf:
    case Enc::mtbuf:
      if (sc.isCs) {
        if (in.enc == Enc::mubuf) emitCsMubuf(t, in, sc);
        else sc.csUnsupported = true;  // tbuffer (typed) not modelled
      }
      break;
    case Enc::ds:
      if (sc.isCs) sc.csUnsupported = true;  // LDS/GDS not modelled
      break;
    case Enc::mimg: {
      if (sc.isCs) { emitCsMimg(t, in, sc); break; }
      if (!sc.isPs) break;
      emitMimg(t, in.opcode, w, w1, *sc.r);
      break;
    }
    case Enc::exp: {
      if (sc.isCs) { sc.csUnsupported = true; break; }  // no exports in compute
      uint32_t en = w & 0xF, target = (w >> 4) & 0x3F, compr = (w >> 10) & 1;
      uint32_t v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF, (w1 >> 24) & 0xFF};
      if (sc.isPs) {
        if (target <= 7 && en) {
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
          t.m.store(psColorOut(t, sc, target), col);
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
                  const std::unordered_set<uint32_t> &flatAttrs, Tr &t) {
  uint64_t fetch = (static_cast<uint64_t>(vsUserData[1] & 0xFFFF) << 32) | vsUserData[0];
  auto attrs = parseFetch(fetch);
  t.initTypes();

  // Inputs + position/param outputs.
  std::vector<Id> iface;
  Id posOut = t.m.variable(t.m.typePointer(spv::StorageClass::Output, t.tV4),
                           spv::StorageClass::Output);
  t.m.decorate(posOut, spv::Decoration::BuiltIn, {(uint32_t)spv::BuiltIn::Position});
  iface.push_back(posOut);

  Id main = t.m.beginFunction(t.tVoid, t.tFn);

  if (attrs.empty()) {
    // A procedural VS has no fetch shader. On GFX6-8, VertexID enters in v0;
    // InstanceID/StepRate0 enters in v1 and raw InstanceID is available in v3.
    // Seed those ABI inputs from Vulkan's draw built-ins for fullscreen RECTLISTs.
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/be00f53d4d50b87a87f83e8fa243b77e614eb0b8/src/gallium/drivers/radeonsi/gfx/si_state_shaders.cpp#L269-307
    Id pInU = t.m.typePointer(spv::StorageClass::Input, t.tU);
    Id vertexIndex = t.m.variable(pInU, spv::StorageClass::Input);
    Id instanceIndex = t.m.variable(pInU, spv::StorageClass::Input);
    t.m.decorate(vertexIndex, spv::Decoration::BuiltIn,
                 {(uint32_t)spv::BuiltIn::VertexIndex});
    t.m.decorate(instanceIndex, spv::Decoration::BuiltIn,
                 {(uint32_t)spv::BuiltIn::InstanceIndex});
    iface.push_back(vertexIndex);
    iface.push_back(instanceIndex);
    t.stVg(0, t.m.load(t.tU, vertexIndex));
    Id instance = t.m.load(t.tU, instanceIndex);
    t.stVg(1, instance);
    t.stVg(3, instance);
  }

  // Seed destination VGPRs from fetched vertex attributes when present.
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

  auto insts = decodeShader(vsCode, 4096);
  std::unordered_map<uint32_t, uint32_t> cbufBindings;
  if (!planCbufs(insts, 0, r.vsCbufs, cbufBindings)) return false;

  StageCtx sc;
  sc.r = &r;
  sc.iface = &iface;
  sc.posOut = posOut;
  sc.cbufBind = cbufBindings;
  sc.flatAttrs = &flatAttrs;

  // Branchy shaders take the CFG (while-switch) path so their control flow (the GCN
  // alpha-test/discard idiom, conditional shading) is honoured; single-basic-block
  // shaders emit the same instruction stream straight-line. DELTA_GPU_SPIRV_CFG
  // forces the CFG path even for single-BB shaders (machinery test).
  static const bool forceCfg = std::getenv("DELTA_GPU_SPIRV_CFG") != nullptr;
  if (forceCfg || hasControlFlow(insts)) {
    t.seedExec();
    emitCFG(t, insts, sc);
  } else {
    for (auto &in : insts) {
      if (in.enc == Enc::sopp && in.opcode == 1) break;  // s_endpgm
      emitInst(t, in, sc);
    }
  }
  r.numParams = sc.maxParam;

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
bool translatePs(const uint32_t *psCode, Recompiled &r,
                  const std::unordered_set<uint32_t> &flatAttrs, Tr &t) {
  auto insts = decodeShader(psCode, 4096);
  std::unordered_map<uint32_t, uint32_t> cbufBindings;
  if (!planCbufs(insts, r.vsCbufs.size(), r.psCbufs, cbufBindings)) return false;

  // Color outputs (psColorOut) are declared lazily per MRT target (location ==
  // target index), so a shader exporting to MRT0..7 produces a multi-attachment
  // fragment output. Most 2D titles only export MRT0 -> a single location-0 output.
  // PS inputs (interpolants, psInputVar) are likewise declared as they are read.
  std::vector<Id> iface;
  StageCtx sc;
  sc.isPs = true;
  sc.r = &r;
  sc.iface = &iface;
  sc.cbufBind = cbufBindings;
  sc.flatAttrs = &flatAttrs;

  Id main = t.m.beginFunction(t.tVoid, t.tFn);

  static const bool forceCfg = std::getenv("DELTA_GPU_SPIRV_CFG") != nullptr;
  if (forceCfg || hasControlFlow(insts)) {  // branchy PS: honour control flow / discard
    // Default MRT0 to transparent so a fragment that never reaches an export leaves
    // a defined value (not garbage) even if the discard lowering is bypassed.
    t.m.store(psColorOut(t, sc, 0), t.m.constComposite(t.tV4,
              {t.fconst(0.f), t.fconst(0.f), t.fconst(0.f), t.fconst(0.f)}));
    sc.colorWrittenVar = t.m.variable(t.pPrivU, spv::StorageClass::Private, t.m.constNull(t.tU));
    t.seedExec();
    emitCFG(t, insts, sc);
  } else {
    for (auto &in : insts) {
      if (in.enc == Enc::sopp && in.opcode == 1) break;  // s_endpgm
      emitInst(t, in, sc);
    }
  }

  if (!sc.wroteColor) {
    // Shader has no color export at all: opaque white fallback.
    t.m.store(psColorOut(t, sc, 0), t.m.constComposite(t.tV4,
              {t.fconst(1.f), t.fconst(1.f), t.fconst(1.f), t.fconst(1.f)}));
  } else if (sc.colorWrittenVar) {
    // GCN alpha-test/kill idiom (CFG path only): control flow branches over the color
    // export for failing fragments (e.g. s_cmp + s_cbranch_scc0 -> s_endpgm). Discard
    // those (OpKill) instead of leaving the output undefined.
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
  }

  t.m.returnVoid();
  t.m.endFunction();
  t.m.entryPoint(spv::ExecutionModel::Fragment, main, "main", iface);
  t.m.execMode(main, spv::ExecutionMode::OriginUpperLeft);
  return true;
}

// GCN permits depth-only rasterization with SPI_SHADER_PGM_LO_PS=0. Vulkan still
// requires a fragment stage for the ordinary graphics pipeline used here, so emit
// an empty stage: fixed-function depth testing/writes run, while no color location
// is declared or written.
bool translateDepthOnlyPs(Tr &t) {
  Id main = t.m.beginFunction(t.tVoid, t.tFn);
  t.m.returnVoid();
  t.m.endFunction();
  t.m.entryPoint(spv::ExecutionModel::Fragment, main, "main", {});
  t.m.execMode(main, spv::ExecutionMode::OriginUpperLeft);
  return true;
}

// ---- CS ---------------------------------------------------------------------
// Scan the CS for the memory resources (source/dest buffers/images) it touches,
// keyed by the descriptor SGPR. Fills r.resources + `bind` (baseSgpr -> binding).
// Returns false if a memory op the compute backend can't model is present.
bool planCsResources(const std::vector<Inst> &insts, RecompiledCs &r,
                     std::unordered_map<uint32_t, uint32_t> &bind) {
  auto resource = [&](uint32_t baseSgpr, uint8_t kind, bool written,
                      uint32_t minBytes) -> bool {
    auto it = bind.find(baseSgpr);
    if (it != bind.end()) {
      CsResource &res = r.resources[it->second];
      if (res.kind != kind) return false;  // same SGPR as both buffer and image
      res.written = res.written || written;
      if (minBytes > res.minBytes) res.minBytes = minBytes;
      return true;
    }
    uint32_t idx = (uint32_t)r.resources.size();
    if (idx >= 8) return false;  // cap the storage-buffer binding count
    bind[baseSgpr] = idx;
    r.resources.push_back({baseSgpr, idx, kind, written, minBytes});
    return true;
  };
  for (auto &in : insts) {
    uint32_t w = in.raw[0], w1 = in.raw[1];
    switch (in.enc) {
      case Enc::smrd: {
        uint32_t op = in.opcode, sbase = (w >> 9) & 0x3F;
        bool imm = (w >> 8) & 1; uint32_t off = w & 0xFF;
        if (op > 0x0c) return false;  // beyond s_buffer_load_dwordx16
        uint32_t n = op < 0x08 ? (1u << op)
                   : op == 0x08 ? 1 : op == 0x09 ? 2 : op == 0x0a ? 4 : op == 0x0b ? 8 : 16;
        uint32_t bytes = imm ? (off + n) * 4 : 0;
        if (!resource(sbase * 2, 0, false, bytes)) return false;
        break;
      }
      case Enc::mubuf: {
        uint32_t op = (w >> 18) & 0x7F, srsrc = ((w1 >> 16) & 0x1F) * 4;
        if (op >= 0x18) return false;  // buffer store: not handled
        if (!resource(srsrc, 0, false, 0)) return false;
        break;
      }
      case Enc::mimg: {
        uint32_t op = (w >> 18) & 0x7F, srsrc = ((w1 >> 16) & 0x1F) * 4;
        bool r128 = (w & 0x8000) != 0;
        bool store = (op == 0x08 || op == 0x09);
        bool load = (op == 0x00 || op == 0x01);
        if (!store && !load || r128 || srsrc + 7 >= 16) return false;
        if (!resource(srsrc, 1, store, 0)) return false;
        break;
      }
      case Enc::mtbuf:
      case Enc::ds:
        return false;  // typed buffer / LDS not modelled
      default:
        break;
    }
  }
  return true;
}

// Translate a CS to a GLCompute SPIR-V module. Models guest memory as per-resource
// storage buffers (see the compute mem-op emitters); user data is passed as push
// constants and seeds the register file; thread ids come from the builtins.
bool translateCs(const uint32_t *csCode, uint32_t ntx, uint32_t nty, uint32_t ntz,
                 uint32_t userSgpr, uint32_t tgidEnable, RecompiledCs &r, Tr &t) {
  auto insts = decode(csCode, 2048);
  if (insts.empty()) return false;
  std::unordered_map<uint32_t, uint32_t> bind;
  if (!planCsResources(insts, r, bind) || r.resources.empty()) return false;

  t.initTypes();
  // Storage buffers: Buf { uint data[]; } at set 0, binding = resource index.
  Id tRun = t.m.typeRuntimeArray(t.tU);
  t.m.decorate(tRun, spv::Decoration::ArrayStride, {4});
  Id tBuf = t.m.typeStruct({tRun});
  t.m.decorate(tBuf, spv::Decoration::Block);
  t.m.memberDecorate(tBuf, 0, spv::Decoration::Offset, {0});
  Id pBuf = t.m.typePointer(spv::StorageClass::StorageBuffer, tBuf);
  std::vector<Id> ssbo(r.resources.size());
  for (auto &res : r.resources) {
    Id v = t.m.variable(pBuf, spv::StorageClass::StorageBuffer);
    t.m.decorate(v, spv::Decoration::DescriptorSet, {0});
    t.m.decorate(v, spv::Decoration::Binding, {res.binding});
    ssbo[res.binding] = v;
  }

  // Push constant: the 16 COMPUTE_USER_DATA dwords seed s0.. (descriptors + params).
  Id tUArr16 = t.m.typeArray(t.tU, 16);
  t.m.decorate(tUArr16, spv::Decoration::ArrayStride, {4});
  Id tPc = t.m.typeStruct({tUArr16});
  t.m.decorate(tPc, spv::Decoration::Block);
  t.m.memberDecorate(tPc, 0, spv::Decoration::Offset, {0});
  Id pPc = t.m.typePointer(spv::StorageClass::PushConstant, tPc);
  Id pcVar = t.m.variable(pPc, spv::StorageClass::PushConstant);

  // Builtins: gl_LocalInvocationID (-> v0..v2) and gl_WorkGroupID (-> s after user data).
  Id tUV3 = t.m.typeVec(t.tU, 3);
  Id pInV3 = t.m.typePointer(spv::StorageClass::Input, tUV3);
  Id lid = t.m.variable(pInV3, spv::StorageClass::Input);
  t.m.decorate(lid, spv::Decoration::BuiltIn, {(uint32_t)spv::BuiltIn::LocalInvocationId});
  Id wid = t.m.variable(pInV3, spv::StorageClass::Input);
  t.m.decorate(wid, spv::Decoration::BuiltIn, {(uint32_t)spv::BuiltIn::WorkgroupId});
  std::vector<Id> iface{lid, wid};

  Id main = t.m.beginFunction(t.tVoid, t.tFn);
  Id pcU = t.m.typePointer(spv::StorageClass::PushConstant, t.tU);
  for (uint32_t i = 0; i < 16; i++)  // user data -> s0..s15
    t.stSg(i, t.m.load(t.tU, t.m.accessChain(pcU, pcVar, {t.m.constU32(0), t.m.constU32(i)})));
  Id pInU = t.m.typePointer(spv::StorageClass::Input, t.tU);
  auto widComp = [&](uint32_t c) { return t.m.load(t.tU, t.m.accessChain(pInU, wid, {t.m.constU32(c)})); };
  uint32_t sg = userSgpr;  // workgroup id -> s[userSgpr..] per tgid_enable
  if ((tgidEnable & 1) && sg < 106) t.stSg(sg++, widComp(0));
  if ((tgidEnable & 2) && sg < 106) t.stSg(sg++, widComp(1));
  if ((tgidEnable & 4) && sg < 106) t.stSg(sg++, widComp(2));
  for (uint32_t c = 0; c < 3; c++)  // local invocation id (tidig) -> v0..v2
    t.stVg(c, t.m.load(t.tU, t.m.accessChain(pInU, lid, {t.m.constU32(c)})));
  t.seedExec();

  StageCtx sc; sc.isCs = true; sc.csBind = bind; sc.csSsbo = ssbo;
  emitCFG(t, insts, sc);
  if (sc.csUnsupported) return false;
  t.m.returnVoid();
  t.m.endFunction();
  t.m.entryPoint(spv::ExecutionModel::GLCompute, main, "main", iface);
  t.m.execMode(main, spv::ExecutionMode::LocalSize,
               {ntx ? ntx : 1, nty ? nty : 1, ntz ? ntz : 1});
  r.localSize[0] = ntx ? ntx : 1;
  r.localSize[1] = nty ? nty : 1;
  r.localSize[2] = ntz ? ntz : 1;
  return true;
}

// RECTLIST consumes three post-VS corners and rasterizes the fourth corner as a
// second triangle. Vulkan has no matching input topology, so insert a geometry
// stage that performs the fixed-function expansion without making assumptions
// about the guest VS or its source vertex format.
std::vector<uint32_t> emitRectListGeometry(
    uint32_t numParams, const std::unordered_set<uint32_t> &flatAttrs) {
  S::Module m;
  m.capability(spv::Capability::Geometry);
  Id tVoid = m.typeVoid(), tF = m.typeFloat(32), tV4 = m.typeVec(tF, 4);
  Id tFn = m.typeFunction(tVoid);
  Id tInV4 = m.typeArray(tV4, 3);
  Id pInV4 = m.typePointer(spv::StorageClass::Input, tV4);
  Id pOutV4 = m.typePointer(spv::StorageClass::Output, tV4);

  std::vector<Id> iface;
  Id inPos = m.variable(m.typePointer(spv::StorageClass::Input, tInV4),
                        spv::StorageClass::Input);
  Id outPos = m.variable(pOutV4, spv::StorageClass::Output);
  m.decorate(inPos, spv::Decoration::BuiltIn,
             {(uint32_t)spv::BuiltIn::Position});
  m.decorate(outPos, spv::Decoration::BuiltIn,
             {(uint32_t)spv::BuiltIn::Position});
  iface.push_back(inPos);
  iface.push_back(outPos);

  std::vector<Id> inputs(numParams), outputs(numParams);
  for (uint32_t p = 0; p < numParams; p++) {
    inputs[p] = m.variable(m.typePointer(spv::StorageClass::Input, tInV4),
                           spv::StorageClass::Input);
    outputs[p] = m.variable(pOutV4, spv::StorageClass::Output);
    m.decorate(inputs[p], spv::Decoration::Location, {p});
    m.decorate(outputs[p], spv::Decoration::Location, {p});
    if (flatAttrs.count(p)) {
      m.decorate(inputs[p], spv::Decoration::Flat);
      m.decorate(outputs[p], spv::Decoration::Flat);
    }
    iface.push_back(inputs[p]);
    iface.push_back(outputs[p]);
  }

  Id main = m.beginFunction(tVoid, tFn);
  auto loadInput = [&](Id input, uint32_t vertex) {
    return m.load(tV4, m.accessChain(pInV4, input, {m.constU32(vertex)}));
  };
  auto fourthCorner = [&](Id input) {
    Id v0 = loadInput(input, 0), v1 = loadInput(input, 1);
    Id v2 = loadInput(input, 2);
    return m.emit(spv::Op::OpFSub, tV4,
                  {m.emit(spv::Op::OpFAdd, tV4, {v1, v2}), v0});
  };
  Id pos3 = fourthCorner(inPos);
  std::vector<Id> param3(numParams);
  for (uint32_t p = 0; p < numParams; p++)
    if (!flatAttrs.count(p)) param3[p] = fourthCorner(inputs[p]);

  for (uint32_t vertex = 0; vertex < 4; vertex++) {
    m.store(outPos, vertex < 3 ? loadInput(inPos, vertex) : pos3);
    for (uint32_t p = 0; p < numParams; p++) {
      Id value = flatAttrs.count(p) ? loadInput(inputs[p], 0)
                 : vertex < 3 ? loadInput(inputs[p], vertex) : param3[p];
      m.store(outputs[p], value);
    }
    m.emitVoid(spv::Op::OpEmitVertex, {});
  }
  m.emitVoid(spv::Op::OpEndPrimitive, {});
  m.returnVoid();
  m.endFunction();
  m.entryPoint(spv::ExecutionModel::Geometry, main, "main", iface);
  m.execMode(main, spv::ExecutionMode::Triangles);
  m.execMode(main, spv::ExecutionMode::OutputTriangleStrip);
  m.execMode(main, spv::ExecutionMode::OutputVertices, {4});
  return m.assemble();
}

}  // namespace

bool recompileSpirv(const uint32_t *vsCode, const uint32_t *psCode,
                     const uint32_t *vsUserData, const uint32_t *psUserData,
                     Recompiled &r) {
  if (!vsCode || !vsUserData || !psUserData) return false;
  // One-shot disassembly (DELTA_GPU_SHDIS): for the first branchy shader, list each
  // instruction's encoding + opcode so we can see which ops the CFG path must handle.
  if (std::getenv("DELTA_GPU_SHDIS")) {
    static int dn = 0;
    auto dump = [&](const char *tag, const uint32_t *code) {
      auto ins = decodeShader(code, 4096);
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
  // V_INTERP_MOV P0 reads a per-primitive parameter rather than a smoothly
  // interpolated value. Represent those locations as flat varyings in both stages.
  std::unordered_set<uint32_t> flatAttrs;
  if (psCode)
    for (const Inst &in : decodeShader(psCode, 4096))
      if (in.enc == Enc::vintrp && in.opcode == 2 && (in.raw[0] & 0xFF) == 2)
        flatAttrs.insert((in.raw[0] >> 10) & 0x3F);

  // VS and PS are separate SPIR-V modules (separate Tr/Module each).
  Tr tv;
  if (!translateVs(vsCode, vsUserData, r, flatAttrs, tv)) {
    if (g_dbg) std::fprintf(stderr, "[gcnspv] VS translation rejected @%p\n",
                            static_cast<const void *>(vsCode));
    return false;
  }
  Tr tp;
  tp.initTypes();
  if (psCode ? !translatePs(psCode, r, flatAttrs, tp) : !translateDepthOnlyPs(tp)) {
    if (g_dbg) std::fprintf(stderr, "[gcnspv] PS translation rejected @%p\n",
                            static_cast<const void *>(psCode));
    return false;
  }

  auto vs = tv.m.assemble();
  auto gs = emitRectListGeometry(r.numParams, flatAttrs);
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
  if (!spirv::validate(gs, &err)) {
    if (g_dbg) std::fprintf(stderr, "[gcnspv] RECTLIST GS invalid: %s\n", err.c_str());
    return false;
  }
  // DELTA_GPU_SPIRV_NOOPT: skip the optimize pass (use the naive memory-backed
  // register SPIR-V). Diagnostic: isolates an emission bug from a spirv-opt
  // mis-promotion (the naive form keeps the register file in memory, always correct).
  static const bool noOpt = std::getenv("DELTA_GPU_SPIRV_NOOPT") != nullptr;
  r.vsSpirv = noOpt ? vs : spirv::optimize(vs);
  r.gsSpirv = noOpt ? gs : spirv::optimize(gs);
  r.fsSpirv = noOpt ? ps : spirv::optimize(ps);
  r.ok = !r.vsSpirv.empty() && !r.gsSpirv.empty() && !r.fsSpirv.empty();
  // Tally (DELTA_GPU_SPIRV): how many shaders the direct SPIR-V backend accepted vs
  // had to decline (-> GLSL fallback), and how many used the CFG path. Confirms the
  // backend is actually in use rather than silently falling back.
  static const bool tally = std::getenv("DELTA_GPU_SPIRV") != nullptr;
  if (tally) {
    static int okN = 0, cfN = 0; static int logged = 0;
    if (r.ok) okN++;
    if (hasControlFlow(decodeShader(vsCode, 4096)) || hasControlFlow(decodeShader(psCode, 4096))) cfN++;
    if (logged < 12) { logged++;
      std::fprintf(stderr, "[gcnspv] recompiled ok=%d (cfg-shaders=%d) this=%s\n", okN, cfN,
                   r.ok ? "spirv" : "FALLBACK"); }
  }
  return r.ok;
}

bool recompileComputeSpirv(const uint32_t *csCode, uint32_t ntx, uint32_t nty,
                           uint32_t ntz, uint32_t userSgpr, uint32_t tgidEnable,
                           RecompiledCs &r) {
  if (!csCode) return false;
  Tr t;
  RecompiledCs tmp;  // build into a temp so a mid-emit failure leaves r untouched
  if (!translateCs(csCode, ntx, nty, ntz, userSgpr, tgidEnable, tmp, t)) return false;
  auto spv = t.m.assemble();
  std::string err;
  if (!spirv::validate(spv, &err)) {
    if (g_dbg) std::fprintf(stderr, "[gcnspv] CS invalid: %s\n", err.c_str());
    return false;
  }
  static const bool noOpt = std::getenv("DELTA_GPU_SPIRV_NOOPT") != nullptr;
  tmp.spirv = noOpt ? spv : spirv::optimize(spv);
  if (tmp.spirv.empty()) return false;
  tmp.ok = true;
  r = std::move(tmp);
  return true;
}

}  // namespace gpu::gcn

#endif  // DELTA_HAVE_SPIRV_BACKEND
