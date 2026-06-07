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

#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include <spirv/unified1/GLSL.std.450.h>

namespace gpu::gcn {
namespace {

namespace S = ::gpu::gcn::spirv;
using Id = S::Id;

bool inGuest(uint64_t a) { return a >= 0x1000000000ull && a < 0x20000000000ull; }
const bool g_dbg = std::getenv("DELTA_GPU_SHTRACE") != nullptr;

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
  Id pcVar = 0;          // push-constant block (cbuffer)
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
  }

  // Declare the push-constant cbuffer: PC { uvec4 data[8]; } (matches the GLSL
  // backend's layout so the renderer's push range is identical).
  void ensurePc() {
    if (havePc) return;
    havePc = true;
    Id tUV4 = m.typeVec(tU, 4);
    Id arr = m.typeArray(tUV4, 8);
    m.decorate(arr, spv::Decoration::ArrayStride, {16});
    Id st = m.typeStruct({arr});
    m.decorate(st, spv::Decoration::Block);
    m.memberDecorate(st, 0, spv::Decoration::Offset, {0});
    pcVar = m.variable(m.typePointer(spv::StorageClass::PushConstant, st),
                       spv::StorageClass::PushConstant);
  }
  // Read cbuffer dword k (== uvec4 data[k>>2][k&3]) as a uint Id.
  Id pcDword(uint32_t k) {
    ensurePc();
    Id pPushU = m.typePointer(spv::StorageClass::PushConstant, tU);
    Id ch = m.accessChain(pPushU, pcVar,
                          {m.constU32(0), m.constU32(k >> 2), m.constU32(k & 3)});
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
void emitVop2(Tr &t, uint32_t op, uint32_t vdst, Id s0, Id s1);

void emitVop1(Tr &t, uint32_t op, uint32_t vdst, Id s0) {
  auto setF = [&](Id f) { t.stVgF(vdst, f); };
  auto setU = [&](Id u) { t.stVg(vdst, u); };
  switch (op) {
    case 0x01: setU(t.m.bitcast(t.tU, s0)); break;                       // mov_b32
    case 0x05: setF(t.m.emit(spv::Op::OpConvertSToF, t.tF, {t.m.bitcast(t.tI, s0)})); break;  // f32_i32
    case 0x06: setF(t.m.emit(spv::Op::OpConvertUToF, t.tF, {t.m.bitcast(t.tU, s0)})); break;  // f32_u32
    case 0x07: setU(t.m.emit(spv::Op::OpConvertFToU, t.tU, {s0})); break;  // u32_f32
    case 0x08: setU(t.m.bitcast(t.tU, t.m.emit(spv::Op::OpConvertFToS, t.tI, {s0}))); break;  // i32_f32
    case 0x0d: setU(t.m.bitcast(t.tU, t.m.emit(spv::Op::OpConvertFToS, t.tI,
                  {t.ext1(GLSLstd450Floor, t.fadd(s0, t.fconst(0.5f)))}))); break;  // rpi
    case 0x0e: setU(t.m.bitcast(t.tU, t.m.emit(spv::Op::OpConvertFToS, t.tI,
                  {t.ext1(GLSLstd450Floor, s0)}))); break;               // flr
    case 0x21: setF(t.ext1(GLSLstd450Fract, s0)); break;
    case 0x22: setF(t.ext1(GLSLstd450Trunc, s0)); break;
    case 0x23: setF(t.ext1(GLSLstd450Ceil, s0)); break;
    case 0x24: setF(t.ext1(GLSLstd450RoundEven, s0)); break;
    case 0x25: setF(t.ext1(GLSLstd450Floor, s0)); break;
    case 0x2a: setF(t.ext1(GLSLstd450Exp2, s0)); break;
    case 0x2c: setF(t.ext1(GLSLstd450Log2, s0)); break;
    case 0x2d: case 0x2e: setF(t.fdiv(t.fconst(1.0f), s0)); break;       // rcp
    case 0x2f: setF(t.ext1(GLSLstd450InverseSqrt, s0)); break;           // rsq
    case 0x33: setF(t.ext1(GLSLstd450Sqrt, s0)); break;
    case 0x35: setF(t.ext1(GLSLstd450Sin, s0)); break;
    case 0x36: setF(t.ext1(GLSLstd450Cos, s0)); break;
    default: setU(t.m.bitcast(t.tU, s0)); break;                         // mov fallback
  }
}

void emitVop2(Tr &t, uint32_t op, uint32_t vdst, Id s0, Id s1) {
  auto setF = [&](Id f) { t.stVgF(vdst, f); };
  auto setU = [&](Id u) { t.stVg(vdst, u); };
  Id u0 = t.m.bitcast(t.tU, s0), u1 = t.m.bitcast(t.tU, s1);
  switch (op) {
    case 0x00: {  // cndmask: VCC ? s1 : s0 (VCC stored as raw 1u/0u by VOPC)
      Id cond = t.m.emit(spv::Op::OpINotEqual, t.tBool, {t.ldSg(106), t.m.constU32(0)});
      setF(t.m.emit(spv::Op::OpSelect, t.tF, {cond, s1, s0}));
      break;
    }
    case 0x01: case 0x02: case 0x03: setF(t.fadd(s0, s1)); break;
    case 0x04: setF(t.fsub(s0, s1)); break;
    case 0x05: setF(t.fsub(s1, s0)); break;
    case 0x06: setF(t.fmul(s0, s1)); break;
    case 0x08: setF(t.fmul(s0, s1)); break;
    case 0x0a: setF(t.ext2(GLSLstd450FMin, s0, s1)); break;
    case 0x0b: setF(t.ext2(GLSLstd450FMax, s0, s1)); break;
    case 0x1f: setF(t.fadd(t.fmul(s0, s1), t.ldVgF(vdst))); break;       // mac_f32
    case 0x25: setU(t.m.emit(spv::Op::OpBitwiseAnd, t.tU, {u0, u1})); break;
    case 0x26: setU(t.m.emit(spv::Op::OpBitwiseOr, t.tU, {u0, u1})); break;
    case 0x27: setU(t.m.emit(spv::Op::OpBitwiseXor, t.tU, {u0, u1})); break;
    case 0x2f: setU(t.m.extInst(t.tU, GLSLstd450PackHalf2x16,
                  {t.m.compositeConstruct(t.tV2, {s0, s1})})); break;    // cvt_pkrtz
    default: setF(t.fmul(s0, s1)); break;
  }
}

// VOPC: vector compare -> VCC (sgpr 106) as raw 1u/0u. The low nibble selects the
// predicate (1=lt 2=eq 3=le 4=gt 5=ne 6=ge) for all of f32 (op 0x00-0x1f, incl. the
// cmpx EXEC-writing variants which we treat the same), i32 (0x80-0x9f) and u32
// (0xc0-0xdf). s0f/s1f are the float operands, s0u/s1u the raw uint operands.
void emitVopc(Tr &t, uint32_t op, Id s0f, Id s1f, Id s0u, Id s1u) {
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
    t.stSg(106, t.m.emit(spv::Op::OpSelect, t.tU, {cond, t.m.constU32(1), t.m.constU32(0)}));
}

void emitVop3(Tr &t, uint32_t op, uint32_t vdst, Id s0, Id s1, Id s2) {
  if (op >= 0x100 && op < 0x140) { emitVop2(t, op - 0x100, vdst, s0, s1); return; }
  if (op >= 0x180 && op < 0x200) { emitVop1(t, op - 0x180, vdst, s0); return; }
  auto setF = [&](Id f) { t.stVgF(vdst, f); };
  switch (op) {
    case 0x141: case 0x14b: case 0x143: setF(t.fadd(t.fmul(s0, s1), s2)); break;  // mad/fma
    case 0x151: setF(t.ext2(GLSLstd450FMin, t.ext2(GLSLstd450FMin, s0, s1), s2)); break;
    case 0x154: setF(t.ext2(GLSLstd450FMax, t.ext2(GLSLstd450FMax, s0, s1), s2)); break;
    case 0x157: {  // med3 = clamp(s2, min(s0,s1), max(s0,s1))
      Id lo = t.ext2(GLSLstd450FMin, s0, s1), hi = t.ext2(GLSLstd450FMax, s0, s1);
      setF(t.m.extInst(t.tF, GLSLstd450FClamp, {s2, lo, hi}));
      break;
    }
    default: setF(s0); break;
  }
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
    for (uint32_t c = 0; c < a.numComps; c++) {
      Id comp = a.numComps == 1 ? val : t.m.compositeExtract(t.tF, val, c);
      t.stVgF(a.destVgpr + c, comp);
    }
    r.attrs.push_back({a.semantic, a.numComps, a.tableSgpr, a.dwordOff});
  }

  auto insts = decode(vsCode, 4096);
  uint32_t maxParam = 0;
  std::unordered_map<uint32_t, Id> paramOuts;  // param index -> Output var
  bool haveCbuf = false;

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
        emitVop2(t, op, vdst, t.srcF(src0, in.literal), t.srcF(256 + vsrc1, in.literal));
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
  Id colorOut = t.m.variable(t.m.typePointer(spv::StorageClass::Output, t.tV4),
                             spv::StorageClass::Output);
  t.m.decorate(colorOut, spv::Decoration::Location, {0});
  iface.push_back(colorOut);

  // Sampled-image type for any texture.
  Id imgTy = t.m.typeImage(t.tF, spv::Dim::Dim2D, 0, 0, 0, 1, spv::ImageFormat::Unknown);
  Id sampImgTy = t.m.typeSampledImage(imgTy);
  Id pSampImg = t.m.typePointer(spv::StorageClass::UniformConstant, sampImgTy);

  // PS inputs (interpolants) are declared lazily as they are read.
  std::unordered_map<uint32_t, Id> inVars;  // attr index -> Input vec4 var
  uint32_t maxIn = 0;
  std::vector<Id> texVars;
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
        emitVop2(t, op, vdst, t.srcF(src0, in.literal), t.srcF(256 + vsrc1, in.literal));
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
        uint32_t dmask = (w >> 8) & 0xF, vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
        uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
        uint32_t bind = (uint32_t)r.psTexs.size();
        r.psTexs.push_back({bind, srsrc});
        Id texVar = t.m.variable(pSampImg, spv::StorageClass::UniformConstant);
        t.m.decorate(texVar, spv::Decoration::DescriptorSet, {0});
        t.m.decorate(texVar, spv::Decoration::Binding, {bind});
        texVars.push_back(texVar);
        Id uv = t.m.compositeConstruct(t.tV2, {t.ldVgF(vaddr), t.ldVgF(vaddr + 1)});
        Id si = t.m.load(sampImgTy, texVar);
        Id texel = t.m.emit(spv::Op::OpImageSampleImplicitLod, t.tV4, {si, uv});
        uint32_t comp = 0;
        for (int i = 0; i < 4; i++)
          if (dmask & (1 << i))
            t.stVgF(vdata + comp++, t.m.compositeExtract(t.tF, texel, i));
        break;
      }
      case Enc::exp: {
        uint32_t en = w & 0xF, target = (w >> 4) & 0x3F, compr = (w >> 10) & 1;
        uint32_t v[4] = {w1 & 0xFF, (w1 >> 8) & 0xFF, (w1 >> 16) & 0xFF, (w1 >> 24) & 0xFF};
        if (target <= 7) {  // MRT0
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
          t.m.store(colorOut, col);
        }
        break;
      }
      default: break;
    }
    if (in.enc == Enc::sopp && in.opcode == 1) break;
  }
  if (!wroteColor)
    t.m.store(colorOut, t.m.constComposite(t.tV4,
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
  r.vsSpirv = spirv::optimize(vs);
  r.fsSpirv = spirv::optimize(ps);
  r.ok = !r.vsSpirv.empty() && !r.fsSpirv.empty();
  return r.ok;
}

}  // namespace gpu::gcn

#endif  // DELTA_HAVE_SPIRV_BACKEND
