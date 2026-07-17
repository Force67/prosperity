#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Internal shared state of the GCN -> SPIR-V translator. Only the backend TUs
 * (gcn_spirv.cpp, translate_alu.cpp, translate_mem.cpp) include this; the
 * public surface is gcn_spirv.h / gcn_translate.h.
 *
 * The translator models the GCN register file as Private-storage arrays
 * (sgpr[128] / vgpr[256]) plus SCC/EXEC/state scalars, emits straight
 * load/compute/store SPIR-V per instruction, and relies on spirv-opt's SSA
 * rewrite to clean it up. One wave lane == one SPIR-V invocation: EXEC is a
 * single "this lane active" bit, VCC (sgpr[106]) a 0/1 scalar.
 *
 * ISA reference: AMD Sea Islands (GFX7) ISA,
 * https://docs.amd.com/v/u/en-US/sea-islands-instruction-set-architecture_0
 */

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <spirv/unified1/GLSL.std.450.h>

#include "../gcn_decode.h"
#include "../gcn_resource.h"
#include "../gcn_translate.h"
#include "spv_emit.h"

namespace gpu::gcn {

using spirv::Id;

constexpr uint64_t kGuestLo = 0x1000000000ull;
constexpr uint64_t kGuestHi = 0x20000000000ull;
inline bool InGuest(uint64_t a) { return a >= kGuestLo && a < kGuestHi; }

// Loud, deduplicated report of an instruction the translator does not
// implement, so it falls back to an approximation. Logged once per distinct
// (encoding, opcode): silent wrong codegen is never acceptable, but a
// per-frame flood is useless.
void WarnUnsupported(const char* enc, uint32_t op, uint32_t w0 = 0,
                     uint32_t w1 = 0);

// DELTA_GPU_SHTRACE: translator debug logging.
bool TraceEnabled();

// Translator context: one SPIR-V module + the register-file model.
struct Translator {
  spirv::Module m;
  Id t_void = 0, t_fn = 0, t_u = 0, t_i = 0, t_f = 0, t_bool = 0;
  Id t_v2 = 0, t_v3 = 0, t_v4 = 0;
  Id p_priv_u = 0, sgpr = 0, vgpr = 0;
  Id scc_var = 0;    // scalar condition code
  Id state_var = 0;  // CFG block index for the while-switch dispatch
  Id cbuf_type = 0;  // shared CB { uvec4 data[64]; } type
  std::unordered_map<uint32_t, Id> cbuf_vars;  // binding -> cbuffer UBO var
  Id img_types[4] = {};      // sampled 2D / 2D-array, color / depth images
  Id sampled_types[4] = {};  // corresponding combined image-sampler types
  Id sampled_ptrs[4] = {};   // UniformConstant pointers to sampled_types
  bool image_query = false;

  void InitTypes() {
    t_void = m.TypeVoid();
    t_fn = m.TypeFunction(t_void);
    t_u = m.TypeInt(32, false);
    t_i = m.TypeInt(32, true);
    t_f = m.TypeFloat(32);
    t_bool = m.TypeBool();
    t_v2 = m.TypeVec(t_f, 2);
    t_v3 = m.TypeVec(t_f, 3);
    t_v4 = m.TypeVec(t_f, 4);
    p_priv_u = m.TypePointer(spv::StorageClass::Private, t_u);
    const Id arr_sg = m.TypeArray(t_u, 128), arr_vg = m.TypeArray(t_u, 256);
    sgpr = m.Variable(m.TypePointer(spv::StorageClass::Private, arr_sg),
                      spv::StorageClass::Private, m.ConstNull(arr_sg));
    vgpr = m.Variable(m.TypePointer(spv::StorageClass::Private, arr_vg),
                      spv::StorageClass::Private, m.ConstNull(arr_vg));
    m.Name(sgpr, "sgpr");
    m.Name(vgpr, "vgpr");
    scc_var = m.Variable(p_priv_u, spv::StorageClass::Private, m.ConstNull(t_u));
    state_var = m.Variable(p_priv_u, spv::StorageClass::Private, m.ConstNull(t_u));
    m.Name(scc_var, "scc");
    m.Name(state_var, "state");
  }

  // Seed EXEC all-active (sgpr[126] = 1) at the start of the function body.
  // Must run after BeginFunction (emits an OpStore).
  void SeedExec() { SetSg(126, U32(1)); }

  void RequireImageQuery() {
    if (image_query) return;
    m.Capability(spv::Capability::ImageQuery);
    image_query = true;
  }

  // ---- SCC / EXEC / CFG-state ----
  Id Scc() { return m.Load(t_u, scc_var); }
  void SetScc(Id v) { m.Store(scc_var, v); }
  void SetSccBool(Id b) { SetScc(SelectB(b, U32(1), U32(0))); }
  Id Exec() { return Sg(126); }
  Id State() { return m.Load(t_u, state_var); }
  void SetState(uint32_t s) { m.Store(state_var, U32(s)); }
  void SetStateId(Id s) { m.Store(state_var, s); }

  // ---- register file ----
  Id SgPtr(uint32_t i) { return m.AccessChain(p_priv_u, sgpr, {U32(i)}); }
  Id VgPtr(uint32_t i) { return m.AccessChain(p_priv_u, vgpr, {U32(i)}); }
  Id Sg(uint32_t i) { return m.Load(t_u, SgPtr(i)); }
  Id Vg(uint32_t i) { return m.Load(t_u, VgPtr(i)); }
  void SetSg(uint32_t i, Id v) { m.Store(SgPtr(i), v); }
  void SetVg(uint32_t i, Id v) { m.Store(VgPtr(i), v); }
  Id VgF(uint32_t i) { return m.Bitcast(t_f, Vg(i)); }
  void SetVgF(uint32_t i, Id f) { SetVg(i, m.Bitcast(t_u, f)); }

  // ---- constants ----
  Id U32(uint32_t v) { return m.ConstU32(v); }
  Id F32(float v) { return m.ConstF32(v); }

  // ---- float ALU ----
  Id Ext1(uint32_t op, Id a) { return m.ExtInst(t_f, op, {a}); }
  Id Ext2(uint32_t op, Id a, Id b) { return m.ExtInst(t_f, op, {a, b}); }
  Id FMul(Id a, Id b) { return m.Emit(spv::Op::OpFMul, t_f, {a, b}); }
  Id FAdd(Id a, Id b) { return m.Emit(spv::Op::OpFAdd, t_f, {a, b}); }
  Id FSub(Id a, Id b) { return m.Emit(spv::Op::OpFSub, t_f, {a, b}); }
  Id FDiv(Id a, Id b) { return m.Emit(spv::Op::OpFDiv, t_f, {a, b}); }
  Id FNeg(Id a) { return m.Emit(spv::Op::OpFNegate, t_f, {a}); }
  Id FClamp01(Id f) {
    return m.ExtInst(t_f, GLSLstd450FClamp, {f, F32(0.0f), F32(1.0f)});
  }

  // ---- integer ALU (uint domain; signed ops bitcast through t_i). Shifts
  // mask the amount to [4:0] as GCN does. ----
  Id Add(Id a, Id b) { return m.Emit(spv::Op::OpIAdd, t_u, {a, b}); }
  Id Sub(Id a, Id b) { return m.Emit(spv::Op::OpISub, t_u, {a, b}); }
  Id Mul(Id a, Id b) { return m.Emit(spv::Op::OpIMul, t_u, {a, b}); }
  Id And(Id a, Id b) { return m.Emit(spv::Op::OpBitwiseAnd, t_u, {a, b}); }
  Id Or(Id a, Id b) { return m.Emit(spv::Op::OpBitwiseOr, t_u, {a, b}); }
  Id Xor(Id a, Id b) { return m.Emit(spv::Op::OpBitwiseXor, t_u, {a, b}); }
  Id Not(Id a) { return m.Emit(spv::Op::OpNot, t_u, {a}); }
  Id Shl(Id a, Id s) {
    return m.Emit(spv::Op::OpShiftLeftLogical, t_u, {a, And(s, U32(31))});
  }
  Id Shr(Id a, Id s) {
    return m.Emit(spv::Op::OpShiftRightLogical, t_u, {a, And(s, U32(31))});
  }
  Id Sar(Id a, Id s) {
    return m.Bitcast(t_u, m.Emit(spv::Op::OpShiftRightArithmetic, t_i,
                                 {m.Bitcast(t_i, a), And(s, U32(31))}));
  }
  Id SMin(Id a, Id b) {
    return m.Bitcast(t_u, m.ExtInst(t_i, GLSLstd450SMin,
                                    {m.Bitcast(t_i, a), m.Bitcast(t_i, b)}));
  }
  Id SMax(Id a, Id b) {
    return m.Bitcast(t_u, m.ExtInst(t_i, GLSLstd450SMax,
                                    {m.Bitcast(t_i, a), m.Bitcast(t_i, b)}));
  }
  Id UMin(Id a, Id b) { return m.ExtInst(t_u, GLSLstd450UMin, {a, b}); }
  Id UMax(Id a, Id b) { return m.ExtInst(t_u, GLSLstd450UMax, {a, b}); }
  Id PopCount(Id a) { return m.Emit(spv::Op::OpBitCount, t_u, {a}); }
  Id BitRev(Id a) { return m.Emit(spv::Op::OpBitReverse, t_u, {a}); }
  Id Sext24(Id a) {  // sign-extend the low 24 bits
    return m.Bitcast(t_u, m.Emit(spv::Op::OpBitFieldSExtract, t_i,
                                 {m.Bitcast(t_i, a), U32(0), U32(24)}));
  }
  Id Low24(Id a) { return And(a, U32(0xFFFFFF)); }

  // ---- comparisons / selects (Bool domain) ----
  Id IsNonZero(Id u) { return m.Emit(spv::Op::OpINotEqual, t_bool, {u, U32(0)}); }
  Id IsZero(Id u) { return m.Emit(spv::Op::OpIEqual, t_bool, {u, U32(0)}); }
  Id Eq(Id a, Id b) { return m.Emit(spv::Op::OpIEqual, t_bool, {a, b}); }
  Id Ult(Id a, Id b) { return m.Emit(spv::Op::OpULessThan, t_bool, {a, b}); }
  Id Ule(Id a, Id b) { return m.Emit(spv::Op::OpULessThanEqual, t_bool, {a, b}); }
  Id Uge(Id a, Id b) {
    return m.Emit(spv::Op::OpUGreaterThanEqual, t_bool, {a, b});
  }
  Id LAnd(Id a, Id b) { return m.Emit(spv::Op::OpLogicalAnd, t_bool, {a, b}); }
  Id SelectB(Id cond, Id a, Id b) {
    return m.Emit(spv::Op::OpSelect, t_u, {cond, a, b});
  }
  Id SelectNz(Id cond_u, Id a, Id b) { return SelectB(IsNonZero(cond_u), a, b); }
  Id SelectF(Id cond, Id a, Id b) {
    return m.Emit(spv::Op::OpSelect, t_f, {cond, a, b});
  }

  Id PairType() { return m.TypeStruct({t_u, t_u}); }  // {result, carry/hi}

  // ---- constant buffers (graphics SMRD model) ----
  // Declared as CB { uvec4 data[64]; } at set 1. Separate bindings preserve
  // the distinct V# resources selected by each s_buffer_load.
  Id EnsureCbuf(uint32_t binding) {
    auto it = cbuf_vars.find(binding);
    if (it != cbuf_vars.end()) return it->second;
    if (!cbuf_type) {
      const Id arr = m.TypeArray(m.TypeVec(t_u, 4), 64);
      m.Decorate(arr, spv::Decoration::ArrayStride, {16});
      cbuf_type = m.TypeStruct({arr});
      m.Decorate(cbuf_type, spv::Decoration::Block);
      m.MemberDecorate(cbuf_type, 0, spv::Decoration::Offset, {0});
    }
    const Id v = m.Variable(m.TypePointer(spv::StorageClass::Uniform, cbuf_type),
                            spv::StorageClass::Uniform);
    m.Decorate(v, spv::Decoration::DescriptorSet, {1});
    m.Decorate(v, spv::Decoration::Binding, {binding});
    cbuf_vars[binding] = v;
    return v;
  }
  // Read cbuffer dword k (== uvec4 data[k>>2][k&3]) as a uint. The uvec4 index
  // clamps into the 64-element (1 KiB) window so an out-of-range constant
  // index cannot produce an invalid access chain.
  Id CbufDword(uint32_t binding, uint32_t k) {
    const Id var = EnsureCbuf(binding);
    const Id p_u = m.TypePointer(spv::StorageClass::Uniform, t_u);
    const Id ch = m.AccessChain(
        p_u, var, {U32(0), U32((k >> 2) & 63), U32(k & 3)});
    return m.Load(t_u, ch);
  }
  Id CbufDwordId(uint32_t binding, Id k) {
    const Id var = EnsureCbuf(binding);
    const Id v4 = UMin(Shr(k, U32(2)), U32(63));
    const Id p_u = m.TypePointer(spv::StorageClass::Uniform, t_u);
    const Id ch = m.AccessChain(p_u, var, {U32(0), v4, And(k, U32(3))});
    return m.Load(t_u, ch);
  }

  // ---- operand sources ----
  // Raw uint of a source operand field (SSRC/VSRC encoding).
  Id SrcRaw(uint32_t field, uint32_t literal) {
    if (field <= 127) return Sg(field);
    if (field == 128) return U32(0);
    if (field >= 129 && field <= 192) return U32(field - 128);
    if (field >= 193 && field <= 208)
      return U32(static_cast<uint32_t>(-static_cast<int>(field - 192)));
    switch (field) {  // inline float constants
      case 240: return U32(0x3f000000u);
      case 241: return U32(0xbf000000u);
      case 242: return U32(0x3f800000u);
      case 243: return U32(0xbf800000u);
      case 244: return U32(0x40000000u);
      case 245: return U32(0xc0000000u);
      case 246: return U32(0x40800000u);
      case 247: return U32(0xc0800000u);
    }
    if (field == 255) return U32(literal);
    if (field >= 256) return Vg(field - 256);
    return U32(0);
  }
  // High dword of a 64-bit source. Register operands use the adjacent
  // SGPR/VGPR; inline and literal operands are extended from their low dword.
  Id SrcRawHi(uint32_t field, uint32_t literal, bool sign_extend) {
    if (field <= 126) return Sg(field + 1);
    if (field >= 256 && field <= 510) return Vg(field - 255);
    const Id lo = SrcRaw(field, literal);
    return sign_extend ? Sar(lo, U32(31)) : U32(0);
  }
  // Float source with the VOP3 neg/abs input modifiers applied.
  Id SrcF(uint32_t field, uint32_t literal, bool neg = false, bool abs = false) {
    Id f = m.Bitcast(t_f, SrcRaw(field, literal));
    if (abs) f = Ext1(GLSLstd450FAbs, f);
    if (neg) f = FNeg(f);
    return f;
  }
};

// Per-stage state carried into the shared per-instruction emitter (EmitInst).
struct StageContext {
  bool is_ps = false;
  bool is_cs = false;
  Recompiled* r = nullptr;
  std::vector<Id>* iface = nullptr;
  Id main_fn = 0;  // entry function (for stage-wide ExecMode additions)

  // VS
  Id pos_out = 0;
  std::unordered_map<uint32_t, Id> param_outs;
  uint32_t max_param = 0;

  // PS
  Id color_outs[8] = {};  // lazily declared per MRT target (location == target)
  Id depth_out = 0;       // MRTZ -> FragDepth (lazily declared)
  std::unordered_map<uint32_t, Id> in_vars;
  bool wrote_color = false;  // compile-time: shader has a color export
  Id color_written_var = 0;  // runtime: this fragment reached a color export
  const std::unordered_set<uint32_t>* flat_attrs = nullptr;
  // Sampler-binding plan (see PlanMimgBindings): MIMGs referencing the same
  // descriptor share one set-0 binding; variables are created lazily per
  // binding. The plan is also what TrackTextures pairs against at draw time.
  const MimgBindingPlan* mimg_plan = nullptr;
  static constexpr uint32_t kMaxPsSamplers = 8;  // == vk_render State::kMaxTex
  Id tex_vars[kMaxPsSamplers] = {};
  uint32_t tex_types[kMaxPsSamplers] = {};

  // shared graphics
  std::unordered_map<uint32_t, uint32_t> cbuf_bind;  // V# SGPR -> set-1 binding

  // Compute: storage buffers modelling the guest memory the CS reads/writes.
  std::unordered_map<uint32_t, uint32_t> cs_bind;  // base SGPR -> binding
  std::vector<Id> cs_ssbo;                         // binding -> SSBO variable
  Id lds_var = 0;          // Workgroup-storage uint array (0 = no LDS)
  uint32_t lds_dwords = 0;  // its length
  bool cs_unsupported = false;  // op the compute backend can't model
};

// ---- stage-io helpers (gcn_spirv.cpp) --------------------------------------
Id PsInputVar(Translator& t, StageContext& sc, uint32_t attr);
Id VsParamOut(Translator& t, StageContext& sc, uint32_t p);
Id PsColorOut(Translator& t, StageContext& sc, uint32_t target);
Id PsDepthOut(Translator& t, StageContext& sc);

// ---- ALU emitters (translate_alu.cpp) --------------------------------------
void EmitSop1(Translator& t, const Inst& inst);
void EmitSop2(Translator& t, const Inst& inst);
void EmitSopc(Translator& t, const Inst& inst);
void EmitSopk(Translator& t, const Inst& inst);
void EmitVop1(Translator& t, uint32_t op, uint32_t vdst, Id s0,
              bool clamp = false);
void EmitVop2(Translator& t, uint32_t op, uint32_t vdst, Id s0, Id s1,
              uint32_t literal = 0, bool clamp = false);
void EmitVop3(Translator& t, uint32_t op, uint32_t vdst, Id s0, Id s1, Id s2,
              Id s2_hi, uint32_t sdst, bool clamp);
// Vector compare: writes the 0/1 predicate to sgpr[dst]; the cmpx forms also
// replace EXEC.
void EmitVopc(Translator& t, uint32_t op, Id s0f, Id s1f, Id s0u, Id s1u,
              uint32_t dst = 106);
bool IsVop3b(uint32_t op);

// ---- memory emitters (translate_mem.cpp) -----------------------------------
uint32_t SmrdLoadCount(uint32_t op);
// Per-instruction reachability from the entry block (program-index aligned).
// Instructions after an early-out s_endpgm that no branch targets are dead --
// typically OrbShdr footer padding decoded past the real code -- and must not
// influence translation (a garbage "memory op" would decline a compute shader).
std::vector<uint8_t> ComputeReachability(const Program& program);
bool PlanCbufs(const Program& program, uint32_t first_binding,
               std::vector<ShaderCbuf>& cbufs,
               std::unordered_map<uint32_t, uint32_t>& bindings);
void EmitCbufSmrd(Translator& t, const Inst& inst,
                  const std::unordered_map<uint32_t, uint32_t>& bindings);
void EmitMimg(Translator& t, const Inst& inst, StageContext& sc);
bool PlanCsResources(const Program& program, const uint8_t* reachable,
                     uint32_t lds_dwords, RecompiledCs& r,
                     std::unordered_map<uint32_t, uint32_t>& bind);
void EmitCsSmrd(Translator& t, const Inst& inst, StageContext& sc);
void EmitCsMubuf(Translator& t, const Inst& inst, StageContext& sc);
void EmitCsMtbuf(Translator& t, const Inst& inst, StageContext& sc);
void EmitCsMimg(Translator& t, const Inst& inst, StageContext& sc);
void EmitDs(Translator& t, const Inst& inst, StageContext& sc);

}  // namespace gpu::gcn
