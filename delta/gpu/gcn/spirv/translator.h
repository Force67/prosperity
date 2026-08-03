#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Internal shared state of the GCN -> SPIR-V translator. Only the backend TUs
 * (gcn_spirv.cc, translate_alu.cc, translate_mem.cc) include this; the
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

#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gcn/spirv/spv_emit.h"

namespace gpu::gcn {

using spirv::Id;

constexpr uint64_t kGuestLo = 0x1000000000ull;
constexpr uint64_t kGuestHi = 0x20000000000ull;
inline bool InGuest(uint64_t a) {
  return a >= kGuestLo && a < kGuestHi;
}

// Loud, deduplicated report of an instruction the translator does not
// implement, so it falls back to an approximation. Logged once per distinct
// (encoding, opcode): silent wrong codegen is never acceptable, but a
// per-frame flood is useless.
void WarnUnsupported(const char* enc,
                     uint32_t op,
                     uint32_t w0 = 0,
                     uint32_t w1 = 0);
// Translated, but not to the letter of the spec: recorded, not rejected.
void NoteApproximated(const char* enc, uint32_t op);
void ResetUnsupported();
bool HadUnsupported();
const std::string& UnsupportedOps();

// DELTA_GPU_SHTRACE: translator debug logging.
bool TraceEnabled();

// Translator context: one SPIR-V module + the register-file model.
struct Translator {
  spirv::Module m;
  Id t_void = 0, t_fn = 0, t_u = 0, t_i = 0, t_f = 0, t_bool = 0;
  Id t_v2 = 0, t_v3 = 0, t_v4 = 0;
  Id p_priv_u = 0, sgpr = 0, vgpr = 0;
  bool predicate_vector = false;
  bool rdna_sources = false;
  // Guest address of dword 0 of the program being translated. Guest memory is
  // identity-mapped, so this is just the code pointer. s_getpc_b64 needs it:
  // shaders form absolute addresses from their own PC to reach descriptors
  // stored alongside the code, and a PC of zero sends those loads to null.
  uint64_t program_base = 0;
  // Graphics stages under PushCodeBase(): the {lo, hi} u32 members of the push
  // block that carry the stage's code address per draw. s_getpc_b64 then reads
  // the address at draw time instead of baking program_base into the module,
  // so one content-keyed module serves every address the title streams the
  // shader to. Zero = bake program_base (compute, or the 128-byte push floor).
  Id pc_base_var = 0;
  uint32_t pc_base_member = 0;
  Id scc_var = 0;    // scalar condition code
  Id state_var = 0;  // CFG block index for the while-switch dispatch
  Id cbuf_type = 0;  // shared CB { uvec4 data[64]; } type
  std::unordered_map<uint32_t, Id> cbuf_vars;  // binding -> cbuffer UBO var
  Id gfx_buf_type = 0;  // shared Buf { uint data[]; } type (set 2)
  std::unordered_map<uint32_t, Id> gfx_buf_vars;  // binding -> raw-buffer SSBO
  // Indexed by (arrayed ? 1 : 0) | (dref ? 2 : 0) | (3D ? 4 : 0).
  Id img_types[8] = {};          // sampled 2D / 2D-array / 3D, color / depth
  Id sampled_types[8] = {};      // corresponding combined image-sampler types
  Id sampled_ptrs[8] = {};       // UniformConstant pointers to sampled_types
  Id storage_img_types[2] = {};  // storage 2D / 2D-array images
  Id storage_img_ptrs[2] = {};   // UniformConstant pointers to storage images
  bool image_query = false;
  // DELTA_GPU_PSTEX: the most recent image sample, so the PS epilogue can
  // export it instead of the shader's own colour maths (see gcn_spirv.cc).
  Id last_texel = 0;
  Id dbg_file = 0;  // OpString for OpLine pc markers (DELTA_GPU_SHDUMP)

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
    // A T# can start at s124 and extend through s131.
    const Id arr_sg = m.TypeArray(t_u, 136), arr_vg = m.TypeArray(t_u, 256);
    sgpr = m.Variable(m.TypePointer(spv::StorageClass::Private, arr_sg),
                      spv::StorageClass::Private, m.ConstNull(arr_sg));
    vgpr = m.Variable(m.TypePointer(spv::StorageClass::Private, arr_vg),
                      spv::StorageClass::Private, m.ConstNull(arr_vg));
    m.Name(sgpr, "sgpr");
    m.Name(vgpr, "vgpr");
    scc_var =
        m.Variable(p_priv_u, spv::StorageClass::Private, m.ConstNull(t_u));
    state_var =
        m.Variable(p_priv_u, spv::StorageClass::Private, m.ConstNull(t_u));
    m.Name(scc_var, "scc");
    m.Name(state_var, "state");
  }

  // Seed EXEC all-active (sgpr[126] = 1) at the start of the function body.
  // Must run after BeginFunction (emits an OpStore).
  void SeedExec() { SetSg(126, U32(1)); }

  void RequireImageQuery() {
    if (image_query)
      return;
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
  Id Sg(uint32_t i) {
    return rdna_sources && i == 125 ? U32(0) : m.Load(t_u, SgPtr(i));
  }
  Id Vg(uint32_t i) { return m.Load(t_u, VgPtr(i)); }
  void SetSg(uint32_t i, Id v) {
    if (!rdna_sources || i != 125)
      m.Store(SgPtr(i), v);
  }
  Id Sdst(uint32_t base, uint32_t offset = 0) {
    return rdna_sources && base == 125 ? U32(0) : Sg(base + offset);
  }
  void SetSdst(uint32_t base, uint32_t offset, Id v) {
    if (!rdna_sources || base != 125)
      SetSg(base + offset, v);
  }
  void SetVg(uint32_t i, Id v) {
    if (predicate_vector)
      v = SelectNz(Exec(), v, Vg(i));
    m.Store(VgPtr(i), v);
  }
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

  // The two saturating primitives the Sea Islands ISA writes its `_clamp` and
  // `_legacy` transcendentals in terms of (v_rcp_clamp_f32 =
  // ClampInfToFltMax(Rcp(x)), v_rcp_legacy_f32 = ConvertInfToZero(Rcp(x)),
  // and likewise for log/rsq). Both must touch ONLY infinities: a GLSL
  // FMin/FMax/FClamp against +/-FLT_MAX would also fold a NaN into a finite
  // bound, because GLSL's min/max return the non-NaN operand. Selecting on
  // OpIsInf leaves NaN -- and every finite value -- exactly as produced.
  Id IsInf(Id f) { return m.Emit(spv::Op::OpIsInf, t_bool, {f}); }
  Id FLt(Id a, Id b) { return m.Emit(spv::Op::OpFOrdLessThan, t_bool, {a, b}); }
  Id FGt(Id a, Id b) {
    return m.Emit(spv::Op::OpFOrdGreaterThan, t_bool, {a, b});
  }
  static constexpr float kFltMax = 3.402823466e+38f;
  Id ClampInfToFltMax(Id f) {  // +/-Inf -> +/-FLT_MAX
    const Id lim = SelectF(FLt(f, F32(0.0f)), F32(-kFltMax), F32(kFltMax));
    return SelectF(IsInf(f), lim, f);
  }
  Id ConvertInfToZero(Id f) {  // +/-Inf -> +0.0 (legacy DX9)
    return SelectF(IsInf(f), F32(0.0f), f);
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
  Id IsNonZero(Id u) {
    return m.Emit(spv::Op::OpINotEqual, t_bool, {u, U32(0)});
  }
  Id IsZero(Id u) { return m.Emit(spv::Op::OpIEqual, t_bool, {u, U32(0)}); }
  Id Eq(Id a, Id b) { return m.Emit(spv::Op::OpIEqual, t_bool, {a, b}); }
  Id Ult(Id a, Id b) { return m.Emit(spv::Op::OpULessThan, t_bool, {a, b}); }
  Id Ule(Id a, Id b) {
    return m.Emit(spv::Op::OpULessThanEqual, t_bool, {a, b});
  }
  Id Uge(Id a, Id b) {
    return m.Emit(spv::Op::OpUGreaterThanEqual, t_bool, {a, b});
  }
  Id LAnd(Id a, Id b) { return m.Emit(spv::Op::OpLogicalAnd, t_bool, {a, b}); }
  Id SelectB(Id cond, Id a, Id b) {
    return m.Emit(spv::Op::OpSelect, t_u, {cond, a, b});
  }
  Id SelectNz(Id cond_u, Id a, Id b) {
    return SelectB(IsNonZero(cond_u), a, b);
  }
  Id SelectF(Id cond, Id a, Id b) {
    return m.Emit(spv::Op::OpSelect, t_f, {cond, a, b});
  }

  Id PairType() { return m.TypeStruct({t_u, t_u}); }  // {result, carry/hi}

  // ---- constant buffers (graphics SMRD model) ----
  // Declared as CB { uvec4 data[]; } at set 1. Separate bindings preserve
  // the distinct V# resources selected by each s_buffer_load.
  Id EnsureCbuf(uint32_t binding) {
    auto it = cbuf_vars.find(binding);
    if (it != cbuf_vars.end())
      return it->second;
    if (!cbuf_type) {
      const Id arr = m.TypeArray(m.TypeVec(t_u, 4), kCbufDwords / 4);
      m.Decorate(arr, spv::Decoration::ArrayStride, {16});
      cbuf_type = m.TypeStruct({arr});
      m.Decorate(cbuf_type, spv::Decoration::Block);
      m.MemberDecorate(cbuf_type, 0, spv::Decoration::Offset, {0});
    }
    const Id v =
        m.Variable(m.TypePointer(spv::StorageClass::Uniform, cbuf_type),
                   spv::StorageClass::Uniform);
    m.Decorate(v, spv::Decoration::DescriptorSet, {1});
    m.Decorate(v, spv::Decoration::Binding, {binding});
    m.Name(v, "cbuf" + std::to_string(binding));
    cbuf_vars[binding] = v;
    return v;
  }
  // Read cbuffer dword k (== uvec4 data[k>>2][k&3]) as a uint. The uvec4 index
  // clamps into the declared window so an out-of-range constant
  // index cannot produce an invalid access chain.
  Id CbufDword(uint32_t binding, uint32_t k) {
    const Id var = EnsureCbuf(binding);
    const Id p_u = m.TypePointer(spv::StorageClass::Uniform, t_u);
    const Id ch = m.AccessChain(
        p_u, var,
        {U32(0), U32(std::min(k >> 2, kCbufDwords / 4 - 1)), U32(k & 3)});
    return m.Load(t_u, ch);
  }
  Id CbufDwordId(uint32_t binding, Id k) {
    const Id var = EnsureCbuf(binding);
    const Id v4 = UMin(Shr(k, U32(2)), U32(kCbufDwords / 4 - 1));
    const Id p_u = m.TypePointer(spv::StorageClass::Uniform, t_u);
    const Id ch = m.AccessChain(p_u, var, {U32(0), v4, And(k, U32(3))});
    return m.Load(t_u, ch);
  }

  // ---- raw buffers (graphics MUBUF model) ----
  // Declared as Buf { uint data[]; } at set 2, one binding per distinct V# the
  // stage loads through. Storage rather than uniform because the address is a
  // per-lane index, not a constant offset.
  Id EnsureGfxBuffer(uint32_t binding) {
    auto it = gfx_buf_vars.find(binding);
    if (it != gfx_buf_vars.end())
      return it->second;
    if (!gfx_buf_type) {
      const Id run = m.TypeRuntimeArray(t_u);
      m.Decorate(run, spv::Decoration::ArrayStride, {4});
      gfx_buf_type = m.TypeStruct({run});
      m.Decorate(gfx_buf_type, spv::Decoration::Block);
      m.MemberDecorate(gfx_buf_type, 0, spv::Decoration::Offset, {0});
    }
    const Id v = m.Variable(
        m.TypePointer(spv::StorageClass::StorageBuffer, gfx_buf_type),
        spv::StorageClass::StorageBuffer);
    m.Decorate(v, spv::Decoration::DescriptorSet, {2});
    m.Decorate(v, spv::Decoration::Binding, {binding});
    m.Name(v, "rawbuf" + std::to_string(binding));
    gfx_buf_vars[binding] = v;
    return v;
  }

  // ---- operand sources ----
  // Raw uint of a source operand field (SSRC/VSRC encoding).
  Id SrcRaw(uint32_t field, uint32_t literal) {
    if (field <= 127)
      return Sg(field);
    if (field == 128)
      return U32(0);
    if (field >= 129 && field <= 192)
      return U32(field - 128);
    if (field >= 193 && field <= 208)
      return U32(static_cast<uint32_t>(-static_cast<int>(field - 192)));
    switch (field) {  // inline float constants
      case 240:
        return U32(0x3f000000u);
      case 241:
        return U32(0xbf000000u);
      case 242:
        return U32(0x3f800000u);
      case 243:
        return U32(0xbf800000u);
      case 244:
        return U32(0x40000000u);
      case 245:
        return U32(0xc0000000u);
      case 246:
        return U32(0x40800000u);
      case 247:
        return U32(0xc0800000u);
      case 248:
        return U32(0x3e22f983u);  // INV_2PI
      case 251:
        return SelectB(IsZero(Sg(106)), U32(1), U32(0));
      case 252:
        return SelectB(IsZero(Exec()), U32(1), U32(0));
      case 253:
        return Scc();
    }
    if (field == 255)
      return U32(literal);
    if (field >= 256)
      return Vg(field - 256);
    return U32(0);
  }
  // High dword of a 64-bit source. Register operands use the adjacent
  // SGPR/VGPR; inline and literal operands are extended from their low dword.
  Id SrcRawHi(uint32_t field, uint32_t literal, bool sign_extend) {
    if (rdna_sources && field == 125)
      return U32(0);
    if (field <= 126)
      return Sg(field + 1);
    if (field >= 256 && field <= 510)
      return Vg(field - 255);
    const Id lo = SrcRaw(field, literal);
    // An integer inline constant is a 64-bit value in a 64-bit operand, so
    // -1..-16 fill the high dword too. Sign-extending 0..64 is a no-op, so this
    // needs no per-op opt-in. Without it `s_lshr_b64 exec, -1, n` -- the NGG
    // prologue's lane mask -- yields EXEC 0 and masks off the whole shader.
    if (field >= 128 && field <= 208)
      return Sar(lo, U32(31));
    return sign_extend ? Sar(lo, U32(31)) : U32(0);
  }
  // Float source with the VOP3 neg/abs input modifiers applied.
  Id SrcF(uint32_t field,
          uint32_t literal,
          bool neg = false,
          bool abs = false) {
    Id f = m.Bitcast(t_f, SrcRaw(field, literal));
    if (abs)
      f = Ext1(GLSLstd450FAbs, f);
    if (neg)
      f = FNeg(f);
    return f;
  }
};

// SPI_PS_INPUT_ENA lays system inputs into initial VGPRs in ascending bit
// order. Interpolated parameters are modeled as Vulkan Location inputs by the
// instruction translator; seed the system values that shaders read directly.
inline void SeedPsInputVgprs(Translator& t,
                             uint32_t ena,
                             std::vector<Id>& iface) {
  if (!ena)
    return;
  static constexpr uint8_t width[16] = {2, 2, 2, 3, 2, 2, 2, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1};
  uint32_t vg[16] = {}, next = 0;
  for (uint32_t bit = 0; bit < 16; bit++)
    if (ena & (1u << bit)) {
      vg[bit] = next;
      next += width[bit];
    }

  if ((ena >> 8) & 0xF) {
    const Id frag_coord =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_v4),
                     spv::StorageClass::Input);
    t.m.Decorate(frag_coord, spv::Decoration::BuiltIn,
                 {static_cast<uint32_t>(spv::BuiltIn::FragCoord)});
    iface.push_back(frag_coord);
    const Id value = t.m.Load(t.t_v4, frag_coord);
    for (uint32_t component = 0; component < 4; component++)
      if (ena & (1u << (8 + component)))
        t.SetVgF(vg[8 + component],
                 t.m.CompositeExtract(t.t_f, value, component));
  }

  if (ena & (1u << 12)) {
    const Id front_facing =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_bool),
                     spv::StorageClass::Input);
    t.m.Decorate(front_facing, spv::Decoration::BuiltIn,
                 {static_cast<uint32_t>(spv::BuiltIn::FrontFacing)});
    iface.push_back(front_facing);
    t.SetVg(vg[12],
            t.SelectB(t.m.Load(t.t_bool, front_facing), t.U32(0xFFFFFFFFu),
                      t.U32(0)));
  }
}

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
  std::unordered_set<uint32_t>
      direct_vfetch;  // MUBUF pc seeded as vertex input
  uint32_t max_param = 0;
  // PS5 inline vertex fetch: (Location input, first dest VGPR, component count)
  // keyed by the fetch MUBUF's pc. The destination VGPRs are (re)seeded from
  // the input AT the fetch instruction, because an NGG merged-wave's index math
  // can overwrite them (e.g. v0, reused as the fetch index) between the
  // function prologue and the position transform.
  struct VfetchSeed {
    Id in_var;
    uint32_t dest_vgpr, num_comps;
  };
  std::unordered_map<uint32_t, VfetchSeed> vfetch_seed;
  // DELTA_GPU_DBGPOS: the position input and the cbuffer holding a 4x4
  // transform, so the position export can be recomputed the way the draw's
  // host-side state says it should be. Splits "the lifted math is wrong" from
  // "the values reaching the shader are wrong".
  Id dbg_pos_in = 0;
  uint32_t dbg_pos_comps = 0;
  int dbg_pos_cbuf = -1;
  uint32_t dbg_pos_dword = 0;
  int dbg_pos_world = -1;

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
  // Set 0 is shared by both stages, so a VS's samplers are numbered after the
  // PS's. tex_vars[] stays indexed by the stage-local binding.
  uint32_t tex_binding_base = 0;
  static constexpr uint32_t kMaxPsSamplers = 24;  // == gpu::vk::kMaxTex
  Id tex_vars[kMaxPsSamplers] = {};
  uint32_t tex_types[kMaxPsSamplers] = {};
  // Bit i set: binding i's T# is SQ_RSRC_IMG_3D. Nothing in the MIMG encoding
  // says so (a 3D descriptor leaves DA 0), so the caller has to supply it from
  // the decoded descriptors, and it belongs in the shader cache key.
  uint32_t tex_3d_mask = 0;
  // Bit i set: binding i's T# is SQ_RSRC_IMG_1D[_ARRAY]. Same reasoning as
  // tex_3d_mask: the instruction encodes only DA, not the dimensionality. The
  // resource is bound as a height-1 2D image; the address body carries x
  // (+layer) and y is fixed at the row centre.
  uint32_t tex_1d_mask = 0;

  // shared graphics
  std::unordered_map<uint32_t, uint32_t> cbuf_bind;  // V# SGPR -> set-1 binding
  // Raw MUBUF loads: instruction pc -> set-2 storage-buffer binding (see
  // PlanGfxBuffers). Keyed per instruction, not per SGPR, because one SGPR quad
  // can hold several descriptors over a shader's life.
  std::unordered_map<uint32_t, uint32_t> gfx_buf_bind;
  // Per-instruction cbuf bindings for constant buffer_loads whose srsrc SGPRs
  // are reused (PS5 table-chained descriptors); takes precedence over
  // cbuf_bind.
  std::unordered_map<uint32_t, uint32_t> mubuf_cbuf_by_pc;
  // Per-instruction RDNA SMEM binding when one sbase has multiple producers.
  std::unordered_map<uint32_t, uint32_t> smem_cbuf_by_pc;
  // pcs of `s_mov exec, sN` movs where sN holds unmodelled SPI launch state
  // (e.g. the PS coverage mask); emitting them would zero EXEC and skip every
  // export in the CFG path, so they are dropped (EXEC keeps its all-on seed).
  std::unordered_set<uint32_t> skip_launch_movs;

  // Compute: storage buffers modelling the guest memory the CS reads/writes.
  std::unordered_map<uint32_t, uint32_t> cs_bind;  // instruction pc -> binding
  std::vector<Id> cs_ssbo;                         // binding -> SSBO variable
  // Attributes that v_interp_mov_f32 reads as P10 or P20. Those are the
  // per-vertex DELTAS (P1-P0, P2-P0), which an interpolated fragment input
  // cannot supply, so the whole Location is declared PerVertexKHR -- an
  // array[3] of the triangle's vertex values. A Location cannot be both that
  // and an ordinary interpolated input, and two variables cannot share one, so
  // the choice is per attribute: everything here goes through the array (its
  // interpolated value recomputed from BaryCoordKHR), everything else keeps
  // the plain input.
  std::unordered_set<uint32_t> pervertex_attrs;
  std::unordered_map<uint32_t, Id> pervertex_vars;
  Id bary_var = 0;  // BaryCoordKHR, declared on first use

  Id lds_var = 0;           // uint array backing LDS (0 = no LDS)
  uint32_t lds_dwords = 0;  // its length
  // Workgroup in a CS. A fragment shader cannot declare Workgroup storage at
  // all -- SPIR-V allows that class only in GLCompute/Kernel/Task/Mesh -- so a
  // graphics stage backs LDS with Private, one copy per invocation. That is
  // exact precisely when every address is the lane's OWN slot, which is what
  // `v_mbcnt_{lo,hi}(-1)` computes: the ISA counts bits in the explicit vsrc
  // mask, so an all-ones mask yields the raw thread id regardless of EXEC.
  // The idiom that would break it is mbcnt applied to EXEC (a compaction
  // index), where lanes share and reuse slots over time.
  spv::StorageClass lds_storage = spv::StorageClass::Workgroup;
  Id subgroup_local_id = 0;     // SubgroupLocalInvocationId for DS swizzles
  // Instruction indices (sorted) at which a workgroup barrier must be emitted
  // because the guest compiler omitted one it was entitled to omit on a
  // 64-lane wave. See PlanLdsBarriers.
  std::vector<uint32_t> lds_barrier_at;
  bool cs_unsupported = false;  // op the compute backend can't model
};

// ---- stage-io helpers (gcn_spirv.cc) --------------------------------------
Id PsInputVar(Translator& t, StageContext& sc, uint32_t attr);
Id VsParamOut(Translator& t, StageContext& sc, uint32_t p);
Id PsColorOut(Translator& t, StageContext& sc, uint32_t target);
Id PsDepthOut(Translator& t, StageContext& sc);

// ---- ALU emitters (translate_alu.cc) --------------------------------------
void EmitSop1(Translator& t, const Inst& inst);
void EmitSop2(Translator& t, const Inst& inst);
void EmitSopc(Translator& t, const Inst& inst);
void EmitSopk(Translator& t, const Inst& inst);
void EmitVop1(Translator& t,
              uint32_t op,
              uint32_t vdst,
              Id s0,
              bool clamp = false,
              uint32_t omod = 0);
void EmitVop2(Translator& t,
              uint32_t op,
              uint32_t vdst,
              Id s0,
              Id s1,
              uint32_t literal = 0,
              bool clamp = false,
              uint32_t omod = 0);
void EmitVop3(Translator& t,
              uint32_t op,
              uint32_t vdst,
              Id s0,
              Id s0_hi,
              Id s1,
              Id s2,
              Id s2_hi,
              uint32_t sdst,
              bool clamp,
              uint32_t omod = 0,
              Id s1_hi = 0);
// Vector compare: writes the 0/1 predicate to sgpr[dst]; the cmpx forms also
// replace EXEC.
// s0_hi/s1_hi carry the high dword of a 64-bit operand pair; the f64/i64/u64
// opcode families need both halves to compare a whole value.
void EmitVopc(Translator& t,
              uint32_t op,
              Id s0f,
              Id s1f,
              Id s0u,
              Id s1u,
              uint32_t dst = 106,
              Id s0_hi = 0,
              Id s1_hi = 0);
bool IsVop3b(uint32_t op);
bool EmitNeoVop1(Translator& t, const Inst& inst);
bool EmitNeoVop2(Translator& t, const Inst& inst);
bool EmitNeoVopc(Translator& t,
                 uint32_t op,
                 uint32_t dst,
                 uint32_t src0,
                 uint32_t src1,
                 uint32_t literal,
                 bool src0_high = false,
                 bool src1_high = false,
                 bool src0_neg = false,
                 bool src1_neg = false,
                 bool src0_abs = false,
                 bool src1_abs = false);
bool EmitNeoVop3(Translator& t, const Inst& inst);
bool EmitNeoVop3p(Translator& t, const Inst& inst);

// ---- memory emitters (translate_mem.cc) -----------------------------------
uint32_t SmrdLoadCount(uint32_t op);
uint32_t SmrdDwordCount(uint32_t op);
bool PlanCbufs(const Program& program,
               uint32_t first_binding,
               std::vector<ShaderCbuf>& cbufs,
               std::unordered_map<uint32_t, uint32_t>& bindings,
               const uint8_t* reachable = nullptr);
void EmitCbufSmrd(Translator& t,
                  const Inst& inst,
                  const std::unordered_map<uint32_t, uint32_t>& bindings);
// True for the MIMG ops that state their own LOD, i.e. the ones legal outside a
// fragment shader (see the definition for the opcode-bit reasoning).
bool MimgNamesItsLod(uint32_t op);
// DS ops a graphics stage may run against Private-backed LDS: the plain loads
// and stores only. Atomics are excluded because Vulkan does not allow them on a
// Private pointer, and everything else declines.
bool DsGraphicsSupported(uint32_t op);
// Dwords of Private LDS a graphics program needs, from the largest address its
// DS instructions can reach. 0 if it has none. Sized statically rather than
// from M0: these shaders set M0 to 0x10000, the whole 64 KB, which is an upper
// bound meaning "unrestricted", not an allocation.
uint32_t GraphicsLdsDwords(const Program& program, const uint8_t* reachable);
void EmitMimg(Translator& t,
              const Inst& inst,
              StageContext& sc,
              const Id* address = nullptr);
// Assign a set-2 storage-buffer binding to every raw MUBUF/MTBUF load in a
// graphics stage, skipping the instructions `claimed` already serves as vertex
// inputs. Instructions sharing one descriptor (same SGPR quad, same producing
// scalar load) share a binding; anything past MaxGfxBuffers() is left unplanned
// and warns at emit time, exactly as an unimplemented op would.
void PlanGfxBuffers(const Program& program,
                    uint32_t first_binding,
                    const std::unordered_set<uint32_t>* claimed,
                    std::vector<ShaderBuffer>& buffers,
                    std::unordered_map<uint32_t, uint32_t>& bindings,
                    const uint8_t* reachable = nullptr);
void EmitGfxMubuf(Translator& t, const Inst& inst, StageContext& sc);
// Typed buffer op in a graphics stage. Loads go through the same set-2 window
// as EmitGfxMubuf; a store is dropped, because that window is not written back.
void EmitGfxMtbuf(Translator& t, const Inst& inst, StageContext& sc);
bool PlanCsResources(const Program& program,
                     const uint8_t* reachable,
                     uint32_t lds_dwords,
                     RecompiledCs& r,
                     std::unordered_map<uint32_t, uint32_t>& bind);
void EmitCsSmrd(Translator& t, const Inst& inst, StageContext& sc);
void EmitCsMubuf(Translator& t, const Inst& inst, StageContext& sc);
void EmitCsMtbuf(Translator& t, const Inst& inst, StageContext& sc);
void EmitCsMimg(Translator& t,
                const Inst& inst,
                StageContext& sc,
                const Id* address = nullptr);
void EmitDs(Translator& t, const Inst& inst, StageContext& sc);

// The compute resource model: a guest range aliased as Buf { uint data[]; },
// addressed by dword index. ISA-neutral, so the RDNA2 path binds the same
// buffers and only has to decode its own (differently encoded) scalar loads.
Id CsSsboPtr(Translator& t, StageContext& sc, uint32_t binding, Id dword_idx);
Id CsSsboLoad(Translator& t, StageContext& sc, uint32_t binding, Id dword_idx);
void CsSsboStore(Translator& t,
                 StageContext& sc,
                 uint32_t binding,
                 Id dword_idx,
                 Id value);
// Storage-buffer binding planned for the instruction at pc, or -1.
int CsBindingFor(StageContext& sc, uint32_t pc);

// RECTLIST expansion stage: three post-VS corners in, two triangles out. Shared
// with the RDNA2 path, which reaches RECTLIST under a different primitive-type
// number but needs the identical fixed-function expansion.
std::vector<uint32_t> EmitRectListGeometry(
    uint32_t num_params,
    const std::unordered_set<uint32_t>& flat_attrs);

}  // namespace gpu::gcn
