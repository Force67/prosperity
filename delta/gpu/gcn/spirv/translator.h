#pragma once

/*
 * PS4Delta: internal state of the GCN -> SPIR-V translator. Backend TUs only
 * (gcn_spirv.cc, translate_alu.cc, translate_mem.cc); public surface is
 * gcn_spirv.h / gcn_translate.h. The register file is Private arrays plus
 * SCC/EXEC scalars; spirv-opt's SSA rewrite cleans the straight-line output.
 * ISA reference: AMD Sea Islands (GFX7),
 * https://docs.amd.com/v/u/en-US/sea-islands-instruction-set-architecture_0
 */

#include "base/arch.h"
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

constexpr u64 kGuestLo = 0x1000000000ull;
constexpr u64 kGuestHi = 0x20000000000ull;
inline bool InGuest(u64 a) {
  return a >= kGuestLo && a < kGuestHi;
}

// Loud, deduplicated report of an unimplemented instruction (once per encoding+opcode).
void WarnUnsupported(const char* enc,
                     u32 op,
                     u32 w0 = 0,
                     u32 w1 = 0);
// Translated, but not to the letter of the spec: recorded, not rejected.
void NoteApproximated(const char* enc, u32 op);
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
  // Mesh: only waves whose private PC names the selected block may change state.
  Id block_active = 0;
  bool rdna_sources = false;
  // Guest address of dword 0; s_getpc_b64 forms absolute addresses from it.
  u64 program_base = 0;
  // Push-constant {lo,hi} of the code address under PushCodeBase(); 0 = bake program_base.
  Id pc_base_var = 0;
  u32 pc_base_member = 0;
  Id scc_var = 0;    // scalar condition code
  Id state_var = 0;  // CFG block index for the while-switch dispatch
  Id cbuf_type = 0;  // shared CB { uvec4 data[64]; } type
  bool indirect_cbufs = false;
  u32 user_data_slot = 0;  // 0 = vertex/geometry, 1 = pixel
  Id cbuf_ring = 0, cbuf_offsets = 0;
  std::unordered_map<u32, Id> cbuf_vars;  // binding -> cbuffer UBO var
  Id gfx_buf_type = 0;  // shared Buf { uint data[]; } type (set 2)
  Id lds_buf_var = 0;   // shared-LDS storage buffer (set 3), see EnsureLdsBuffer
  std::unordered_map<u32, Id> gfx_buf_vars;  // binding -> raw-buffer SSBO
  // Index: arrayed | dref<<1 | 3d<<2 | integer<<3 (integer images yield uvec4).
  Id img_types[16] = {};      // sampled 2D / 2D-array / 3D, color / depth
  Id sampled_types[16] = {};  // corresponding combined image-sampler types
  Id sampled_ptrs[16] = {};   // UniformConstant pointers to sampled_types
  Id storage_img_types[2] = {};  // storage 2D / 2D-array images
  Id storage_img_ptrs[2] = {};   // UniformConstant pointers to storage images
  bool image_query = false;
  // DELTA_GPU_PSTEX: most recent image sample, exported by the PS epilogue.
  Id last_texel = 0;
  // Same, in a Private var: a sample inside a branch does not dominate the epilogue.
  Id last_texel_var = 0;
  // DELTA_GPU_PSVGPR_AT: three VGPRs snapshotted after one instruction.
  Id probe_var = 0;
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

  // Seed EXEC all-active; must run after BeginFunction (emits an OpStore).
  void SeedExec() {
    SetSg(126, lane_masks ? U32(~0u) : wave_masks ? BallotLow(True()) : U32(1));
    if (lane_masks)
      SetSg(127, U32(~0u));
  }

  // ---- wave masks ------------------------------------------------------
  // EXEC and v_cmp results are 64-bit lane masks shaders read as numbers; a
  // one-bit-per-invocation model collapses NGG's v_cmpx lane gating onto lane 0.
  bool wave_masks = false;  // true mask = subgroup ballot; off (GFX7) = one-bit model
  // Per-invocation mask keeps its hardware bit position in the 64-bit pair.
  bool lane_masks = false;
  bool full_wave_masks = false;
  Id mask_lane_id = 0;
  Id MaskLaneHigh() { return IsNonZero(And(mask_lane_id, U32(32))); }
  Id MaskWord(Id lo, Id hi) {
    return lane_masks ? SelectB(MaskLaneHigh(), hi, lo) : lo;
  }
  Id SgMask(u32 sgpr) { return MaskWord(Sg(sgpr), Sg(sgpr + 1)); }
  // A wave32 mask is one SGPR; the next one is an ordinary register the
  // compiler may still hold a live value in (a descriptor table pointer, say).
  void SetMask(u32 sgpr, Id value) {
    if (full_wave_masks) {
      WavePublish(value);
      Barrier();
      const Id lo = WaveFetch(U32(0)), hi = WaveFetch(U32(32));
      Barrier();
      SetSg(sgpr, lo);
      if (wave_size != 32)
        SetSg(sgpr + 1, hi);
    } else if (lane_masks && wave_size == 32) {
      SetSg(sgpr, value);
    } else if (lane_masks) {
      const Id high = MaskLaneHigh();
      SetSg(sgpr, SelectB(high, U32(0), value));
      SetSg(sgpr + 1, SelectB(high, value, U32(0)));
    } else {
      SetSg(sgpr, value);
    }
  }
  Id BallotLow(Id cond) {
    RequireSubgroup(spv::Capability::GroupNonUniformBallot);
    const Id b = m.Emit(spv::Op::OpGroupNonUniformBallot, TypeV4u(),
                        {U32(static_cast<u32>(spv::Scope::Subgroup)), cond});
    return m.CompositeExtract(t_u, b, 0);
  }
  // A predicate as the wave mask an instruction writes to an SGPR.
  Id MaskOf(Id cond) {
    if (full_wave_masks) {
      const Id half = XchgAt(Add(wave_base, And(WaveLane(), U32(32))));
      const Id init = m.NewBlock(), ready = m.NewBlock();
      const Id leader = IsZero(And(WaveLane(), U32(31)));
      m.SelectionMerge(ready);
      m.BranchConditional(leader, init, ready);
      m.OpenBlock(init);
      m.Store(half, U32(0));
      m.Branch(ready);
      m.OpenBlock(ready);
      Barrier();
      m.Emit(spv::Op::OpAtomicOr, t_u,
             {half, U32(2), U32(0x108),
              SelectB(cond, Shl(U32(1), And(WaveLane(), U32(31))), U32(0))});
      Barrier();
      const Id mask = m.Load(t_u, half);
      Barrier();
      return mask;
    }
    return wave_masks ? BallotLow(cond)
                      : SelectB(cond, lane_masks ? Shl(U32(1), And(mask_lane_id, U32(31)))
                                                : U32(1), U32(0));
  }
  // Whether this invocation's lane is set in a wave mask.
  Id LaneActive(Id mask) {
    return InBlock((wave_masks || lane_masks)
               ? IsNonZero(And(Shr(mask, And(lane_masks ? mask_lane_id : WaveLane(), U32(31))),
                               U32(1)))
               : IsNonZero(mask));
  }
  Id InBlock(Id condition) {
    return block_active ? m.Emit(spv::Op::OpLogicalAnd, t_bool,
                                 {block_active, condition}) : condition;
  }
  void StorePrivate(Id pointer, Id value) {
    m.Store(pointer, block_active
                         ? SelectB(block_active, value, m.Load(t_u, pointer))
                         : value);
  }
  // A carry/borrow out: one bit per lane on the hardware, so it is a mask too.
  void SetLaneFlag(u32 sgpr, Id flag) {
    SetMask(sgpr, (wave_masks || lane_masks)
                      ? And(MaskOf(IsNonZero(flag)), Exec()) : flag);
  }
  Id LaneFlag(u32 sgpr) {
    return (wave_masks || lane_masks)
               ? SelectB(LaneActive(SgMask(sgpr)), U32(1), U32(0))
               : And(Sg(sgpr), U32(1));
  }

  // ---- cross-lane ------------------------------------------------------
  // A wave wider than the host subgroup cannot always use a subgroup shuffle;
  // compute also has a Workgroup-array channel (two barriers, uniform points
  // only; see uniform_here).
  Id lane_id = 0;    // this invocation's lane within its wave (0..63)
  u32 wave_size = 64;  // GCN uses 64; RDNA can request 32 at dispatch.
  Id wave_base = 0;  // LocalInvocationIndex of lane 0 of this wave
  Id xchg_var = 0;   // the exchange array, 2 slots per invocation
  u32 xchg_lanes = 0;  // invocations per workgroup (0 = no channel)
  Id xchg_index = 0;        // LocalInvocationIndex
  bool uniform_here = false;
  // This instruction's v_readfirstlane source is provably wave-uniform.
  bool readfirstlane_uniform = false;

  bool CanExchange() const { return xchg_var && uniform_here; }

  // Lane within the GCN wave; 0 where no lane index exists (graphics).
  Id WaveLane() { return lane_id ? lane_id : U32(0); }

  // Cross-wave publish/fetch via the Workgroup array; `slot` is one of the
  // two words a barrier pair can carry. Bracket with Barrier().
  void WavePublish(Id value, u32 slot = 0) {
    m.Store(XchgAt(Add(U32(slot * xchg_lanes), xchg_index)), value);
  }
  Id WaveFetch(Id src_lane, u32 slot = 0) {
    return m.Load(t_u, XchgAt(Add(U32(slot * xchg_lanes),
                                  Add(wave_base, And(src_lane, U32(wave_size - 1))))));
  }
  Id XchgAt(Id index) {
    return m.AccessChain(m.TypePointer(spv::StorageClass::Workgroup, t_u),
                         xchg_var, {index});
  }

  // The value lane `src_lane` of this wave holds, exactly.
  Id WaveExchange(Id value, Id src_lane) {
    WavePublish(value);
    Barrier();
    const Id r = WaveFetch(src_lane);
    Barrier();
    return r;
  }

  void Barrier() {
    m.EmitVoid(spv::Op::OpControlBarrier, {U32(2), U32(2), U32(0x108)});
  }

  // Exact within one subgroup; crossing the boundary needs WaveExchange.
  void RequireSubgroup(spv::Capability extra) {
    m.Capability(spv::Capability::GroupNonUniform);
    m.Capability(extra);
  }
  Id SubgroupShuffle(Id value, Id lane) {
    RequireSubgroup(spv::Capability::GroupNonUniformShuffle);
    // Shuffle is poison past the subgroup width; mask the index (folds to a constant).
    return m.Emit(spv::Op::OpGroupNonUniformShuffle, t_u,
                  {U32(3), value, And(lane, U32(HostSubgroupSize() - 1))});
  }

  void RequireImageQuery() {
    if (image_query)
      return;
    m.Capability(spv::Capability::ImageQuery);
    image_query = true;
  }

  // ---- SCC / EXEC / CFG-state ----
  Id Scc() { return m.Load(t_u, scc_var); }
  void SetScc(Id v) { StorePrivate(scc_var, v); }
  void SetSccBool(Id b) { SetScc(SelectB(b, U32(1), U32(0))); }
  Id Exec() { return SgMask(126); }
  Id State() { return m.Load(t_u, state_var); }
  void SetState(u32 s) { StorePrivate(state_var, U32(s)); }
  void SetStateId(Id s) { StorePrivate(state_var, s); }

  // ---- register file ----
  Id SgPtr(u32 i) { return m.AccessChain(p_priv_u, sgpr, {U32(i)}); }
  Id VgPtr(u32 i) { return m.AccessChain(p_priv_u, vgpr, {U32(i)}); }
  Id Sg(u32 i) {
    return rdna_sources && i == 125 ? U32(0) : m.Load(t_u, SgPtr(i));
  }
  Id Vg(u32 i) { return m.Load(t_u, VgPtr(i)); }
  void SetSg(u32 i, Id v) {
    if (!rdna_sources || i != 125)
      StorePrivate(SgPtr(i), v);
  }
  Id Sdst(u32 base, u32 offset = 0) {
    return rdna_sources && base == 125 ? U32(0) : Sg(base + offset);
  }
  void SetSdst(u32 base, u32 offset, Id v) {
    if (!rdna_sources || base != 125)
      SetSg(base + offset, v);
  }
  void SetVg(u32 i, Id v) {
    if (predicate_vector)
      v = SelectB(LaneActive(Exec()), v, Vg(i));
    StorePrivate(VgPtr(i), v);
  }
  Id VgF(u32 i) { return m.Bitcast(t_f, Vg(i)); }
  void SetVgF(u32 i, Id f) { SetVg(i, m.Bitcast(t_u, f)); }

  // ---- SGPR spill slots ------------------------------------------------
  // A scalar-starved shader parks scalars in VGPR lanes (v_writelane_b32);
  // both lane operands are wave-uniform by encoding, so a Private array per
  // spilled VGPR reproduces the lane file exactly, with no cross-lane channel.
  std::unordered_set<u32> spill_vgprs;  // pre-populated by PlanLaneSpills
  std::unordered_map<u32, Id> spill_vars;
  bool IsSpillVgpr(u32 v) const { return spill_vgprs.count(v) != 0; }
  Id SpillAt(u32 vgpr, Id lane) {
    Id& var = spill_vars[vgpr];
    if (!var) {
      const Id arr = m.TypeArray(t_u, 64);
      var = m.Variable(m.TypePointer(spv::StorageClass::Private, arr),
                       spv::StorageClass::Private, m.ConstNull(arr));
      m.Name(var, "lane_spill");
    }
    return m.AccessChain(p_priv_u, var, {And(lane, U32(63))});
  }

  // ---- constants ----
  Id U32(u32 v) { return m.ConstU32(v); }
  Id F32(float v) { return m.ConstF32(v); }

  // ---- float ALU ----
  Id Ext1(u32 op, Id a) { return m.ExtInst(t_f, op, {a}); }
  Id Ext2(u32 op, Id a, Id b) { return m.ExtInst(t_f, op, {a, b}); }
  Id FMul(Id a, Id b) { return m.Emit(spv::Op::OpFMul, t_f, {a, b}); }
  // V_*_LEGACY_F32: zero times anything is zero, even inf/NaN.
  Id LegacyMul(Id a, Id b) {
    const Id zero = F32(0.f);
    const Id any_zero = m.Emit(spv::Op::OpLogicalOr, t_bool,
                               {m.Emit(spv::Op::OpFOrdEqual, t_bool, {a, zero}),
                                m.Emit(spv::Op::OpFOrdEqual, t_bool, {b, zero})});
    return SelectF(any_zero, zero, FMul(a, b));
  }
  Id FAdd(Id a, Id b) { return m.Emit(spv::Op::OpFAdd, t_f, {a, b}); }
  Id FSub(Id a, Id b) { return m.Emit(spv::Op::OpFSub, t_f, {a, b}); }
  Id FDiv(Id a, Id b) { return m.Emit(spv::Op::OpFDiv, t_f, {a, b}); }
  Id FNeg(Id a) { return m.Emit(spv::Op::OpFNegate, t_f, {a}); }
  Id FClamp01(Id f) {
    return m.ExtInst(t_f, GLSLstd450FClamp, {f, F32(0.0f), F32(1.0f)});
  }

  // Saturating primitives behind the _clamp/_legacy transcendentals; NaN must
  // survive (select on OpIsInf, not GLSL min/max, which fold NaN).
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

  // ---- integer ALU (uint domain; signed bitcast; shifts mask to [4:0]) ----
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

  Id TypeV4u() { return m.TypeVec(t_u, 4); }
  Id True() { return IsNonZero(U32(1)); }

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
  // CB { uvec4 data[]; } at set 1, one binding per distinct V#.
  Id EnsureCbuf(u32 binding) {
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
  // Read cbuffer dword k; the uvec4 index clamps into the declared window.
  Id CbufDword(u32 binding, u32 k) {
    if (indirect_cbufs)
      return CbufDwordId(binding, U32(k));
    const Id var = EnsureCbuf(binding);
    const Id p_u = m.TypePointer(spv::StorageClass::Uniform, t_u);
    const Id ch = m.AccessChain(
        p_u, var,
        {U32(0), U32(std::min(k >> 2, kCbufDwords / 4 - 1)), U32(k & 3)});
    return m.Load(t_u, ch);
  }
  Id DrawDataDword(u32 index) {
    if (!cbuf_offsets) {
      const Id array = m.TypeArray(TypeV4u(), kIndirectDrawDwords / 4);
      m.Decorate(array, spv::Decoration::ArrayStride, {16});
      const Id block = m.TypeStruct({array});
      m.Decorate(block, spv::Decoration::Block);
      m.MemberDecorate(block, 0, spv::Decoration::Offset, {0});
      cbuf_offsets = m.Variable(m.TypePointer(spv::StorageClass::Uniform,
          block), spv::StorageClass::Uniform);
      m.Decorate(cbuf_offsets, spv::Decoration::DescriptorSet, {1});
      m.Decorate(cbuf_offsets, spv::Decoration::Binding, {1});
    }
    return m.Load(t_u, m.AccessChain(
        m.TypePointer(spv::StorageClass::Uniform, t_u), cbuf_offsets,
        {U32(0), U32(index / 4), U32(index % 4)}));
  }
  Id CbufDwordId(u32 binding, Id k) {
    if (indirect_cbufs) {
      if (!cbuf_ring) {
        cbuf_ring = m.Variable(m.TypePointer(spv::StorageClass::StorageBuffer,
            EnsureRawBlockType()), spv::StorageClass::StorageBuffer);
        m.Decorate(cbuf_ring, spv::Decoration::DescriptorSet, {1});
        m.Decorate(cbuf_ring, spv::Decoration::Binding, {0});
        m.Decorate(cbuf_ring, spv::Decoration::NonWritable);
      }
      const Id base = DrawDataDword(binding);
      const Id index = Add(base, Add(Mul(UMin(Shr(k, U32(2)),
          U32(kCbufDwords / 4 - 1)), U32(4)), And(k, U32(3))));
      return m.Load(t_u, m.AccessChain(
          m.TypePointer(spv::StorageClass::StorageBuffer, t_u), cbuf_ring,
          {U32(0), index}));
    }
    const Id var = EnsureCbuf(binding);
    const Id v4 = UMin(Shr(k, U32(2)), U32(kCbufDwords / 4 - 1));
    const Id p_u = m.TypePointer(spv::StorageClass::Uniform, t_u);
    const Id ch = m.AccessChain(p_u, var, {U32(0), v4, And(k, U32(3))});
    return m.Load(t_u, ch);
  }

  // ---- raw buffers (graphics MUBUF model) ----
  // Buf { uint data[]; } at set 2, one binding per distinct V# (per-lane
  // index, so storage rather than uniform).
  Id EnsureLdsBuffer() {
    if (lds_buf_var)
      return lds_buf_var;
    lds_buf_var = m.Variable(
        m.TypePointer(spv::StorageClass::StorageBuffer, EnsureRawBlockType()),
        spv::StorageClass::StorageBuffer);
    m.Decorate(lds_buf_var, spv::Decoration::DescriptorSet, {3});
    m.Decorate(lds_buf_var, spv::Decoration::Binding, {0});
    m.Name(lds_buf_var, "lds");
    return lds_buf_var;
  }

  // Shared by every storage buffer; the builder dedups type ids, so a second
  // decorated copy would decorate the same id twice and fail spirv-val.
  Id EnsureRawBlockType() {
    if (!gfx_buf_type) {
      const Id run = m.TypeRuntimeArray(t_u);
      m.Decorate(run, spv::Decoration::ArrayStride, {4});
      gfx_buf_type = m.TypeStruct({run});
      m.Decorate(gfx_buf_type, spv::Decoration::Block);
      m.MemberDecorate(gfx_buf_type, 0, spv::Decoration::Offset, {0});
    }
    return gfx_buf_type;
  }

  Id EnsureGfxBuffer(u32 binding) {
    auto it = gfx_buf_vars.find(binding);
    if (it != gfx_buf_vars.end())
      return it->second;
    const Id v = m.Variable(
        m.TypePointer(spv::StorageClass::StorageBuffer, EnsureRawBlockType()),
        spv::StorageClass::StorageBuffer);
    m.Decorate(v, spv::Decoration::DescriptorSet, {2});
    m.Decorate(v, spv::Decoration::Binding, {binding});
    m.Name(v, "rawbuf" + std::to_string(binding));
    gfx_buf_vars[binding] = v;
    return v;
  }

  // ---- operand sources ----
  // Raw uint of a source operand field (SSRC/VSRC encoding).
  Id SrcRaw(u32 field, u32 literal) {
    if (field <= 127)
      return Sg(field);
    if (field == 128)
      return U32(0);
    if (field >= 129 && field <= 192)
      return U32(field - 128);
    if (field >= 193 && field <= 208)
      return U32(static_cast<u32>(-static_cast<int>(field - 192)));
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
        return SelectB(IsZero(full_wave_masks ? Or(Sg(106), Sg(107))
                                             : SgMask(106)), U32(1), U32(0));
      case 252:
        return SelectB(IsZero(full_wave_masks ? Or(Sg(126), Sg(127))
                                             : Exec()), U32(1), U32(0));
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
  Id SrcRawHi(u32 field, u32 literal, bool sign_extend) {
    if (rdna_sources && field == 125)
      return U32(0);
    if (field <= 126)
      return Sg(field + 1);
    if (field >= 256 && field <= 510)
      return Vg(field - 255);
    const Id lo = SrcRaw(field, literal);
    // Inline integer constants -1..-16 fill the high dword of a 64-bit operand too.
    if (field >= 128 && field <= 208)
      return Sar(lo, U32(31));
    return sign_extend ? Sar(lo, U32(31)) : U32(0);
  }
  // Float source with the VOP3 neg/abs input modifiers applied.
  Id SrcF(u32 field,
          u32 literal,
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

// Seed the system values SPI_PS_INPUT_ENA lays into initial VGPRs (ascending
// bit order); interpolated parameters come through as Location inputs.
inline void SeedPsInputVgprs(Translator& t,
                             u32 ena,
                             std::vector<Id>& iface) {
  if (!ena)
    return;
  static constexpr u8 width[16] = {2, 2, 2, 3, 2, 2, 2, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1};
  u32 vg[16] = {}, next = 0;
  for (u32 bit = 0; bit < 16; bit++)
    if (ena & (1u << bit)) {
      vg[bit] = next;
      next += width[bit];
    }

  if ((ena >> 8) & 0xF) {
    const Id frag_coord =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_v4),
                     spv::StorageClass::Input);
    t.m.Decorate(frag_coord, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::FragCoord)});
    iface.push_back(frag_coord);
    const Id value = t.m.Load(t.t_v4, frag_coord);
    for (u32 component = 0; component < 4; component++)
      if (ena & (1u << (8 + component)))
        t.SetVgF(vg[8 + component],
                 t.m.CompositeExtract(t.t_f, value, component));
  }

  if (ena & (1u << 12)) {
    const Id front_facing =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_bool),
                     spv::StorageClass::Input);
    t.m.Decorate(front_facing, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::FrontFacing)});
    iface.push_back(front_facing);
    t.SetVg(vg[12],
            t.SelectB(t.m.Load(t.t_bool, front_facing), t.U32(0xFFFFFFFFu),
                      t.U32(0)));
  }
}

// Per-stage state carried into the shared per-instruction emitter (EmitInst).
// One channel of a copy-shader export: GSVS ring component `comp`, or the
// constant `bits` when comp < 0.
struct GsCopyExport {
  u32 target = 0, chan = 0;
  int comp = -1;
  u32 bits = 0;
};

struct StageContext {
  bool is_ps = false;
  // SPI_PS_INPUT_CNTL: VS param each PS attr slot reads; null = attr_i -> param_i.
  const u32* ps_in_cntl = nullptr;
  // NUM_INTERP: slots at or above it are don't-care and read 0.
  u32 ps_num_interp = 0;
  // VS param exports that exist; OFFSET indexes the export cache, not always
  // the param number.
  const std::vector<u32>* vs_exported_params = nullptr;
  bool is_cs = false;
  // PS5 compute whose 64-lane wave spans two host subgroups: every wave runs
  // the same block at once (the mesh scheduler), and LDS write/read turns get
  // the barrier the guest never needed.
  bool wave_lockstep = false;
  bool lds_sync = false;
  Id cs_probe_var = 0;  // DELTA_GPU_CSVGPR snapshot
  Recompiled* r = nullptr;
  std::vector<Id>* iface = nullptr;
  Id main_fn = 0;  // entry function (for stage-wide ExecMode additions)

  // VS
  bool is_mesh = false;
  Id mesh_counts = 0;     // Workgroup {vertex count, primitive count}
  Id mesh_primitive = 0;  // Private packed NGG connectivity
  Id mesh_local_index = 0;
  Id pos_out = 0;
  std::unordered_map<u32, Id> param_outs;
  std::unordered_set<u32>
      direct_vfetch;  // MUBUF pc seeded as vertex input
  u32 max_param = 0;
  // PS5 inline vertex fetch: (input, first dest VGPR, comps) per fetch pc;
  // reseeded at the fetch because NGG index math can overwrite the VGPRs.
  struct VfetchSeed {
    Id in_var;
    u32 dest_vgpr, num_comps;
  };
  std::unordered_map<u32, VfetchSeed> vfetch_seed;
  // DELTA_GPU_DBGPOS: position input + cbuf transform, to split bad lifted
  // math from bad input values.
  Id dbg_pos_in = 0;
  u32 dbg_pos_comps = 0;
  int dbg_pos_cbuf = -1;
  u32 dbg_pos_dword = 0;
  int dbg_pos_world = -1;

  // PS
  Id color_outs[8] = {};  // lazily declared per MRT target (location == target)
  // Bit n: MRT n is integer format (output uvec4); tex_uint_mask says the
  // same for sampler bindings.
  u32 mrt_uint_mask = 0;
  // Bit n: the pass binds colour attachment n; unbound exports write nowhere.
  u32 mrt_bound_mask = 0xFF;
  u64 tex_uint_mask = 0;
  Id depth_out = 0;       // MRTZ -> FragDepth (lazily declared)
  std::unordered_map<u32, Id> in_vars;
  bool wrote_color = false;  // compile-time: shader has a color export
  Id color_written_var = 0;  // runtime: this fragment reached a color export
  const std::unordered_set<u32>* flat_attrs = nullptr;
  // MIMGs sharing a descriptor share one set-0 binding; TrackTextures pairs
  // against this plan at draw time.
  const MimgBindingPlan* mimg_plan = nullptr;
  // Set 0 is shared: VS samplers number after the PS's; tex_vars stays stage-local.
  u32 tex_binding_base = 0;
  static constexpr u32 kMaxPsSamplers = 64;  // == gpu::vk::kMaxTex
  Id tex_vars[kMaxPsSamplers] = {};
  u32 tex_types[kMaxPsSamplers] = {};
  // Bit i: binding i's T# is SQ_RSRC_IMG_3D; invisible in MIMG (DA 0), so
  // caller-supplied and cache-keyed.
  u64 tex_3d_mask = 0;
  // Bit i: binding i's T# is 1D[_ARRAY]; bound as height-1 2D, y at row centre.
  u64 tex_1d_mask = 0;

  // shared graphics
  std::unordered_map<u32, u32> cbuf_bind;  // V# SGPR -> set-1 binding
  // Raw MUBUF: pc -> set-2 binding (per pc, not SGPR: one quad can hold
  // several descriptors over a shader's life).
  std::unordered_map<u32, u32> gfx_buf_bind;
  // Per-pc cbuf bindings for reused srsrc SGPRs (PS5 table chains); beats cbuf_bind.
  std::unordered_map<u32, u32> mubuf_cbuf_by_pc;
  // Per-instruction RDNA SMEM binding when one sbase has multiple producers.
  std::unordered_map<u32, u32> smem_cbuf_by_pc;
  // set-1 binding -> staged window start (ShaderCbuf::first_dword); absent = 0.
  std::unordered_map<u32, u32> cbuf_first_dword;
  // pcs of `s_mov exec, sN` with unmodelled launch state; emitting would zero
  // EXEC and skip every export.
  std::unordered_set<u32> skip_launch_movs;

  // Compute: storage buffers modelling the guest memory the CS reads/writes.
  std::unordered_map<u32, u32> cs_bind;  // instruction pc -> binding
  u32 cs_cur_pc = 0;                     // instruction being emitted
  std::vector<Id> cs_ssbo;                         // binding -> SSBO variable
  // Push constants {user_data[16], bounds[64]}: SSBO access clamps to the
  // per-binding bounds, as the hardware drops out-of-bound stores.
  Id cs_bounds_var = 0;
  Id cs_guest_table = 0, cs_guest_translate = 0;
  std::unordered_map<u32, std::pair<u32, u32>> cs_runtime_resources; // binding -> kind, SGPR
  std::unordered_set<u32> cs_runtime_images;

  // Attrs v_interp_mov_f32 reads as P10/P20 (per-vertex deltas); their whole
  // Location becomes a PerVertexKHR array[3] recomputed from BaryCoordKHR.
  std::unordered_set<u32> pervertex_attrs;
  std::unordered_map<u32, Id> pervertex_vars;
  Id bary_var = 0;  // BaryCoordKHR, declared on first use
  Id bary_nopersp_var = 0;  // BaryCoordNoPerspKHR, likewise
  // DELTA_GPU_PSVGPR_BLOCK: a saved copy of the probed VGPR, taken at
  // the top of one CFG block instead of at the export.
  Id vgpr_snap_var = 0;

  Id gds_var = 0;   // GDS counters (ds_append / ds_consume), a storage buffer
  Id lds_var = 0;           // uint array backing LDS (0 = no LDS)
  u32 lds_dwords = 0;  // its length
  // Workgroup in a CS; graphics backs LDS with Private (exact only when every
  // address is the lane's own slot, e.g. v_mbcnt with an explicit all-ones
  // mask, not mbcnt-on-EXEC compaction indices).
  spv::StorageClass lds_storage = spv::StorageClass::Workgroup;
  // Nonzero = shared-LDS path instead: set-3 buffer, one block per wave, which
  // an NGG vertex program needs (it reads slots its own lane never wrote).
  Id lds_wave_base = 0;  // dword index of this wave's block (0 = Private LDS)
  // Shared LDS keys its block by a wave-agreed value; the vertex index is the
  // only one a VS has.
  bool is_vs_shared_lds_capable = false;
  Id vertex_index_value = 0;
  std::unordered_set<u32> ds_own_lane;
  Id subgroup_local_id = 0;     // SubgroupLocalInvocationId for DS swizzles
  // Sorted instruction indices needing a barrier the guest omitted. See PlanLdsBarriers.
  std::vector<u32> lds_barrier_at;
  // Per instruction: v_readfirstlane source proven wave-uniform.
  std::vector<u8> uniform_readfirstlane;
  // Per instruction: may the group be synchronised there? See UniformPoints.
  std::vector<u8> uniform_points;
  // Barrier once per dispatch-loop iteration (see LdsBarrierPlan).
  bool lockstep_loop = false;
  bool cs_unsupported = false;  // op the compute backend can't model

  // Legacy ES -> GS pipeline (translate_gs.cc). ES: es_ring is the uvec4 output
  // array its ring stores land in. GS: es_ring is the per-vertex input array,
  // gsvs the Private copy of the GSVS ring, and each emit writes one vertex
  // through the copy shader's export map.
  Id es_ring = 0;
  u32 es_ring_vec4s = 0;
  u32 gs_input_verts = 0;
  Id gsvs = 0;
  u32 gsvs_dwords = 0;
  Id gs_emitted = 0;
  u32 gs_max_vert_out = 0;
  const std::vector<GsCopyExport>* gs_exports = nullptr;
  Id layer_out = 0;
  bool gl_clip = false;
};

// ---- stage-io helpers (gcn_spirv.cc) --------------------------------------
Id PsInputVar(Translator& t, StageContext& sc, u32 attr);
// Locations a reachable v_interp_mov_f32 reads as P10/P20 become PerVertexKHR
// arrays; settled before the first read.
std::unordered_set<u32> PlanPerVertexAttrs(const Program& program,
                                           const u8* reachable);
void EmitVintrp(Translator& t, u32 w, StageContext& sc);
Id VsParamOut(Translator& t, StageContext& sc, u32 p);
Id PsColorOut(Translator& t, StageContext& sc, u32 target);
Id PsDepthOut(Translator& t, StageContext& sc);
// Seed the barycentric I/J VGPRs (bottom of the file per SPI_PS_INPUT_ENA)
// from BaryCoordKHR, which SeedPsInputVgprs cannot supply.
void SeedPsBarycentrics(Translator& t, u32 ena, StageContext& sc);

// ---- ALU emitters (translate_alu.cc) --------------------------------------
void EmitSop1(Translator& t, const Inst& inst);
void EmitSop2(Translator& t, const Inst& inst);
void EmitSopc(Translator& t, const Inst& inst);
void EmitSopk(Translator& t, const Inst& inst);
void EmitVop1(Translator& t,
              u32 op,
              u32 vdst,
              Id s0,
              bool clamp = false,
              u32 omod = 0);
void EmitVop2(Translator& t,
              u32 op,
              u32 vdst,
              Id s0,
              Id s1,
              u32 literal = 0,
              bool clamp = false,
              u32 omod = 0);
void EmitVop3(Translator& t,
              u32 op,
              u32 vdst,
              Id s0,
              Id s0_hi,
              Id s1,
              Id s2,
              Id s2_hi,
              u32 sdst,
              bool clamp,
              u32 omod = 0,
              Id s1_hi = 0);
// Vector compare: 0/1 predicate to sgpr[dst], cmpx also replaces EXEC;
// s0_hi/s1_hi carry the high dwords of 64-bit operand pairs.
void EmitVopc(Translator& t,
              u32 op,
              Id s0f,
              Id s1f,
              Id s0u,
              Id s1u,
              u32 dst = 106,
              Id s0_hi = 0,
              Id s1_hi = 0);
bool IsVop3b(u32 op);
// Every v_writelane_b32 destination (incl. VOP3). See Translator::spill_vgprs.
std::unordered_set<u32> PlanLaneSpills(const Program& program,
                                            const u8* reachable = nullptr);
// v_readlane/writelane against a spill VGPR, exact in every stage; false
// leaves the general cross-lane lowering.
bool EmitLaneSpill(Translator& t,
                   u32 op,
                   u32 dst,
                   u32 src0,
                   u32 src1,
                   u32 literal);
bool EmitNeoVop1(Translator& t, const Inst& inst);
bool EmitNeoVop2(Translator& t, const Inst& inst);
bool EmitNeoVopc(Translator& t,
                 u32 op,
                 u32 dst,
                 u32 src0,
                 u32 src1,
                 u32 literal,
                 bool src0_high = false,
                 bool src1_high = false,
                 bool src0_neg = false,
                 bool src1_neg = false,
                 bool src0_abs = false,
                 bool src1_abs = false);
bool EmitNeoVop3(Translator& t, const Inst& inst);
bool EmitNeoVop3p(Translator& t, const Inst& inst);

// ---- memory emitters (translate_mem.cc) -----------------------------------
u32 SmrdLoadCount(u32 op);
u32 SmrdDwordCount(u32 op);
bool PlanCbufs(const Program& program,
               u32 first_binding,
               std::vector<ShaderCbuf>& cbufs,
               std::unordered_map<u32, u32>& bindings,
               const u8* reachable = nullptr);
void EmitCbufSmrd(Translator& t,
                  const Inst& inst,
                  const std::unordered_map<u32, u32>& bindings);
// True for MIMG ops stating their own LOD (legal outside fragment shaders).
bool MimgNamesItsLod(u32 op);
// DS ops legal on Private-backed LDS: plain loads/stores only (no atomics on
// a Private pointer).
bool DsGraphicsSupported(u32 op);
// Dwords of Private LDS a graphics program needs, static from its DS
// addresses (M0 0x10000 means "unrestricted", not an allocation).
u32 GraphicsLdsDwords(const Program& program, const u8* reachable);
// DS instructions whose address is provably the lane's own slot.
std::unordered_set<u32> PlanDsOwnLane(const Program& program,
                                      const u8* reachable);
// Declare SubgroupLocalInvocationId + shuffle capability (ds_swizzle/DPP channel).
void EnableDsSwizzle(Translator& t, StageContext& sc, std::vector<Id>& iface);
bool UsesDsSwizzle(const Program& program, const u8* reachable);
void EmitMimg(Translator& t,
              const Inst& inst,
              StageContext& sc,
              const Id* address = nullptr);
// Set-2 binding per raw MUBUF/MTBUF load not already claimed as a vertex
// input; sharers share a binding, past the cap warns at emit time.
void PlanGfxBuffers(const Program& program,
                    u32 first_binding,
                    const std::unordered_set<u32>* claimed,
                    std::vector<ShaderBuffer>& buffers,
                    std::unordered_map<u32, u32>& bindings,
                    const u8* reachable = nullptr);
void EmitGfxMubuf(Translator& t, const Inst& inst, StageContext& sc);
// Typed buffer op in a graphics stage; loads share the set-2 window, stores
// are dropped (the window is not written back).
void EmitGfxMtbuf(Translator& t, const Inst& inst, StageContext& sc);
bool PlanCsResources(const Program& program,
                     const u8* reachable,
                     u32 lds_dwords,
                     RecompiledCs& r,
                     std::unordered_map<u32, u32>& bind);
void EmitCsSmrd(Translator& t, const Inst& inst, StageContext& sc);
void EmitCsMubuf(Translator& t, const Inst& inst, StageContext& sc);
void EmitCsGlobal(Translator& t, const Inst& inst, StageContext& sc);
void EmitGdsCounter(Translator& t, const Inst& inst, StageContext& sc);
void EmitCsMtbuf(Translator& t, const Inst& inst, StageContext& sc);
void EmitCsMimg(Translator& t,
                const Inst& inst,
                StageContext& sc,
                const Id* address = nullptr);
void EmitDs(Translator& t, const Inst& inst, StageContext& sc);

// Compute resource model: a guest range aliased as Buf { uint data[]; },
// addressed by dword index; ISA-neutral, so RDNA2 binds the same buffers.
Id CsSsboPtr(Translator& t, StageContext& sc, u32 binding, Id dword_idx);
Id CsSsboLoad(Translator& t, StageContext& sc, u32 binding, Id dword_idx);
Id CsSsboBound(Translator& t, StageContext& sc, u32 binding);
void DeclareGuestMemory(Translator& t, StageContext& sc, u32 binding);
Id CsGuestBase(Translator& t, StageContext& sc, u32 binding);
Id CsGuestAddress(Translator& t, StageContext& sc, Id address, Id bytes, bool write = false);
void CsGuestStore(Translator& t, StageContext& sc, u32 binding, Id dword_idx, Id value);
void EmitGuestGlobal(Translator& t, const Inst& inst, StageContext& sc);
Id CsGuestLoad(Translator& t, StageContext& sc, u32 binding, Id dword_idx,
               Id required_end = 0);
Id CsPhysicalLoad(Translator& t, Id address);
void CsGuestStoreMasked(Translator& t, StageContext& sc, Id address, Id value, Id mask);
Id CsLinearImageEligible(Translator& t, const Inst& inst);
void EmitCsLinearImage(Translator& t, const Inst& inst, StageContext& sc,
                       const Id* address);

void CsSsboStore(Translator& t,
                 StageContext& sc,
                 u32 binding,
                 Id dword_idx,
                 Id value);
// Storage-buffer binding planned for the instruction at pc, or -1.
int CsBindingFor(StageContext& sc, u32 pc);

// ---- legacy geometry shaders (translate_gs.cc) ----------------------------
// Which GSVS component each export channel of a copy shader carries; false for
// a copy shader this model cannot follow.
bool ParseCopyShader(const Program& program,
                     u32 max_vert_out,
                     std::vector<GsCopyExport>& out);
void EmitEsRingStore(Translator& t, const Inst& inst, StageContext& sc);
void EmitGsRingAccess(Translator& t, const Inst& inst, StageContext& sc);
// s_sendmsg in a GS: emit and cut become EmitVertex / EndPrimitive.
void EmitGsMessage(Translator& t, const Inst& inst, StageContext& sc);

// RECTLIST expansion: three post-VS corners in, two triangles out; shared
// with the RDNA2 path.
std::vector<u32> EmitRectListGeometry(
    u32 num_params,
    const std::unordered_set<u32>& flat_attrs);

}  // namespace gpu::gcn
