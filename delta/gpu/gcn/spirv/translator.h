#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"

#include <spirv/unified1/GLSL.std.450.h>

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/pair.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/strings/to_string.h"
#include "base/strings/xstring.h"
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

void WarnUnsupported(const char* enc, u32 op, u32 w0 = 0, u32 w1 = 0);
void NoteApproximated(const char* enc, u32 op);
void ResetUnsupported();
bool HadUnsupported();
const base::String& UnsupportedOps();

bool TraceEnabled();

inline u32 RelativeVgprFloor(
    const Program& program,
    const base::HashMap<u32, base::Vector<u32>>& known = {}) {
  u32 lo = 256;
  for (const Inst& inst : program) {
    if (inst.enc != Enc::kVop1 || known.count(inst.pc))
      continue;
    const u32 w = inst.raw[0], op = inst.opcode;
    const u32 vdst = (w >> 17) & 0xFF, src0 = w & 0x1FF;
    if (op == 0x42 || op == 0x44)
      lo = base::Min(lo, vdst);
    if ((op == 0x43 || op == 0x44) && src0 >= 256)
      lo = base::Min(lo, src0 - 256);
  }
  return lo;
}

inline void BoundMovrels(const Program& program,
                         base::HashMap<u32, base::Vector<u32>>& known) {
  u32 lo = 256;
  for (const Inst& inst : program) {
    if (inst.enc != Enc::kVop1 || inst.opcode < 0x42 || inst.opcode > 0x44 ||
        known.count(inst.pc))
      continue;
    const u32 w = inst.raw[0];
    if (inst.opcode != 0x43)
      lo = base::Min(lo, (w >> 17) & 0xFF);
    if (inst.opcode != 0x42 && (w & 0x1FF) >= 256)
      lo = base::Min(lo, (w & 0x1FF) - 256);
  }
  if (lo == 256)
    return;
  base::Vector<u32> values;
  for (u32 v = 0; lo + v < 256; v++)
    values.push_back(v);
  for (const Inst& inst : program)
    if (inst.enc == Enc::kVop1 && inst.opcode >= 0x42 && inst.opcode <= 0x44 &&
        !known.count(inst.pc))
      known[inst.pc] = values;
}

struct Translator {
  spirv::Module m;
  Id t_void = 0, t_fn = 0, t_u = 0, t_i = 0, t_f = 0, t_bool = 0;
  Id t_v2 = 0, t_v3 = 0, t_v4 = 0;
  Id p_priv_u = 0, sgpr = 0, vgpr = 0;
  u32 vgpr_dyn_lo = 0;
  Id vgpr_regs[256] = {};
  base::HashMap<u32, base::Vector<u32>> movrel_m0;
  bool predicate_vector = false;
  Id vector_write_guard = 0;
  Id block_active = 0;
  bool rdna_sources = false;
  u64 program_base = 0;
  Id pc_base_var = 0;
  u32 pc_base_member = 0;
  Id scc_var = 0;
  Id state_var = 0;
  Id cbuf_type = 0;
  bool indirect_cbufs = false;
  u32 tex_cube_mask = 0;
  u32 user_data_slot = 0;
  Id cbuf_ring = 0, cbuf_offsets = 0;
  base::HashMap<u32, Id> cbuf_vars;
  Id gfx_buf_type = 0;
  Id lds_buf_var = 0;
  base::HashMap<u32, Id> gfx_buf_vars;
  Id img_types[16] = {};
  Id sampled_types[16] = {};
  Id sampled_ptrs[16] = {};
  Id storage_img_types[2] = {};
  Id storage_img_ptrs[2] = {};
  bool image_query = false;
  Id last_texel = 0;
  Id last_texel_var = 0;
  Id probe_var = 0;
  Id dbg_file = 0;

  void InitTypes(const Program* program = nullptr,
                 const Program* second = nullptr) {
    if (program && !second)
      BoundMovrels(*program, movrel_m0);
    vgpr_dyn_lo = program ? RelativeVgprFloor(*program, movrel_m0) : 0u;
    if (second)
      vgpr_dyn_lo = base::Min(vgpr_dyn_lo, RelativeVgprFloor(*second));
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

  void SeedExec() {
    SetSg(126, lane_masks ? U32(~0u) : wave_masks ? BallotLow(True()) : U32(1));
    if (lane_masks)
      SetSg(127, U32(~0u));
  }

  bool wave_masks = false;
  bool lane_masks = false;
  bool full_wave_masks = false;
  Id mask_lane_id = 0;
  Id MaskLaneHigh() { return IsNonZero(And(mask_lane_id, U32(32))); }
  Id MaskWord(Id lo, Id hi) {
    if (dual)
      return half ? hi : lo;
    return lane_masks ? SelectB(MaskLaneHigh(), hi, lo) : lo;
  }
  // Whole-wave view of a mask register, for scalar tests such as EXECZ.
  Id WaveMask(u32 sgpr) {
    return full_wave_masks || dual ? Or(Sg(sgpr), Sg(sgpr + 1)) : SgMask(sgpr);
  }

  // Dual-lane mode: an invocation runs lanes `sid` and `sid + 32` of a wave64
  // and one subgroup holds the wave, so SGPRs and masks stay wave-uniform.
  // Vector instructions are emitted once per half, each half with its own
  // VGPRs and mask word; the SGPRs they write are held until both are done.
  bool dual = false;
  u32 half = 0;
  Id half_lane[2] = {};
  Id other_vgpr = 0;
  Id other_vgpr_regs[256] = {};
  bool hold_scalar_writes = false;
  struct HeldWrite {
    u32 sgpr;
    Id value;
  };
  base::Vector<HeldWrite> held_writes;
  Id first_half_pick = 0;
  void SetHalf(u32 h) {
    if (h == half)
      return;
    const Id v = vgpr;
    vgpr = other_vgpr;
    other_vgpr = v;
    for (u32 i = 0; i < 256; i++) {
      const Id r = vgpr_regs[i];
      vgpr_regs[i] = other_vgpr_regs[i];
      other_vgpr_regs[i] = r;
    }
    for (u32 i = kRegVgpr; i < kRegScc; i++)
      known_regs[i] = {};
    exec_memo = {};
    lane_active_memo = {};
    half = h;
    lane_id = mask_lane_id = half_lane[h];
  }
  Id VgHalf(u32 i, u32 h) {
    const u32 was = half;
    SetHalf(h);
    const Id v = Vg(i);
    SetHalf(was);
    return v;
  }
  // `candidate` is this half's reading of `lane`; returns the reading from
  // the half that owns the lane. Called once per half, low half first.
  Id PickHalf(Id candidate, Id lane) {
    if (half == 0) {
      first_half_pick = candidate;
      return candidate;
    }
    return SelectB(Uge(lane, U32(32)), candidate, first_half_pick);
  }
  void FlushHeldWrites() {
    hold_scalar_writes = false;
    for (const HeldWrite& w : held_writes)
      SetSg(w.sgpr, w.value);
    held_writes.clear();
  }
  Id SgMask(u32 sgpr) { return MaskWord(Sg(sgpr), Sg(sgpr + 1)); }

  static constexpr u32 kRegVgpr = 136, kRegScc = kRegVgpr + 256,
                       kRegState = kRegScc + 1, kRegCount = kRegState + 1;
  struct KnownValue {
    Id block = 0, value = 0;
  };
  KnownValue known_regs[kRegCount];
  Id KnownReg(u32 key) const {
    const KnownValue& k = known_regs[key];
    return k.block && k.block == m.CurrentBlock() ? k.value : 0;
  }
  void KnowReg(u32 key, Id value) {
    known_regs[key] = {m.CurrentBlock(), value};
  }
  Id LoadReg(u32 key, Id pointer) {
    if (const Id v = KnownReg(key))
      return v;
    const Id v = m.Load(t_u, pointer);
    KnowReg(key, v);
    return v;
  }
  void StoreReg(u32 key, Id pointer, Id value) {
    if (block_active)
      value = SelectB(block_active, value, LoadReg(key, pointer));
    m.Store(pointer, value);
    KnowReg(key, value);
  }
  bool CachedVgpr(u32 i) const { return i < vgpr_dyn_lo && i < 256; }

  struct Memo {
    Id block = 0, a = 0, b = 0, value = 0;
  };
  template <typename Make>
  Id Memoized(Memo& memo, Id a, Id b, Make make) {
    const Id block = m.CurrentBlock();
    if (block && memo.block == block && memo.a == a && memo.b == b)
      return memo.value;
    const Id value = make();
    memo = {block, a, b, value};
    return value;
  }
  Memo exec_memo, lane_active_memo;
  void SetMask(u32 sgpr, Id value) {
    if (dual) {
      if (!rdna_sources || sgpr != 125)
        SetSg(sgpr + half, value);
      return;
    }
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
    return wave_masks
               ? BallotLow(cond)
               : SelectB(cond,
                         lane_masks ? Shl(U32(1), And(mask_lane_id, U32(31)))
                                    : U32(1),
                         U32(0));
  }
  Id LaneActive(Id mask) {
    return Memoized(lane_active_memo, mask, block_active, [&] {
      if (!wave_masks && !lane_masks)
        return InBlock(IsNonZero(mask));
      const Id lane_bit = IsNonZero(
          And(Shr(mask, And(lane_masks ? mask_lane_id : WaveLane(), U32(31))),
              U32(1)));
      return InBlock(
          m.Emit(spv::Op::OpLogicalOr, t_bool, {Eq(mask, U32(~0u)), lane_bit}));
    });
  }
  Id InBlock(Id condition) {
    return block_active ? m.Emit(spv::Op::OpLogicalAnd, t_bool,
                                 {block_active, condition})
                        : condition;
  }
  void StorePrivate(Id pointer, Id value) {
    for (u32 i = kRegVgpr; i < kRegScc; i++)
      known_regs[i] = {};
    StoreUncached(pointer, value);
  }
  void StoreUncached(Id pointer, Id value) {
    m.Store(pointer, block_active
                         ? SelectB(block_active, value, m.Load(t_u, pointer))
                         : value);
  }
  void SetLaneFlag(u32 sgpr, Id flag) {
    SetMask(sgpr, (wave_masks || lane_masks)
                      ? And(MaskOf(IsNonZero(flag)), Exec())
                      : flag);
  }
  Id LaneFlag(u32 sgpr) {
    return (wave_masks || lane_masks)
               ? SelectB(LaneActive(SgMask(sgpr)), U32(1), U32(0))
               : And(Sg(sgpr), U32(1));
  }

  Id lane_id = 0;
  u32 wave_size = 64;
  Id wave_base = 0;
  Id xchg_var = 0;
  u32 xchg_lanes = 0;
  Id xchg_index = 0;
  bool uniform_here = false;
  bool readfirstlane_uniform = false;

  bool CanExchange() const { return xchg_var && uniform_here; }

  Id WaveLane() { return lane_id ? lane_id : U32(0); }

  void WavePublish(Id value, u32 slot = 0) {
    m.Store(XchgAt(Add(U32(slot * xchg_lanes), xchg_index)), value);
  }
  Id WaveFetch(Id src_lane, u32 slot = 0) {
    return m.Load(
        t_u, XchgAt(Add(U32(slot * xchg_lanes),
                        Add(wave_base, And(src_lane, U32(wave_size - 1))))));
  }
  Id XchgAt(Id index) {
    return m.AccessChain(m.TypePointer(spv::StorageClass::Workgroup, t_u),
                         xchg_var, {index});
  }

  Id WaveExchange(Id value, Id src_lane) {
    WavePublish(value);
    Barrier();
    const Id r = WaveFetch(src_lane);
    Barrier();
    return r;
  }

  // AcquireRelease | WorkgroupMemory, plus Uniform and Image memory when a
  // split wave orders guest memory as well as LDS.
  static constexpr u32 kBarrierLdsOnly = 0x108, kBarrierWithMemory = 0x948;
  u32 barrier_semantics = kBarrierLdsOnly;
  void Barrier() {
    const u32 scope = static_cast<u32>(dual ? spv::Scope::Subgroup
                                            : spv::Scope::Workgroup);
    m.EmitVoid(spv::Op::OpControlBarrier,
               {U32(scope), U32(scope), U32(barrier_semantics)});
  }

  void RequireSubgroup(spv::Capability extra) {
    m.Capability(spv::Capability::GroupNonUniform);
    m.Capability(extra);
  }
  Id SubgroupShuffle(Id value, Id lane) {
    RequireSubgroup(spv::Capability::GroupNonUniformShuffle);
    return m.Emit(spv::Op::OpGroupNonUniformShuffle, t_u,
                  {U32(3), value, And(lane, U32(HostSubgroupSize() - 1))});
  }

  void RequireImageQuery() {
    if (image_query)
      return;
    m.Capability(spv::Capability::ImageQuery);
    image_query = true;
  }

  Id Scc() { return LoadReg(kRegScc, scc_var); }
  void SetScc(Id v) { StoreReg(kRegScc, scc_var, v); }
  void SetSccBool(Id b) { SetScc(SelectB(b, U32(1), U32(0))); }
  Id Exec() {
    const Id lo = Sg(126), hi = Sg(127);
    return Memoized(exec_memo, lo, hi, [&] { return MaskWord(lo, hi); });
  }
  Id State() { return LoadReg(kRegState, state_var); }
  void SetState(u32 s) { StoreReg(kRegState, state_var, U32(s)); }
  void SetStateId(Id s) { StoreReg(kRegState, state_var, s); }

  Id SgPtr(u32 i) { return m.AccessChain(p_priv_u, sgpr, {U32(i)}); }
  Id VgPtr(u32 i) {
    if (i >= vgpr_dyn_lo || i >= 256)
      return m.AccessChain(p_priv_u, vgpr, {U32(i)});
    if (!vgpr_regs[i])
      vgpr_regs[i] =
          m.Variable(p_priv_u, spv::StorageClass::Private, m.ConstNull(t_u));
    return vgpr_regs[i];
  }
  Id Sg(u32 i) {
    if (rdna_sources && i == 125)
      return U32(0);
    if (const Id v = KnownReg(i))
      return v;
    return LoadReg(i, SgPtr(i));
  }
  Id Vg(u32 i) {
    if (!CachedVgpr(i))
      return m.Load(t_u, VgPtr(i));
    if (const Id v = KnownReg(kRegVgpr + i))
      return v;
    return LoadReg(kRegVgpr + i, VgPtr(i));
  }
  void SetSg(u32 i, Id v) {
    if (hold_scalar_writes) {
      held_writes.push_back({i, v});
      return;
    }
    if (!rdna_sources || i != 125)
      StoreReg(i, SgPtr(i), v);
  }
  Id Sdst(u32 base, u32 offset = 0) {
    return rdna_sources && base == 125 ? U32(0) : Sg(base + offset);
  }
  void SetSdst(u32 base, u32 offset, Id v) {
    if (!rdna_sources || base != 125)
      SetSg(base + offset, v);
  }
  void SetVg(u32 i, Id v) {
    if (vector_write_guard)
      v = SelectB(vector_write_guard, v, Vg(i));
    if (predicate_vector)
      v = SelectB(LaneActive(Exec()), v, Vg(i));
    if (CachedVgpr(i))
      StoreReg(kRegVgpr + i, VgPtr(i), v);
    else
      StoreUncached(VgPtr(i), v);
  }
  Id VgRelPtr(u32 i) {
    return m.AccessChain(p_priv_u, vgpr,
                         {UMin(Add(U32(i), Sg(124)), U32(255))});
  }
  void SetVgRel(u32 i, Id v) {
    const Id ptr = VgRelPtr(i);
    if (predicate_vector)
      v = SelectB(LaneActive(Exec()), v, m.Load(t_u, ptr));
    StorePrivate(ptr, v);
  }
  Id VgIndexed(u32 i, const base::Vector<u32>& values) {
    const Id m0 = Sg(124);
    Id v = Vg(base::Min(i + values.back(), 255u));
    for (size_t k = values.size() - 1; k-- > 0;)
      v = SelectB(Eq(m0, U32(values[k])), Vg(base::Min(i + values[k], 255u)),
                  v);
    return v;
  }
  void SetVgIndexed(u32 i, const base::Vector<u32>& values, Id v) {
    const Id m0 = Sg(124);
    for (u32 value : values) {
      const u32 r = base::Min(i + value, 255u);
      SetVg(r, SelectB(Eq(m0, U32(value)), v, Vg(r)));
    }
  }
  Id VgF(u32 i) { return m.Bitcast(t_f, Vg(i)); }
  void SetVgF(u32 i, Id f) { SetVg(i, m.Bitcast(t_u, f)); }

  base::HashSet<u32> spill_vgprs;
  base::HashMap<u32, Id> spill_vars;
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

  Id U32(u32 v) { return m.ConstU32(v); }
  Id F32(float v) { return m.ConstF32(v); }

  Id Ext1(u32 op, Id a) { return m.ExtInst(t_f, op, {a}); }
  Id Ext2(u32 op, Id a, Id b) { return m.ExtInst(t_f, op, {a, b}); }
  Id FMul(Id a, Id b) {
    const Id value = m.Emit(spv::Op::OpFMul, t_f, {a, b});
    m.Decorate(value, spv::Decoration::NoContraction);
    return value;
  }
  Id LegacyMul(Id a, Id b) {
    const Id zero = F32(0.f);
    const Id any_zero =
        m.Emit(spv::Op::OpLogicalOr, t_bool,
               {m.Emit(spv::Op::OpFOrdEqual, t_bool, {a, zero}),
                m.Emit(spv::Op::OpFOrdEqual, t_bool, {b, zero})});
    return SelectF(any_zero, zero, FMul(a, b));
  }
  Id FAdd(Id a, Id b) {
    const Id value = m.Emit(spv::Op::OpFAdd, t_f, {a, b});
    m.Decorate(value, spv::Decoration::NoContraction);
    return value;
  }
  Id FSub(Id a, Id b) {
    const Id value = m.Emit(spv::Op::OpFSub, t_f, {a, b});
    m.Decorate(value, spv::Decoration::NoContraction);
    return value;
  }
  Id FDiv(Id a, Id b) { return m.Emit(spv::Op::OpFDiv, t_f, {a, b}); }
  Id FNeg(Id a) { return m.Emit(spv::Op::OpFNegate, t_f, {a}); }
  Id FClamp01(Id f) {
    return m.ExtInst(t_f, GLSLstd450NClamp, {f, F32(0.0f), F32(1.0f)});
  }

  Id IsInf(Id f) { return m.Emit(spv::Op::OpIsInf, t_bool, {f}); }
  Id FLt(Id a, Id b) { return m.Emit(spv::Op::OpFOrdLessThan, t_bool, {a, b}); }
  Id FGt(Id a, Id b) {
    return m.Emit(spv::Op::OpFOrdGreaterThan, t_bool, {a, b});
  }
  static constexpr float kFltMax = 3.402823466e+38f;
  Id ClampInfToFltMax(Id f) {
    const Id lim = SelectF(FLt(f, F32(0.0f)), F32(-kFltMax), F32(kFltMax));
    return SelectF(IsInf(f), lim, f);
  }
  Id ConvertInfToZero(Id f) { return SelectF(IsInf(f), F32(0.0f), f); }

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
  Id Sext24(Id a) {
    return m.Bitcast(t_u, m.Emit(spv::Op::OpBitFieldSExtract, t_i,
                                 {m.Bitcast(t_i, a), U32(0), U32(24)}));
  }
  Id Low24(Id a) { return And(a, U32(0xFFFFFF)); }

  Id TypeV4u() { return m.TypeVec(t_u, 4); }
  Id True() { return IsNonZero(U32(1)); }

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

  Id PairType() { return m.TypeStruct({t_u, t_u}); }

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
    m.Name(v, "cbuf" + base::ToString(binding));
    cbuf_vars[binding] = v;
    return v;
  }
  Id CbufDword(u32 binding, u32 k) {
    if (indirect_cbufs)
      return CbufDwordId(binding, U32(k));
    const Id var = EnsureCbuf(binding);
    const Id p_u = m.TypePointer(spv::StorageClass::Uniform, t_u);
    const Id ch = m.AccessChain(
        p_u, var,
        {U32(0), U32(base::Min(k >> 2, kCbufDwords / 4 - 1)), U32(k & 3)});
    return m.Load(t_u, ch);
  }
  Id DrawDataDword(u32 index) {
    if (!cbuf_offsets) {
      const Id array = m.TypeArray(TypeV4u(), kIndirectDrawDwords / 4);
      m.Decorate(array, spv::Decoration::ArrayStride, {16});
      const Id block = m.TypeStruct({array});
      m.Decorate(block, spv::Decoration::Block);
      m.MemberDecorate(block, 0, spv::Decoration::Offset, {0});
      cbuf_offsets =
          m.Variable(m.TypePointer(spv::StorageClass::Uniform, block),
                     spv::StorageClass::Uniform);
      m.Decorate(cbuf_offsets, spv::Decoration::DescriptorSet, {1});
      m.Decorate(cbuf_offsets, spv::Decoration::Binding, {1});
    }
    return m.Load(
        t_u,
        m.AccessChain(m.TypePointer(spv::StorageClass::Uniform, t_u),
                      cbuf_offsets, {U32(0), U32(index / 4), U32(index % 4)}));
  }
  Id CbufDwordId(u32 binding, Id k) {
    if (indirect_cbufs) {
      if (!cbuf_ring) {
        cbuf_ring = m.Variable(m.TypePointer(spv::StorageClass::StorageBuffer,
                                             EnsureRawBlockType()),
                               spv::StorageClass::StorageBuffer);
        m.Decorate(cbuf_ring, spv::Decoration::DescriptorSet, {1});
        m.Decorate(cbuf_ring, spv::Decoration::Binding, {0});
        m.Decorate(cbuf_ring, spv::Decoration::NonWritable);
      }
      const Id base = DrawDataDword(binding);
      const Id index = Add(
          base, Add(Mul(UMin(Shr(k, U32(2)), U32(kCbufDwords / 4 - 1)), U32(4)),
                    And(k, U32(3))));
      return m.Load(
          t_u,
          m.AccessChain(m.TypePointer(spv::StorageClass::StorageBuffer, t_u),
                        cbuf_ring, {U32(0), index}));
    }
    const Id var = EnsureCbuf(binding);
    const Id v4 = UMin(Shr(k, U32(2)), U32(kCbufDwords / 4 - 1));
    const Id p_u = m.TypePointer(spv::StorageClass::Uniform, t_u);
    const Id ch = m.AccessChain(p_u, var, {U32(0), v4, And(k, U32(3))});
    return m.Load(t_u, ch);
  }

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
    m.Decorate(v, spv::Decoration::NonWritable);
    m.Name(v, "rawbuf" + base::ToString(binding));
    gfx_buf_vars[binding] = v;
    return v;
  }

  Id SrcRaw(u32 field, u32 literal) {
    if (field <= 127)
      return Sg(field);
    if (field == 128)
      return U32(0);
    if (field >= 129 && field <= 192)
      return U32(field - 128);
    if (field >= 193 && field <= 208)
      return U32(static_cast<u32>(-static_cast<int>(field - 192)));
    switch (field) {
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
        return U32(0x3e22f983u);
      case 251:
        return SelectB(
            IsZero(WaveMask(106)),
            U32(1), U32(0));
      case 252:
        return SelectB(IsZero(WaveMask(126)),
                       U32(1), U32(0));
      case 253:
        return Scc();
    }
    if (field == 255)
      return U32(literal);
    if (field >= 256)
      return Vg(field - 256);
    return U32(0);
  }
  Id SrcRawHi(u32 field, u32 literal, bool sign_extend) {
    if (rdna_sources && field == 125)
      return U32(0);
    if (field <= 126)
      return Sg(field + 1);
    if (field >= 256 && field <= 510)
      return Vg(field - 255);
    const Id lo = SrcRaw(field, literal);
    if (field >= 128 && field <= 208)
      return Sar(lo, U32(31));
    return sign_extend ? Sar(lo, U32(31)) : U32(0);
  }
  Id SrcF(u32 field, u32 literal, bool neg = false, bool abs = false) {
    Id f = m.Bitcast(t_f, SrcRaw(field, literal));
    if (abs)
      f = Ext1(GLSLstd450FAbs, f);
    if (neg)
      f = FNeg(f);
    return f;
  }
};

inline void SeedPsInputVgprs(Translator& t, u32 ena, base::Vector<Id>& iface) {
  if (!ena)
    return;
  static constexpr u8 kWidth[16] = {2, 2, 2, 3, 2, 2, 2, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1};
  u32 vg[16] = {}, next = 0;
  for (u32 bit = 0; bit < 16; bit++)
    if (ena & (1u << bit)) {
      vg[bit] = next;
      next += kWidth[bit];
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
    t.SetVg(vg[12], t.SelectB(t.m.Load(t.t_bool, front_facing),
                              t.U32(0xFFFFFFFFu), t.U32(0)));
  }

  if (ena & (1u << 13)) {
    t.m.Capability(spv::Capability::Geometry);
    const Id layer =
        t.m.Variable(t.m.TypePointer(spv::StorageClass::Input, t.t_i),
                     spv::StorageClass::Input);
    t.m.Decorate(layer, spv::Decoration::BuiltIn,
                 {static_cast<u32>(spv::BuiltIn::Layer)});
    t.m.Decorate(layer, spv::Decoration::Flat);
    iface.push_back(layer);
    t.SetVg(vg[13], t.Shl(t.And(t.m.Bitcast(t.t_u, t.m.Load(t.t_i, layer)),
                                t.U32(0x7FF)),
                          t.U32(16)));
  }
}

struct GsCopyExport {
  u32 target = 0, chan = 0;
  int comp = -1;
  u32 bits = 0;
};

struct StageContext {
  bool is_ps = false;
  const u32* ps_in_cntl = nullptr;
  u32 ps_num_interp = 0;
  const base::Vector<u32>* vs_exported_params = nullptr;
  bool is_cs = false;
  bool wave_lockstep = false;
  bool lds_sync = false;
  Id cs_probe_var = 0;
  Recompiled* r = nullptr;
  base::Vector<Id>* iface = nullptr;
  Id main_fn = 0;

  bool is_mesh = false;
  Id mesh_counts = 0;
  Id mesh_primitive = 0;
  Id mesh_local_index = 0;
  Id pos_out = 0;
  base::HashMap<u32, Id> param_outs;
  base::HashSet<u32> direct_vfetch;
  u32 max_param = 0;
  struct VfetchSeed {
    Id in_var;
    u32 dest_vgpr, num_comps;
  };
  base::HashMap<u32, VfetchSeed> vfetch_seed;
  Id dbg_pos_in = 0;
  u32 dbg_pos_comps = 0;
  int dbg_pos_cbuf = -1;
  u32 dbg_pos_dword = 0;
  int dbg_pos_world = -1;

  Id color_outs[8] = {};
  u32 mrt_uint_mask = 0;
  u32 mrt_bound_mask = 0xFF;
  bool gcn_tsharp = false;
  u32 col_format = kColFormatUnknown;
  u64 tex_uint_mask = 0;
  Id depth_out = 0;
  base::HashMap<u32, Id> in_vars;
  bool wrote_color = false;
  Id color_written_var = 0;
  const base::HashSet<u32>* flat_attrs = nullptr;
  const MimgBindingPlan* mimg_plan = nullptr;
  u32 tex_binding_base = 0;
  static constexpr u32 kMaxPsSamplers = 64;
  Id tex_vars[kMaxPsSamplers] = {};
  bool bindless = false;
  Id bindless_images[kBindlessKinds] = {};
  Id bindless_samplers = 0;
  Id bindless_tables[2] = {};
  Id bindless_lookup[2] = {};
  u32 tex_types[kMaxPsSamplers] = {};
  u64 tex_3d_mask = 0;
  u64 tex_1d_mask = 0;

  base::HashMap<u32, u32> cbuf_bind;
  base::HashMap<u32, u32> gfx_buf_bind;
  base::HashMap<u32, u32> smem_cbuf_by_pc;
  base::HashMap<u32, u32> cbuf_first_dword;
  base::HashSet<u32> skip_launch_movs;

  base::HashMap<u32, u32> cs_bind;
  u32 cs_cur_pc = 0;
  base::Vector<Id> cs_ssbo;
  Id cs_bounds_var = 0;
  Id cs_guest_table = 0, cs_guest_translate = 0;
  base::HashMap<u32, base::Pair<u32, u32>> cs_runtime_resources;
  bool direct = false;
  Id direct_formats = 0;
  base::HashSet<u32> direct_vertex_fetch;
  Id direct_vertex_index = 0;
  base::HashSet<u32> cs_runtime_images;
  base::HashSet<u32> cs_native;
  base::HashSet<u32> cs_written;

  base::HashSet<u32> pervertex_attrs;
  base::HashMap<u32, Id> pervertex_vars;
  Id bary_var = 0;
  Id bary_nopersp_var = 0;
  Id vgpr_snap_var = 0;

  Id gds_var = 0;
  Id lds_var = 0;
  u32 lds_dwords = 0;
  spv::StorageClass lds_storage = spv::StorageClass::Workgroup;
  Id lds_wave_base = 0;
  bool is_vs_shared_lds_capable = false;
  Id vertex_index_value = 0;
  base::HashSet<u32> ds_own_lane;
  Id subgroup_id = 0;
  Id subgroup_local_id = 0;
  base::Vector<u32> lds_barrier_at;
  base::Vector<u8> uniform_readfirstlane;
  base::Vector<u8> uniform_points;
  bool lockstep_loop = false;
  bool cs_unsupported = false;

  Id es_ring = 0;
  u32 es_ring_vec4s = 0;
  u32 gs_input_verts = 0;
  Id gsvs = 0;
  u32 gsvs_dwords = 0;
  Id gs_emitted = 0;
  u32 gs_max_vert_out = 0;
  const base::Vector<GsCopyExport>* gs_exports = nullptr;
  Id layer_out = 0;
  bool gl_clip = false;
};

Id PsInputVar(Translator& t, StageContext& sc, u32 attr);
base::HashSet<u32> PlanPerVertexAttrs(const Program& program,
                                      const u8* reachable);
void EmitVintrp(Translator& t, u32 w, StageContext& sc);
Id VsParamOut(Translator& t, StageContext& sc, u32 p);
Id PsColorOut(Translator& t, StageContext& sc, u32 target);
Id PsDepthOut(Translator& t, StageContext& sc);
void SeedPsBarycentrics(Translator& t, u32 ena, StageContext& sc);

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
base::HashSet<u32> PlanLaneSpills(const Program& program,
                                  const u8* reachable = nullptr);
bool EmitLaneSpill(Translator& t,
                   u32 op,
                   u32 dst,
                   u32 src0,
                   u32 src1,
                   u32 literal);
bool EmitMbcntOfExec(Translator& t,
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

u32 SmrdLoadCount(u32 op);
u32 SmrdDwordCount(u32 op);
bool PlanCbufs(const Program& program,
               u32 first_binding,
               base::Vector<ShaderCbuf>& cbufs,
               base::HashMap<u32, u32>& bindings,
               const u8* reachable = nullptr,
               u32 max_bindings = kMaxCbufBindings);
void EmitCbufSmrd(Translator& t,
                  const Inst& inst,
                  const base::HashMap<u32, u32>& bindings);
bool MimgNamesItsLod(u32 op);
bool DsGraphicsSupported(u32 op);
u32 GraphicsLdsDwords(const Program& program, const u8* reachable);
base::HashSet<u32> PlanDsOwnLane(const Program& program, const u8* reachable);
void EnableDsSwizzle(Translator& t, StageContext& sc, base::Vector<Id>& iface);
bool UsesDsSwizzle(const Program& program, const u8* reachable);
Id BindlessLookup(Translator& t, StageContext& sc, u32 which);
void EmitMimg(Translator& t,
              const Inst& inst,
              StageContext& sc,
              const Id* address = nullptr);
void PlanGfxBuffers(const Program& program,
                    u32 first_binding,
                    const base::HashSet<u32>* claimed,
                    base::Vector<ShaderBuffer>& buffers,
                    base::HashMap<u32, u32>& bindings,
                    const u8* reachable = nullptr);
void EmitGfxMubuf(Translator& t, const Inst& inst, StageContext& sc);
void EmitGfxMtbuf(Translator& t, const Inst& inst, StageContext& sc);
bool PlanCsResources(const Program& program,
                     const u8* reachable,
                     u32 lds_dwords,
                     RecompiledCs& r,
                     base::HashMap<u32, u32>& bind);
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

Id CsSsboPtr(Translator& t, StageContext& sc, u32 binding, Id dword_idx);
Id CsSsboLoad(Translator& t, StageContext& sc, u32 binding, Id dword_idx);
Id CsSsboBound(Translator& t, StageContext& sc, u32 binding);
void DeclareGuestMemory(Translator& t, StageContext& sc, u32 binding);
void DeclareDirectMemory(Translator& t, StageContext& sc);
bool EmitDirectLoad(Translator& t, const Inst& inst, StageContext& sc);
Id CsGuestBase(Translator& t, StageContext& sc, u32 binding);
Id CsGuestAddress(Translator& t,
                  StageContext& sc,
                  Id address,
                  Id bytes,
                  bool write = false);
void CsGuestStore(Translator& t,
                  StageContext& sc,
                  u32 binding,
                  Id dword_idx,
                  Id value);
void EmitGuestGlobal(Translator& t, const Inst& inst, StageContext& sc);
Id CsGuestLoad(Translator& t,
               StageContext& sc,
               u32 binding,
               Id dword_idx,
               Id required_end = 0);
Id CsPhysicalLoad(Translator& t, StageContext& sc, Id address);
void CsGuestStoreMasked(Translator& t,
                        StageContext& sc,
                        Id address,
                        Id value,
                        Id mask);
Id CsLinearImageEligible(Translator& t, const Inst& inst);
void EmitCsLinearImage(Translator& t,
                       const Inst& inst,
                       StageContext& sc,
                       const Id* address);

void CsSsboStore(Translator& t,
                 StageContext& sc,
                 u32 binding,
                 Id dword_idx,
                 Id value);
int CsBindingFor(StageContext& sc, u32 pc);

bool ParseCopyShader(const Program& program,
                     u32 max_vert_out,
                     base::Vector<GsCopyExport>& out);
void EmitEsRingStore(Translator& t, const Inst& inst, StageContext& sc);
void EmitGsRingAccess(Translator& t, const Inst& inst, StageContext& sc);
void EmitGsMessage(Translator& t, const Inst& inst, StageContext& sc);

base::Vector<u32> EmitRectListGeometry(u32 num_params,
                                       const base::HashSet<u32>& flat_attrs);

}  // namespace gpu::gcn
