/*
 * PS4Delta : PS4 emulation and research project
 */

#ifdef DELTA_HAVE_SPIRV_BACKEND

#include "base/arch.h"
#include "base/containers/hash_map.h"
#include "base/containers/set.h"
#include "gpu/gcn/spirv/translator.h"

namespace gpu::gcn {
namespace {

struct CarryResult {
  Id value;
  Id flag;
};

CarryResult AddCarry(Translator& t, Id a, Id b, Id carry = 0) {
  const Id p = t.m.Emit(spv::Op::OpIAddCarry, t.PairType(), {a, b});
  Id value = t.m.CompositeExtract(t.t_u, p, 0);
  Id flag = t.m.CompositeExtract(t.t_u, p, 1);
  if (carry) {
    const Id q = t.m.Emit(spv::Op::OpIAddCarry, t.PairType(),
                          {value, t.And(carry, t.U32(1))});
    value = t.m.CompositeExtract(t.t_u, q, 0);
    flag = t.Or(flag, t.m.CompositeExtract(t.t_u, q, 1));
  }
  return {value, flag};
}

CarryResult SubBorrow(Translator& t, Id a, Id b, Id borrow = 0) {
  const Id p = t.m.Emit(spv::Op::OpISubBorrow, t.PairType(), {a, b});
  Id value = t.m.CompositeExtract(t.t_u, p, 0);
  Id flag = t.m.CompositeExtract(t.t_u, p, 1);
  if (borrow) {
    const Id q = t.m.Emit(spv::Op::OpISubBorrow, t.PairType(),
                          {value, t.And(borrow, t.U32(1))});
    value = t.m.CompositeExtract(t.t_u, q, 0);
    flag = t.Or(flag, t.m.CompositeExtract(t.t_u, q, 1));
  }
  return {value, flag};
}

Id SignedAddOverflow(Translator& t, Id a, Id b, Id r) {
  return t.IsNonZero(
      t.And(t.And(t.Xor(a, r), t.Xor(b, r)), t.U32(0x80000000u)));
}

enum class ExecOp {
  kAnd,
  kOr,
  kXor,
  kAndn2,
  kOrn2,
  kNand,
  kNor,
  kXnor,
  kAndn1,
  kOrn1,
};

Id CombineExec(Translator& t, ExecOp kind, Id exec, Id src) {
  switch (kind) {
    case ExecOp::kOr:
      return t.Or(src, exec);
    case ExecOp::kXor:
      return t.Xor(src, exec);
    case ExecOp::kAndn2:
      return t.And(src, t.Not(exec));
    case ExecOp::kOrn2:
      return t.Or(src, t.Not(exec));
    case ExecOp::kNand:
      return t.Not(t.And(src, exec));
    case ExecOp::kNor:
      return t.Not(t.Or(src, exec));
    case ExecOp::kXnor:
      return t.Not(t.Xor(src, exec));
    case ExecOp::kAndn1:
      return t.And(t.Not(src), exec);
    case ExecOp::kOrn1:
      return t.Or(t.Not(src), exec);
    default:
      return t.And(src, exec);
  }
}

void SaveExec(Translator& t,
              ExecOp kind,
              u32 sdst,
              Id src,
              bool wide,
              bool write_modified,
              Id src_hi = 0) {
  if (t.lane_masks || t.wave_masks) {
    const Id old_lo = t.Sg(126), old_hi = t.Sg(127);
    const Id new_lo = CombineExec(t, kind, old_lo, src);
    const Id new_hi =
        wide ? CombineExec(t, kind, old_hi, src_hi ? src_hi : t.U32(0))
             : old_hi;
    t.SetSdst(sdst, 0, write_modified ? new_lo : old_lo);
    if (wide)
      t.SetSdst(sdst, 1, write_modified ? new_hi : old_hi);
    t.SetSg(126, new_lo);
    if (wide)
      t.SetSg(127, new_hi);
    t.SetSccBool(t.IsNonZero(wide ? t.Or(new_lo, new_hi) : new_lo));
    return;
  }
  const Id old_exec = t.Exec();
  const Id new_exec = t.And(CombineExec(t, kind, old_exec, src), t.U32(1));
  t.SetSdst(sdst, 0, write_modified ? new_exec : old_exec);
  if (wide)
    t.SetSdst(sdst, 1, t.U32(0));
  t.SetSg(126, new_exec);
  t.SetSccBool(t.IsNonZero(new_exec));
}

Id FirstSet64(Translator& t, Id lo, Id hi) {
  const Id none = t.U32(0xFFFFFFFFu);
  const Id l = t.m.ExtInst(t.t_u, GLSLstd450FindILsb, {lo});
  const Id h = t.m.ExtInst(t.t_u, GLSLstd450FindILsb, {hi});
  const Id upper = t.SelectB(t.Eq(h, none), none, t.Add(h, t.U32(32)));
  return t.SelectB(t.Eq(l, none), upper, l);
}

Id LeadingZeros64(Translator& t, Id lo, Id hi) {
  const Id l = t.m.ExtInst(t.t_u, GLSLstd450FindUMsb, {lo});
  const Id h = t.m.ExtInst(t.t_u, GLSLstd450FindUMsb, {hi});
  const Id from_lo =
      t.SelectB(t.IsZero(lo), t.U32(0xFFFFFFFFu), t.Sub(t.U32(63), l));
  return t.SelectB(t.IsZero(hi), from_lo, t.Sub(t.U32(31), h));
}

bool EmitGfx10Sop1(Translator& t, u32 op, u32 sdst, Id a, Id a_hi) {
  switch (op) {
    case 0x37:
      SaveExec(t, ExecOp::kAndn1, sdst, a, true, false, a_hi);
      return true;
    case 0x38:
      SaveExec(t, ExecOp::kOrn1, sdst, a, true, false, a_hi);
      return true;
    case 0x39:
      SaveExec(t, ExecOp::kAndn1, sdst, a, true, true, a_hi);
      return true;
    case 0x3a:
      SaveExec(t, ExecOp::kAndn2, sdst, a, true, true, a_hi);
      return true;
    case 0x3b: {
      const auto spread = [&](Id half) {
        Id x = half;
        x = t.And(t.Or(x, t.Shl(x, t.U32(8))), t.U32(0x00FF00FFu));
        x = t.And(t.Or(x, t.Shl(x, t.U32(4))), t.U32(0x0F0F0F0Fu));
        x = t.And(t.Or(x, t.Shl(x, t.U32(2))), t.U32(0x33333333u));
        x = t.And(t.Or(x, t.Shl(x, t.U32(1))), t.U32(0x55555555u));
        return t.Or(x, t.Shl(x, t.U32(1)));
      };
      t.SetSdst(sdst, 0, spread(t.And(a, t.U32(0xFFFF))));
      t.SetSdst(sdst, 1, spread(t.Shr(a, t.U32(16))));
      return true;
    }
    case 0x3c:
    case 0x3d:
    case 0x3e:
    case 0x3f:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
      SaveExec(t, static_cast<ExecOp>(op - 0x3c), sdst, a, false, false);
      return true;
    case 0x44:
      SaveExec(t, ExecOp::kAndn1, sdst, a, false, false);
      return true;
    case 0x45:
      SaveExec(t, ExecOp::kOrn1, sdst, a, false, false);
      return true;
    case 0x46:
      SaveExec(t, ExecOp::kAndn1, sdst, a, false, true);
      return true;
    case 0x47:
      SaveExec(t, ExecOp::kAndn2, sdst, a, false, true);
      return true;
    default:
      return false;
  }
}

Id ApplyOutputModifier(Translator& t, Id value, u32 omod) {
  switch (omod) {
    case 1:
      return t.FMul(value, t.F32(2.0f));
    case 2:
      return t.FMul(value, t.F32(4.0f));
    case 3:
      return t.FMul(value, t.F32(0.5f));
    default:
      return value;
  }
}

}  // namespace

bool IsVop3b(u32 op) {
  return (op >= 0x125 && op <= 0x12a) || op == 0x16d || op == 0x16e;
}

void EmitSop1(Translator& t, const Inst& inst) {
  const u32 w = inst.raw[0];
  const u32 op = inst.opcode, sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF;
  const Id a = t.SrcRaw(ssrc0, inst.literal);
  const Id a_hi = t.SrcRawHi(ssrc0, inst.literal, false);
  switch (op) {
    case 0x03:
      t.SetSdst(sdst, 0, a);
      break;
    case 0x04:
      t.SetSdst(sdst, 0, a);
      t.SetSdst(sdst, 1, a_hi);
      break;
    case 0x05:
      t.SetSdst(sdst, 0, t.SelectNz(t.Scc(), a, t.Sdst(sdst)));
      break;
    case 0x06:
      t.SetSdst(sdst, 0, t.SelectNz(t.Scc(), a, t.Sdst(sdst)));
      t.SetSdst(sdst, 1, t.SelectNz(t.Scc(), a_hi, t.Sdst(sdst, 1)));
      break;
    case 0x07: {
      const Id r = t.Not(a);
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(t.IsNonZero(r));
      break;
    }
    case 0x08: {
      const Id lo = t.Not(a), hi = t.Not(a_hi);
      t.SetSdst(sdst, 0, lo);
      t.SetSdst(sdst, 1, hi);
      t.SetSccBool(t.IsNonZero(t.Or(lo, hi)));
      break;
    }
    case 0x09:
      t.SetSdst(sdst, 0, a);
      t.SetSccBool(t.IsNonZero(a));
      break;
    case 0x0a:
      t.SetSdst(sdst, 0, a);
      t.SetSdst(sdst, 1, a_hi);
      t.SetSccBool(t.IsNonZero(t.Or(a, a_hi)));
      break;
    case 0x0b:
      t.SetSdst(sdst, 0, t.BitRev(a));
      break;
    case 0x0d: {
      const Id r = t.Sub(t.U32(32), t.PopCount(a));
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(t.IsNonZero(r));
      break;
    }
    case 0x0e: {
      const Id r = t.Sub(t.U32(64), t.Add(t.PopCount(a), t.PopCount(a_hi)));
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(t.IsNonZero(r));
      break;
    }
    case 0x0f: {
      const Id r = t.PopCount(a);
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(t.IsNonZero(r));
      break;
    }
    case 0x10: {
      const Id r = t.Add(t.PopCount(a), t.PopCount(a_hi));
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(t.IsNonZero(r));
      break;
    }
    case 0x11:
      t.SetSdst(sdst, 0, t.m.ExtInst(t.t_u, GLSLstd450FindILsb, {t.Not(a)}));
      break;
    case 0x12:
      t.SetSdst(sdst, 0, FirstSet64(t, t.Not(a), t.Not(a_hi)));
      break;
    case 0x13:
      t.SetSdst(sdst, 0, t.m.ExtInst(t.t_u, GLSLstd450FindILsb, {a}));
      break;
    case 0x14:
      t.SetSdst(sdst, 0, FirstSet64(t, a, a_hi));
      break;
    case 0x15: {
      const Id msb = t.m.ExtInst(t.t_u, GLSLstd450FindUMsb, {a});
      t.SetSdst(
          sdst, 0,
          t.SelectB(t.IsZero(a), t.U32(0xFFFFFFFFu), t.Sub(t.U32(31), msb)));
      break;
    }
    case 0x16:
      t.SetSdst(sdst, 0, LeadingZeros64(t, a, a_hi));
      break;
    case 0x17: {
      const Id smsb = t.m.Bitcast(t.t_u, t.m.ExtInst(t.t_i, GLSLstd450FindSMsb,
                                                     {t.m.Bitcast(t.t_i, a)}));
      t.SetSdst(sdst, 0,
                t.SelectB(t.Eq(smsb, t.U32(0xFFFFFFFFu)), t.U32(0xFFFFFFFFu),
                          t.Sub(t.U32(31), smsb)));
      break;
    }
    case 0x19:
    case 0x1a:
      t.SetSdst(sdst, 0,
                t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_u,
                         {a, t.U32(0), t.U32(op == 0x19 ? 8 : 16)}));
      break;
    case 0x1b:
    case 0x1d: {
      const Id bit = t.Shl(t.U32(1), t.And(a, t.U32(31)));
      t.SetSdst(sdst, 0,
                op == 0x1b ? t.And(t.Sdst(sdst), t.Not(bit))
                           : t.Or(t.Sdst(sdst), bit));
      break;
    }
    case 0x1c:
    case 0x1e: {
      const Id n = t.And(a, t.U32(63));
      const Id bit = t.Shl(t.U32(1), t.And(n, t.U32(31)));
      const Id upper = t.Uge(n, t.U32(32));
      const Id lo_bit = t.SelectB(upper, t.U32(0), bit);
      const Id hi_bit = t.SelectB(upper, bit, t.U32(0));
      const bool clear = op == 0x1c;
      t.SetSdst(sdst, 0,
                clear ? t.And(t.Sdst(sdst, 0), t.Not(lo_bit))
                      : t.Or(t.Sdst(sdst, 0), lo_bit));
      t.SetSdst(sdst, 1,
                clear ? t.And(t.Sdst(sdst, 1), t.Not(hi_bit))
                      : t.Or(t.Sdst(sdst, 1), hi_bit));
      break;
    }
    case 0x20:
    case 0x21:
      break;
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2a:
    case 0x2b:
      SaveExec(t, static_cast<ExecOp>(op - 0x24), sdst, a, true, false, a_hi);
      break;
    case 0x34: {
      const Id r = t.m.Bitcast(
          t.t_u, t.m.ExtInst(t.t_i, GLSLstd450SAbs, {t.m.Bitcast(t.t_i, a)}));
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(t.IsNonZero(r));
      break;
    }
    case 0x1f: {
      const u32 next_off = (inst.pc + 1) * 4u;
      if (t.pc_base_var) {
        const Id p_u = t.m.TypePointer(spv::StorageClass::PushConstant, t.t_u);
        const auto half = [&](u32 i) {
          return t.m.Load(t.t_u,
                          t.m.AccessChain(p_u, t.pc_base_var,
                                          {t.U32(t.pc_base_member + i)}));
        };
        const Id lo = t.Add(half(0), t.U32(next_off));
        const Id carry =
            t.SelectB(t.Ult(lo, t.U32(next_off)), t.U32(1), t.U32(0));
        t.SetSdst(sdst, 0, lo);
        t.SetSdst(sdst, 1, t.Add(half(1), carry));
      } else {
        const u64 next = t.program_base + next_off;
        t.SetSdst(sdst, 0, t.U32(static_cast<u32>(next)));
        t.SetSdst(sdst, 1, t.U32(static_cast<u32>(next >> 32)));
      }
      break;
    }
    default:
      if (!t.rdna_sources || !EmitGfx10Sop1(t, op, sdst, a, a_hi))
        WarnUnsupported("sop1", op);
      break;
  }
}

void EmitSop2(Translator& t, const Inst& inst) {
  const u32 w = inst.raw[0];
  const u32 op = inst.opcode, sdst = (w >> 16) & 0x7F;
  const u32 s0f = w & 0xFF, s1f = (w >> 8) & 0xFF;
  const Id a = t.SrcRaw(s0f, inst.literal), b = t.SrcRaw(s1f, inst.literal);
  const Id a_hi = t.SrcRawHi(s0f, inst.literal, op == 0x23);
  const Id b_hi = t.SrcRawHi(s1f, inst.literal, false);
  Id r = 0, r_hi = 0;
  bool scc = false, wide_scc = false;

  const auto shift64 = [&](u32 kind) {
    const Id n = t.And(b, t.U32(63)), n_lo = t.And(n, t.U32(31));
    const Id ge32 = t.Uge(n, t.U32(32));
    const Id zero = t.IsZero(n);
    const Id inv = t.And(t.Sub(t.U32(32), n_lo), t.U32(31));
    if (kind == 0) {
      const Id cross = t.SelectB(zero, t.U32(0), t.Shr(a, inv));
      const Id hi_small = t.Or(t.Shl(a_hi, n_lo), cross);
      r = t.SelectB(ge32, t.U32(0), t.Shl(a, n_lo));
      r_hi = t.SelectB(ge32, t.Shl(a, n_lo), hi_small);
    } else {
      const Id cross = t.SelectB(zero, t.U32(0), t.Shl(a_hi, inv));
      const Id lo_small = t.Or(t.Shr(a, n_lo), cross);
      const Id hi_shift = kind == 1 ? t.Shr(a_hi, n_lo) : t.Sar(a_hi, n_lo);
      r = t.SelectB(ge32, hi_shift, lo_small);
      r_hi = t.SelectB(ge32, kind == 1 ? t.U32(0) : t.Sar(a_hi, t.U32(31)),
                       hi_shift);
    }
    scc = wide_scc = true;
  };

  switch (op) {
    case 0x00: {
      const CarryResult c = AddCarry(t, a, b);
      r = c.value;
      t.SetSccBool(t.IsNonZero(c.flag));
      break;
    }
    case 0x01: {
      const CarryResult c = SubBorrow(t, a, b);
      r = c.value;
      t.SetSccBool(t.IsNonZero(c.flag));
      break;
    }
    case 0x02:
      r = t.Add(a, b);
      t.SetSccBool(SignedAddOverflow(t, a, b, r));
      break;
    case 0x03:
      r = t.Sub(a, b);
      t.SetSccBool(t.IsNonZero(
          t.And(t.And(t.Xor(a, b), t.Xor(a, r)), t.U32(0x80000000u))));
      break;
    case 0x04: {
      const CarryResult c = AddCarry(t, a, b, t.Scc());
      r = c.value;
      t.SetSccBool(t.IsNonZero(c.flag));
      break;
    }
    case 0x05: {
      const CarryResult c = SubBorrow(t, a, b, t.Scc());
      r = c.value;
      t.SetSccBool(t.IsNonZero(c.flag));
      break;
    }
    case 0x06:
      r = t.SMin(a, b);
      t.SetSccBool(t.m.Emit(spv::Op::OpSLessThan, t.t_bool,
                            {t.m.Bitcast(t.t_i, a), t.m.Bitcast(t.t_i, b)}));
      break;
    case 0x07:
      r = t.UMin(a, b);
      t.SetSccBool(t.Ult(a, b));
      break;
    case 0x08:
      r = t.SMax(a, b);
      t.SetSccBool(t.m.Emit(spv::Op::OpSGreaterThan, t.t_bool,
                            {t.m.Bitcast(t.t_i, a), t.m.Bitcast(t.t_i, b)}));
      break;
    case 0x09:
      r = t.UMax(a, b);
      t.SetSccBool(t.m.Emit(spv::Op::OpUGreaterThan, t.t_bool, {a, b}));
      break;
    case 0x0a:
      r = t.SelectNz(t.Scc(), a, b);
      break;
    case 0x0b:
      r = t.SelectNz(t.Scc(), a, b);
      r_hi = t.SelectNz(t.Scc(), a_hi, b_hi);
      break;
    case 0x0e:
      r = t.And(a, b);
      scc = true;
      break;
    case 0x0f:
      r = t.And(a, b);
      r_hi = t.And(a_hi, b_hi);
      scc = wide_scc = true;
      break;
    case 0x10:
      r = t.Or(a, b);
      scc = true;
      break;
    case 0x11:
      r = t.Or(a, b);
      r_hi = t.Or(a_hi, b_hi);
      scc = wide_scc = true;
      break;
    case 0x12:
      r = t.Xor(a, b);
      scc = true;
      break;
    case 0x13:
      r = t.Xor(a, b);
      r_hi = t.Xor(a_hi, b_hi);
      scc = wide_scc = true;
      break;
    case 0x14:
      r = t.And(a, t.Not(b));
      scc = true;
      break;
    case 0x15:
      r = t.And(a, t.Not(b));
      r_hi = t.And(a_hi, t.Not(b_hi));
      scc = wide_scc = true;
      break;
    case 0x16:
      r = t.Or(a, t.Not(b));
      scc = true;
      break;
    case 0x17:
      r = t.Or(a, t.Not(b));
      r_hi = t.Or(a_hi, t.Not(b_hi));
      scc = wide_scc = true;
      break;
    case 0x18:
      r = t.Not(t.And(a, b));
      scc = true;
      break;
    case 0x19:
      r = t.Not(t.And(a, b));
      r_hi = t.Not(t.And(a_hi, b_hi));
      scc = wide_scc = true;
      break;
    case 0x1a:
      r = t.Not(t.Or(a, b));
      scc = true;
      break;
    case 0x1b:
      r = t.Not(t.Or(a, b));
      r_hi = t.Not(t.Or(a_hi, b_hi));
      scc = wide_scc = true;
      break;
    case 0x1c:
      r = t.Not(t.Xor(a, b));
      scc = true;
      break;
    case 0x1d:
      r = t.Not(t.Xor(a, b));
      r_hi = t.Not(t.Xor(a_hi, b_hi));
      scc = wide_scc = true;
      break;
    case 0x1e:
      r = t.Shl(a, b);
      scc = true;
      break;
    case 0x1f:
      shift64(0);
      break;
    case 0x20:
      r = t.Shr(a, b);
      scc = true;
      break;
    case 0x21:
      shift64(1);
      break;
    case 0x22:
      r = t.Sar(a, b);
      scc = true;
      break;
    case 0x23:
      shift64(2);
      break;
    case 0x24: {
      const Id width = t.And(a, t.U32(31)), off = t.And(b, t.U32(31));
      r = t.Shl(t.Sub(t.Shl(t.U32(1), width), t.U32(1)), off);
      break;
    }
    case 0x25: {
      const Id width = t.And(a, t.U32(63)), off = t.And(b, t.U32(63));
      const Id w_wide = t.Uge(width, t.U32(32));
      const Id lo_ones =
          t.SelectB(w_wide, t.U32(0xFFFFFFFFu),
                    t.Sub(t.Shl(t.U32(1), t.And(width, t.U32(31))), t.U32(1)));
      const Id hi_ones = t.SelectB(
          w_wide,
          t.Sub(t.Shl(t.U32(1), t.And(t.Sub(width, t.U32(32)), t.U32(31))),
                t.U32(1)),
          t.U32(0));
      const Id n_lo = t.And(off, t.U32(31)), o_wide = t.Uge(off, t.U32(32));
      const Id inv = t.And(t.Sub(t.U32(32), n_lo), t.U32(31));
      const Id cross = t.SelectB(t.IsZero(off), t.U32(0), t.Shr(lo_ones, inv));
      r = t.SelectB(o_wide, t.U32(0), t.Shl(lo_ones, n_lo));
      r_hi = t.SelectB(o_wide, t.Shl(lo_ones, n_lo),
                       t.Or(t.Shl(hi_ones, n_lo), cross));
      break;
    }
    case 0x26:
      r = t.Mul(a, b);
      break;
    case 0x27: {
      const Id off = t.And(b, t.U32(31));
      const Id width = t.UMin(t.And(t.Shr(b, t.U32(16)), t.U32(0x7F)),
                              t.Sub(t.U32(32), off));
      r = t.m.Emit(spv::Op::OpBitFieldUExtract, t.t_u, {a, off, width});
      scc = true;
      break;
    }
    case 0x28: {
      const Id off = t.And(b, t.U32(31));
      const Id width = t.UMin(t.And(t.Shr(b, t.U32(16)), t.U32(0x7F)),
                              t.Sub(t.U32(32), off));
      r = t.m.Bitcast(t.t_u, t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                                      {t.m.Bitcast(t.t_i, a), off, width}));
      scc = true;
      break;
    }
    case 0x29: {
      const Id off = t.And(b, t.U32(63));
      const Id width = t.UMin(t.And(t.Shr(b, t.U32(16)), t.U32(0x7F)),
                              t.Sub(t.U32(64), off));
      const Id n_lo = t.And(off, t.U32(31));
      const Id ge32 = t.Uge(off, t.U32(32));
      const Id inv = t.And(t.Sub(t.U32(32), n_lo), t.U32(31));
      const Id cross = t.SelectB(t.IsZero(off), t.U32(0), t.Shl(a_hi, inv));
      const Id lo_small = t.Or(t.Shr(a, n_lo), cross);
      const Id sh_lo = t.SelectB(ge32, t.Shr(a_hi, n_lo), lo_small);
      const Id sh_hi = t.SelectB(ge32, t.U32(0), t.Shr(a_hi, n_lo));
      const Id wge32 = t.Uge(width, t.U32(32));
      const Id wge64 = t.Uge(width, t.U32(64));
      const Id w_lo = t.And(width, t.U32(31));
      const Id low_part = t.Sub(t.Shl(t.U32(1), w_lo), t.U32(1));
      const Id hi_width = t.SelectB(wge32, t.Sub(width, t.U32(32)), t.U32(0));
      const Id hi_part =
          t.Sub(t.Shl(t.U32(1), t.And(hi_width, t.U32(31))), t.U32(1));
      r = t.And(sh_lo, t.SelectB(wge32, t.U32(0xFFFFFFFFu), low_part));
      r_hi = t.And(sh_hi, t.SelectB(wge64, t.U32(0xFFFFFFFFu),
                                    t.SelectB(wge32, hi_part, t.U32(0))));
      scc = wide_scc = true;
      break;
    }
    case 0x2c:
      r = t.m.Bitcast(t.t_u, t.m.ExtInst(t.t_i, GLSLstd450SAbs,
                                         {t.m.Bitcast(t.t_i, t.Sub(a, b))}));
      scc = true;
      break;
    case 0x2e:
    case 0x2f:
    case 0x30:
    case 0x31: {
      if (!t.rdna_sources) {
        WarnUnsupported("sop2", op);
        r = a;
        break;
      }
      const CarryResult c = AddCarry(t, t.Shl(a, t.U32(op - 0x2d)), b);
      r = c.value;
      t.SetSccBool(t.IsNonZero(c.flag));
      break;
    }
    case 0x32:
      if (inst.isa != IsaMode::kNeo) {
        WarnUnsupported("sop2", op);
        r = a;
        break;
      }
      r = t.Or(t.And(a, t.U32(0xFFFF)),
               t.Shl(t.And(b, t.U32(0xFFFF)), t.U32(16)));
      break;
    case 0x33:
      if (inst.isa != IsaMode::kNeo) {
        WarnUnsupported("sop2", op);
        r = a;
        break;
      }
      r = t.Or(t.And(a, t.U32(0xFFFF)), t.And(b, t.U32(0xFFFF0000u)));
      break;
    case 0x34:
      if (inst.isa != IsaMode::kNeo) {
        WarnUnsupported("sop2", op);
        r = a;
        break;
      }
      r = t.Or(t.Shr(a, t.U32(16)), t.And(b, t.U32(0xFFFF0000u)));
      break;
    default:
      WarnUnsupported("sop2", op);
      r = a;
      break;
  }
  if (r) {
    t.SetSdst(sdst, 0, r);
    if (r_hi)
      t.SetSdst(sdst, 1, r_hi);
    if (scc)
      t.SetSccBool(t.IsNonZero(wide_scc ? t.Or(r, r_hi) : r));
  }
}

void EmitSopc(Translator& t, const Inst& inst) {
  const u32 w = inst.raw[0];
  const u32 op = inst.opcode, s0f = w & 0xFF, s1f = (w >> 8) & 0xFF;
  const Id a = t.SrcRaw(s0f, inst.literal), b = t.SrcRaw(s1f, inst.literal);
  const Id ai = t.m.Bitcast(t.t_i, a), bi = t.m.Bitcast(t.t_i, b);
  Id c = 0;
  switch (op) {
    case 0x00:
      c = t.Eq(a, b);
      break;
    case 0x01:
      c = t.m.Emit(spv::Op::OpINotEqual, t.t_bool, {a, b});
      break;
    case 0x02:
      c = t.m.Emit(spv::Op::OpSGreaterThan, t.t_bool, {ai, bi});
      break;
    case 0x03:
      c = t.m.Emit(spv::Op::OpSGreaterThanEqual, t.t_bool, {ai, bi});
      break;
    case 0x04:
      c = t.m.Emit(spv::Op::OpSLessThan, t.t_bool, {ai, bi});
      break;
    case 0x05:
      c = t.m.Emit(spv::Op::OpSLessThanEqual, t.t_bool, {ai, bi});
      break;
    case 0x06:
      c = t.Eq(a, b);
      break;
    case 0x07:
      c = t.m.Emit(spv::Op::OpINotEqual, t.t_bool, {a, b});
      break;
    case 0x08:
      c = t.m.Emit(spv::Op::OpUGreaterThan, t.t_bool, {a, b});
      break;
    case 0x09:
      c = t.Uge(a, b);
      break;
    case 0x0a:
      c = t.Ult(a, b);
      break;
    case 0x0b:
      c = t.Ule(a, b);
      break;
    case 0x0c: {
      c = t.IsZero(t.And(t.Shr(a, b), t.U32(1)));
      break;
    }
    case 0x0d: {
      c = t.IsNonZero(t.And(t.Shr(a, b), t.U32(1)));
      break;
    }
    case 0x12:
    case 0x13: {
      if (!t.rdna_sources) {
        WarnUnsupported("sopc", op);
        break;
      }
      const Id ahi = t.SrcRawHi(s0f, inst.literal, false);
      const Id bhi = t.SrcRawHi(s1f, inst.literal, false);
      const Id equal = t.LAnd(t.Eq(a, b), t.Eq(ahi, bhi));
      c = op == 0x12 ? equal
                     : t.m.Emit(spv::Op::OpLogicalNot, t.t_bool, {equal});
      break;
    }
    default:
      WarnUnsupported("sopc", op);
      break;
  }
  if (c)
    t.SetSccBool(c);
}

void EmitSopk(Translator& t, const Inst& inst) {
  const u32 w = inst.raw[0];
  const u32 op = inst.opcode, sdst = (w >> 16) & 0x7F;
  const u32 simm_bits = w & 0xFFFF;
  const u32 sext =
      static_cast<u32>(static_cast<i32>(static_cast<i16>(simm_bits)));
  const Id imm_i = t.U32(sext);
  const Id imm_u = t.U32(simm_bits);

  const auto cmp = [&](spv::Op cmp_op, bool is_signed) {
    const Id s0 = t.Sg(sdst);
    const Id imm = is_signed ? imm_i : imm_u;
    if (is_signed)
      t.SetSccBool(t.m.Emit(cmp_op, t.t_bool,
                            {t.m.Bitcast(t.t_i, s0), t.m.Bitcast(t.t_i, imm)}));
    else
      t.SetSccBool(t.m.Emit(cmp_op, t.t_bool, {s0, imm}));
  };

  switch (op) {
    case 0x00:
      t.SetSdst(sdst, 0, imm_i);
      break;
    case 0x02:
      t.SetSdst(sdst, 0, t.SelectNz(t.Scc(), imm_i, t.Sdst(sdst)));
      break;
    case 0x03:
      cmp(spv::Op::OpIEqual, true);
      break;
    case 0x04:
      cmp(spv::Op::OpINotEqual, true);
      break;
    case 0x05:
      cmp(spv::Op::OpSGreaterThan, true);
      break;
    case 0x06:
      cmp(spv::Op::OpSGreaterThanEqual, true);
      break;
    case 0x07:
      cmp(spv::Op::OpSLessThan, true);
      break;
    case 0x08:
      cmp(spv::Op::OpSLessThanEqual, true);
      break;
    case 0x09:
      cmp(spv::Op::OpIEqual, false);
      break;
    case 0x0a:
      cmp(spv::Op::OpINotEqual, false);
      break;
    case 0x0b:
      cmp(spv::Op::OpUGreaterThan, false);
      break;
    case 0x0c:
      cmp(spv::Op::OpUGreaterThanEqual, false);
      break;
    case 0x0d:
      cmp(spv::Op::OpULessThan, false);
      break;
    case 0x0e:
      cmp(spv::Op::OpULessThanEqual, false);
      break;
    case 0x0f: {
      const Id s0 = t.Sg(sdst);
      const Id r = t.Add(s0, imm_i);
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(SignedAddOverflow(t, s0, imm_i, r));
      break;
    }
    case 0x10: {
      const Id lhs = t.Sg(sdst);
      const Id product =
          t.m.Emit(spv::Op::OpSMulExtended, t.PairType(), {lhs, imm_i});
      const Id lo = t.m.CompositeExtract(t.t_u, product, 0);
      const Id hi = t.m.CompositeExtract(t.t_u, product, 1);
      t.SetSdst(sdst, 0, lo);
      t.SetSccBool(
          t.m.Emit(spv::Op::OpINotEqual, t.t_bool, {hi, t.Sar(lo, t.U32(31))}));
      break;
    }
    case 0x12:
      t.SetSdst(sdst, 0, t.U32(0));
      WarnUnsupported("sopk", op);
      break;
    case 0x13:
      WarnUnsupported("sopk", op);
      break;
    default:
      WarnUnsupported("sopk", op);
      break;
  }
}

base::HashSet<u32> PlanLaneSpills(const Program& program, const u8* reachable) {
  base::HashSet<u32> spills;
  u32 index = 0;
  for (const Inst& inst : program) {
    const u32 i = index++;
    if (reachable && !reachable[i])
      continue;
    if (inst.enc == Enc::kVop2 && inst.opcode == 0x02)
      spills.insert((inst.raw[0] >> 17) & 0xFF);
    else if (inst.enc == Enc::kVop3 && inst.opcode == 0x102)
      spills.insert(inst.raw[0] & 0xFF);
  }
  return spills;
}

bool EmitLaneSpill(Translator& t,
                   u32 op,
                   u32 dst,
                   u32 src0,
                   u32 src1,
                   u32 literal) {
  if (op == 0x01) {
    if (src0 < 256 || !t.IsSpillVgpr(src0 - 256))
      return false;
    t.SetSg(dst,
            t.m.Load(t.t_u, t.SpillAt(src0 - 256, t.SrcRaw(src1, literal))));
    return true;
  }
  if (op == 0x02) {
    if (!t.IsSpillVgpr(dst))
      return false;
    t.StorePrivate(t.SpillAt(dst, t.SrcRaw(src1, literal)),
                   t.SrcRaw(src0, literal));
    return false;
  }
  return false;
}

bool EmitMbcntOfExec(Translator& t,
                     u32 op,
                     u32 dst,
                     u32 src0,
                     u32 src1,
                     u32 literal) {
  if ((op != 0x23 && op != 0x24) || (src0 != 126 && src0 != 127) ||
      t.wave_masks || t.lane_masks || t.full_wave_masks)
    return false;
  t.SetVg(dst, t.SrcRaw(src1, literal));
  return true;
}

Id ReadLane(Translator& t, Id value, Id lane) {
  if (!t.lane_id)
    return value;
  lane = t.And(lane, t.U32(t.wave_size - 1));
  if (t.CanExchange())
    return t.WaveExchange(value, lane);
  if (t.wave_size == 64 && WaveSplitsAcrossSubgroups())
    NoteApproximated("v_readlane.no-sync-point", 0x01);
  return t.SubgroupShuffle(value, lane);
}

Id ReadFirstLane(Translator& t, Id value) {
  if (t.readfirstlane_uniform)
    return value;
  if (!t.xchg_lanes)
    return value;
  if (t.full_wave_masks) {
    const Id lo = t.Sg(126), hi = t.Sg(127);
    const Id first_lo = t.m.ExtInst(t.t_u, GLSLstd450FindILsb, {lo});
    const Id first_hi = t.m.ExtInst(t.t_u, GLSLstd450FindILsb, {hi});
    const Id first = t.SelectB(
        t.IsNonZero(lo), first_lo,
        t.SelectB(t.IsNonZero(hi), t.Add(first_hi, t.U32(32)), t.U32(0)));
    return t.WaveExchange(value, first);
  }
  t.RequireSubgroup(spv::Capability::GroupNonUniformBallot);
  t.RequireSubgroup(spv::Capability::GroupNonUniformVote);
  const Id scope = t.U32(3);
  const Id active = t.LaneActive(t.Exec());
  const Id ballot = t.m.Emit(spv::Op::OpGroupNonUniformBallot,
                             t.m.TypeVec(t.t_u, 4), {scope, active});
  const Id any =
      t.m.Emit(spv::Op::OpGroupNonUniformAny, t.t_bool, {scope, active});
  const Id first =
      t.m.Emit(spv::Op::OpGroupNonUniformBallotFindLSB, t.t_u, {scope, ballot});
  const Id half = t.SubgroupShuffle(value, t.SelectB(any, first, t.U32(0)));
  if (!WaveSplitsAcrossSubgroups())
    return half;
  if (!t.CanExchange()) {
    NoteApproximated("v_readfirstlane.no-sync-point", 0x02);
    return half;
  }
  t.WavePublish(half, 0);
  t.WavePublish(t.SelectB(any, t.U32(1), t.U32(0)), 1);
  t.Barrier();
  const Id lo = t.WaveFetch(t.U32(0), 0);
  const Id lo_any = t.WaveFetch(t.U32(0), 1);
  const Id hi = t.WaveFetch(t.U32(32), 0);
  t.Barrier();
  return t.SelectB(t.IsNonZero(lo_any), lo, hi);
}

void EmitVop1(Translator& t, u32 op, u32 vdst, Id s0, bool clamp, u32 omod) {
  const auto set_f = [&](Id f) {
    f = ApplyOutputModifier(t, f, omod);
    t.SetVgF(vdst, clamp ? t.FClamp01(f) : f);
  };
  const auto set_u = [&](Id u) { t.SetVg(vdst, u); };
  const Id u0 = t.m.Bitcast(t.t_u, s0);
  switch (op) {
    case 0x00:
      break;
    case 0x01:
      set_u(u0);
      break;
    case 0x02:
      t.SetSg(vdst, ReadFirstLane(t, u0));
      break;
    case 0x05:
      set_f(t.m.Emit(spv::Op::OpConvertSToF, t.t_f, {t.m.Bitcast(t.t_i, s0)}));
      break;
    case 0x06:
      set_f(t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {u0}));
      break;
    case 0x07:
      set_u(t.m.Emit(spv::Op::OpConvertFToU, t.t_u, {s0}));
      break;
    case 0x08:
      set_u(t.m.Bitcast(t.t_u, t.m.Emit(spv::Op::OpConvertFToS, t.t_i, {s0})));
      break;
    case 0x0c:
      set_u(t.m.Bitcast(
          t.t_u, t.m.Emit(spv::Op::OpConvertFToS, t.t_i,
                          {t.Ext1(GLSLstd450Floor, t.FAdd(s0, t.F32(0.5f)))})));
      break;
    case 0x0d:
      set_u(t.m.Bitcast(t.t_u, t.m.Emit(spv::Op::OpConvertFToS, t.t_i,
                                        {t.Ext1(GLSLstd450Floor, s0)})));
      break;
    case 0x0e: {
      const Id nibble = t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                                 {t.m.Bitcast(t.t_i, u0), t.U32(0), t.U32(4)});
      set_f(t.FMul(t.m.Emit(spv::Op::OpConvertSToF, t.t_f, {nibble}),
                   t.F32(1.0f / 16.0f)));
      break;
    }
    case 0x0a: {
      Id value = ApplyOutputModifier(t, s0, omod);
      if (clamp)
        value = t.FClamp01(value);
      t.SetVg(vdst, t.m.ExtInst(
                        t.t_u, GLSLstd450PackHalf2x16,
                        {t.m.CompositeConstruct(t.t_v2, {value, t.F32(0.f)})}));
      break;
    }
    case 0x0b:
      set_f(t.m.CompositeExtract(
          t.t_f, t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {u0}), 0));
      break;
    case 0x11:
    case 0x12:
    case 0x13:
    case 0x14: {
      const Id b = t.And(t.Shr(u0, t.U32((op - 0x11) * 8)), t.U32(0xFF));
      set_f(t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {b}));
      break;
    }
    case 0x20:
      set_f(t.Ext1(GLSLstd450Fract, s0));
      break;
    case 0x21:
      set_f(t.Ext1(GLSLstd450Trunc, s0));
      break;
    case 0x22:
      set_f(t.Ext1(GLSLstd450Ceil, s0));
      break;
    case 0x23:
      set_f(t.Ext1(GLSLstd450RoundEven, s0));
      break;
    case 0x24:
      set_f(t.Ext1(GLSLstd450Floor, s0));
      break;
    case 0x25:
      set_f(t.Ext1(GLSLstd450Exp2, s0));
      break;
    case 0x26:
      set_f(t.ClampInfToFltMax(t.Ext1(GLSLstd450Log2, s0)));
      break;
    case 0x27:
      set_f(t.Ext1(GLSLstd450Log2, s0));
      break;
    case 0x28:
      set_f(t.ClampInfToFltMax(t.FDiv(t.F32(1.0f), s0)));
      break;
    case 0x29:
      set_f(t.ConvertInfToZero(t.FDiv(t.F32(1.0f), s0)));
      break;
    case 0x2a:
    case 0x2b:
      set_f(t.FDiv(t.F32(1.0f), s0));
      break;
    case 0x2c:
      set_f(t.ClampInfToFltMax(t.Ext1(GLSLstd450InverseSqrt, s0)));
      break;
    case 0x2d:
      set_f(t.ConvertInfToZero(t.Ext1(GLSLstd450InverseSqrt, s0)));
      break;
    case 0x2e:
      set_f(t.Ext1(GLSLstd450InverseSqrt, s0));
      break;
    case 0x33:
      set_f(t.Ext1(GLSLstd450Sqrt, s0));
      break;
    case 0x35:
      set_f(t.Ext1(GLSLstd450Sin, t.FMul(s0, t.F32(6.28318530718f))));
      break;
    case 0x36:
      set_f(t.Ext1(GLSLstd450Cos, t.FMul(s0, t.F32(6.28318530718f))));
      break;
    case 0x37:
      set_u(t.Not(u0));
      break;
    case 0x38:
      set_u(t.BitRev(u0));
      break;
    case 0x39: {
      const Id msb = t.m.ExtInst(t.t_u, GLSLstd450FindUMsb, {u0});
      set_u(t.SelectB(t.IsZero(u0), t.U32(0xFFFFFFFFu), t.Sub(t.U32(31), msb)));
      break;
    }
    case 0x3a:
      set_u(t.m.ExtInst(t.t_u, GLSLstd450FindILsb, {u0}));
      break;
    case 0x3b: {
      const Id smsb = t.m.Bitcast(t.t_u, t.m.ExtInst(t.t_i, GLSLstd450FindSMsb,
                                                     {t.m.Bitcast(t.t_i, u0)}));
      set_u(t.SelectB(t.Eq(smsb, t.U32(0xFFFFFFFFu)), t.U32(0xFFFFFFFFu),
                      t.Sub(t.U32(31), smsb)));
      break;
    }
    default:
      WarnUnsupported("vop1", op);
      set_u(u0);
      break;
  }
}

void EmitVop2(Translator& t,
              u32 op,
              u32 vdst,
              Id s0,
              Id s1,
              u32 literal,
              bool clamp,
              u32 omod) {
  const auto set_f = [&](Id f) {
    f = ApplyOutputModifier(t, f, omod);
    t.SetVgF(vdst, clamp ? t.FClamp01(f) : f);
  };
  const auto set_u = [&](Id u) { t.SetVg(vdst, u); };
  const Id u0 = t.m.Bitcast(t.t_u, s0), u1 = t.m.Bitcast(t.t_u, s1);
  const Id i0 = t.m.Bitcast(t.t_i, s0), i1 = t.m.Bitcast(t.t_i, s1);
  const auto mul24_hi = [&](spv::Op wide_mul, Id a, Id b) {
    const Id prod = t.m.Emit(wide_mul, t.PairType(), {a, b});
    return t.m.CompositeExtract(t.t_u, prod, 1);
  };
  switch (op) {
    case 0x00: {
      set_f(t.SelectF(t.LaneActive(t.SgMask(106)), s1, s0));
      break;
    }
    case 0x01:
      t.SetSg(vdst, ReadLane(t, u0, u1));
      break;
    case 0x02: {
      const Id keep = t.m.Load(t.t_u, t.VgPtr(vdst));
      t.StorePrivate(
          t.VgPtr(vdst),
          t.SelectB(t.Eq(t.WaveLane(), t.And(u1, t.U32(63))), u0, keep));
      break;
    }
    case 0x03:
      set_f(t.FAdd(s0, s1));
      break;
    case 0x04:
      set_f(t.FSub(s0, s1));
      break;
    case 0x05:
      set_f(t.FSub(s1, s0));
      break;
    case 0x06:
      set_f(t.FAdd(t.LegacyMul(s0, s1), t.VgF(vdst)));
      break;
    case 0x07:
      set_f(t.LegacyMul(s0, s1));
      break;
    case 0x08:
      set_f(t.FMul(s0, s1));
      break;
    case 0x09:
      set_u(t.Mul(t.Sext24(u0), t.Sext24(u1)));
      break;
    case 0x0a:
      set_u(mul24_hi(spv::Op::OpSMulExtended, t.Sext24(u0), t.Sext24(u1)));
      break;
    case 0x0b:
      set_u(t.Mul(t.Low24(u0), t.Low24(u1)));
      break;
    case 0x0c:
      set_u(mul24_hi(spv::Op::OpUMulExtended, t.Low24(u0), t.Low24(u1)));
      break;
    case 0x0d:
      // Legacy min/max return vsrc1 on NaN, unlike FMin/FMax.
      set_f(t.SelectF(t.FLt(s0, s1), s0, s1));
      break;
    case 0x0f:
      set_f(t.Ext2(GLSLstd450NMin, s0, s1));
      break;
    case 0x0e:
      set_f(t.SelectF(t.FGt(s0, s1), s0, s1));
      break;
    case 0x10:
      set_f(t.Ext2(GLSLstd450NMax, s0, s1));
      break;
    case 0x11:
      set_u(t.SMin(u0, u1));
      break;
    case 0x12:
      set_u(t.SMax(u0, u1));
      break;
    case 0x13:
      set_u(t.UMin(u0, u1));
      break;
    case 0x14:
      set_u(t.UMax(u0, u1));
      break;
    case 0x15:
      set_u(t.Shr(u0, u1));
      break;
    case 0x16:
      set_u(t.Shr(u1, u0));
      break;
    case 0x17:
      set_u(t.Sar(u0, u1));
      break;
    case 0x18:
      set_u(t.Sar(u1, u0));
      break;
    case 0x19:
      set_u(t.Shl(u0, u1));
      break;
    case 0x1a:
      set_u(t.Shl(u1, u0));
      break;
    case 0x1b:
      set_u(t.And(u0, u1));
      break;
    case 0x1c:
      set_u(t.Or(u0, u1));
      break;
    case 0x1d:
      set_u(t.Xor(u0, u1));
      break;
    case 0x1e: {
      const Id ones = t.Sub(t.Shl(t.U32(1), u0), t.U32(1));
      set_u(t.Shl(ones, u1));
      break;
    }
    case 0x1f:
      set_f(t.FAdd(t.FMul(s0, s1), t.VgF(vdst)));
      break;
    case 0x20:
      set_f(t.FAdd(t.FMul(s0, t.m.Bitcast(t.t_f, t.U32(literal))), s1));
      break;
    case 0x21:
      set_f(t.FAdd(t.FMul(s0, s1), t.m.Bitcast(t.t_f, t.U32(literal))));
      break;
    case 0x22:
      set_u(t.Add(t.PopCount(u0), u1));
      break;
    case 0x23:
    case 0x24: {
      const Id lane = t.WaveLane();
      const Id lo_half = t.Ult(lane, t.U32(32));
      const Id mask =
          op == 0x23
              ? t.SelectB(lo_half, t.Sub(t.Shl(t.U32(1), lane), t.U32(1)),
                          t.U32(0xFFFFFFFFu))
              : t.SelectB(
                    lo_half, t.U32(0),
                    t.Sub(t.Shl(t.U32(1), t.Sub(lane, t.U32(32))), t.U32(1)));
      set_u(t.Add(t.PopCount(t.And(u0, mask)), u1));
      break;
    }
    case 0x25: {
      const CarryResult r = AddCarry(t, u0, u1);
      set_u(t.Add(u0, u1));
      t.SetLaneFlag(106, r.flag);
      break;
    }
    case 0x26: {
      const CarryResult r = SubBorrow(t, u0, u1);
      set_u(r.value);
      t.SetLaneFlag(106, r.flag);
      break;
    }
    case 0x27: {
      const CarryResult r = SubBorrow(t, u1, u0);
      set_u(r.value);
      t.SetLaneFlag(106, r.flag);
      break;
    }
    case 0x28: {
      const CarryResult r = AddCarry(t, u0, u1, t.LaneFlag(106));
      set_u(r.value);
      t.SetLaneFlag(106, r.flag);
      break;
    }
    case 0x29: {
      const CarryResult r = SubBorrow(t, u0, u1, t.LaneFlag(106));
      set_u(r.value);
      t.SetLaneFlag(106, r.flag);
      break;
    }
    case 0x2a: {
      const CarryResult r = SubBorrow(t, u1, u0, t.LaneFlag(106));
      set_u(r.value);
      t.SetLaneFlag(106, r.flag);
      break;
    }
    case 0x2b:
      set_f(t.m.ExtInst(t.t_f, GLSLstd450Ldexp, {s0, i1}));
      break;
    case 0x2d:
      set_u(t.m.ExtInst(t.t_u, GLSLstd450PackSnorm2x16,
                        {t.m.CompositeConstruct(t.t_v2, {s0, s1})}));
      break;
    case 0x2e:
      set_u(t.m.ExtInst(t.t_u, GLSLstd450PackUnorm2x16,
                        {t.m.CompositeConstruct(t.t_v2, {s0, s1})}));
      break;
    case 0x2f:
      if (clamp || omod)
        WarnUnsupported("vop2.cvt_pkrtz-output-modifier", op);
      t.SetVg(vdst, t.m.ExtInst(t.t_u, GLSLstd450PackHalf2x16,
                                {t.m.CompositeConstruct(t.t_v2, {s0, s1})}));
      break;
    case 0x30: {
      const Id lo = t.m.ExtInst(t.t_u, GLSLstd450UMin, {u0, t.U32(0xFFFFu)});
      const Id hi = t.m.ExtInst(t.t_u, GLSLstd450UMin, {u1, t.U32(0xFFFFu)});
      set_u(t.Or(lo, t.Shl(hi, t.U32(16))));
      break;
    }
    case 0x31: {
      const auto sat = [&](Id v) {
        const Id c =
            t.m.ExtInst(t.t_i, GLSLstd450SClamp,
                        {v, t.m.ConstI32(-32768), t.m.ConstI32(32767)});
        return t.And(t.m.Bitcast(t.t_u, c), t.U32(0xFFFFu));
      };
      set_u(t.Or(sat(i0), t.Shl(sat(i1), t.U32(16))));
      break;
    }
    default:
      WarnUnsupported("vop2", op);
      set_f(t.FMul(s0, s1));
      break;
  }
  (void)i0;
}

namespace {

Id FloatPredicate(Translator& t, u32 lo, Id a, Id b) {
  const auto f = [&](spv::Op o) { return t.m.Emit(o, t.t_bool, {a, b}); };
  const auto is_nan = [&](Id x) {
    return t.m.Emit(spv::Op::OpIsNan, t.t_bool, {x});
  };
  switch (lo) {
    case 1:
      return f(spv::Op::OpFOrdLessThan);
    case 2:
      return f(spv::Op::OpFOrdEqual);
    case 3:
      return f(spv::Op::OpFOrdLessThanEqual);
    case 4:
      return f(spv::Op::OpFOrdGreaterThan);
    case 5:
      return f(spv::Op::OpFOrdNotEqual);
    case 6:
      return f(spv::Op::OpFOrdGreaterThanEqual);
    case 7:
      return t.m.Emit(
          spv::Op::OpLogicalNot, t.t_bool,
          {t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {is_nan(a), is_nan(b)})});
    case 8:
      return t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {is_nan(a), is_nan(b)});
    case 9:
      return f(spv::Op::OpFUnordLessThan);
    case 10:
      return f(spv::Op::OpFUnordEqual);
    case 11:
      return f(spv::Op::OpFUnordLessThanEqual);
    case 12:
      return f(spv::Op::OpFUnordGreaterThan);
    case 13:
      return f(spv::Op::OpFUnordNotEqual);
    case 14:
      return f(spv::Op::OpFUnordGreaterThanEqual);
    case 15:
      return t.m.ConstBool(true);
    default:
      return 0;
  }
}

Id IntPredicate(Translator& t, u32 lo, bool is_signed, Id a, Id b) {
  const Id ai = t.m.Bitcast(t.t_i, a), bi = t.m.Bitcast(t.t_i, b);
  const auto s = [&](spv::Op o) { return t.m.Emit(o, t.t_bool, {ai, bi}); };
  const auto u = [&](spv::Op o) { return t.m.Emit(o, t.t_bool, {a, b}); };
  switch (lo) {
    case 1:
      return is_signed ? s(spv::Op::OpSLessThan) : u(spv::Op::OpULessThan);
    case 2:
      return t.Eq(a, b);
    case 3:
      return is_signed ? s(spv::Op::OpSLessThanEqual)
                       : u(spv::Op::OpULessThanEqual);
    case 4:
      return is_signed ? s(spv::Op::OpSGreaterThan)
                       : u(spv::Op::OpUGreaterThan);
    case 5:
      return t.m.Emit(spv::Op::OpINotEqual, t.t_bool, {a, b});
    case 6:
      return is_signed ? s(spv::Op::OpSGreaterThanEqual)
                       : u(spv::Op::OpUGreaterThanEqual);
    case 7:
      return t.m.ConstBool(true);
    default:
      return 0;
  }
}

struct Dword2 {
  Id lo, hi;
};

Id LOr(Translator& t, Id a, Id b) {
  return t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {a, b});
}
Id LNot(Translator& t, Id a) {
  return t.m.Emit(spv::Op::OpLogicalNot, t.t_bool, {a});
}
Id Slt32(Translator& t, Id a, Id b) {
  return t.m.Emit(spv::Op::OpSLessThan, t.t_bool,
                  {t.m.Bitcast(t.t_i, a), t.m.Bitcast(t.t_i, b)});
}
Id U64Eq(Translator& t, Dword2 a, Dword2 b) {
  return t.LAnd(t.Eq(a.lo, b.lo), t.Eq(a.hi, b.hi));
}
Id U64Lt(Translator& t, Dword2 a, Dword2 b) {
  return LOr(t, t.Ult(a.hi, b.hi), t.LAnd(t.Eq(a.hi, b.hi), t.Ult(a.lo, b.lo)));
}
Id I64Lt(Translator& t, Dword2 a, Dword2 b) {
  return LOr(t, Slt32(t, a.hi, b.hi),
             t.LAnd(t.Eq(a.hi, b.hi), t.Ult(a.lo, b.lo)));
}

Id Int64Predicate(Translator& t, u32 lo, bool is_signed, Dword2 a, Dword2 b) {
  const auto lt = [&](Dword2 x, Dword2 y) {
    return is_signed ? I64Lt(t, x, y) : U64Lt(t, x, y);
  };
  switch (lo) {
    case 0:
      return t.m.ConstBool(false);
    case 1:
      return lt(a, b);
    case 2:
      return U64Eq(t, a, b);
    case 3:
      return LOr(t, lt(a, b), U64Eq(t, a, b));
    case 4:
      return lt(b, a);
    case 5:
      return LNot(t, U64Eq(t, a, b));
    case 6:
      return LNot(t, lt(a, b));
    case 7:
      return t.m.ConstBool(true);
    default:
      return 0;
  }
}

Id F64IsNan(Translator& t, Dword2 x) {
  return t.LAnd(
      t.Eq(t.And(x.hi, t.U32(0x7ff00000u)), t.U32(0x7ff00000u)),
      LOr(t, t.IsNonZero(t.And(x.hi, t.U32(0x000fffffu))), t.IsNonZero(x.lo)));
}
Id F64IsZero(Translator& t, Dword2 x) {
  return t.LAnd(t.IsZero(t.And(x.hi, t.U32(0x7fffffffu))), t.IsZero(x.lo));
}
Dword2 F64OrderKey(Translator& t, Dword2 x) {
  const Id neg = t.IsNonZero(t.And(x.hi, t.U32(0x80000000u)));
  return {t.SelectB(neg, t.Not(x.lo), x.lo),
          t.SelectB(neg, t.Not(x.hi), t.Xor(x.hi, t.U32(0x80000000u)))};
}
Id F64Eq(Translator& t, Dword2 a, Dword2 b) {
  return t.LAnd(
      LNot(t, LOr(t, F64IsNan(t, a), F64IsNan(t, b))),
      LOr(t, U64Eq(t, a, b), t.LAnd(F64IsZero(t, a), F64IsZero(t, b))));
}
Id F64Lt(Translator& t, Dword2 a, Dword2 b) {
  return t.LAnd(LNot(t, LOr(t, F64IsNan(t, a), F64IsNan(t, b))),
                t.LAnd(U64Lt(t, F64OrderKey(t, a), F64OrderKey(t, b)),
                       LNot(t, t.LAnd(F64IsZero(t, a), F64IsZero(t, b)))));
}

Id Float64Predicate(Translator& t, u32 lo, Dword2 a, Dword2 b) {
  const Id nan = LOr(t, F64IsNan(t, a), F64IsNan(t, b));
  switch (lo) {
    case 0:
      return t.m.ConstBool(false);
    case 1:
      return F64Lt(t, a, b);
    case 2:
      return F64Eq(t, a, b);
    case 3:
      return LOr(t, F64Lt(t, a, b), F64Eq(t, a, b));
    case 4:
      return F64Lt(t, b, a);
    case 5:
      return t.LAnd(LNot(t, nan), LNot(t, F64Eq(t, a, b)));
    case 6:
      return LOr(t, F64Lt(t, b, a), F64Eq(t, a, b));
    case 7:
      return LNot(t, nan);
    case 8:
      return nan;
    case 9:
      return LOr(t, nan, F64Lt(t, a, b));
    case 10:
      return LOr(t, nan, F64Eq(t, a, b));
    case 11:
      return LOr(t, nan, LOr(t, F64Lt(t, a, b), F64Eq(t, a, b)));
    case 12:
      return LOr(t, nan, F64Lt(t, b, a));
    case 13:
      return LNot(t, F64Eq(t, a, b));
    case 14:
      return LOr(t, nan, LOr(t, F64Lt(t, b, a), F64Eq(t, a, b)));
    case 15:
      return t.m.ConstBool(true);
    default:
      return 0;
  }
}

Id FloatClassPredicate(Translator& t, Id bits, Id mask) {
  const Id sign = t.IsNonZero(t.And(bits, t.U32(0x80000000u)));
  const Id exponent = t.And(bits, t.U32(0x7f800000u));
  const Id mantissa = t.And(bits, t.U32(0x007fffffu));
  const Id exp_zero = t.IsZero(exponent);
  const Id exp_all_ones = t.Eq(exponent, t.U32(0x7f800000u));
  const Id mantissa_zero = t.IsZero(mantissa);
  const auto logical_not = [&](Id value) {
    return t.m.Emit(spv::Op::OpLogicalNot, t.t_bool, {value});
  };
  const auto logical_or = [&](Id a, Id b) {
    return t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {a, b});
  };
  const Id negative = sign, positive = logical_not(sign);
  const Id nan = t.LAnd(exp_all_ones, logical_not(mantissa_zero));
  const Id quiet = t.IsNonZero(t.And(mantissa, t.U32(0x00400000u)));
  const Id infinity = t.LAnd(exp_all_ones, mantissa_zero);
  const Id zero = t.LAnd(exp_zero, mantissa_zero);
  const Id subnormal = t.LAnd(exp_zero, logical_not(mantissa_zero));
  const Id normal = logical_not(logical_or(exp_zero, exp_all_ones));

  Id classes = t.U32(0);
  const auto add_class = [&](u32 bit, Id condition) {
    classes = t.Or(classes, t.SelectB(condition, t.U32(1u << bit), t.U32(0)));
  };
  add_class(0, t.LAnd(nan, logical_not(quiet)));
  add_class(1, t.LAnd(nan, quiet));
  add_class(2, t.LAnd(infinity, negative));
  add_class(3, t.LAnd(normal, negative));
  add_class(4, t.LAnd(subnormal, negative));
  add_class(5, t.LAnd(zero, negative));
  add_class(6, t.LAnd(zero, positive));
  add_class(7, t.LAnd(subnormal, positive));
  add_class(8, t.LAnd(normal, positive));
  add_class(9, t.LAnd(infinity, positive));
  return t.IsNonZero(t.And(classes, mask));
}

}  // namespace

void EmitVopc(Translator& t,
              u32 op,
              Id s0f,
              Id s1f,
              Id s0u,
              Id s1u,
              u32 dst,
              Id s0_hi,
              Id s1_hi) {
  const Dword2 a{s0u, s0_hi ? s0_hi : t.U32(0)};
  const Dword2 b{s1u, s1_hi ? s1_hi : t.U32(0)};
  Id cond = 0;
  if (op == 0x88 || op == 0x98) {
    cond = FloatClassPredicate(t, s0u, s1u);
  } else if (op <= 0x3F) {
    cond = op >= 0x20 ? Float64Predicate(t, op & 0xF, a, b)
                      : FloatPredicate(t, op & 0xF, s0f, s1f);
  } else if (op <= 0x7F) {
    cond = op >= 0x60 ? Float64Predicate(t, op & 0xF, a, b)
                      : FloatPredicate(t, op & 0xF, s0f, s1f);
  } else {
    const bool is_signed = op <= 0xBF;
    const bool is_64 = (op >= 0xA0 && op <= 0xBF) || op >= 0xE0;
    cond = is_64 ? Int64Predicate(t, op & 0x7, is_signed, a, b)
                 : IntPredicate(t, op & 0x7, is_signed, s0u, s1u);
  }
  const Id predicate = cond ? t.MaskOf(cond) : t.U32(0);
  const Id result = t.And(predicate, t.Exec());
  if (!(op & 0x10) || !t.rdna_sources)
    t.SetMask(dst, result);
  if (op & 0x10)
    t.SetMask(126, result);
}

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
              u32 omod,
              Id s1_hi) {
  const Id u0 = t.m.Bitcast(t.t_u, s0), u1 = t.m.Bitcast(t.t_u, s1),
           u2 = t.m.Bitcast(t.t_u, s2);
  const auto set_f = [&](Id f) {
    f = ApplyOutputModifier(t, f, omod);
    t.SetVgF(vdst, clamp ? t.FClamp01(f) : f);
  };
  const auto set_u = [&](Id u) { t.SetVg(vdst, u); };

  if (op < 0x100) {
    EmitVopc(t, op, s0, s1, u0, u1, vdst, s0_hi, s1_hi);
    return;
  }
  if (op == 0x100) {
    set_u(t.SelectB(t.LaneActive(u2), u1, u0));
    return;
  }
  if (op >= 0x125 && op <= 0x12a) {
    CarryResult r;
    if (op == 0x125)
      r = AddCarry(t, u0, u1);
    else if (op == 0x126)
      r = SubBorrow(t, u0, u1);
    else if (op == 0x127)
      r = SubBorrow(t, u1, u0);
    else if (op == 0x128)
      r = AddCarry(t, u0, u1, u2);
    else if (op == 0x129)
      r = SubBorrow(t, u0, u1, u2);
    else
      r = SubBorrow(t, u1, u0, u2);
    set_u(r.value);
    t.SetLaneFlag(sdst, r.flag);
    return;
  }
  if (op >= 0x100 && op < 0x140) {
    EmitVop2(t, op - 0x100, vdst, s0, s1, 0, clamp, omod);
    return;
  }
  if (op >= 0x180 && op < 0x200) {
    if (op == 0x18b) {
      set_f(s0);
      return;
    }
    EmitVop1(t, op - 0x180, vdst, s0, clamp, omod);
    return;
  }

  const auto mul_hi = [&](spv::Op wide_mul) {
    return t.m.CompositeExtract(t.t_u,
                                t.m.Emit(wide_mul, t.PairType(), {u0, u1}), 1);
  };
  const auto med3 = [&](Id (Translator::*mn)(Id, Id),
                        Id (Translator::*mx)(Id, Id)) {
    return (t.*mx)((t.*mn)(u0, u1), (t.*mn)((t.*mx)(u0, u1), u2));
  };
  switch (op) {
    case 0x140:
      set_f(t.FAdd(t.LegacyMul(s0, s1), s2));
      break;
    case 0x141:
      set_f(t.FAdd(t.FMul(s0, s1), s2));
      break;
    case 0x14b:
      set_f(t.m.ExtInst(t.t_f, GLSLstd450Fma, {s0, s1, s2}));
      break;
    case 0x142:
      set_u(t.Add(t.Mul(t.Sext24(u0), t.Sext24(u1)), u2));
      break;
    case 0x143:
      set_u(t.Add(t.Mul(t.Low24(u0), t.Low24(u1)), u2));
      break;
    case 0x148: {
      const Id off = t.And(u1, t.U32(31));
      const Id width = t.UMin(t.And(u2, t.U32(31)), t.Sub(t.U32(32), off));
      set_u(t.m.Emit(spv::Op::OpBitFieldUExtract, t.t_u, {u0, off, width}));
      break;
    }
    case 0x149: {
      const Id off = t.And(u1, t.U32(31));
      const Id width = t.UMin(t.And(u2, t.U32(31)), t.Sub(t.U32(32), off));
      set_u(t.m.Bitcast(t.t_u, t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                                        {t.m.Bitcast(t.t_i, u0), off, width})));
      break;
    }
    case 0x14a:
      set_u(t.Or(t.And(u0, u1), t.And(t.Not(u0), u2)));
      break;
    case 0x144:
    case 0x145:
    case 0x146:
    case 0x147: {
      const Id ax = t.Ext1(GLSLstd450FAbs, s0);
      const Id ay = t.Ext1(GLSLstd450FAbs, s1);
      const Id az = t.Ext1(GLSLstd450FAbs, s2);
      const Id z_ge_x =
          t.m.Emit(spv::Op::OpFOrdGreaterThanEqual, t.t_bool, {az, ax});
      const Id z_ge_y =
          t.m.Emit(spv::Op::OpFOrdGreaterThanEqual, t.t_bool, {az, ay});
      const Id z_major = t.LAnd(z_ge_x, z_ge_y);
      const Id y_ge_x =
          t.m.Emit(spv::Op::OpFOrdGreaterThanEqual, t.t_bool, {ay, ax});
      const Id not_z = t.m.Emit(spv::Op::OpLogicalNot, t.t_bool, {z_major});
      const Id y_major = t.LAnd(not_z, y_ge_x);
      const Id x_neg =
          t.m.Emit(spv::Op::OpFOrdLessThan, t.t_bool, {s0, t.F32(0.0f)});
      const Id y_neg =
          t.m.Emit(spv::Op::OpFOrdLessThan, t.t_bool, {s1, t.F32(0.0f)});
      const Id z_neg =
          t.m.Emit(spv::Op::OpFOrdLessThan, t.t_bool, {s2, t.F32(0.0f)});

      Id result;
      if (op == 0x144) {
        const Id x_face = t.SelectF(x_neg, t.F32(1.0f), t.F32(0.0f));
        const Id y_face = t.SelectF(y_neg, t.F32(3.0f), t.F32(2.0f));
        const Id z_face = t.SelectF(z_neg, t.F32(5.0f), t.F32(4.0f));
        result = t.SelectF(z_major, z_face, t.SelectF(y_major, y_face, x_face));
      } else if (op == 0x145) {
        const Id x_sc = t.SelectF(x_neg, s2, t.FNeg(s2));
        const Id z_sc = t.SelectF(z_neg, t.FNeg(s0), s0);
        result = t.SelectF(z_major, z_sc, t.SelectF(y_major, s0, x_sc));
      } else if (op == 0x146) {
        const Id y_tc = t.SelectF(y_neg, t.FNeg(s2), s2);
        result = t.SelectF(y_major, y_tc, t.FNeg(s1));
      } else {
        const Id major = t.SelectF(z_major, s2, t.SelectF(y_major, s1, s0));
        result = t.FMul(t.F32(2.0f), major);
      }
      set_f(result);
      break;
    }
    case 0x14d: {
      Id r = t.U32(0);
      for (int b = 0; b < 4; b++) {
        const Id sh = t.U32(static_cast<u32>(b) * 8), mask = t.U32(0xFF);
        const Id a = t.And(t.Shr(u0, sh), mask),
                 bb = t.And(t.Shr(u1, sh), mask);
        const Id cc = t.And(t.Shr(u2, sh), t.U32(1));
        const Id avg = t.Shr(t.Add(t.Add(a, bb), cc), t.U32(1));
        r = t.Or(r, t.Shl(avg, sh));
      }
      set_u(r);
      break;
    }
    case 0x14e: {
      const Id sh = t.And(u2, t.U32(31));
      const Id lo = t.Shr(u1, sh);
      const Id hi =
          t.SelectB(t.IsZero(sh), t.U32(0), t.Shl(u0, t.Sub(t.U32(32), sh)));
      set_u(t.Or(lo, hi));
      break;
    }
    case 0x14f: {
      const Id sh = t.Mul(t.And(u2, t.U32(3)), t.U32(8));
      const Id lo = t.Shr(u1, sh);
      const Id hi =
          t.SelectB(t.IsZero(sh), t.U32(0), t.Shl(u0, t.Sub(t.U32(32), sh)));
      set_u(t.Or(lo, hi));
      break;
    }
    case 0x151:
      set_f(t.Ext2(GLSLstd450NMin, t.Ext2(GLSLstd450NMin, s0, s1), s2));
      break;
    case 0x152:
      set_u(t.SMin(t.SMin(u0, u1), u2));
      break;
    case 0x153:
      set_u(t.UMin(t.UMin(u0, u1), u2));
      break;
    case 0x154:
      set_f(t.Ext2(GLSLstd450NMax, t.Ext2(GLSLstd450NMax, s0, s1), s2));
      break;
    case 0x155:
      set_u(t.SMax(t.SMax(u0, u1), u2));
      break;
    case 0x156:
      set_u(t.UMax(t.UMax(u0, u1), u2));
      break;
    case 0x157: {
      const Id lo = t.Ext2(GLSLstd450NMin, s0, s1);
      const Id hi = t.Ext2(GLSLstd450NMax, s0, s1);
      set_f(t.m.ExtInst(t.t_f, GLSLstd450NClamp, {s2, lo, hi}));
      break;
    }
    case 0x158:
      set_u(med3(&Translator::SMin, &Translator::SMax));
      break;
    case 0x159:
      set_u(med3(&Translator::UMin, &Translator::UMax));
      break;
    case 0x15d:
      set_u(t.Add(t.Sub(t.UMax(u0, u1), t.UMin(u0, u1)), u2));
      break;
    case 0x15e: {
      const Id sat = t.m.ExtInst(t.t_f, GLSLstd450NClamp,
                                 {s0, t.F32(0.0f), t.F32(255.0f)});
      const Id byte =
          t.And(t.m.Emit(spv::Op::OpConvertFToU, t.t_u, {sat}), t.U32(0xFF));
      const Id shift = t.Mul(t.And(u1, t.U32(3)), t.U32(8));
      const Id mask = t.Shl(t.U32(0xFF), shift);
      set_u(t.Or(t.And(u2, t.Not(mask)), t.And(t.Shl(byte, shift), mask)));
      break;
    }
    case 0x15f:
      WarnUnsupported("vop3.div-fixup", op);
      set_f(t.FDiv(s2, s1));
      break;
    case 0x16d:
      WarnUnsupported("vop3.div-scale", op);
      set_f(s0);
      t.SetSdst(sdst, 0, t.U32(0));
      break;
    case 0x16f:
      set_f(t.m.ExtInst(t.t_f, GLSLstd450Fma, {s0, s1, s2}));
      break;
    case 0x169:
    case 0x16b:
      set_u(t.Mul(u0, u1));
      break;
    case 0x16a:
      set_u(mul_hi(spv::Op::OpUMulExtended));
      break;
    case 0x16c:
      set_u(mul_hi(spv::Op::OpSMulExtended));
      break;
    case 0x161:
    case 0x162:
    case 0x163: {
      const Id n = t.And(u1, t.U32(63)), n_lo = t.And(n, t.U32(31));
      const Id ge32 = t.Uge(n, t.U32(32));
      const Id zero = t.IsZero(n);
      const Id inv = t.And(t.Sub(t.U32(32), n_lo), t.U32(31));
      const Id lo_in = u0, hi_in = s0_hi;
      Id lo, hi;
      if (op == 0x161) {
        const Id cross = t.SelectB(zero, t.U32(0), t.Shr(lo_in, inv));
        lo = t.SelectB(ge32, t.U32(0), t.Shl(lo_in, n_lo));
        hi = t.SelectB(ge32, t.Shl(lo_in, n_lo),
                       t.Or(t.Shl(hi_in, n_lo), cross));
      } else {
        const Id cross = t.SelectB(zero, t.U32(0), t.Shl(hi_in, inv));
        const Id hi_shift =
            op == 0x162 ? t.Shr(hi_in, n_lo) : t.Sar(hi_in, n_lo);
        lo = t.SelectB(ge32, hi_shift, t.Or(t.Shr(lo_in, n_lo), cross));
        hi = t.SelectB(ge32, op == 0x162 ? t.U32(0) : t.Sar(hi_in, t.U32(31)),
                       hi_shift);
      }
      set_u(lo);
      if (vdst + 1 < 256)
        t.SetVg(vdst + 1, hi);
      break;
    }
    case 0x176:
    case 0x177: {
      const bool sgn = (op == 0x177);
      const Id prod =
          t.m.Emit(sgn ? spv::Op::OpSMulExtended : spv::Op::OpUMulExtended,
                   t.PairType(), {u0, u1});
      const Id p_lo = t.m.CompositeExtract(t.t_u, prod, 0);
      const Id p_hi = t.m.CompositeExtract(t.t_u, prod, 1);
      const CarryResult lo = AddCarry(t, p_lo, u2);
      const CarryResult hi = AddCarry(t, p_hi, s2_hi, lo.flag);
      set_u(lo.value);
      if (vdst + 1 < 256)
        t.SetVg(vdst + 1, hi.value);
      Id bit64 = hi.flag;
      if (sgn)
        bit64 = t.Xor(bit64,
                      t.Xor(t.Shr(p_hi, t.U32(31)), t.Shr(s2_hi, t.U32(31))));
      t.SetSdst(sdst, 0, t.And(bit64, t.U32(1)));
      break;
    }
    case 0x360:
    case 0x361:
      if (!t.rdna_sources) {
        WarnUnsupported("vop3", op);
        set_f(s0);
        break;
      }
      EmitVop2(t, op == 0x360 ? 0x01 : 0x02, vdst, s0, s1, 0, false, 0);
      break;
    default:
      WarnUnsupported("vop3", op);
      set_f(s0);
      break;
  }
}

}  // namespace gpu::gcn

#endif
