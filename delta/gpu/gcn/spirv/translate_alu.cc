/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN -> SPIR-V: scalar + vector ALU emitters. Shared by all stages (VS/PS/CS):
 * integer ops bitcast through the uint/int types as needed, so passing
 * float-typed sources is lossless. Opcode numbering is the GFX7 (Sea Islands /
 * Liverpool) ISA:
 * https://docs.amd.com/v/u/en-US/sea-islands-instruction-set-architecture_0
 */

#ifdef DELTA_HAVE_SPIRV_BACKEND

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

// Signed-overflow bit for a + b = r: (a^r) & (b^r), sign bit.
Id SignedAddOverflow(Translator& t, Id a, Id b, Id r) {
  return t.IsNonZero(
      t.And(t.And(t.Xor(a, r), t.Xor(b, r)), t.U32(0x80000000u)));
}

Id ApplyOutputModifier(Translator& t, Id value, uint32_t omod) {
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

bool IsVop3b(uint32_t op) {
  return (op >= 0x125 && op <= 0x12a) || op == 0x16d || op == 0x16e;
}

// ---- SOP1 -------------------------------------------------------------------
void EmitSop1(Translator& t, const Inst& inst) {
  const uint32_t w = inst.raw[0];
  const uint32_t op = inst.opcode, sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF;
  const Id a = t.SrcRaw(ssrc0, inst.literal);
  const Id a_hi = t.SrcRawHi(ssrc0, inst.literal, false);
  switch (op) {
    case 0x03:
      t.SetSdst(sdst, 0, a);
      break;    // s_mov_b32
    case 0x04:  // s_mov_b64
      t.SetSdst(sdst, 0, a);
      t.SetSdst(sdst, 1, a_hi);
      break;
    case 0x05:  // s_cmov_b32: SCC ? src : dst
      t.SetSdst(sdst, 0, t.SelectNz(t.Scc(), a, t.Sdst(sdst)));
      break;
    case 0x06:  // s_cmov_b64
      t.SetSdst(sdst, 0, t.SelectNz(t.Scc(), a, t.Sdst(sdst)));
      t.SetSdst(sdst, 1, t.SelectNz(t.Scc(), a_hi, t.Sdst(sdst, 1)));
      break;
    case 0x07: {  // s_not_b32
      const Id r = t.Not(a);
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(t.IsNonZero(r));
      break;
    }
    case 0x08: {  // s_not_b64
      const Id lo = t.Not(a), hi = t.Not(a_hi);
      t.SetSdst(sdst, 0, lo);
      t.SetSdst(sdst, 1, hi);
      t.SetSccBool(t.IsNonZero(t.Or(lo, hi)));
      break;
    }
    // s_wqm (whole quad mode): a lane's bit is set if any lane in its quad is.
    // Single-lane model -> identity; SCC = (result != 0).
    case 0x09:  // s_wqm_b32
      t.SetSdst(sdst, 0, a);
      t.SetSccBool(t.IsNonZero(a));
      break;
    case 0x0a:  // s_wqm_b64
      t.SetSdst(sdst, 0, a);
      t.SetSdst(sdst, 1, a_hi);
      t.SetSccBool(t.IsNonZero(t.Or(a, a_hi)));
      break;
    case 0x0b:
      t.SetSdst(sdst, 0, t.BitRev(a));
      break;      // s_brev_b32
    case 0x0d: {  // s_bcnt0_i32_b32
      const Id r = t.Sub(t.U32(32), t.PopCount(a));
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(t.IsNonZero(r));
      break;
    }
    case 0x0e: {  // s_bcnt0_i32_b64
      const Id r = t.Sub(t.U32(64), t.Add(t.PopCount(a), t.PopCount(a_hi)));
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(t.IsNonZero(r));
      break;
    }
    case 0x0f: {  // s_bcnt1_i32_b32
      const Id r = t.PopCount(a);
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(t.IsNonZero(r));
      break;
    }
    case 0x10: {  // s_bcnt1_i32_b64
      const Id r = t.Add(t.PopCount(a), t.PopCount(a_hi));
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(t.IsNonZero(r));
      break;
    }
    case 0x11:  // s_ff0_i32_b32
      t.SetSdst(sdst, 0, t.m.ExtInst(t.t_u, GLSLstd450FindILsb, {t.Not(a)}));
      break;
    case 0x13:  // s_ff1_i32_b32
      t.SetSdst(sdst, 0, t.m.ExtInst(t.t_u, GLSLstd450FindILsb, {a}));
      break;
    case 0x15: {  // s_flbit_i32_b32: count from the MSB; -1 if src == 0
      const Id msb = t.m.ExtInst(t.t_u, GLSLstd450FindUMsb, {a});
      t.SetSdst(
          sdst, 0,
          t.SelectB(t.IsZero(a), t.U32(0xFFFFFFFFu), t.Sub(t.U32(31), msb)));
      break;
    }
    case 0x17: {  // s_flbit_i32: leading-sign-bit count; -1 if src is 0 or -1
      const Id smsb = t.m.Bitcast(t.t_u, t.m.ExtInst(t.t_i, GLSLstd450FindSMsb,
                                                     {t.m.Bitcast(t.t_i, a)}));
      t.SetSdst(sdst, 0,
                t.SelectB(t.Eq(smsb, t.U32(0xFFFFFFFFu)), t.U32(0xFFFFFFFFu),
                          t.Sub(t.U32(31), smsb)));
      break;
    }
    // The main VS uses these to call/return from its fetch shader. Vertex
    // attributes are decoded from that fetch program and supplied as Vulkan
    // inputs, so no runtime jump remains in the translated shader.
    case 0x20:
    case 0x21:
      break;  // s_setpc_b64 / s_swappc_b64
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2a:
    case 0x2b: {
      // s_{and,or,xor,andn2,orn2,nand,nor,xnor}_saveexec_b64
      const Id old_exec = t.Exec();
      Id new_exec;
      if (op == 0x24)
        new_exec = t.And(old_exec, a);
      else if (op == 0x25)
        new_exec = t.Or(old_exec, a);
      else if (op == 0x26)
        new_exec = t.Xor(old_exec, a);
      else if (op == 0x27)
        new_exec = t.And(a, t.Not(old_exec));
      else if (op == 0x28)
        new_exec = t.Or(a, t.Not(old_exec));
      else if (op == 0x29)
        new_exec = t.Not(t.And(a, old_exec));
      else if (op == 0x2a)
        new_exec = t.Not(t.Or(a, old_exec));
      else
        new_exec = t.Not(t.Xor(a, old_exec));
      new_exec = t.And(new_exec, t.U32(1));
      t.SetSdst(sdst, 0, old_exec);
      t.SetSdst(sdst, 1, t.U32(0));
      t.SetSg(126, new_exec);
      t.SetSccBool(t.IsNonZero(new_exec));
      break;
    }
    case 0x34: {  // s_abs_i32
      const Id r = t.m.Bitcast(
          t.t_u, t.m.ExtInst(t.t_i, GLSLstd450SAbs, {t.m.Bitcast(t.t_i, a)}));
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(t.IsNonZero(r));
      break;
    }
    case 0x1f: {  // s_getpc_b64: sdst = PC + 4, i.e. the NEXT instruction
      // Shaders use this to build an absolute address from their own location
      // -- typically s_getpc_b64 / s_add_u32 / s_addc_u32 / s_load_dwordx*, to
      // reach a descriptor the toolchain stored next to the code. Guest memory
      // is identity-mapped, so the program's guest address is its code pointer,
      // and s_getpc is one dword wide.
      const uint32_t next_off = (inst.pc + 1) * 4u;
      if (t.pc_base_var) {
        // Graphics: the module is cached by code CONTENT and reused wherever
        // the title streams this shader, so the address arrives per draw in
        // the push range. 64-bit add from u32 halves (no Int64 capability);
        // the carry is (lo + off) having wrapped below off.
        const Id p_u = t.m.TypePointer(spv::StorageClass::PushConstant, t.t_u);
        const auto half = [&](uint32_t i) {
          return t.m.Load(t.t_u,
                          t.m.AccessChain(p_u, t.pc_base_var,
                                          {t.U32(t.pc_base_member + i)}));
        };
        const Id lo = t.Add(half(0), t.U32(next_off));
        const Id carry =
            t.SelectB(t.Ult(lo, t.U32(next_off)), t.U32(1), t.U32(0));
        t.SetSdst(sdst, 0, lo);
        t.SetSdst(sdst, 1, t.Add(half(1), carry));
      } else {  // compute (address-keyed cache), or the 128-byte push floor
        const uint64_t next = t.program_base + next_off;
        t.SetSdst(sdst, 0, t.U32(static_cast<uint32_t>(next)));
        t.SetSdst(sdst, 1, t.U32(static_cast<uint32_t>(next >> 32)));
      }
      break;
    }
    default:
      WarnUnsupported("sop1", op);
      break;
  }
}

// ---- SOP2 -------------------------------------------------------------------
void EmitSop2(Translator& t, const Inst& inst) {
  const uint32_t w = inst.raw[0];
  const uint32_t op = inst.opcode, sdst = (w >> 16) & 0x7F;
  const uint32_t s0f = w & 0xFF, s1f = (w >> 8) & 0xFF;
  const Id a = t.SrcRaw(s0f, inst.literal), b = t.SrcRaw(s1f, inst.literal);
  const Id a_hi = t.SrcRawHi(s0f, inst.literal, op == 0x23);
  const Id b_hi = t.SrcRawHi(s1f, inst.literal, false);
  Id r = 0, r_hi = 0;
  bool scc = false, wide_scc = false;

  // 64-bit shifts: kind 0 = logical left, 1 = logical right, 2 = arithmetic.
  const auto shift64 = [&](uint32_t kind) {
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
    case 0x00: {  // s_add_u32: SCC = unsigned carry-out
      const CarryResult c = AddCarry(t, a, b);
      r = c.value;
      t.SetSccBool(t.IsNonZero(c.flag));
      break;
    }
    case 0x01: {  // s_sub_u32: SCC = unsigned borrow
      const CarryResult c = SubBorrow(t, a, b);
      r = c.value;
      t.SetSccBool(t.IsNonZero(c.flag));
      break;
    }
    case 0x02:  // s_add_i32: SCC = signed overflow
      r = t.Add(a, b);
      t.SetSccBool(SignedAddOverflow(t, a, b, r));
      break;
    case 0x03:  // s_sub_i32: overflow = (a^b) & (a^r), sign bit
      r = t.Sub(a, b);
      t.SetSccBool(t.IsNonZero(
          t.And(t.And(t.Xor(a, b), t.Xor(a, r)), t.U32(0x80000000u))));
      break;
    case 0x04: {  // s_addc_u32: a + b + SCC, SCC = carry-out
      const CarryResult c = AddCarry(t, a, b, t.Scc());
      r = c.value;
      t.SetSccBool(t.IsNonZero(c.flag));
      break;
    }
    case 0x05: {  // s_subb_u32: a - b - SCC, SCC = borrow-out
      const CarryResult c = SubBorrow(t, a, b, t.Scc());
      r = c.value;
      t.SetSccBool(t.IsNonZero(c.flag));
      break;
    }
    case 0x06:  // s_min_i32
      r = t.SMin(a, b);
      t.SetSccBool(t.m.Emit(spv::Op::OpSLessThan, t.t_bool,
                            {t.m.Bitcast(t.t_i, a), t.m.Bitcast(t.t_i, b)}));
      break;
    case 0x07:  // s_min_u32
      r = t.UMin(a, b);
      t.SetSccBool(t.Ult(a, b));
      break;
    case 0x08:  // s_max_i32
      r = t.SMax(a, b);
      t.SetSccBool(t.m.Emit(spv::Op::OpSGreaterThan, t.t_bool,
                            {t.m.Bitcast(t.t_i, a), t.m.Bitcast(t.t_i, b)}));
      break;
    case 0x09:  // s_max_u32
      r = t.UMax(a, b);
      t.SetSccBool(t.m.Emit(spv::Op::OpUGreaterThan, t.t_bool, {a, b}));
      break;
    case 0x0a:
      r = t.SelectNz(t.Scc(), a, b);
      break;    // s_cselect_b32
    case 0x0b:  // s_cselect_b64
      r = t.SelectNz(t.Scc(), a, b);
      r_hi = t.SelectNz(t.Scc(), a_hi, b_hi);
      break;
    case 0x0e:
      r = t.And(a, b);
      scc = true;
      break;  // s_and_b32
    case 0x0f:
      r = t.And(a, b);
      r_hi = t.And(a_hi, b_hi);
      scc = wide_scc = true;
      break;
    case 0x10:
      r = t.Or(a, b);
      scc = true;
      break;  // s_or_b32
    case 0x11:
      r = t.Or(a, b);
      r_hi = t.Or(a_hi, b_hi);
      scc = wide_scc = true;
      break;
    case 0x12:
      r = t.Xor(a, b);
      scc = true;
      break;  // s_xor_b32
    case 0x13:
      r = t.Xor(a, b);
      r_hi = t.Xor(a_hi, b_hi);
      scc = wide_scc = true;
      break;
    case 0x14:
      r = t.And(a, t.Not(b));
      scc = true;
      break;  // s_andn2_b32
    case 0x15:
      r = t.And(a, t.Not(b));
      r_hi = t.And(a_hi, t.Not(b_hi));
      scc = wide_scc = true;
      break;
    case 0x16:
      r = t.Or(a, t.Not(b));
      scc = true;
      break;  // s_orn2_b32
    case 0x17:
      r = t.Or(a, t.Not(b));
      r_hi = t.Or(a_hi, t.Not(b_hi));
      scc = wide_scc = true;
      break;
    case 0x18:
      r = t.Not(t.And(a, b));
      scc = true;
      break;  // s_nand_b32
    case 0x19:
      r = t.Not(t.And(a, b));
      r_hi = t.Not(t.And(a_hi, b_hi));
      scc = wide_scc = true;
      break;
    case 0x1a:
      r = t.Not(t.Or(a, b));
      scc = true;
      break;  // s_nor_b32
    case 0x1b:
      r = t.Not(t.Or(a, b));
      r_hi = t.Not(t.Or(a_hi, b_hi));
      scc = wide_scc = true;
      break;
    case 0x1c:
      r = t.Not(t.Xor(a, b));
      scc = true;
      break;  // s_xnor_b32
    case 0x1d:
      r = t.Not(t.Xor(a, b));
      r_hi = t.Not(t.Xor(a_hi, b_hi));
      scc = wide_scc = true;
      break;
    case 0x1e:
      r = t.Shl(a, b);
      scc = true;
      break;  // s_lshl_b32
    case 0x1f:
      shift64(0);
      break;  // s_lshl_b64
    case 0x20:
      r = t.Shr(a, b);
      scc = true;
      break;  // s_lshr_b32
    case 0x21:
      shift64(1);
      break;  // s_lshr_b64
    case 0x22:
      r = t.Sar(a, b);
      scc = true;
      break;  // s_ashr_i32
    case 0x23:
      shift64(2);
      break;      // s_ashr_i64
    case 0x24: {  // s_bfm_b32: mask = ((1 << width) - 1) << offset
      const Id width = t.And(a, t.U32(31)), off = t.And(b, t.U32(31));
      r = t.Shl(t.Sub(t.Shl(t.U32(1), width), t.U32(1)), off);
      break;
    }
    case 0x26:
      r = t.Mul(a, b);
      break;      // s_mul_i32
    case 0x27: {  // s_bfe_u32: offset = b[4:0], width = b[22:16]
      const Id off = t.And(b, t.U32(31));
      const Id width = t.UMin(t.And(t.Shr(b, t.U32(16)), t.U32(0x7F)),
                              t.Sub(t.U32(32), off));
      r = t.m.Emit(spv::Op::OpBitFieldUExtract, t.t_u, {a, off, width});
      scc = true;
      break;
    }
    case 0x28: {  // s_bfe_i32: signed
      const Id off = t.And(b, t.U32(31));
      const Id width = t.UMin(t.And(t.Shr(b, t.U32(16)), t.U32(0x7F)),
                              t.Sub(t.U32(32), off));
      r = t.m.Bitcast(t.t_u, t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                                      {t.m.Bitcast(t.t_i, a), off, width}));
      scc = true;
      break;
    }
    case 0x29: {  // s_bfe_u64: 64-bit unsigned bitfield extract (off=b[5:0],
                  // width=b[22:16])
      const Id off = t.And(b, t.U32(63));
      const Id width = t.UMin(t.And(t.Shr(b, t.U32(16)), t.U32(0x7F)),
                              t.Sub(t.U32(64), off));
      // 64-bit logical shift-right of {a, a_hi} by off.
      const Id n_lo = t.And(off, t.U32(31));
      const Id ge32 = t.Uge(off, t.U32(32));
      const Id inv = t.And(t.Sub(t.U32(32), n_lo), t.U32(31));
      const Id cross = t.SelectB(t.IsZero(off), t.U32(0), t.Shl(a_hi, inv));
      const Id lo_small = t.Or(t.Shr(a, n_lo), cross);
      const Id sh_lo = t.SelectB(ge32, t.Shr(a_hi, n_lo), lo_small);
      const Id sh_hi = t.SelectB(ge32, t.U32(0), t.Shr(a_hi, n_lo));
      // Mask the low/high words to `width` bits.
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
    case 0x2c:  // s_absdiff_i32: |a - b|
      r = t.m.Bitcast(t.t_u, t.m.ExtInst(t.t_i, GLSLstd450SAbs,
                                         {t.m.Bitcast(t.t_i, t.Sub(a, b))}));
      scc = true;
      break;
    case 0x2e:  // s_lshl1_add_u32 .. s_lshl4_add_u32: (a << n) + b. RDNA-only
    case 0x2f:  // numbering -- these opcodes mean something else pre-gfx10, so
    case 0x30:  // the PS4 decoder must not reach them.
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
    case 0x32:  // s_pack_ll_b32_b16: {b[15:0], a[15:0]}
      if (inst.isa != IsaMode::kNeo) {
        WarnUnsupported("sop2", op);
        r = a;
        break;
      }
      r = t.Or(t.And(a, t.U32(0xFFFF)),
               t.Shl(t.And(b, t.U32(0xFFFF)), t.U32(16)));
      break;
    case 0x33:  // s_pack_lh_b32_b16: {b[31:16], a[15:0]}
      if (inst.isa != IsaMode::kNeo) {
        WarnUnsupported("sop2", op);
        r = a;
        break;
      }
      r = t.Or(t.And(a, t.U32(0xFFFF)), t.And(b, t.U32(0xFFFF0000u)));
      break;
    case 0x34:  // s_pack_hh_b32_b16: {b[31:16], a[31:16]}
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

// ---- SOPC (s_cmp_* -> SCC) --------------------------------------------------
void EmitSopc(Translator& t, const Inst& inst) {
  const uint32_t w = inst.raw[0];
  const uint32_t op = inst.opcode, s0f = w & 0xFF, s1f = (w >> 8) & 0xFF;
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
    case 0x0c: {  // s_bitcmp0_b32: SCC = (a[b[4:0]] == 0)
      c = t.IsZero(t.And(t.Shr(a, b), t.U32(1)));
      break;
    }
    case 0x0d: {  // s_bitcmp1_b32
      c = t.IsNonZero(t.And(t.Shr(a, b), t.U32(1)));
      break;
    }
    default:
      WarnUnsupported("sopc", op);
      break;
  }
  if (c)
    t.SetSccBool(c);
}

// ---- SOPK -------------------------------------------------------------------
void EmitSopk(Translator& t, const Inst& inst) {
  const uint32_t w = inst.raw[0];
  const uint32_t op = inst.opcode, sdst = (w >> 16) & 0x7F;
  const uint32_t simm_bits = w & 0xFFFF;
  const uint32_t sext = static_cast<uint32_t>(
      static_cast<int32_t>(static_cast<int16_t>(simm_bits)));
  const Id imm_i = t.U32(sext);       // sign-extended immediate
  const Id imm_u = t.U32(simm_bits);  // zero-extended immediate

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
      break;    // s_movk_i32
    case 0x02:  // s_cmovk_i32
      t.SetSdst(sdst, 0, t.SelectNz(t.Scc(), imm_i, t.Sdst(sdst)));
      break;
    case 0x03:
      cmp(spv::Op::OpIEqual, true);
      break;  // s_cmpk_eq_i32
    case 0x04:
      cmp(spv::Op::OpINotEqual, true);
      break;  // s_cmpk_lg_i32
    case 0x05:
      cmp(spv::Op::OpSGreaterThan, true);
      break;  // s_cmpk_gt_i32
    case 0x06:
      cmp(spv::Op::OpSGreaterThanEqual, true);
      break;  // s_cmpk_ge_i32
    case 0x07:
      cmp(spv::Op::OpSLessThan, true);
      break;  // s_cmpk_lt_i32
    case 0x08:
      cmp(spv::Op::OpSLessThanEqual, true);
      break;  // s_cmpk_le_i32
    case 0x09:
      cmp(spv::Op::OpIEqual, false);
      break;  // s_cmpk_eq_u32
    case 0x0a:
      cmp(spv::Op::OpINotEqual, false);
      break;  // s_cmpk_lg_u32
    case 0x0b:
      cmp(spv::Op::OpUGreaterThan, false);
      break;  // s_cmpk_gt_u32
    case 0x0c:
      cmp(spv::Op::OpUGreaterThanEqual, false);
      break;  // s_cmpk_ge_u32
    case 0x0d:
      cmp(spv::Op::OpULessThan, false);
      break;  // s_cmpk_lt_u32
    case 0x0e:
      cmp(spv::Op::OpULessThanEqual, false);
      break;      // s_cmpk_le_u32
    case 0x0f: {  // s_addk_i32: sdst += simm16, SCC = signed overflow
      const Id s0 = t.Sg(sdst);
      const Id r = t.Add(s0, imm_i);
      t.SetSdst(sdst, 0, r);
      t.SetSccBool(SignedAddOverflow(t, s0, imm_i, r));
      break;
    }
    case 0x10: {  // s_mulk_i32: SCC reports signed overflow
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
    // s_getreg/s_setreg touch hardware state registers (mode/trap) that have
    // no analogue here; reading returns 0.
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

// ---- VOP1 -------------------------------------------------------------------
void EmitVop1(Translator& t,
              uint32_t op,
              uint32_t vdst,
              Id s0,
              bool clamp,
              uint32_t omod) {
  const auto set_f = [&](Id f) {
    f = ApplyOutputModifier(t, f, omod);
    t.SetVgF(vdst, clamp ? t.FClamp01(f) : f);
  };
  // OMOD on an integer result is IGNORED by the hardware, not unsupported:
  // "Integer and non-specific instructions (such as moves) ignore output
  // modifiers" (Sea Islands ISA, Output Modifier Options). Warning here set
  // HadUnsupported() and DECLINED the whole shader over a modifier the GPU
  // discards. Same reasoning at the vop2/vop3 set_u below.
  const auto set_u = [&](Id u) { t.SetVg(vdst, u); };
  const Id u0 = t.m.Bitcast(t.t_u, s0);
  switch (op) {
    case 0x00:
      break;  // v_nop
    case 0x01:
      set_u(u0);
      break;  // v_mov_b32
    case 0x02:
      t.SetSg(vdst, u0);
      break;    // v_readfirstlane_b32 (single lane)
    case 0x05:  // v_cvt_f32_i32
      set_f(t.m.Emit(spv::Op::OpConvertSToF, t.t_f,
                     {t.m.Bitcast(t.t_i, s0)}));
      break;
    case 0x06:
      set_f(t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {u0}));
      break;  // cvt_f32_u32
    case 0x07:
      set_u(t.m.Emit(spv::Op::OpConvertFToU, t.t_u, {s0}));
      break;    // cvt_u32_f32
    case 0x08:  // cvt_i32_f32
      set_u(t.m.Bitcast(t.t_u, t.m.Emit(spv::Op::OpConvertFToS, t.t_i, {s0})));
      break;
    case 0x0c:  // cvt_rpi_i32_f32: floor(s + 0.5)
      set_u(t.m.Bitcast(
          t.t_u, t.m.Emit(spv::Op::OpConvertFToS, t.t_i,
                          {t.Ext1(GLSLstd450Floor, t.FAdd(s0, t.F32(0.5f)))})));
      break;
    case 0x0d:  // cvt_flr_i32_f32
      set_u(t.m.Bitcast(t.t_u, t.m.Emit(spv::Op::OpConvertFToS, t.t_i,
                                        {t.Ext1(GLSLstd450Floor, s0)})));
      break;
    case 0x0e: {  // v_cvt_off_f32_i4: signed low nibble in sixteenths
      const Id nibble = t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                                 {t.m.Bitcast(t.t_i, u0), t.U32(0), t.U32(4)});
      set_f(t.FMul(t.m.Emit(spv::Op::OpConvertSToF, t.t_f, {nibble}),
                   t.F32(1.0f / 16.0f)));
      break;
    }
    // f16 <-> f32. cvt_f16_f32 packs the half into the low half-word (high
    // half zero); cvt_f32_f16 reads it back.
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
    case 0x14: {  // cvt_f32_ubyte0..3
      const Id b = t.And(t.Shr(u0, t.U32((op - 0x11) * 8)), t.U32(0xFF));
      set_f(t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {b}));
      break;
    }
    case 0x20:
      set_f(t.Ext1(GLSLstd450Fract, s0));
      break;  // v_fract_f32
    case 0x21:
      set_f(t.Ext1(GLSLstd450Trunc, s0));
      break;  // v_trunc_f32
    case 0x22:
      set_f(t.Ext1(GLSLstd450Ceil, s0));
      break;  // v_ceil_f32
    case 0x23:
      set_f(t.Ext1(GLSLstd450RoundEven, s0));
      break;  // v_rndne_f32
    case 0x24:
      set_f(t.Ext1(GLSLstd450Floor, s0));
      break;  // v_floor_f32
    case 0x25:
      set_f(t.Ext1(GLSLstd450Exp2, s0));
      break;  // v_exp_f32
    case 0x26:  // v_log_clamp_f32 = ClampInfToFltMax(Log2(x))
      set_f(t.ClampInfToFltMax(t.Ext1(GLSLstd450Log2, s0)));
      break;
    case 0x27:
      set_f(t.Ext1(GLSLstd450Log2, s0));
      break;  // v_log_f32
    case 0x28:  // v_rcp_clamp_f32 = ClampInfToFltMax(Rcp(x))
      set_f(t.ClampInfToFltMax(t.FDiv(t.F32(1.0f), s0)));
      break;
    case 0x29:  // v_rcp_legacy_f32 = ConvertInfToZero(Rcp(x))
      set_f(t.ConvertInfToZero(t.FDiv(t.F32(1.0f), s0)));
      break;
    case 0x2a:
    case 0x2b:
      set_f(t.FDiv(t.F32(1.0f), s0));
      break;  // v_rcp[_iflag]_f32
    case 0x2c:  // v_rsq_clamp_f32 = ClampInfToFltMax(Rsqrt(x)). The old FMin
                // against FLT_MAX also turned a NaN result into +FLT_MAX, since
                // GLSL FMin returns the non-NaN operand.
      set_f(t.ClampInfToFltMax(t.Ext1(GLSLstd450InverseSqrt, s0)));
      break;
    case 0x2d:  // v_rsq_legacy_f32 = ConvertInfToZero(Rsqrt(x))
      set_f(t.ConvertInfToZero(t.Ext1(GLSLstd450InverseSqrt, s0)));
      break;
    case 0x2e:
      set_f(t.Ext1(GLSLstd450InverseSqrt, s0));
      break;  // v_rsq_f32
    case 0x33:
      set_f(t.Ext1(GLSLstd450Sqrt, s0));
      break;  // v_sqrt_f32
    // GCN trig takes the argument in revolutions (1.0 == 2*pi).
    case 0x35:
      set_f(t.Ext1(GLSLstd450Sin, t.FMul(s0, t.F32(6.28318530718f))));
      break;
    case 0x36:
      set_f(t.Ext1(GLSLstd450Cos, t.FMul(s0, t.F32(6.28318530718f))));
      break;
    case 0x37:
      set_u(t.Not(u0));
      break;  // v_not_b32
    case 0x38:
      set_u(t.BitRev(u0));
      break;      // v_bfrev_b32
    case 0x39: {  // v_ffbh_u32: count leading zeros; -1 if src == 0
      const Id msb = t.m.ExtInst(t.t_u, GLSLstd450FindUMsb, {u0});
      set_u(t.SelectB(t.IsZero(u0), t.U32(0xFFFFFFFFu), t.Sub(t.U32(31), msb)));
      break;
    }
    case 0x3a:
      set_u(t.m.ExtInst(t.t_u, GLSLstd450FindILsb, {u0}));
      break;      // v_ffbl_b32
    case 0x3b: {  // v_ffbh_i32: leading-sign-bit count; -1 if src is 0 or -1
      const Id smsb = t.m.Bitcast(t.t_u, t.m.ExtInst(t.t_i, GLSLstd450FindSMsb,
                                                     {t.m.Bitcast(t.t_i, u0)}));
      set_u(t.SelectB(t.Eq(smsb, t.U32(0xFFFFFFFFu)), t.U32(0xFFFFFFFFu),
                      t.Sub(t.U32(31), smsb)));
      break;
    }
    default:
      WarnUnsupported("vop1", op);
      set_u(u0);  // mov fallback
      break;
  }
}

// ---- VOP2 -------------------------------------------------------------------
void EmitVop2(Translator& t,
              uint32_t op,
              uint32_t vdst,
              Id s0,
              Id s1,
              uint32_t literal,
              bool clamp,
              uint32_t omod) {
  const auto set_f = [&](Id f) {
    f = ApplyOutputModifier(t, f, omod);
    t.SetVgF(vdst, clamp ? t.FClamp01(f) : f);
  };
  const auto set_u = [&](Id u) { t.SetVg(vdst, u); };  // OMOD ignored: see EmitVop1
  const Id u0 = t.m.Bitcast(t.t_u, s0), u1 = t.m.Bitcast(t.t_u, s1);
  const Id i0 = t.m.Bitcast(t.t_i, s0), i1 = t.m.Bitcast(t.t_i, s1);
  const auto mul24_hi = [&](spv::Op wide_mul, Id a, Id b) {
    const Id prod = t.m.Emit(wide_mul, t.PairType(), {a, b});
    return t.m.CompositeExtract(t.t_u, prod, 1);
  };
  switch (op) {
    case 0x00: {  // v_cndmask_b32: VCC ? s1 : s0 (VCC stored as raw 1u/0u)
      set_f(t.SelectF(t.IsNonZero(t.Sg(106)), s1, s0));
      break;
    }
    // v_readlane / v_writelane: pick/write a specific wave lane. Per-invocation
    // model (one lane): the value is just s0.
    case 0x01:
      t.SetSg(vdst, u0);
      break;
    case 0x02:
      // Lane writes ignore EXEC. In the scalarized wave model this remains a
      // selected-lane approximation, but it must not use SetVg predication.
      t.m.Store(t.VgPtr(vdst), u0);
      break;
    case 0x03:
      set_f(t.FAdd(s0, s1));
      break;  // v_add_f32
    case 0x04:
      set_f(t.FSub(s0, s1));
      break;  // v_sub_f32
    case 0x05:
      set_f(t.FSub(s1, s0));
      break;  // v_subrev_f32
    // The legacy multiply forms differ from the IEEE ones only in returning
    // zero when a multiplicand is zero and the other is inf or NaN. Nothing
    // feeds those values here, so they lower the same way.
    case 0x06:
      set_f(t.FAdd(t.FMul(s0, s1), t.VgF(vdst)));
      break;  // v_mac_legacy_f32
    case 0x07:
      set_f(t.FMul(s0, s1));
      break;  // v_mul_legacy_f32
    case 0x08:
      set_f(t.FMul(s0, s1));
      break;  // v_mul_f32
    case 0x09:
      set_u(t.Mul(t.Sext24(u0), t.Sext24(u1)));
      break;    // v_mul_i32_i24
    case 0x0a:  // v_mul_hi_i32_i24
      set_u(mul24_hi(spv::Op::OpSMulExtended, t.Sext24(u0), t.Sext24(u1)));
      break;
    case 0x0b:
      set_u(t.Mul(t.Low24(u0), t.Low24(u1)));
      break;    // v_mul_u32_u24
    case 0x0c:  // v_mul_hi_u32_u24
      set_u(mul24_hi(spv::Op::OpUMulExtended, t.Low24(u0), t.Low24(u1)));
      break;
    // v_min_legacy_f32 / v_max_legacy_f32 = min_dx9 / max_dx9. The ISA is
    // explicit: "If one or both inputs are NaN values then vsrc1 is always
    // returned", and IEEE mode has no effect. An ordered compare is exactly
    // that -- it is false whenever either operand is NaN, so the select falls
    // to vsrc1. GLSL FMin/FMax return the *non-NaN* operand instead, which is
    // the opposite answer when vsrc1 is the NaN.
    case 0x0d:
      set_f(t.SelectF(t.FLt(s0, s1), s0, s1));
      break;
    case 0x0f:
      set_f(t.Ext2(GLSLstd450FMin, s0, s1));
      break;  // v_min_f32
    case 0x0e:
      set_f(t.SelectF(t.FGt(s0, s1), s0, s1));
      break;
    case 0x10:
      set_f(t.Ext2(GLSLstd450FMax, s0, s1));
      break;  // v_max_f32
    case 0x11:
      set_u(t.SMin(u0, u1));
      break;  // v_min_i32
    case 0x12:
      set_u(t.SMax(u0, u1));
      break;  // v_max_i32
    case 0x13:
      set_u(t.UMin(u0, u1));
      break;  // v_min_u32
    case 0x14:
      set_u(t.UMax(u0, u1));
      break;  // v_max_u32
    case 0x15:
      set_u(t.Shr(u0, u1));
      break;  // v_lshr_b32
    case 0x16:
      set_u(t.Shr(u1, u0));
      break;  // v_lshrrev_b32
    case 0x17:
      set_u(t.Sar(u0, u1));
      break;  // v_ashr_i32
    case 0x18:
      set_u(t.Sar(u1, u0));
      break;  // v_ashrrev_i32
    case 0x19:
      set_u(t.Shl(u0, u1));
      break;  // v_lshl_b32
    case 0x1a:
      set_u(t.Shl(u1, u0));
      break;  // v_lshlrev_b32
    case 0x1b:
      set_u(t.And(u0, u1));
      break;  // v_and_b32
    case 0x1c:
      set_u(t.Or(u0, u1));
      break;  // v_or_b32
    case 0x1d:
      set_u(t.Xor(u0, u1));
      break;      // v_xor_b32
    case 0x1e: {  // v_bfm_b32: ((1 << s0[4:0]) - 1) << s1[4:0]
      const Id ones = t.Sub(t.Shl(t.U32(1), u0), t.U32(1));
      set_u(t.Shl(ones, u1));
      break;
    }
    case 0x1f:
      set_f(t.FAdd(t.FMul(s0, s1), t.VgF(vdst)));
      break;    // v_mac_f32
    case 0x20:  // v_madmk_f32: s0 * K + s1
      set_f(t.FAdd(t.FMul(s0, t.m.Bitcast(t.t_f, t.U32(literal))), s1));
      break;
    case 0x21:  // v_madak_f32: s0 * s1 + K
      set_f(t.FAdd(t.FMul(s0, s1), t.m.Bitcast(t.t_f, t.U32(literal))));
      break;
    case 0x22:
      set_u(t.Add(t.PopCount(u0), u1));
      break;  // v_bcnt_u32_b32
    // v_mbcnt_lo/hi: this lane's index within the wave. Single lane -> no
    // prior lanes; the running accumulator s1 passes through.
    case 0x23:
    case 0x24:
      set_u(u1);
      break;
    case 0x25: {  // v_add_i32: carry-out -> VCC
      const CarryResult r = AddCarry(t, u0, u1);
      set_u(t.Add(u0, u1));
      t.SetSg(106, r.flag);
      break;
    }
    case 0x26: {  // v_sub_i32
      const CarryResult r = SubBorrow(t, u0, u1);
      set_u(r.value);
      t.SetSg(106, r.flag);
      break;
    }
    case 0x27: {  // v_subrev_i32
      const CarryResult r = SubBorrow(t, u1, u0);
      set_u(r.value);
      t.SetSg(106, r.flag);
      break;
    }
    case 0x28: {  // v_addc_u32: s0 + s1 + VCC, carry-out -> VCC
      const CarryResult r = AddCarry(t, u0, u1, t.Sg(106));
      set_u(r.value);
      t.SetSg(106, r.flag);
      break;
    }
    case 0x29: {  // v_subb_u32
      const CarryResult r = SubBorrow(t, u0, u1, t.Sg(106));
      set_u(r.value);
      t.SetSg(106, r.flag);
      break;
    }
    case 0x2a: {  // v_subbrev_u32
      const CarryResult r = SubBorrow(t, u1, u0, t.Sg(106));
      set_u(r.value);
      t.SetSg(106, r.flag);
      break;
    }
    case 0x2b:  // v_ldexp_f32
      set_f(t.m.ExtInst(t.t_f, GLSLstd450Ldexp, {s0, i1}));
      break;
    case 0x2f:  // v_cvt_pkrtz_f16_f32
      if (clamp || omod)
        WarnUnsupported("vop2.cvt_pkrtz-output-modifier", op);
      t.SetVg(vdst, t.m.ExtInst(t.t_u, GLSLstd450PackHalf2x16,
                                {t.m.CompositeConstruct(t.t_v2, {s0, s1})}));
      break;
    default:
      WarnUnsupported("vop2", op);
      set_f(t.FMul(s0, s1));
      break;
  }
  (void)i0;
}

// ---- VOPC -------------------------------------------------------------------
namespace {

// Float predicate for the low opcode nibble (F/LT/EQ/LE/GT/LG/GE/O/U/NGE/NLG/
// NGT/NLE/NEQ/NLT/TRU). Returns 0 for none (F handled by caller).
Id FloatPredicate(Translator& t, uint32_t lo, Id a, Id b) {
  const auto F = [&](spv::Op o) { return t.m.Emit(o, t.t_bool, {a, b}); };
  const auto is_nan = [&](Id x) {
    return t.m.Emit(spv::Op::OpIsNan, t.t_bool, {x});
  };
  switch (lo) {
    case 1:
      return F(spv::Op::OpFOrdLessThan);
    case 2:
      return F(spv::Op::OpFOrdEqual);
    case 3:
      return F(spv::Op::OpFOrdLessThanEqual);
    case 4:
      return F(spv::Op::OpFOrdGreaterThan);
    case 5:
      return F(spv::Op::OpFOrdNotEqual);  // LG
    case 6:
      return F(spv::Op::OpFOrdGreaterThanEqual);
    case 7:  // O: neither operand NaN
      return t.m.Emit(
          spv::Op::OpLogicalNot, t.t_bool,
          {t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {is_nan(a), is_nan(b)})});
    case 8:  // U: either operand NaN
      return t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {is_nan(a), is_nan(b)});
    case 9:
      return F(spv::Op::OpFUnordLessThan);  // NGE
    case 10:
      return F(spv::Op::OpFUnordEqual);  // NLG
    case 11:
      return F(spv::Op::OpFUnordLessThanEqual);  // NGT
    case 12:
      return F(spv::Op::OpFUnordGreaterThan);  // NLE
    case 13:
      return F(spv::Op::OpFUnordNotEqual);  // NEQ
    case 14:
      return F(spv::Op::OpFUnordGreaterThanEqual);  // NLT
    case 15:
      return t.m.ConstBool(true);  // TRU
    default:
      return 0;  // F (always false)
  }
}

// Integer predicate for the low 3 bits (F/LT/EQ/LE/GT/NE/GE/T).
Id IntPredicate(Translator& t, uint32_t lo, bool is_signed, Id a, Id b) {
  const Id ai = t.m.Bitcast(t.t_i, a), bi = t.m.Bitcast(t.t_i, b);
  const auto S = [&](spv::Op o) { return t.m.Emit(o, t.t_bool, {ai, bi}); };
  const auto U = [&](spv::Op o) { return t.m.Emit(o, t.t_bool, {a, b}); };
  switch (lo) {
    case 1:
      return is_signed ? S(spv::Op::OpSLessThan) : U(spv::Op::OpULessThan);
    case 2:
      return t.Eq(a, b);
    case 3:
      return is_signed ? S(spv::Op::OpSLessThanEqual)
                       : U(spv::Op::OpULessThanEqual);
    case 4:
      return is_signed ? S(spv::Op::OpSGreaterThan)
                       : U(spv::Op::OpUGreaterThan);
    case 5:
      return t.m.Emit(spv::Op::OpINotEqual, t.t_bool, {a, b});
    case 6:
      return is_signed ? S(spv::Op::OpSGreaterThanEqual)
                       : U(spv::Op::OpUGreaterThanEqual);
    case 7:
      return t.m.ConstBool(true);
    default:
      return 0;
  }
}

// ---- exact 64-bit compares, assembled from dword pairs ----------------------
// These families used to be evaluated on the low dword alone, which is simply a
// different answer for any value whose halves disagree. Building them out of
// 32-bit halves (rather than OpTypeInt 64 / OpTypeFloat 64) keeps the emitted
// module free of the Int64 and Float64 capabilities, and so free of the
// shaderInt64 / shaderFloat64 device features -- which the Android targets do
// not all advertise.
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
// Lexicographic on (hi, lo); only the high half's comparison differs in sign.
Id U64Lt(Translator& t, Dword2 a, Dword2 b) {
  return LOr(t, t.Ult(a.hi, b.hi),
             t.LAnd(t.Eq(a.hi, b.hi), t.Ult(a.lo, b.lo)));
}
Id I64Lt(Translator& t, Dword2 a, Dword2 b) {
  return LOr(t, Slt32(t, a.hi, b.hi),
             t.LAnd(t.Eq(a.hi, b.hi), t.Ult(a.lo, b.lo)));
}

Id Int64Predicate(Translator& t,
                  uint32_t lo,
                  bool is_signed,
                  Dword2 a,
                  Dword2 b) {
  const auto lt = [&](Dword2 x, Dword2 y) {
    return is_signed ? I64Lt(t, x, y) : U64Lt(t, x, y);
  };
  switch (lo) {
    case 0: return t.m.ConstBool(false);
    case 1: return lt(a, b);
    case 2: return U64Eq(t, a, b);
    case 3: return LOr(t, lt(a, b), U64Eq(t, a, b));
    case 4: return lt(b, a);
    case 5: return LNot(t, U64Eq(t, a, b));
    case 6: return LNot(t, lt(a, b));
    case 7: return t.m.ConstBool(true);
    default: return 0;
  }
}

// IEEE-754 binary64 predicates over the raw halves.
Id F64IsNan(Translator& t, Dword2 x) {
  return t.LAnd(
      t.Eq(t.And(x.hi, t.U32(0x7ff00000u)), t.U32(0x7ff00000u)),
      LOr(t, t.IsNonZero(t.And(x.hi, t.U32(0x000fffffu))), t.IsNonZero(x.lo)));
}
Id F64IsZero(Translator& t, Dword2 x) {  // +0.0 and -0.0 alike
  return t.LAnd(t.IsZero(t.And(x.hi, t.U32(0x7fffffffu))), t.IsZero(x.lo));
}
// Monotonic unsigned key: ordering the key as a 64-bit unsigned reproduces the
// ordering of the doubles (negatives invert, positives gain the sign bit).
Dword2 F64OrderKey(Translator& t, Dword2 x) {
  const Id neg = t.IsNonZero(t.And(x.hi, t.U32(0x80000000u)));
  return {t.SelectB(neg, t.Not(x.lo), x.lo),
          t.SelectB(neg, t.Not(x.hi), t.Xor(x.hi, t.U32(0x80000000u)))};
}
Id F64Eq(Translator& t, Dword2 a, Dword2 b) {
  return t.LAnd(LNot(t, LOr(t, F64IsNan(t, a), F64IsNan(t, b))),
                LOr(t, U64Eq(t, a, b),
                    t.LAnd(F64IsZero(t, a), F64IsZero(t, b))));
}
// The order key would rank -0.0 below +0.0, so equal zeroes are excluded.
Id F64Lt(Translator& t, Dword2 a, Dword2 b) {
  return t.LAnd(
      LNot(t, LOr(t, F64IsNan(t, a), F64IsNan(t, b))),
      t.LAnd(U64Lt(t, F64OrderKey(t, a), F64OrderKey(t, b)),
             LNot(t, t.LAnd(F64IsZero(t, a), F64IsZero(t, b)))));
}

Id Float64Predicate(Translator& t, uint32_t lo, Dword2 a, Dword2 b) {
  const Id nan = LOr(t, F64IsNan(t, a), F64IsNan(t, b));
  switch (lo) {
    case 0: return t.m.ConstBool(false);                        // F
    case 1: return F64Lt(t, a, b);                              // LT
    case 2: return F64Eq(t, a, b);                              // EQ
    case 3: return LOr(t, F64Lt(t, a, b), F64Eq(t, a, b));      // LE
    case 4: return F64Lt(t, b, a);                              // GT
    case 5: return t.LAnd(LNot(t, nan), LNot(t, F64Eq(t, a, b)));  // LG
    case 6: return LOr(t, F64Lt(t, b, a), F64Eq(t, a, b));      // GE
    case 7: return LNot(t, nan);                                // O
    case 8: return nan;                                         // U
    case 9: return LOr(t, nan, F64Lt(t, a, b));                 // NGE
    case 10: return LOr(t, nan, F64Eq(t, a, b));                // NLG
    case 11: return LOr(t, nan, LOr(t, F64Lt(t, a, b), F64Eq(t, a, b)));  // NGT
    case 12: return LOr(t, nan, F64Lt(t, b, a));                // NLE
    case 13: return LNot(t, F64Eq(t, a, b));                    // NEQ
    case 14: return LOr(t, nan, LOr(t, F64Lt(t, b, a), F64Eq(t, a, b)));  // NLT
    case 15: return t.m.ConstBool(true);                        // TRU
    default: return 0;
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
  const auto add_class = [&](uint32_t bit, Id condition) {
    classes = t.Or(classes, t.SelectB(condition, t.U32(1u << bit), t.U32(0)));
  };
  add_class(0, t.LAnd(nan, logical_not(quiet)));  // signaling NaN
  add_class(1, t.LAnd(nan, quiet));               // quiet NaN
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
              uint32_t op,
              Id s0f,
              Id s1f,
              Id s0u,
              Id s1u,
              uint32_t dst,
              Id s0_hi,
              Id s1_hi) {
  // Opcode space: f32 0x00-0x1F, f64 0x20-0x3F, i32 0x80-0x9F, i64 0xA0-0xBF,
  // u32 0xC0-0xDF, u64 0xE0-0xFF. Bit 4 of each 32-op family selects the
  // EXEC-writing cmpx form. The 64-bit families read a register PAIR, so they
  // are evaluated on both halves (see the Dword2 predicates above). The only
  // caller that omits the high dwords is translate_neo's f16 path, which maps
  // onto the 32-bit families exclusively, so the zero default is dead there --
  // it is not a safe fallback, since {lo, 0} reads as a denormal.
  const Dword2 a{s0u, s0_hi ? s0_hi : t.U32(0)};
  const Dword2 b{s1u, s1_hi ? s1_hi : t.U32(0)};
  Id cond = 0;
  if (op == 0x88 || op == 0x98) {
    cond = FloatClassPredicate(t, s0u, s1u);
  } else if (op <= 0x3F) {
    cond = op >= 0x20 ? Float64Predicate(t, op & 0xF, a, b)
                      : FloatPredicate(t, op & 0xF, s0f, s1f);
  } else if (op <= 0x7F) {
    // 0x40-0x7F are V_CMPS[X]_*: the signalling forms of 0x00-0x3F. They differ
    // only in raising an FP exception on a quiet NaN, which nothing here
    // models, so they take the same predicate.
    cond = op >= 0x60 ? Float64Predicate(t, op & 0xF, a, b)
                      : FloatPredicate(t, op & 0xF, s0f, s1f);
  } else {
    const bool is_signed = op <= 0xBF;
    const bool is_64 = (op >= 0xA0 && op <= 0xBF) || op >= 0xE0;
    cond = is_64 ? Int64Predicate(t, op & 0x7, is_signed, a, b)
                 : IntPredicate(t, op & 0x7, is_signed, s0u, s1u);
  }
  const Id predicate =
      cond ? t.SelectB(cond, t.U32(1), t.U32(0)) : t.U32(0);  // F -> 0
  const Id result = t.And(predicate, t.Exec());
  t.SetSg(dst, result);
  if (op & 0x10)
    t.SetSg(126, result);  // cmpx: replace EXEC
}

// ---- VOP3 -------------------------------------------------------------------
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
              uint32_t omod,
              Id s1_hi) {
  // VOP3 reflects the VOPC (0x000-0x0FF), VOP2 (0x100-0x13F) and VOP1
  // (0x180-0x1FF) encodings; only 0x140-0x17F are VOP3-exclusive.
  const Id u0 = t.m.Bitcast(t.t_u, s0), u1 = t.m.Bitcast(t.t_u, s1),
           u2 = t.m.Bitcast(t.t_u, s2);
  const auto set_f = [&](Id f) {
    f = ApplyOutputModifier(t, f, omod);
    t.SetVgF(vdst, clamp ? t.FClamp01(f) : f);
  };
  const auto set_u = [&](Id u) { t.SetVg(vdst, u); };  // OMOD ignored: see EmitVop1

  if (op < 0x100) {  // VOPC in VOP3 form: predicate written to sgpr[vdst]
    // OMOD applies only to instructions with a float OUTPUT; a VOPC writes a
    // lane mask to an SGPR, so the hardware ignores the modifier. Dropping it
    // is exact, not an approximation.
    EmitVopc(t, op, s0, s1, u0, u1, vdst, s0_hi, s1_hi);
    return;
  }
  if (op == 0x100) {  // VOP3 cndmask uses explicit S2 instead of implicit VCC
    set_u(t.SelectNz(t.And(u2, t.U32(1)), u1, u0));
    return;
  }
  if (op >= 0x125 && op <= 0x12a) {  // VOP3B integer add/sub + explicit SDST
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
    t.SetSdst(sdst, 0, r.flag);
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

  const auto mul_hi = [&](spv::Op wide_mul) {  // high 32 bits of the product
    return t.m.CompositeExtract(t.t_u,
                                t.m.Emit(wide_mul, t.PairType(), {u0, u1}), 1);
  };
  // Median of 3 (no GLSL medN): max(min(a,b), min(max(a,b), c)).
  const auto med3 = [&](Id (Translator::*mn)(Id, Id),
                        Id (Translator::*mx)(Id, Id)) {
    return (t.*mx)((t.*mn)(u0, u1), (t.*mn)((t.*mx)(u0, u1), u2));
  };
  switch (op) {
    case 0x140:  // v_mad_legacy_f32: legacy zero handling, see EmitVop2 0x06
      set_f(t.FAdd(t.FMul(s0, s1), s2));
      break;
    case 0x141:
      set_f(t.FAdd(t.FMul(s0, s1), s2));
      break;
    case 0x14b:  // v_fma_f32
      set_f(t.m.ExtInst(t.t_f, GLSLstd450Fma, {s0, s1, s2}));
      break;
    case 0x142:  // v_mad_i32_i24
      set_u(t.Add(t.Mul(t.Sext24(u0), t.Sext24(u1)), u2));
      break;
    case 0x143:  // v_mad_u32_u24
      set_u(t.Add(t.Mul(t.Low24(u0), t.Low24(u1)), u2));
      break;
    case 0x148:  // v_bfe_u32
    {
      const Id off = t.And(u1, t.U32(31));
      const Id width = t.UMin(t.And(u2, t.U32(31)), t.Sub(t.U32(32), off));
      set_u(t.m.Emit(spv::Op::OpBitFieldUExtract, t.t_u, {u0, off, width}));
      break;
    }
    case 0x149:  // v_bfe_i32
    {
      const Id off = t.And(u1, t.U32(31));
      const Id width = t.UMin(t.And(u2, t.U32(31)), t.Sub(t.U32(32), off));
      set_u(t.m.Bitcast(t.t_u, t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                                        {t.m.Bitcast(t.t_i, u0), off, width})));
      break;
    }
    case 0x14a:  // v_bfi_b32: (s0 & s1) | (~s0 & s2)
      set_u(t.Or(t.And(u0, u1), t.And(t.Not(u0), u2)));
      break;
    case 0x144:
    case 0x145:
    case 0x146:
    case 0x147: {  // v_cube{id,sc,tc,ma}_f32
      // GFX7 cube-coordinate preparation, with the ISA's Z > Y > X tie
      // priority.
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
      if (op == 0x144) {  // face: +X,-X,+Y,-Y,+Z,-Z => 0..5
        const Id x_face = t.SelectF(x_neg, t.F32(1.0f), t.F32(0.0f));
        const Id y_face = t.SelectF(y_neg, t.F32(3.0f), t.F32(2.0f));
        const Id z_face = t.SelectF(z_neg, t.F32(5.0f), t.F32(4.0f));
        result = t.SelectF(z_major, z_face, t.SelectF(y_major, y_face, x_face));
      } else if (op == 0x145) {  // horizontal face coordinate
        const Id x_sc = t.SelectF(x_neg, s2, t.FNeg(s2));
        const Id z_sc = t.SelectF(z_neg, t.FNeg(s0), s0);
        result = t.SelectF(z_major, z_sc, t.SelectF(y_major, s0, x_sc));
      } else if (op == 0x146) {  // vertical face coordinate
        const Id y_tc = t.SelectF(y_neg, t.FNeg(s2), s2);
        result = t.SelectF(y_major, y_tc, t.FNeg(s1));
      } else {  // signed twice-major-axis value used for normalization
        const Id major = t.SelectF(z_major, s2, t.SelectF(y_major, s1, s0));
        result = t.FMul(t.F32(2.0f), major);
      }
      set_f(result);
      break;
    }
    case 0x14d: {  // v_lerp_u8: per-byte (a + b + (c & 1)) >> 1
      Id r = t.U32(0);
      for (int b = 0; b < 4; b++) {
        const Id sh = t.U32(static_cast<uint32_t>(b) * 8), mask = t.U32(0xFF);
        const Id a = t.And(t.Shr(u0, sh), mask),
                 bb = t.And(t.Shr(u1, sh), mask);
        const Id cc = t.And(t.Shr(u2, sh), t.U32(1));
        const Id avg = t.Shr(t.Add(t.Add(a, bb), cc), t.U32(1));
        r = t.Or(r, t.Shl(avg, sh));
      }
      set_u(r);
      break;
    }
    case 0x14e: {  // v_alignbit_b32: ({S0,S1} >> S2[4:0])[31:0] (S0 hi, S1 lo)
      const Id sh = t.And(u2, t.U32(31));
      const Id lo = t.Shr(u1, sh);
      const Id hi =
          t.SelectB(t.IsZero(sh), t.U32(0), t.Shl(u0, t.Sub(t.U32(32), sh)));
      set_u(t.Or(lo, hi));
      break;
    }
    case 0x14f: {  // v_alignbyte_b32: byte-granular funnel shift
      const Id sh = t.Mul(t.And(u2, t.U32(3)), t.U32(8));
      const Id lo = t.Shr(u1, sh);
      const Id hi =
          t.SelectB(t.IsZero(sh), t.U32(0), t.Shl(u0, t.Sub(t.U32(32), sh)));
      set_u(t.Or(lo, hi));
      break;
    }
    case 0x151:
      set_f(t.Ext2(GLSLstd450FMin, t.Ext2(GLSLstd450FMin, s0, s1), s2));
      break;  // v_min3_f32
    case 0x152:
      set_u(t.SMin(t.SMin(u0, u1), u2));
      break;  // v_min3_i32
    case 0x153:
      set_u(t.UMin(t.UMin(u0, u1), u2));
      break;  // v_min3_u32
    case 0x154:
      set_f(t.Ext2(GLSLstd450FMax, t.Ext2(GLSLstd450FMax, s0, s1), s2));
      break;  // v_max3_f32
    case 0x155:
      set_u(t.SMax(t.SMax(u0, u1), u2));
      break;  // v_max3_i32
    case 0x156:
      set_u(t.UMax(t.UMax(u0, u1), u2));
      break;       // v_max3_u32
    case 0x157: {  // v_med3_f32 = clamp(s2, min(s0,s1), max(s0,s1))
      const Id lo = t.Ext2(GLSLstd450FMin, s0, s1);
      const Id hi = t.Ext2(GLSLstd450FMax, s0, s1);
      set_f(t.m.ExtInst(t.t_f, GLSLstd450FClamp, {s2, lo, hi}));
      break;
    }
    case 0x158:
      set_u(med3(&Translator::SMin, &Translator::SMax));
      break;  // v_med3_i32
    case 0x159:
      set_u(med3(&Translator::UMin, &Translator::UMax));
      break;     // v_med3_u32
    case 0x15d:  // v_sad_u32: |s0 - s1| + s2
      set_u(t.Add(t.Sub(t.UMax(u0, u1), t.UMin(u0, u1)), u2));
      break;
    case 0x15e: {  // v_cvt_pk_u8_f32: insert cvt_u8(S0) into byte S1 of S2
      const Id sat = t.m.ExtInst(t.t_f, GLSLstd450FClamp,
                                 {s0, t.F32(0.0f), t.F32(255.0f)});
      const Id byte =
          t.And(t.m.Emit(spv::Op::OpConvertFToU, t.t_u, {sat}), t.U32(0xFF));
      const Id shift = t.Mul(t.And(u1, t.U32(3)), t.U32(8));
      const Id mask = t.Shl(t.U32(0xFF), shift);
      set_u(t.Or(t.And(u2, t.Not(mask)), t.And(t.Shl(byte, shift), mask)));
      break;
    }
    // IEEE divide sequence (div_scale -> rcp -> div_fmas -> div_fixup),
    // shortened to an exact divide at the fixup (S2/S1): div_scale is an
    // identity passthrough and div_fmas an FMA feeding the estimate the fixup
    // ignores.
    case 0x15f:
      WarnUnsupported("vop3.div-fixup", op);
      set_f(t.FDiv(s2, s1));
      break;     // v_div_fixup_f32
    case 0x16d:  // v_div_scale_f32: identity
      WarnUnsupported("vop3.div-scale", op);
      set_f(s0);
      t.SetSdst(sdst, 0, t.U32(0));
      break;
    case 0x16f:
      set_f(t.m.ExtInst(t.t_f, GLSLstd450Fma, {s0, s1, s2}));
      break;  // v_div_fmas_f32
    case 0x169:
    case 0x16b:
      set_u(t.Mul(u0, u1));
      break;  // v_mul_lo_u32/i32
    case 0x16a:
      set_u(mul_hi(spv::Op::OpUMulExtended));
      break;  // v_mul_hi_u32
    case 0x16c:
      set_u(mul_hi(spv::Op::OpSMulExtended));
      break;  // v_mul_hi_i32
    case 0x161:
    case 0x162:
    case 0x163: {  // v_lshl/lshr/ashr_b64 (low result)
      // 64-bit shifts on a VGPR pair; reuse the scalar shift64 shape.
      const Id n = t.And(u1, t.U32(63)), n_lo = t.And(n, t.U32(31));
      const Id ge32 = t.Uge(n, t.U32(32));
      const Id zero = t.IsZero(n);
      const Id inv = t.And(t.Sub(t.U32(32), n_lo), t.U32(31));
      const Id lo_in = u0, hi_in = s0_hi;
      Id lo, hi;
      if (op == 0x161) {  // lshl
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
    case 0x177: {  // v_mad_u64_u32 / v_mad_i64_i32
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
    default:
      WarnUnsupported("vop3", op);
      set_f(s0);
      break;
  }
}

}  // namespace gpu::gcn

#endif  // DELTA_HAVE_SPIRV_BACKEND
