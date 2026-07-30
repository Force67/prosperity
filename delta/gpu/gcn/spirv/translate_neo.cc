/*
 * PS4Delta : PS4 Pro/Neo shader instructions from the Sony Neo ISA delta.
 */

#include "gpu/gcn/spirv/translator.h"

#ifdef DELTA_HAVE_SPIRV_BACKEND

namespace gpu::gcn {
namespace {

Id InlineHalf(Translator& t, uint32_t field) {
  switch (field) {
    case 240:
      return t.U32(0x3800);  // 0.5
    case 241:
      return t.U32(0xb800);  // -0.5
    case 242:
      return t.U32(0x3c00);  // 1.0
    case 243:
      return t.U32(0xbc00);  // -1.0
    case 244:
      return t.U32(0x4000);  // 2.0
    case 245:
      return t.U32(0xc000);  // -2.0
    case 246:
      return t.U32(0x4400);  // 4.0
    case 247:
      return t.U32(0xc400);  // -4.0
    case 248:
      return t.U32(0x3118);  // 1 / (2*pi)
    default:
      return 0;
  }
}

bool IsInlineFloat(uint32_t field) {
  return field >= 240 && field <= 248;
}

Id Source16(Translator& t,
            uint32_t field,
            uint32_t literal,
            bool high,
            bool sign_extend = false) {
  if (IsInlineFloat(field))
    return InlineHalf(t, field);
  Id value = t.SrcRaw(field, literal);
  if (high)
    value = t.Shr(value, t.U32(16));
  value = t.And(value, t.U32(0xffff));
  if (!sign_extend)
    return value;
  return t.m.Bitcast(
      t.t_u, t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                      {t.m.Bitcast(t.t_i, value), t.U32(0), t.U32(16)}));
}

Id SignExtend16(Translator& t, Id value) {
  return t.m.Bitcast(
      t.t_u, t.m.Emit(spv::Op::OpBitFieldSExtract, t.t_i,
                      {t.m.Bitcast(t.t_i, value), t.U32(0), t.U32(16)}));
}

Id HalfFloat(Translator& t, Id bits) {
  const Id unpacked = t.m.ExtInst(t.t_v2, GLSLstd450UnpackHalf2x16, {bits});
  return t.m.CompositeExtract(t.t_f, unpacked, 0);
}

Id FloatToHalf(Translator& t, Id value) {
  const Id pair = t.m.CompositeConstruct(t.t_v2, {value, t.F32(0.0f)});
  return t.And(t.m.ExtInst(t.t_u, GLSLstd450PackHalf2x16, {pair}),
               t.U32(0xffff));
}

void Write16(Translator& t, uint32_t vdst, Id value, bool high) {
  const Id old = t.Vg(vdst);
  const Id result =
      high ? t.Or(t.And(old, t.U32(0x0000ffff)),
                  t.Shl(t.And(value, t.U32(0xffff)), t.U32(16)))
           : t.Or(t.And(old, t.U32(0xffff0000)), t.And(value, t.U32(0xffff)));
  t.SetVg(vdst, result);
}

Id ApplyFloatMods(Translator& t, Id value, bool neg, bool abs) {
  if (abs)
    value = t.Ext1(GLSLstd450FAbs, value);
  return neg ? t.FNeg(value) : value;
}

Id ApplyOmod(Translator& t, Id value, uint32_t omod) {
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

Id MadF32(Translator& t, Id a, Id b, Id c) {
  const Id product = t.FMul(a, b);
  t.m.Decorate(product, spv::Decoration::NoContraction);
  const Id result = t.FAdd(product, c);
  t.m.Decorate(result, spv::Decoration::NoContraction);
  return result;
}

Id MadF16(Translator& t, Id a, Id b, Id c) {
  const Id product = HalfFloat(t, FloatToHalf(t, t.FMul(a, b)));
  return t.FAdd(product, c);
}

Id ClampLdexpF16Overflow(Translator& t, Id input, Id result) {
  const Id input_special =
      t.m.Emit(spv::Op::OpLogicalOr, t.t_bool,
               {t.m.Emit(spv::Op::OpIsInf, t.t_bool, {input}),
                t.m.Emit(spv::Op::OpIsNan, t.t_bool, {input})});
  const Id magnitude = t.Ext1(GLSLstd450FAbs, result);
  const Id overflow = t.m.Emit(spv::Op::OpFOrdGreaterThan, t.t_bool,
                               {magnitude, t.F32(65504.0f)});
  const Id negative =
      t.m.Emit(spv::Op::OpFOrdLessThan, t.t_bool, {result, t.F32(0.0f)});
  const Id max_finite = t.SelectF(negative, t.F32(-65504.0f), t.F32(65504.0f));
  return t.SelectF(input_special, result,
                   t.SelectF(overflow, max_finite, result));
}

Id Clamp16(Translator& t, Id value, bool is_signed, bool clamp) {
  if (clamp) {
    value = is_signed ? t.SMin(t.SMax(value, t.U32(0xffff8000)), t.U32(0x7fff))
                      : t.UMin(value, t.U32(0xffff));
  }
  return t.And(value, t.U32(0xffff));
}

Id SaturatingAddU32(Translator& t, Id a, Id b) {
  const Id pair = t.m.Emit(spv::Op::OpIAddCarry, t.PairType(), {a, b});
  const Id value = t.m.CompositeExtract(t.t_u, pair, 0);
  const Id carry = t.m.CompositeExtract(t.t_u, pair, 1);
  return t.SelectB(t.IsNonZero(carry), t.U32(0xffffffff), value);
}

Id SaturatingAddI32(Translator& t, Id a, Id b) {
  const Id value = t.Add(a, b);
  const Id overflow = t.IsNonZero(
      t.And(t.And(t.Xor(a, value), t.Xor(b, value)), t.U32(0x80000000)));
  const Id limit = t.SelectB(t.IsNonZero(t.And(a, t.U32(0x80000000))),
                             t.U32(0x80000000), t.U32(0x7fffffff));
  return t.SelectB(overflow, limit, value);
}

Id Shift16Amount(Translator& t, Id value) {
  return t.And(value, t.U32(15));
}

Id SafeFloat(Translator& t, Id value) {
  const Id is_nan = t.m.Emit(spv::Op::OpIsNan, t.t_bool, {value});
  return t.SelectF(is_nan, t.F32(0.0f), value);
}

Id ClampFloat(Translator& t, Id value, float lo, float hi) {
  return t.m.ExtInst(t.t_f, GLSLstd450FClamp,
                     {SafeFloat(t, value), t.F32(lo), t.F32(hi)});
}

Id Normalized16(Translator& t, Id value, bool is_signed) {
  const float scale = is_signed ? 32767.0f : 65535.0f;
  const Id clamped = ClampFloat(t, value, is_signed ? -1.0f : 0.0f, 1.0f);
  const Id rounded = t.Ext1(GLSLstd450RoundEven, t.FMul(clamped, t.F32(scale)));
  return is_signed ? t.m.Bitcast(t.t_u, t.m.Emit(spv::Op::OpConvertFToS, t.t_i,
                                                 {rounded}))
                   : t.m.Emit(spv::Op::OpConvertFToU, t.t_u, {rounded});
}

Id FrexpExponent16(Translator& t, Id bits) {
  const Id exponent = t.And(t.Shr(bits, t.U32(10)), t.U32(0x1f));
  const Id fraction = t.And(bits, t.U32(0x3ff));
  const Id msb = t.m.ExtInst(t.t_u, GLSLstd450FindUMsb, {fraction});
  const Id denorm = t.Add(msb, t.U32(static_cast<uint32_t>(-23)));
  const Id normal = t.Add(exponent, t.U32(static_cast<uint32_t>(-14)));
  const Id result = t.SelectB(t.IsZero(exponent), denorm, normal);
  const Id zero_or_special = t.m.Emit(
      spv::Op::OpLogicalOr, t.t_bool,
      {t.IsZero(t.And(bits, t.U32(0x7fff))), t.Eq(exponent, t.U32(31))});
  return t.SelectB(zero_or_special, t.U32(0), result);
}

Id FrexpMantissa16(Translator& t, Id bits) {
  const Id exponent = t.And(t.Shr(bits, t.U32(10)), t.U32(0x1f));
  const Id fraction = t.And(bits, t.U32(0x3ff));
  const Id sign = t.And(bits, t.U32(0x8000));
  const Id normal = t.Or(sign, t.Or(t.U32(0x3800), fraction));
  const Id msb = t.m.ExtInst(t.t_u, GLSLstd450FindUMsb, {fraction});
  const Id shift = t.Sub(t.U32(10), msb);
  const Id denorm_fraction = t.And(t.Shl(fraction, shift), t.U32(0x3ff));
  const Id denorm = t.Or(sign, t.Or(t.U32(0x3800), denorm_fraction));
  const Id finite = t.SelectB(t.IsZero(exponent), denorm, normal);
  const Id zero = t.IsZero(fraction);
  const Id zero_result = t.SelectB(zero, sign, finite);
  return t.SelectB(t.Eq(exponent, t.U32(31)), bits, zero_result);
}

Id F16Binary(Translator& t, uint32_t op, Id a, Id b) {
  switch (op) {
    case 0x32:
      return t.FAdd(a, b);
    case 0x33:
      return t.FSub(a, b);
    case 0x34:
      return t.FSub(b, a);
    case 0x35:
      return t.FMul(a, b);
    case 0x39:
      return t.Ext2(GLSLstd450FMax, a, b);
    case 0x3a:
      return t.Ext2(GLSLstd450FMin, a, b);
    default:
      return 0;
  }
}

Id MedianF(Translator& t, Id a, Id b, Id c) {
  const Id lo = t.Ext2(GLSLstd450FMin, a, b);
  const Id hi = t.Ext2(GLSLstd450FMax, a, b);
  return t.m.ExtInst(t.t_f, GLSLstd450FClamp, {c, lo, hi});
}

Id MedianI16(Translator& t, Id a, Id b, Id c, bool is_signed) {
  return is_signed ? t.SMax(t.SMin(a, b), t.SMin(t.SMax(a, b), c))
                   : t.UMax(t.UMin(a, b), t.UMin(t.UMax(a, b), c));
}

Id PermuteByte(Translator& t, Id src0, Id src1, Id select) {
  Id result = t.U32(0xff);
  result = t.SelectB(t.Eq(select, t.U32(12)), t.U32(0), result);
  for (uint32_t index = 8; index <= 11; index++) {
    const uint32_t byte_index = (index - 8) * 2 + 1;
    const Id source = byte_index < 4 ? src1 : src0;
    const Id byte =
        t.And(t.Shr(source, t.U32((byte_index & 3) * 8)), t.U32(0xff));
    const Id sign =
        t.SelectB(t.IsNonZero(t.And(byte, t.U32(0x80))), t.U32(0xff), t.U32(0));
    result = t.SelectB(t.Eq(select, t.U32(index)), sign, result);
  }
  for (uint32_t index = 0; index < 8; index++) {
    const Id source = index < 4 ? src1 : src0;
    const Id byte = t.And(t.Shr(source, t.U32((index & 3) * 8)), t.U32(0xff));
    result = t.SelectB(t.Eq(select, t.U32(index)), byte, result);
  }
  return result;
}

Id FloatClass16(Translator& t, Id bits, Id mask) {
  const Id exponent = t.And(t.Shr(bits, t.U32(10)), t.U32(0x1f));
  const Id fraction = t.And(bits, t.U32(0x3ff));
  const Id sign = t.IsNonZero(t.And(bits, t.U32(0x8000)));
  const Id not_sign = t.m.Emit(spv::Op::OpLogicalNot, t.t_bool, {sign});
  const Id exp_zero = t.IsZero(exponent);
  const Id exp_max = t.Eq(exponent, t.U32(31));
  const Id frac_zero = t.IsZero(fraction);
  const Id frac_nonzero = t.IsNonZero(fraction);
  const Id is_nan = t.LAnd(exp_max, frac_nonzero);
  const Id is_inf = t.LAnd(exp_max, frac_zero);
  const Id is_zero = t.LAnd(exp_zero, frac_zero);
  const Id is_subnormal = t.LAnd(exp_zero, frac_nonzero);
  const Id finite_exp =
      t.m.Emit(spv::Op::OpLogicalNot, t.t_bool,
               {t.m.Emit(spv::Op::OpLogicalOr, t.t_bool, {exp_zero, exp_max})});
  const Id quiet = t.IsNonZero(t.And(fraction, t.U32(0x200)));
  const Id signaling = t.m.Emit(spv::Op::OpLogicalNot, t.t_bool, {quiet});
  const Id classes[] = {
      t.LAnd(is_nan, signaling),    t.LAnd(is_nan, quiet),
      t.LAnd(is_inf, sign),         t.LAnd(finite_exp, sign),
      t.LAnd(is_subnormal, sign),   t.LAnd(is_zero, sign),
      t.LAnd(is_zero, not_sign),    t.LAnd(is_subnormal, not_sign),
      t.LAnd(finite_exp, not_sign), t.LAnd(is_inf, not_sign),
  };
  Id result = t.m.ConstBool(false);
  for (uint32_t i = 0; i < 10; i++) {
    const Id enabled = t.IsNonZero(t.And(mask, t.U32(1u << i)));
    result = t.m.Emit(spv::Op::OpLogicalOr, t.t_bool,
                      {result, t.LAnd(enabled, classes[i])});
  }
  return result;
}

bool EmitNeoVop1At(Translator& t,
                   uint32_t op,
                   uint32_t vdst,
                   uint32_t field,
                   uint32_t literal,
                   bool src_high,
                   bool dst_high,
                   bool neg,
                   bool abs,
                   bool clamp,
                   uint32_t omod,
                   bool compact) {
  if (op < 0x50 || op > 0x65)
    return false;
  if (op == 0x65) {
    if (field < 256 || src_high || dst_high || neg || abs || clamp || omod) {
      WarnUnsupported("v_swap_b32.modifier", op);
      return true;
    }
    const uint32_t source = field - 256;
    const Id old_dst = t.Vg(vdst), old_source = t.Vg(source);
    t.SetVg(vdst, old_source);
    t.SetVg(source, old_dst);
    return true;
  }
  if (op == 0x62) {
    if (!compact) {
      WarnUnsupported("v_sat_pk_u8_i16.vop3", op);
      return true;
    }
    const Id packed = t.SrcRaw(field, literal);
    const Id lo = SignExtend16(t, t.And(packed, t.U32(0xffff)));
    const Id hi = SignExtend16(t, t.Shr(packed, t.U32(16)));
    const Id lo_sat = t.UMin(t.SMax(lo, t.U32(0)), t.U32(0xff));
    const Id hi_sat = t.UMin(t.SMax(hi, t.U32(0)), t.U32(0xff));
    t.SetVg(vdst, t.Or(lo_sat, t.Shl(hi_sat, t.U32(8))));
    return true;
  }
  if ((op == 0x50 || op == 0x51) && (neg || abs)) {
    WarnUnsupported("vop1.i16-modifier", op);
    return true;
  }

  const Id bits = Source16(t, field, literal, src_high);
  Id value = HalfFloat(t, bits);
  if (op != 0x50 && op != 0x51)
    value = ApplyFloatMods(t, value, neg, abs);
  const Id operand_bits = (neg || abs) ? FloatToHalf(t, value) : bits;
  Id result = 0;
  bool float_result = false;
  switch (op) {
    case 0x50:
      result = FloatToHalf(t, t.m.Emit(spv::Op::OpConvertUToF, t.t_f, {bits}));
      float_result = true;
      break;
    case 0x51: {
      const Id signed_bits = SignExtend16(t, bits);
      result = FloatToHalf(t, t.m.Emit(spv::Op::OpConvertSToF, t.t_f,
                                       {t.m.Bitcast(t.t_i, signed_bits)}));
      float_result = true;
      break;
    }
    case 0x52:
      result = t.m.Emit(spv::Op::OpConvertFToU, t.t_u,
                        {ClampFloat(t, value, 0.0f, 65535.0f)});
      break;
    case 0x53:
      result = t.m.Bitcast(
          t.t_u, t.m.Emit(spv::Op::OpConvertFToS, t.t_i,
                          {ClampFloat(t, value, -32768.0f, 32767.0f)}));
      break;
    case 0x54:
      result = FloatToHalf(t, t.FDiv(t.F32(1.0f), value));
      float_result = true;
      break;
    case 0x55:
      result = FloatToHalf(t, t.Ext1(GLSLstd450Sqrt, value));
      float_result = true;
      break;
    case 0x56:
      result = FloatToHalf(t, t.Ext1(GLSLstd450InverseSqrt, value));
      float_result = true;
      break;
    case 0x57:
      result = FloatToHalf(t, t.Ext1(GLSLstd450Log2, value));
      float_result = true;
      break;
    case 0x58:
      result = FloatToHalf(t, t.Ext1(GLSLstd450Exp2, value));
      float_result = true;
      break;
    case 0x59:
      result = FrexpMantissa16(t, operand_bits);
      float_result = true;
      break;
    case 0x5a:
      result = FrexpExponent16(t, operand_bits);
      break;
    case 0x5b:
      result = FloatToHalf(t, t.Ext1(GLSLstd450Floor, value));
      float_result = true;
      break;
    case 0x5c:
      result = FloatToHalf(t, t.Ext1(GLSLstd450Ceil, value));
      float_result = true;
      break;
    case 0x5d:
      result = FloatToHalf(t, t.Ext1(GLSLstd450Trunc, value));
      float_result = true;
      break;
    case 0x5e:
      result = FloatToHalf(t, t.Ext1(GLSLstd450RoundEven, value));
      float_result = true;
      break;
    case 0x5f:
      result = FloatToHalf(t, t.Ext1(GLSLstd450Fract, value));
      float_result = true;
      break;
    case 0x60:
      result = FloatToHalf(
          t, t.Ext1(GLSLstd450Sin, t.FMul(value, t.F32(6.28318530718f))));
      float_result = true;
      break;
    case 0x61:
      result = FloatToHalf(
          t, t.Ext1(GLSLstd450Cos, t.FMul(value, t.F32(6.28318530718f))));
      float_result = true;
      break;
    case 0x63: {
      result = Normalized16(t, value, true);
      break;
    }
    case 0x64: {
      result = Normalized16(t, value, false);
      break;
    }
    default:
      WarnUnsupported("vop1.neo", op);
      return true;
  }
  if (omod && !float_result) {
    WarnUnsupported("vop1.integer-omod", op);
    return true;
  }
  if (float_result && (omod || clamp)) {
    Id output = ApplyOmod(t, HalfFloat(t, result), omod);
    if (clamp)
      output = t.FClamp01(output);
    result = FloatToHalf(t, output);
  }
  Write16(t, vdst, result, dst_high);
  return true;
}

}  // namespace

bool EmitNeoVop1(Translator& t, const Inst& inst) {
  const uint32_t w = inst.raw[0], op = inst.opcode;
  const uint32_t vdst = (w >> 17) & 0xff, field = w & 0x1ff;
  if (op == 0x0a) {
    Write16(t, vdst,
            FloatToHalf(t, t.m.Bitcast(t.t_f, t.SrcRaw(field, inst.literal))),
            false);
    return true;
  }
  if (op == 0x0b) {
    t.SetVgF(vdst, HalfFloat(t, Source16(t, field, inst.literal, false)));
    return true;
  }
  return EmitNeoVop1At(t, op, vdst, field, inst.literal, false, false, false,
                       false, false, 0, true);
}

bool EmitNeoVop2(Translator& t, const Inst& inst) {
  const uint32_t w = inst.raw[0], op = inst.opcode;
  if (op < 0x32 || op > 0x3b)
    return false;
  const uint32_t vdst = (w >> 17) & 0xff;
  const Id a = HalfFloat(t, Source16(t, w & 0x1ff, inst.literal, false));
  const Id b =
      HalfFloat(t, Source16(t, 256 + ((w >> 9) & 0xff), inst.literal, false));
  Id result;
  if (op == 0x36) {
    result = MadF16(t, a, b, HalfFloat(t, t.Vg(vdst)));
  } else if (op == 0x37 || op == 0x38) {
    const Id constant = HalfFloat(t, t.U32(inst.literal & 0xffff));
    result = op == 0x37 ? MadF16(t, a, constant, b) : MadF16(t, a, b, constant);
  } else if (op == 0x3b) {
    const Id exponent =
        Source16(t, 256 + ((w >> 9) & 0xff), inst.literal, false, true);
    result = ClampLdexpF16Overflow(
        t, a,
        t.m.ExtInst(t.t_f, GLSLstd450Ldexp, {a, t.m.Bitcast(t.t_i, exponent)}));
  } else {
    result = F16Binary(t, op, a, b);
  }
  Write16(t, vdst, FloatToHalf(t, result), false);
  return true;
}

bool EmitNeoVopc(Translator& t,
                 uint32_t op,
                 uint32_t dst,
                 uint32_t src0,
                 uint32_t src1,
                 uint32_t literal,
                 bool src0_high,
                 bool src1_high,
                 bool src0_neg,
                 bool src1_neg,
                 bool src0_abs,
                 bool src1_abs) {
  uint32_t mapped = 0;
  bool is_float = false, is_signed = false;
  if ((op >= 0x89 && op <= 0x8e) || (op >= 0x99 && op <= 0x9e)) {
    mapped = (op >= 0x90 ? 0x90 : 0x80) | (op & 7);
    is_signed = true;
  } else if ((op >= 0xa9 && op <= 0xae) || (op >= 0xb9 && op <= 0xbe)) {
    mapped = (op >= 0xb0 ? 0xd0 : 0xc0) | (op & 7);
  } else if ((op >= 0xc8 && op <= 0xcf) || (op >= 0xd8 && op <= 0xdf) ||
             (op >= 0xe8 && op <= 0xef) || (op >= 0xf8 && op <= 0xff)) {
    const uint32_t condition = (op & 7) | (op >= 0xe0 ? 8 : 0);
    mapped = condition | ((op & 0x10) ? 0x10 : 0);
    is_float = true;
  } else if (op == 0x8f || op == 0x9f) {
    if (src0_neg || src1_neg || src0_abs || src1_abs) {
      WarnUnsupported("vopc.f16-class.modifier", op);
      return true;
    }
    const Id value = Source16(t, src0, literal, src0_high);
    const Id mask = Source16(t, src1, literal, src1_high);
    const Id condition = FloatClass16(t, value, mask);
    const Id result = t.And(t.SelectB(condition, t.U32(1), t.U32(0)), t.Exec());
    t.SetSg(dst, result);
    if (op == 0x9f)
      t.SetSg(126, result);
    return true;
  } else {
    return false;
  }

  const Id a_bits = Source16(t, src0, literal, src0_high, is_signed);
  const Id b_bits = Source16(t, src1, literal, src1_high, is_signed);
  if (!is_float && (src0_neg || src1_neg || src0_abs || src1_abs)) {
    WarnUnsupported("vopc.i16-modifier", op);
    return true;
  }
  const Id a_float =
      is_float ? ApplyFloatMods(t, HalfFloat(t, a_bits), src0_neg, src0_abs)
               : t.F32(0.0f);
  const Id b_float =
      is_float ? ApplyFloatMods(t, HalfFloat(t, b_bits), src1_neg, src1_abs)
               : t.F32(0.0f);
  EmitVopc(t, mapped, a_float, b_float, a_bits, b_bits, dst);
  return true;
}

bool EmitNeoVop3(Translator& t, const Inst& inst) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1], op = inst.opcode;
  const uint32_t vdst = w & 0xff, op_sel = (w >> 12) & 0xf;
  const uint32_t fields[3] = {w1 & 0x1ff, (w1 >> 9) & 0x1ff,
                              (w1 >> 18) & 0x1ff};
  const uint32_t neg = (w1 >> 29) & 7, abs = (w >> 8) & 7;
  const uint32_t omod = (w1 >> 27) & 3;
  if (op == 0x18a || op == 0x18b) {
    if (op_sel) {
      WarnUnsupported("vop3.cvt_f16.op_sel", op, w, w1);
      return true;
    }
    if (op == 0x18a) {
      Id value = ApplyFloatMods(
          t, t.m.Bitcast(t.t_f, t.SrcRaw(fields[0], inst.literal)), neg & 1,
          abs & 1);
      value = ApplyOmod(t, value, omod);
      if ((w >> 11) & 1)
        value = t.FClamp01(value);
      Write16(t, vdst, FloatToHalf(t, value), false);
    } else {
      Id value = ApplyFloatMods(
          t, HalfFloat(t, Source16(t, fields[0], inst.literal, false)), neg & 1,
          abs & 1);
      value = ApplyOmod(t, value, omod);
      if ((w >> 11) & 1)
        value = t.FClamp01(value);
      t.SetVgF(vdst, value);
    }
    return true;
  }
  if (op < 0x100 && omod) {
    WarnUnsupported("vopc.omod", op, w, w1);
    return true;
  }
  if (op < 0x100 &&
      EmitNeoVopc(t, op, vdst, fields[0], fields[1], inst.literal, op_sel & 1,
                  op_sel & 2, neg & 1, neg & 2, abs & 1, abs & 2))
    return true;

  const bool clamp = (w >> 11) & 1;
  const bool dst_high = op_sel & 8;
  if (op >= 0x1d0 && op <= 0x1e5)
    return EmitNeoVop1At(t, op - 0x180, vdst, fields[0], inst.literal,
                         op_sel & 1, dst_high, neg & 1, abs & 1, clamp, omod,
                         false);

  Id raw[3], signed16[3], f16[3];
  for (uint32_t i = 0; i < 3; i++) {
    raw[i] = Source16(t, fields[i], inst.literal, op_sel & (1u << i));
    signed16[i] = SignExtend16(t, raw[i]);
    f16[i] = ApplyFloatMods(t, HalfFloat(t, raw[i]), neg & (1u << i),
                            abs & (1u << i));
  }

  if ((op >= 0x132 && op <= 0x136) || (op >= 0x139 && op <= 0x13b)) {
    Id result;
    if (op == 0x136) {
      const Id addend =
          HalfFloat(t, Source16(t, 256 + vdst, inst.literal, dst_high));
      result = MadF16(t, f16[0], f16[1], addend);
    } else if (op == 0x13b)
      result = ClampLdexpF16Overflow(
          t, f16[0],
          t.m.ExtInst(t.t_f, GLSLstd450Ldexp,
                      {f16[0], t.m.Bitcast(t.t_i, signed16[1])}));
    else
      result = F16Binary(t, op - 0x100, f16[0], f16[1]);
    result = ApplyOmod(t, result, omod);
    if (clamp)
      result = t.FClamp01(result);
    Write16(t, vdst, FloatToHalf(t, result), dst_high);
    return true;
  }

  const bool has_float_inputs = (op >= 0x311 && op <= 0x313) || op == 0x341 ||
                                op == 0x34b || op == 0x351 || op == 0x354 ||
                                op == 0x357 || op == 0x35f;
  const bool has_float_output = op == 0x341 || op == 0x34b || op == 0x351 ||
                                op == 0x354 || op == 0x357 || op == 0x35f;
  if ((neg || abs) && !has_float_inputs) {
    WarnUnsupported("vop3.i16-modifier", op, w, w1);
    return true;
  }
  if (omod && !has_float_output) {
    WarnUnsupported("vop3.integer-omod", op, w, w1);
    return true;
  }
  const bool writes_32 = (op >= 0x344 && op <= 0x347) || op == 0x36d ||
                         op == 0x36f || op == 0x371 || op == 0x372 ||
                         op == 0x373 || op == 0x375;
  if (clamp && writes_32 && op != 0x373 && op != 0x375) {
    WarnUnsupported("vop3.i32-clamp", op, w, w1);
    return true;
  }

  Id result = 0;
  bool signed_result = false, writes_16 = true;
  switch (op) {
    case 0x303:
    case 0x30d:
      result =
          op == 0x30d ? t.Add(signed16[0], signed16[1]) : t.Add(raw[0], raw[1]);
      signed_result = op == 0x30d;
      break;
    case 0x304:
    case 0x30e:
      result =
          op == 0x30e ? t.Sub(signed16[0], signed16[1]) : t.Sub(raw[0], raw[1]);
      if (op == 0x304 && clamp)
        result = t.SelectB(t.Ult(raw[0], raw[1]), t.U32(0), result);
      signed_result = op == 0x30e;
      break;
    case 0x305:
      result = t.Mul(raw[0], raw[1]);
      break;
    case 0x307:
      result = t.Shr(raw[1], Shift16Amount(t, raw[0]));
      break;
    case 0x308:
      result = t.Sar(signed16[1], Shift16Amount(t, raw[0]));
      signed_result = true;
      break;
    case 0x309:
      result = t.UMax(raw[0], raw[1]);
      break;
    case 0x30a:
      result = t.SMax(signed16[0], signed16[1]);
      signed_result = true;
      break;
    case 0x30b:
      result = t.UMin(raw[0], raw[1]);
      break;
    case 0x30c:
      result = t.SMin(signed16[0], signed16[1]);
      signed_result = true;
      break;
    case 0x311:
      if (clamp) {
        WarnUnsupported("v_pack_b32_f16.clamp", op, w, w1);
        return true;
      }
      result = t.Or(FloatToHalf(t, f16[1]),
                    t.Shl(FloatToHalf(t, f16[0]), t.U32(16)));
      writes_16 = false;
      break;
    case 0x312:
    case 0x313:
      result =
          t.Or(t.And(Normalized16(t, f16[0], op == 0x312), t.U32(0xffff)),
               t.Shl(t.And(Normalized16(t, f16[1], op == 0x312), t.U32(0xffff)),
                     t.U32(16)));
      writes_16 = false;
      break;
    case 0x314:
      result = t.Shl(raw[1], Shift16Amount(t, raw[0]));
      break;
    case 0x340:
    case 0x35e:
      result = t.Add(t.Mul(op == 0x35e ? signed16[0] : raw[0],
                           op == 0x35e ? signed16[1] : raw[1]),
                     op == 0x35e ? signed16[2] : raw[2]);
      signed_result = op == 0x35e;
      break;
    case 0x341:
    case 0x34b: {
      Id value = op == 0x34b ? t.m.ExtInst(t.t_f, GLSLstd450Fma,
                                           {f16[0], f16[1], f16[2]})
                             : MadF16(t, f16[0], f16[1], f16[2]);
      value = ApplyOmod(t, value, omod);
      if (clamp)
        value = t.FClamp01(value);
      Write16(t, vdst, FloatToHalf(t, value), dst_high);
      return true;
    }
    case 0x345:
      result = t.Add(t.Xor(t.SrcRaw(fields[0], inst.literal),
                           t.SrcRaw(fields[1], inst.literal)),
                     t.SrcRaw(fields[2], inst.literal));
      writes_16 = false;
      break;
    case 0x346:
      result = t.Add(t.Shl(t.SrcRaw(fields[0], inst.literal),
                           t.SrcRaw(fields[1], inst.literal)),
                     t.SrcRaw(fields[2], inst.literal));
      writes_16 = false;
      break;
    case 0x347:
      result = t.Shl(t.Add(t.SrcRaw(fields[0], inst.literal),
                           t.SrcRaw(fields[1], inst.literal)),
                     t.SrcRaw(fields[2], inst.literal));
      writes_16 = false;
      break;
    case 0x351:
      result = t.Ext2(GLSLstd450FMin, t.Ext2(GLSLstd450FMin, f16[0], f16[1]),
                      f16[2]);
      result = ApplyOmod(t, result, omod);
      if (clamp)
        result = t.FClamp01(result);
      Write16(t, vdst, FloatToHalf(t, result), dst_high);
      return true;
    case 0x352:
    case 0x353:
      result = op == 0x352
                   ? t.SMin(t.SMin(signed16[0], signed16[1]), signed16[2])
                   : t.UMin(t.UMin(raw[0], raw[1]), raw[2]);
      signed_result = op == 0x352;
      break;
    case 0x354:
      result = t.Ext2(GLSLstd450FMax, t.Ext2(GLSLstd450FMax, f16[0], f16[1]),
                      f16[2]);
      result = ApplyOmod(t, result, omod);
      if (clamp)
        result = t.FClamp01(result);
      Write16(t, vdst, FloatToHalf(t, result), dst_high);
      return true;
    case 0x355:
    case 0x356:
      result = op == 0x355
                   ? t.SMax(t.SMax(signed16[0], signed16[1]), signed16[2])
                   : t.UMax(t.UMax(raw[0], raw[1]), raw[2]);
      signed_result = op == 0x355;
      break;
    case 0x357:
      result = MedianF(t, f16[0], f16[1], f16[2]);
      result = ApplyOmod(t, result, omod);
      if (clamp)
        result = t.FClamp01(result);
      Write16(t, vdst, FloatToHalf(t, result), dst_high);
      return true;
    case 0x358:
    case 0x359:
      result = MedianI16(t, op == 0x358 ? signed16[0] : raw[0],
                         op == 0x358 ? signed16[1] : raw[1],
                         op == 0x358 ? signed16[2] : raw[2], op == 0x358);
      signed_result = op == 0x358;
      break;
    case 0x35f:
      WarnUnsupported("v_div_fixup_f16", op, w, w1);
      return true;
    case 0x36d:
      result = t.Add(t.Add(t.SrcRaw(fields[0], inst.literal),
                           t.SrcRaw(fields[1], inst.literal)),
                     t.SrcRaw(fields[2], inst.literal));
      writes_16 = false;
      break;
    case 0x36f:
      result = t.Or(t.Shl(t.SrcRaw(fields[0], inst.literal),
                          t.SrcRaw(fields[1], inst.literal)),
                    t.SrcRaw(fields[2], inst.literal));
      writes_16 = false;
      break;
    case 0x371:
      result = t.Or(t.And(t.SrcRaw(fields[0], inst.literal),
                          t.SrcRaw(fields[1], inst.literal)),
                    t.SrcRaw(fields[2], inst.literal));
      writes_16 = false;
      break;
    case 0x372:
      result = t.Or(t.Or(t.SrcRaw(fields[0], inst.literal),
                         t.SrcRaw(fields[1], inst.literal)),
                    t.SrcRaw(fields[2], inst.literal));
      writes_16 = false;
      break;
    case 0x373:
      result = t.Mul(raw[0], raw[1]);
      result =
          clamp ? SaturatingAddU32(t, result, t.SrcRaw(fields[2], inst.literal))
                : t.Add(result, t.SrcRaw(fields[2], inst.literal));
      writes_16 = false;
      break;
    case 0x375:
      result = t.Mul(signed16[0], signed16[1]);
      result =
          clamp ? SaturatingAddI32(t, result, t.SrcRaw(fields[2], inst.literal))
                : t.Add(result, t.SrcRaw(fields[2], inst.literal));
      writes_16 = false;
      break;
    case 0x344: {
      const Id source0 = t.SrcRaw(fields[0], inst.literal);
      const Id source1 = t.SrcRaw(fields[1], inst.literal);
      const Id select = t.SrcRaw(fields[2], inst.literal);
      result = t.U32(0);
      for (uint32_t byte = 0; byte < 4; byte++) {
        const Id selector = t.And(t.Shr(select, t.U32(byte * 8)), t.U32(0xff));
        result = t.Or(result, t.Shl(PermuteByte(t, source0, source1, selector),
                                    t.U32(byte * 8)));
      }
      writes_16 = false;
      break;
    }
    default:
      return false;
  }
  if (writes_16)
    Write16(t, vdst, Clamp16(t, result, signed_result, clamp), dst_high);
  else
    t.SetVg(vdst, result);
  return true;
}

bool EmitNeoVop3p(Translator& t, const Inst& inst) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1], op = inst.opcode;
  const uint32_t fields[3] = {w1 & 0x1ff, (w1 >> 9) & 0x1ff,
                              (w1 >> 18) & 0x1ff};
  const uint32_t op_sel = (w >> 11) & 7;
  const uint32_t op_sel_hi = ((w1 >> 27) & 3) | (((w >> 14) & 1) << 2);
  const uint32_t neg_lo = (w1 >> 29) & 7, neg_hi = (w >> 8) & 7;
  const bool clamp = (w >> 15) & 1;
  if (op >= 0x20 && op <= 0x22) {
    Id source[3];
    for (uint32_t i = 0; i < 3; i++) {
      source[i] = (op_sel_hi & (1u << i))
                      ? HalfFloat(t, Source16(t, fields[i], inst.literal,
                                              op_sel & (1u << i)))
                      : t.m.Bitcast(t.t_f, t.SrcRaw(fields[i], inst.literal));
      source[i] =
          ApplyFloatMods(t, source[i], neg_lo & (1u << i), neg_hi & (1u << i));
    }
    Id result = MadF32(t, source[0], source[1], source[2]);
    if (clamp)
      result = t.FClamp01(result);
    if (op == 0x20)
      t.SetVgF(w & 0xff, result);
    else
      Write16(t, w & 0xff, FloatToHalf(t, result), op == 0x22);
    return true;
  }
  if (op > 0x12)
    return false;
  Id packed = t.U32(0);
  for (uint32_t lane = 0; lane < 2; lane++) {
    const uint32_t selections = lane ? op_sel_hi : op_sel;
    const uint32_t negations = lane ? neg_hi : neg_lo;
    Id raw[3], signed16[3], floats[3];
    for (uint32_t i = 0; i < 3; i++) {
      raw[i] = Source16(t, fields[i], inst.literal, selections & (1u << i));
      if (negations & (1u << i))
        raw[i] = t.And(t.Sub(t.U32(0), raw[i]), t.U32(0xffff));
      signed16[i] = SignExtend16(t, raw[i]);
      floats[i] = HalfFloat(t, raw[i]);
      if (negations & (1u << i)) {
        // Integer packed operations consume the two's-complement value above;
        // floating packed operations negate the original half value instead.
        const Id original =
            Source16(t, fields[i], inst.literal, selections & (1u << i));
        floats[i] = HalfFloat(t, original);
        floats[i] = t.FNeg(floats[i]);
      }
    }
    Id result;
    bool is_float = false, is_signed = false;
    switch (op) {
      case 0x00:
        result = t.Add(t.Mul(signed16[0], signed16[1]), signed16[2]);
        is_signed = true;
        break;
      case 0x01:
        result = t.Mul(raw[0], raw[1]);
        break;
      case 0x02:
        result = t.Add(signed16[0], signed16[1]);
        is_signed = true;
        break;
      case 0x03:
        result = t.Sub(signed16[0], signed16[1]);
        is_signed = true;
        break;
      case 0x04:
        result = t.Shl(raw[1], Shift16Amount(t, raw[0]));
        break;
      case 0x05:
        result = t.Shr(raw[1], Shift16Amount(t, raw[0]));
        break;
      case 0x06:
        result = t.Sar(signed16[1], Shift16Amount(t, raw[0]));
        is_signed = true;
        break;
      case 0x07:
        result = t.SMax(signed16[0], signed16[1]);
        is_signed = true;
        break;
      case 0x08:
        result = t.SMin(signed16[0], signed16[1]);
        is_signed = true;
        break;
      case 0x09:
        result = t.Add(t.Mul(raw[0], raw[1]), raw[2]);
        break;
      case 0x0a:
        result = t.Add(raw[0], raw[1]);
        break;
      case 0x0b:
        result = t.Sub(raw[0], raw[1]);
        if (clamp)
          result = t.SelectB(t.Ult(raw[0], raw[1]), t.U32(0), result);
        break;
      case 0x0c:
        result = t.UMax(raw[0], raw[1]);
        break;
      case 0x0d:
        result = t.UMin(raw[0], raw[1]);
        break;
      case 0x0e:
        result = MadF16(t, floats[0], floats[1], floats[2]);
        is_float = true;
        break;
      case 0x0f:
        result = t.FAdd(floats[0], floats[1]);
        is_float = true;
        break;
      case 0x10:
        result = t.FMul(floats[0], floats[1]);
        is_float = true;
        break;
      case 0x11:
        result = t.Ext2(GLSLstd450FMin, floats[0], floats[1]);
        is_float = true;
        break;
      case 0x12:
        result = t.Ext2(GLSLstd450FMax, floats[0], floats[1]);
        is_float = true;
        break;
      default:
        return false;
    }
    if (is_float) {
      if (clamp)
        result = t.FClamp01(result);
      result = FloatToHalf(t, result);
    } else {
      const bool saturating_mad = op == 0x00 || op == 0x09;
      result = Clamp16(t, result, is_signed, clamp || saturating_mad);
    }
    packed = t.Or(packed, lane ? t.Shl(result, t.U32(16)) : result);
  }
  t.SetVg(w & 0xff, packed);
  return true;
}

}  // namespace gpu::gcn

#endif  // DELTA_HAVE_SPIRV_BACKEND
