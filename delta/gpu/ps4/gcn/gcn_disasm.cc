/*
 * PS4Delta : PS4 emulation and research project
 *
 * GCN GFX7 disassembler. See gcn_disasm.h. Mnemonic tables follow the Sea
 * Islands ISA opcode numbering; gaps render as "<enc>_op0x<n>" so an unmapped
 * opcode is still uniquely identifiable in logs and audit reports.
 */

#include "gpu/ps4/gcn/gcn_disasm.h"

#include <cstdio>

namespace gpu::gcn {
namespace {

std::string Hex(uint32_t v) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%x", v);
  return buf;
}

// ---- register / operand naming ---------------------------------------------

// Scalar register name for a plain 7-bit SGPR-file index (SDST fields).
std::string SName(uint32_t i) {
  if (i <= 103)
    return "s" + std::to_string(i);
  switch (i) {
    case 104:
      return "flat_scratch_lo";
    case 105:
      return "flat_scratch_hi";
    case 106:
      return "vcc_lo";
    case 107:
      return "vcc_hi";
    case 124:
      return "m0";
    case 126:
      return "exec_lo";
    case 127:
      return "exec_hi";
    default:
      break;
  }
  if (i >= 112 && i <= 123)
    return "ttmp" + std::to_string(i - 112);
  return "s" + std::to_string(i);
}

// Aligned scalar range (64-bit pairs, V#/T#/S# quads and up).
std::string SRange(uint32_t base, uint32_t count) {
  if (count <= 1)
    return SName(base);
  if (count == 2) {
    if (base == 106)
      return "vcc";
    if (base == 126)
      return "exec";
    if (base == 104)
      return "flat_scratch";
  }
  return "s[" + std::to_string(base) + ":" + std::to_string(base + count - 1) +
         "]";
}

std::string VRange(uint32_t base, uint32_t count) {
  if (count <= 1)
    return "v" + std::to_string(base);
  return "v[" + std::to_string(base) + ":" + std::to_string(base + count - 1) +
         "]";
}

// Full source-operand field (SSRC/SRC encoding: SGPRs, inline constants,
// literal, VGPRs).
std::string Src(uint32_t field, const Inst& inst, uint32_t count = 1) {
  if (field <= 127)
    return SRange(field, count);
  if (field == 128)
    return "0";
  if (field >= 129 && field <= 192)
    return std::to_string(field - 128);
  if (field >= 193 && field <= 208)
    return std::to_string(-static_cast<int>(field - 192));
  switch (field) {
    case 240:
      return "0.5";
    case 241:
      return "-0.5";
    case 242:
      return "1.0";
    case 243:
      return "-1.0";
    case 244:
      return "2.0";
    case 245:
      return "-2.0";
    case 246:
      return "4.0";
    case 247:
      return "-4.0";
    case 248:
      if (inst.isa == IsaMode::kNeo)
        return "0.15915494";
      break;
    case 251:
      return "vccz";
    case 252:
      return "execz";
    case 253:
      return "scc";
    case 254:
      return "lds_direct";
    case 255:
      return Hex(inst.literal);
    default:
      break;
  }
  if (field >= 256)
    return VRange(field - 256, count);
  return "src" + std::to_string(field);
}

// A mnemonic naming a 64-bit operation operates on register pairs.
bool Is64(const std::string& name) {
  return name.size() >= 2 && name.compare(name.size() - 2, 2, "64") == 0;
}

std::string Fallback(const char* enc, uint32_t op) {
  return std::string(enc) + "_op" + Hex(op);
}

const char* Lookup(const char* const* table, uint32_t n, uint32_t op) {
  return op < n ? table[op] : nullptr;
}

// ---- opcode tables (Sea Islands numbering) ----------------------------------

const char* const kSop1[] = {
    // clang-format off
    nullptr, nullptr, nullptr,
    "s_mov_b32", "s_mov_b64", "s_cmov_b32", "s_cmov_b64",
    "s_not_b32", "s_not_b64", "s_wqm_b32", "s_wqm_b64",
    "s_brev_b32", "s_brev_b64",
    "s_bcnt0_i32_b32", "s_bcnt0_i32_b64", "s_bcnt1_i32_b32", "s_bcnt1_i32_b64",
    "s_ff0_i32_b32", "s_ff0_i32_b64", "s_ff1_i32_b32", "s_ff1_i32_b64",
    "s_flbit_i32_b32", "s_flbit_i32_b64", "s_flbit_i32", "s_flbit_i32_i64",
    "s_sext_i32_i8", "s_sext_i32_i16",
    "s_bitset0_b32", "s_bitset0_b64", "s_bitset1_b32", "s_bitset1_b64",
    "s_getpc_b64", "s_setpc_b64", "s_swappc_b64", "s_rfe_b64", nullptr,
    "s_and_saveexec_b64", "s_or_saveexec_b64", "s_xor_saveexec_b64",
    "s_andn2_saveexec_b64", "s_orn2_saveexec_b64", "s_nand_saveexec_b64",
    "s_nor_saveexec_b64", "s_xnor_saveexec_b64",
    "s_quadmask_b32", "s_quadmask_b64",
    "s_movrels_b32", "s_movrels_b64", "s_movreld_b32", "s_movreld_b64",
    "s_cbranch_join", nullptr, "s_abs_i32", "s_mov_fed_b32",
    // clang-format on
};

const char* const kSop2[] = {
    // clang-format off
    "s_add_u32", "s_sub_u32", "s_add_i32", "s_sub_i32",
    "s_addc_u32", "s_subb_u32",
    "s_min_i32", "s_min_u32", "s_max_i32", "s_max_u32",
    "s_cselect_b32", "s_cselect_b64", nullptr, nullptr,
    "s_and_b32", "s_and_b64", "s_or_b32", "s_or_b64",
    "s_xor_b32", "s_xor_b64", "s_andn2_b32", "s_andn2_b64",
    "s_orn2_b32", "s_orn2_b64", "s_nand_b32", "s_nand_b64",
    "s_nor_b32", "s_nor_b64", "s_xnor_b32", "s_xnor_b64",
    "s_lshl_b32", "s_lshl_b64", "s_lshr_b32", "s_lshr_b64",
    "s_ashr_i32", "s_ashr_i64", "s_bfm_b32", "s_bfm_b64",
    "s_mul_i32", "s_bfe_u32", "s_bfe_i32", "s_bfe_u64", "s_bfe_i64",
    "s_cbranch_g_fork", "s_absdiff_i32",
    // clang-format on
};

const char* NeoSop2Name(uint32_t op) {
  switch (op) {
    case 0x32:
      return "s_pack_ll_b32_b16";
    case 0x33:
      return "s_pack_lh_b32_b16";
    case 0x34:
      return "s_pack_hh_b32_b16";
    default:
      return nullptr;
  }
}

const char* const kSopk[] = {
    // clang-format off
    "s_movk_i32", nullptr, "s_cmovk_i32",
    "s_cmpk_eq_i32", "s_cmpk_lg_i32", "s_cmpk_gt_i32", "s_cmpk_ge_i32",
    "s_cmpk_lt_i32", "s_cmpk_le_i32",
    "s_cmpk_eq_u32", "s_cmpk_lg_u32", "s_cmpk_gt_u32", "s_cmpk_ge_u32",
    "s_cmpk_lt_u32", "s_cmpk_le_u32",
    "s_addk_i32", "s_mulk_i32", "s_cbranch_i_fork",
    "s_getreg_b32", "s_setreg_b32", nullptr, "s_setreg_imm32_b32",
    // clang-format on
};

const char* const kSopc[] = {
    // clang-format off
    "s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32",
    "s_cmp_lt_i32", "s_cmp_le_i32",
    "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32",
    "s_cmp_lt_u32", "s_cmp_le_u32",
    "s_bitcmp0_b32", "s_bitcmp1_b32", "s_bitcmp0_b64", "s_bitcmp1_b64",
    "s_setvskip",
    // clang-format on
};

const char* const kSopp[] = {
    // clang-format off
    "s_nop", "s_endpgm", "s_branch", nullptr,
    "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz",
    "s_cbranch_execz", "s_cbranch_execnz",
    "s_barrier", nullptr, "s_waitcnt", "s_sethalt", "s_sleep", "s_setprio",
    "s_sendmsg", "s_sendmsghalt", "s_trap", "s_icache_inv",
    "s_incperflevel", "s_decperflevel", "s_ttracedata",
    "s_cbranch_cdbgsys", "s_cbranch_cdbguser",
    "s_cbranch_cdbgsys_or_user", "s_cbranch_cdbgsys_and_user",
    // clang-format on
};

const char* const kSmrd[] = {
    // clang-format off
    "s_load_dword", "s_load_dwordx2", "s_load_dwordx4",
    "s_load_dwordx8", "s_load_dwordx16", nullptr, nullptr, nullptr,
    "s_buffer_load_dword", "s_buffer_load_dwordx2", "s_buffer_load_dwordx4",
    "s_buffer_load_dwordx8", "s_buffer_load_dwordx16", nullptr, nullptr,
    nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    "s_dcache_inv_vol", "s_memtime", "s_dcache_inv",
    // clang-format on
};

const char* const kVop1[] = {
    // clang-format off
    "v_nop", "v_mov_b32", "v_readfirstlane_b32",
    "v_cvt_i32_f64", "v_cvt_f64_i32", "v_cvt_f32_i32", "v_cvt_f32_u32",
    "v_cvt_u32_f32", "v_cvt_i32_f32", "v_mov_fed_b32",
    "v_cvt_f16_f32", "v_cvt_f32_f16", "v_cvt_rpi_i32_f32", "v_cvt_flr_i32_f32",
    "v_cvt_off_f32_i4", "v_cvt_f32_f64", "v_cvt_f64_f32",
    "v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1", "v_cvt_f32_ubyte2",
    "v_cvt_f32_ubyte3", "v_cvt_u32_f64", "v_cvt_f64_u32",
    "v_trunc_f64", "v_ceil_f64", "v_rndne_f64", "v_floor_f64",
    nullptr, nullptr, nullptr, nullptr, nullptr,
    "v_fract_f32", "v_trunc_f32", "v_ceil_f32", "v_rndne_f32", "v_floor_f32",
    "v_exp_f32", "v_log_clamp_f32", "v_log_f32",
    "v_rcp_clamp_f32", "v_rcp_legacy_f32", "v_rcp_f32", "v_rcp_iflag_f32",
    "v_rsq_clamp_f32", "v_rsq_legacy_f32", "v_rsq_f32",
    "v_rcp_f64", "v_rcp_clamp_f64", "v_rsq_f64", "v_rsq_clamp_f64",
    "v_sqrt_f32", "v_sqrt_f64", "v_sin_f32", "v_cos_f32",
    "v_not_b32", "v_bfrev_b32", "v_ffbh_u32", "v_ffbl_b32", "v_ffbh_i32",
    "v_frexp_exp_i32_f64", "v_frexp_mant_f64", "v_fract_f64",
    "v_frexp_exp_i32_f32", "v_frexp_mant_f32", "v_clrexcp",
    "v_movreld_b32", "v_movrels_b32", "v_movrelsd_b32",
    // clang-format on
};

const char* NeoVop1Name(uint32_t op) {
  static const char* const kNeo[] = {
      // clang-format off
      "v_cvt_f16_u16", "v_cvt_f16_i16", "v_cvt_u16_f16",
      "v_cvt_i16_f16", "v_rcp_f16", "v_sqrt_f16", "v_rsq_f16",
      "v_log_f16", "v_exp_f16", "v_frexp_mant_f16",
      "v_frexp_exp_i16_f16", "v_floor_f16", "v_ceil_f16", "v_trunc_f16",
      "v_rndne_f16", "v_fract_f16", "v_sin_f16", "v_cos_f16",
      "v_sat_pk_u8_i16", "v_cvt_norm_i16_f16", "v_cvt_norm_u16_f16",
      "v_swap_b32",
      // clang-format on
  };
  return op >= 0x50 ? Lookup(kNeo, sizeof(kNeo) / sizeof(kNeo[0]), op - 0x50)
                    : nullptr;
}

const char* const kVop2[] = {
    // clang-format off
    "v_cndmask_b32", "v_readlane_b32", "v_writelane_b32",
    "v_add_f32", "v_sub_f32", "v_subrev_f32",
    "v_mac_legacy_f32", "v_mul_legacy_f32", "v_mul_f32",
    "v_mul_i32_i24", "v_mul_hi_i32_i24", "v_mul_u32_u24", "v_mul_hi_u32_u24",
    "v_min_legacy_f32", "v_max_legacy_f32", "v_min_f32", "v_max_f32",
    "v_min_i32", "v_max_i32", "v_min_u32", "v_max_u32",
    "v_lshr_b32", "v_lshrrev_b32", "v_ashr_i32", "v_ashrrev_i32",
    "v_lshl_b32", "v_lshlrev_b32", "v_and_b32", "v_or_b32", "v_xor_b32",
    "v_bfm_b32", "v_mac_f32", "v_madmk_f32", "v_madak_f32",
    "v_bcnt_u32_b32", "v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32",
    "v_add_i32", "v_sub_i32", "v_subrev_i32",
    "v_addc_u32", "v_subb_u32", "v_subbrev_u32",
    "v_ldexp_f32", "v_cvt_pkaccum_u8_f32",
    "v_cvt_pknorm_i16_f32", "v_cvt_pknorm_u16_f32", "v_cvt_pkrtz_f16_f32",
    "v_cvt_pk_u16_u32", "v_cvt_pk_i16_i32",
    // clang-format on
};

const char* NeoVop2Name(uint32_t op) {
  switch (op) {
    case 0x32:
      return "v_add_f16";
    case 0x33:
      return "v_sub_f16";
    case 0x34:
      return "v_subrev_f16";
    case 0x35:
      return "v_mul_f16";
    case 0x36:
      return "v_mac_f16";
    case 0x37:
      return "v_madmk_f16";
    case 0x38:
      return "v_madak_f16";
    case 0x39:
      return "v_max_f16";
    case 0x3a:
      return "v_min_f16";
    case 0x3b:
      return "v_ldexp_f16";
    default:
      return nullptr;
  }
}

// VOP3-only range (0x140..0x17f).
const char* const kVop3Only[] = {
    // clang-format off
    "v_mad_legacy_f32", "v_mad_f32", "v_mad_i32_i24", "v_mad_u32_u24",
    "v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32", "v_cubema_f32",
    "v_bfe_u32", "v_bfe_i32", "v_bfi_b32", "v_fma_f32", "v_fma_f64",
    "v_lerp_u8", "v_alignbit_b32", "v_alignbyte_b32", "v_mullit_f32",
    "v_min3_f32", "v_min3_i32", "v_min3_u32",
    "v_max3_f32", "v_max3_i32", "v_max3_u32",
    "v_med3_f32", "v_med3_i32", "v_med3_u32",
    "v_sad_u8", "v_sad_hi_u8", "v_sad_u16", "v_sad_u32",
    "v_cvt_pk_u8_f32", "v_div_fixup_f32", "v_div_fixup_f64",
    "v_lshl_b64", "v_lshr_b64", "v_ashr_i64",
    "v_add_f64", "v_mul_f64", "v_min_f64", "v_max_f64", "v_ldexp_f64",
    "v_mul_lo_u32", "v_mul_hi_u32", "v_mul_lo_i32", "v_mul_hi_i32",
    "v_div_scale_f32", "v_div_scale_f64", "v_div_fmas_f32", "v_div_fmas_f64",
    "v_msad_u8", "v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8", "v_trig_preop_f64",
    "v_mqsad_u32_u8", "v_mad_u64_u32", "v_mad_i64_i32",
    // clang-format on
};

std::string VopcName(uint32_t op, IsaMode isa) {
  if (isa == IsaMode::kNeo) {
    static const char* const kIntCond[6] = {"lt", "eq", "le", "gt", "ne", "ge"};
    if ((op >= 0x89 && op <= 0x8e) || (op >= 0x99 && op <= 0x9e)) {
      const bool cmpx = op >= 0x90;
      return std::string(cmpx ? "v_cmpx_" : "v_cmp_") +
             kIntCond[(op & 0xf) - 9] + "_i16";
    }
    if (op == 0x8f)
      return "v_cmp_class_f16";
    if (op == 0x9f)
      return "v_cmpx_class_f16";
    if ((op >= 0xa9 && op <= 0xae) || (op >= 0xb9 && op <= 0xbe)) {
      const bool cmpx = op >= 0xb0;
      return std::string(cmpx ? "v_cmpx_" : "v_cmp_") +
             kIntCond[(op & 0xf) - 9] + "_u16";
    }
    if ((op >= 0xc8 && op <= 0xcf) || (op >= 0xd8 && op <= 0xdf) ||
        (op >= 0xe8 && op <= 0xef) || (op >= 0xf8 && op <= 0xff)) {
      static const char* const kFloatCond[16] = {
          "f", "lt",  "eq",  "le",  "gt",  "lg",  "ge",  "o",
          "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"};
      const bool cmpx = (op & 0x10) != 0;
      const uint32_t cond = (op & 7) | ((op >= 0xe0) ? 8 : 0);
      return std::string(cmpx ? "v_cmpx_" : "v_cmp_") + kFloatCond[cond] +
             "_f16";
    }
  }
  switch (op) {
    case 0x88:
      return "v_cmp_class_f32";
    case 0x98:
      return "v_cmpx_class_f32";
    case 0xa8:
      return "v_cmp_class_f64";
    case 0xb8:
      return "v_cmpx_class_f64";
    default:
      break;
  }
  static const char* const kFloatCond[16] = {
      "f", "lt",  "eq",  "le",  "gt",  "lg",  "ge",  "o",
      "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"};
  static const char* const kIntCond[8] = {"f",  "lt", "eq", "le",
                                          "gt", "ne", "ge", "t"};
  // Row layout: op[7:4] selects (cmp family, type), op[3:0] the condition.
  static const struct {
    const char* prefix;
    const char* type;
    bool is_float;
  } kRow[16] = {
      {"v_cmp_", "_f32", true},  {"v_cmpx_", "_f32", true},
      {"v_cmp_", "_f64", true},  {"v_cmpx_", "_f64", true},
      {"v_cmps_", "_f32", true}, {"v_cmpsx_", "_f32", true},
      {"v_cmps_", "_f64", true}, {"v_cmpsx_", "_f64", true},
      {"v_cmp_", "_i32", false}, {"v_cmpx_", "_i32", false},
      {"v_cmp_", "_i64", false}, {"v_cmpx_", "_i64", false},
      {"v_cmp_", "_u32", false}, {"v_cmpx_", "_u32", false},
      {"v_cmp_", "_u64", false}, {"v_cmpx_", "_u64", false},
  };
  const uint32_t row = (op >> 4) & 15, cond = op & 15;
  if (!kRow[row].is_float && cond >= 8)
    return Fallback("vopc", op);
  const char* c = kRow[row].is_float ? kFloatCond[cond] : kIntCond[cond];
  return std::string(kRow[row].prefix) + c + kRow[row].type;
}

const char* NeoVop3Name(uint32_t op) {
  switch (op) {
    case 0x303:
      return "v_add_u16";
    case 0x304:
      return "v_sub_u16";
    case 0x305:
      return "v_mul_lo_u16";
    case 0x307:
      return "v_lshrrev_b16";
    case 0x308:
      return "v_ashrrev_i16";
    case 0x309:
      return "v_max_u16";
    case 0x30a:
      return "v_max_i16";
    case 0x30b:
      return "v_min_u16";
    case 0x30c:
      return "v_min_i16";
    case 0x30d:
      return "v_add_i16";
    case 0x30e:
      return "v_sub_i16";
    case 0x311:
      return "v_pack_b32_f16";
    case 0x312:
      return "v_cvt_pknorm_i16_f16";
    case 0x313:
      return "v_cvt_pknorm_u16_f16";
    case 0x314:
      return "v_lshlrev_b16";
    case 0x340:
      return "v_mad_u16";
    case 0x341:
      return "v_mad_f16";
    case 0x342:
      return "v_interp_p1ll_f16";
    case 0x344:
      return "v_perm_b32";
    case 0x345:
      return "v_xad_u32";
    case 0x346:
      return "v_lshl_add_u32";
    case 0x347:
      return "v_add_lshl_u32";
    case 0x34b:
      return "v_fma_f16";
    case 0x351:
      return "v_min3_f16";
    case 0x352:
      return "v_min3_i16";
    case 0x353:
      return "v_min3_u16";
    case 0x354:
      return "v_max3_f16";
    case 0x355:
      return "v_max3_i16";
    case 0x356:
      return "v_max3_u16";
    case 0x357:
      return "v_med3_f16";
    case 0x358:
      return "v_med3_i16";
    case 0x359:
      return "v_med3_u16";
    case 0x35a:
      return "v_interp_p2_f16";
    case 0x35e:
      return "v_mad_i16";
    case 0x35f:
      return "v_div_fixup_f16";
    case 0x36d:
      return "v_add3_u32";
    case 0x36f:
      return "v_lshl_or_b32";
    case 0x371:
      return "v_and_or_b32";
    case 0x372:
      return "v_or3_b32";
    case 0x373:
      return "v_mad_u32_u16";
    case 0x375:
      return "v_mad_i32_i16";
    default:
      return nullptr;
  }
}

std::string Vop3Name(uint32_t op, IsaMode isa) {
  if (isa == IsaMode::kNeo) {
    if (const char* n = NeoVop3Name(op))
      return n;
  }
  if (op < 0x100)
    return VopcName(op, isa);
  if (op >= 0x100 && op < 0x140) {
    if (isa == IsaMode::kNeo) {
      const uint32_t reflected = op - 0x100;
      if (reflected != 0x37 && reflected != 0x38)
        if (const char* n = NeoVop2Name(reflected))
          return n;
    }
    if (const char* n =
            Lookup(kVop2, sizeof(kVop2) / sizeof(kVop2[0]), op - 0x100))
      return n;
    return Fallback("vop3", op);
  }
  if (op >= 0x140 && op < 0x140 + sizeof(kVop3Only) / sizeof(kVop3Only[0])) {
    if (const char* n = kVop3Only[op - 0x140])
      return n;
  }
  if (op >= 0x180) {
    if (isa == IsaMode::kNeo) {
      if (const char* n = NeoVop1Name(op - 0x180))
        return n;
    }
    if (const char* n =
            Lookup(kVop1, sizeof(kVop1) / sizeof(kVop1[0]), op - 0x180))
      return n;
  }
  return Fallback("vop3", op);
}

std::string Vop3pName(uint32_t op) {
  static const char* const kLow[] = {
      // clang-format off
      "v_pk_mad_i16", "v_pk_mul_lo_u16", "v_pk_add_i16", "v_pk_sub_i16",
      "v_pk_lshlrev_b16", "v_pk_lshrrev_b16", "v_pk_ashrrev_i16",
      "v_pk_max_i16", "v_pk_min_i16", "v_pk_mad_u16", "v_pk_add_u16",
      "v_pk_sub_u16", "v_pk_max_u16", "v_pk_min_u16", "v_pk_mad_f16",
      "v_pk_add_f16", "v_pk_mul_f16", "v_pk_min_f16", "v_pk_max_f16",
      // clang-format on
  };
  if (const char* n = Lookup(kLow, sizeof(kLow) / sizeof(kLow[0]), op))
    return n;
  switch (op) {
    case 0x20:
      return "v_mad_mix_f32";
    case 0x21:
      return "v_mad_mixlo_f16";
    case 0x22:
      return "v_mad_mixhi_f16";
    default:
      return Fallback("vop3p", op);
  }
}

std::string DsName(uint32_t op) {
  static const char* const kDs[] = {
      // clang-format off
      "ds_add_u32", "ds_sub_u32", "ds_rsub_u32", "ds_inc_u32", "ds_dec_u32",
      "ds_min_i32", "ds_max_i32", "ds_min_u32", "ds_max_u32",
      "ds_and_b32", "ds_or_b32", "ds_xor_b32", "ds_mskor_b32",
      "ds_write_b32", "ds_write2_b32", "ds_write2st64_b32",
      "ds_cmpst_b32", "ds_cmpst_f32", "ds_min_f32", "ds_max_f32",
      "ds_nop", nullptr, nullptr, nullptr,
      "ds_gws_sema_release_all", "ds_gws_init", "ds_gws_sema_v",
      "ds_gws_sema_br", "ds_gws_sema_p", "ds_gws_barrier",
      "ds_write_b8", "ds_write_b16",
      "ds_add_rtn_u32", "ds_sub_rtn_u32", "ds_rsub_rtn_u32", "ds_inc_rtn_u32",
      "ds_dec_rtn_u32", "ds_min_rtn_i32", "ds_max_rtn_i32", "ds_min_rtn_u32",
      "ds_max_rtn_u32", "ds_and_rtn_b32", "ds_or_rtn_b32", "ds_xor_rtn_b32",
      "ds_mskor_rtn_b32", "ds_wrxchg_rtn_b32", "ds_wrxchg2_rtn_b32",
      "ds_wrxchg2st64_rtn_b32", "ds_cmpst_rtn_b32", "ds_cmpst_rtn_f32",
      "ds_min_rtn_f32", "ds_max_rtn_f32", "ds_wrap_rtn_b32", "ds_swizzle_b32",
      "ds_read_b32", "ds_read2_b32", "ds_read2st64_b32",
      "ds_read_i8", "ds_read_u8", "ds_read_i16", "ds_read_u16",
      "ds_consume", "ds_append", "ds_ordered_count",
      "ds_add_u64", "ds_sub_u64", "ds_rsub_u64", "ds_inc_u64", "ds_dec_u64",
      "ds_min_i64", "ds_max_i64", "ds_min_u64", "ds_max_u64",
      "ds_and_b64", "ds_or_b64", "ds_xor_b64", "ds_mskor_b64",
      "ds_write_b64", "ds_write2_b64", "ds_write2st64_b64",
      "ds_cmpst_b64", "ds_cmpst_f64", "ds_min_f64", "ds_max_f64",
      // clang-format on
  };
  if (const char* n = Lookup(kDs, sizeof(kDs) / sizeof(kDs[0]), op))
    return n;
  switch (op) {
    case 0x60:
      return "ds_add_rtn_u64";
    case 0x61:
      return "ds_sub_rtn_u64";
    case 0x62:
      return "ds_rsub_rtn_u64";
    case 0x63:
      return "ds_inc_rtn_u64";
    case 0x64:
      return "ds_dec_rtn_u64";
    case 0x65:
      return "ds_min_rtn_i64";
    case 0x66:
      return "ds_max_rtn_i64";
    case 0x67:
      return "ds_min_rtn_u64";
    case 0x68:
      return "ds_max_rtn_u64";
    case 0x69:
      return "ds_and_rtn_b64";
    case 0x6a:
      return "ds_or_rtn_b64";
    case 0x6b:
      return "ds_xor_rtn_b64";
    case 0x6c:
      return "ds_mskor_rtn_b64";
    case 0x6d:
      return "ds_wrxchg_rtn_b64";
    case 0x6e:
      return "ds_wrxchg2_rtn_b64";
    case 0x6f:
      return "ds_wrxchg2st64_rtn_b64";
    case 0x70:
      return "ds_cmpst_rtn_b64";
    case 0x71:
      return "ds_cmpst_rtn_f64";
    case 0x72:
      return "ds_min_rtn_f64";
    case 0x73:
      return "ds_max_rtn_f64";
    case 0x76:
      return "ds_read_b64";
    case 0x77:
      return "ds_read2_b64";
    case 0x78:
      return "ds_read2st64_b64";
    case 0x7e:
      return "ds_condxchg32_rtn_b64";
    case 0x80:
      return "ds_add_src2_u32";
    case 0x81:
      return "ds_sub_src2_u32";
    case 0x82:
      return "ds_rsub_src2_u32";
    case 0x83:
      return "ds_inc_src2_u32";
    case 0x84:
      return "ds_dec_src2_u32";
    case 0x85:
      return "ds_min_src2_i32";
    case 0x86:
      return "ds_max_src2_i32";
    case 0x87:
      return "ds_min_src2_u32";
    case 0x88:
      return "ds_max_src2_u32";
    case 0x89:
      return "ds_and_src2_b32";
    case 0x8a:
      return "ds_or_src2_b32";
    case 0x8b:
      return "ds_xor_src2_b32";
    case 0x8d:
      return "ds_write_src2_b32";
    case 0x92:
      return "ds_min_src2_f32";
    case 0x93:
      return "ds_max_src2_f32";
    case 0xc0:
      return "ds_add_src2_u64";
    case 0xc1:
      return "ds_sub_src2_u64";
    case 0xc2:
      return "ds_rsub_src2_u64";
    case 0xc3:
      return "ds_inc_src2_u64";
    case 0xc4:
      return "ds_dec_src2_u64";
    case 0xc5:
      return "ds_min_src2_i64";
    case 0xc6:
      return "ds_max_src2_i64";
    case 0xc7:
      return "ds_min_src2_u64";
    case 0xc8:
      return "ds_max_src2_u64";
    case 0xc9:
      return "ds_and_src2_b64";
    case 0xca:
      return "ds_or_src2_b64";
    case 0xcb:
      return "ds_xor_src2_b64";
    case 0xcd:
      return "ds_write_src2_b64";
    case 0xd2:
      return "ds_min_src2_f64";
    case 0xd3:
      return "ds_max_src2_f64";
    default:
      return Fallback("ds", op);
  }
}

std::string MubufName(uint32_t op) {
  static const char* const kFmt[] = {"x", "xy", "xyz", "xyzw"};
  if (op <= 0x03)
    return std::string("buffer_load_format_") + kFmt[op];
  if (op >= 0x04 && op <= 0x07)
    return std::string("buffer_store_format_") + kFmt[op - 4];
  switch (op) {
    case 0x08:
      return "buffer_load_ubyte";
    case 0x09:
      return "buffer_load_sbyte";
    case 0x0a:
      return "buffer_load_ushort";
    case 0x0b:
      return "buffer_load_sshort";
    case 0x0c:
      return "buffer_load_dword";
    case 0x0d:
      return "buffer_load_dwordx2";
    case 0x0e:
      return "buffer_load_dwordx4";
    case 0x0f:
      return "buffer_load_dwordx3";
    case 0x18:
      return "buffer_store_byte";
    case 0x1a:
      return "buffer_store_short";
    case 0x1c:
      return "buffer_store_dword";
    case 0x1d:
      return "buffer_store_dwordx2";
    case 0x1e:
      return "buffer_store_dwordx4";
    case 0x1f:
      return "buffer_store_dwordx3";
    case 0x70:
      return "buffer_wbinvl1_sc";
    case 0x71:
      return "buffer_wbinvl1";
    default:
      break;
  }
  static const char* const kAtomic[] = {
      "swap", "cmpswap", "add",      "sub",  "rsub", "smin",
      "umin", "smax",    "umax",     "and",  "or",   "xor",
      "inc",  "dec",     "fcmpswap", "fmin", "fmax"};
  if (op >= 0x30 && op <= 0x40)
    return std::string("buffer_atomic_") + kAtomic[op - 0x30];
  if (op >= 0x50 && op <= 0x60)
    return std::string("buffer_atomic_") + kAtomic[op - 0x50] + "_x2";
  return Fallback("mubuf", op);
}

std::string MimgName(uint32_t op) {
  static const char* const kLoadStore[] = {
      // clang-format off
      "image_load", "image_load_mip", "image_load_pck", "image_load_pck_sgn",
      "image_load_mip_pck", "image_load_mip_pck_sgn", nullptr, nullptr,
      "image_store", "image_store_mip", "image_store_pck",
      "image_store_mip_pck", nullptr, nullptr,
      "image_get_resinfo",
      "image_atomic_swap", "image_atomic_cmpswap", "image_atomic_add",
      "image_atomic_sub", "image_atomic_rsub", "image_atomic_smin",
      "image_atomic_umin", "image_atomic_smax", "image_atomic_umax",
      "image_atomic_and", "image_atomic_or", "image_atomic_xor",
      "image_atomic_inc", "image_atomic_dec", "image_atomic_fcmpswap",
      "image_atomic_fmin", "image_atomic_fmax",
      // clang-format on
  };
  if (op < sizeof(kLoadStore) / sizeof(kLoadStore[0]) && kLoadStore[op])
    return kLoadStore[op];
  static const char* const kSample[16] = {
      "",   "_cl",   "_d",   "_d_cl",   "_l",   "_b",   "_b_cl",   "_lz",
      "_c", "_c_cl", "_c_d", "_c_d_cl", "_c_l", "_c_b", "_c_b_cl", "_c_lz"};
  if (op >= 0x20 && op <= 0x2f)
    return std::string("image_sample") + kSample[op - 0x20];
  if (op >= 0x30 && op <= 0x3f)
    return std::string("image_sample") + kSample[op - 0x30] + "_o";
  static const char* const kGather[16] = {
      "",   "_cl",   nullptr, nullptr, "_l",   "_b",   "_b_cl",   "_lz",
      "_c", "_c_cl", nullptr, nullptr, "_c_l", "_c_b", "_c_b_cl", "_c_lz"};
  if (op >= 0x40 && op <= 0x5f) {
    const char* suffix = kGather[(op - 0x40) & 15];
    if (suffix)
      return std::string("image_gather4") + suffix + ((op & 0x10) ? "_o" : "");
  }
  if (op == 0x60)
    return "image_get_lod";
  static const char* const kCd[8] = {"_cd",      "_cd_cl",    "_c_cd",
                                     "_c_cd_cl", "_cd_o",     "_cd_cl_o",
                                     "_c_cd_o",  "_c_cd_cl_o"};
  if (op >= 0x68 && op <= 0x6f)
    return std::string("image_sample") + kCd[op - 0x68];
  return Fallback("mimg", op);
}

std::string ExpTarget(uint32_t target) {
  if (target <= 7)
    return "mrt" + std::to_string(target);
  if (target == 8)
    return "mrtz";
  if (target == 9)
    return "null";
  if (target >= 12 && target <= 15)
    return "pos" + std::to_string(target - 12);
  if (target >= 32 && target <= 63)
    return "param" + std::to_string(target - 32);
  return "target" + std::to_string(target);
}

// SMRD destination width in dwords (from the opcode's x2/x4/... suffix).
uint32_t SmrdCount(uint32_t op) {
  switch (op & 7) {
    case 0:
      return 1;
    case 1:
      return 2;
    case 2:
      return 4;
    case 3:
      return 8;
    case 4:
      return 16;
    default:
      return 1;
  }
}

// MUBUF data-register count.
uint32_t MubufCount(uint32_t op) {
  if (op <= 0x07)
    return (op & 3) + 1;  // format x..xyzw
  switch (op) {
    case 0x0d:
    case 0x1d:
      return 2;
    case 0x0f:
    case 0x1f:
      return 3;
    case 0x0e:
    case 0x1e:
      return 4;
    default:
      return 1;
  }
}

uint32_t PopCount4(uint32_t v) {
  v &= 0xF;
  v = (v & 5) + ((v >> 1) & 5);
  return (v & 3) + ((v >> 2) & 3);
}

std::string SourceMods(std::string src, bool neg, bool abs) {
  if (abs)
    src = "|" + src + "|";
  if (neg)
    src = "-" + src;
  return src;
}

const char* SdwaSelName(uint32_t sel) {
  static const char* const kSel[] = {"BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3",
                                     "WORD_0", "WORD_1", "DWORD"};
  return sel < sizeof(kSel) / sizeof(kSel[0]) ? kSel[sel] : "RESERVED";
}

const char* SdwaUnusedName(uint32_t value) {
  static const char* const kUnused[] = {"UNUSED_PAD", "UNUSED_SEXT",
                                        "UNUSED_PRESERVE", "RESERVED"};
  return kUnused[value & 3];
}

std::string CompactSource(const Inst& inst,
                          uint32_t index,
                          uint32_t vsrc1,
                          uint32_t count) {
  const uint32_t m = inst.raw[1];
  if (inst.extension == InstExtension::kSdwa) {
    const bool scalar = index == 0 ? ((m >> 23) & 1) : ((m >> 31) & 1);
    const uint32_t reg = index == 0 ? (m & 0xff) : vsrc1;
    const uint32_t field = reg + (scalar ? 0 : 256);
    const bool neg = (m >> (index == 0 ? 20 : 28)) & 1;
    const bool abs = (m >> (index == 0 ? 21 : 29)) & 1;
    return SourceMods(Src(field, inst, count), neg, abs);
  }
  if (inst.extension == InstExtension::kDpp) {
    const uint32_t field = (index == 0 ? (m & 0xff) : vsrc1) + 256;
    const bool neg = (m >> (index == 0 ? 20 : 22)) & 1;
    const bool abs = (m >> (index == 0 ? 21 : 23)) & 1;
    return SourceMods(Src(field, inst, count), neg, abs);
  }
  return index == 0 ? Src(inst.raw[0] & 0x1ff, inst, count)
                    : VRange(vsrc1, count);
}

std::string SdwaControls(const Inst& inst, bool has_src1, bool is_vopc) {
  const uint32_t m = inst.raw[1];
  std::string s;
  if (!is_vopc) {
    s += " dst_sel:";
    s += SdwaSelName((m >> 8) & 7);
    s += " dst_unused:";
    s += SdwaUnusedName((m >> 11) & 3);
    if ((m >> 13) & 1)
      s += " clamp";
    const uint32_t omod = (m >> 14) & 3;
    if (omod)
      s += omod == 1 ? " mul:2" : omod == 2 ? " mul:4" : " div:2";
  }
  s += " src0_sel:";
  s += SdwaSelName((m >> 16) & 7);
  if ((m >> 19) & 1)
    s += " src0_sext";
  if (has_src1) {
    s += " src1_sel:";
    s += SdwaSelName((m >> 24) & 7);
    if ((m >> 27) & 1)
      s += " src1_sext";
  }
  return s;
}

std::string DppControlName(uint32_t ctrl) {
  if (ctrl <= 0xff) {
    return "quad_perm:[" + std::to_string(ctrl & 3) + "," +
           std::to_string((ctrl >> 2) & 3) + "," +
           std::to_string((ctrl >> 4) & 3) + "," +
           std::to_string((ctrl >> 6) & 3) + "]";
  }
  if (ctrl >= 0x100 && ctrl <= 0x10f)
    return "row_shl:" + std::to_string(ctrl & 0xf);
  if (ctrl >= 0x110 && ctrl <= 0x11f)
    return "row_shr:" + std::to_string(ctrl & 0xf);
  if (ctrl >= 0x120 && ctrl <= 0x12f)
    return "row_ror:" + std::to_string(ctrl & 0xf);
  switch (ctrl) {
    case 0x130:
      return "wave_shl:1";
    case 0x134:
      return "wave_rol:1";
    case 0x138:
      return "wave_shr:1";
    case 0x13c:
      return "wave_ror:1";
    case 0x140:
      return "row_mirror";
    case 0x141:
      return "row_half_mirror";
    case 0x142:
      return "row_bcast:15";
    case 0x143:
      return "row_bcast:31";
    default:
      return "dpp_ctrl:" + Hex(ctrl);
  }
}

std::string DppControls(const Inst& inst) {
  const uint32_t m = inst.raw[1];
  std::string s = " " + DppControlName((m >> 8) & 0x1ff) +
                  " row_mask:" + Hex((m >> 28) & 0xf) +
                  " bank_mask:" + Hex((m >> 24) & 0xf);
  if ((m >> 19) & 1)
    s += " bound_ctrl:1";
  if ((m >> 18) & 1)
    s += " fi:1";
  return s;
}

std::string CompactControls(const Inst& inst, bool has_src1, bool is_vopc) {
  if (inst.extension == InstExtension::kSdwa)
    return SdwaControls(inst, has_src1, is_vopc);
  if (inst.extension == InstExtension::kDpp)
    return DppControls(inst);
  return "";
}

// ---- per-encoding operand rendering
// ------------------------------------------

std::string OperandsSop1(const Inst& inst, const std::string& name) {
  const uint32_t w = inst.raw[0];
  const uint32_t sdst = (w >> 16) & 0x7F, ssrc = w & 0xFF;
  const uint32_t n = Is64(name) ? 2 : 1;
  return SRange(sdst, n) + ", " + Src(ssrc, inst, n);
}

std::string OperandsSop2(const Inst& inst, const std::string& name) {
  const uint32_t w = inst.raw[0];
  const uint32_t sdst = (w >> 16) & 0x7F;
  const uint32_t n = Is64(name) ? 2 : 1;
  return SRange(sdst, n) + ", " + Src(w & 0xFF, inst, n) + ", " +
         Src((w >> 8) & 0xFF, inst, n);
}

std::string OperandsSopc(const Inst& inst, const std::string& name) {
  const uint32_t w = inst.raw[0];
  const uint32_t n = Is64(name) ? 2 : 1;
  return Src(w & 0xFF, inst, n) + ", " + Src((w >> 8) & 0xFF, inst, n);
}

std::string OperandsSopk(const Inst& inst) {
  const uint32_t w = inst.raw[0];
  return SName((w >> 16) & 0x7F) + ", " + Hex(w & 0xFFFF);
}

std::string OperandsSopp(const Inst& inst) {
  const uint32_t w = inst.raw[0];
  const uint32_t simm = w & 0xFFFF;
  switch (inst.opcode) {
    case 0x01:  // s_endpgm
      return "";
    case 0x02:
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07:
    case 0x08:
    case 0x09: {  // branches: render the absolute dword target
      const int32_t rel = static_cast<int16_t>(simm);
      const int64_t target = static_cast<int64_t>(inst.pc) + inst.size + rel;
      char buf[32];
      std::snprintf(buf, sizeof(buf), "pc%+d -> %04llx", rel,
                    static_cast<unsigned long long>(target));
      return buf;
    }
    case 0x0c: {  // s_waitcnt bitfields
      char buf[64];
      std::snprintf(buf, sizeof(buf), "vmcnt(%u) expcnt(%u) lgkmcnt(%u)",
                    simm & 0xF, (simm >> 4) & 0x7, (simm >> 8) & 0x1F);
      return buf;
    }
    default:
      return simm ? Hex(simm) : "";
  }
}

std::string OperandsSmrd(const Inst& inst) {
  const uint32_t w = inst.raw[0];
  const uint32_t sdst = (w >> 15) & 0x7F, sbase = ((w >> 9) & 0x3F) * 2;
  const bool imm = (w >> 8) & 1;
  const uint32_t off = w & 0xFF;
  const bool buffer = inst.opcode >= 8;
  std::string s = SRange(sdst, SmrdCount(inst.opcode)) + ", " +
                  SRange(sbase, buffer ? 4 : 2) + ", ";
  if (imm)
    s += Hex(off);  // dword offset
  else if (off == 255)
    s += Hex(inst.literal);
  else
    s += SName(off);
  return s;
}

std::string OperandsVop1(const Inst& inst, const std::string& name) {
  const uint32_t w = inst.raw[0];
  const uint32_t n = Is64(name) ? 2 : 1;
  return VRange((w >> 17) & 0xFF, n) + ", " + CompactSource(inst, 0, 0, n) +
         CompactControls(inst, false, false);
}

std::string OperandsVop2(const Inst& inst) {
  const uint32_t w = inst.raw[0];
  const uint32_t vsrc1 = (w >> 9) & 0xff;
  const std::string src0 = CompactSource(inst, 0, vsrc1, 1);
  const std::string src1 = CompactSource(inst, 1, vsrc1, 1);
  std::string s = VRange((w >> 17) & 0xFF, 1) + ", " + src0 + ", ";
  if (inst.isa == IsaMode::kNeo && inst.opcode == 0x37)
    s += Hex(inst.literal) + ", " + src1;
  else
    s += src1;
  // v_madmk/v_madak carry a mandatory literal K.
  if (inst.opcode == 0x20 || inst.opcode == 0x21 ||
      (inst.isa == IsaMode::kNeo && inst.opcode == 0x38))
    s += ", " + Hex(inst.literal);
  s += CompactControls(inst, true, false);
  return s;
}

std::string OperandsVopc(const Inst& inst, const std::string& name) {
  const uint32_t w = inst.raw[0];
  const uint32_t n = Is64(name) ? 2 : 1;
  const uint32_t vsrc1 = (w >> 9) & 0xff;
  std::string dst = "vcc";
  if (inst.extension == InstExtension::kSdwa && ((inst.raw[1] >> 15) & 1))
    dst = SRange((inst.raw[1] >> 8) & 0x7f, 2);
  return dst + ", " + CompactSource(inst, 0, vsrc1, n) + ", " +
         CompactSource(inst, 1, vsrc1, n) + CompactControls(inst, true, true);
}

uint32_t Vop3NumSources(uint32_t op) {
  if (op >= 0x300)
    return op < 0x340 ? 2 : 3;
  if (op >= 0x180 && op < 0x200)
    return 1;
  return op < 0x140 ? 2 : 3;
}

std::string OperandsVop3(const Inst& inst, const std::string& name) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t op = inst.opcode;
  const uint32_t n = Is64(name) ? 2 : 1;
  const uint32_t neg = (w1 >> 29) & 7;
  const bool vop3b = (op >= 0x125 && op <= 0x12a) || op == 0x16d ||
                     op == 0x16e || op == 0x176 || op == 0x177;
  std::string s;
  if (op < 0x100) {  // VOPC via VOP3: destination is an SGPR pair
    s = SRange(w & 0xFF, 2);
  } else {
    s = VRange(w & 0xFF, n);
    // VOP3b (carry ops, div_scale): explicit scalar carry destination.
    const uint32_t sdst = (w >> 8) & 0x7F;
    if (vop3b)
      s += ", " + SRange(sdst, 2);
  }
  const uint32_t srcs[3] = {w1 & 0x1FF, (w1 >> 9) & 0x1FF, (w1 >> 18) & 0x1FF};
  const uint32_t abs = (w >> 8) & 7;
  const uint32_t num_srcs = Vop3NumSources(op);
  for (uint32_t i = 0; i < num_srcs; i++) {
    std::string v = Src(srcs[i], inst, n);
    if (abs & (1u << i))
      v = "|" + v + "|";
    if (neg & (1u << i))
      v = "-" + v;
    s += ", " + v;
  }
  if (inst.isa == IsaMode::kNeo && !vop3b) {
    const uint32_t op_sel = (w >> 12) & 0xf;
    if (op_sel) {
      s += " op_sel:[";
      for (uint32_t i = 0; i < num_srcs; i++) {
        if (i)
          s += ",";
        s += std::to_string((op_sel >> i) & 1);
      }
      s += "," + std::to_string((op_sel >> 3) & 1) + "]";
    }
  }
  if ((w >> 11) & 1)
    s += " clamp";
  const uint32_t omod = (w1 >> 27) & 3;
  if (omod)
    s += omod == 1 ? " mul:2" : omod == 2 ? " mul:4" : " div:2";
  return s;
}

uint32_t Vop3pNumSources(uint32_t op) {
  switch (op) {
    case 0x00:
    case 0x09:
    case 0x0e:
    case 0x20:
    case 0x21:
    case 0x22:
      return 3;
    default:
      return 2;
  }
}

std::string PackedControl(const char* name, uint32_t value, uint32_t count) {
  std::string s = " ";
  s += name;
  s += ":[";
  for (uint32_t i = 0; i < count; i++) {
    if (i)
      s += ",";
    s += std::to_string((value >> i) & 1);
  }
  return s + "]";
}

std::string OperandsVop3p(const Inst& inst) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t num_srcs = Vop3pNumSources(inst.opcode);
  const uint32_t srcs[3] = {w1 & 0x1ff, (w1 >> 9) & 0x1ff, (w1 >> 18) & 0x1ff};
  std::string s = VRange(w & 0xff, 1);
  for (uint32_t i = 0; i < num_srcs; i++)
    s += ", " + Src(srcs[i], inst);

  const uint32_t op_sel = (w >> 11) & 7;
  const uint32_t op_sel_hi = ((w1 >> 27) & 3) | (((w >> 14) & 1) << 2);
  const uint32_t neg_lo = (w1 >> 29) & 7;
  const uint32_t neg_hi = (w >> 8) & 7;
  const uint32_t mask = (1u << num_srcs) - 1;
  if (op_sel & mask)
    s += PackedControl("op_sel", op_sel, num_srcs);
  const uint32_t default_hi = inst.opcode < 0x20 ? mask : 0;
  if ((op_sel_hi & mask) != default_hi)
    s += PackedControl("op_sel_hi", op_sel_hi, num_srcs);
  if (neg_lo & mask)
    s += PackedControl("neg_lo", neg_lo, num_srcs);
  if (neg_hi & mask)
    s += PackedControl("neg_hi", neg_hi, num_srcs);
  if ((w >> 15) & 1)
    s += " clamp";
  return s;
}

std::string OperandsVintrp(const Inst& inst) {
  const uint32_t w = inst.raw[0];
  const uint32_t vsrc = w & 0xFF, chan = (w >> 8) & 3, attr = (w >> 10) & 0x3F;
  const uint32_t vdst = (w >> 18) & 0xFF;
  static const char kChan[4] = {'x', 'y', 'z', 'w'};
  std::string s = VRange(vdst, 1) + ", ";
  if (inst.opcode == 2)  // v_interp_mov: source is P10/P20/P0 selector
    s += "p" + std::to_string(vsrc);
  else
    s += VRange(vsrc, 1);
  s += ", attr" + std::to_string(attr) + ".";
  s += kChan[chan];
  return s;
}

std::string OperandsDs(const Inst& inst) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t off0 = w & 0xFF, off1 = (w >> 8) & 0xFF, gds = (w >> 17) & 1;
  const uint32_t addr = w1 & 0xFF, d0 = (w1 >> 8) & 0xFF,
                 d1 = (w1 >> 16) & 0xFF, vdst = (w1 >> 24) & 0xFF;
  char buf[96];
  std::snprintf(buf, sizeof(buf), "v%u, v%u, v%u, v%u offset0:%u offset1:%u%s",
                vdst, addr, d0, d1, off0, off1, gds ? " gds" : "");
  return buf;
}

std::string OperandsMubuf(const Inst& inst, uint32_t count) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t offset = w & 0xFFF;
  const bool offen = (w >> 12) & 1, idxen = (w >> 13) & 1, glc = (w >> 14) & 1;
  const uint32_t vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
  const bool slc = (w1 >> 22) & 1, tfe = (w1 >> 23) & 1;
  const uint32_t soffset = (w1 >> 24) & 0xFF;
  std::string s = VRange(vdata, count) + ", " +
                  VRange(vaddr, (idxen && offen) ? 2 : 1) + ", " +
                  SRange(srsrc, 4) + ", " + Src(soffset, inst);
  if (idxen)
    s += " idxen";
  if (offen)
    s += " offen";
  if (offset)
    s += " offset:" + Hex(offset);
  if (glc)
    s += " glc";
  if (slc)
    s += " slc";
  if (tfe)
    s += " tfe";
  return s;
}

std::string OperandsMtbuf(const Inst& inst) {
  const uint32_t w = inst.raw[0];
  uint32_t count = (inst.opcode & 3) + 1;
  if (inst.isa == IsaMode::kNeo && inst.opcode >= 8)
    count = (count + 1) / 2;
  std::string s = OperandsMubuf(inst, count);
  s += " dfmt:" + std::to_string((w >> 19) & 0xF) +
       " nfmt:" + std::to_string((w >> 23) & 0x7);
  return s;
}

std::string OperandsMimg(const Inst& inst) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t dmask = (w >> 8) & 0xF;
  const bool unorm = (w >> 12) & 1, glc = (w >> 13) & 1, da = (w >> 14) & 1;
  const uint32_t vaddr = w1 & 0xFF, vdata = (w1 >> 8) & 0xFF;
  const uint32_t srsrc = ((w1 >> 16) & 0x1F) * 4;
  const uint32_t ssamp = ((w1 >> 21) & 0x1F) * 4;
  const uint32_t n = PopCount4(dmask);
  std::string s = VRange(vdata, n ? n : 1) + ", " + VRange(vaddr, 1) + ", " +
                  SRange(srsrc, 8);
  if (inst.opcode >= 0x20)  // sample/gather ops also name an S#
    s += ", " + SRange(ssamp, 4);
  s += " dmask:" + Hex(dmask);
  if (unorm)
    s += " unorm";
  if (glc)
    s += " glc";
  if (da)
    s += " da";
  return s;
}

std::string OperandsExp(const Inst& inst) {
  const uint32_t w = inst.raw[0], w1 = inst.raw[1];
  const uint32_t en = w & 0xF, target = (w >> 4) & 0x3F;
  const bool compr = (w >> 10) & 1, done = (w >> 11) & 1, vm = (w >> 12) & 1;
  std::string s = ExpTarget(target);
  for (uint32_t i = 0; i < 4; i++) {
    s += i == 0 ? " " : ", ";
    s += (en & (1u << i)) ? "v" + std::to_string((w1 >> (8 * i)) & 0xFF)
                          : std::string("off");
  }
  if (compr)
    s += " compr";
  if (done)
    s += " done";
  if (vm)
    s += " vm";
  return s;
}

}  // namespace

std::string Mnemonic(const Inst& inst) {
  const uint32_t op = inst.opcode;
  const char* n = nullptr;
  switch (inst.enc) {
    case Enc::kSop1:
      n = Lookup(kSop1, sizeof(kSop1) / sizeof(kSop1[0]), op);
      return n ? n : Fallback("sop1", op);
    case Enc::kSop2:
      if (inst.isa == IsaMode::kNeo)
        n = NeoSop2Name(op);
      if (n)
        return n;
      n = Lookup(kSop2, sizeof(kSop2) / sizeof(kSop2[0]), op);
      return n ? n : Fallback("sop2", op);
    case Enc::kSopk:
      n = Lookup(kSopk, sizeof(kSopk) / sizeof(kSopk[0]), op);
      return n ? n : Fallback("sopk", op);
    case Enc::kSopc:
      n = Lookup(kSopc, sizeof(kSopc) / sizeof(kSopc[0]), op);
      return n ? n : Fallback("sopc", op);
    case Enc::kSopp:
      n = Lookup(kSopp, sizeof(kSopp) / sizeof(kSopp[0]), op);
      return n ? n : Fallback("sopp", op);
    case Enc::kSmrd:
      n = Lookup(kSmrd, sizeof(kSmrd) / sizeof(kSmrd[0]), op);
      return n ? n : Fallback("smrd", op);
    case Enc::kVop1:
      if (inst.isa == IsaMode::kNeo)
        n = NeoVop1Name(op);
      if (n)
        return n;
      n = Lookup(kVop1, sizeof(kVop1) / sizeof(kVop1[0]), op);
      return n ? n : Fallback("vop1", op);
    case Enc::kVop2:
      if (inst.isa == IsaMode::kNeo)
        n = NeoVop2Name(op);
      if (n)
        return n;
      n = Lookup(kVop2, sizeof(kVop2) / sizeof(kVop2[0]), op);
      return n ? n : Fallback("vop2", op);
    case Enc::kVop3:
      return Vop3Name(op, inst.isa);
    case Enc::kVop3p:
      return Vop3pName(op);
    case Enc::kVopc:
      return VopcName(op, inst.isa);
    case Enc::kVintrp:
      switch (op) {
        case 0:
          return "v_interp_p1_f32";
        case 1:
          return "v_interp_p2_f32";
        case 2:
          return "v_interp_mov_f32";
        default:
          return Fallback("vintrp", op);
      }
    case Enc::kDs:
      return DsName(op);
    case Enc::kMubuf:
      return MubufName(op);
    case Enc::kMtbuf:
      if (inst.isa == IsaMode::kNeo && op >= 8 && op < 16) {
        static const char* const kD16Fmt[] = {"x", "xy", "xyz", "xyzw"};
        return std::string(op < 12 ? "tbuffer_load_format_d16_"
                                   : "tbuffer_store_format_d16_") +
               kD16Fmt[op & 3];
      }
      if (op < 4)
        return std::string("tbuffer_load_format_") + (op == 0   ? "x"
                                                      : op == 1 ? "xy"
                                                      : op == 2 ? "xyz"
                                                                : "xyzw");
      if (op < 8)
        return std::string("tbuffer_store_format_") + (op == 4   ? "x"
                                                       : op == 5 ? "xy"
                                                       : op == 6 ? "xyz"
                                                                 : "xyzw");
      return Fallback("mtbuf", op);
    case Enc::kMimg:
      return MimgName(op);
    case Enc::kExp:
      return "exp";
    default:
      return Fallback("unk", inst.raw[0]);
  }
}

std::string DisasmInst(const Inst& inst) {
  const std::string name = Mnemonic(inst);
  std::string ops;
  switch (inst.enc) {
    case Enc::kSop1:
      ops = OperandsSop1(inst, name);
      break;
    case Enc::kSop2:
      ops = OperandsSop2(inst, name);
      break;
    case Enc::kSopc:
      ops = OperandsSopc(inst, name);
      break;
    case Enc::kSopk:
      ops = OperandsSopk(inst);
      break;
    case Enc::kSopp:
      ops = OperandsSopp(inst);
      break;
    case Enc::kSmrd:
      ops = OperandsSmrd(inst);
      break;
    case Enc::kVop1:
      ops = OperandsVop1(inst, name);
      break;
    case Enc::kVop2:
      ops = OperandsVop2(inst);
      break;
    case Enc::kVopc:
      ops = OperandsVopc(inst, name);
      break;
    case Enc::kVop3:
      ops = OperandsVop3(inst, name);
      break;
    case Enc::kVop3p:
      ops = OperandsVop3p(inst);
      break;
    case Enc::kVintrp:
      ops = OperandsVintrp(inst);
      break;
    case Enc::kDs:
      ops = OperandsDs(inst);
      break;
    case Enc::kMubuf:
      ops = OperandsMubuf(inst, MubufCount(inst.opcode));
      break;
    case Enc::kMtbuf:
      ops = OperandsMtbuf(inst);
      break;
    case Enc::kMimg:
      ops = OperandsMimg(inst);
      break;
    case Enc::kExp:
      ops = OperandsExp(inst);
      break;
    default:
      break;
  }
  return ops.empty() ? name : name + " " + ops;
}

std::string DisasmLine(const Inst& inst) {
  char head[40];
  if (inst.size >= 2 && !inst.has_literal)
    std::snprintf(head, sizeof(head), "%04x: %08x %08x          ", inst.pc,
                  inst.raw[0], inst.raw[1]);
  else if (inst.size >= 3)  // two encoding words + literal
    std::snprintf(head, sizeof(head), "%04x: %08x %08x %08x ", inst.pc,
                  inst.raw[0], inst.raw[1], inst.literal);
  else if (inst.has_literal)
    std::snprintf(head, sizeof(head), "%04x: %08x %08x          ", inst.pc,
                  inst.raw[0], inst.literal);
  else
    std::snprintf(head, sizeof(head), "%04x: %08x                   ", inst.pc,
                  inst.raw[0]);
  return std::string(head) + DisasmInst(inst);
}

void Disassemble(const uint32_t* code, uint32_t max_dwords, const char* tag) {
  const Program program = Decode(code, max_dwords);
  std::fprintf(stderr, "[gcn] %s: %zu instructions\n", tag, program.size());
  for (const Inst& inst : program)
    std::fprintf(stderr, "[gcn]   %s\n", DisasmLine(inst).c_str());
}

}  // namespace gpu::gcn
