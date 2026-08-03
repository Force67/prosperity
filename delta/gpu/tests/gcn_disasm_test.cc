#include <cstdint>

#include <gtest/gtest.h>

#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_disasm.h"

namespace {

gpu::gcn::Inst DecodeOne(const uint32_t* code,
                         uint32_t dwords,
                         gpu::gcn::IsaMode mode = gpu::gcn::IsaMode::kBase) {
  const gpu::gcn::Program program = gpu::gcn::Decode(code, dwords, true, mode);
  EXPECT_FALSE(program.empty());
  return program.empty() ? gpu::gcn::Inst{} : program[0];
}

std::string Name(gpu::gcn::Enc enc,
                 uint32_t opcode,
                 gpu::gcn::IsaMode mode = gpu::gcn::IsaMode::kBase) {
  gpu::gcn::Inst inst;
  inst.isa = mode;
  inst.enc = enc;
  inst.opcode = opcode;
  return gpu::gcn::Mnemonic(inst);
}

TEST(GcnDisasm, Sop1MovRegister) {
  const uint32_t code[] = {0xbe800301};  // s_mov_b32 s0, s1
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 1)), "s_mov_b32 s0, s1");
}

TEST(GcnDisasm, Sop1MovInlineZero) {
  const uint32_t code[] = {0xbe800380};  // s_mov_b32 s0, 0
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 1)), "s_mov_b32 s0, 0");
}

TEST(GcnDisasm, Sop1Mov64UsesPairs) {
  // s_mov_b64 s[2:3], vcc (op 0x04, sdst=2, ssrc=106)
  const uint32_t code[] = {0xbe82046a};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 1)), "s_mov_b64 s[2:3], vcc");
}

TEST(GcnDisasm, SmrdLoadWithLiteralOffset) {
  const uint32_t code[] = {
      0xc0c216ff,  // s_load_dwordx8 s[4:11], s[22:23], 0x1c14
      0x00001c14,
  };
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 2)),
            "s_load_dwordx8 s[4:11], s[22:23], 0x1c14");
}

TEST(GcnDisasm, Vop2AddF32) {
  // v_add_f32 v1, s2, v3 (op 3, vdst=1, vsrc1=3, src0=2)
  const uint32_t code[] = {0x06020602};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 1)), "v_add_f32 v1, s2, v3");
}

TEST(GcnDisasm, VopcGeneratedName) {
  // v_cmp_lt_f32 vcc, v1, v0 (op 1, src0=v1=257, vsrc1=0)
  const uint32_t code[] = {0x7c020101};
  const gpu::gcn::Inst inst = DecodeOne(code, 1);
  EXPECT_EQ(gpu::gcn::Mnemonic(inst), "v_cmp_lt_f32");
  EXPECT_EQ(gpu::gcn::DisasmInst(inst), "v_cmp_lt_f32 vcc, v1, v0");
}

TEST(GcnDisasm, Vop3MadF32) {
  // v_mad_f32 v0, v0, v1, v2 (op 0x141)
  const uint32_t code[] = {0xd2820000, 0x040a0300};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 2)),
            "v_mad_f32 v0, v0, v1, v2");
}

TEST(GcnDisasm, MubufFormatLoad) {
  // buffer_load_format_xyzw v[4:7], v0, s[8:11], 0 idxen
  const uint32_t code[] = {0xe00c2000, 0x80020400};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 2)),
            "buffer_load_format_xyzw v[4:7], v0, s[8:11], 0 idxen");
}

TEST(GcnDisasm, ExpPositionExport) {
  // exp pos0 v0, v1, v2, v3 (en=0xf, target=12)
  const uint32_t code[] = {0xf80000cf, 0x03020100};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 2)),
            "exp pos0 v0, v1, v2, v3");
}

TEST(GcnDisasm, BranchRendersAbsoluteTarget) {
  // s_cbranch_scc0 +3 at pc 0 -> target 0x4
  const uint32_t code[] = {0xbf840003};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 1)),
            "s_cbranch_scc0 pc+3 -> 0004");
}

TEST(GcnDisasm, WaitcntDecodesFields) {
  // s_waitcnt vmcnt(0) expcnt(7) lgkmcnt(0)
  const uint32_t code[] = {0xbf8c0070};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 1)),
            "s_waitcnt vmcnt(0) expcnt(7) lgkmcnt(0)");
}

TEST(GcnDisasm, UnknownOpcodeFallsBackGreppable) {
  // SOP1 with an out-of-table opcode (0xf0)
  const uint32_t code[] = {0xbe80f001};
  EXPECT_EQ(gpu::gcn::Mnemonic(DecodeOne(code, 1)), "sop1_op0xf0");
}

TEST(GcnDisasm, BaseDsInventoryNames) {
  struct Case {
    uint32_t opcode;
    const char* name;
  };
  static const Case cases[] = {
      {0x60, "ds_add_rtn_u64"},        {0x61, "ds_sub_rtn_u64"},
      {0x62, "ds_rsub_rtn_u64"},       {0x63, "ds_inc_rtn_u64"},
      {0x64, "ds_dec_rtn_u64"},        {0x65, "ds_min_rtn_i64"},
      {0x66, "ds_max_rtn_i64"},        {0x67, "ds_min_rtn_u64"},
      {0x68, "ds_max_rtn_u64"},        {0x69, "ds_and_rtn_b64"},
      {0x6a, "ds_or_rtn_b64"},         {0x6b, "ds_xor_rtn_b64"},
      {0x6c, "ds_mskor_rtn_b64"},      {0x6d, "ds_wrxchg_rtn_b64"},
      {0x6e, "ds_wrxchg2_rtn_b64"},    {0x6f, "ds_wrxchg2st64_rtn_b64"},
      {0x70, "ds_cmpst_rtn_b64"},      {0x71, "ds_cmpst_rtn_f64"},
      {0x72, "ds_min_rtn_f64"},        {0x73, "ds_max_rtn_f64"},
      {0x7e, "ds_condxchg32_rtn_b64"}, {0x80, "ds_add_src2_u32"},
      {0x81, "ds_sub_src2_u32"},       {0x82, "ds_rsub_src2_u32"},
      {0x83, "ds_inc_src2_u32"},       {0x84, "ds_dec_src2_u32"},
      {0x85, "ds_min_src2_i32"},       {0x86, "ds_max_src2_i32"},
      {0x87, "ds_min_src2_u32"},       {0x88, "ds_max_src2_u32"},
      {0x89, "ds_and_src2_b32"},       {0x8a, "ds_or_src2_b32"},
      {0x8b, "ds_xor_src2_b32"},       {0x8c, "ds_write_src2_b32"},
      {0x92, "ds_min_src2_f32"},       {0x93, "ds_max_src2_f32"},
      {0xc0, "ds_add_src2_u64"},       {0xc1, "ds_sub_src2_u64"},
      {0xc2, "ds_rsub_src2_u64"},      {0xc3, "ds_inc_src2_u64"},
      {0xc4, "ds_dec_src2_u64"},       {0xc5, "ds_min_src2_i64"},
      {0xc6, "ds_max_src2_i64"},       {0xc7, "ds_min_src2_u64"},
      {0xc8, "ds_max_src2_u64"},       {0xc9, "ds_and_src2_b64"},
      {0xca, "ds_or_src2_b64"},        {0xcb, "ds_xor_src2_b64"},
      {0xcc, "ds_write_src2_b64"},     {0xd2, "ds_min_src2_f64"},
      {0xd3, "ds_max_src2_f64"},       {0xde, "ds_write_b96"},
      {0xdf, "ds_write_b128"},         {0xfd, "ds_condxchg32_rtn_b128"},
      {0xfe, "ds_read_b96"},           {0xff, "ds_read_b128"},
  };

  for (const Case& c : cases)
    EXPECT_EQ(Name(gpu::gcn::Enc::kDs, c.opcode), c.name) << c.opcode;
}

TEST(GcnDisasm, BaseMubufCacheInvalidateNames) {
  EXPECT_EQ(Name(gpu::gcn::Enc::kMubuf, 0x70), "buffer_wbinvl1_vol");
  EXPECT_EQ(Name(gpu::gcn::Enc::kMubuf, 0x71), "buffer_wbinvl1");
}

TEST(GcnDisasm, SeaIslandsSparseOpcodeNames) {
  using gpu::gcn::Enc;
  EXPECT_EQ(Name(Enc::kSopp, 0x0b), "s_setkill");
  EXPECT_EQ(Name(Enc::kVop1, 0x45), "v_log_legacy_f32");
  EXPECT_EQ(Name(Enc::kVop1, 0x46), "v_exp_legacy_f32");
  EXPECT_EQ(Name(Enc::kVop3, 0x120), "vop3_op0x120");
  EXPECT_EQ(Name(Enc::kVop3, 0x121), "vop3_op0x121");
  EXPECT_EQ(Name(Enc::kDs, 0x8d), "ds_op0x8d");
  EXPECT_EQ(Name(Enc::kDs, 0xcd), "ds_op0xcd");
  EXPECT_EQ(Name(Enc::kMubuf, 0x34), "mubuf_op0x34");
  EXPECT_EQ(Name(Enc::kMubuf, 0x54), "mubuf_op0x54");
  EXPECT_EQ(Name(Enc::kMimg, 0x13), "mimg_op0x13");
  // gather4 has no derivative forms, so 0x42/0x43 and 0x4a/0x4b are holes and
  // the entries after them are NOT shifted up.
  EXPECT_EQ(Name(Enc::kMimg, 0x42), "mimg_op0x42");
  EXPECT_EQ(Name(Enc::kMimg, 0x44), "image_gather4_l");
  EXPECT_EQ(Name(Enc::kMimg, 0x47), "image_gather4_lz");
  EXPECT_EQ(Name(Enc::kMimg, 0x48), "image_gather4_c");
  EXPECT_EQ(Name(Enc::kMimg, 0x4b), "mimg_op0x4b");
  EXPECT_EQ(Name(Enc::kMimg, 0x57), "image_gather4_lz_o");
  EXPECT_EQ(Name(Enc::kFlat, 0x0c), "flat_load_dword");
  EXPECT_EQ(Name(Enc::kFlat, 0x31), "flat_atomic_cmpswap");
}

TEST(GcnDisasm, MixedScalarOperandWidthsAndSpecialForms) {
  const uint32_t count64[] = {0xbe821004};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(count64, 1)),
            "s_bcnt1_i32_b64 s2, s[4:5]");

  const uint32_t setpc[] = {0xbe802002};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(setpc, 1)), "s_setpc_b64 s[2:3]");

  const uint32_t setreg[] = {(0xbu << 28) | (0x15u << 23) | 0x1234u,
                             0x89abcdef};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(setreg, 2)),
            "s_setreg_imm32_b32 0x1234, 0x89abcdef");

  const uint32_t memtime[] = {(0x18u << 27) | (0x1eu << 22) | (2u << 15)};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(memtime, 1)), "s_memtime s[2:3]");
}

TEST(GcnDisasm, VectorSpecialOperandsAndVop3Arities) {
  const uint32_t nop[] = {(0x3fu << 25)};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(nop, 1)), "v_nop");

  const uint32_t readfirst[] = {(0x3fu << 25) | (2u << 17) | (2u << 9) | 260u};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(readfirst, 1)),
            "v_readfirstlane_b32 s2, v4");

  const uint32_t readlane[] = {(1u << 25) | (2u << 17) | (3u << 9) | 260u};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(readlane, 1)),
            "v_readlane_b32 s2, v4, s3");

  const uint32_t cndmask[] = {
      (0x34u << 26) | (0x100u << 17),
      257u | (258u << 9) | (259u << 18),
  };
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(cndmask, 2)),
            "v_cndmask_b32 v0, v1, v2, v3");

  const uint32_t shift64[] = {
      (0x34u << 26) | (0x161u << 17),
      257u | (258u << 9) | (259u << 18),
  };
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(shift64, 2)),
            "v_lshl_b64 v[0:1], v[1:2], v2");

  const uint32_t addc[] = {
      (0x34u << 26) | (0x128u << 17) | (14u << 8),
      257u | (258u << 9) | (259u << 18),
  };
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(addc, 2)),
            "v_addc_u32 v0, s[14:15], v1, v2, v3");

  const uint32_t mad64[] = {
      (0x34u << 26) | (0x176u << 17),
      257u | (258u << 9) | (259u << 18),
  };
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(mad64, 2)),
            "v_mad_u64_u32 v[0:1], vcc, v1, v2, v[3:4]");
}

TEST(GcnDisasm, MemoryFieldsAndCompressedExport) {
  const uint32_t mubuf[] = {
      (0x38u << 26) | (0x0cu << 18) | (1u << 15),
      (4u << 8) | (128u << 24),
  };
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(mubuf, 2)),
            "buffer_load_dword v4, v[0:1], s[0:3], 0 addr64");

  const uint32_t mimg[] = {
      (0x3cu << 26) | (0x20u << 18) | (0xfu << 8) | (1u << 15) | (1u << 16) |
          (1u << 17) | (1u << 25),
      (4u << 8) | (2u << 16) | (2u << 21),
  };
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(mimg, 2)),
            "image_sample v[4:8], v0, s[8:11], s[8:11] dmask:0xf r128 tfe "
            "lwe slc");

  const uint32_t exp[] = {0xf80004c5, 0x04030201};
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(exp, 2)), "exp pos0 v1, v2 compr");

  const uint32_t flat[] = {
      (0x37u << 26) | (0x0cu << 18),
      2u << 24,
  };
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(flat, 2)),
            "flat_load_dword v2, v[0:1]");
}

TEST(GcnDisasm, NeoNamesAreModeDependent) {
  using gpu::gcn::Enc;
  using gpu::gcn::IsaMode;

  EXPECT_EQ(Name(Enc::kSop2, 0x32, IsaMode::kBase), "sop2_op0x32");
  EXPECT_EQ(Name(Enc::kSop2, 0x32, IsaMode::kNeo), "s_pack_ll_b32_b16");
  EXPECT_EQ(Name(Enc::kVop1, 0x50, IsaMode::kBase), "vop1_op0x50");
  EXPECT_EQ(Name(Enc::kVop1, 0x50, IsaMode::kNeo), "v_cvt_f16_u16");
  EXPECT_EQ(Name(Enc::kVop2, 0x32, IsaMode::kBase), "vop2_op0x32");
  EXPECT_EQ(Name(Enc::kVop2, 0x32, IsaMode::kNeo), "v_add_f16");
  EXPECT_EQ(Name(Enc::kVop2, 0x36, IsaMode::kNeo), "v_mac_f16");
  EXPECT_EQ(Name(Enc::kVop2, 0x37, IsaMode::kNeo), "v_madmk_f16");
  EXPECT_EQ(Name(Enc::kVop2, 0x38, IsaMode::kNeo), "v_madak_f16");
  EXPECT_EQ(Name(Enc::kVopc, 0xc9, IsaMode::kBase), "vopc_op0xc9");
  EXPECT_EQ(Name(Enc::kVopc, 0xc9, IsaMode::kNeo), "v_cmp_lt_f16");
  EXPECT_EQ(Name(Enc::kVop3, 0x132, IsaMode::kNeo), "v_add_f16");
  EXPECT_EQ(Name(Enc::kVop3, 0x136, IsaMode::kNeo), "v_mac_f16");
  EXPECT_EQ(Name(Enc::kVop3, 0x1d0, IsaMode::kNeo), "v_cvt_f16_u16");
  EXPECT_EQ(Name(Enc::kVop3, 0x36d, IsaMode::kNeo), "v_add3_u32");
  EXPECT_EQ(Name(Enc::kVop3p, 0x22, IsaMode::kNeo), "v_mad_mixhi_f16");
  EXPECT_EQ(Name(Enc::kMtbuf, 0x8, IsaMode::kBase), "mtbuf_op0x8");
  EXPECT_EQ(Name(Enc::kMtbuf, 0x8, IsaMode::kNeo), "tbuffer_load_format_d16_x");
  EXPECT_EQ(Name(Enc::kMtbuf, 0xf, IsaMode::kNeo),
            "tbuffer_store_format_d16_xyzw");
}

TEST(GcnDisasm, NeoFp16MadLiteralOrder) {
  const uint32_t madmk[] = {
      (0x37u << 25) | (1u << 17) | (2u << 9) | 256u,
      0x00003c00,
  };
  const uint32_t madak[] = {
      (0x38u << 25) | (1u << 17) | (2u << 9) | 256u,
      0x00003c00,
  };

  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(madmk, 2, gpu::gcn::IsaMode::kNeo)),
            "v_madmk_f16 v1, v0, 0x3c00, v2");
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(madak, 2, gpu::gcn::IsaMode::kNeo)),
            "v_madak_f16 v1, v0, v2, 0x3c00");
}

TEST(GcnDisasm, NeoVopcSixteenBitFamilies) {
  static const char* const int_cond[] = {"lt", "eq", "le", "gt", "ne", "ge"};
  for (uint32_t i = 0; i < 6; i++) {
    EXPECT_EQ(Name(gpu::gcn::Enc::kVopc, 0x89 + i, gpu::gcn::IsaMode::kNeo),
              std::string("v_cmp_") + int_cond[i] + "_i16");
    EXPECT_EQ(Name(gpu::gcn::Enc::kVopc, 0x99 + i, gpu::gcn::IsaMode::kNeo),
              std::string("v_cmpx_") + int_cond[i] + "_i16");
    EXPECT_EQ(Name(gpu::gcn::Enc::kVopc, 0xa9 + i, gpu::gcn::IsaMode::kNeo),
              std::string("v_cmp_") + int_cond[i] + "_u16");
    EXPECT_EQ(Name(gpu::gcn::Enc::kVopc, 0xb9 + i, gpu::gcn::IsaMode::kNeo),
              std::string("v_cmpx_") + int_cond[i] + "_u16");
  }
  EXPECT_EQ(Name(gpu::gcn::Enc::kVopc, 0x8f, gpu::gcn::IsaMode::kNeo),
            "v_cmp_class_f16");
  EXPECT_EQ(Name(gpu::gcn::Enc::kVopc, 0x9f, gpu::gcn::IsaMode::kNeo),
            "v_cmpx_class_f16");

  static const char* const float_cond[] = {
      "f", "lt",  "eq",  "le",  "gt",  "lg",  "ge",  "o",
      "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"};
  for (uint32_t i = 0; i < 16; i++) {
    const uint32_t cmp = (i < 8 ? 0xc8 : 0xe0) + i;
    EXPECT_EQ(Name(gpu::gcn::Enc::kVopc, cmp, gpu::gcn::IsaMode::kNeo),
              std::string("v_cmp_") + float_cond[i] + "_f16");
    EXPECT_EQ(Name(gpu::gcn::Enc::kVopc, cmp + 0x10, gpu::gcn::IsaMode::kNeo),
              std::string("v_cmpx_") + float_cond[i] + "_f16");
  }
}

TEST(GcnDisasm, NeoVop3OpSelAndModifiers) {
  constexpr uint32_t op = 0x340;  // v_mad_u16
  const uint32_t code[] = {
      (0x34u << 26) | ((op & 0x1ff) << 17) | ((op >> 9) << 16) | 4u |
          (0xdu << 12) | (1u << 11) | (1u << 8),
      261u | (262u << 9) | (263u << 18) | (1u << 30) | (1u << 27),
  };
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 2, gpu::gcn::IsaMode::kNeo)),
            "v_mad_u16 v4, |v5|, -v6, v7 op_sel:[1,0,1,1] clamp mul:2");
}

TEST(GcnDisasm, NeoVop3pPackedControls) {
  const uint32_t code[] = {
      (0x33u << 26) | 4u | (1u << 9) | (1u << 11) | (1u << 13) | (1u << 15),
      257u | (258u << 9) | (259u << 18) | (1u << 28) | (1u << 29) | (1u << 31),
  };
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 2, gpu::gcn::IsaMode::kNeo)),
            "v_pk_mad_i16 v4, v1, v2, v3 op_sel:[1,0,1] "
            "op_sel_hi:[0,1,0] neg_lo:[1,0,1] neg_hi:[0,1,0] clamp");
}

TEST(GcnDisasm, NeoSdwaExposesSourcesAndControls) {
  const uint32_t code[] = {
      (0x32u << 25) | (3u << 17) | (2u << 9) | 249u,
      5u | (5u << 8) | (2u << 11) | (1u << 13) | (1u << 14) | (2u << 16) |
          (1u << 20) | (1u << 21) | (4u << 24) | (1u << 28) | (1u << 29) |
          (1u << 31),
  };
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 2, gpu::gcn::IsaMode::kNeo)),
            "v_add_f16 v3, -|v5|, -|s2| dst_sel:WORD_1 "
            "dst_unused:UNUSED_PRESERVE clamp mul:2 src0_sel:BYTE_2 "
            "src1_sel:WORD_0");
}

TEST(GcnDisasm, NeoDppExposesSourcesAndControls) {
  const uint32_t code[] = {
      (3u << 25) | (4u << 17) | (6u << 9) | 250u,
      5u | (0x127u << 8) | (1u << 19) | (1u << 20) | (1u << 21) | (1u << 22) |
          (1u << 23) | (0xcu << 24) | (3u << 28),
  };
  EXPECT_EQ(gpu::gcn::DisasmInst(DecodeOne(code, 2, gpu::gcn::IsaMode::kNeo)),
            "v_add_f32 v4, -|v5|, -|v6| row_ror:7 row_mask:0x3 "
            "bank_mask:0xc bound_ctrl:1");
}

// Junk must never crash the renderer's diagnostics: disassemble arbitrary
// words through every encoding classifier.
TEST(GcnDisasm, ArbitraryWordsNeverCrash) {
  uint32_t lcg = 0x12345678;
  for (int i = 0; i < 20000; i++) {
    lcg = lcg * 1664525u + 1013904223u;
    const uint32_t code[2] = {lcg, lcg ^ 0xdeadbeef};
    const gpu::gcn::Program program =
        gpu::gcn::Decode(code, 2, /*stop_at_endpgm=*/false);
    for (const gpu::gcn::Inst& inst : program)
      EXPECT_FALSE(gpu::gcn::DisasmInst(inst).empty());
  }
}

}  // namespace
