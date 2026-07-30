#include <cstdint>

#include <gtest/gtest.h>

#include "gpu/ps5/rdna/rdna_decode.h"

namespace {

uint32_t Vop2(uint32_t op, uint32_t src0 = 256, uint32_t vsrc1 = 0) {
  return (op << 25) | (vsrc1 << 9) | src0;
}

uint32_t Sopp(uint32_t op) {
  return (0x17Fu << 23) | (op << 16);
}

uint32_t Sopk(uint32_t op) {
  return (0xBu << 28) | (op << 23);
}

void Vop3(uint32_t* out,
          uint32_t op,
          uint32_t src0,
          uint32_t src1,
          uint32_t src2) {
  out[0] = (0x35u << 26) | (op << 16);
  out[1] = src0 | (src1 << 9) | (src2 << 18);
}

TEST(RdnaDecode, Dpp8AndDpp8FiConsumeControlDwords) {
  const uint32_t code[] = {
      Vop2(0x03, 233), 0x01234567, Vop2(0x03, 234), 0x89abcdef, Sopp(0x01),
  };

  const gpu::gcn::Program program = gpu::rdna::Decode(code, 5);

  ASSERT_EQ(program.size(), 3u);
  EXPECT_EQ(program[0].extension, gpu::gcn::InstExtension::kDpp8);
  EXPECT_FALSE(program[0].has_literal);
  EXPECT_EQ(program[0].raw[1], 0x01234567u);
  EXPECT_EQ(program[0].size, 2u);
  EXPECT_EQ(program[1].extension, gpu::gcn::InstExtension::kDpp8Fi);
  EXPECT_FALSE(program[1].has_literal);
  EXPECT_EQ(program[1].raw[1], 0x89abcdefu);
  EXPECT_EQ(program[1].pc, 2u);
  EXPECT_EQ(program[2].pc, 4u);
}

TEST(RdnaDecode, SdwaUsesExtensionMetadataInsteadOfLiteralMetadata) {
  const uint32_t code[] = {Vop2(0x03, 249), 0x12345678, Sopp(0x01)};

  const gpu::gcn::Program program = gpu::rdna::Decode(code, 3);

  ASSERT_EQ(program.size(), 2u);
  EXPECT_EQ(program[0].extension, gpu::gcn::InstExtension::kSdwa);
  EXPECT_FALSE(program[0].has_literal);
  EXPECT_EQ(program[0].raw[1], 0x12345678u);
}

TEST(RdnaDecode, Vop3pAndFlatHaveDistinctFamilies) {
  const uint32_t code[] = {
      (0x33u << 26) | (0x0Fu << 16) | (1u << 14),
      3u << 27,
      (0x37u << 26) | (0x12u << 18),
      0,
  };

  const gpu::gcn::Program program = gpu::rdna::Decode(code, 4, false);

  ASSERT_EQ(program.size(), 2u);
  EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kVop3p);
  EXPECT_EQ(program[0].opcode, 0x0Fu);
  EXPECT_EQ(program[1].enc, gpu::gcn::Enc::kFlat);
  EXPECT_EQ(program[1].opcode, 0x12u);
}

TEST(RdnaDecode, ReservedVop3pPrefixesRemainUnknown) {
  const uint32_t code[] = {0xCD000000u, (0x33u << 26) | (1u << 23), 0};

  const gpu::gcn::Program program = gpu::rdna::Decode(code, 3, false);

  ASSERT_EQ(program.size(), 3u);
  EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kUnknown);
  EXPECT_EQ(program[1].enc, gpu::gcn::Enc::kUnknown);
}

TEST(RdnaDecode, MandatoryImmediateDwordsPreserveInstructionBoundaries) {
  const uint32_t code[] = {
      Sopk(0x15), 0x11111111,  // s_setreg_imm32_b32
      Vop2(0x2C), 0x44444444,  // v_fmamk_f32
      Vop2(0x2D), 0x55555555,  // v_fmaak_f32
      Vop2(0x37), 0x66666666,  // v_fmamk_f16
      Vop2(0x38), 0x77777777,  // v_fmaak_f16
      Vop2(0x1F),              // ordinary one-dword gfx10 VOP2
      Sopp(0x01),
  };

  const gpu::gcn::Program program = gpu::rdna::Decode(code, 12, false);

  ASSERT_EQ(program.size(), 7u);
  for (uint32_t i = 0; i < 5; i++) {
    EXPECT_EQ(program[i].pc, i * 2);
    EXPECT_EQ(program[i].size, 2u);
    EXPECT_TRUE(program[i].has_literal);
  }
  EXPECT_EQ(program[5].pc, 10u);
  EXPECT_EQ(program[5].size, 1u);
  EXPECT_EQ(program[6].pc, 11u);
}

TEST(RdnaDecode, ReservedGcnMadkSlotsDoNotConsumeFollowingInstructions) {
  for (uint32_t op : {0x20u, 0x21u}) {
    const uint32_t code[] = {Vop2(op), Sopp(0x01)};
    const gpu::gcn::Program program = gpu::rdna::Decode(code, 2, false);
    ASSERT_EQ(program.size(), 2u);
    EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kVop2);
    EXPECT_EQ(program[0].size, 1u);
    EXPECT_EQ(program[1].enc, gpu::gcn::Enc::kSopp);
    EXPECT_EQ(program[1].pc, 1u);
  }
}

TEST(RdnaDecode, FmacImplicitAccumulatorDoesNotSelectLiteral) {
  uint32_t code[3] = {};
  Vop3(code, 0x12B, 256, 257, 255);
  code[2] = Sopp(0x01);

  const gpu::gcn::Program program = gpu::rdna::Decode(code, 3, false);

  ASSERT_EQ(program.size(), 2u);
  EXPECT_EQ(program[0].opcode, 0x12Bu);
  EXPECT_FALSE(program[0].has_literal);
  EXPECT_EQ(program[0].size, 2u);
  EXPECT_EQ(program[1].pc, 2u);
}

// Anchored on a real gfx1030 encoding rather than on our own field choice:
// `llvm-mc -arch=amdgcn -mcpu=gfx1030` assembles ds_write_b32 (opcode 13) to
// 0xd8340000. Reading the gfx9 field [24:17] instead yields 26, i.e. op*2.
TEST(RdnaDecode, DsOpcodeUsesBits25Through18) {
  const uint32_t code[] = {0xd8340000u, 0};
  const gpu::gcn::Program program = gpu::rdna::Decode(code, 2, false);
  ASSERT_EQ(program.size(), 1u);
  EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kDs);
  EXPECT_EQ(program[0].opcode, 13u);
}

// GDS moved to bit 17, so it must not bleed into the opcode.
TEST(RdnaDecode, DsGdsBitIsNotPartOfTheOpcode) {
  const uint32_t code[] = {0xd8340000u | (1u << 17), 0};
  const gpu::gcn::Program program = gpu::rdna::Decode(code, 2, false);
  ASSERT_EQ(program.size(), 1u);
  EXPECT_EQ(program[0].opcode, 13u);
}

TEST(RdnaDecode, VopcDppFormsRetainTheirControlDword) {
  for (uint32_t source : {233u, 234u, 250u}) {
    const uint32_t code[] = {(0x3eu << 25) | source, 0x12345678};
    const gpu::gcn::Program program = gpu::rdna::Decode(code, 2, false);
    ASSERT_EQ(program.size(), 1u);
    EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kVopc);
    EXPECT_EQ(program[0].size, 2u);
  }
}

TEST(RdnaDecode, ObservedNggExportPrefixOnlyAcceptsNullExports) {
  const uint32_t code[] = {
      0x31u << 26,
      0,
      (0x31u << 26) | 1u,
      0,
  };

  const gpu::gcn::Program program = gpu::rdna::Decode(code, 4, false);

  ASSERT_EQ(program.size(), 2u);
  EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kExp);
  EXPECT_EQ(program[0].size, 2u);
  EXPECT_EQ(program[1].enc, gpu::gcn::Enc::kUnknown);
  EXPECT_EQ(program[1].size, 2u);
}

TEST(RdnaDecode, MimgNsa3PreservesAllFiveWords) {
  const uint32_t code[] = {
      (0x3Cu << 26) | (3u << 1), 0x11111111, 0x22222222, 0x33333333, 0x44444444,
  };

  const gpu::gcn::Program program = gpu::rdna::Decode(code, 5, false);

  ASSERT_EQ(program.size(), 1u);
  EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kMimg);
  EXPECT_EQ(program[0].size, 5u);
  EXPECT_EQ(program[0].raw[0], code[0]);
  EXPECT_EQ(program[0].raw[1], code[1]);
  EXPECT_EQ(program[0].raw[2], code[2]);
  EXPECT_EQ(program[0].raw[3], code[3]);
  EXPECT_EQ(program[0].raw[4], code[4]);
}

TEST(RdnaDecode, TruncatedMultiwordInstructionsBecomeBoundedUnknowns) {
  const uint32_t vop3[] = {(0x35u << 26) | (0x141u << 16)};
  const uint32_t nsa3[] = {(0x3Cu << 26) | (3u << 1), 1, 2, 3};
  const uint32_t dpp8[] = {Vop2(0x03, 233)};

  const auto truncated_vop3 = gpu::rdna::Decode(vop3, 1, false);
  const auto truncated_nsa = gpu::rdna::Decode(nsa3, 4, false);
  const auto truncated_dpp = gpu::rdna::Decode(dpp8, 1, false);

  ASSERT_EQ(truncated_vop3.size(), 1u);
  EXPECT_EQ(truncated_vop3[0].enc, gpu::gcn::Enc::kUnknown);
  EXPECT_EQ(truncated_vop3[0].size, 1u);
  ASSERT_EQ(truncated_nsa.size(), 1u);
  EXPECT_EQ(truncated_nsa[0].enc, gpu::gcn::Enc::kUnknown);
  EXPECT_EQ(truncated_nsa[0].size, 4u);
  ASSERT_EQ(truncated_dpp.size(), 1u);
  EXPECT_EQ(truncated_dpp[0].enc, gpu::gcn::Enc::kUnknown);
  EXPECT_EQ(truncated_dpp[0].size, 1u);
}

TEST(RdnaDecode, OrderedEndpgmAndCodeEndTerminateAppropriately) {
  const uint32_t ordered[] = {Sopp(0x1E), Vop2(0x03), Sopp(0x01)};
  const uint32_t code_end[] = {Sopp(0x1F), Vop2(0x03)};

  EXPECT_EQ(gpu::rdna::Decode(ordered, 3).size(), 1u);
  EXPECT_EQ(gpu::rdna::Decode(ordered, 3, false).size(), 3u);
  EXPECT_EQ(gpu::rdna::Decode(code_end, 2, false).size(), 1u);
}

TEST(RdnaDecode, EndpgmSavedTerminatesAndCallTargetsRemainReachable) {
  const uint32_t saved[] = {Sopp(0x1B), Vop2(0x03)};
  EXPECT_EQ(gpu::rdna::Decode(saved, 2).size(), 1u);
  EXPECT_EQ(gpu::rdna::Decode(saved, 2, false).size(), 2u);

  const uint32_t call[] = {
      Sopk(0x16) | 1u,  // target pc2; pc1 is the return address
      Sopp(0x01),
      Vop2(0x03),
      Sopp(0x01),
  };
  const gpu::gcn::Program reachable =
      gpu::rdna::ReachableProgram(gpu::rdna::Decode(call, 4, false));
  ASSERT_EQ(reachable.size(), 4u);
  EXPECT_EQ(reachable[2].pc, 2u);
}

}  // namespace
