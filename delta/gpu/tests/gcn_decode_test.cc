#include <cstdint>

#include <gtest/gtest.h>

#include "gpu/ps4/gcn/gcn_decode.h"

namespace {

TEST(GcnDecode, SmrdSoffsetLiteralConsumesTrailingDword) {
  const uint32_t code[] = {
      0xc0c216ff,  // s_load_dwordx8 s[4:11], s[22:23], 0x1c14
      0x00001c14,
      0xbf810000,  // s_endpgm
  };

  const gpu::gcn::Program program = gpu::gcn::Decode(code, 3);

  ASSERT_EQ(program.size(), 2u);
  EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kSmrd);
  EXPECT_EQ(program[0].pc, 0u);
  EXPECT_EQ(program[0].size, 2u);
  EXPECT_TRUE(program[0].has_literal);
  EXPECT_EQ(program[0].literal, 0x1c14u);
  EXPECT_EQ(program[1].enc, gpu::gcn::Enc::kSopp);
  EXPECT_EQ(program[1].pc, 2u);
}

TEST(GcnDecode, SmrdImmediateOffsetDoesNotConsumeTrailingDword) {
  const uint32_t code[] = {
      0xc0c217ff,  // s_load_dwordx8 s[4:11], s[22:23], 0xff
      0xbf810000,  // s_endpgm
  };

  const gpu::gcn::Program program = gpu::gcn::Decode(code, 2);

  ASSERT_EQ(program.size(), 2u);
  EXPECT_EQ(program[0].size, 1u);
  EXPECT_FALSE(program[0].has_literal);
  EXPECT_EQ(program[1].pc, 1u);
}

TEST(GcnDecode, SopkSetregImmediateConsumesTrailingDword) {
  const uint32_t code[] = {
      (0xbu << 28) | (0x15u << 23) | 0x1234u,
      0x89abcdef,
      0xbf810000,
  };

  const gpu::gcn::Program program = gpu::gcn::Decode(code, 3);

  ASSERT_EQ(program.size(), 2u);
  EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kSopk);
  EXPECT_EQ(program[0].opcode, 0x15u);
  EXPECT_EQ(program[0].size, 2u);
  EXPECT_EQ(program[0].literal, 0x89abcdefu);
  EXPECT_EQ(program[1].pc, 2u);
}

TEST(GcnDecode, NeoVop3UsesSplitTenthOpcodeBit) {
  constexpr uint32_t op = 0x36d;  // v_add3_u32
  const uint32_t code[] = {
      (0x34u << 26) | ((op & 0x1ff) << 17) | ((op >> 9) << 16),
      0,
  };

  const gpu::gcn::Program base =
      gpu::gcn::Decode(code, 2, false, gpu::gcn::IsaMode::kBase);
  const gpu::gcn::Program neo =
      gpu::gcn::Decode(code, 2, false, gpu::gcn::IsaMode::kNeo);

  ASSERT_EQ(base.size(), 1u);
  ASSERT_EQ(neo.size(), 1u);
  EXPECT_EQ(base[0].opcode, op & 0x1ff);
  EXPECT_EQ(neo[0].opcode, op);
  EXPECT_EQ(neo[0].isa, gpu::gcn::IsaMode::kNeo);
}

TEST(GcnDecode, NeoVop3pIsASeparateEncoding) {
  constexpr uint32_t op = 0x20;  // v_mad_mix_f32
  const uint32_t code[] = {(0x33u << 26) | (op << 16), 0};

  const gpu::gcn::Program program =
      gpu::gcn::Decode(code, 2, false, gpu::gcn::IsaMode::kNeo);

  ASSERT_EQ(program.size(), 1u);
  EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kVop3p);
  EXPECT_EQ(program[0].opcode, op);
  EXPECT_EQ(program[0].size, 2u);
}

TEST(GcnDecode, NeoVop3AndVop3pConsumeTrailingLiterals) {
  const uint32_t code[] = {
      (0x34u << 26) | ((0x303u & 0x1ff) << 17) | ((0x303u >> 9) << 16),
      255u,
      0x12345678,
      (0x33u << 26) | (1u << 16),
      255u,
      0x89abcdef,
      0xbf810000,
  };

  const gpu::gcn::Program program =
      gpu::gcn::Decode(code, 7, true, gpu::gcn::IsaMode::kNeo);

  ASSERT_EQ(program.size(), 3u);
  EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kVop3);
  EXPECT_TRUE(program[0].has_literal);
  EXPECT_EQ(program[0].literal, 0x12345678u);
  EXPECT_EQ(program[0].size, 3u);
  EXPECT_EQ(program[1].enc, gpu::gcn::Enc::kVop3p);
  EXPECT_TRUE(program[1].has_literal);
  EXPECT_EQ(program[1].literal, 0x89abcdefu);
  EXPECT_EQ(program[1].pc, 3u);
  EXPECT_EQ(program[1].size, 3u);
  EXPECT_EQ(program[2].pc, 6u);
}

TEST(GcnDecode, NeoFp16MadVop2ConsumesMandatoryLiteral) {
  const uint32_t code[] = {
      (0x37u << 25) | 256u,
      0x00003c00,
      0xbf810000,
  };

  const gpu::gcn::Program program =
      gpu::gcn::Decode(code, 3, true, gpu::gcn::IsaMode::kNeo);

  ASSERT_EQ(program.size(), 2u);
  EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kVop2);
  EXPECT_EQ(program[0].opcode, 0x37u);
  EXPECT_TRUE(program[0].has_literal);
  EXPECT_EQ(program[0].literal, 0x00003c00u);
  EXPECT_EQ(program[0].size, 2u);
  EXPECT_EQ(program[1].pc, 2u);
}

TEST(GcnDecode, TruncatedMultiwordFormsBecomeUnknown) {
  const uint32_t vop3[] = {(0x34u << 26) | (0x303u & 0x1ff) << 17 |
                           (0x303u >> 9) << 16};
  const uint32_t sdwa[] = {(0x3fu << 25) | (0x50u << 9) | 249u};
  const uint32_t literal[] = {(0x37u << 25) | 256u};

  for (const auto& test :
       {gpu::gcn::Decode(vop3, 1, true, gpu::gcn::IsaMode::kNeo),
        gpu::gcn::Decode(sdwa, 1, true, gpu::gcn::IsaMode::kNeo),
        gpu::gcn::Decode(literal, 1, true, gpu::gcn::IsaMode::kNeo)}) {
    ASSERT_EQ(test.size(), 1u);
    EXPECT_EQ(test[0].enc, gpu::gcn::Enc::kUnknown);
  }
}

TEST(GcnDecode, NeoSdwaAndDppConsumeControlDwords) {
  const uint32_t code[] = {
      (0x3fu << 25) | (1u << 9) | 249u,  // v_mov_b32 v0, sdwa
      0x01234567,
      3u << 25 | 250u,  // v_add_f32 v0, dpp, v0
      0x89abcdef,
      0xbf810000,
  };

  const gpu::gcn::Program program =
      gpu::gcn::Decode(code, 5, true, gpu::gcn::IsaMode::kNeo);

  ASSERT_EQ(program.size(), 3u);
  EXPECT_EQ(program[0].extension, gpu::gcn::InstExtension::kSdwa);
  EXPECT_EQ(program[0].raw[1], 0x01234567u);
  EXPECT_EQ(program[0].size, 2u);
  EXPECT_EQ(program[1].extension, gpu::gcn::InstExtension::kDpp);
  EXPECT_EQ(program[1].raw[1], 0x89abcdefu);
  EXPECT_EQ(program[1].pc, 2u);
  EXPECT_EQ(program[2].pc, 4u);
}

TEST(GcnDecode, NeoMtbufReadsFourthOpcodeBitFromSecondWord) {
  const uint32_t code[] = {(0x3au << 26), 1u << 21};

  const gpu::gcn::Program program =
      gpu::gcn::Decode(code, 2, false, gpu::gcn::IsaMode::kNeo);

  ASSERT_EQ(program.size(), 1u);
  EXPECT_EQ(program[0].enc, gpu::gcn::Enc::kMtbuf);
  EXPECT_EQ(program[0].opcode, 8u);
}

TEST(GcnDecode, ReachabilityExcludesUnconditionalBranchFallthrough) {
  const uint32_t code[] = {
      0xbf820003,  // s_branch pc+4
      0xffffffff, 0xffffffff,
      0xbf810000,  // s_endpgm in dead fallthrough
      0xbe800380,  // s_mov_b32 s0, 0
      0xbf810000,  // reachable s_endpgm
  };

  const gpu::gcn::Program program =
      gpu::gcn::Decode(code, 6, /*stop_at_endpgm=*/false);
  const std::vector<uint8_t> reachable = gpu::gcn::ComputeReachability(program);

  ASSERT_EQ(reachable.size(), 6u);
  EXPECT_EQ(reachable[0], 1u);
  EXPECT_EQ(reachable[1], 0u);
  EXPECT_EQ(reachable[2], 0u);
  EXPECT_EQ(reachable[3], 0u);
  EXPECT_EQ(reachable[4], 1u);
  EXPECT_EQ(reachable[5], 1u);
}

}  // namespace
