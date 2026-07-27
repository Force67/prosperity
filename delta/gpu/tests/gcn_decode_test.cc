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
