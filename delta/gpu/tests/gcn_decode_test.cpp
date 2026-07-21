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

}  // namespace
