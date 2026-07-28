#include <cstdint>

#include <gtest/gtest.h>

#include "gpu/ps4/gcn/gcn_decode.h"
#include "gpu/ps4/gcn/gcn_disasm.h"

namespace {

gpu::gcn::Inst DecodeOne(const uint32_t* code, uint32_t dwords) {
  const gpu::gcn::Program program = gpu::gcn::Decode(code, dwords);
  EXPECT_FALSE(program.empty());
  return program.empty() ? gpu::gcn::Inst{} : program[0];
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
