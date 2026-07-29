#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "gpu/ps4/gcn/gcn_decode.h"
#include "gpu/ps4/gcn/gcn_translate.h"

namespace {

constexpr uint32_t kEndPgm = 0xbf810000;

class IsaScope {
 public:
  explicit IsaScope(gpu::gcn::IsaMode mode)
      : old_mode_(gpu::gcn::DefaultIsaMode()) {
    gpu::gcn::SetDefaultIsaMode(mode);
  }
  ~IsaScope() { gpu::gcn::SetDefaultIsaMode(old_mode_); }

 private:
  gpu::gcn::IsaMode old_mode_;
};

uint32_t Vop1(uint32_t op, uint32_t source = 256) {
  return (0x3fu << 25) | (op << 9) | source;
}

uint32_t Vop2(uint32_t op) {
  return (op << 25) | 256u;
}

uint32_t Vopc(uint32_t op) {
  return (0x3eu << 25) | (op << 17) | 256u;
}

void AppendVop3(std::vector<uint32_t>& code, uint32_t op) {
  code.push_back((0x34u << 26) | ((op & 0x1ff) << 17) | ((op >> 9) << 16));
  code.push_back(256u | (256u << 9) | (256u << 18));
}

void AppendVop3Literal(std::vector<uint32_t>& code, uint32_t op) {
  code.push_back((0x34u << 26) | ((op & 0x1ff) << 17) | ((op >> 9) << 16));
  code.push_back(255u | (256u << 9) | (256u << 18));
  code.push_back(0x00010001);
}

void AppendVop3p(std::vector<uint32_t>& code, uint32_t op) {
  code.push_back((0x33u << 26) | (op << 16));
  code.push_back(256u | (256u << 9) | (256u << 18));
}

bool Recompile(std::vector<uint32_t> code) {
  code.push_back(kEndPgm);
  const uint32_t user_data[16] = {};
  return gpu::gcn::Recompile(code.data(), nullptr, user_data, user_data).ok;
}

TEST(GcnSpirv, AcceptsImplementedNeoVectorFamilies) {
  const IsaScope neo(gpu::gcn::IsaMode::kNeo);
  EXPECT_TRUE(Recompile({}));

  std::vector<uint32_t> code;
  code.push_back(Vop1(0x0a));
  code.push_back(Vop1(0x0b));
  for (uint32_t op = 0x50; op <= 0x65; op++)
    code.push_back(Vop1(op));
  EXPECT_TRUE(Recompile(std::move(code))) << "VOP1";

  code.clear();
  for (uint32_t op : {0x32, 0x33, 0x34, 0x35, 0x36, 0x39, 0x3a, 0x3b})
    code.push_back(Vop2(op));
  for (uint32_t op : {0x37, 0x38}) {
    code.push_back(Vop2(op));
    code.push_back(0x00003c00);  // Mandatory FP16 literal.
  }
  EXPECT_TRUE(Recompile(std::move(code))) << "VOP2";

  code.clear();
  for (uint32_t op : {0x89, 0x8f, 0xa9, 0xc9, 0xe9})
    code.push_back(Vopc(op));
  EXPECT_TRUE(Recompile(std::move(code))) << "VOPC";

  code.clear();
  AppendVop3(code, 0x18a);
  AppendVop3(code, 0x18b);
  for (uint32_t op = 0x1d0; op <= 0x1e5; op++) {
    if (op != 0x1e2)  // v_sat_pk_u8_i16 is VOP1-only.
      AppendVop3(code, op);
  }
  EXPECT_TRUE(Recompile(std::move(code))) << "VOP3 reflected VOP1";

  code.clear();
  for (uint32_t op :
       {0x132, 0x133, 0x134, 0x135, 0x136, 0x139, 0x13a, 0x13b, 0x303, 0x304,
        0x305, 0x307, 0x308, 0x309, 0x30a, 0x30b, 0x30c, 0x30d, 0x30e, 0x311,
        0x312, 0x313, 0x314, 0x340, 0x341, 0x344, 0x345, 0x346, 0x347, 0x34b,
        0x351, 0x352, 0x353, 0x354, 0x355, 0x356, 0x357, 0x358, 0x359, 0x35e,
        0x36d, 0x36f, 0x371, 0x372, 0x373, 0x375}) {
    AppendVop3(code, op);
  }
  AppendVop3Literal(code, 0x303);
  AppendVop3(code, 0x341);
  code.back() |= 1u << 27;  // OMOD:*2
  AppendVop3(code, 0x373);
  code[code.size() - 2] |= 1u << 11;  // Saturating U32 clamp.
  AppendVop3(code, 0x375);
  code[code.size() - 2] |= 1u << 11;  // Saturating I32 clamp.
  EXPECT_TRUE(Recompile(std::move(code))) << "VOP3";

  code.clear();
  for (uint32_t op = 0; op <= 0x12; op++)
    AppendVop3p(code, op);
  for (uint32_t op = 0x20; op <= 0x22; op++)
    AppendVop3p(code, op);
  EXPECT_TRUE(Recompile(std::move(code))) << "VOP3P";
}

TEST(GcnSpirv, RejectsUnsupportedNeoForms) {
  const IsaScope neo(gpu::gcn::IsaMode::kNeo);

  EXPECT_FALSE(Recompile({Vop1(0x50, 249), 0}));  // SDWA control dword.

  std::vector<uint32_t> interp;
  AppendVop3(interp, 0x342);  // Requires pixel-stage interpolation state.
  EXPECT_FALSE(Recompile(std::move(interp)));

  std::vector<uint32_t> div_fixup;
  AppendVop3(div_fixup, 0x35f);
  EXPECT_FALSE(Recompile(std::move(div_fixup)));

  EXPECT_FALSE(Recompile({Vop1(0x50, 254)}));  // LDS_DIRECT is not modeled.
}

TEST(GcnSpirv, RejectsNeoOnlyEncodingsInBaseMode) {
  const IsaScope base(gpu::gcn::IsaMode::kBase);
  const uint32_t sop2_pack = (2u << 30) | (0x32u << 23) | (128u << 8) | 128u;

  EXPECT_FALSE(Recompile({sop2_pack}));
  EXPECT_FALSE(Recompile({(0x17du << 23) | (3u << 8) | 248u}));
  EXPECT_FALSE(Recompile({Vop1(0x01, 248)}));  // Neo INV_2PI inline value.

  std::vector<uint32_t> literal;
  AppendVop3(literal, 0x103);
  literal.back() = (literal.back() & ~0x1ffu) | 255u;
  EXPECT_FALSE(Recompile(std::move(literal)));
}

TEST(GcnSpirv, BaseVop3OutputModifiersAreNotIgnored) {
  const IsaScope base(gpu::gcn::IsaMode::kBase);
  std::vector<uint32_t> floating;
  AppendVop3(floating, 0x103);  // v_add_f32
  floating.back() |= 1u << 27;  // OMOD:*2
  EXPECT_TRUE(Recompile(std::move(floating)));

  std::vector<uint32_t> cvt_f16;
  AppendVop3(cvt_f16, 0x18a);
  cvt_f16.back() |= 1u << 27;
  AppendVop3(cvt_f16, 0x18b);
  cvt_f16.back() |= 1u << 27;
  EXPECT_TRUE(Recompile(std::move(cvt_f16)));

  std::vector<uint32_t> integer;
  AppendVop3(integer, 0x169);  // v_mul_lo_u32
  integer.back() |= 1u << 27;
  EXPECT_FALSE(Recompile(std::move(integer)));
}

}  // namespace
