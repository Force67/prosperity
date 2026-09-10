#include "base/arch.h"
#include <vector>

#include <gtest/gtest.h>

#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps5/rdna/rdna_decode.h"
#include "gpu/ps5/rdna/rdna_resource.h"
#include "gpu/ps5/rdna/rdna_translate.h"

namespace {

constexpr u32 kEndPgm = 0xbf810000;

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

u32 Vop1(u32 op, u32 source = 256) {
  return (0x3fu << 25) | (op << 9) | source;
}

u32 Vop2(u32 op) {
  return (op << 25) | 256u;
}

u32 Vopc(u32 op) {
  return (0x3eu << 25) | (op << 17) | 256u;
}

void AppendVop3(std::vector<u32>& code, u32 op) {
  code.push_back((0x34u << 26) | ((op & 0x1ff) << 17) | ((op >> 9) << 16));
  code.push_back(256u | (256u << 9) | (256u << 18));
}

void AppendVop3Literal(std::vector<u32>& code, u32 op) {
  code.push_back((0x34u << 26) | ((op & 0x1ff) << 17) | ((op >> 9) << 16));
  code.push_back(255u | (256u << 9) | (256u << 18));
  code.push_back(0x00010001);
}

void AppendVop3p(std::vector<u32>& code, u32 op) {
  code.push_back((0x33u << 26) | (op << 16));
  code.push_back(256u | (256u << 9) | (256u << 18));
}

bool HasBuiltin(const std::vector<u32>& spirv, u32 builtin) {
  for (size_t i = 5; i < spirv.size();) {
    const u32 word_count = spirv[i] >> 16;
    const u32 opcode = spirv[i] & 0xFFFF;
    if (!word_count || i + word_count > spirv.size())
      return false;
    // OpDecorate target BuiltIn <builtin>.
    if (opcode == 71 && word_count >= 4 && spirv[i + 2] == 11 &&
        spirv[i + 3] == builtin)
      return true;
    i += word_count;
  }
  return false;
}

bool Recompile(std::vector<u32> code) {
  code.push_back(kEndPgm);
  const u32 user_data[16] = {};
  return gpu::gcn::Recompile(code.data(), nullptr, user_data, user_data).ok;
}

TEST(RdnaScalarReplay, CompareWritesOnlyItsSelectedMaskRegister) {
  // Astro's compare uses VCC. Bits 14:8 are zero but do not name s[0:1]
  // unless SD is set. Inventing that write rejected its later buffer atomic.
  const u32 code[] = {0x7c041ef9, 0x86860080, kEndPgm};
  auto program = gpu::rdna::DecodeShader(code, std::size(code));
  auto writes = gpu::rdna::PossibleScalarWrites(program[0]);
  EXPECT_EQ(writes.range[0].first, 106u);
  EXPECT_EQ(writes.range[0].count, 2u);
  EXPECT_EQ(writes.range[1].count, 0u);
  auto replay = gpu::rdna::PlanScalarReplay(program);
  EXPECT_EQ(replay.LossAt(0, 4, 1), gpu::rdna::ScalarReplayPlan::kCovered);

  // Selecting s[20:21] leaves VCC intact instead.
  program[0].raw[1] = 0x0686946a;
  writes = gpu::rdna::PossibleScalarWrites(program[0]);
  EXPECT_EQ(writes.range[0].first, 20u);
  EXPECT_EQ(writes.range[1].count, 0u);
  replay = gpu::rdna::PlanScalarReplay(program);
  EXPECT_EQ(replay.LossAt(106, 2, 1), gpu::rdna::ScalarReplayPlan::kCovered);
  EXPECT_EQ(replay.LossAt(20, 2, 1), gpu::rdna::ScalarReplayPlan::kUnmodelled);

  // RDNA CMPX writes EXEC, regardless of SDST or the VOP3 destination field.
  program[0].opcode |= 0x10;
  writes = gpu::rdna::PossibleScalarWrites(program[0]);
  EXPECT_EQ(writes.range[0].first, 126u);
  program[0].enc = gpu::gcn::Enc::kVop3;
  program[0].raw[0] = 0xd5120000;
  writes = gpu::rdna::PossibleScalarWrites(program[0]);
  EXPECT_EQ(writes.range[0].first, 126u);
  EXPECT_EQ(writes.range[1].count, 0u);
}

TEST(RdnaSpirv, NoColorExportDoesNotSynthesizeWhiteOutput) {
  const u32 user_data[32] = {};
  const u32 vs[64] = {kEndPgm};
  const u32 no_export[64] = {kEndPgm};
  const u32 null_export[64] = {0xf8000890, 0, kEndPgm};
  const u32 color_export[64] = {
      0x7e0002f2, 0xf800080f, 0, kEndPgm};  // v0=1; exp mrt0 v0,v0,v0,v0
  for (const u32* ps : {no_export, null_export, color_export}) {
    gpu::rdna::NextProgramGeneration();
    const auto result = gpu::rdna::Recompile(vs, ps, user_data, user_data);
    ASSERT_TRUE(result.ok);
    EXPECT_EQ(result.ps_mrt_mask, ps == color_export ? 1 : 0);
  }
}

TEST(GcnSpirv, AcceptsImplementedNeoVectorFamilies) {
  const IsaScope neo(gpu::gcn::IsaMode::kNeo);
  EXPECT_TRUE(Recompile({}));

  std::vector<u32> code;
  code.push_back(Vop1(0x0a));
  code.push_back(Vop1(0x0b));
  for (u32 op = 0x50; op <= 0x65; op++)
    code.push_back(Vop1(op));
  EXPECT_TRUE(Recompile(std::move(code))) << "VOP1";

  code.clear();
  for (u32 op : {0x32, 0x33, 0x34, 0x35, 0x36, 0x39, 0x3a, 0x3b})
    code.push_back(Vop2(op));
  for (u32 op : {0x37, 0x38}) {
    code.push_back(Vop2(op));
    code.push_back(0x00003c00);  // Mandatory FP16 literal.
  }
  EXPECT_TRUE(Recompile(std::move(code))) << "VOP2";

  code.clear();
  for (u32 op : {0x89, 0x8f, 0xa9, 0xc9, 0xe9})
    code.push_back(Vopc(op));
  EXPECT_TRUE(Recompile(std::move(code))) << "VOPC";

  code.clear();
  AppendVop3(code, 0x18a);
  AppendVop3(code, 0x18b);
  for (u32 op = 0x1d0; op <= 0x1e5; op++) {
    if (op != 0x1e2)  // v_sat_pk_u8_i16 is VOP1-only.
      AppendVop3(code, op);
  }
  EXPECT_TRUE(Recompile(std::move(code))) << "VOP3 reflected VOP1";

  code.clear();
  for (u32 op :
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
  for (u32 op = 0; op <= 0x12; op++)
    AppendVop3p(code, op);
  for (u32 op = 0x20; op <= 0x22; op++)
    AppendVop3p(code, op);
  EXPECT_TRUE(Recompile(std::move(code))) << "VOP3P";
}

TEST(GcnSpirv, RejectsUnsupportedNeoForms) {
  const IsaScope neo(gpu::gcn::IsaMode::kNeo);

  EXPECT_FALSE(Recompile({Vop1(0x50, 249), 0}));  // SDWA control dword.

  std::vector<u32> interp;
  AppendVop3(interp, 0x342);  // Requires pixel-stage interpolation state.
  EXPECT_FALSE(Recompile(std::move(interp)));

  std::vector<u32> div_fixup;
  AppendVop3(div_fixup, 0x35f);
  EXPECT_FALSE(Recompile(std::move(div_fixup)));

  EXPECT_FALSE(Recompile({Vop1(0x50, 254)}));  // LDS_DIRECT is not modeled.
}

// A wave64 guest compiler omits the s_barrier between an LDS write and an LDS
// read when the threadgroup is one wave; on a 32-wide host those 64 lanes are
// two subgroups and the read races the write.
class WaveScope {
 public:
  explicit WaveScope(u32 lanes)
      : old_(gpu::gcn::HostSubgroupSize()) {
    gpu::gcn::SetHostSubgroupSize(lanes);
  }
  ~WaveScope() { gpu::gcn::SetHostSubgroupSize(old_); }

 private:
  u32 old_;
};

// ds_write_b32 v0, v0  /  ds_read_b32 v1, v0
constexpr u32 kDsWrite0 = 0xd8340000, kDsWrite1 = 0x00000000;
constexpr u32 kDsRead0 = 0xd8d80000, kDsRead1 = 0x01000000;

gpu::gcn::LdsBarrierPlan PlanBarriers(std::vector<u32> code,
                                      u32 threads) {
  const gpu::gcn::Program p = gpu::gcn::Decode(code.data(),
                                               (u32)code.size(), false);
  const std::vector<u8> reach = gpu::gcn::ComputeReachability(p);
  return gpu::gcn::PlanLdsBarriers(p, reach.data(), threads);
}

bool NoBarriers(const gpu::gcn::LdsBarrierPlan& p) {
  return p.at.empty() && !p.lockstep;
}

TEST(GcnSpirv, Wave64LdsBarriersAreReinsertedOnlyWhereNeeded) {
  const WaveScope split(32);

  // Straight-line write-then-read: one barrier, immediately before the read.
  const std::vector<u32> raw{kDsWrite0, kDsWrite1, kDsRead0, kDsRead1,
                                  kEndPgm};
  const gpu::gcn::LdsBarrierPlan plan = PlanBarriers(raw, 64);
  ASSERT_EQ(plan.at.size(), 1u);
  EXPECT_EQ(plan.at[0], 1u);  // instruction index of the ds_read
  EXPECT_FALSE(plan.lockstep);

  // More than one wave per group: the guest compiler had to emit its own.
  EXPECT_TRUE(NoBarriers(PlanBarriers(raw, 128)));
  EXPECT_TRUE(NoBarriers(PlanBarriers(raw, 256)));

  // Reads only, or writes only: nothing to order.
  EXPECT_TRUE(NoBarriers(PlanBarriers({kDsRead0, kDsRead1, kEndPgm}, 64)));
  EXPECT_TRUE(NoBarriers(PlanBarriers({kDsWrite0, kDsWrite1, kEndPgm}, 64)));

  // A host subgroup at least a wave wide keeps the guest's own guarantee.
  const WaveScope whole(64);
  EXPECT_TRUE(NoBarriers(PlanBarriers(raw, 64)));
}

TEST(GcnSpirv, BranchyWave64LdsSyncsPerDispatchIteration) {
  const WaveScope split(32);
  // write ; if(execz) skip a second write ; read
  const std::vector<u32> code{
      kDsWrite0, kDsWrite1,
      0xbf880002,  // s_cbranch_execz pc+2 -> the read
      kDsWrite0,  kDsWrite1,
      kDsRead0,   kDsRead1,
      kEndPgm,
  };
  const gpu::gcn::LdsBarrierPlan plan = PlanBarriers(code, 64);
  // A branchy shader gets wave-uniform control flow, which puts every
  // invocation in the same block on the same iteration; the barrier can then
  // go inline at the read, after both writes.
  EXPECT_TRUE(plan.lockstep);
  ASSERT_EQ(plan.at.size(), 1u);
  EXPECT_EQ(plan.at[0], 3u);  // the ds_read
}

TEST(GcnSpirv, OnlyTheEntryBlockIsAUniformPointInABranchyShader) {
  std::vector<u32> code{
      kDsWrite0, kDsWrite1,
      0xbf880002,  // s_cbranch_execz
      kDsWrite0,  kDsWrite1,
      kDsRead0,   kDsRead1,
      kEndPgm,
  };
  const gpu::gcn::Program p =
      gpu::gcn::Decode(code.data(), (u32)code.size(), false);
  const std::vector<u8> at = gpu::gcn::UniformPoints(p);
  ASSERT_GE(at.size(), 4u);
  EXPECT_TRUE(at[0]);   // ds_write, entry block
  EXPECT_TRUE(at[1]);   // the branch itself, still the entry block
  EXPECT_FALSE(at[2]);  // conditional block
  EXPECT_FALSE(at[3]);  // reached on differing iterations

  // With no branches at all every point is uniform.
  const std::vector<u32> flat{kDsWrite0, kDsWrite1, kDsRead0, kDsRead1,
                                   kEndPgm};
  const gpu::gcn::Program fp =
      gpu::gcn::Decode(flat.data(), (u32)flat.size(), false);
  for (u8 u : gpu::gcn::UniformPoints(fp))
    EXPECT_TRUE(u);
}

TEST(GcnSpirv, RejectsNeoOnlyEncodingsInBaseMode) {
  const IsaScope base(gpu::gcn::IsaMode::kBase);
  const u32 sop2_pack = (2u << 30) | (0x32u << 23) | (128u << 8) | 128u;

  EXPECT_FALSE(Recompile({sop2_pack}));
  EXPECT_FALSE(Recompile({(0x17du << 23) | (3u << 8) | 248u}));
  EXPECT_FALSE(Recompile({Vop1(0x01, 248)}));  // Neo INV_2PI inline value.

  std::vector<u32> literal;
  AppendVop3(literal, 0x103);
  literal.back() = (literal.back() & ~0x1ffu) | 255u;
  EXPECT_FALSE(Recompile(std::move(literal)));
}

TEST(GcnSpirv, BaseVop3OutputModifiersAreNotIgnored) {
  const IsaScope base(gpu::gcn::IsaMode::kBase);
  std::vector<u32> floating;
  AppendVop3(floating, 0x103);  // v_add_f32
  floating.back() |= 1u << 27;  // OMOD:*2
  EXPECT_TRUE(Recompile(std::move(floating)));

  std::vector<u32> cvt_f16;
  AppendVop3(cvt_f16, 0x18a);
  cvt_f16.back() |= 1u << 27;
  AppendVop3(cvt_f16, 0x18b);
  cvt_f16.back() |= 1u << 27;
  EXPECT_TRUE(Recompile(std::move(cvt_f16)));

  // An integer result IGNORES the output modifier on hardware ("Integer and
  // non-specific instructions ignore output modifiers", Sea Islands ISA), so
  // dropping it is exact. Declining the shader over it was the defect.
  std::vector<u32> integer;
  AppendVop3(integer, 0x169);  // v_mul_lo_u32
  integer.back() |= 1u << 27;
  EXPECT_TRUE(Recompile(std::move(integer)));
}

TEST(GcnSpirv, BaseSeaIslandsCorrectionsAreAcceptedOrRejectedExplicitly) {
  const IsaScope base(gpu::gcn::IsaMode::kBase);

  EXPECT_TRUE(Recompile({Vopc(0x88)}));                    // v_cmp_class_f32
  EXPECT_TRUE(Recompile({(0xbu << 28) | (0x10u << 23)}));  // s_mulk_i32
  EXPECT_TRUE(Recompile({0xbf960000}));   // s_cbranch_cdbgsys to fallthrough
  EXPECT_FALSE(Recompile({0xbf960001}));  // meaningful debug branch unsupported

  std::vector<u32> shift64;
  AppendVop3(shift64, 0x161);
  EXPECT_TRUE(Recompile(std::move(shift64)));

  std::vector<u32> mad64;
  AppendVop3(mad64, 0x176);
  EXPECT_TRUE(Recompile(std::move(mad64)));

  // The legacy multiply differs from the IEEE one only for zero times inf or
  // NaN, so it lowers to the same OpFMul rather than costing the shader.
  EXPECT_TRUE(Recompile({Vop2(0x07)}));

  std::vector<u32> div_scale;
  AppendVop3(div_scale, 0x16d);
  EXPECT_FALSE(Recompile(std::move(div_scale)));

  EXPECT_FALSE(Recompile({
      (0x37u << 26) | (0x0cu << 18),  // recognized but untranslated FLAT
      2u << 24,
  }));
}

TEST(GcnSpirv, PlansScalarLoadedCbufferDescriptor) {
  const IsaScope base(gpu::gcn::IsaMode::kBase);
  const u32 code[] = {
      0xbeeb03ff, 0x00000002,  // Shader footer starts at dword 6.
      0xc08e0104,  // s_load_dwordx4 s[28:31], s[0:1], 0x4
      0xc28c1d08,  // s_buffer_load_dwordx4 s[24:27], s[28:31], 0x8
      0xc2d41d00,  // s_buffer_load_dwordx8 s[40:47], s[28:31], 0x0
      kEndPgm,
      0x5362724f, 0x00726468,  // "OrbShdr"
  };
  const u32 user_data[16] = {};
  const auto recompiled =
      gpu::gcn::Recompile(code, nullptr, user_data, user_data);

  ASSERT_TRUE(recompiled.ok);
  ASSERT_EQ(recompiled.vs_cbufs.size(), 2u);
  EXPECT_EQ(recompiled.vs_cbufs[1].ud_sgpr, 28u);
  EXPECT_FALSE(recompiled.vs_cbufs[1].pointer);
  EXPECT_EQ(recompiled.vs_cbufs[1].num_dwords, 12u);
}

TEST(GcnSpirv, SeedsPixelPositionFromFragCoord) {
  const IsaScope base(gpu::gcn::IsaMode::kBase);
  const u32 vs[] = {kEndPgm};
  const u32 ps[] = {
      0x7e080f02,           // v_cvt_u32_f32 v4, v2
      0x7e0a0f03,           // v_cvt_u32_f32 v5, v3
      0xf800000f, 0x07060504,  // exp mrt0 v4, v5, v6, v7
      kEndPgm,
  };
  const u32 user_data[16] = {};
  // PERSP_CENTER consumes v0:v1, placing POS_X/Y at v2:v3.
  const u32 ps_input_ena = (1u << 1) | (1u << 8) | (1u << 9);

  const auto recompiled =
      gpu::gcn::Recompile(vs, ps, user_data, user_data, ps_input_ena);

  ASSERT_TRUE(recompiled.ok);
  EXPECT_TRUE(HasBuiltin(recompiled.fs_spirv, 15));  // FragCoord
}

}  // namespace
