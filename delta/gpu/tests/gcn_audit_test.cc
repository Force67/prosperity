#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "gpu/ps4/gcn/gcn_audit.h"

namespace {

std::string ReportString() {
  std::FILE* f = std::tmpfile();
  gpu::gcn::WriteAuditReport(f);
  std::fseek(f, 0, SEEK_SET);
  std::string s;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
    s.append(buf, n);
  std::fclose(f);
  return s;
}

// One process-wide fixture: the audit latches its env gates on first use.
class GcnAuditTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { setenv("DELTA_GPU_SHAUDIT", "1", 1); }
};

TEST_F(GcnAuditTest, ClassifiesSilentAndUnsupportedAndDeduplicates) {
  ASSERT_TRUE(gpu::gcn::ShaderDebugEnabled());
  const uint32_t code[] = {
      0xbe800301,              // s_mov_b32 s0, s1
      0x7e020280,              // v_mov_b32 v1, 0
      0xd2820000, 0x040a0300,  // v_mad_f32 v0, v0, v1, v2
      0xbf810000,              // s_endpgm
  };
  const gpu::gcn::Program program = gpu::gcn::Decode(code, 5);
  ASSERT_EQ(program.size(), 4u);

  for (int round = 0; round < 2; round++) {  // second round must deduplicate
    gpu::gcn::AuditBegin("vs", code, program);
    gpu::gcn::AuditInstBegin(0, 0);
    gpu::gcn::AuditInstEnd(0, 3);  // translated: emitted 3 words
    gpu::gcn::AuditInstBegin(1, 1);
    gpu::gcn::AuditInstEnd(1, 0);  // silent: 0 words, no warning
    gpu::gcn::AuditInstBegin(2, 2);
    gpu::gcn::AuditNote("vop3", 0x141);  // warned unsupported
    gpu::gcn::AuditInstEnd(2, 0);
    gpu::gcn::AuditEnd(nullptr);
  }

  const std::string report = ReportString();
  EXPECT_NE(report.find("unique shaders: 1 (vs=1)"), std::string::npos)
      << report;
  EXPECT_NE(report.find("vop3 op=0x141 (v_mad_f32)"), std::string::npos)
      << report;
  // The silent drop is reported under its mnemonic; the translated and the
  // warned instructions are not.
  EXPECT_NE(report.find("v_mov_b32"), std::string::npos) << report;
  EXPECT_EQ(report.find("s_mov_b32"), std::string::npos) << report;
}

TEST_F(GcnAuditTest, DeclinedShaderIsListedWithReason) {
  const uint32_t code[] = {
      0xbf8c0070,  // s_waitcnt
      0xbf810000,  // s_endpgm
  };
  const gpu::gcn::Program program = gpu::gcn::Decode(code, 2);
  gpu::gcn::AuditBegin("cs", code, program);
  gpu::gcn::AuditDecline("cs translation rejected");
  gpu::gcn::AuditEnd(nullptr);

  const std::string report = ReportString();
  EXPECT_NE(report.find("declined: 1"), std::string::npos) << report;
  EXPECT_NE(report.find("cs translation rejected"), std::string::npos)
      << report;
}

TEST_F(GcnAuditTest, ExpectedNoOpsAreNotSilent) {
  const uint32_t code[] = {
      0xbf8c0070,  // s_waitcnt: expected to emit nothing
      0xbf810000,  // s_endpgm
  };
  const gpu::gcn::Program program = gpu::gcn::Decode(code, 2);
  gpu::gcn::AuditBegin("ps", code, program);
  gpu::gcn::AuditInstBegin(0, 0);
  gpu::gcn::AuditInstEnd(0, 0);
  gpu::gcn::AuditEnd(nullptr);

  const std::string report = ReportString();
  EXPECT_EQ(report.find("s_waitcnt"), std::string::npos) << report;
}

}  // namespace
