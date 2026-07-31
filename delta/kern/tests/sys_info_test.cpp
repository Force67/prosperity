#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "kern/ps4/hardware_mode.h"
#include "kern/ps4/lv2/sys_info.h"

namespace {

uint32_t ReadSysctlByName(const char *name) {
  int translate_mib[] = {0, 3};
  int mib[4]{};
  size_t mib_size = sizeof(mib);
  EXPECT_EQ(krnl::sys_sysctl(translate_mib, 2, mib, &mib_size, name,
                             std::strlen(name)),
            0);

  uint32_t value = UINT32_MAX;
  size_t value_size = sizeof(value);
  EXPECT_EQ(krnl::sys_sysctl(mib, static_cast<uint32_t>(mib_size / sizeof(int)),
                             &value, &value_size, nullptr, 0),
            0);
  EXPECT_EQ(value_size, sizeof(value));
  return value;
}

class TitleAttributesScope {
public:
  TitleAttributesScope() : saved_(krnl::ps4::titleAttributes()) {}
  ~TitleAttributesScope() { krnl::ps4::setTitleAttributes(saved_); }

private:
  uint32_t saved_;
};

} // namespace

TEST(SysInfo, ReportsPs4PageSize) {
  int mib[] = {6, 7};
  uint32_t page_size = 0;
  size_t result_size = sizeof(page_size);

  EXPECT_EQ(krnl::sys_sysctl(mib, 2, &page_size, &result_size, nullptr, 0), 0);
  EXPECT_EQ(result_size, sizeof(page_size));
  EXPECT_EQ(page_size, 0x4000u);
}

TEST(SysInfo, ReportsConfiguredPs4HardwareMode) {
  const TitleAttributesScope restore_attributes;
  base::InitOptionsFromEnv();
  const bool expect_neo_hardware = krnl::ps4::kNeoMode;
  const auto &profile = krnl::ps4::hardwareModeProfile();

  EXPECT_EQ(profile.mode, expect_neo_hardware ? krnl::ps4::HardwareMode::neo
                                              : krnl::ps4::HardwareMode::base);
  EXPECT_EQ(profile.mainSocId, expect_neo_hardware ? 0x740f30u : 0x710f10u);

  krnl::ps4::setTitleAttributes(0);
  EXPECT_FALSE(krnl::ps4::isNeoMode());
  EXPECT_STREQ(krnl::ps4::gnmDriverModule(), "libSceGnmDriver");
  EXPECT_EQ(ReadSysctlByName("kern.neomode"), 0u);

  krnl::ps4::setTitleAttributes(1u << 23);
  EXPECT_EQ(krnl::ps4::isNeoMode(), expect_neo_hardware);
  EXPECT_STREQ(krnl::ps4::gnmDriverModule(), expect_neo_hardware
                                                 ? "libSceGnmDriverForNeoMode"
                                                 : "libSceGnmDriver");
  EXPECT_EQ(ReadSysctlByName("kern.neomode"), expect_neo_hardware ? 1u : 0u);
}

TEST(SysInfo, ReportsCpuModeFromTitleAttributes) {
  const TitleAttributesScope restore_attributes;
  struct Case {
    uint32_t attributes;
    uint32_t expected;
  };
  constexpr Case cases[] = {
      {0, 0},
      {1u << 15, 0},
      {1u << 16, 5},
      {(1u << 15) | (1u << 16), 2},
  };

  int cpu_mode_mib[] = {1, 14, 42};
  for (const Case &test : cases) {
    krnl::ps4::setTitleAttributes(test.attributes);
    uint32_t direct_cpu_mode = UINT32_MAX;
    size_t cpu_mode_size = sizeof(direct_cpu_mode);
    ASSERT_EQ(krnl::sys_sysctl(cpu_mode_mib, 3, &direct_cpu_mode,
                               &cpu_mode_size, nullptr, 0),
              0);
    EXPECT_EQ(cpu_mode_size, sizeof(direct_cpu_mode));
    EXPECT_EQ(direct_cpu_mode, test.expected);
    EXPECT_EQ(ReadSysctlByName("kern.cpumode"), test.expected);
  }
}
