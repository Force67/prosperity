#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "base/arch.h"

#include <gtest/gtest.h>

#include "kern/lv2/sys_info.h"
#include "kern/process.h"
#include "kern/ps4/hardware_mode.h"

namespace {

u32 ReadSysctlByName(const char* name) {
  int translate_mib[] = {0, 3};
  int mib[4]{};
  size_t mib_size = sizeof(mib);
  EXPECT_EQ(kern::sys_sysctl(translate_mib, 2, mib, &mib_size, name,
                             std::strlen(name)),
            0);

  u32 value = UINT32_MAX;
  size_t value_size = sizeof(value);
  EXPECT_EQ(kern::sys_sysctl(mib, static_cast<u32>(mib_size / sizeof(int)),
                             &value, &value_size, nullptr, 0),
            0);
  EXPECT_EQ(value_size, sizeof(value));
  return value;
}

class TitleAttributesScope {
 public:
  TitleAttributesScope() : saved_(kern::ps4::TitleAttributes()) {}
  ~TitleAttributesScope() { kern::ps4::SetTitleAttributes(saved_); }

 private:
  u32 saved_;
};

}  // namespace

TEST(SysInfo, ReportsPs4PageSize) {
  int mib[] = {6, 7};
  u32 page_size = 0;
  size_t result_size = sizeof(page_size);

  EXPECT_EQ(kern::sys_sysctl(mib, 2, &page_size, &result_size, nullptr, 0), 0);
  EXPECT_EQ(result_size, sizeof(page_size));
  EXPECT_EQ(page_size, 0x4000u);
}

TEST(SysInfo, ReportsFirmwareSdkSeparatelyFromTitleSdk) {
  static kern::Process process;
  process.SetPlatform(kern::Process::Platform::kPs5);
  process.SetSdkVersion(0x09000000);
  struct Restore {
    kern::Process& process;
    ~Restore() {
      process.GetModuleList().clear();
      process.SetPlatform(kern::Process::Platform::kPs4);
      process.SetSdkVersion(0);
    }
  } restore{process};
  u32 param[8]{};
  param[4] = 0x13590001;
  param[5] = 0x13600007;
  kern::ModulePtr kernel(new kern::Module(&process));
  kernel->GetInfo().name = "libkernel";
  kernel->GetInfo().module_param = reinterpret_cast<u8*>(param);
  kernel->GetInfo().module_param_size = sizeof(param);
  process.GetModuleList().push_back(base::move(kernel));

  EXPECT_EQ(ReadSysctlByName("kern.sdk_version"), 0x13600007u);
  int compiled_mib[] = {1, 14, 36};
  u32 compiled = 0;
  size_t size = sizeof(compiled);
  ASSERT_EQ(kern::sys_sysctl(compiled_mib, 3, &compiled, &size, nullptr, 0), 0);
  EXPECT_EQ(compiled, 0x09000000u);
}

TEST(SysInfo, ReportsConfiguredPs4HardwareMode) {
  const TitleAttributesScope restore_attributes;
  base::InitOptionsFromEnv();
  const bool expect_neo_hardware = kern::ps4::kNeoMode;
  const auto& profile = kern::ps4::GetHardwareModeProfile();

  EXPECT_EQ(profile.mode, expect_neo_hardware ? kern::ps4::HardwareMode::kNeo
                                              : kern::ps4::HardwareMode::kBase);
  EXPECT_EQ(profile.main_soc_id, expect_neo_hardware ? 0x740f30u : 0x710f10u);

  kern::ps4::SetTitleAttributes(0);
  EXPECT_FALSE(kern::ps4::IsNeoMode());
  EXPECT_STREQ(kern::ps4::GnmDriverModule(), "libSceGnmDriver");
  EXPECT_EQ(ReadSysctlByName("kern.neomode"), 0u);

  kern::ps4::SetTitleAttributes(1u << 23);
  EXPECT_EQ(kern::ps4::IsNeoMode(), expect_neo_hardware);
  EXPECT_STREQ(kern::ps4::GnmDriverModule(), expect_neo_hardware
                                                 ? "libSceGnmDriverForNeoMode"
                                                 : "libSceGnmDriver");
  EXPECT_EQ(ReadSysctlByName("kern.neomode"), expect_neo_hardware ? 1u : 0u);
}

TEST(SysInfo, ReportsCpuModeFromTitleAttributes) {
  const TitleAttributesScope restore_attributes;
  struct Case {
    u32 attributes;
    u32 expected;
  };
  constexpr Case kCases[] = {
      {0, 0},
      {1u << 15, 0},
      {1u << 16, 5},
      {(1u << 15) | (1u << 16), 2},
  };

  int cpu_mode_mib[] = {1, 14, 42};
  for (const Case& test : kCases) {
    kern::ps4::SetTitleAttributes(test.attributes);
    u32 direct_cpu_mode = UINT32_MAX;
    size_t cpu_mode_size = sizeof(direct_cpu_mode);
    ASSERT_EQ(kern::sys_sysctl(cpu_mode_mib, 3, &direct_cpu_mode,
                               &cpu_mode_size, nullptr, 0),
              0);
    EXPECT_EQ(cpu_mode_size, sizeof(direct_cpu_mode));
    EXPECT_EQ(direct_cpu_mode, test.expected);
    EXPECT_EQ(ReadSysctlByName("kern.cpumode"), test.expected);
  }
}
