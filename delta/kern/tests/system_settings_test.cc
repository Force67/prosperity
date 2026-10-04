#include <gtest/gtest.h>

#include <array>
#include <cstring>

#include "base/option_file.h"
#include "guest/session.h"
#include "kern/ipmi/ipmi.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_event_flag.h"
#include "kern/lv2/sys_mem.h"
#include "kern/lv2/sys_regmgr.h"
#include "kern/lv2/sys_semaphore.h"
#include "kern/process.h"
#include "options/options.h"

namespace {
constexpr u64 kLanguage = 0x12356328ecf5617b;
constexpr u64 kEnterButton = 0x22666251fe7becff;
constexpr u32 kUnavailable = 0x800d0203;
class SystemSettingsTest : public testing::Test {
 protected:
  struct IntRequest {
    u64 key;
    u32 index;
    i32 value;
  };
  void SetUp() override {
    options::Init();
    guest::BeginSession();
    guest::ResetSessionResources();
    base::FindOption("DELTA_SYS_LANG")->SetFromString("en");
  }
  void TearDown() override {
    guest::RequestStop();
    guest::JoinThreads();
    process.GetObjTable().Reset();
    guest::ResetSessionResources();
    base::FindOption("DELTA_SYS_LANG")->Reset();
    guest::BeginSession();
  }
  i32 Get(u64 key) {
    IntRequest request{key, 0, 99};
    u32 result = 99;
    EXPECT_EQ(
        kern::sys_regmgr_call(0x19, 0, &result, &request, sizeof(request)), 0);
    EXPECT_EQ(result, 0u);
    EXPECT_EQ(request.key, key);
    EXPECT_EQ(request.index, 0u);
    return request.value;
  }
  void Set(u64 key, i32 value) {
    IntRequest request{key, 0, value};
    u32 result = 99;
    ASSERT_EQ(
        kern::sys_regmgr_call(0x18, 0, &result, &request, sizeof(request)), 0);
    EXPECT_EQ(result, 0u);
    EXPECT_EQ(request.value, value);
  }
  kern::Process process;
};

TEST_F(SystemSettingsTest, ConsoleDefaults) {
  EXPECT_EQ(Get(kLanguage), 1);
  EXPECT_EQ(Get(kEnterButton), 1);
  EXPECT_EQ(Get(0x596b60e8fda47e4c), 0);
  EXPECT_EQ(Get(0x1bd16a3e9b03ed9a), 1);
  EXPECT_EQ(Get(0x123d6328eff5667b), 0);
  EXPECT_EQ(Get(0x3a73688d92700623), 0);
  EXPECT_EQ(Get(0x03d261027dfff0a4), 0);
}

TEST_F(SystemSettingsTest, VideoControlReadySurvivesRepeatedWaits) {
  const int flag = kern::sys_evf_open("SceAvControlEvf");
  ASSERT_GT(flag, 0);
  for (const u64 pattern : {1ull << 34, 1ull << 40, 1ull << 34}) {
    u64 result = 0;
    u32 timeout = 0;
    ASSERT_EQ(kern::sys_evf_wait(flag, pattern,
                               kern::kEvfAnd | kern::kEvfClearPat, &result,
                               &timeout),
              0);
    EXPECT_EQ(result, (1ull << 34) | (1ull << 40));
  }
  u64 result = 0;
  EXPECT_EQ(kern::sys_evf_trywait(flag, 1, kern::kEvfAnd, &result),
            -static_cast<i32>(kern::SysError::eBUSY));
}

TEST_F(SystemSettingsTest, ForegroundSuspendPermitIsExclusiveAndReusable) {
  const int semaphore = kern::sys_osem_open("SceLncSuspendBlock60000001");
  ASSERT_GT(semaphore, 0);
  EXPECT_EQ(kern::sys_osem_open("SceLncSuspendBlock60000001"), semaphore);
  for (int i = 0; i < 100; ++i) {
    ASSERT_EQ(kern::sys_osem_trywait(semaphore, 1), 0);
    EXPECT_EQ(kern::sys_osem_trywait(semaphore, 1),
              -static_cast<i32>(kern::SysError::eBUSY));
    ASSERT_EQ(kern::sys_osem_post(semaphore, 1), 0);
  }
  EXPECT_EQ(kern::sys_osem_post(semaphore, 1),
            -static_cast<i32>(kern::SysError::eINVAL));
  for (const char* name : {"SceLncSuspendBlock60000002",
                           "SceLncSuspendBlockCount", "SceAppMessaging"}) {
    const int other = kern::sys_osem_open(name);
    ASSERT_GT(other, 0);
    EXPECT_EQ(kern::sys_osem_trywait(other, 1),
              -static_cast<i32>(kern::SysError::eBUSY));
  }
}

TEST_F(SystemSettingsTest, LanguageCodesAndRawIdsDoNotCacheAcrossSessions) {
  for (const auto& [name, id] : {std::pair{"ja", 0},
                                 {"DE", 4},
                                 {"pt-br", 17},
                                 {"en-gb", 18},
                                 {"zh-hant", 10},
                                 {"29", 29},
                                 {"30", 1},
                                 {"2garbage", 1},
                                 {"", 1}}) {
    guest::ResetSessionResources();
    base::FindOption("DELTA_SYS_LANG")->SetFromString(name);
    EXPECT_EQ(Get(kLanguage), id);
    EXPECT_EQ(Get(kEnterButton), id != 0);
  }
}

TEST_F(SystemSettingsTest, SetAndResetUseTypedRegistryBacking) {
  Set(kLanguage, 0);
  EXPECT_EQ(Get(kLanguage), 0);
  EXPECT_EQ(Get(kEnterButton), 0);
  Set(kEnterButton, 1);
  EXPECT_EQ(Get(kEnterButton), 1);
  Set(0x1bd16a3e9b03ed9a, 0);
  EXPECT_EQ(Get(0x1bd16a3e9b03ed9a), 0);
  guest::ResetSessionResources();
  EXPECT_EQ(Get(kLanguage), 1);
  EXPECT_EQ(Get(kEnterButton), 1);
  EXPECT_EQ(Get(0x1bd16a3e9b03ed9a), 1);
}

TEST_F(SystemSettingsTest, UnsupportedKeyAndIndexUseMethodResult) {
  for (IntRequest request : {IntRequest{123, 0, 81}, {kLanguage, 1, 81}}) {
    u32 result[] = {99, 97};
    ASSERT_EQ(kern::sys_regmgr_call(0x19, 0, result, &request, sizeof(request)),
              0);
    EXPECT_EQ(result[0], kUnavailable);
    EXPECT_EQ(result[1], 97u);
    EXPECT_EQ(request.value, 0);
    request.value = 81;
    ASSERT_EQ(kern::sys_regmgr_call(0x18, 0, result, &request, sizeof(request)),
              0);
    EXPECT_EQ(result[0], kUnavailable);
    EXPECT_EQ(request.value, 81);
  }
  EXPECT_EQ(Get(kLanguage), 1);
}

TEST_F(SystemSettingsTest, InvalidIntDescriptorsLeaveOutputsUntouched) {
  IntRequest request{kLanguage, 0, 71};
  u32 result = 91;
  EXPECT_EQ(kern::sys_regmgr_call(0x19, 0, &result, &request, 15),
            -static_cast<i32>(kern::SysError::eINVAL));
  EXPECT_EQ(result, 91u);
  EXPECT_EQ(request.value, 71);
  EXPECT_EQ(kern::sys_regmgr_call(0x19, 0, &result, nullptr, 16),
            -static_cast<i32>(kern::SysError::eFAULT));
  EXPECT_EQ(result, 91u);
  EXPECT_EQ(kern::sys_regmgr_call(0x19, 0, nullptr, &request, 16),
            -static_cast<i32>(kern::SysError::eFAULT));
  EXPECT_EQ(request.value, 71);
}

TEST_F(SystemSettingsTest, BinaryReplyValidatesSizeAndPreservesCanary) {
  struct Binary {
    u64 key = 1, index = 0, size = 17;
    std::array<u8, 18> data;
  } request;
  request.data.fill(83);
  u32 result = 91;
  ASSERT_EQ(kern::sys_regmgr_call(0x1b, 0, &result, &request, 41), 0);
  EXPECT_EQ(result, 0u);
  for (size_t i = 0; i < 17; ++i)
    EXPECT_EQ(request.data[i], 0);
  EXPECT_EQ(request.data[17], 83);
  result = 91;
  request.data.fill(83);
  EXPECT_EQ(kern::sys_regmgr_call(0x1d, 0, &result, &request, 40),
            -static_cast<i32>(kern::SysError::eINVAL));
  EXPECT_EQ(result, 91u);
  EXPECT_EQ(request.data[0], 83);
}

TEST_F(SystemSettingsTest, ShellCoreDaemonPublishesHeaderAndParentalLevel) {
  const int fd = kern::sys_shm_open("/SceShellCoreUtil", 0, 0);
  ASSERT_GE(fd, 0);
  auto* view = kern::sys_mmap(nullptr, 0x4000, 1, 1, fd, 0);
  ASSERT_FALSE(kern::IsErrnoPtr(view));
  EXPECT_EQ(*reinterpret_cast<const u64*>(view), 0x6ed81ede6df17259u);
  EXPECT_EQ(*reinterpret_cast<const i32*>(view + 0x14), 1);
  const int reopened = kern::sys_shm_open("/SceShellCoreUtil", 0, 0);
  ASSERT_GE(reopened, 0);
  auto* other = kern::sys_mmap(nullptr, 0x4000, 1, 1, reopened, 0);
  ASSERT_FALSE(kern::IsErrnoPtr(other));
  ASSERT_NE(view, other);
  EXPECT_EQ(*reinterpret_cast<const i32*>(other + 0x14), 1);
}

TEST_F(SystemSettingsTest,
       LaunchDaemonRepliesWithPlatformSpecificStatusLayout) {
  const char* name = "SceLncService";
  u64 client = reinterpret_cast<u64>(name);
  u32 kid;
  ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, &client, 8), 0);
  for (const auto& [method, size] :
       {std::pair{0x30013u, 12u}, {0x30010u, 32u}}) {
    std::array<u32, 9> status;
    status.fill(83);
    u64 out[] = {reinterpret_cast<u64>(status.data()), size, 1};
    i32 result = -1;
    u32 manager_result = 99;
    struct Request {
      u32 method, num_in, num_out, pad;
      const u64* in;
      const u64* out;
      i32* result;
    } request{method, 0, 1, 0, nullptr, out, &result};
    ASSERT_EQ(kern::ipmi::ManagerCall(800, kid, &manager_result, &request,
                                      sizeof(request)),
              0);
    EXPECT_EQ(manager_result, 0u);
    EXPECT_EQ(result, 0);
    EXPECT_EQ(status[0], 0x60000001u);
    EXPECT_EQ(status[1], 0u);
    EXPECT_EQ(status[2], 4u);
    for (u32 i = 3; i < size / 4; ++i)
      EXPECT_EQ(status[i], 0u);
    EXPECT_EQ(status[size / 4], 83u);
    status.fill(83);
    out[1] = size - 1;
    ASSERT_EQ(kern::ipmi::ManagerCall(800, kid, &manager_result, &request,
                                      sizeof(request)),
              0);
    EXPECT_EQ(result, -static_cast<i32>(kern::SysError::eINVAL));
    for (auto word : status)
      EXPECT_EQ(word, 83u);
  }
}
}  // namespace
