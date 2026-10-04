#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "base/threading/thread.h"
#include "base/time/time.h"
#include "guest/session.h"
#include "kern/ipmi/common_dialog.h"
#include "kern/ipmi/ipmi.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_mem.h"
#include "options/options.h"

namespace {
constexpr size_t kPayload = 0x250f0, kReply = 0x400;
constexpr char kName[] = "/00010001SaveDataDialog";
class CommonDialogTest : public testing::Test {
 protected:
  void SetUp() override {
    options::Init();
    guest::RequestStop();
    guest::JoinThreads();
    guest::ResetSessionResources();
    guest::BeginSession();
    kid = Client();
    Create(kName);
  }
  void TearDown() override {
    guest::RequestStop();
    guest::JoinThreads();
    guest::ResetSessionResources();
    guest::BeginSession();
  }
  u32 Client() {
    const char* service = "SceLncService";
    u64 input = reinterpret_cast<u64>(service);
    u32 id = 0;
    EXPECT_EQ(kern::ipmi::ManagerCall(2, 0, &id, &input, sizeof(input)), 0);
    return id;
  }
  void Create(const char* name, size_t size = kReply + kPayload + 16) {
    ASSERT_EQ(kern::ShmCreate(name, size), 0);
    const u32 header[] = {1, 1, 0x400, 1, kPayload, kPayload, 0, 0};
    ASSERT_EQ(kern::ShmWrite(name, 0, header, sizeof(header)), 0);
    request.assign(kPayload, 0);
    const u32 mode = 5;
    std::memcpy(request.data() + 0x38, &mode, 4);
    ASSERT_EQ(kern::ShmWrite(name, 0x400, request.data(),
                             std::min(request.size(), size - 0x400)),
              0);
  }
  i32 Invoke(u32 method,
             const void* data = nullptr,
             u64 size = 0,
             u32 client = 0,
             bool output = false) {
    const u64 in[] = {reinterpret_cast<u64>(data), size};
    u32 sentinel = 0x37533753;
    const u64 out[] = {reinterpret_cast<u64>(&sentinel), sizeof(sentinel), 1};
    i32 result = 73;
    struct Request {
      u32 method, num_in, num_out, pad;
      const u64* in;
      const u64* out;
      i32* result;
    } request{method, data ? 1u : 0u, output ? 1u : 0u, 0, in, out, &result};
    u32 manager = 73;
    EXPECT_EQ(kern::ipmi::ManagerCall(800, client ? client : kid, &manager,
                                      &request, sizeof(request)),
              0);
    EXPECT_EQ(manager, 0u);
    EXPECT_EQ(sentinel, 0x37533753u);
    return result;
  }
  i32 Register(const char* name = kName, u32 client = 0) {
    return Invoke(0x30044, name, std::strlen(name) + 1, client);
  }
  u32 State(const char* name = kName) {
    u32 state = ~0u;
    EXPECT_EQ(kern::ShmRead(name, 0x18, &state, sizeof(state)), 0);
    return state;
  }
  void Close(const char* name = kName) {
    const u8 close = 1;
    ASSERT_EQ(kern::ShmWrite(name, 0x1c, &close, 1), 0);
  }
  i32 LaunchProcess(u32& token,
                    u32 client = 0,
                    const char* title = "NPXS22010") {
    const u32 user = 0;
    const u32 param[6] = {24, ~0u, 0, 1, 0, 0};
    const u64 inputs[] = {reinterpret_cast<u64>(&user),
                          4,
                          reinterpret_cast<u64>(title),
                          10,
                          0,
                          0,
                          reinterpret_cast<u64>(param),
                          sizeof(param)};
    struct Request {
      u32 method, timeout;
      u64 reserved;
      u32 count, pad;
      const u64* inputs;
      u32* token;
      u32 flags, pad1;
    } request{0x30040, ~0u, 0, 4, 0, inputs, &token, 1, 0};
    u32 result = 73;
    EXPECT_EQ(kern::ipmi::ManagerCall(577, client ? client : kid, &result,
                                      &request, sizeof(request)),
              0);
    return static_cast<i32>(result);
  }
  i32 PollProcess(u32 token,
                  i32& result,
                  u32 client = 0,
                  u32 method = 0x30041) {
    struct Request {
      u32 method, token;
      i32* result;
      u32 count, pad;
      const u64* outputs;
      u32 flags, pad1;
    } request{method, token, &result, 0, 0, nullptr, 1, 0};
    u32 manager = 73;
    EXPECT_EQ(kern::ipmi::ManagerCall(579, client ? client : kid, &manager,
                                      &request, sizeof(request)),
              0);
    return static_cast<i32>(manager);
  }
  i32 KillProcess(u32& token, u32 app = 0x60000002) {
    const u64 inputs[] = {reinterpret_cast<u64>(&app), 4};
    struct Request {
      u32 method, timeout;
      u64 reserved;
      u32 count, pad;
      const u64* inputs;
      u32* token;
      u32 flags, pad1;
    } request{0x30042, ~0u, 0, 1, 0, inputs, &token, 1, 0};
    u32 result = 73;
    EXPECT_EQ(
        kern::ipmi::ManagerCall(577, kid, &result, &request, sizeof(request)),
        0);
    return static_cast<i32>(result);
  }
  void CreateWork() {
    ASSERT_EQ(kern::ShmCreate("/60000001CdlgShWk", 4096), 0);
  }
  u32 kid;
  std::vector<u8> request;
};

TEST_F(CommonDialogTest, AsyncLaunchResultsAreOwnedConsumedAndDriveHeartbeat) {
  CreateWork();
  u32 token = 0x37533753;
  ASSERT_EQ(LaunchProcess(token), 0);
  EXPECT_NE(token, 0x37533753u);
  EXPECT_NE(token, 0u);
  EXPECT_EQ(
      kern::ipmi::CommonDialogProcessStatus("SceLncUtilAppStatus60000002"), 1u);
  const auto deadline = base::TimeTicks::Now() + base::Seconds(2);
  u64 heartbeat = 0;
  while (!heartbeat && base::TimeTicks::Now() < deadline) {
    ASSERT_EQ(kern::ShmRead("/60000001CdlgShWk", 0x18, &heartbeat, 8), 0);
    base::SleepForMilliseconds(1);
  }
  EXPECT_GT(heartbeat, 0u);
  i32 result = 0x37533753;
  EXPECT_EQ(PollProcess(token, result, Client()), -kern::SysError::eNOENT);
  EXPECT_EQ(PollProcess(token, result, 0, 0x30043), -kern::SysError::eNOENT);
  EXPECT_EQ(result, 0x37533753);
  ASSERT_EQ(PollProcess(token, result), 0);
  EXPECT_EQ(result, 0x60000002);
  result = 0x37533753;
  EXPECT_EQ(PollProcess(token, result), -kern::SysError::eNOENT);
  EXPECT_EQ(result, 0x37533753);
  u32 manager;
  ASSERT_EQ(kern::ipmi::ManagerCall(3, kid, &manager, nullptr, 0), 0);
  EXPECT_EQ(
      kern::ipmi::CommonDialogProcessStatus("SceLncUtilAppStatus60000002"), 4u);
}

TEST_F(CommonDialogTest, AsyncKillStopsProcessAndRejectsUnknownApplications) {
  CreateWork();
  u32 token;
  ASSERT_EQ(LaunchProcess(token), 0);
  i32 result;
  ASSERT_EQ(PollProcess(token, result), 0);
  token = 0x37533753;
  EXPECT_EQ(KillProcess(token, 0x60000001), -kern::SysError::eNOENT);
  EXPECT_EQ(token, 0x37533753u);
  ASSERT_EQ(KillProcess(token), 0);
  EXPECT_EQ(
      kern::ipmi::CommonDialogProcessStatus("SceLncUtilAppStatus60000002"), 4u);
  result = 73;
  ASSERT_EQ(PollProcess(token, result, 0, 0x30043), 0);
  EXPECT_EQ(result, 0);
  u64 before, after;
  ASSERT_EQ(kern::ShmRead("/60000001CdlgShWk", 0x18, &before, 8), 0);
  base::SleepForMilliseconds(20);
  ASSERT_EQ(kern::ShmRead("/60000001CdlgShWk", 0x18, &after, 8), 0);
  EXPECT_EQ(before, after);
  EXPECT_EQ(KillProcess(token), -kern::SysError::eNOENT);
  ASSERT_EQ(LaunchProcess(token), 0);
  EXPECT_EQ(
      kern::ipmi::CommonDialogProcessStatus("SceLncUtilAppStatus60000002"), 1u);
}

TEST_F(CommonDialogTest, AsyncManagerRejectsBadBuffersWithoutWritingOutputs) {
  u32 manager = 0x37533753, token = 0x37533753;
  struct Request {
    u32 method, timeout;
    u64 reserved;
    u32 count, pad;
    const u64* inputs;
    u32* token;
    u32 flags, pad1;
  } request{0x30040, ~0u, 0, 1, 0, nullptr, &token, 1, 0};
  EXPECT_EQ(kern::ipmi::ManagerCall(577, kid, &manager, &request,
                                    sizeof(request) - 1),
            -kern::SysError::eINVAL);
  EXPECT_EQ(
      kern::ipmi::ManagerCall(577, kid, &manager, &request, sizeof(request)),
      -kern::SysError::eINVAL);
  const u64 input[] = {1, 4};
  request.inputs = input;
  EXPECT_EQ(
      kern::ipmi::ManagerCall(577, kid, &manager, &request, sizeof(request)),
      -kern::SysError::eFAULT);
  EXPECT_EQ(manager, 0x37533753u);
  EXPECT_EQ(token, 0x37533753u);
  EXPECT_EQ(guest::LiveThreads(), 0u);
}

TEST_F(CommonDialogTest, AsyncLaunchFailurePreservesRequestToken) {
  u32 token = 0x37533753;
  EXPECT_EQ(LaunchProcess(token), -kern::SysError::eNOENT);
  EXPECT_EQ(token, 0x37533753u);
  CreateWork();
  EXPECT_EQ(LaunchProcess(token, 0, "NPXS99999"), -kern::SysError::eNOSYS);
  EXPECT_EQ(token, 0x37533753u);
  EXPECT_EQ(guest::LiveThreads(), 0u);
  guest::RequestStop();
  EXPECT_EQ(LaunchProcess(token), -kern::SysError::eIO);
  EXPECT_EQ(token, 0x37533753u);
}

TEST_F(CommonDialogTest, AutomaticAcceptPreservesFirstSelectionAndReplyBounds) {
  for (u32 mode : {1u, 2u, 3u, 4u, 6u, 7u}) {
    SCOPED_TRACE(mode);
    request.assign(kPayload, 0);
    const u32 count = 2;
    std::memcpy(request.data() + 0x38, &mode, 4);
    std::memcpy(request.data() + 0x805c, &count, 4);
    std::memcpy(request.data() + 0x5c, "FIRST", 6);
    std::memcpy(request.data() + 0x7c, "SECOND", 7);
    ASSERT_EQ(kern::ShmWrite(kName, kReply, request.data(), request.size()), 0);
    const u32 zero = 0;
    ASSERT_EQ(kern::ShmWrite(kName, 0x18, &zero, 4), 0);
    ASSERT_EQ(Register(), 0);
    ASSERT_EQ(Invoke(0x30033), 0);
    const auto deadline = base::TimeTicks::Now() + base::Seconds(2);
    while (!State() && base::TimeTicks::Now() < deadline)
      base::SleepForMilliseconds(1);
    ASSERT_EQ(State(), 1u);
    std::vector<u8> read(kPayload);
    ASSERT_EQ(kern::ShmRead(kName, kReply, read.data(), read.size()), 0);
    const u32 button = 1;
    std::memcpy(request.data() + 0x24b98, &button, 4);
    std::memcpy(request.data() + 0x24b9c, request.data() + 0x5c, 32);
    EXPECT_EQ(read, request);
    EXPECT_EQ(Invoke(0x30034), 0);
    EXPECT_EQ(Invoke(0x30045, kName, sizeof(kName)), 0);
  }
}

TEST_F(CommonDialogTest, DeactivationStopsProgressWithoutPublishingAResult) {
  ASSERT_EQ(Register(), 0);
  ASSERT_EQ(Invoke(0x30033), 0);
  ASSERT_EQ(Invoke(0x30034), 0);
  Close();
  base::SleepForMilliseconds(20);
  EXPECT_EQ(State(), 0u);
}

TEST_F(CommonDialogTest, ProgressStaysRunningUntilCloseAndPublishesReply) {
  std::vector<u8> sentinel = request;
  sentinel.resize(kPayload + 16, 0xa5);
  ASSERT_EQ(kern::ShmWrite(kName, kReply, sentinel.data(), sentinel.size()), 0);
  ASSERT_EQ(Register(), 0);
  ASSERT_EQ(Invoke(0x30033), 0);
  u64 ready = 0;
  ASSERT_EQ(kern::ShmRead(kName, 0x28, &ready, sizeof(ready)), 0);
  EXPECT_EQ(ready, 0x102u);
  base::SleepForMilliseconds(20);
  EXPECT_EQ(State(), 0u);
  std::vector<u8> read(sentinel.size());
  ASSERT_EQ(kern::ShmRead(kName, kReply, read.data(), read.size()), 0);
  EXPECT_EQ(read, sentinel);
  Close();
  const auto deadline = base::TimeTicks::Now() + base::Seconds(2);
  while (!State() && base::TimeTicks::Now() < deadline)
    base::SleepForMilliseconds(1);
  ASSERT_EQ(State(), 1u);
  ASSERT_EQ(kern::ShmRead(kName, kReply, read.data(), read.size()), 0);
  request.resize(kPayload + 16, 0xa5);
  const u32 button = 1;
  std::memcpy(request.data() + 0x24b98, &button, 4);
  EXPECT_EQ(read, request);
  EXPECT_EQ(Invoke(0x30045, kName, sizeof(kName)), 0);
  EXPECT_EQ(Invoke(0x30034), 0);
  EXPECT_NE(kern::ShmIdentity(kName), 0u);
}

TEST_F(CommonDialogTest, RegistrationRejectsMalformedNamesAndMissingBacking) {
  EXPECT_EQ(Register("/00010002SaveDataDialog"), -kern::SysError::eNOENT);
  EXPECT_EQ(kern::ShmIdentity("/00010002SaveDataDialog"), 0u);
  std::array<char, 24> name;
  std::memcpy(name.data(), kName, sizeof(kName));
  for (size_t offset : {0u, 1u, 8u, 9u, 23u}) {
    const char original = name[offset];
    name[offset] = '!';
    EXPECT_EQ(Invoke(0x30044, name.data(), name.size()),
              -kern::SysError::eINVAL);
    name[offset] = original;
  }
  EXPECT_EQ(Invoke(0x30044, kName, sizeof(kName) - 1), -kern::SysError::eINVAL);
  EXPECT_EQ(Invoke(0x30044, kName, sizeof(kName), 0, true),
            -kern::SysError::eINVAL);
  ASSERT_EQ(Register(), 0);
  EXPECT_EQ(Register(), -kern::SysError::eBUSY);
  EXPECT_EQ(Register(kName, Client()), -kern::SysError::eBUSY);
}

TEST_F(CommonDialogTest, CloseCommandPreservesCancelAndRejectsInvalidResult) {
  ASSERT_EQ(Register(), 0);
  ASSERT_EQ(Invoke(0x30033), 0);
  u64 command = 3 | (u64(2) << 32);
  ASSERT_EQ(kern::ShmWrite(kName, 0x20, &command, sizeof(command)), 0);
  Close();
  base::SleepForMilliseconds(20);
  EXPECT_EQ(State(), 0u);
  command = 3 | (u64(1) << 32);
  ASSERT_EQ(kern::ShmWrite(kName, 0x20, &command, sizeof(command)), 0);
  const auto deadline = base::TimeTicks::Now() + base::Seconds(2);
  while (!State() && base::TimeTicks::Now() < deadline)
    base::SleepForMilliseconds(1);
  ASSERT_EQ(State(), 1u);
  u32 result = ~0u;
  ASSERT_EQ(kern::ShmRead(kName, kReply + 0x24b94, &result, 4), 0);
  EXPECT_EQ(result, 1u);
}

TEST_F(CommonDialogTest, LifecycleBelongsToTheRegisteringClient) {
  ASSERT_EQ(Register(), 0);
  const u32 other = Client();
  EXPECT_EQ(Invoke(0x30033, nullptr, 0, other), -kern::SysError::eNOENT);
  EXPECT_EQ(Invoke(0x30034, nullptr, 0, other), 0);
  EXPECT_EQ(Invoke(0x30045, kName, sizeof(kName), other),
            -kern::SysError::eNOENT);
  ASSERT_EQ(Invoke(0x30033), 0);
  EXPECT_EQ(State(), 0u);
  EXPECT_EQ(Invoke(0x30033, kName, sizeof(kName)), -kern::SysError::eINVAL);
  EXPECT_EQ(Invoke(0x30034, nullptr, 0, 0, true), -kern::SysError::eINVAL);
}

TEST_F(CommonDialogTest, UnsupportedModesAndMalformedHeadersDoNotLaunch) {
  ASSERT_EQ(Register(), 0);
  const u32 list = 8;
  ASSERT_EQ(kern::ShmWrite(kName, 0x438, &list, 4), 0);
  EXPECT_EQ(Invoke(0x30033), -kern::SysError::eNOSYS);
  EXPECT_EQ(Invoke(0x30034), 0);
  ASSERT_EQ(kern::ShmWrite(kName, 0x400, request.data(), request.size()), 0);
  for (size_t offset : {0u, 4u, 8u, 16u, 20u}) {
    u32 original;
    ASSERT_EQ(kern::ShmRead(kName, offset, &original, 4), 0);
    const u32 invalid = 0xffffffff;
    ASSERT_EQ(kern::ShmWrite(kName, offset, &invalid, 4), 0);
    EXPECT_EQ(Invoke(0x30033), -kern::SysError::eINVAL);
    EXPECT_EQ(Invoke(0x30034), 0);
    ASSERT_EQ(kern::ShmWrite(kName, offset, &original, 4), 0);
  }
  EXPECT_EQ(guest::LiveThreads(), 0u);
  ASSERT_EQ(Invoke(0x30033), 0);
}

TEST_F(CommonDialogTest, ReplyCapacityIsCheckedBeforeStartingWorker) {
  ASSERT_EQ(kern::sys_shm_unlink(kName), 0);
  Create(kName, kReply + kPayload - 1);
  ASSERT_EQ(Register(), 0);
  EXPECT_EQ(Invoke(0x30033), -kern::SysError::eINVAL);
  EXPECT_EQ(guest::LiveThreads(), 0u);
  EXPECT_EQ(State(), 0u);
}

TEST_F(CommonDialogTest, ReplacedChannelIsNeverWrittenByOldWorker) {
  ASSERT_EQ(Register(), 0);
  ASSERT_EQ(Invoke(0x30033), 0);
  const u64 identity = kern::ShmIdentity(kName);
  ASSERT_EQ(kern::sys_shm_unlink(kName), 0);
  Create(kName);
  ASSERT_NE(kern::ShmIdentity(kName), identity);
  const u32 sentinel = 0xa537a537;
  ASSERT_EQ(kern::ShmWrite(kName, kReply, &sentinel, 4), 0);
  Close();
  base::SleepForMilliseconds(20);
  EXPECT_EQ(Invoke(0x30034), 0);
  EXPECT_EQ(State(), 0u);
  u32 read;
  ASSERT_EQ(kern::ShmRead(kName, kReply, &read, 4), 0);
  EXPECT_EQ(read, sentinel);
  EXPECT_EQ(kern::ShmWrite(kName, kReply, &read, 4, identity),
            -kern::SysError::eNOENT);
  read = 0x37;
  EXPECT_EQ(kern::ShmRead(kName, kReply, &read, 4, identity),
            -kern::SysError::eNOENT);
  EXPECT_EQ(read, 0x37u);
  ASSERT_EQ(Invoke(0x30045, kName, sizeof(kName)), 0);
  EXPECT_NE(kern::ShmIdentity(kName), 0u);
}

TEST_F(CommonDialogTest, ClientDestructionRetiresWorkWithoutUnlinkingChannel) {
  ASSERT_EQ(Register(), 0);
  ASSERT_EQ(Invoke(0x30033), 0);
  u32 result = 73;
  ASSERT_EQ(kern::ipmi::ManagerCall(3, kid, &result, nullptr, 0), 0);
  Close();
  base::SleepForMilliseconds(20);
  EXPECT_EQ(State(), 0u);
  EXPECT_NE(kern::ShmIdentity(kName), 0u);
  kid = Client();
  EXPECT_EQ(Invoke(0x30034), 0);
  ASSERT_EQ(Register(), 0);
}

TEST_F(CommonDialogTest, OnlyOneDialogProcessCanRunAndStoppingRejectsLaunch) {
  ASSERT_EQ(Register(), 0);
  ASSERT_EQ(Invoke(0x30033), 0);
  const char* other_name = "/00010002SaveDataDialog";
  Create(other_name);
  const u32 other = Client();
  ASSERT_EQ(Register(other_name, other), 0);
  EXPECT_EQ(Invoke(0x30033, nullptr, 0, other), -kern::SysError::eBUSY);
  EXPECT_EQ(State(other_name), 0u);
  ASSERT_EQ(Invoke(0x30045, kName, sizeof(kName)), 0);
  ASSERT_EQ(Invoke(0x30033, nullptr, 0, other), 0);
  guest::RequestStop();
  guest::JoinThreads();
  EXPECT_EQ(guest::LiveThreads(), 0u);
  EXPECT_EQ(Invoke(0x30033, nullptr, 0, other), -kern::SysError::eIO);
}
}  // namespace
