#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <thread>

#include "base/atomic.h"
#include "guest/session.h"
#include "kern/ipmi/ipmi.h"
#include "options/options.h"

namespace {
class UserServiceTest : public testing::Test {
 protected:
  void SetUp() override {
    options::Init();
    guest::ResetSessionResources();
  }
  void TearDown() override { guest::ResetSessionResources(); }

  u32 CreateClient() {
    const char* service = "SceUserService";
    u64 request = reinterpret_cast<u64>(service);
    u32 kid = 0;
    EXPECT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, &request, sizeof(request)),
              0);
    return kid;
  }

  struct Request {
    u32 method, num_in, num_out, pad;
    const u64* in;
    const u64* out;
    i32* result;
  };
  static_assert(sizeof(Request) == 40);
  i32 Invoke(u32 kid,
             u32 method,
             const void* input,
             u64 input_size,
             void* output,
             u64 capacity) {
    i32 result = -1;
    u32 manager_result = ~0u;
    u64 in[] = {reinterpret_cast<u64>(input), input_size};
    u64 out[] = {reinterpret_cast<u64>(output), capacity, 1};
    Request request{method, input ? 1u : 0u, 1, 0, in, out, &result};
    EXPECT_EQ(kern::ipmi::ManagerCall(800, kid, &manager_result, &request,
                                      sizeof(request)),
              0);
    EXPECT_EQ(manager_result, 0u);
    return result;
  }
  int Receive(u32 kid, void* data, u64 capacity, u64* size) {
    u64 request[] = {0, reinterpret_cast<u64>(data),
                     reinterpret_cast<u64>(size), capacity};
    u32 result = ~0u;
    const int status =
        kern::ipmi::ManagerCall(594, kid, &result, request, sizeof(request));
    EXPECT_EQ(result, status == 0 ? 0u : ~0u);
    return status;
  }
};

TEST_F(UserServiceTest, InitialAndForegroundUserAndNameMatch) {
  const u32 kid = CreateClient();
  i32 user = -1;
  for (u32 method : {0x30011u, 0x3000cu}) {
    ASSERT_EQ(Invoke(kid, method, nullptr, 0, &user, sizeof(user)), 0);
    EXPECT_EQ(user, 1);
  }
  struct Setting {
    i32 user;
    u32 key;
    u64 capacity;
  } setting{1, 1, 16};
  char name[16];
  std::memset(name, 0xcc, sizeof(name));
  ASSERT_EQ(Invoke(kid, 0x30004, &setting, sizeof(setting), name, sizeof(name)),
            0);
  EXPECT_STREQ(name, "Player");
  EXPECT_EQ(name[15], 0);
  setting.user = 2;
  EXPECT_EQ(Invoke(kid, 0x30004, &setting, sizeof(setting), name, sizeof(name)),
            static_cast<i32>(0x80960009u));
  setting.user = 1;
  setting.capacity = 3;
  std::memset(name, 0xcc, sizeof(name));
  EXPECT_EQ(Invoke(kid, 0x30004, &setting, sizeof(setting), name, sizeof(name)),
            static_cast<i32>(0x8096000au));
  EXPECT_EQ(static_cast<u8>(name[0]), 0xcc);
}

TEST_F(UserServiceTest, LoginListsIncludeOnlyTheLocalUser) {
  const u32 kid = CreateClient();
  for (u32 method : {0x3000eu, 0x3000fu}) {
    const size_t count = method == 0x3000e ? 4 : 20;
    std::array<i32, 21> users;
    users.fill(73);
    ASSERT_EQ(Invoke(kid, method, nullptr, 0, users.data(), count * 4), 0);
    EXPECT_EQ(users[0], 1);
    for (size_t i = 1; i < count; ++i)
      EXPECT_EQ(users[i], -1);
    EXPECT_EQ(users[count], 73);
    users.fill(73);
    EXPECT_EQ(Invoke(kid, method, nullptr, 0, users.data(), count * 4 - 1),
              static_cast<i32>(0x80960005u));
    EXPECT_EQ(users[0], 73);
  }
}

TEST_F(UserServiceTest, LoginDeliveredOncePerClientWithoutOverwritingTail) {
  for (u32 kid : {CreateClient(), CreateClient()}) {
    i32 event[] = {-1, -1, 73};
    u64 size = 0;
    ASSERT_EQ(Receive(kid, event, sizeof(event), &size), 0);
    EXPECT_EQ(size, 8u);
    EXPECT_EQ(event[0], 0);
    EXPECT_EQ(event[1], 1);
    EXPECT_EQ(event[2], 73);
    event[0] = -1;
    EXPECT_EQ(Receive(kid, event, sizeof(event), &size), -1);
    EXPECT_EQ(event[0], -1);
  }
}

TEST_F(UserServiceTest,
       InvalidBuffersDoNotConsumeLoginAndResetRestartsClients) {
  u32 kid = CreateClient();
  i32 event[] = {-1, -1};
  u64 size = 0;
  EXPECT_EQ(Receive(kid, event, 7, &size), -1);
  EXPECT_EQ(Receive(kid, event, sizeof(event), reinterpret_cast<u64*>(1)), -1);
  EXPECT_EQ(Receive(kid, reinterpret_cast<void*>(1), sizeof(event), &size), -1);
  EXPECT_EQ(Receive(kid, event, sizeof(event), &size), 0);
  EXPECT_EQ(event[1], 1);
  u32 result = ~0u;
  EXPECT_EQ(kern::ipmi::ManagerCall(3, kid, &result, nullptr, 0), 0);
  EXPECT_EQ(Receive(kid, event, sizeof(event), &size), -1);
  guest::ResetSessionResources();
  kid = CreateClient();
  EXPECT_EQ(kid, 1u);
  EXPECT_EQ(Receive(kid, event, sizeof(event), &size), 0);
}

TEST_F(UserServiceTest, ConcurrentReceiversConsumeOneLogin) {
  const u32 kid = CreateClient();
  base::Atomic<u32> delivered{0};
  std::array<std::thread, 4> threads;
  for (auto& thread : threads) {
    thread = std::thread([&] {
      i32 event[2];
      u64 size = 0;
      if (Receive(kid, event, sizeof(event), &size) == 0)
        delivered.fetch_add(1);
    });
  }
  for (auto& thread : threads)
    thread.join();
  EXPECT_EQ(delivered.load(), 1u);
}

TEST_F(UserServiceTest, InvalidDescriptorArraysReturnMethodErrors) {
  const u32 kid = CreateClient();
  i32 method_result = 0;
  u32 result = ~0u;
  Request request{
      0x30004,       1, 0, 0, reinterpret_cast<const u64*>(1), nullptr,
      &method_result};
  ASSERT_EQ(
      kern::ipmi::ManagerCall(800, kid, &result, &request, sizeof(request)), 0);
  EXPECT_EQ(method_result, static_cast<i32>(0x80960005u));
  request.method = 0x30011;
  request.num_in = 0;
  request.num_out = 1;
  request.out = reinterpret_cast<const u64*>(1);
  ASSERT_EQ(
      kern::ipmi::ManagerCall(800, kid, &result, &request, sizeof(request)), 0);
  EXPECT_EQ(method_result, static_cast<i32>(0x80960005u));
}
}  // namespace
