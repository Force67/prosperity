#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <thread>

#include "guest/session.h"
#include "kern/ipmi/ipmi.h"
#include "options/options.h"

namespace {
class NetCtlTest : public testing::Test {
 protected:
  void SetUp() override {
    options::Init();
    guest::ResetSessionResources();
    CreateClient();
  }
  void TearDown() override { guest::ResetSessionResources(); }
  void CreateClient() {
    const char* name = "SceNetCtl";
    u64 request = reinterpret_cast<u64>(name);
    ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, &request, sizeof(request)),
              0);
  }
  i32 Invoke(u32 method,
             const void* input,
             u64 input_size,
             void* output,
             u64 capacity) {
    i32 result = -1;
    u32 manager_result = ~0u;
    u64 in[] = {reinterpret_cast<u64>(input), input_size};
    u64 out[] = {reinterpret_cast<u64>(output), capacity, 1};
    struct Request {
      u32 method, num_in, num_out, pad;
      const u64* in;
      const u64* out;
      i32* result;
    } request{method, input ? 1u : 0u, output ? 1u : 0u, 0, in, out, &result};
    EXPECT_EQ(kern::ipmi::ManagerCall(800, kid, &manager_result, &request,
                                      sizeof(request)),
              0);
    EXPECT_EQ(manager_result, 0u);
    return result;
  }
  i32 Info(u32 selector, void* output, u64 capacity = 256) {
    return Invoke(0x20005, &selector, sizeof(selector), output, capacity);
  }
  u32 kid = 0;
};

TEST_F(NetCtlTest, ConfiguredWiredStateAndIdentity) {
  u32 state[] = {77, 77};
  ASSERT_EQ(Invoke(0x20004, nullptr, 0, state, 4), 0);
  EXPECT_EQ(state[0], 3u);
  EXPECT_EQ(state[1], 77u);
  std::array<u8, 257> info;
  const std::array<u32, 4> expected{0, 0, 1500, 1};
  for (u32 selector : {1u, 3u, 4u}) {
    info.fill(77);
    ASSERT_EQ(Info(selector, info.data()), 0);
    u32 value;
    std::memcpy(&value, info.data(), sizeof(value));
    EXPECT_EQ(value, expected[selector - 1]);
    for (u32 i = 4; i < 256; ++i)
      EXPECT_EQ(info[i], 0);
    EXPECT_EQ(info[256], 77);
  }
  ASSERT_EQ(Info(2, info.data()), 0);
  const u8 mac[] = {0x02, 0x50, 0x54, 0, 0, 1};
  EXPECT_EQ(std::memcmp(info.data(), mac, sizeof(mac)), 0);
  EXPECT_EQ(info[256], 77);
}

TEST_F(NetCtlTest, AddressAndNameSelectorsAreTerminatedAndPadded) {
  struct Expected {
    u32 selector;
    const char* value;
  };
  for (auto [selector, value] :
       {Expected{12, "ps4"}, Expected{14, "192.168.1.100"},
        Expected{15, "255.255.255.0"}, Expected{16, "192.168.1.1"},
        Expected{17, "8.8.8.8"}, Expected{18, "8.8.4.4"}}) {
    std::array<char, 257> info;
    info.fill(77);
    ASSERT_EQ(Info(selector, info.data()), 0);
    EXPECT_STREQ(info.data(), value);
    for (u32 i = std::strlen(value); i < 256; ++i)
      EXPECT_EQ(info[i], 0);
    EXPECT_EQ(info[256], 77);
  }
}

TEST_F(NetCtlTest, WiredInterfaceHasNoWirelessOrProxySettings) {
  std::array<u8, 257> info;
  for (u32 selector : {5u, 6u, 7u, 8u, 9u, 10u, 11u, 13u, 19u, 20u, 21u}) {
    info.fill(77);
    ASSERT_EQ(Info(selector, info.data()), 0);
    for (u32 i = 0; i < 256; ++i)
      EXPECT_EQ(info[i], 0);
    EXPECT_EQ(info[256], 77);
  }
}

TEST_F(NetCtlTest, InvalidSelectorsAndBuffersLeaveCallerMemoryUnchanged) {
  std::array<u8, 256> info;
  info.fill(77);
  for (u32 selector : {0u, 22u, ~0u}) {
    EXPECT_EQ(Info(selector, info.data()), static_cast<i32>(0x80412102u));
    EXPECT_EQ(info[0], 77);
    EXPECT_EQ(info[255], 77);
  }
  for (u64 capacity : {0u, 4u, 255u}) {
    EXPECT_EQ(Info(14, info.data(), capacity), static_cast<i32>(0x80412107u));
    EXPECT_EQ(info[0], 77);
    EXPECT_EQ(info[255], 77);
  }
  EXPECT_EQ(Info(14, reinterpret_cast<void*>(1)),
            static_cast<i32>(0x80412107u));
  u32 selector = 14;
  for (u64 size : {0u, 3u, 8u})
    EXPECT_EQ(Invoke(0x20005, &selector, size, info.data(), info.size()),
              static_cast<i32>(0x80412107u));
  EXPECT_EQ(Invoke(0x20005, reinterpret_cast<void*>(1), 4, info.data(), 256),
            static_cast<i32>(0x80412107u));
  EXPECT_EQ(info[0], 77);
  EXPECT_EQ(info[255], 77);
  u32 state = 77;
  EXPECT_EQ(Invoke(0x20004, nullptr, 0, &state, 3),
            static_cast<i32>(0x80412107u));
  EXPECT_EQ(state, 77u);
}

TEST_F(NetCtlTest, IndependentClientsAndResetKeepInterfaceConsistent) {
  const u32 first = kid;
  CreateClient();
  EXPECT_NE(kid, first);
  u32 state = 77;
  EXPECT_EQ(Invoke(0x20004, nullptr, 0, &state, 4), 0);
  EXPECT_EQ(state, 3u);
  const u32 second = kid;
  kid = first;
  EXPECT_EQ(Invoke(0x20001, nullptr, 0, nullptr, 0), 0);
  kid = second;
  EXPECT_EQ(Invoke(0x20004, nullptr, 0, &state, 4), 0);
  EXPECT_EQ(state, 3u);
  guest::ResetSessionResources();
  CreateClient();
  EXPECT_EQ(Invoke(0x20004, nullptr, 0, &state, 4), 0);
  EXPECT_EQ(state, 3u);
}

TEST_F(NetCtlTest, ConcurrentReadersGetStableInterfaceData) {
  std::array<std::thread, 4> threads;
  for (auto& thread : threads) {
    thread = std::thread([&] {
      for (u32 i = 0; i < 20; ++i) {
        std::array<char, 256> info;
        EXPECT_EQ(Info(14, info.data()), 0);
        EXPECT_STREQ(info.data(), "192.168.1.100");
      }
    });
  }
  for (auto& thread : threads)
    thread.join();
}
}  // namespace
