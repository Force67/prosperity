#include <gtest/gtest.h>

#include <array>
#include <cstring>

#include "guest/session.h"
#include "kern/ipmi/ipmi.h"
#include "kern/lv2/error_table.h"
#include "options/options.h"

namespace {
class ImeTest : public testing::Test {
 protected:
  void SetUp() override {
    options::Init();
    guest::ResetSessionResources();
    const char* name = "SceImeService";
    u64 request = reinterpret_cast<u64>(name);
    ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, &request, sizeof(request)),
              0);
    const u32 type = 4, user = 1;
    std::memcpy(data.data(), &type, 4);
    std::memcpy(data.data() + 20, &user, 4);
  }
  void TearDown() override { guest::ResetSessionResources(); }

  struct ConnectRequest {
    const void* data;
    u64 size;
    i32* result;
    const void* timeout;
  };
  int Connect(const void* input, u64 size, i32* result) {
    ConnectRequest request{input, size, result, nullptr};
    manager_result = 73;
    return kern::ipmi::ManagerCall(1024, kid, &manager_result, &request,
                                   sizeof(request));
  }
  void Reject(u32 expected) {
    i32 result = 73;
    EXPECT_EQ(Connect(data.data(), data.size(), &result),
              -static_cast<i32>(kern::SysError::eACCES));
    EXPECT_EQ(static_cast<u32>(result), expected);
    EXPECT_EQ(manager_result, 73u);
  }
  u32 kid = 0, manager_result = 0;
  std::array<u8, 24> data{};
};

TEST_F(ImeTest, DisconnectedKeyboardRejectsRepeatedHandshakes) {
  Reject(0x80bc0004);
  Reject(0x80bc0004);
  const u32 user = 0xfe;
  std::memcpy(data.data() + 20, &user, 4);
  Reject(0x80bc0004);
}

TEST_F(ImeTest, InvalidUsersAndReservedFieldsAreRejected) {
  const u32 user = 2;
  std::memcpy(data.data() + 20, &user, 4);
  Reject(0x80bc0010);
  for (size_t i = 12; i < 20; ++i) {
    data[i] = 1;
    Reject(0x80bc0032);
    data[i] = 0;
  }
}

TEST_F(ImeTest, UnsupportedContextsDoNotPretendToOpen) {
  data[0] = 1;
  Reject(0x80bc00ff);
  i32 result = 73;
  EXPECT_EQ(Connect(data.data(), 23, &result),
            -static_cast<i32>(kern::SysError::eACCES));
  EXPECT_EQ(static_cast<u32>(result), 0x80bc0032u);
}

TEST_F(ImeTest, UnmappedPayloadAndResultLeaveResultsUnchanged) {
  i32 result = 73;
  EXPECT_EQ(Connect(reinterpret_cast<void*>(1), 24, &result),
            -static_cast<i32>(kern::SysError::eFAULT));
  EXPECT_EQ(result, 73);
  EXPECT_EQ(Connect(data.data(), 24, reinterpret_cast<i32*>(1)),
            -static_cast<i32>(kern::SysError::eFAULT));
  EXPECT_EQ(manager_result, 73u);
  EXPECT_EQ(kern::ipmi::ManagerCall(1024, kid, &manager_result,
                                    reinterpret_cast<void*>(1), 32),
            -static_cast<i32>(kern::SysError::eFAULT));
}

TEST_F(ImeTest, NullServiceResultDoesNotCreateAKeyboard) {
  EXPECT_EQ(Connect(data.data(), 24, nullptr),
            -static_cast<i32>(kern::SysError::eACCES));
}

TEST_F(ImeTest, TruncatedConnectDoesNotWriteServiceResult) {
  i32 result = 73;
  ConnectRequest request{data.data(), data.size(), &result, nullptr};
  EXPECT_EQ(kern::ipmi::ManagerCall(1024, kid, &manager_result, &request,
                                    sizeof(request) - 1),
            -static_cast<i32>(kern::SysError::eINVAL));
  EXPECT_EQ(result, 73);
}

TEST_F(ImeTest, OtherServicesStillAcceptEmptyHandshakes) {
  const char* name = "SceUserService";
  u64 request = reinterpret_cast<u64>(name);
  ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, &request, sizeof(request)), 0);
  i32 results[] = {73, 74};
  ASSERT_EQ(Connect(nullptr, 0, results), 0);
  EXPECT_EQ(results[0], 0);
  EXPECT_EQ(results[1], 74);
  EXPECT_EQ(manager_result, 0u);
  EXPECT_EQ(Connect(nullptr, 0, nullptr), 0);
}

TEST_F(ImeTest, MethodsWithoutAnOpenContextFailWithoutTouchingOutput) {
  i32 result = 73;
  u32 output = 73;
  u64 out[] = {reinterpret_cast<u64>(&output), sizeof(output), 1};
  struct {
    u32 method, num_in, num_out, pad;
    const u64* in;
    const u64* out;
    i32* result;
  } request{0, 0, 1, 0, nullptr, out, &result};
  EXPECT_EQ(kern::ipmi::ManagerCall(800, kid, &manager_result, &request,
                                    sizeof(request)),
            0);
  EXPECT_EQ(static_cast<u32>(result), 0x80bc0001u);
  EXPECT_EQ(output, 73u);
}
}  // namespace
