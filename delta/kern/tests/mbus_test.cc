#include <gtest/gtest.h>

#include <array>

#include "guest/session.h"
#include "kern/ipmi/ipmi.h"
#include "kern/object_table.h"
#include "kern/ps4/controller_ports.h"
#include "kern/ps4/dev/hid_dev.h"
#include "options/options.h"

namespace {
std::array<int, 3> g_opened_port;
u32 g_open_count = 0;
void RecordOpen(int user, int type, int index) {
  g_opened_port = {user, type, index};
  ++g_open_count;
}
class MbusTest : public testing::Test {
 protected:
  void SetUp() override {
    options::Init();
    guest::ResetSessionResources();
    kern::SetControllerOpen(nullptr);
    g_open_count = 0;
    const char* name = "SceMbusIpc";
    u64 request = reinterpret_cast<u64>(name);
    ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &kid, &request, sizeof(request)),
              0);
    u32 create[] = {~u32(0), 3, 0};
    const i32 result = device.Ioctl(0x800c4802, create);
    ASSERT_GT(result, 0);
    handle = result;
  }
  void TearDown() override {
    kern::SetControllerOpen(nullptr);
    guest::ResetSessionResources();
  }
  i32 Invoke(u32 method, const void* input, u64 size) {
    i32 result = -1;
    u32 transport = ~0u;
    u64 descriptor[] = {reinterpret_cast<u64>(input), size};
    struct {
      u32 method, num_in, num_out, pad;
      const u64* in;
      const u64* out;
      i32* result;
    } request{method, 1, 0, 0, descriptor, nullptr, &result};
    EXPECT_EQ(kern::ipmi::ManagerCall(800, kid, &transport, &request,
                                      sizeof(request)),
              0);
    EXPECT_EQ(transport, 0u);
    return result;
  }
  i32 Bind(u32 pid = 4919) {
    u32 args[] = {0, handle, 1, 0, 0, 0, pid};
    return Invoke(0xce110007, args, sizeof(args));
  }
  i32 Unbind(u32 pid = 4919) {
    u32 args[] = {0, handle, pid};
    return Invoke(0xce11000a, args, sizeof(args));
  }
  kern::ObjectTable objects;
  kern::HidDevice& device = *new kern::HidDevice(objects);
  u32 kid = 0, handle = 0;
};

TEST_F(MbusTest, NativeBindingNotifiesHostBeforeAnyInputRead) {
  kern::SetControllerOpen(RecordOpen);
  ASSERT_EQ(Bind(), 0);
  EXPECT_EQ(g_open_count, 1u);
  EXPECT_EQ(g_opened_port, (std::array<int, 3>{1, 0, 0}));
  ASSERT_EQ(Unbind(), 0);
  ASSERT_EQ(Bind(), 0);
  EXPECT_EQ(g_open_count, 2u);
  u32 invalid[] = {0, 0, 1, 0, 0, 0, 4919};
  EXPECT_EQ(Invoke(0xce110007, invalid, sizeof(invalid)),
            -kern::SysError::eBADF);
  EXPECT_EQ(g_open_count, 2u);
}

TEST_F(MbusTest, NativeRegistrationBindsALiveHidHandleToItsUserPort) {
  kern::ControllerPort port{};
  EXPECT_FALSE(kern::GetControllerPort(handle, port));
  ASSERT_EQ(Bind(), 0);
  ASSERT_TRUE(kern::GetControllerPort(handle, port));
  EXPECT_EQ(port.user, 1u);
  EXPECT_EQ(port.type, 0u);
  EXPECT_EQ(port.index, 0u);
  EXPECT_EQ(port.flags, 0u);
  EXPECT_EQ(port.pid, 4919u);
  EXPECT_EQ(Unbind(77), -kern::SysError::eINVAL);
  EXPECT_TRUE(kern::GetControllerPort(handle, port));
  ASSERT_EQ(Unbind(), 0);
  EXPECT_FALSE(kern::GetControllerPort(handle, port));
  ASSERT_EQ(Bind(), 0);
  EXPECT_TRUE(kern::GetControllerPort(handle, port));
}

TEST_F(MbusTest, ClosingHidHandleRetiresItsPort) {
  ASSERT_EQ(Bind(), 0);
  u32 close[] = {3, handle};
  ASSERT_EQ(device.Ioctl(0x80084803, close), 0);
  kern::ControllerPort port{};
  EXPECT_FALSE(kern::GetControllerPort(handle, port));
  EXPECT_EQ(Bind(), -kern::SysError::eBADF);
  EXPECT_EQ(Unbind(), -kern::SysError::eBADF);
}

TEST_F(MbusTest, HandlesAreUniqueAcrossOpenHidDevicesAndOwnedByTheirFd) {
  auto& other = *new kern::HidDevice(objects);
  u32 create[] = {~u32(0), 3, 0};
  const i32 second = other.Ioctl(0x800c4802, create);
  ASSERT_GT(second, 0);
  EXPECT_NE(static_cast<u32>(second), handle);
  ASSERT_EQ(Bind(), 0);
  u32 close[] = {3, handle};
  EXPECT_EQ(other.Ioctl(0x80084803, close), -kern::SysError::eBADF);
  kern::ControllerPort port{};
  EXPECT_TRUE(kern::GetControllerPort(handle, port));
}

TEST_F(MbusTest, ReleasingHidDeviceRetiresUnclosedHandles) {
  ASSERT_EQ(Bind(), 0);
  objects.Reset();
  kern::ControllerPort port{};
  EXPECT_FALSE(kern::GetControllerPort(handle, port));
  EXPECT_EQ(Bind(), -kern::SysError::eBADF);
}

TEST_F(MbusTest, MalformedRegistrationsLeaveExistingPortIntact) {
  ASSERT_EQ(Bind(), 0);
  std::array<u32, 7> args{0, handle, 2, 2, 1, 0, 4919};
  EXPECT_EQ(Invoke(0xce110007, args.data(), 27), -kern::SysError::eINVAL);
  EXPECT_EQ(Invoke(0xce110007, args.data(), 29), -kern::SysError::eINVAL);
  EXPECT_EQ(Invoke(0xce110007, reinterpret_cast<void*>(1), 28),
            -kern::SysError::eINVAL);
  EXPECT_EQ(Invoke(0xce11000a, args.data(), 8), -kern::SysError::eINVAL);
  args[0] = 99;
  EXPECT_EQ(Invoke(0xce110007, args.data(), 28), -kern::SysError::eNOTSUP);
  kern::ControllerPort port{};
  ASSERT_TRUE(kern::GetControllerPort(handle, port));
  EXPECT_EQ(port.user, 1u);
  EXPECT_EQ(port.index, 0u);
}

TEST_F(MbusTest, AudioRegistrationRetainsTheExistingAudioDaemonPath) {
  u32 args[] = {1, 0x20000007, 255, 0, 0, 0, 4919};
  EXPECT_EQ(Invoke(0xce110007, args, sizeof(args)), 0);
  u32 close[] = {1, 0x20000007, 4919};
  EXPECT_EQ(Invoke(0xce11000a, close, sizeof(close)), 0);
  kern::ControllerPort port{};
  EXPECT_FALSE(kern::GetControllerPort(args[1], port));
}

TEST_F(MbusTest, SessionResetDropsPortsAndNeverReusesStaleHandles) {
  ASSERT_EQ(Bind(), 0);
  guest::ResetSessionResources();
  kern::ControllerPort port{};
  EXPECT_FALSE(kern::GetControllerPort(handle, port));
  EXPECT_FALSE(kern::IsControllerHandle(handle));
  u32 create[] = {~u32(0), 3, 0};
  const i32 next = device.Ioctl(0x800c4802, create);
  ASSERT_GT(next, 0);
  EXPECT_NE(static_cast<u32>(next), handle);
}
}  // namespace
