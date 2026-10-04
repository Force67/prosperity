#include <array>
#include <cstring>
#include <vector>
#include "base/arch.h"

#include <gtest/gtest.h>

#include "base/strings/format.h"
#include "guest/session.h"
#include "kern/lv2/error_table.h"
#include "kern/object_table.h"
#include "kern/ps4/controller_ports.h"
#include "kern/ps4/dev/file_dev.h"
#include "kern/ps4/dev/hid_dev.h"

namespace {
std::vector<std::array<u8, 2>> g_rumble_events;
std::vector<std::array<u8, 3>> g_light_events;
void RecordLight(u8 red, u8 green, u8 blue) {
  g_light_events.push_back({red, green, blue});
}
void RecordRumble(u8 large, u8 small) {
  g_rumble_events.push_back({large, small});
}

class HidDevice : public ::testing::Test {
 protected:
  // A device registers itself in the object table it is handed, which then owns
  // it, so it has to be built the way the kernel builds one. But that table is
  // the whole of what it needs, so no process is involved.
  HidDevice() : dev_(*new kern::HidDevice(objects_)) {}
  void SetUp() override {
    kern::SetControllerRumble(nullptr);
    kern::SetControllerLight(nullptr);
    g_rumble_events.clear();
    g_light_events.clear();
  }
  void TearDown() override {
    kern::SetControllerRumble(nullptr);
    kern::SetControllerLight(nullptr);
    kern::SetControllerSource(nullptr);
  }

  kern::ObjectTable objects_;
  kern::HidDevice& dev_;
};

TEST_F(HidDevice, NativeLightBarRoutesColorResetAndCloseToHost) {
  kern::SetControllerSource([](guest::ControllerState&) { return true; });
  kern::SetControllerLight(RecordLight);
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  ASSERT_EQ(kern::BindControllerPort(handle, {1, 0, 0, 0, 4919}), 0);
  u8 color[] = {42, 100, 251, 0};
  u64 args[] = {static_cast<u32>(handle), reinterpret_cast<u64>(color)};
  ASSERT_EQ(dev_.Ioctl(0x80104821, args), 0);
  ASSERT_EQ(g_light_events.size(), 1u);
  EXPECT_EQ(g_light_events.back(), (std::array<u8, 3>{42, 100, 251}));
  EXPECT_EQ(color[3], 0u);
  u32 reset = handle;
  ASSERT_EQ(dev_.Ioctl(0x80044825, &reset), 0);
  ASSERT_EQ(g_light_events.size(), 2u);
  EXPECT_EQ(g_light_events.back(), (std::array<u8, 3>{0, 0, 255}));
  ASSERT_EQ(dev_.Ioctl(0x80104821, args), 0);
  u32 close[] = {3, static_cast<u32>(handle)};
  ASSERT_EQ(dev_.Ioctl(0x80084803, close), 0);
  ASSERT_EQ(g_light_events.size(), 4u);
  EXPECT_EQ(g_light_events.back(), (std::array<u8, 3>{0, 0, 255}));
  EXPECT_EQ(dev_.Ioctl(0x80044825, &reset),
            -static_cast<i32>(kern::SysError::eBADF));
  EXPECT_EQ(g_light_events.size(), 4u);
}

TEST_F(HidDevice, InvalidZeroHandleCleanupDoesNotChangeHostOutputs) {
  kern::SetControllerLight(RecordLight);
  kern::SetControllerRumble(RecordRumble);
  EXPECT_FALSE(kern::ReleaseControllerHandle(0));
  EXPECT_TRUE(g_light_events.empty());
  EXPECT_TRUE(g_rumble_events.empty());
}

TEST_F(HidDevice, Ps5MultiplexedControlResetsLightAndValidatesRequests) {
  kern::SetControllerLight(RecordLight);
  kern::SetControllerSource([](guest::ControllerState&) { return true; });
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  ASSERT_EQ(kern::BindControllerPort(handle, {1, 0, 0, 0, 4919}), 0);
  u32 request[] = {4, 0xa5a5a5a5, static_cast<u32>(handle), 0xa5a5a5a5};
  ASSERT_EQ(dev_.Ioctl(0x80104850, request), 0);
  ASSERT_EQ(g_light_events.size(), 1u);
  EXPECT_EQ(g_light_events.back(), (std::array<u8, 3>{0, 0, 255}));
  EXPECT_EQ(request[1], 0xa5a5a5a5u);
  EXPECT_EQ(request[3], 0xa5a5a5a5u);
  request[0] = 99;
  EXPECT_EQ(dev_.Ioctl(0x80104850, request),
            -static_cast<i32>(kern::SysError::eNOTSUP));
  EXPECT_EQ(dev_.Ioctl(0x80104850, nullptr),
            -static_cast<i32>(kern::SysError::eFAULT));
  request[0] = 4;
  request[2] = 0;
  EXPECT_EQ(dev_.Ioctl(0x80104850, request),
            -static_cast<i32>(kern::SysError::eBADF));
  EXPECT_EQ(g_light_events.size(), 1u);
}

TEST_F(HidDevice, MalformedAndDisconnectedLightBarRequestsDoNotReachHost) {
  kern::SetControllerLight(RecordLight);
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  EXPECT_EQ(dev_.Ioctl(0x80104821, nullptr),
            -static_cast<i32>(kern::SysError::eFAULT));
  EXPECT_EQ(dev_.Ioctl(0x80044825, nullptr),
            -static_cast<i32>(kern::SysError::eFAULT));
  u64 args[] = {static_cast<u32>(handle), 1};
  EXPECT_EQ(dev_.Ioctl(0x80104821, args),
            -static_cast<i32>(kern::SysError::eFAULT));
  u8 color[] = {42, 100, 251, 0};
  args[1] = reinterpret_cast<u64>(color);
  EXPECT_EQ(dev_.Ioctl(0x80104821, args),
            -static_cast<i32>(kern::SysError::eNODEV));
  kern::SetControllerSource([](guest::ControllerState&) { return true; });
  ASSERT_EQ(kern::BindControllerPort(handle, {1, 0, 0, 0, 4919}), 0);
  color[3] = 1;
  EXPECT_EQ(dev_.Ioctl(0x80104821, args),
            -static_cast<i32>(kern::SysError::eNOTSUP));
  args[0] = 0;
  EXPECT_EQ(dev_.Ioctl(0x80104821, args),
            -static_cast<i32>(kern::SysError::eBADF));
  EXPECT_TRUE(g_light_events.empty());
}

TEST_F(HidDevice, OnlyActiveLightOwnerCleanupRestoresDefaultColor) {
  kern::SetControllerLight(RecordLight);
  kern::SetControllerSource([](guest::ControllerState&) { return true; });
  u32 create[] = {~u32(0), 3, 0};
  const i32 first = dev_.Ioctl(0x800c4802, create);
  const i32 second = dev_.Ioctl(0x800c4802, create);
  ASSERT_EQ(kern::BindControllerPort(first, {1, 0, 0, 0, 4919}), 0);
  ASSERT_EQ(kern::BindControllerPort(second, {1, 0, 0, 0, 4919}), 0);
  u8 color[] = {42, 100, 251, 0};
  u64 args[] = {static_cast<u32>(second), reinterpret_cast<u64>(color)};
  ASSERT_EQ(dev_.Ioctl(0x80104821, args), 0);
  ASSERT_EQ(kern::UnbindControllerPort(first, 4919), 0);
  EXPECT_EQ(g_light_events.size(), 1u);
  ASSERT_EQ(kern::UnbindControllerPort(second, 4919), 0);
  ASSERT_EQ(g_light_events.size(), 2u);
  EXPECT_EQ(g_light_events.back(), (std::array<u8, 3>{0, 0, 255}));
  ASSERT_EQ(kern::BindControllerPort(second, {1, 0, 0, 0, 4919}), 0);
  ASSERT_EQ(dev_.Ioctl(0x80104821, args), 0);
  guest::ResetSessionResources();
  ASSERT_EQ(g_light_events.size(), 4u);
  EXPECT_EQ(g_light_events.back(), (std::array<u8, 3>{0, 0, 255}));
  EXPECT_EQ(dev_.Ioctl(0x80104821, args),
            -static_cast<i32>(kern::SysError::eBADF));
}

TEST_F(HidDevice, NativeVibrationDrivesBothMotorsAndUnbindStopsOutput) {
  kern::SetControllerSource([](guest::ControllerState&) { return true; });
  kern::SetControllerRumble(RecordRumble);
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  ASSERT_EQ(kern::BindControllerPort(handle, {1, 0, 0, 0, 4919}), 0);
  u8 motors[] = {7, 249, 0, 0};
  u64 args[] = {static_cast<u32>(handle), reinterpret_cast<u64>(motors), 0};
  ASSERT_EQ(dev_.Ioctl(0x80184822, args), 0);
  ASSERT_EQ(g_rumble_events.size(), 1u);
  EXPECT_EQ(g_rumble_events[0], (std::array<u8, 2>{7, 249}));
  EXPECT_EQ(motors[0], 7u);
  EXPECT_EQ(motors[1], 249u);
  ASSERT_EQ(kern::UnbindControllerPort(handle, 4919), 0);
  ASSERT_EQ(g_rumble_events.size(), 2u);
  EXPECT_EQ(g_rumble_events[1], (std::array<u8, 2>{0, 0}));
  EXPECT_EQ(dev_.Ioctl(0x80184822, args),
            -static_cast<i32>(kern::SysError::eNODEV));
  EXPECT_EQ(g_rumble_events.size(), 2u);
}

TEST_F(HidDevice, ClosingTheActiveVibrationHandleStopsItsMotors) {
  kern::SetControllerSource([](guest::ControllerState&) { return true; });
  kern::SetControllerRumble(RecordRumble);
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  ASSERT_EQ(kern::BindControllerPort(handle, {1, 0, 0, 0, 4919}), 0);
  u8 motors[] = {255, 128, 0, 0};
  u64 args[] = {static_cast<u32>(handle), reinterpret_cast<u64>(motors), 0};
  ASSERT_EQ(dev_.Ioctl(0x80184822, args), 0);
  u32 close[] = {3, static_cast<u32>(handle)};
  ASSERT_EQ(dev_.Ioctl(0x80084803, close), 0);
  ASSERT_EQ(g_rumble_events.size(), 2u);
  EXPECT_EQ(g_rumble_events.back(), (std::array<u8, 2>{0, 0}));
  EXPECT_EQ(dev_.Ioctl(0x80184822, args),
            -static_cast<i32>(kern::SysError::eBADF));
  EXPECT_EQ(g_rumble_events.size(), 2u);
}

TEST_F(HidDevice, SessionResetStopsActiveMotorsAndRejectsStaleOutputHandles) {
  kern::SetControllerSource([](guest::ControllerState&) { return true; });
  kern::SetControllerRumble(RecordRumble);
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  ASSERT_EQ(kern::BindControllerPort(handle, {1, 0, 0, 0, 4919}), 0);
  u8 motors[] = {99, 33, 0, 0};
  u64 args[] = {static_cast<u32>(handle), reinterpret_cast<u64>(motors), 0};
  ASSERT_EQ(dev_.Ioctl(0x80184822, args), 0);
  guest::ResetSessionResources();
  ASSERT_EQ(g_rumble_events.size(), 2u);
  EXPECT_EQ(g_rumble_events.back(), (std::array<u8, 2>{0, 0}));
  EXPECT_EQ(dev_.Ioctl(0x80184822, args),
            -static_cast<i32>(kern::SysError::eBADF));
}

TEST_F(HidDevice, MalformedNativeVibrationDoesNotReachTheHost) {
  kern::SetControllerSource([](guest::ControllerState&) { return true; });
  kern::SetControllerRumble(RecordRumble);
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  ASSERT_EQ(kern::BindControllerPort(handle, {1, 0, 0, 0, 4919}), 0);
  u64 args[] = {static_cast<u32>(handle), 1, 0};
  EXPECT_EQ(dev_.Ioctl(0x80184822, nullptr),
            -static_cast<i32>(kern::SysError::eFAULT));
  EXPECT_EQ(dev_.Ioctl(0x80184822, args),
            -static_cast<i32>(kern::SysError::eFAULT));
  u8 motors[] = {7, 9, 0, 0};
  args[1] = reinterpret_cast<u64>(motors);
  args[2] = 1;
  EXPECT_EQ(dev_.Ioctl(0x80184822, args),
            -static_cast<i32>(kern::SysError::eNOTSUP));
  args[0] = 0;
  EXPECT_EQ(dev_.Ioctl(0x80184822, args),
            -static_cast<i32>(kern::SysError::eBADF));
  EXPECT_TRUE(g_rumble_events.empty());
}

TEST_F(HidDevice, ConnectedReadUsesBoundHostInputAndWritesOnlyOneReport) {
  kern::SetControllerSource([](guest::ControllerState& state) {
    state = {0x4108, 7, 42, 123, 250, 11, 222, 987654321};
    return true;
  });
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  ASSERT_EQ(kern::BindControllerPort(handle, {1, 0, 0, 0, 4919}), 0);
  u8 reports[2 * 0xa8];
  std::memset(reports, 0xaa, sizeof(reports));
  u32 connection[] = {123, 0xabcdef01}, extended[] = {123, 0xabcdef01};
  u64 read[] = {
      static_cast<u32>(handle),          reinterpret_cast<u64>(reports),  2,
      reinterpret_cast<u64>(connection), reinterpret_cast<u64>(extended), 0};
  ASSERT_EQ(dev_.Ioctl(0x8030482e, read), 1);
  EXPECT_EQ(connection[0], 1u);
  EXPECT_EQ(extended[0], 0u);
  EXPECT_EQ(connection[1], 0xabcdef01u);
  EXPECT_EQ(extended[1], 0xabcdef01u);
  u64 timestamp;
  u32 buttons;
  std::memcpy(&timestamp, reports, sizeof(timestamp));
  std::memcpy(&buttons, reports + 0xc, sizeof(buttons));
  EXPECT_EQ(timestamp, 987654321u);
  EXPECT_EQ(buttons, 0x4108u);
  const u8 axes[] = {7, 42, 123, 250, 11, 222};
  EXPECT_EQ(std::memcmp(reports + 0x10, axes, sizeof(axes)), 0);
  for (size_t i = 0xa8; i < sizeof(reports); ++i)
    EXPECT_EQ(reports[i], 0xaa);
  ASSERT_EQ(kern::UnbindControllerPort(handle, 4919), 0);
  std::memset(reports, 0xaa, sizeof(reports));
  EXPECT_EQ(dev_.Ioctl(0x8030482e, read), 0);
  EXPECT_EQ(connection[0], 0u);
  for (u8 byte : reports)
    EXPECT_EQ(byte, 0xaa);
}

TEST_F(HidDevice, Ps5MultiplexedReadUsesExpandedReportStride) {
  kern::SetControllerSource([](guest::ControllerState& state) {
    state = {0x4108, 7, 42, 123, 250, 11, 222, 987654321};
    return true;
  });
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  ASSERT_EQ(kern::BindControllerPort(handle, {1, 0, 0, 0, 4919}), 0);
  u8 reports[2 * 0xb8];
  std::memset(reports, 0xaa, sizeof(reports));
  u32 connection[] = {123, 0xabcdef01}, extended[] = {123, 0xabcdef01};
  struct {
    u32 operation, handle, count, reserved1, flags, reserved2;
    u64 reserved3;
    u32* connection;
    u32* extended;
    u64 reserved4[2];
    void* reports;
  } args{6,        static_cast<u32>(handle),
         2,        0,
         0,        0,
         0,        connection,
         extended, {0, 0},
         reports};
  static_assert(sizeof(args) == 0x48);
  ASSERT_EQ(dev_.Ioctl(0xc0484851, &args), 1);
  EXPECT_EQ(connection[0], 1u);
  EXPECT_EQ(extended[0], 0u);
  EXPECT_EQ(connection[1], 0xabcdef01u);
  EXPECT_EQ(extended[1], 0xabcdef01u);
  u32 buttons;
  std::memcpy(&buttons, reports + 0xc, sizeof(buttons));
  EXPECT_EQ(buttons, 0x4108u);
  EXPECT_EQ(reports[0x10], 7);
  EXPECT_EQ(reports[0x15], 222);
  for (size_t i = 0xa8; i < 0xb8; ++i)
    EXPECT_EQ(reports[i], 0u);
  for (size_t i = 0xb8; i < sizeof(reports); ++i)
    EXPECT_EQ(reports[i], 0xaa);
  connection[0] = extended[0] = 123;
  args.count = 65;
  EXPECT_EQ(dev_.Ioctl(0xc0484851, &args),
            -static_cast<i32>(kern::SysError::eINVAL));
  args.count = 1;
  args.extended = reinterpret_cast<u32*>(1);
  EXPECT_EQ(dev_.Ioctl(0xc0484851, &args),
            -static_cast<i32>(kern::SysError::eFAULT));
  args.operation = 99;
  EXPECT_EQ(dev_.Ioctl(0xc0484851, &args),
            -static_cast<i32>(kern::SysError::eNOTSUP));
  EXPECT_EQ(connection[0], 123u);
  EXPECT_EQ(extended[0], 123u);
  EXPECT_EQ(dev_.Ioctl(0xc0484851, nullptr),
            -static_cast<i32>(kern::SysError::eFAULT));
}

TEST_F(HidDevice, Ps5InformationUsesItsExpandedOutputLayout) {
  kern::SetControllerSource([](guest::ControllerState&) { return true; });
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  ASSERT_EQ(kern::BindControllerPort(handle, {1, 0, 0, 0, 4919}), 0);
  u8 info[0xa9];
  std::memset(info, 0xaa, sizeof(info));
  u64 args[] = {1 | u64(handle) << 32, 0, reinterpret_cast<u64>(info)};
  ASSERT_EQ(dev_.Ioctl(0xc0184852, args), 0);
  EXPECT_EQ(info[4], 1u);
  u16 vendor, product;
  std::memcpy(&vendor, info + 0x10, sizeof(vendor));
  std::memcpy(&product, info + 0x12, sizeof(product));
  EXPECT_EQ(vendor, 0x054c);
  EXPECT_EQ(product, 0x05c4);
  EXPECT_EQ(info[0xa4], 12u);
  EXPECT_EQ(info[0xa5], 12u);
  EXPECT_EQ(info[0xa8], 0xaa);
  args[0] = u64(1) << 32;
  ASSERT_EQ(dev_.Ioctl(0xc0184852, args), 0);
  std::memcpy(&vendor, info + 0x10, sizeof(vendor));
  std::memcpy(&product, info + 0x12, sizeof(product));
  EXPECT_EQ(vendor, 0x054c);
  EXPECT_EQ(product, 0x05c4);
  EXPECT_EQ(info[0xa8], 0xaa);
  args[0] = 99;
  std::memset(info, 0xaa, sizeof(info));
  EXPECT_EQ(dev_.Ioctl(0xc0184852, args),
            -static_cast<i32>(kern::SysError::eNOTSUP));
  for (u8 byte : info)
    EXPECT_EQ(byte, 0xaa);
}

TEST_F(HidDevice, Ps5CompatibilityModesRequireLiveHandles) {
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  u32 args[] = {static_cast<u32>(handle), 0};
  EXPECT_EQ(dev_.Ioctl(0x8008485b, args), 0);
  args[1] = 1;
  EXPECT_EQ(dev_.Ioctl(0x8008485b, args), 0);
  args[1] = 2;
  EXPECT_EQ(dev_.Ioctl(0x8008485b, args),
            -static_cast<i32>(kern::SysError::eINVAL));
  u32 close[] = {3, static_cast<u32>(handle)};
  ASSERT_EQ(dev_.Ioctl(0x80084803, close), 0);
  args[1] = 0;
  EXPECT_EQ(dev_.Ioctl(0x8008485b, args),
            -static_cast<i32>(kern::SysError::eBADF));
}

TEST_F(HidDevice, NativeInformationDescribesTheBoundVirtualController) {
  kern::SetControllerSource([](guest::ControllerState&) { return true; });
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  ASSERT_EQ(kern::BindControllerPort(handle, {1, 0, 0, 0, 4919}), 0);
  u8 info[0x41];
  std::memset(info, 0xaa, sizeof(info));
  u64 args[] = {static_cast<u32>(handle), reinterpret_cast<u64>(info)};
  ASSERT_EQ(dev_.Ioctl(0x8010480e, args), 0);
  EXPECT_EQ(info[4], 1u);
  EXPECT_EQ(info[0x3c], 12u);
  EXPECT_EQ(info[0x3d], 12u);
  EXPECT_EQ(info[0x40], 0xaa);
  args[0] = 1;
  ASSERT_EQ(dev_.Ioctl(0x80104801, args), 0);
  u16 vendor, product;
  std::memcpy(&vendor, info + 0x10, sizeof(vendor));
  std::memcpy(&product, info + 0x12, sizeof(product));
  EXPECT_EQ(vendor, 0x054c);
  EXPECT_EQ(product, 0x05c4);
  EXPECT_EQ(info[0x40], 0xaa);
  args[0] = handle;
  ASSERT_EQ(kern::UnbindControllerPort(handle, 4919), 0);
  std::memset(info, 0xaa, sizeof(info));
  EXPECT_EQ(dev_.Ioctl(0x8010480e, args), -kern::SysError::eNODEV);
  for (u8 byte : info)
    EXPECT_EQ(byte, 0xaa);
}

TEST_F(HidDevice, ControllerHandlesAreDistinctAndClosedHandlesStayInvalid) {
  u32 create[] = {~u32(0), 3, 0};
  const i32 first = dev_.Ioctl(0x800c4802, create);
  const i32 second = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(first, 0);
  ASSERT_GT(second, first);
  EXPECT_EQ(create[0], ~u32(0));
  u32 close[] = {3, static_cast<u32>(first)};
  EXPECT_EQ(dev_.Ioctl(0x80084803, close), 0);
  EXPECT_EQ(dev_.Ioctl(0x80084803, close), -kern::SysError::eBADF);
  EXPECT_GT(dev_.Ioctl(0x800c4802, create), second);
  close[1] = second;
  EXPECT_EQ(dev_.Ioctl(0x80084803, close), 0);
}

TEST_F(HidDevice, ControllerHandleRequestsValidateArguments) {
  EXPECT_EQ(dev_.Ioctl(0x800c4802, nullptr), -kern::SysError::eFAULT);
  EXPECT_EQ(dev_.Ioctl(0x80084803, nullptr), -kern::SysError::eFAULT);
  EXPECT_EQ(dev_.Ioctl(0x800c4802, reinterpret_cast<void*>(1)),
            -kern::SysError::eFAULT);
  u32 create[] = {0, 3, 0};
  EXPECT_EQ(dev_.Ioctl(0x800c4802, create), -kern::SysError::eINVAL);
  create[0] = ~u32(0);
  create[1] = 0;
  EXPECT_EQ(dev_.Ioctl(0x800c4802, create), -kern::SysError::eINVAL);
  create[1] = 3;
  create[2] = 1;
  EXPECT_EQ(dev_.Ioctl(0x800c4802, create), -kern::SysError::eINVAL);
  u32 close[] = {0, 1};
  EXPECT_EQ(dev_.Ioctl(0x80084803, close), -kern::SysError::eINVAL);
  close[0] = 3;
  EXPECT_EQ(dev_.Ioctl(0x80084803, close), -kern::SysError::eBADF);
}

TEST_F(HidDevice,
       NativeControllerReadReportsDisconnectedWithoutTouchingReports) {
  u32 create[] = {~u32(0), 3, 0};
  const i32 handle = dev_.Ioctl(0x800c4802, create);
  ASSERT_GT(handle, 0);
  u8 reports[0xa8];
  std::memset(reports, 0xaa, sizeof(reports));
  u32 connection[] = {123, 0xabcdef01}, extended[] = {123, 0xabcdef01};
  struct {
    u32 handle, reserved;
    void* buffer;
    u32 count, reserved1;
    u32* connection;
    u32* extended;
    u32 flags, reserved2;
  } read{
      static_cast<u32>(handle), 0, reports, 1, 0, connection, extended, 0, 0};
  static_assert(sizeof(read) == 0x30);
  ASSERT_EQ(dev_.Ioctl(0x8030482e, &read), 0);
  EXPECT_EQ(connection[0], 0u);
  EXPECT_EQ(extended[0], 0u);
  EXPECT_EQ(connection[1], 0xabcdef01u);
  EXPECT_EQ(extended[1], 0xabcdef01u);
  for (u8 byte : reports)
    EXPECT_EQ(byte, 0xaa);
  connection[0] = extended[0] = 123;
  read.count = 65;
  EXPECT_EQ(dev_.Ioctl(0x8030482e, &read), -kern::SysError::eINVAL);
  read.count = 0;
  EXPECT_EQ(dev_.Ioctl(0x8030482e, &read), -kern::SysError::eINVAL);
  read.count = 1;
  read.extended = nullptr;
  EXPECT_EQ(dev_.Ioctl(0x8030482e, &read), -kern::SysError::eFAULT);
  EXPECT_EQ(connection[0], 123u);
  EXPECT_EQ(extended[0], 123u);
  read.extended = reinterpret_cast<u32*>(1);
  EXPECT_EQ(dev_.Ioctl(0x8030482e, &read), -kern::SysError::eFAULT);
  read.extended = extended;
  read.connection = reinterpret_cast<u32*>(1);
  EXPECT_EQ(dev_.Ioctl(0x8030482e, &read), -kern::SysError::eFAULT);
  read.connection = connection;
  read.buffer = reinterpret_cast<void*>(1);
  EXPECT_EQ(dev_.Ioctl(0x8030482e, &read), -kern::SysError::eFAULT);
  EXPECT_EQ(connection[0], 123u);
  EXPECT_EQ(extended[0], 123u);
  u32 close[] = {3, static_cast<u32>(handle)};
  ASSERT_EQ(dev_.Ioctl(0x80084803, close), 0);
  EXPECT_EQ(dev_.Ioctl(0x8030482e, &read), -kern::SysError::eBADF);
  EXPECT_EQ(dev_.Ioctl(0x8030482e, nullptr), -kern::SysError::eFAULT);
}

// The read commands request up to 16 reports (controller: 64), but the real
// device hands out none to a non-system caller and the emulator's passthrough
// is off by default, so every read has to report zero reports.
TEST_F(HidDevice, ReadsProduceNothing) {
  struct Case {
    u32 cmd;
  };
  const Case cases[] = {
      {0x80204814},  // KeyboardRead
      {0x80204815},  // KeyboardReadPort
      {0x80204819},  // MouseRead
      {0x8020481a},  // MouseReadPort
      {0x80204820},  // ControllerRead
      {0x80204829},  // ControllerReadPort
      {0x8028482e},  // ControllerRead2
  };
  for (const auto& c : cases) {
    u8 buf[0x40];
    std::memset(buf, 0xAA, sizeof(buf));
    EXPECT_EQ(dev_.Ioctl(c.cmd, buf), 0)
        << base::Format("{:#x}", c.cmd).c_str();
    // A read that produced nothing must not have touched the guest buffer.
    for (u8 b : buf)
      EXPECT_EQ(b, 0xAA) << base::Format("{:#x}", c.cmd).c_str();
  }
}

TEST_F(HidDevice, OutputRequestsDoNotRewriteTheirArguments) {
  u8 buf[0x10];
  std::memset(buf, 0x5A, sizeof(buf));
  EXPECT_EQ(dev_.Ioctl(0x80104822, buf), 0);  // SetVibration
  EXPECT_EQ(dev_.Ioctl(0x80104821, buf),
            -static_cast<i32>(kern::SysError::eBADF));
  EXPECT_EQ(dev_.Ioctl(0x80044825, buf),
            -static_cast<i32>(kern::SysError::eBADF));
  for (u8 b : buf)
    EXPECT_EQ(b, 0x5A);  // a setter reads its argument, it does not rewrite it
}

// An unknown ioctl soft-succeeds, but an IOC_OUT one still owes the caller a
// defined buffer.
TEST_F(HidDevice, UnknownIoctlZeroesAnOutBuffer) {
  u8 buf[0x20];
  std::memset(buf, 0xAA, sizeof(buf));
  EXPECT_EQ(dev_.Ioctl(0x40107499, buf), 0);  // IOC_OUT, 0x10 bytes
  for (u32 i = 0; i < 0x10; i++)
    EXPECT_EQ(buf[i], 0) << "byte " << i;
  EXPECT_EQ(buf[0x10], 0xAA);

  // IOC_VOID carries no payload, so nothing may be written through the pointer.
  std::memset(buf, 0xAA, sizeof(buf));
  EXPECT_EQ(dev_.Ioctl(0x20007499, buf), 0);
  for (u8 b : buf)
    EXPECT_EQ(b, 0xAA);
}

TEST_F(HidDevice, ReportsItselfAsACharacterDevice) {
  kern::SceKernelStat st;
  std::memset(&st, 0xAA, sizeof(st));
  EXPECT_EQ(dev_.Fstat(&st), 0);
  EXPECT_EQ(st.st_mode & 0xF000, 0x2000);  // S_IFCHR
  EXPECT_EQ(st.st_size, 0);
  EXPECT_EQ(dev_.Fstat(nullptr), -static_cast<int>(kern::SysError::eFAULT));
}

}  // namespace
