#include <sys/mman.h>
#include <array>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "base/threading/thread.h"
#include "base/time/time.h"
#include "guest/session.h"
#include "kern/object_table.h"
#include "kern/ps4/controller_ports.h"
#include "kern/ps4/dev/usb_ds4.h"
#include "kern/ps4/dev/usbctl_dev.h"

namespace {
std::vector<std::array<u8, 2>> g_rumble;
std::vector<std::array<u8, 3>> g_light;
class UsbDeviceTest : public testing::Test {
 protected:
  void SetUp() override {
    guest::BeginSession();
    guest::ResetSessionResources();
    kern::SetControllerSource([](guest::ControllerState& state) {
      state = {0x104128, 255, 128, 128, 0, 255, 0, 12345};
      return true;
    });
    g_rumble.clear();
    g_light.clear();
  }
  void TearDown() override {
    kern::SetControllerSource(nullptr);
    kern::SetControllerRumble(nullptr);
    kern::SetControllerLight(nullptr);
  }
  struct Directory {
    void* data;
    u32 start, length;
  };
  struct Descriptor {
    void* data;
    u16 language, maximum, actual, offset;
    u8 configuration, string, interface, alternate, endpoint, report_type;
    u8 reserved[8];
  };
  struct Endpoint {
    void** buffers;
    u32* lengths;
    u32 frames, actual_frames;
    u16 flags, timeout, completion_time;
    i32 status;
  };
  struct Open {
    u32 buffer, frames;
    u16 packet;
    u8 device, index, endpoint, reserved[3];
  };
  void InitTransfers() {
    struct Init {
      Endpoint* endpoints;
      u8 count, reserved[7];
    };
    Init init{endpoints, 3, {}};
    ASSERT_EQ(device.Ioctl(0x801055c3, &init), 0);
  }
  void OpenEndpoint(u8 index, u8 endpoint, u32 maximum = 256) {
    Open open{maximum, 2, 0, 0, index, endpoint, {}};
    ASSERT_EQ(device.Ioctl(0xc01055c5, &open), 0);
    EXPECT_EQ(open.packet, 64);
  }
  i32 Complete(u8& index) {
    const auto deadline = base::TimeTicks::Now() + base::Seconds(2);
    i32 result;
    do {
      result = device.Ioctl(0x400155c2, &index);
      if (result != -static_cast<i32>(kern::SysError::eBUSY))
        return result;
      base::SleepForMilliseconds(1);
    } while (base::TimeTicks::Now() < deadline);
    return result;
  }
  Endpoint endpoints[3]{};
  kern::ObjectTable objects;
  kern::UsbctlDevice& control = *new kern::UsbctlDevice(objects);
  kern::UgenDevice& device = *new kern::UgenDevice(objects);
};

TEST_F(UsbDeviceTest, ControlEndpointCompletesSetupAndReturnsShortDescriptor) {
  InitTransfers();
  OpenEndpoint(0, 0);
  u8 setup[] = {0x80, 6, 0, 1, 0, 0, 64, 0};
  std::array<u8, 64> buffer;
  buffer.fill(0xa5);
  void* buffers[] = {setup, buffer.data()};
  u32 lengths[] = {8, 64};
  endpoints[0] = {buffers, lengths, 2, 0, 3, 1000, 0, 0};
  u8 index = 0;
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  ASSERT_EQ(Complete(index), 0);
  EXPECT_EQ(endpoints[0].status, 0);
  EXPECT_EQ(endpoints[0].actual_frames, 2u);
  EXPECT_EQ(lengths[0], 8u);
  EXPECT_EQ(lengths[1], 18u);
  EXPECT_EQ(buffer[0], 18);
  EXPECT_EQ(buffer[18], 0xa5);
}

TEST_F(UsbDeviceTest, OutputReportRoutesMotorsAndLightAndResetStopsOutputs) {
  kern::SetControllerRumble(
      [](u8 large, u8 small) { g_rumble.push_back({large, small}); });
  kern::SetControllerLight(
      [](u8 red, u8 green, u8 blue) { g_light.push_back({red, green, blue}); });
  InitTransfers();
  OpenEndpoint(0, 3);
  u8 report[32] = {5, 3, 0, 0, 42, 100, 25, 80, 220};
  void* input = report;
  u32 length = sizeof(report);
  endpoints[0] = {&input, &length, 1, 0, 3, 1000, 0, 0};
  u8 index = 0;
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  ASSERT_EQ(Complete(index), 0);
  EXPECT_EQ(endpoints[0].status, 0);
  EXPECT_EQ(length, 32u);
  ASSERT_EQ(g_rumble.size(), 1u);
  EXPECT_EQ(g_rumble[0], (std::array<u8, 2>{100, 42}));
  ASSERT_EQ(g_light.size(), 1u);
  EXPECT_EQ(g_light[0], (std::array<u8, 3>{25, 80, 220}));
  i32 reset = 0;
  ASSERT_EQ(device.Ioctl(0x80045506, &reset), 0);
  ASSERT_EQ(g_rumble.size(), 2u);
  EXPECT_EQ(g_rumble.back(), (std::array<u8, 2>{0, 0}));
  ASSERT_EQ(g_light.size(), 2u);
  EXPECT_EQ(g_light.back(), (std::array<u8, 3>{0, 0, 255}));
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  ASSERT_EQ(Complete(index), 0);
  EXPECT_EQ(endpoints[0].status, 0);
  ASSERT_EQ(g_rumble.size(), 3u);
}

TEST_F(UsbDeviceTest, InterfaceClaimIsExclusiveAndClosingOwnerReleasesIt) {
  auto& other = *new kern::UgenDevice(objects);
  u32 interface = 0;
  ASSERT_EQ(other.Ioctl(0x80045598, &interface), 0);
  EXPECT_EQ(device.Ioctl(0x80045598, &interface), -kern::SysError::eBUSY);
  EXPECT_EQ(device.Ioctl(0x80045599, &interface), -kern::SysError::eINVAL);
  ASSERT_TRUE(objects.Release(other.handle()));
  ASSERT_EQ(device.Ioctl(0x80045598, &interface), 0);
  ASSERT_EQ(device.Ioctl(0x80045599, &interface), 0);
}

TEST_F(UsbDeviceTest, ResetCancelsPendingTransfersAcrossDeviceHandles) {
  InitTransfers();
  OpenEndpoint(0, 0x84);
  std::array<u8, 64> buffer;
  buffer.fill(0xa5);
  void* output = buffer.data();
  u32 length = buffer.size();
  endpoints[0] = {&output, &length, 1, 0, 3, 1000, 0, 0};
  u8 index = 0;
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  auto& other = *new kern::UgenDevice(objects);
  i32 reset = 0;
  ASSERT_EQ(other.Ioctl(0x80045506, &reset), 0);
  EXPECT_EQ(device.Poll(1), 1);
  ASSERT_EQ(Complete(index), 0);
  EXPECT_EQ(endpoints[0].status, 5);
  EXPECT_EQ(buffer[0], 0xa5);
}

TEST_F(UsbDeviceTest,
       ConfigurationChangesCancelTransfersAndRejectInvalidIndex) {
  InitTransfers();
  OpenEndpoint(0, 0x84);
  u8 buffer[64]{};
  void* output = buffer;
  u32 length = sizeof(buffer);
  endpoints[0] = {&output, &length, 1, 0, 3, 1000, 0, 0};
  u8 index = 0;
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  i32 configuration = 255;
  ASSERT_EQ(device.Ioctl(0x80045565, &configuration), 0);
  ASSERT_EQ(Complete(index), 0);
  EXPECT_EQ(endpoints[0].status, 5);
  configuration = 1;
  EXPECT_EQ(device.Ioctl(0x80045565, &configuration), -kern::SysError::eINVAL);
  ASSERT_EQ(device.Ioctl(0x40045564, &configuration), 0);
  EXPECT_EQ(configuration, 255);
  configuration = 0;
  ASSERT_EQ(device.Ioctl(0x80045565, &configuration), 0);
  length = sizeof(buffer);
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  ASSERT_EQ(Complete(index), 0);
  EXPECT_EQ(endpoints[0].status, 0);
}

TEST_F(UsbDeviceTest, Ps5InformationExtendsMetadataWithoutOverwritingCanary) {
  std::array<u8, 0x180> buffer;
  buffer.fill(0xa5);
  ASSERT_EQ(device.Ioctl(0x41785570, buffer.data()), 0);
  EXPECT_EQ(buffer[0], 0xc4);
  EXPECT_EQ(buffer[2], 0x4c);
  EXPECT_EQ(buffer[16], 2);
  for (size_t i = 0x170; i < 0x178; ++i)
    EXPECT_EQ(buffer[i], 0);
  EXPECT_EQ(buffer[0x178], 0xa5);
  EXPECT_EQ(device.Ioctl(0x41785570, nullptr), -kern::SysError::eFAULT);
}

TEST_F(UsbDeviceTest, UnmappedSubmittedBufferCompletesWithBadAddress) {
  InitTransfers();
  OpenEndpoint(0, 0x84);
  void* output = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(output, MAP_FAILED);
  u32 length = 64;
  endpoints[0] = {&output, &length, 1, 0, 3, 1000, 0, 0};
  u8 index = 0;
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  ASSERT_EQ(::munmap(output, 4096), 0);
  ASSERT_EQ(Complete(index), 0);
  EXPECT_EQ(endpoints[0].status, 6);
  EXPECT_EQ(endpoints[0].actual_frames, 0u);
  EXPECT_EQ(length, 0u);
}

TEST_F(UsbDeviceTest, ConfigurationAndStringsAreBoundedAndHonorOffsets) {
  std::array<u8, 80> buffer;
  buffer.fill(0xa5);
  Descriptor query{buffer.data(), 0, 9, 0xffff, 0, 0, 0, 0, 0, 0, 0, {}};
  ASSERT_EQ(device.Ioctl(0xc020556d, &query), 0);
  EXPECT_EQ(query.actual, 9);
  EXPECT_EQ(buffer[2], 41);
  EXPECT_EQ(buffer[4], 1);
  EXPECT_EQ(buffer[9], 0xa5);
  query.maximum = 80;
  ASSERT_EQ(device.Ioctl(0xc020556d, &query), 0);
  EXPECT_EQ(query.actual, 41);
  EXPECT_EQ(buffer[41], 0xa5);
  EXPECT_EQ(buffer[29], 0x84);
  EXPECT_EQ(buffer[36], 3);
  EXPECT_EQ(buffer[25] | (buffer[26] << 8), sizeof(kern::kDs4ReportDescriptor));
  query.offset = 27;
  ASSERT_EQ(device.Ioctl(0xc020556d, &query), 0);
  EXPECT_EQ(query.actual, 14);
  EXPECT_EQ(buffer[0], 7);
  EXPECT_EQ(buffer[2], 0x84);
  query.configuration = 1;
  EXPECT_EQ(device.Ioctl(0xc020556d, &query), -kern::SysError::eINVAL);
  query.configuration = 0;
  query.offset = 0;
  query.string = 2;
  query.language = 0x409;
  ASSERT_EQ(device.Ioctl(0xc020556e, &query), 0);
  EXPECT_EQ(query.actual, 40);
  EXPECT_EQ(buffer[0], 40);
  EXPECT_EQ(buffer[1], 3);
  EXPECT_EQ(buffer[2], 'W');
  EXPECT_EQ(buffer[3], 0);
  query.string = 0;
  ASSERT_EQ(device.Ioctl(0xc020556e, &query), 0);
  EXPECT_EQ(query.actual, 4);
  EXPECT_EQ(buffer[2], 9);
  EXPECT_EQ(buffer[3], 4);
}

TEST_F(UsbDeviceTest, NativeControlRequestUsesPackedSetupAndShortResponse) {
  std::array<u8, 64> buffer;
  buffer.fill(0xa5);
  alignas(8) std::array<u8, 24> request{};
  void* output = buffer.data();
  std::memcpy(request.data(), &output, sizeof(output));
  const u8 setup[] = {0x80, 6, 0, 1, 0, 0, 64, 0};
  std::memcpy(request.data() + 13, setup, sizeof(setup));
  ASSERT_EQ(device.Ioctl(0xc018556f, request.data()), 0);
  EXPECT_EQ(request[10], 18);
  EXPECT_EQ(request[11], 0);
  EXPECT_EQ(buffer[0], 18);
  EXPECT_EQ(buffer[18], 0xa5);
  request[16] = 0xff;
  EXPECT_EQ(device.Ioctl(0xc018556f, request.data()), -kern::SysError::ePIPE);
  EXPECT_EQ(request[10], 0);
}

TEST_F(UsbDeviceTest, InputCompletionCopiesOneFreshDs4ReportAndReleasesSlot) {
  InitTransfers();
  OpenEndpoint(0, 0x84);
  std::array<u8, 128> buffer;
  buffer.fill(0xa5);
  void* output = buffer.data();
  u32 length = buffer.size();
  endpoints[0] = {&output, &length, 1, 0xa5a5a5a5, 3, 1000, 0, -1};
  u8 index = 0;
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  EXPECT_EQ(device.Ioctl(0x800155c0, &index), -kern::SysError::eBUSY);
  ASSERT_EQ(Complete(index), 0);
  EXPECT_EQ(index, 0);
  EXPECT_EQ(endpoints[0].status, 0);
  EXPECT_EQ(endpoints[0].actual_frames, 1u);
  EXPECT_EQ(length, 64u);
  EXPECT_EQ(buffer[0], 1);
  EXPECT_EQ(buffer[1], 255);
  EXPECT_EQ(buffer[4], 0);
  EXPECT_EQ(buffer[5], 0x22);
  EXPECT_EQ(buffer[6], 0x24);
  EXPECT_EQ(buffer[7] & 3, 2);
  EXPECT_EQ(buffer[8], 255);
  EXPECT_EQ(buffer[35], 0x80);
  EXPECT_EQ(buffer[39], 0x80);
  EXPECT_EQ(buffer[64], 0xa5);
  EXPECT_EQ(device.Poll(1), 0);
  EXPECT_EQ(device.Ioctl(0x400155c2, &index), -kern::SysError::eBUSY);
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  ASSERT_EQ(Complete(index), 0);
  EXPECT_EQ(buffer[7] >> 2, 1);
}

TEST_F(UsbDeviceTest, CancellationAndTimeoutCompleteWithoutTouchingBuffer) {
  InitTransfers();
  OpenEndpoint(0, 0x84);
  std::array<u8, 64> buffer;
  buffer.fill(0xa5);
  void* output = buffer.data();
  u32 length = buffer.size();
  endpoints[0] = {&output, &length, 1, 0, 3, 1000, 0, 0};
  u8 index = 0;
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  ASSERT_EQ(device.Ioctl(0x800155c1, &index), 0);
  EXPECT_EQ(device.Poll(1), 1);
  ASSERT_EQ(Complete(index), 0);
  EXPECT_EQ(endpoints[0].status, 5);
  EXPECT_EQ(length, 0u);
  EXPECT_EQ(endpoints[0].actual_frames, 0u);
  length = buffer.size();
  endpoints[0].timeout = 1;
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  ASSERT_EQ(Complete(index), 0);
  EXPECT_EQ(endpoints[0].status, 20);
  EXPECT_EQ(length, 0u);
  for (auto value : buffer)
    EXPECT_EQ(value, 0xa5);
}

TEST_F(UsbDeviceTest, ClosingAndUninitializingRetireQueuedBufferReferences) {
  InitTransfers();
  OpenEndpoint(0, 0x84);
  u8 buffer[64]{};
  void* output = buffer;
  u32 length = sizeof(buffer);
  endpoints[0] = {&output, &length, 1, 0, 3, 1000, 0, 0};
  u8 index = 0;
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  ASSERT_EQ(device.Ioctl(0x800155c6, &index), 0);
  EXPECT_EQ(device.Poll(1), 0);
  EXPECT_EQ(device.Ioctl(0x400155c2, &index), -kern::SysError::eBUSY);
  OpenEndpoint(0, 0x84);
  ASSERT_EQ(device.Ioctl(0x800155c0, &index), 0);
  ASSERT_EQ(device.Ioctl(0x800155c4, &index), 0);
  EXPECT_EQ(device.Ioctl(0x400155c2, &index), -kern::SysError::eINVAL);
  EXPECT_EQ(length, 64u);
  EXPECT_EQ(buffer[0], 0);
  InitTransfers();
}

TEST_F(UsbDeviceTest,
       RejectsInvalidEndpointBuffersAndFrameCountsBeforeSubmission) {
  InitTransfers();
  Open invalid{128, 1, 0, 0, 0, 0x81, {}};
  EXPECT_EQ(device.Ioctl(0xc01055c5, &invalid), -kern::SysError::eINVAL);
  OpenEndpoint(0, 0x84, 64);
  u8 index = 0;
  EXPECT_EQ(device.Ioctl(0x800155c0, &index), -kern::SysError::eINVAL);
  endpoints[0].frames = 3;
  EXPECT_EQ(device.Ioctl(0x800155c0, &index), -kern::SysError::eINVAL);
  endpoints[0].frames = 1;
  EXPECT_EQ(device.Ioctl(0x800155c0, &index), -kern::SysError::eFAULT);
  void* output = reinterpret_cast<void*>(1);
  u32 length = 64;
  endpoints[0].buffers = &output;
  endpoints[0].lengths = &length;
  EXPECT_EQ(device.Ioctl(0x800155c0, &index), -kern::SysError::eFAULT);
  length = 65;
  EXPECT_EQ(device.Ioctl(0x800155c0, &index), -kern::SysError::eINVAL);
  EXPECT_EQ(device.Poll(1), 0);
}

TEST_F(UsbDeviceTest, EnumeratesOneDeviceThenTerminatesWithoutChangingCursor) {
  std::array<u8, 256> buffer;
  buffer.fill(0xa5);
  Directory request{buffer.data(), 0, buffer.size()};
  ASSERT_EQ(control.Ioctl(0x8010557f, &request), 0);
  EXPECT_EQ(buffer[0], 19);
  EXPECT_STREQ(reinterpret_cast<char*>(buffer.data() + 1), "ugen1.1");
  EXPECT_STREQ(reinterpret_cast<char*>(buffer.data() + 9), "usb/1.1.0");
  EXPECT_EQ(buffer[19], 0);
  EXPECT_EQ(buffer[20], 0xa5);
  EXPECT_EQ(request.start, 0u);
  buffer.fill(0xa5);
  request.start = 1;
  ASSERT_EQ(control.Ioctl(0x8010557f, &request), 0);
  EXPECT_EQ(buffer[0], 0);
  EXPECT_EQ(buffer[1], 0xa5);
  request.start = ~u32(0);
  ASSERT_EQ(control.Ioctl(0x8010557f, &request), 0);
  EXPECT_EQ(buffer[0], 0);
}

TEST_F(UsbDeviceTest, HandlesShortDirectoryBuffersAndRejectsInvalidPointers) {
  std::array<u8, 32> buffer;
  buffer.fill(0xa5);
  Directory request{buffer.data(), 0, 18};
  ASSERT_EQ(control.Ioctl(0x8010557f, &request), 0);
  EXPECT_EQ(buffer[0], 0);
  EXPECT_EQ(buffer[1], 0xa5);
  request.length = 19;
  ASSERT_EQ(control.Ioctl(0x8010557f, &request), 0);
  EXPECT_EQ(buffer[0], 19);
  EXPECT_EQ(buffer[18], 0);
  EXPECT_EQ(buffer[19], 0xa5);
  EXPECT_EQ(control.Ioctl(0x8010557f, nullptr), -kern::SysError::eFAULT);
  request.data = reinterpret_cast<void*>(1);
  request.length = 0;
  EXPECT_EQ(control.Ioctl(0x8010557f, &request), 0);
  request.length = 20;
  EXPECT_EQ(control.Ioctl(0x8010557f, &request), -kern::SysError::eFAULT);
  EXPECT_EQ(control.Ioctl(0, buffer.data()), -kern::SysError::eNOTTY);
}

TEST_F(UsbDeviceTest, DescriptorAndInformationAgreeAndRespectOutputSizes) {
  std::array<u8, 0x180> buffer;
  buffer.fill(0xa5);
  ASSERT_EQ(device.Ioctl(0x40125569, buffer.data()), 0);
  const u8 descriptor[] = {18, 1,    0, 2, 0, 0, 0, 64, 0x4c,
                           5,  0xc4, 5, 0, 1, 1, 2, 0,  1};
  EXPECT_EQ(std::memcmp(buffer.data(), descriptor, sizeof(descriptor)), 0);
  EXPECT_EQ(buffer[18], 0xa5);
  buffer.fill(0xa5);
  ASSERT_EQ(device.Ioctl(0x41705570, buffer.data()), 0);
  auto word = [&](size_t offset) {
    u16 value;
    std::memcpy(&value, buffer.data() + offset, sizeof(value));
    return value;
  };
  EXPECT_EQ(word(0), 0x05c4);
  EXPECT_EQ(word(2), 0x054c);
  EXPECT_EQ(word(4), 0x0100);
  EXPECT_EQ(buffer[8], 1);
  EXPECT_EQ(buffer[9], 1);
  EXPECT_EQ(buffer[14], 1);
  EXPECT_EQ(buffer[16], 2);
  EXPECT_EQ(buffer[17], 0);
  EXPECT_STREQ(reinterpret_cast<char*>(buffer.data() + 0x28),
               "Wireless Controller");
  EXPECT_EQ(buffer[0x170], 0xa5);
  u32 first = 0, second = 0;
  ASSERT_EQ(device.Ioctl(0x4004557e, &first), 0);
  ASSERT_EQ(device.Ioctl(0x4004557e, &second), 0);
  EXPECT_NE(first, 0u);
  EXPECT_EQ(first, second);
}

TEST_F(UsbDeviceTest, RejectsUnknownRequestsAndInvalidOutputPointers) {
  for (auto command : {0x40125569u, 0x41705570u, 0x4004557eu}) {
    EXPECT_EQ(device.Ioctl(command, nullptr), -kern::SysError::eFAULT);
    EXPECT_EQ(device.Ioctl(command, reinterpret_cast<void*>(1)),
              -kern::SysError::eFAULT);
  }
  EXPECT_EQ(device.Ioctl(0, nullptr), -kern::SysError::eNOTTY);
}
}  // namespace
