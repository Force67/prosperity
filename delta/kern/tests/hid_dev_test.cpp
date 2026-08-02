#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "kern/proc.h"
#include "kern/ps4/dev/file_dev.h"
#include "kern/ps4/dev/hid_dev.h"
#include "kern/ps4/lv2/error_table.h"

// Without DELTA_HID_PASSTHROUGH the device has no host input behind it, so it
// has to soft-succeed exactly like the real device does for non-system
// processes: accepted, but the read commands produce nothing and the write
// commands are ignored.
namespace {

class HidDevice : public ::testing::Test {
protected:
  // A device registers itself in its process' object table, which then owns it,
  // so it has to be built the way the kernel builds one.
  HidDevice() : dev_(*new krnl::hidDevice(&proc_)) {}

  krnl::proc proc_;
  krnl::hidDevice &dev_;
};

// The read commands request up to 16 reports (controller: 64), but the real
// device hands out none to a non-system caller and the emulator's passthrough
// is off by default, so every read has to report zero reports.
TEST_F(HidDevice, ReadsProduceNothing) {
  struct Case {
    uint32_t cmd;
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
  for (const auto &c : cases) {
    uint8_t buf[0x40];
    std::memset(buf, 0xAA, sizeof(buf));
    EXPECT_EQ(dev_.ioctl(c.cmd, buf), 0) << std::hex << c.cmd;
    // A read that produced nothing must not have touched the guest buffer.
    for (uint8_t b : buf)
      EXPECT_EQ(b, 0xAA) << std::hex << c.cmd;
  }
}

// The write commands drive the uinput device when passthrough is on; with it
// off they are accepted and ignored, and must not rewrite their argument.
TEST_F(HidDevice, WritesAreAcceptedAndIgnored) {
  uint8_t buf[0x10];
  std::memset(buf, 0x5A, sizeof(buf));
  EXPECT_EQ(dev_.ioctl(0x80104822, buf), 0);  // SetVibration
  EXPECT_EQ(dev_.ioctl(0x80104821, buf), 0);  // SetLightBar
  EXPECT_EQ(dev_.ioctl(0x80044825, buf), 0);  // ResetLightBar
  for (uint8_t b : buf)
    EXPECT_EQ(b, 0x5A);  // a setter reads its argument, it does not rewrite it
}

// An unknown ioctl soft-succeeds, but an IOC_OUT one still owes the caller a
// defined buffer.
TEST_F(HidDevice, UnknownIoctlZeroesAnOutBuffer) {
  uint8_t buf[0x20];
  std::memset(buf, 0xAA, sizeof(buf));
  EXPECT_EQ(dev_.ioctl(0x40107499, buf), 0);  // IOC_OUT, 0x10 bytes
  for (uint32_t i = 0; i < 0x10; i++)
    EXPECT_EQ(buf[i], 0) << "byte " << i;
  EXPECT_EQ(buf[0x10], 0xAA);

  // IOC_VOID carries no payload, so nothing may be written through the pointer.
  std::memset(buf, 0xAA, sizeof(buf));
  EXPECT_EQ(dev_.ioctl(0x20007499, buf), 0);
  for (uint8_t b : buf)
    EXPECT_EQ(b, 0xAA);
}

TEST_F(HidDevice, ReportsItselfAsACharacterDevice) {
  krnl::SceKernelStat st;
  std::memset(&st, 0xAA, sizeof(st));
  EXPECT_EQ(dev_.fstat(&st), 0);
  EXPECT_EQ(st.st_mode & 0xF000, 0x2000);  // S_IFCHR
  EXPECT_EQ(st.st_size, 0);
  EXPECT_EQ(dev_.fstat(nullptr), -static_cast<int>(krnl::SysError::eFAULT));
}

}  // namespace
