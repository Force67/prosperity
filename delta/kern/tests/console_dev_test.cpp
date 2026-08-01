#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include "kern/proc.h"
#include "kern/ps4/dev/console_dev.h"
#include "kern/ps4/dev/file_dev.h"
#include "kern/ps4/lv2/error_table.h"

// No title exercises /dev/console beyond opening it, so its behaviour has to be
// pinned here rather than by a boot: every guest that does start using it will
// come through these entry points.
namespace {

class ConsoleDevice : public ::testing::Test {
protected:
  // A device registers itself in its process' object table, which then owns it,
  // so it has to be built the way the kernel builds one.
  ConsoleDevice() : dev_(*new krnl::consoleDevice(&proc_)) {}

  krnl::proc proc_;
  krnl::consoleDevice &dev_;
};

TEST_F(ConsoleDevice, ReadsReportEndOfFile) {
  char buf[8];
  std::memset(buf, 0xAA, sizeof(buf));
  EXPECT_EQ(dev_.read(buf, sizeof(buf)), 0);
}

TEST_F(ConsoleDevice, WriteReportsEveryByteConsumed) {
  const char msg[] = "guest console output\n";
  EXPECT_EQ(dev_.write(msg, sizeof(msg) - 1),
            static_cast<int64_t>(sizeof(msg) - 1));
  // An empty write is not an error, and a null buffer must not be dereferenced.
  EXPECT_EQ(dev_.write(msg, 0), 0);
  EXPECT_EQ(dev_.write(nullptr, 4), 0);
}

TEST_F(ConsoleDevice, ReportsItselfAsACharacterDevice) {
  krnl::SceKernelStat st;
  std::memset(&st, 0xAA, sizeof(st));
  EXPECT_EQ(dev_.fstat(&st), 0);
  EXPECT_EQ(st.st_mode & 0xF000, 0x2000);  // S_IFCHR
  EXPECT_EQ(st.st_size, 0);
  // A null out-pointer is the caller's fault, not a crash.
  EXPECT_EQ(dev_.fstat(nullptr), -static_cast<int>(krnl::SysError::eFAULT));
}

// A tty getter has to leave a defined value behind: the caller reads the buffer
// back, and stack garbage there reads as a configured line.
TEST_F(ConsoleDevice, GettersZeroTheirOutputBuffer) {
  struct Case {
    uint32_t cmd;
    uint32_t bytes;
  };
  const Case cases[] = {
      {0x402c7413, 0x2c},  // TIOCGETA
      {0x40087468, 8},     // TIOCGWINSZ
      {0x40047477, 4},     // TIOCGPGRP
      {0x4004667f, 4},     // FIONREAD
  };
  for (const auto &c : cases) {
    uint8_t buf[0x40];
    std::memset(buf, 0xAA, sizeof(buf));
    EXPECT_EQ(dev_.ioctl(c.cmd, buf), 0) << std::hex << c.cmd;
    for (uint32_t i = 0; i < c.bytes; i++)
      EXPECT_EQ(buf[i], 0) << std::hex << c.cmd << " byte " << i;
    // Only the payload is touched.
    EXPECT_EQ(buf[c.bytes], 0xAA) << std::hex << c.cmd;
  }
}

TEST_F(ConsoleDevice, SettersAreAcceptedAndIgnored) {
  uint8_t buf[0x2c];
  std::memset(buf, 0x5A, sizeof(buf));
  EXPECT_EQ(dev_.ioctl(0x802c7414, buf), 0);  // TIOCSETA
  EXPECT_EQ(dev_.ioctl(0x80087467, buf), 0);  // TIOCSWINSZ
  EXPECT_EQ(dev_.ioctl(0x2000745e, nullptr), 0);  // TIOCDRAIN, no argument
  for (uint8_t b : buf)
    EXPECT_EQ(b, 0x5A);  // a setter reads its argument, it does not rewrite it
}

// An unknown ioctl soft-succeeds, but an IOC_OUT one still owes the caller a
// defined buffer.
TEST_F(ConsoleDevice, UnknownIoctlZeroesAnOutBuffer) {
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

}  // namespace
