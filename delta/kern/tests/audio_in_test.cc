#include <gtest/gtest.h>

#include <array>
#include <cstring>

#include "base/threading/thread.h"
#include "base/time/time.h"
#include "guest/session.h"
#include "kern/ipmi/ipmi.h"
#include "kern/lv2/sys_event_flag.h"
#include "kern/object_table.h"
#include "kern/ps4/audio_daemon.h"
#include "options/options.h"

namespace {
class AudioInTest : public testing::Test {
 protected:
  alignas(8) std::array<u8, 0x4f8> control{};
  kern::ObjectTable objects;
  kern::EventFlag* flag;
  u32 client;
  static constexpr u32 kPid = 0x1337;
  void SetUp() override {
    options::Init();
    guest::BeginSession();
    guest::ResetSessionResources();
    flag = new kern::EventFlag(objects, "/evfAin00001337", 0);
    const char* name = "SceSysAudioSystemIpc";
    u64 request = reinterpret_cast<u64>(name);
    ASSERT_EQ(kern::ipmi::ManagerCall(2, 0, &client, &request, sizeof(request)),
              0);
  }
  void TearDown() override {
    guest::RequestStop();
    guest::JoinThreads();
    guest::ResetSessionResources();
    guest::BeginSession();
  }
  template <class T>
  void Field(u32 index, size_t offset, T value) {
    std::memcpy(control.data() + 0x78 + index * 0x90 + offset, &value,
                sizeof(value));
  }
  u32 Word(u32 index, size_t offset) {
    u32 value;
    std::memcpy(&value, control.data() + 0x78 + index * 0x90 + offset, 4);
    return value;
  }
  void Map(size_t size = 0x4f8) {
    kern::AudioDaemonNoticeShm("/shmAin00001337", control.data(), size);
  }
  void Configure(u32 index, u32 rate = 16000, u32 channels = 1) {
    Field(index, 0x08, 256u);
    Field(index, 0x1c, rate);
    Field(index, 0x20, channels);
  }
  i32 Invoke(u32 method, u32 pid, u32 handle, u64 size = 8) {
    u32 input[] = {pid, handle};
    u64 descriptor[] = {reinterpret_cast<u64>(input), size};
    i32 result = -1;
    struct {
      u32 method, num_in, num_out, pad;
      const u64* in;
      const u64* out;
      i32* result;
    } request{method, 1, 0, 0, descriptor, nullptr, &result};
    u32 manager = ~0u;
    EXPECT_EQ(kern::ipmi::ManagerCall(800, client, &manager, &request,
                                      sizeof(request)),
              0);
    EXPECT_EQ(manager, 0u);
    return result;
  }
  bool Take(u32 index) {
    u32 timeout = 500000;
    return flag->Wait(1ull << index, kern::kEvfAnd | kern::kEvfClearPat,
                      nullptr, &timeout) == 0;
  }
};

TEST_F(AudioInTest, CapturesDisconnectedSilenceAtThePortClock) {
  Configure(7);
  Field(7, 0x28, 1u);
  Field(7, 0x0c, 256u);
  Field(7, 0x00, u64{0x12345678});
  Map();
  EXPECT_NE(flag->Trywait(1u << 7, kern::kEvfAnd, nullptr), 0);
  const auto start = base::TimeTicks::Now();
  ASSERT_EQ(Invoke(0x12340018, kPid, 0x30000007), 0);
  ASSERT_TRUE(Take(7));
  EXPECT_GE(base::TimeTicks::Now() - start, base::Milliseconds(12));
  EXPECT_EQ(Word(7, 0x28), 0u);
  EXPECT_EQ(Word(7, 0x2c), 256u);
  EXPECT_EQ(Word(7, 0x0c), 0u);
  EXPECT_EQ(Word(7, 0x00), 0x12345678u);
  ASSERT_TRUE(Take(7));
  EXPECT_GE(base::TimeTicks::Now() - start, base::Milliseconds(26));
}

TEST_F(AudioInTest, StereoHighRateAndIndependentPortBits) {
  Configure(0, 48000, 2);
  Field(0, 0x08, 128u);
  Configure(1);
  Map();
  ASSERT_EQ(Invoke(0x12340018, kPid, 0x30000000), 0);
  ASSERT_EQ(Invoke(0x12340018, kPid, 0x30000001), 0);
  ASSERT_TRUE(Take(0));
  ASSERT_TRUE(Take(1));
  EXPECT_EQ(Word(0, 0x2c), 256u);
  EXPECT_EQ(Word(1, 0x2c), 256u);
}

TEST_F(AudioInTest, CloseAndUnmapStopCaptureGrants) {
  Configure(0);
  Map();
  ASSERT_EQ(Invoke(0x12340018, kPid, 0x30000000), 0);
  ASSERT_TRUE(Take(0));
  ASSERT_EQ(Invoke(0x12340019, kPid, 0x30000000), 0);
  flag->Trywait(1, kern::kEvfAnd | kern::kEvfClearPat, nullptr);
  base::SleepForMilliseconds(25);
  EXPECT_NE(flag->Trywait(1, kern::kEvfAnd, nullptr), 0);
  ASSERT_EQ(Invoke(0x12340018, kPid, 0x30000000), 0);
  ASSERT_TRUE(Take(0));
  kern::AudioDaemonForgetRange(control.data(), control.size());
  control.fill(0xa5);
  flag->Trywait(1, kern::kEvfAnd | kern::kEvfClearPat, nullptr);
  base::SleepForMilliseconds(25);
  EXPECT_NE(flag->Trywait(1, kern::kEvfAnd, nullptr), 0);
  for (u8 byte : control)
    EXPECT_EQ(byte, 0xa5);
  EXPECT_EQ(Invoke(0x12340018, kPid, 0x30000000),
            static_cast<i32>(0x80260109u));
}

TEST_F(AudioInTest, RejectsMalformedRequestsAndMissingMappings) {
  EXPECT_EQ(Invoke(0x12340018, kPid, 0x30000000, 4),
            static_cast<i32>(0x80260106u));
  EXPECT_EQ(Invoke(0x12340018, kPid, 0x30000000),
            static_cast<i32>(0x80260109u));
  Configure(0);
  Map(control.size() - 1);
  EXPECT_EQ(Invoke(0x12340018, kPid, 0x30000000),
            static_cast<i32>(0x80260109u));
  Map();
  EXPECT_EQ(Invoke(0x12340018, kPid + 1, 0x30000000),
            static_cast<i32>(0x80260109u));
  for (u32 handle : {0u, 0x30000008u, 0x40000000u})
    EXPECT_EQ(Invoke(0x12340018, kPid, handle), static_cast<i32>(0x80260101u));
}

TEST_F(AudioInTest, ReopenDropsThePreviousPortGrant) {
  Configure(0);
  Map();
  ASSERT_EQ(Invoke(0x12340018, kPid, 0x30000000), 0);
  base::SleepForMilliseconds(30);
  ASSERT_EQ(flag->Trywait(1, kern::kEvfAnd, nullptr), 0);
  ASSERT_EQ(Invoke(0x12340019, kPid, 0x30000000), 0);
  EXPECT_NE(flag->Trywait(1, kern::kEvfAnd, nullptr), 0);
  Field(0, 0x08, 128u);
  ASSERT_EQ(Invoke(0x12340018, kPid, 0x30000000), 0);
  EXPECT_NE(flag->Trywait(1, kern::kEvfAnd, nullptr), 0);
  ASSERT_TRUE(Take(0));
  EXPECT_EQ(Word(0, 0x2c), 128u);
}

TEST_F(AudioInTest, InvalidSharedFormatIsNotConsumed) {
  Configure(0, 44100);
  Field(0, 0x2c, 0xdeadbeefu);
  Map();
  ASSERT_EQ(Invoke(0x12340018, kPid, 0x30000000), 0);
  base::SleepForMilliseconds(25);
  EXPECT_NE(flag->Trywait(1, kern::kEvfAnd, nullptr), 0);
  EXPECT_EQ(Word(0, 0x2c), 0xdeadbeefu);
}
}  // namespace
