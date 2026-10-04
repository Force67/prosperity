#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <mutex>
#include <vector>

#include "base/threading/thread.h"
#include "base/time/time.h"
#include "guest/session.h"
#include "kern/lv2/sys_event_flag.h"
#include "kern/object_table.h"
#include "kern/ps4/audio_daemon.h"
#include "kern/ps4/audio_sink.h"
#include "options/options.h"

namespace {
std::mutex mutex;
std::vector<std::vector<u8>> blocks;
std::vector<base::TimeTicks> times;
int opened, closed, sample_bytes;
u32 sink_channels;
int Open(u32 rate, u32 channels, int is_float) {
  std::lock_guard lock(mutex);
  EXPECT_EQ(rate, 48000u);
  sink_channels = channels;
  sample_bytes = is_float ? 4 : 2;
  return opened++;
}
int Output(int, const void* samples, u32 frames) {
  std::lock_guard lock(mutex);
  const auto* bytes = static_cast<const u8*>(samples);
  blocks.emplace_back(bytes, bytes + frames * sink_channels * sample_bytes);
  times.push_back(base::TimeTicks::Now());
  return frames;
}
void Close(int) {
  std::lock_guard lock(mutex);
  ++closed;
}

class AudioDaemonTest : public testing::Test {
 protected:
  kern::ObjectTable objects;
  kern::EventFlag* flag;
  alignas(8) std::array<u8, 0x4000> control{};
  alignas(8) std::array<u8, 0x10000> samples{};
  u8* slot = control.data() + 0x20;
  template <class T>
  void Field(size_t at, T value) {
    std::memcpy(slot + at, &value, sizeof(value));
  }
  void SetUp() override {
    options::Init();
    guest::BeginSession();
    guest::ResetSessionResources();
    blocks.clear();
    times.clear();
    opened = closed = 0;
    kern::ps4::SetAudioSink({Open, Output, nullptr, Close});
    flag = new kern::EventFlag(objects, "sceAudioOutMix1337", 0);
    Field(0x08, 4u);
    Field(0x0c, 0.5f);
    Field(0x10, 0.25f);
    Field(0x34, 48000u);
    Field(0x60, 2048u);
    Field(0x00, u64{1});
  }
  void TearDown() override {
    guest::RequestStop();
    guest::JoinThreads();
    guest::ResetSessionResources();
    kern::ps4::SetAudioSink({});
    guest::BeginSession();
  }
  void Start(size_t size = 0x10000) {
    kern::AudioDaemonNoticeShm("/shm_4919_0_A", samples.data(), size);
    kern::AudioDaemonNoticeShm("/shm_4919_C", control.data(), control.size());
  }
  bool Wait(size_t count) {
    auto deadline = base::TimeTicks::Now() + base::Seconds(2);
    do {
      {
        std::lock_guard lock(mutex);
        if (blocks.size() >= count)
          return true;
      }
      base::SleepForMilliseconds(1);
    } while (base::TimeTicks::Now() < deadline);
    return false;
  }
};

TEST_F(AudioDaemonTest, CopiesPcmAppliesChannelGainsAndPacesBlocks) {
  auto* pcm = reinterpret_cast<i16*>(samples.data());
  for (size_t i = 0; i < 4096; i += 2) {
    pcm[i] = 20000;
    pcm[i + 1] = -20000;
  }
  Start();
  ASSERT_TRUE(Wait(1));
  EXPECT_EQ(*reinterpret_cast<u64*>(slot), 0u);
  u64 bits = 0;
  EXPECT_EQ(flag->Trywait(1, kern::kEvfAnd | kern::kEvfClearPat, &bits), 0);
  EXPECT_EQ(bits, 1u);
  EXPECT_NE(flag->Trywait(1, kern::kEvfAnd, &bits), 0);
  EXPECT_EQ(pcm[0], 20000);
  {
    std::lock_guard lock(mutex);
    const auto* output = reinterpret_cast<const i16*>(blocks[0].data());
    EXPECT_EQ(output[0], 10000);
    EXPECT_EQ(output[1], -5000);
  }
  Field(0x00, u64{1});
  ASSERT_TRUE(Wait(2));
  {
    std::lock_guard lock(mutex);
    EXPECT_GE(times[1] - times[0], base::Milliseconds(30));
    EXPECT_EQ(opened, 1);
  }
  kern::AudioDaemonForgetRange(samples.data(), samples.size());
  base::SleepForMilliseconds(10);
  std::lock_guard lock(mutex);
  EXPECT_EQ(closed, 1);
}

TEST_F(AudioDaemonTest, FloatPcmAndStopCloseTheSink) {
  Field(0x08, 8u);
  Field(0x2c, 1u);
  auto* pcm = reinterpret_cast<float*>(samples.data());
  pcm[0] = 0.8f;
  pcm[1] = -0.8f;
  Start();
  ASSERT_TRUE(Wait(1));
  guest::RequestStop();
  guest::JoinThreads();
  std::lock_guard lock(mutex);
  auto* output = reinterpret_cast<const float*>(blocks[0].data());
  EXPECT_FLOAT_EQ(output[0], 0.4f);
  EXPECT_FLOAT_EQ(output[1], -0.2f);
  EXPECT_EQ(closed, 1);
}

TEST_F(AudioDaemonTest, Signed32PcmConvertsToFloat) {
  Field(0x08, 8u);
  Field(0x2c, 2u);
  auto* pcm = reinterpret_cast<i32*>(samples.data());
  pcm[0] = 1073741824;
  pcm[1] = -1073741824;
  Start();
  ASSERT_TRUE(Wait(1));
  std::lock_guard lock(mutex);
  auto* output = reinterpret_cast<const float*>(blocks[0].data());
  EXPECT_FLOAT_EQ(output[0], 0.25f);
  EXPECT_FLOAT_EQ(output[1], -0.125f);
}

TEST_F(AudioDaemonTest, ShortRegionDoesNotAcknowledgeOrReadPastMapping) {
  Start(4);
  base::SleepForMilliseconds(20);
  {
    std::lock_guard lock(mutex);
    EXPECT_TRUE(blocks.empty());
    EXPECT_EQ(opened, 0);
  }
  EXPECT_EQ(*reinterpret_cast<u64*>(slot), 1u);
  kern::AudioDaemonNoticeShm("/shm_4919_0_A", samples.data(), samples.size());
  ASSERT_TRUE(Wait(1));
}

TEST_F(AudioDaemonTest, InvalidFormatDoesNotAcknowledge) {
  Field(0x34, 44100u);
  Start();
  base::SleepForMilliseconds(20);
  guest::RequestStop();
  guest::JoinThreads();
  EXPECT_EQ(*reinterpret_cast<u64*>(slot), 1u);
  EXPECT_EQ(opened, 0);
}

TEST_F(AudioDaemonTest, MultiplePortsGrantTheirOwnBitsAndRetireWithControl) {
  u8* second = slot + 0x250;
  std::memcpy(second, slot, 0x250);
  const u32 index = 1;
  std::memcpy(second + 0x50, &index, sizeof(index));
  kern::AudioDaemonNoticeShm("/shm_4919_1_A", samples.data(), samples.size());
  Start();
  ASSERT_TRUE(Wait(2));
  u64 bits = 0;
  EXPECT_EQ(flag->Trywait(3, kern::kEvfAnd | kern::kEvfClearPat, &bits), 0);
  EXPECT_EQ(bits, 3u);
  kern::AudioDaemonForgetRange(control.data(), control.size());
  base::SleepForMilliseconds(10);
  std::lock_guard lock(mutex);
  EXPECT_EQ(opened, 2);
  EXPECT_EQ(closed, 2);
  EXPECT_EQ(blocks.size(), 2u);
}
}  // namespace
