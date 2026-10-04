#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "kern/object_table.h"
#include "kern/ps4/dev/ajm_dev.h"

namespace {

// Builds client batch commands for the audio job device.
class BatchWriter {
 public:
  explicit BatchWriter(std::vector<u8>& buffer) : buffer_(buffer) {}

  void JobHeader(u32 instance, size_t job_size) {
    const u32 total = (u32)job_size - 8;
    U32((instance << 6) & 0x3ffffc0);
    U32(total);
  }

  void Input(const void* data, u32 size) {
    U32(1);
    U32(size);
    U64(reinterpret_cast<u64>(data));
  }

  void Output(void* data, u32 size) {
    U32(0x11);
    U32(size);
    U64(reinterpret_cast<u64>(data));
  }

  void SidebandIn(const void* data, u32 size) {
    U32(2);
    U32(size);
    U64(reinterpret_cast<u64>(data));
  }

  void SidebandOut(void* data, u32 size) {
    U32(0x12);
    U32(size);
    U64(reinterpret_cast<u64>(data));
  }

  void Run(u64 flags) {
    U32(4 | (u32)(((flags >> 26) & 0x3fffc0)));
    U32((u32)flags);
  }

  void Control(u64 flags) {
    U32(3 | (u32)(((flags >> 26) & 0x3fffc0)));
    U32((u32)flags);
  }

 private:
  void U32(u32 v) {
    const auto* p = reinterpret_cast<const u8*>(&v);
    buffer_.insert(buffer_.end(), p, p + 4);
  }
  void U64(u64 v) {
    const auto* p = reinterpret_cast<const u8*>(&v);
    buffer_.insert(buffer_.end(), p, p + 8);
  }

  std::vector<u8>& buffer_;
};

// Decode result layout read by the client.
struct DecodeResultBlock {
  u32 result;
  u32 internal_result;
  int32_t consumed;
  int32_t produced;
  u64 total;
  u32 frames;
  u32 reserved;
};

constexpr u32 kCodecAt9 = 1;
constexpr u32 kInstanceFlag2ChS16 = 2 | (0 << 7);
constexpr u64 kFlagStream = 1ull << 47;
constexpr u64 kFlagMultipleFrames = 1ull << 12;
constexpr u64 kFlagControlInitialize = 1ull << 14;

u32 IoctlResult(const std::array<u8, 40>& block) {
  u32 v;
  std::memcpy(&v, block.data(), 4);
  return v;
}

TEST(AjmDevice, InitializeAndInstanceCreate) {
  kern::ObjectTable table;
  auto& device = *new kern::AjmDevice(table);

  std::array<u8, 40> init{};
  ASSERT_EQ(device.Ioctl(0xC02889C8, init.data()), 0);
  EXPECT_EQ(IoctlResult(init), 0u);

  std::array<u8, 40> create{};
  const u64 flags = kInstanceFlag2ChS16;
  std::memcpy(create.data() + 8, &flags, 8);
  u32 codec = kCodecAt9;
  std::memcpy(create.data() + 0x10, &codec, 4);
  ASSERT_EQ(device.Ioctl(0xC0288905, create.data()), 0);
  EXPECT_EQ(IoctlResult(create), 0u);
  u32 instance;
  std::memcpy(&instance, create.data() + 4, 4);
  EXPECT_NE(instance, 0u);

  std::array<u8, 40> bad{};
  codec = 99;
  std::memcpy(bad.data() + 0x10, &codec, 4);
  ASSERT_EQ(device.Ioctl(0xC0288905, bad.data()), 0);
  EXPECT_EQ(IoctlResult(bad), 0x80930005u);

  std::array<u8, 40> destroy{};
  std::memcpy(destroy.data() + 8, &instance, 4);
  ASSERT_EQ(device.Ioctl(0xC0288906, destroy.data()), 0);
  EXPECT_EQ(IoctlResult(destroy), 0u);
}

// Two jobs in one batch: a control job that initializes the AT9 decoder from
// config data (sideband result must read 0), then a run job whose zeroed
// bitstream fails to decode (deterministic codec error written to the
// sideband result, with the stream/mframe blocks still laid out).
TEST(AjmDevice, BatchStartDecodesAt9Frame) {
  kern::ObjectTable table;
  auto& device = *new kern::AjmDevice(table);

  std::array<u8, 40> create{};
  const u64 flags = kInstanceFlag2ChS16;
  std::memcpy(create.data() + 8, &flags, 8);
  u32 codec = kCodecAt9;
  std::memcpy(create.data() + 0x10, &codec, 4);
  ASSERT_EQ(device.Ioctl(0xC0288905, create.data()), 0);
  u32 instance;
  std::memcpy(&instance, create.data() + 4, 4);
  ASSERT_EQ(instance, (kCodecAt9 << 14) | 1);

  // Stereo 48 kHz-class config (valid: 2 ch, 4-byte superframes, 4 frames,
  // 64 samples each).
  const u8 config[4] = {0xFE, 0x12, 0x00, 0x12};
  u8 config_params[8];
  std::memcpy(config_params, config, 4);
  std::memset(config_params + 4, 0, 4);

  u8 init_result[8]{};
  std::vector<u8> bitstream;
  BatchWriter writer(bitstream);
  // Job 1 (control, 48 bytes).
  writer.JobHeader(instance, 8 + 16 + 8 + 16);
  writer.SidebandIn(config_params, sizeof(config_params));
  writer.Control(kFlagControlInitialize);
  writer.SidebandOut(init_result, sizeof(init_result));
  // Job 2 (run, 64 bytes).
  u8 frame[16]{};
  std::vector<u8> pcm(2 * 256 * 2, 0xAA);
  u8 run_result[32]{};
  writer.JobHeader(instance, 8 + 16 + 8 + 16 + 16);
  writer.Input(frame, sizeof(frame));
  writer.Run(kFlagStream | kFlagMultipleFrames);
  writer.Output(pcm.data(), (u32)pcm.size());
  writer.SidebandOut(run_result, sizeof(run_result));

  std::array<u8, 40> start{};
  const u64 buffer_ptr = (u64)bitstream.data();
  std::memcpy(start.data() + 8, &buffer_ptr, 8);
  const u64 offset = bitstream.size();
  std::memcpy(start.data() + 0x10, &offset, 4);
  ASSERT_EQ(device.Ioctl(0xC0288907, start.data()), 0);
  EXPECT_EQ(IoctlResult(start), 0u);

  // The initialize job succeeded.
  u32 result;
  std::memcpy(&result, init_result, 4);
  EXPECT_EQ(result, 0u);

  // The decode job failed on the zeroed frame but still filled the sideband:
  // result, stream {consumed, produced, total}, mframe.
  std::memcpy(&result, run_result, 4);
  EXPECT_EQ(result, 0xC0000000u);  // SCE_AJM_RESULT_CODEC_ERROR | FATAL
  int32_t consumed, produced;
  std::memcpy(&consumed, run_result + 8, 4);
  std::memcpy(&produced, run_result + 12, 4);
  EXPECT_EQ(consumed, 0);
  EXPECT_EQ(produced, 0);
  u32 frames;
  std::memcpy(&frames, run_result + 24, 4);
  EXPECT_EQ(frames, 0u);

  std::array<u8, 40> wait{};
  ASSERT_EQ(device.Ioctl(0xC0288908, wait.data()), 0);
  EXPECT_EQ(IoctlResult(wait), 0u);
}

// A malformed command word must report the malformed-batch error.
TEST(AjmDevice, MalformedBatchIsRejected) {
  kern::ObjectTable table;
  auto& device = *new kern::AjmDevice(table);

  std::vector<u8> bitstream(16, 0);
  BatchWriter writer(bitstream);
  writer.JobHeader(1, 16);
  bitstream[8] = 0x05;  // command type 5 does not exist

  std::array<u8, 40> start{};
  const u64 buffer_ptr = (u64)bitstream.data();
  std::memcpy(start.data() + 8, &buffer_ptr, 8);
  const u64 offset = bitstream.size();
  std::memcpy(start.data() + 0x10, &offset, 4);
  ASSERT_EQ(device.Ioctl(0xC0288907, start.data()), 0);
  EXPECT_EQ(IoctlResult(start), 0x80930011u);
}

}  // namespace
