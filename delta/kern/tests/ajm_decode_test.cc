#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <vector>

#include "kern/object_table.h"
#include "kern/ps4/dev/ajm_dev.h"

namespace {
// Mono 48 kHz, one 64-byte frame per superframe. Ten scale factors of 25,
// gradient 17, and one nonzero spectral coefficient, without Huffman coding.
constexpr std::array<u8, 8> kConfig{0xfe, 0x70, 0x07, 0xe0};
constexpr std::array<u8, 64> kFrame{0x00, 0x00, 0x04, 0x62, 0x07, 0xce,
                                    0x73, 0x9c, 0xe7, 0x39, 0xce, 0x50};
constexpr u64 kStream = 1ull << 47;
constexpr u64 kMultipleFrames = 1ull << 12;

struct StreamResult {
  u32 result, internal;
  i32 consumed, produced;
  u64 samples;
  u32 frames, reserved;
};
static_assert(sizeof(StreamResult) == 32);

class AjmDecodeTest : public testing::Test {
 protected:
  void SetUp() override {
    device = new kern::AjmDevice(objects);
    std::array<u8, 40> command{};
    u32 codec = 1;
    std::memcpy(command.data() + 8, &codec, 4);
    Ioctl(0xc0288903, command);
    command.fill(0);
    const u64 flags = 1;
    std::memcpy(command.data() + 8, &flags, 8);
    std::memcpy(command.data() + 16, &codec, 4);
    Ioctl(0xc0288905, command);
    std::memcpy(&instance, command.data() + 4, 4);
    ASSERT_NE(instance, 0u);
  }
  void Ioctl(u32 code, std::array<u8, 40>& command) {
    ASSERT_EQ(device->Ioctl(code, command.data()), 0);
    u32 result;
    std::memcpy(&result, command.data(), 4);
    ASSERT_EQ(result, 0u);
  }
  void Append(u32 value) {
    const auto* bytes = reinterpret_cast<const u8*>(&value);
    batch.insert(batch.end(), bytes, bytes + sizeof(value));
  }
  void Buffer(u32 type, const void* data, u32 size) {
    Append(type);
    Append(size);
    const u64 address = reinterpret_cast<u64>(data);
    Append(static_cast<u32>(address));
    Append(static_cast<u32>(address >> 32));
  }
  void Flags(u32 type, u64 flags) {
    Append(type | static_cast<u32>((flags >> 26) & 0x3fffc0));
    Append(static_cast<u32>(flags));
  }
  void Start() {
    const u32 size = batch.size() - 8;
    std::memcpy(batch.data() + 4, &size, 4);
    std::array<u8, 40> command{};
    const u64 address = reinterpret_cast<u64>(batch.data());
    const u32 span = batch.size();
    std::memcpy(command.data() + 8, &address, 8);
    std::memcpy(command.data() + 16, &span, 4);
    Ioctl(0xc0288907, command);
    batch.clear();
  }
  void Header() {
    Append(instance << 6);
    Append(0);
  }
  void Control(u64 flags, const void* data = nullptr, u32 size = 0) {
    std::array<u32, 2> result{77, 77};
    Header();
    if (data)
      Buffer(2, data, size);
    Flags(3, flags);
    Buffer(0x12, result.data(), sizeof(result));
    Start();
    ASSERT_EQ(result[0], 0u);
    ASSERT_EQ(result[1], 0u);
  }
  void Initialize() { Control(1ull << 14, kConfig.data(), kConfig.size()); }
  StreamResult Decode(const void* input,
                      u32 input_size,
                      void* pcm,
                      u32 capacity) {
    StreamResult result;
    std::memset(&result, 0xa5, sizeof(result));
    Header();
    Buffer(1, input, input_size);
    Flags(4, kStream | kMultipleFrames);
    Buffer(0x11, pcm, capacity);
    Buffer(0x12, &result, sizeof(result));
    Start();
    return result;
  }
  void CheckPcm(const i16* pcm) {
    const i16 expected[] = {0, 12, 39, 54, -2, -123, -225, -297};
    for (u32 i = 0; i < 8; ++i)
      EXPECT_NEAR(pcm[i * 32], expected[i], 1);
  }
  kern::ObjectTable objects;
  kern::AjmDevice* device = nullptr;
  u32 instance = 0;
  std::vector<u8> batch;
};

TEST_F(AjmDecodeTest, ValidFrameDecodesNonzeroPcmAndResetRestartsHistory) {
  Initialize();
  std::array<i16, 257> pcm;
  pcm.fill(0x5a5a);
  const auto result = Decode(kFrame.data(), kFrame.size(), pcm.data(), 512);
  ASSERT_EQ(result.result, 0u);
  EXPECT_EQ(result.internal, 0u);
  EXPECT_EQ(result.consumed, 64);
  EXPECT_EQ(result.produced, 512);
  EXPECT_EQ(result.samples, 256u);
  EXPECT_EQ(result.frames, 1u);
  EXPECT_EQ(pcm[256], 0x5a5a);
  CheckPcm(pcm.data());
  Control(1ull << 13);
  std::array<i16, 257> repeated;
  repeated.fill(0x5a5a);
  const auto reset = Decode(kFrame.data(), kFrame.size(), repeated.data(), 512);
  EXPECT_EQ(reset.result, 0u);
  EXPECT_EQ(reset.samples, 256u);
  EXPECT_EQ(pcm, repeated);
}

TEST_F(AjmDecodeTest, PartialInputDoesNotConsumeOrWriteAndCanBeRetried) {
  Initialize();
  std::array<i16, 257> pcm;
  pcm.fill(0x5a5a);
  const auto result = Decode(kFrame.data(), 63, pcm.data(), 512);
  EXPECT_EQ(result.result, 8u);
  EXPECT_EQ(result.consumed, 0);
  EXPECT_EQ(result.produced, 0);
  EXPECT_EQ(result.samples, 0u);
  EXPECT_EQ(result.frames, 0u);
  for (auto sample : pcm)
    EXPECT_EQ(sample, 0x5a5a);
  const auto retry = Decode(kFrame.data(), kFrame.size(), pcm.data(), 512);
  EXPECT_EQ(retry.result, 0u);
  EXPECT_EQ(retry.samples, 256u);
  CheckPcm(pcm.data());
  EXPECT_EQ(pcm[256], 0x5a5a);
}

TEST_F(AjmDecodeTest, SmallOutputDoesNotAdvanceDecoderAndCanBeRetried) {
  Initialize();
  std::array<i16, 257> pcm;
  pcm.fill(0x5a5a);
  const auto result = Decode(kFrame.data(), kFrame.size(), pcm.data(), 511);
  EXPECT_EQ(result.result, 16u);
  EXPECT_EQ(result.consumed, 0);
  EXPECT_EQ(result.produced, 0);
  EXPECT_EQ(result.samples, 0u);
  EXPECT_EQ(result.frames, 0u);
  for (auto sample : pcm)
    EXPECT_EQ(sample, 0x5a5a);
  const auto retry = Decode(kFrame.data(), kFrame.size(), pcm.data(), 512);
  EXPECT_EQ(retry.result, 0u);
  EXPECT_EQ(retry.samples, 256u);
  CheckPcm(pcm.data());
  EXPECT_EQ(pcm[256], 0x5a5a);
}

TEST_F(AjmDecodeTest, MultipleSuperframesReportConsumedPaddingAndSampleCount) {
  Initialize();
  std::array<u8, 192> input;
  for (u32 i = 0; i < 3; ++i)
    std::memcpy(input.data() + i * 64, kFrame.data(), kFrame.size());
  std::array<i16, 769> pcm;
  pcm.fill(0x5a5a);
  const auto result = Decode(input.data(), input.size(), pcm.data(), 1536);
  ASSERT_EQ(result.result, 0u);
  EXPECT_EQ(result.consumed, 192);
  EXPECT_EQ(result.produced, 1536);
  EXPECT_EQ(result.samples, 768u);
  EXPECT_EQ(result.frames, 3u);
  CheckPcm(pcm.data());
  EXPECT_EQ(pcm[768], 0x5a5a);
}
}  // namespace
