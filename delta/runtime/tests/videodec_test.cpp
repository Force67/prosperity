#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <vector>
#include "runtime/media/videodec.h"
#include "runtime/vprx/vprx.h"
#include "video_fixture.h"
using namespace runtime::video;

TEST(VideoDecode, ReordersPixelsAndMetadataDrainsAndResets) {
  Config config{};
  config.size = sizeof(config); config.codec = 1;
  config.width = 64; config.height = 48;
  Memory memory{}; memory.size = sizeof(memory);
  ASSERT_EQ(QueryMemory(&config, &memory), 0);
  void* decoder = nullptr;
  ASSERT_EQ(Create(&config, &memory, &decoder), 0);
  struct Cleanup { void* handle; ~Cleanup() { Delete(handle); } } cleanup{decoder};
  std::vector<u8> pixels(memory.frame_bytes + 16, 0xcd);
  constexpr unsigned sizes[] = {698,16,11,12,26,12};
  constexpr unsigned order[] = {0,3,1,2,5,4};
  for (unsigned cycle = 0; cycle < 2; ++cycle) {
    unsigned received = 0, offset = 0;
    auto check = [&](const Output& out) {
      if (!out.valid) return;
      EXPECT_EQ(out.width, 64u); EXPECT_EQ(out.pitch, 64u); EXPECT_EQ(out.height, 48u);
      EXPECT_EQ(out.bytes, 4608u);
      for (unsigned i = 0; i < 64 * 48; ++i)
        ASSERT_NEAR(pixels[i], 32 + received * 16, 2);
      for (unsigned i = 64 * 48; i < out.bytes; i += 2) {
        ASSERT_NEAR(pixels[i], 96, 2); ASSERT_NEAR(pixels[i+1], 160, 2);
      }
      std::array<u64, 16> info{}; info[0] = 0x78; info[15] = 0xdeadbeef;
      EXPECT_EQ(Picture(&out, info.data(), nullptr), 0);
      EXPECT_EQ(info[0], 0x78u); EXPECT_EQ(info[2], received * 512u);
      EXPECT_EQ(info[4], 100u + received); EXPECT_EQ(info[15], 0xdeadbeefu);
      for (unsigned i = out.bytes; i < pixels.size(); ++i) ASSERT_EQ(pixels[i], 0xcd);
      ++received;
    };
    for (unsigned i = 0; i < 6; ++i) {
      Input input{sizeof(Input), kVideoFixture + offset, sizes[i], order[i] * 512u,
                  u64(i64(i) * 512 - 1024), 100u + order[i]};
      FrameBuffer target{sizeof(FrameBuffer), pixels.data(), memory.frame_bytes, false};
      Output out{}; out.size = cycle ? 0x30 : sizeof(out);
      out.format = 0x12345678; out.pitch_bytes = 0x87654321;
      ASSERT_EQ(Decode(decoder, &input, &target, &out), 0);
      EXPECT_EQ(target.accepted, bool(out.valid));
      if (cycle) { EXPECT_EQ(out.format, 0x12345678u); EXPECT_EQ(out.pitch_bytes, 0x87654321u); }
      check(out); offset += sizes[i];
    }
    unsigned drained = 0;
    for (unsigned i = 0; i < 10; ++i) {
      FrameBuffer target{sizeof(FrameBuffer), pixels.data(), memory.frame_bytes, false};
      Output out{}; out.size = sizeof(out);
      ASSERT_EQ(Flush(decoder, &target, &out), 0);
      if (!out.valid) break;
      ++drained; check(out);
    }
    EXPECT_GT(drained, 0u); EXPECT_EQ(received, 6u);
    ASSERT_EQ(Reset(decoder), 0);
  }
}

TEST(VideoDecode, RejectsInvalidArgumentsAndExportsDecoder) {
  EXPECT_NE(QueryMemory(nullptr, nullptr), 0);
  EXPECT_NE(Decode(nullptr, nullptr, nullptr, nullptr), 0);
  EXPECT_NE(Delete(nullptr), 0);
  runtime::vprx_init();
  EXPECT_NE(runtime::vprx_get_forced("libSceVideodec2", 0xF39D85E7EABAFA23ull), 0u);
}
