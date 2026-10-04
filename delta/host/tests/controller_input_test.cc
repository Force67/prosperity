#include <gtest/gtest.h>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "base/option.h"
#include "base/strings/format.h"
#include "base/threading/thread.h"
#include "guest/session.h"
#include "host/controller_input.h"
#include "host/window.h"

namespace {
host::PadKeys g_keys;
bool g_gameplay = false;

void SetOption(const char* name, const char* value) {
  bool found = false;
  base::OptionBase::VisitAll([&](const base::OptionBase* option) {
    if (std::strcmp(option->name(), name) == 0) {
      found = true;
      EXPECT_TRUE(const_cast<base::OptionBase*>(option)->SetFromString(value));
    }
  });
  ASSERT_TRUE(found) << name;
}

class ControllerInputTest : public testing::Test {
 protected:
  void SetUp() override {
    RestartSession();
    g_keys = {};
    g_gameplay = false;
    base::OptionBase::VisitAll([](const base::OptionBase* option) {
      const_cast<base::OptionBase*>(option)->Reset();
    });
  }
  void TearDown() override { RestartSession(); }
  void RestartSession() {
    guest::RequestStop();
    guest::JoinThreads();
    guest::ResetSessionResources();
    guest::BeginSession();
  }
  guest::ControllerState Read() {
    guest::ControllerState state;
    EXPECT_TRUE(host::PollControllerInput(state));
    return state;
  }
};

TEST_F(ControllerInputTest, MapsHostInputAndAdvancesTimestamp) {
  auto neutral = Read();
  EXPECT_EQ(neutral.buttons, 0u);
  EXPECT_EQ(neutral.lx, 128u);
  EXPECT_EQ(neutral.ly, 128u);
  EXPECT_EQ(neutral.rx, 128u);
  EXPECT_EQ(neutral.ry, 128u);
  g_keys.cross = g_keys.l2 = true;
  g_keys.lx = 255;
  g_keys.ry = 0;
  base::SleepForMilliseconds(10);
  auto state = Read();
  EXPECT_EQ(state.buttons, 0x4100u);
  EXPECT_EQ(state.l2, 255u);
  EXPECT_EQ(state.r2, 0u);
  EXPECT_EQ(state.lx, 255u);
  EXPECT_EQ(state.ry, 0u);
  EXPECT_GE(state.timestamp - neutral.timestamp, 1000u);
}

TEST_F(ControllerInputTest, ReadScriptRestartsAndReloadsForEachSession) {
  SetOption("DELTA_PAD_AUTOSKIP", "true");
  SetOption("DELTA_PAD_SCRIPT", "cross:2,none:1");
  EXPECT_EQ(Read().buttons, 0x4000u);
  EXPECT_EQ(Read().buttons, 0x4000u);
  EXPECT_EQ(Read().buttons, 0u);
  RestartSession();
  EXPECT_EQ(Read().buttons, 0x4000u);
  RestartSession();
  SetOption("DELTA_PAD_SCRIPT", "square:1");
  EXPECT_EQ(Read().buttons, 0x8000u);
  EXPECT_EQ(Read().buttons, 0u);
}

TEST_F(ControllerInputTest, ConcurrentReadersConsumeOneSharedScriptSequence) {
  SetOption("DELTA_PAD_AUTOSKIP", "true");
  SetOption("DELTA_PAD_SCRIPT", "cross:64,none:64");
  std::atomic<u32> presses{0};
  std::vector<std::thread> readers;
  for (int i = 0; i < 8; ++i) {
    readers.emplace_back([&] {
      for (int j = 0; j < 16; ++j) {
        guest::ControllerState state;
        host::PollControllerInput(state);
        if (state.buttons == 0x4000)
          ++presses;
      }
    });
  }
  for (auto& reader : readers)
    reader.join();
  EXPECT_EQ(presses, 64u);
  EXPECT_EQ(Read().buttons, 0u);
}

TEST_F(ControllerInputTest, TimeScriptClockRestartsAndReloadsForEachSession) {
  SetOption("DELTA_PAD_SCRIPT", "0:lsright:30");
  EXPECT_EQ(Read().lx, 255u);
  base::SleepForMilliseconds(50);
  EXPECT_EQ(Read().lx, 128u);
  RestartSession();
  EXPECT_EQ(Read().lx, 255u);
  RestartSession();
  SetOption("DELTA_PAD_SCRIPT", "0:cross+rsup:10000");
  auto state = Read();
  EXPECT_EQ(state.buttons, 0x4000u);
  EXPECT_EQ(state.lx, 128u);
  EXPECT_EQ(state.ry, 0u);
}

TEST_F(ControllerInputTest, ExplorationRestartsFromFirstGameplaySample) {
  SetOption("DELTA_PAD_AUTOSKIP", "true");
  SetOption("DELTA_PAD_EXPLORE", "true");
  SetOption("DELTA_PAD_EXPLORE_READS", "2");
  g_gameplay = true;
  EXPECT_EQ(Read().lx, 255u);
  EXPECT_EQ(Read().lx, 255u);
  EXPECT_EQ(Read().lx, 128u);
  RestartSession();
  SetOption("DELTA_PAD_EXPLORE_DIR", "1");
  EXPECT_EQ(Read().lx, 0u);
  EXPECT_EQ(Read().lx, 0u);
  EXPECT_EQ(Read().lx, 128u);
}

TEST_F(ControllerInputTest, ExperimentsStartOncePerSessionAndRestartAfterJoin) {
  u64 watched = 42;
  const auto address = base::Format("{:#x}", reinterpret_cast<u64>(&watched));
  SetOption("DELTA_MEMWATCH", address.c_str());
  for (int session = 0; session < 2; ++session) {
    host::OpenControllerInput(1, 0, 0);
    EXPECT_EQ(guest::LiveThreads(), 1u);
    Read();
    host::OpenControllerInput(1, 0, 0);
    EXPECT_EQ(guest::LiveThreads(), 1u);
    RestartSession();
    EXPECT_EQ(guest::LiveThreads(), 0u);
  }
}
}  // namespace

namespace host {
bool PollKeyboardPad(PadKeys& out) {
  out = g_keys;
  return true;
}
bool InGameplay() {
  return g_gameplay;
}
}  // namespace host
