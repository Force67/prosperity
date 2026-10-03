#include <gtest/gtest.h>

#include <SDL3/SDL.h>

#include "guest/pause.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "memory_debug/memory_debug.h"
#include "ui/input_sdl.h"
#include "ui/mouse_look.h"
#include "ui/overlay.h"
#include "ui/overlay_memory.h"

namespace {
class MemoryOverlayTest : public testing::Test {
 protected:
  void SetUp() override {
    ui::OverlayEnsureImGui();
    unsigned char* pixels;
    int width, height;
    ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    ui::ConfigureMouseLook({false, 100, false});
    ui::SetMemoryOverlayMode(ui::MemoryOverlayMode::kOff);
  }
  void TearDown() override { ui::OverlayShutdownImGui(); }
  void Frame(u32 width = 1280, u32 height = 720) {
    ui::OverlayBuildFrame(width, height, 0, 0);
  }
  void Key(SDL_Scancode key, bool pressed = true, bool repeat = false) {
    SDL_Event event{};
    event.type = pressed ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
    event.key.scancode = key;
    event.key.repeat = repeat;
    ui::ProcessEvent(event, nullptr, 1280, 720);
  }
};

#if defined(DELTA_MEMORY_DEBUG)
TEST_F(MemoryOverlayTest, F4CyclesAndEscapeIsConsumedUntilRelease) {
  Key(SDL_SCANCODE_F4);
  EXPECT_EQ(ui::GetMemoryOverlayMode(), ui::MemoryOverlayMode::kCompact);
  Key(SDL_SCANCODE_F4, true, true);
  EXPECT_EQ(ui::GetMemoryOverlayMode(), ui::MemoryOverlayMode::kCompact);
  Key(SDL_SCANCODE_F4);
  EXPECT_TRUE(ui::MemoryOverlayBlocksInput());
  Key(SDL_SCANCODE_ESCAPE);
  EXPECT_EQ(ui::GetMemoryOverlayMode(), ui::MemoryOverlayMode::kCompact);
  EXPECT_TRUE(ui::MemoryOverlayBlocksInput());
  Key(SDL_SCANCODE_ESCAPE, false);
  EXPECT_FALSE(ui::MemoryOverlayBlocksInput());
  Key(SDL_SCANCODE_F4);
  Key(SDL_SCANCODE_F4);
  EXPECT_EQ(ui::GetMemoryOverlayMode(), ui::MemoryOverlayMode::kOff);
}

TEST_F(MemoryOverlayTest, FullViewFitsAllBucketsAndNarrowViewScrolls) {
  memory_debug::Record image;
  image.Set(memory_debug::Bucket::kTextures, 1024 * 1024, 0);
  memory_debug::Record pool;
  pool.Set(memory_debug::Bucket::kImagePool, 0, 4 * 1024 * 1024);
  ui::SetMemoryOverlayMode(ui::MemoryOverlayMode::kFull);
  Frame();
  Frame();
  auto* full = ImGui::FindWindowByName("##memory_full");
  ASSERT_NE(full, nullptr);
  ASSERT_EQ(full->DC.ChildWindows.size(), 1);
  EXPECT_LE(full->DC.ChildWindows[0]->ScrollMax.y, 1);
  EXPECT_EQ(full->DC.ChildWindows[0]->ScrollMax.x, 0);
  EXPECT_GT(ImGui::GetDrawData()->TotalVtxCount, 1000);
  EXPECT_EQ(ImGui::FindWindowByName("Inspect bucket"), nullptr);
  Frame(640, 480);
  Frame(640, 480);
  EXPECT_GT(full->DC.ChildWindows[0]->ScrollMax.y, 0);
  EXPECT_EQ(full->DC.ChildWindows[0]->ScrollMax.x, 0);
}

TEST_F(MemoryOverlayTest, EscapeFromFullDoesNotPauseTheGuest) {
  ui::SetMemoryOverlayMode(ui::MemoryOverlayMode::kFull);
  Frame();
  Key(SDL_SCANCODE_ESCAPE);
  Frame();
  EXPECT_NE(ImGui::FindWindowByName("##memory_dock"), nullptr);
  EXPECT_FALSE(guest::Paused());
}

TEST_F(MemoryOverlayTest, ControllerCloseIsConsumedUntilRelease) {
  ui::SetMemoryOverlayMode(ui::MemoryOverlayMode::kFull);
  SDL_Event event{};
  event.type = SDL_EVENT_GAMEPAD_BUTTON_DOWN;
  event.gbutton.button = SDL_GAMEPAD_BUTTON_EAST;
  ui::ProcessEvent(event, nullptr, 1280, 720);
  EXPECT_EQ(ui::GetMemoryOverlayMode(), ui::MemoryOverlayMode::kCompact);
  EXPECT_TRUE(ui::MemoryOverlayBlocksInput());
  Key(SDL_SCANCODE_ESCAPE, false);
  EXPECT_TRUE(ui::MemoryOverlayBlocksInput());
  event.type = SDL_EVENT_GAMEPAD_BUTTON_UP;
  ui::ProcessEvent(event, nullptr, 1280, 720);
  EXPECT_FALSE(ui::MemoryOverlayBlocksInput());
}

TEST_F(MemoryOverlayTest, FocusLossClearsDismissalHeldWithoutKeyUp) {
  ui::SetMemoryOverlayMode(ui::MemoryOverlayMode::kFull);
  Key(SDL_SCANCODE_ESCAPE);
  EXPECT_TRUE(ui::MemoryOverlayBlocksInput());
  SDL_Event event{};
  event.type = SDL_EVENT_WINDOW_FOCUS_LOST;
  ui::ProcessEvent(event, nullptr, 1280, 720);
  EXPECT_FALSE(ui::MemoryOverlayBlocksInput());
}
#else
TEST_F(MemoryOverlayTest, DisabledBuildHasNoOverlay) {
  Key(SDL_SCANCODE_F4);
  EXPECT_EQ(ui::GetMemoryOverlayMode(), ui::MemoryOverlayMode::kOff);
  Frame();
  EXPECT_EQ(ImGui::FindWindowByName("##memory_full"), nullptr);
}
#endif
}  // namespace
