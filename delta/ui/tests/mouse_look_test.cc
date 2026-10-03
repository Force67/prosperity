#include <SDL3/SDL.h>
#include <gtest/gtest.h>

#include "ui/home_screen.h"
#include "ui/input_sdl.h"
#include "ui/mouse_look.h"
#include "ui/overlay.h"
#include "ui/pause_menu.h"

namespace {
class MouseLookTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!SDL_Init(SDL_INIT_VIDEO))
      GTEST_SKIP() << "SDL display unavailable: " << SDL_GetError();
    window_ = SDL_CreateWindow("Mouse look test", 640, 480, 0);
    ASSERT_NE(window_, nullptr);
    SDL_RaiseWindow(window_);
    for (int i = 0; i < 50; ++i) {
      SDL_PumpEvents();
      if (SDL_GetWindowFlags(window_) & SDL_WINDOW_INPUT_FOCUS)
        break;
      SDL_Delay(10);
    }
    if (!(SDL_GetWindowFlags(window_) & SDL_WINDOW_INPUT_FOCUS))
      GTEST_SKIP() << "A focused SDL window is required.";
    SDL_FlushEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST);
    ui::OverlayEnsureImGui();
    ready_ = true;
    ui::ConfigureMouseLook({true, 100, false});
    ui::SyncMouseLook(window_);
    ASSERT_TRUE(SDL_GetWindowRelativeMouseMode(window_));
  }
  void TearDown() override {
    if (ready_) {
      ui::EndHomeScreen();
      ui::ResetMouseLook(window_);
      ui::ConfigureMouseLook({});
      ui::OverlayShutdownImGui();
    }
    SDL_DestroyWindow(window_);
    SDL_Quit();
  }
  void Move(float x, float y) {
    SDL_Event event{};
    event.type = SDL_EVENT_MOUSE_MOTION;
    event.motion.xrel = x;
    event.motion.yrel = y;
    ui::ProcessMouseLookEvent(event);
    ui::UpdateMouseLook();
  }
  void Button(u8 button, bool down) {
    SDL_Event event{};
    event.type = down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
    event.button.button = button;
    ui::ProcessMouseLookEvent(event);
  }
  bool Poll() { return ui::PollMouseLook(x_, y_, left_, right_); }
  base::Vector<ui::HomeGame> games_;
  SDL_Window* window_ = nullptr;
  bool ready_ = false;
  u8 x_ = 128, y_ = 128;
  bool left_ = false, right_ = false;
};

TEST_F(MouseLookTest, MotionPersistsAcrossGuestReadsAndReturnsToNeutral) {
  Move(3, -2);
  ASSERT_TRUE(Poll());
  EXPECT_GT(x_, 128);
  EXPECT_LT(y_, 128);
  const auto x = x_, y = y_;
  ASSERT_TRUE(Poll());
  EXPECT_EQ(x_, x);
  EXPECT_EQ(y_, y);
  ui::UpdateMouseLook();
  ASSERT_TRUE(Poll());
  EXPECT_EQ(x_, 128);
  EXPECT_EQ(y_, 128);
  Move(10, 10);
  SDL_Delay(60);
  ASSERT_TRUE(Poll());
  EXPECT_EQ(x_, 128);
  EXPECT_EQ(y_, 128);
}

TEST_F(MouseLookTest, SensitivityInversionAndLargeMovementStayInRange) {
  ui::ConfigureMouseLook({true, 10, true});
  Move(0.5f, 0.5f);
  ASSERT_TRUE(Poll());
  const auto low = x_;
  EXPECT_LT(y_, 128);
  ui::ConfigureMouseLook({true, 300, false});
  Move(0.5f, 0.5f);
  ASSERT_TRUE(Poll());
  EXPECT_GT(x_, low);
  EXPECT_GT(y_, 128);
  Move(10000, -10000);
  ASSERT_TRUE(Poll());
  EXPECT_EQ(x_, 255);
  EXPECT_EQ(y_, 1);
}

TEST_F(MouseLookTest, PauseAndHomeReleaseCaptureAndClearButtons) {
  Button(SDL_BUTTON_LEFT, true);
  Button(SDL_BUTTON_RIGHT, true);
  ASSERT_TRUE(Poll());
  EXPECT_TRUE(left_);
  EXPECT_TRUE(right_);
  ui::PauseMenuGameReady();
  ui::PauseMenuToggle();
  ui::SyncMouseLook(window_);
  EXPECT_FALSE(SDL_GetWindowRelativeMouseMode(window_));
  EXPECT_FALSE(Poll());
  ui::PauseMenuToggle();
  ui::SyncMouseLook(window_);
  EXPECT_TRUE(SDL_GetWindowRelativeMouseMode(window_));
  ASSERT_TRUE(Poll());
  EXPECT_FALSE(left_);
  EXPECT_FALSE(right_);
  EXPECT_EQ(x_, 128);
  ui::BeginHomeScreen({}, true, true);
  ui::SyncMouseLook(window_);
  EXPECT_FALSE(SDL_GetWindowRelativeMouseMode(window_));
  EXPECT_FALSE(Poll());
}

TEST_F(MouseLookTest, FocusLossReleasesCaptureAndClearsInput) {
  Button(SDL_BUTTON_LEFT, true);
  Move(5, 5);
  SDL_HideWindow(window_);
  for (int i = 0; i < 50; ++i) {
    SDL_PumpEvents();
    if (!(SDL_GetWindowFlags(window_) & SDL_WINDOW_INPUT_FOCUS))
      break;
    SDL_Delay(10);
  }
  ASSERT_FALSE(SDL_GetWindowFlags(window_) & SDL_WINDOW_INPUT_FOCUS);
  ui::SyncMouseLook(window_);
  EXPECT_FALSE(SDL_GetWindowRelativeMouseMode(window_));
  EXPECT_FALSE(Poll());
  SDL_ShowWindow(window_);
  SDL_RaiseWindow(window_);
  for (int i = 0; i < 50; ++i) {
    SDL_PumpEvents();
    if (SDL_GetWindowFlags(window_) & SDL_WINDOW_INPUT_FOCUS)
      break;
    SDL_Delay(10);
  }
  ui::SyncMouseLook(window_);
  ASSERT_TRUE(Poll());
  EXPECT_FALSE(left_);
  EXPECT_EQ(x_, 128);
}

TEST_F(MouseLookTest, ShortcutTogglesCaptureWithoutKeyRepeat) {
  SDL_Event event{};
  event.type = SDL_EVENT_KEY_DOWN;
  event.key.scancode = SDL_SCANCODE_F3;
  ui::ProcessEvent(event, window_, 640, 480);
  EXPECT_FALSE(ui::GetMouseLookSettings().enabled);
  EXPECT_FALSE(SDL_GetWindowRelativeMouseMode(window_));
  event.key.repeat = true;
  ui::ProcessEvent(event, window_, 640, 480);
  EXPECT_FALSE(ui::GetMouseLookSettings().enabled);
  event.key.repeat = false;
  ui::ProcessEvent(event, window_, 640, 480);
  EXPECT_TRUE(SDL_GetWindowRelativeMouseMode(window_));
}
}  // namespace
