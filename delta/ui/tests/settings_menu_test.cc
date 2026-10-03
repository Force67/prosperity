#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

#include "base/environment_variables.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "options/options.h"
#include "ui/home_screen.h"
#include "ui/mouse_look.h"
#include "ui/overlay.h"
#include "ui/settings.h"

namespace {
class SettingsMenuTest : public testing::Test {
 protected:
  void SetUp() override {
    char directory[] = "/tmp/prosperity-settings-ui-XXXXXX";
    ASSERT_NE(::mkdtemp(directory), nullptr);
    root_ = directory;
    had_config_ =
        base::GetEnvironmentVariable(u8"XDG_CONFIG_HOME", old_config_);
    ::setenv("XDG_CONFIG_HOME", root_.c_str(), 1);
    ui::BeginHomeScreen(games_, true, true);
    ui::ConfigureSettings({"vulkan", "opengl"}, {"Test GPU"});
    Frame();
  }
  void TearDown() override {
    ui::EndHomeScreen();
    ui::OverlayShutdownImGui();
    if (had_config_)
      ::setenv("XDG_CONFIG_HOME",
               reinterpret_cast<const char*>(old_config_.c_str()), 1);
    else
      ::unsetenv("XDG_CONFIG_HOME");
    std::filesystem::remove_all(root_.c_str());
  }
  void Frame(u32 width = 1280, u32 height = 720) {
    ui::OverlayBuildFrame(width, height, 0, 0);
  }
  void Click(ImVec2 pos) {
    auto& io = ImGui::GetIO();
    io.AddMousePosEvent(pos.x, pos.y);
    Frame();
    io.AddMouseButtonEvent(0, true);
    Frame();
    io.AddMouseButtonEvent(0, false);
    Frame();
    Frame();
  }
  void Key(ImGuiKey key) {
    ImGui::GetIO().AddKeyEvent(key, true);
    Frame();
    ImGui::GetIO().AddKeyEvent(key, false);
    Frame();
  }
  base::Vector<ui::HomeGame> games_;
  base::String root_;
  base::StringU8 old_config_;
  bool had_config_ = false;
};

TEST_F(SettingsMenuTest, GearOpensSettingsAndEscapeProtectsUnsavedChanges) {
  Click(ImVec2(1280 - 64 - 20, 42));
  ASSERT_TRUE(ui::SettingsVisible());
  auto* panel = ImGui::FindWindowByName("##settings_lounge");
  ASSERT_NE(panel, nullptr);
  ASSERT_FALSE(panel->DC.ChildWindows.empty());
  auto* body = panel->DC.ChildWindows[0];
  const float card_width = (body->Size.x - 24) / 3;
  Click(ImVec2(body->Pos.x + card_width + 12 + card_width * 0.5f,
               body->Pos.y + 75));
  Key(ImGuiKey_Escape);
  EXPECT_TRUE(ui::SettingsVisible());
  EXPECT_TRUE(
      ImGui::IsPopupOpen("##settings_unsaved", ImGuiPopupFlags_AnyPopupId));
  EXPECT_FALSE(ui::HomeScreenDone());
  Key(ImGuiKey_Escape);
  EXPECT_TRUE(ui::SettingsVisible());
  EXPECT_FALSE(
      ImGui::IsPopupOpen("##settings_unsaved", ImGuiPopupFlags_AnyPopupId));
  panel = ImGui::FindWindowByName("##settings_lounge");
  Click(ImVec2(panel->Pos.x + panel->Size.x - 108,
               panel->Pos.y + panel->Size.y - 42));
  EXPECT_TRUE(
      std::filesystem::exists((root_ + "/prosperity/settings.txt").c_str()));
  Key(ImGuiKey_Escape);
  EXPECT_FALSE(ui::SettingsVisible());
  EXPECT_FALSE(ui::HomeScreenDone());
  Key(ImGuiKey_S);
  EXPECT_TRUE(ui::SettingsVisible());
  Frame(640, 480);
  Frame(640, 480);
  panel = ImGui::FindWindowByName("##settings_lounge");
  EXPECT_GE(panel->Pos.x, 0);
  EXPECT_GE(panel->Pos.y, 0);
  EXPECT_LE(panel->Pos.x + panel->Size.x, 640);
  EXPECT_LE(panel->Pos.y + panel->Size.y, 480);
}

TEST_F(SettingsMenuTest, MouseLookPreferencesSaveAndApplyImmediately) {
  ui::SettingsOpen();
  Frame();
  auto* panel = ImGui::FindWindowByName("##settings_lounge");
  ASSERT_NE(panel, nullptr);
  const float tab_width = (panel->Size.x - 56 - 12) / 4;
  Click(ImVec2(panel->Pos.x + 28 + 2 * (tab_width + 4) + tab_width * 0.5f,
               panel->Pos.y + 118));
  auto* body = panel->DC.ChildWindows[0];
  const ImVec2 origin(body->Pos.x + body->WindowPadding.x,
                      body->Pos.y + body->WindowPadding.y);
  Click(ImVec2(origin.x + body->Size.x - 224 + 20, origin.y + 28));
  Click(ImVec2(origin.x + body->Size.x - 224 + 20, origin.y + 56 * 2 + 28));
  Click(ImVec2(panel->Pos.x + panel->Size.x - 108,
               panel->Pos.y + panel->Size.y - 42));
  const auto settings = ui::GetMouseLookSettings();
  EXPECT_TRUE(settings.enabled);
  EXPECT_TRUE(settings.invert_y);
  EXPECT_EQ(settings.sensitivity, 100u);
  Frame(640, 480);
  Frame(640, 480);
  EXPECT_LE(panel->Pos.y + panel->Size.y, 480);
  ui::ConfigureMouseLook({});
}
}  // namespace
