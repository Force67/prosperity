#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "base/environment_variables.h"
#include "options/options.h"
#include "options/settings.h"

namespace {
DELTA_OPTION(const char*, kBackend, "DELTA_GPU_BACKEND", "vulkan");
DELTA_OPTION(const char*, kGpu, "DELTA_VK_GPU", nullptr);
DELTA_OPTION(const char*, kHostGpu, "DELTA_VK_GPU", nullptr);
DELTA_OPTION(const char*, kVsync, "DELTA_GPU_VSYNC", nullptr);
DELTA_OPTION(bool, kSync, "DELTA_GPU_SYNCPRESENT", false);
DELTA_OPTION(u32, kBackground, "DELTA_UI_BACKGROUND", 0);
DELTA_OPTION(bool, kPerformance, "DELTA_GPU_OVERLAY", true);
DELTA_OPTION(bool, kValidation, "DELTA_VK_VALIDATE", false);

class SettingsTest : public testing::Test {
 protected:
  void SetUp() override {
    char directory[] = "/tmp/prosperity-settings-XXXXXX";
    ASSERT_NE(::mkdtemp(directory), nullptr);
    root_ = directory;
    had_config_ =
        base::GetEnvironmentVariable(u8"XDG_CONFIG_HOME", old_config_);
    ASSERT_EQ(::setenv("XDG_CONFIG_HOME", root_.c_str(), 1), 0);
    kBackend.Reset();
    kGpu.Reset();
    kHostGpu.Reset();
    kVsync.Reset();
    kSync.Reset();
    kBackground.Reset();
    kPerformance.Reset();
    kValidation.Reset();
  }
  void TearDown() override {
    if (had_config_)
      ::setenv("XDG_CONFIG_HOME",
               reinterpret_cast<const char*>(old_config_.c_str()), 1);
    else
      ::unsetenv("XDG_CONFIG_HOME");
    std::filesystem::remove_all(root_.c_str());
  }
  base::String root_;
  base::StringU8 old_config_;
  bool had_config_ = false;
};

TEST_F(SettingsTest,
       SavesAndLoadsEveryPreferenceWithoutChangingRunningOptions) {
  options::Settings settings;
  settings.backend = "opengl";
  settings.gpu = "GPU with spaces and \"quotes\"";
  settings.vsync = "0";
  settings.async_present = false;
  settings.background = 2;
  settings.performance = false;
  settings.validation = true;
  ASSERT_TRUE(options::SaveSettings(settings));
  EXPECT_EQ(options::ReadSettings(), options::Settings{});
  ASSERT_TRUE(options::LoadFile(options::SettingsPath().c_str()));
  EXPECT_EQ(options::ReadSettings(), settings);
  EXPECT_STREQ(kGpu.get(), kHostGpu.get());
  EXPECT_FALSE(
      std::filesystem::exists((options::SettingsPath() + ".tmp").c_str()));
}

TEST_F(SettingsTest, EnvironmentAndCommandLineOverrideSavedPreferences) {
  options::Settings settings;
  settings.backend = "opengl";
  ASSERT_TRUE(options::SaveSettings(settings));
  base::StringU8 previous;
  const bool had_backend =
      base::GetEnvironmentVariable(u8"DELTA_GPU_BACKEND", previous);
  ASSERT_EQ(::setenv("DELTA_GPU_BACKEND", "d3d12", 1), 0);
  options::Init();
  EXPECT_EQ(options::ReadSettings().backend, "d3d12");
  char executable[] = "ps4delta";
  char argument[] = "+DELTA_GPU_BACKEND=vulkan";
  char* argv[] = {executable, argument, nullptr};
  int argc = 2;
  options::Init(argc, argv);
  EXPECT_EQ(options::ReadSettings().backend, "vulkan");
  EXPECT_EQ(argc, 1);
  if (had_backend)
    ::setenv("DELTA_GPU_BACKEND",
             reinterpret_cast<const char*>(previous.c_str()), 1);
  else
    ::unsetenv("DELTA_GPU_BACKEND");
}

TEST_F(SettingsTest, DefaultsClearDeviceAndVsyncOverrides) {
  options::Settings settings;
  settings.gpu = "Specific GPU";
  settings.vsync = "1";
  ASSERT_TRUE(options::SaveSettings(settings));
  ASSERT_TRUE(options::LoadFile(options::SettingsPath().c_str()));
  ASSERT_TRUE(options::SaveSettings(options::Settings{}));
  ASSERT_TRUE(options::LoadFile(options::SettingsPath().c_str()));
  EXPECT_EQ(options::ReadSettings(), options::Settings{});
  EXPECT_STREQ(kGpu.get(), "");
  EXPECT_STREQ(kHostGpu.get(), "");
  EXPECT_STREQ(kVsync.get(), "");
}

TEST_F(SettingsTest, FailedSavePreservesPreviousPreferences) {
  ASSERT_TRUE(options::SaveSettings(options::Settings{}));
  options::Settings settings;
  settings.backend = "opengl";
  std::filesystem::create_directory((options::SettingsPath() + ".tmp").c_str());
  EXPECT_FALSE(options::SaveSettings(settings));
  ASSERT_TRUE(options::LoadFile(options::SettingsPath().c_str()));
  EXPECT_EQ(options::ReadSettings(), options::Settings{});
}

TEST_F(SettingsTest, RejectsInvalidPreferencesWithoutReplacingSavedFile) {
  ASSERT_TRUE(options::SaveSettings(options::Settings{}));
  options::Settings settings;
  settings.gpu = "GPU\n+DELTA_GPU_BACKEND=opengl";
  EXPECT_FALSE(options::SaveSettings(settings));
  settings = {};
  settings.backend = "unknown";
  EXPECT_FALSE(options::SaveSettings(settings));
  settings = {};
  settings.background = 4;
  EXPECT_FALSE(options::SaveSettings(settings));
  ASSERT_TRUE(options::LoadFile(options::SettingsPath().c_str()));
  EXPECT_EQ(options::ReadSettings(), options::Settings{});
}
}  // namespace
