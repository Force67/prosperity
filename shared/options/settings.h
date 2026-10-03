#pragma once

#include "base/strings/xstring.h"

namespace options {

struct Settings {
  base::String backend = "vulkan";
  base::String gpu;
  base::String vsync;
  bool async_present = true;
  bool display_timing = true;
  u32 background = 0;
  bool performance = true;
  bool validation = false;
  bool mouse_look = false;
  u32 mouse_sensitivity = 100;
  bool mouse_invert_y = false;

  bool operator==(const Settings&) const = default;
};

base::String SettingsPath();
Settings ReadSettings();
bool SaveSettings(const Settings& settings);

}  // namespace options
