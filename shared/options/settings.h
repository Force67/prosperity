#pragma once

#include "base/strings/xstring.h"

namespace options {

struct Settings {
  base::String backend = "vulkan";
  base::String gpu;
  base::String vsync;
  bool async_present = true;
  u32 background = 0;
  bool performance = true;
  bool validation = false;

  bool operator==(const Settings&) const = default;
};

base::String SettingsPath();
Settings ReadSettings();
bool SaveSettings(const Settings& settings);

}  // namespace options
