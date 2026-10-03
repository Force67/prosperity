#include "options/settings.h"

#include <filesystem>
#include <fstream>

#include "base/environment_variables.h"
#include "base/option.h"
#include "base/option_file.h"
#include "base/strings/format.h"

namespace options {
namespace {
base::String Value(const char* name) {
  const auto* option = base::FindOption(name);
  if (!option)
    return {};
  char value[4096];
  option->FormatValue(value, sizeof(value));
  return value;
}
}  // namespace

base::String SettingsPath() {
  base::StringU8 value;
  if (base::GetEnvironmentVariable(u8"XDG_CONFIG_HOME", value) &&
      !value.empty() && value.front() == '/')
    return base::String(reinterpret_cast<const char*>(value.c_str())) +
           "/prosperity/settings.txt";
  if (base::GetEnvironmentVariable(u8"HOME", value) && !value.empty())
    return base::String(reinterpret_cast<const char*>(value.c_str())) +
           "/.config/prosperity/settings.txt";
  return {};
}

Settings ReadSettings() {
  Settings settings;
  const auto backend = Value("DELTA_GPU_BACKEND");
  if (!backend.empty())
    settings.backend = backend;
  settings.gpu = Value("DELTA_VK_GPU");
  settings.vsync = Value("DELTA_GPU_VSYNC");
  settings.async_present = Value("DELTA_GPU_SYNCPRESENT") != "true";
  const auto background = Value("DELTA_UI_BACKGROUND");
  u64 style = 0;
  if (base::ParseUnsigned(background.c_str(), style) && style <= 3)
    settings.background = static_cast<u32>(style);
  settings.performance = Value("DELTA_GPU_OVERLAY") != "false";
  settings.validation = Value("DELTA_VK_VALIDATE") == "true";
  return settings;
}

bool SaveSettings(const Settings& settings) {
  if ((settings.backend != "vulkan" && settings.backend != "opengl" &&
       settings.backend != "d3d12") ||
      (!settings.vsync.empty() && settings.vsync != "0" &&
       settings.vsync != "1") ||
      settings.background > 3 ||
      settings.gpu.find_first_of("\r\n") != base::String::npos)
    return false;
  const auto path = SettingsPath();
  if (path.empty())
    return false;
  std::error_code error;
  std::filesystem::create_directories(
      std::filesystem::path(path.c_str()).parent_path(), error);
  if (error)
    return false;
  const auto text = base::Format(
      "+DELTA_GPU_BACKEND={}\n+DELTA_VK_GPU=\"{}\"\n"
      "+DELTA_GPU_VSYNC=\"{}\"\n+DELTA_GPU_SYNCPRESENT={}\n"
      "+DELTA_UI_BACKGROUND={}\n+DELTA_GPU_OVERLAY={}\n"
      "+DELTA_VK_VALIDATE={}\n",
      settings.backend, settings.gpu, settings.vsync, !settings.async_present,
      settings.background, settings.performance, settings.validation);
  const auto temporary = path + ".tmp";
  std::ofstream file(temporary.c_str(), std::ios::binary | std::ios::trunc);
  file.write(text.data(), text.size());
  file.close();
  if (!file) {
    std::filesystem::remove(temporary.c_str(), error);
    return false;
  }
  std::filesystem::rename(temporary.c_str(), path.c_str(), error);
  if (!error)
    return true;
  std::filesystem::remove(temporary.c_str(), error);
  return false;
}

}  // namespace options
