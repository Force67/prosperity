#pragma once

#include "base/containers/vector.h"
#include "base/strings/xstring.h"

namespace ui {

void ConfigureSettings(const base::Vector<base::String>& backends,
                       const base::Vector<base::String>& devices);
bool SettingsButton();
void SettingsOpen();
bool SettingsVisible();
void SettingsBuild(u32 width, u32 height);
void SettingsReset();

}  // namespace ui
