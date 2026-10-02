#pragma once

#include "base/containers/vector.h"
#include "base/strings/xstring.h"

namespace ui {

struct HomeGame {
  base::String path;
  base::String name;
  base::String title_id;
  bool is_ps5 = false;
  bool available = true;
  base::Vector<u8> icon;
  base::Vector<u8> artwork;
};

// Prepare the artwork before the host creates the UI rendering backend.
void BeginHomeScreen(const base::Vector<HomeGame>& games,
                     bool ps4_ready,
                     bool ps5_ready);
bool HomeScreenDone();
// Returns the selected game path, or an empty string when the window closes.
base::String EndHomeScreen();
struct HomeBackground {
  bool visible = false;
  u32 style = 1;
  float time = 0;
  float pulse = 20;
};

bool HomeScreenActive();
HomeBackground HomeScreenBackground();
void HomeScreenBuild(u32 width, u32 height);

}  // namespace ui
