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
                     bool ps5_ready,
                     base::String (*check_game)(const base::String&) = nullptr);
base::String TakeHomeScreenAddPath();
void HomeScreenSetError(const base::String& error);
bool HomeScreenDone();
// Returns the selected game path, or an empty string when the window closes.
base::String EndHomeScreen();
struct HomeBackground {
  bool visible = false;
  u32 style = 1;
  float time = 0;
  float pulse = 20;
  float opacity = 1;
};

bool HomeScreenActive();
void HomeScreenSetBackground(u32 style);
HomeBackground HomeScreenBackground();
void HomeScreenBuild(u32 width, u32 height);
bool LaunchTransitionActive();
void LaunchTransitionGameReady();
void LaunchTransitionBuild(u32 width, u32 height);
void ReturnTransitionBuild(u32 width,
                           u32 height,
                           const base::String& title,
                           u64 started_ns);

}  // namespace ui
