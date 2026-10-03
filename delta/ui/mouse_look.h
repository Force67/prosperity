#pragma once

#include "base/arch.h"

struct SDL_Window;
union SDL_Event;

namespace ui {

struct MouseLookSettings {
  bool enabled = false;
  u32 sensitivity = 100;
  bool invert_y = false;
};

MouseLookSettings GetMouseLookSettings();
void ConfigureMouseLook(MouseLookSettings settings);
void SyncMouseLook(SDL_Window* window);
void ProcessMouseLookEvent(const SDL_Event& event);
void UpdateMouseLook();
bool PollMouseLook(u8& x, u8& y, bool& left, bool& right);
const char* MouseLookNotice();
void ResetMouseLook(SDL_Window* window);

}  // namespace ui
