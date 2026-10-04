#pragma once

#include "base/arch.h"

union SDL_Event;
struct SDL_Window;

namespace ui {

// True while the last input came from a controller, so hints can show its
// buttons instead of keys.
bool UsingGamepad();
// Label of an SDL_GamepadButton on the controller in use ("A" or "Cross").
const char* GamepadButtonName(int button);

void ProcessEvent(const SDL_Event& event,
                  SDL_Window* window,
                  u32 framebuffer_width,
                  u32 framebuffer_height);

}  // namespace ui
