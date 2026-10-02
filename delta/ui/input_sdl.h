#pragma once

#include "base/arch.h"

union SDL_Event;
struct SDL_Window;

namespace ui {

void ProcessEvent(const SDL_Event& event,
                  SDL_Window* window,
                  u32 framebuffer_width,
                  u32 framebuffer_height);

}  // namespace ui
