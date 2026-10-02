#include "ui/input_sdl.h"

#include <SDL3/SDL.h>

#include "imgui.h"
#include "ui/home_screen.h"
#include "ui/overlay.h"
#include "ui/overlay_log.h"
#include "ui/pause_menu.h"

namespace ui {

void ProcessEvent(const SDL_Event& e,
                  SDL_Window* window,
                  u32 framebuffer_width,
                  u32 framebuffer_height) {
  if (!HomeScreenActive() && e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat &&
      (e.key.scancode == SDL_SCANCODE_LCTRL ||
       e.key.scancode == SDL_SCANCODE_RCTRL)) {
    PauseMenuToggle();
    return;
  }
  if (!HomeScreenActive() && !PauseMenuVisible()) {
    if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat) {
      if (e.key.scancode == SDL_SCANCODE_F1)
        OverlayToggle();
      if (e.key.scancode == SDL_SCANCODE_F2)
        OverlayLogToggle();
    }
    return;
  }
  ImGuiIO& io = ImGui::GetIO();
  int width, height;
  SDL_GetWindowSize(window, &width, &height);
  const float sx = width ? float(framebuffer_width) / width : 1;
  const float sy = height ? float(framebuffer_height) / height : 1;
  if (e.type == SDL_EVENT_MOUSE_MOTION)
    io.AddMousePosEvent(e.motion.x * sx, e.motion.y * sy);
  if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN ||
      e.type == SDL_EVENT_MOUSE_BUTTON_UP) {
    io.AddMousePosEvent(e.button.x * sx, e.button.y * sy);
    if (e.button.button == SDL_BUTTON_LEFT)
      io.AddMouseButtonEvent(0, e.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
  }
  if (e.type == SDL_EVENT_MOUSE_WHEEL)
    io.AddMouseWheelEvent(e.wheel.x, e.wheel.y);
  if (e.type == SDL_EVENT_KEY_DOWN || e.type == SDL_EVENT_KEY_UP) {
    ImGuiKey key = ImGuiKey_None;
    switch (e.key.scancode) {
      case SDL_SCANCODE_UP:
        key = ImGuiKey_UpArrow;
        break;
      case SDL_SCANCODE_DOWN:
        key = ImGuiKey_DownArrow;
        break;
      case SDL_SCANCODE_LEFT:
        key = ImGuiKey_LeftArrow;
        break;
      case SDL_SCANCODE_RIGHT:
        key = ImGuiKey_RightArrow;
        break;
      case SDL_SCANCODE_RETURN:
        key = ImGuiKey_Enter;
        break;
      case SDL_SCANCODE_SPACE:
        key = ImGuiKey_Space;
        break;
      case SDL_SCANCODE_ESCAPE:
        key = ImGuiKey_Escape;
        break;
      case SDL_SCANCODE_O:
        key = ImGuiKey_O;
        break;
      default:
        break;
    }
    if (key != ImGuiKey_None)
      io.AddKeyEvent(key, e.type == SDL_EVENT_KEY_DOWN);
  }
  if (e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN ||
      e.type == SDL_EVENT_GAMEPAD_BUTTON_UP) {
    ImGuiKey key = ImGuiKey_None;
    switch (e.gbutton.button) {
      case SDL_GAMEPAD_BUTTON_DPAD_UP:
        key = ImGuiKey_UpArrow;
        break;
      case SDL_GAMEPAD_BUTTON_DPAD_DOWN:
        key = ImGuiKey_DownArrow;
        break;
      case SDL_GAMEPAD_BUTTON_DPAD_LEFT:
        key = ImGuiKey_LeftArrow;
        break;
      case SDL_GAMEPAD_BUTTON_DPAD_RIGHT:
        key = ImGuiKey_RightArrow;
        break;
      case SDL_GAMEPAD_BUTTON_SOUTH:
        key = ImGuiKey_Enter;
        break;
      case SDL_GAMEPAD_BUTTON_EAST:
        key = ImGuiKey_Escape;
        break;
      case SDL_GAMEPAD_BUTTON_NORTH:
        key = ImGuiKey_O;
        break;
      default:
        break;
    }
    if (key != ImGuiKey_None)
      io.AddKeyEvent(key, e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN);
  }
}

}  // namespace ui
