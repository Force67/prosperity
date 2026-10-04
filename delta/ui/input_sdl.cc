#include "ui/input_sdl.h"

#include <SDL3/SDL.h>

#include "imgui.h"
#include "ui/home_screen.h"
#include "ui/mouse_look.h"
#include "ui/overlay.h"
#include "ui/overlay_log.h"
#include "ui/overlay_memory.h"
#include "ui/pause_menu.h"

namespace ui {
namespace {
bool g_gamepad_input = false;
SDL_JoystickID g_gamepad_id = 0;

void TrackInputDevice(const SDL_Event& e) {
  if (e.type == SDL_EVENT_GAMEPAD_ADDED ||
      e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) {
    g_gamepad_input = true;
    g_gamepad_id = e.type == SDL_EVENT_GAMEPAD_ADDED ? e.gdevice.which
                                                     : e.gbutton.which;
  } else if (e.type == SDL_EVENT_KEY_DOWN)
    g_gamepad_input = false;
  else if (e.type == SDL_EVENT_GAMEPAD_REMOVED)
    g_gamepad_input = SDL_HasGamepad();
}

// View + Menu together, or the guide button where the system leaves it to us.
bool PauseChord(const SDL_GamepadButtonEvent& e) {
  if (e.button == SDL_GAMEPAD_BUTTON_GUIDE)
    return true;
  SDL_Gamepad* pad = SDL_GetGamepadFromID(e.which);
  if (!pad)
    return false;
  return (e.button == SDL_GAMEPAD_BUTTON_START &&
          SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_BACK)) ||
         (e.button == SDL_GAMEPAD_BUTTON_BACK &&
          SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_START));
}
}  // namespace

bool UsingGamepad() {
  return g_gamepad_input;
}

const char* GamepadButtonName(int button) {
  const auto type = SDL_GetGamepadTypeForID(g_gamepad_id);
  switch (SDL_GetGamepadButtonLabelForType(
      type, static_cast<SDL_GamepadButton>(button))) {
    case SDL_GAMEPAD_BUTTON_LABEL_A:
      return "A";
    case SDL_GAMEPAD_BUTTON_LABEL_B:
      return "B";
    case SDL_GAMEPAD_BUTTON_LABEL_X:
      return "X";
    case SDL_GAMEPAD_BUTTON_LABEL_Y:
      return "Y";
    case SDL_GAMEPAD_BUTTON_LABEL_CROSS:
      return "Cross";
    case SDL_GAMEPAD_BUTTON_LABEL_CIRCLE:
      return "Circle";
    case SDL_GAMEPAD_BUTTON_LABEL_SQUARE:
      return "Square";
    case SDL_GAMEPAD_BUTTON_LABEL_TRIANGLE:
      return "Triangle";
    default:
      break;
  }
  const bool playstation =
      type == SDL_GAMEPAD_TYPE_PS4 || type == SDL_GAMEPAD_TYPE_PS5;
  switch (button) {
    case SDL_GAMEPAD_BUTTON_BACK:
      return playstation ? "Create" : "View";
    case SDL_GAMEPAD_BUTTON_START:
      return playstation ? "Options" : "Menu";
    default:
      return "?";
  }
}

void ProcessEvent(const SDL_Event& e,
                  SDL_Window* window,
                  u32 framebuffer_width,
                  u32 framebuffer_height) {
  TrackInputDevice(e);
  if (e.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
    MemoryOverlayEscape(false);
    MemoryOverlayEscape(false, true);
  } else if (e.type == SDL_EVENT_GAMEPAD_REMOVED) {
    MemoryOverlayEscape(false, true);
  }
  if (!HomeScreenActive() && !PauseMenuVisible() &&
      e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat &&
      e.key.scancode == SDL_SCANCODE_F4) {
    MemoryOverlayToggle();
    ImGui::GetIO().ClearInputKeys();
    SyncMouseLook(window);
    return;
  }
  if ((e.type == SDL_EVENT_KEY_DOWN || e.type == SDL_EVENT_KEY_UP) &&
      e.key.scancode == SDL_SCANCODE_ESCAPE &&
      MemoryOverlayEscape(e.type == SDL_EVENT_KEY_DOWN)) {
    ImGui::GetIO().ClearInputKeys();
    SyncMouseLook(window);
    return;
  }
  if ((e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN ||
       e.type == SDL_EVENT_GAMEPAD_BUTTON_UP) &&
      e.gbutton.button == SDL_GAMEPAD_BUTTON_EAST &&
      MemoryOverlayEscape(e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN, true)) {
    ImGui::GetIO().ClearInputKeys();
    SyncMouseLook(window);
    return;
  }
  if (!HomeScreenActive() &&
      ((e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat &&
        (e.key.scancode == SDL_SCANCODE_LCTRL ||
         e.key.scancode == SDL_SCANCODE_RCTRL)) ||
       (e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && PauseChord(e.gbutton)))) {
    PauseMenuToggle();
    SyncMouseLook(window);
    return;
  }
  SyncMouseLook(window);
  ProcessMouseLookEvent(e);
  if (!HomeScreenActive() && !PauseMenuVisible()) {
    if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat) {
      if (e.key.scancode == SDL_SCANCODE_F1)
        OverlayToggle();
      if (e.key.scancode == SDL_SCANCODE_F2)
        OverlayLogToggle();
      if (e.key.scancode == SDL_SCANCODE_F3) {
        auto settings = GetMouseLookSettings();
        settings.enabled = !settings.enabled;
        ConfigureMouseLook(settings);
        SyncMouseLook(window);
      }
    }
    if (GetMemoryOverlayMode() == MemoryOverlayMode::kOff)
      return;
  }
  ImGuiIO& io = ImGui::GetIO();
  int width = framebuffer_width, height = framebuffer_height;
  if (window)
    SDL_GetWindowSize(window, &width, &height);
  const float sx = (width ? float(framebuffer_width) / width : 1) /
                   io.DisplayFramebufferScale.x;
  const float sy = (height ? float(framebuffer_height) / height : 1) /
                   io.DisplayFramebufferScale.y;
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
    io.AddKeyEvent(ImGuiKey_ModShift, (e.key.mod & SDL_KMOD_SHIFT) != 0);
    ImGuiKey key = ImGuiKey_None;
    switch (e.key.scancode) {
      case SDL_SCANCODE_S:
        key = ImGuiKey_S;
        break;
      case SDL_SCANCODE_TAB:
        key = ImGuiKey_Tab;
        break;
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
    // Each button drives the menus' keys and ImGui's own gamepad navigation.
    ImGuiKey key = ImGuiKey_None, nav = ImGuiKey_None;
    switch (e.gbutton.button) {
      case SDL_GAMEPAD_BUTTON_DPAD_UP:
        key = ImGuiKey_UpArrow;
        nav = ImGuiKey_GamepadDpadUp;
        break;
      case SDL_GAMEPAD_BUTTON_DPAD_DOWN:
        key = ImGuiKey_DownArrow;
        nav = ImGuiKey_GamepadDpadDown;
        break;
      case SDL_GAMEPAD_BUTTON_DPAD_LEFT:
        key = ImGuiKey_LeftArrow;
        nav = ImGuiKey_GamepadDpadLeft;
        break;
      case SDL_GAMEPAD_BUTTON_DPAD_RIGHT:
        key = ImGuiKey_RightArrow;
        nav = ImGuiKey_GamepadDpadRight;
        break;
      case SDL_GAMEPAD_BUTTON_SOUTH:
        key = ImGuiKey_Enter;
        nav = ImGuiKey_GamepadFaceDown;
        break;
      case SDL_GAMEPAD_BUTTON_EAST:
        key = ImGuiKey_Escape;
        nav = ImGuiKey_GamepadFaceRight;
        break;
      case SDL_GAMEPAD_BUTTON_NORTH:
        key = ImGuiKey_O;
        break;
      case SDL_GAMEPAD_BUTTON_START:
        key = ImGuiKey_S;
        break;
      default:
        break;
    }
    const bool down = e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
    if (key != ImGuiKey_None)
      io.AddKeyEvent(key, down);
    if (nav != ImGuiKey_None)
      io.AddKeyEvent(nav, down);
  }
}

}  // namespace ui
