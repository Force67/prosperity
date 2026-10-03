#include "ui/mouse_look.h"

#include <SDL3/SDL.h>
#include <cmath>

#include "base/algorithm.h"
#include "base/logging.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "options/options.h"
#include "options/settings.h"
#include "ui/home_screen.h"
#include "ui/overlay_memory.h"
#include "ui/pause_menu.h"

namespace {
DELTA_OPTION(bool, kMouseLook, "DELTA_MOUSE_LOOK", false);
DELTA_OPTION(u32, kMouseSensitivity, "DELTA_MOUSE_SENSITIVITY", 100);
DELTA_OPTION(bool, kMouseInvertY, "DELTA_MOUSE_INVERT_Y", false);
}  // namespace

namespace ui {
namespace {
base::Mutex g_mutex;
MouseLookSettings g_settings;
bool g_loaded = false;
bool g_captured = false;
bool g_failed = false;
float g_dx = 0, g_dy = 0;
u8 g_x = 128, g_y = 128;
bool g_left = false, g_right = false;
SDL_MouseButtonFlags g_blocked_buttons = 0;
u64 g_updated = 0;
u64 g_notice_time = 0;
const char* g_notice = nullptr;

void Load() {
  if (g_loaded)
    return;
  const auto settings = options::ReadSettings();
  g_settings = {settings.mouse_look, settings.mouse_sensitivity,
                settings.mouse_invert_y};
  g_loaded = true;
}

void Clear() {
  g_dx = g_dy = 0;
  g_x = g_y = 128;
  g_left = g_right = false;
  g_updated = SDL_GetTicksNS();
}

u8 Axis(float speed) {
  // The 32-count offset clears typical controller dead zones. Remaining travel
  // scales with mouse speed, reaching full deflection at 1000 pixels/second.
  if (speed == 0)
    return 128;
  const float travel = 32 + base::Min(std::abs(speed) * 0.095f, 95.0f);
  return static_cast<u8>(base::Clamp(
      std::round(128 + std::copysign(travel, speed)), 0.0f, 255.0f));
}
}  // namespace

MouseLookSettings GetMouseLookSettings() {
  base::LockGuard lock(g_mutex);
  Load();
  return g_settings;
}

void ConfigureMouseLook(MouseLookSettings settings) {
  base::LockGuard lock(g_mutex);
  settings.sensitivity = base::Clamp(settings.sensitivity, 10u, 300u);
  if (g_settings.enabled != settings.enabled) {
    g_notice = settings.enabled ? "Mouse look on. Ctrl releases the cursor."
                                : "Mouse look off. Cursor released.";
    g_notice_time = SDL_GetTicksNS();
  }
  g_settings = settings;
  g_loaded = true;
  g_failed = false;
  Clear();
}

void SyncMouseLook(SDL_Window* window) {
  base::LockGuard lock(g_mutex);
  Load();
  const bool capture = window && g_settings.enabled && !HomeScreenActive() &&
                       !LaunchTransitionActive() && !PauseMenuVisible() &&
                       !MemoryOverlayBlocksInput() &&
                       (SDL_GetWindowFlags(window) & SDL_WINDOW_INPUT_FOCUS);
  if (capture == g_captured || (capture && g_failed))
    return;
  if (!SDL_SetWindowRelativeMouseMode(window, capture)) {
    if (capture) {
      BASE_LOGI("input", "Mouse capture failed: {}", SDL_GetError());
      g_failed = true;
      g_notice = "Mouse capture unavailable. Toggle F3 to retry.";
      g_notice_time = SDL_GetTicksNS();
    }
    return;
  }
  g_captured = capture;
  Clear();
  g_blocked_buttons = SDL_GetMouseState(nullptr, nullptr);
}

void ProcessMouseLookEvent(const SDL_Event& event) {
  base::LockGuard lock(g_mutex);
  if (!g_captured)
    return;
  if (event.type == SDL_EVENT_MOUSE_MOTION) {
    g_dx += event.motion.xrel;
    g_dy += event.motion.yrel;
  }
  if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN ||
      event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
    const auto mask = SDL_BUTTON_MASK(event.button.button);
    const bool down = event.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
    if (!down)
      g_blocked_buttons &= ~mask;
    if (g_blocked_buttons & mask)
      return;
    if (event.button.button == SDL_BUTTON_LEFT)
      g_left = down;
    if (event.button.button == SDL_BUTTON_RIGHT)
      g_right = down;
  }
}

void UpdateMouseLook() {
  base::LockGuard lock(g_mutex);
  const auto now = SDL_GetTicksNS();
  const float dt = base::Clamp(float(now - g_updated) / 1e9f, 1.0f / 240, 0.1f);
  g_updated = now;
  const float scale = g_settings.sensitivity / 100.0f / dt;
  g_x = Axis(g_dx * scale);
  g_y = Axis(g_dy * scale * (g_settings.invert_y ? -1 : 1));
  g_dx = g_dy = 0;
}

bool PollMouseLook(u8& x, u8& y, bool& left, bool& right) {
  base::LockGuard lock(g_mutex);
  if (!g_captured || !g_settings.enabled)
    return false;
  const bool fresh = SDL_GetTicksNS() - g_updated < 50'000'000;
  x = fresh ? g_x : 128;
  y = fresh ? g_y : 128;
  left = g_left;
  right = g_right;
  return true;
}

const char* MouseLookNotice() {
  base::LockGuard lock(g_mutex);
  return g_notice && SDL_GetTicksNS() - g_notice_time < 3'000'000'000 ? g_notice
                                                                      : nullptr;
}

void ResetMouseLook(SDL_Window* window) {
  base::LockGuard lock(g_mutex);
  if (window)
    SDL_SetWindowRelativeMouseMode(window, false);
  g_captured = g_failed = false;
  g_notice = nullptr;
  Clear();
}
}  // namespace ui
