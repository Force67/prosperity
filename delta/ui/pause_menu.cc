#include "ui/pause_menu.h"

#if defined(__linux__) && !defined(__ANDROID__)
#include "base/algorithm.h"
#include "base/atomic.h"
#include "base/time/time.h"
#include "guest/pause.h"
#include "imgui.h"
#include "ui/home_screen.h"
#include "ui/overlay_theme.h"

namespace ui {
namespace {
base::String g_title;
bool g_ready = false;
bool g_controls = false;
int g_selection = 0;
base::Atomic<bool> g_exit_requested{false};
float g_visibility = 0;
u64 g_tick_ns = 0;

void Resume() {
  guest::SetPaused(false);
  ImGui::GetIO().ClearInputKeys();
}
}  // namespace

void PauseMenuSetGameTitle(const base::String& title) {
  g_title = title;
}

void PauseMenuGameReady() {
  g_ready = true;
}

void PauseMenuToggle() {
  if (!g_ready || HomeScreenActive() || LaunchTransitionActive())
    return;
  if (guest::Paused()) {
    Resume();
  } else {
    g_controls = false;
    g_selection = 0;
    ImGui::GetIO().ClearInputKeys();
    guest::SetPaused(true);
  }
}

bool PauseMenuVisible() {
  return guest::Paused() || g_visibility > 0;
}

bool PauseMenuExitRequested() {
  return g_exit_requested.load();
}

void PauseMenuBuild(u32 width, u32 height) {
  const auto now = base::TickClock::NowNs();
  const float dt =
      g_tick_ns ? base::Min(float(now - g_tick_ns) / 1e9f, 0.05f) : 1.0f / 60;
  g_tick_ns = now;
  const bool paused = guest::Paused();
  g_visibility =
      base::Clamp(g_visibility + dt / (paused ? 0.18f : -0.20f), 0.0f, 1.0f);
  const float ease = g_visibility * g_visibility * (3 - 2 * g_visibility);
  if (paused) {
    guest::PollPause();
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
      if (g_controls)
        g_controls = false;
      else
        Resume();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow))
      g_selection = (g_selection + 2) % 3;
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow))
      g_selection = (g_selection + 1) % 3;
    if (ImGui::IsKeyPressed(ImGuiKey_Enter)) {
      if (g_controls)
        g_controls = false;
      else if (g_selection == 0)
        Resume();
      else if (g_selection == 1)
        g_controls = true;
      else
        g_exit_requested.store(true);
    }
  }
  const float w = float(width), h = float(height);
  auto* dl = ImGui::GetBackgroundDrawList();
  dl->AddRectFilled(ImVec2(0, 0), ImVec2(w, h),
                    IM_COL32(8, 10, 14, int(175 * ease)));
  const float panel_w = base::Min(480.0f, w - 48);
  const float panel_h = g_controls ? 432.0f : 392.0f;
  const ImVec2 tl((w - panel_w) * 0.5f, (h - panel_h) * 0.5f + 12 * (1 - ease));
  const ImVec2 br(tl.x + panel_w, tl.y + panel_h);
  overlay_theme::DrawPanel(dl, tl, br, ease);
  const auto ink = overlay_theme::WithOpacity(overlay_theme::kText, ease);
  const auto secondary =
      overlay_theme::WithOpacity(overlay_theme::kSecondary, ease);
  dl->AddCircleFilled(ImVec2(tl.x + 48, tl.y + 48), 20,
                      overlay_theme::WithOpacity(overlay_theme::kRaised, ease));
  dl->AddRectFilled(ImVec2(tl.x + 41, tl.y + 40), ImVec2(tl.x + 46, tl.y + 56),
                    ink, 2);
  dl->AddRectFilled(ImVec2(tl.x + 50, tl.y + 40), ImVec2(tl.x + 55, tl.y + 56),
                    ink, 2);
  auto* font = overlay_theme::HeadingFont();
  dl->AddText(font, 28, ImVec2(tl.x + 84, tl.y + 32), ink,
              g_controls ? "Controls" : "Paused");
  dl->AddText(ImGui::GetIO().FontDefault, 15, ImVec2(tl.x + 28, tl.y + 92),
              secondary, g_title.empty() ? "Your game" : g_title.c_str(),
              nullptr, panel_w - 56);

  ImGui::SetNextWindowPos(tl);
  ImGui::SetNextWindowSize(ImVec2(panel_w, panel_h));
  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ease);
  ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 24);
  ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.5f, 0.5f));
  ImGui::Begin("Pause", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoSavedSettings |
                   ImGuiWindowFlags_NoBackground);
  ImGui::BeginDisabled(!paused);
  if (g_controls) {
    const char* rows[][2] = {{"WASD", "Move"},
                             {"Arrow keys", "Aim"},
                             {"Space / Esc / F / R", "Face buttons"},
                             {"Enter / P", "Options"},
                             {"Tab", "Touchpad"}};
    for (int i = 0; i < 5; ++i) {
      dl->AddText(ImVec2(tl.x + 28, tl.y + 140 + i * 32), ink, rows[i][0]);
      dl->AddText(ImVec2(tl.x + panel_w - 144, tl.y + 140 + i * 32), secondary,
                  rows[i][1]);
    }
    ImGui::SetCursorPos(ImVec2(28, panel_h - 100));
    if (ImGui::Button("Back", ImVec2(panel_w - 56, 44)))
      g_controls = false;
  } else {
    ImGui::SetCursorPos(ImVec2(28, 152));
    ImGui::PushStyleColor(ImGuiCol_Button, overlay_theme::kText);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, overlay_theme::kText);
    ImGui::PushStyleColor(ImGuiCol_Text, overlay_theme::kCanvas);
    if (ImGui::Button("Resume game", ImVec2(panel_w - 56, 48)))
      Resume();
    if (g_selection == 0 && paused)
      overlay_theme::DrawFocusHalo(ImGui::GetWindowDrawList(),
                                   ImGui::GetItemRectMin(),
                                   ImGui::GetItemRectMax(), 24, ease);
    ImGui::PopStyleColor(3);
    ImGui::SetCursorPos(ImVec2(28, 212));
    if (ImGui::Button("Controls", ImVec2(panel_w - 56, 44)))
      g_controls = true;
    if (g_selection == 1 && paused)
      overlay_theme::DrawFocusHalo(ImGui::GetWindowDrawList(),
                                   ImGui::GetItemRectMin(),
                                   ImGui::GetItemRectMax(), 24, ease);
    ImGui::SetCursorPos(ImVec2(28, 268));
    ImGui::PushStyleColor(ImGuiCol_Text, overlay_theme::kSecondary);
    if (ImGui::Button("Exit emulator", ImVec2(panel_w - 56, 44)))
      g_exit_requested.store(true);
    ImGui::PopStyleColor();
    if (g_selection == 2 && paused)
      overlay_theme::DrawFocusHalo(ImGui::GetWindowDrawList(),
                                   ImGui::GetItemRectMin(),
                                   ImGui::GetItemRectMax(), 24, ease);
  }
  ImGui::EndDisabled();
  ImGui::End();
  ImGui::PopStyleVar(3);
  dl->AddText(
      ImVec2(tl.x + 28, br.y - 44), secondary,
      g_controls ? "Esc  Back     Ctrl  Resume" : "Ctrl or Esc to resume");
}

void PauseMenuReset() {
  guest::SetPaused(false);
  g_ready = false;
  g_controls = false;
  g_exit_requested.store(false);
  g_visibility = 0;
  g_tick_ns = 0;
  g_title.clear();
}
}  // namespace ui
#else
namespace ui {
void PauseMenuSetGameTitle(const base::String&) {}
void PauseMenuGameReady() {}
void PauseMenuToggle() {}
bool PauseMenuVisible() {
  return false;
}
bool PauseMenuExitRequested() {
  return false;
}
void PauseMenuBuild(u32, u32) {}
void PauseMenuReset() {}
}  // namespace ui
#endif
