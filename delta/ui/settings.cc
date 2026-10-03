#include "ui/settings.h"

#if defined(__linux__) && !defined(__ANDROID__)
#include <cmath>

#include "base/algorithm.h"
#include "base/time/time.h"
#include "imgui.h"
#include "options/settings.h"
#include "ui/home_screen.h"
#include "ui/mouse_look.h"
#include "ui/overlay_theme.h"

namespace ui {
namespace {
base::Vector<base::String> g_backends;
base::Vector<base::String> g_devices;
options::Settings g_saved;
options::Settings g_running;
options::Settings g_draft;
bool g_open = false;
bool g_confirm = false;
bool g_focus = false;
int g_tab = 0;
u64 g_opened_ns = 0;
base::String g_error;
bool g_restart = false;
bool g_loaded = false;
float g_row_x = 0;
ImGuiConfigFlags g_nav_flags = 0;

bool Dirty() {
  return g_draft != g_saved;
}

void Close() {
  g_open = g_confirm = false;
  ImGui::GetIO().ConfigFlags = g_nav_flags;
  ImGui::GetIO().ClearInputKeys();
}

void RequestClose() {
  if (Dirty())
    g_confirm = true;
  else
    Close();
}

bool Save() {
  if (!options::SaveSettings(g_draft)) {
    g_error = "Could not save settings. Check your configuration directory.";
    return false;
  }
  auto running = g_running;
  running.background = g_draft.background;
  running.mouse_look = g_draft.mouse_look;
  running.mouse_sensitivity = g_draft.mouse_sensitivity;
  running.mouse_invert_y = g_draft.mouse_invert_y;
  ConfigureMouseLook(
      {g_draft.mouse_look, g_draft.mouse_sensitivity, g_draft.mouse_invert_y});
  g_restart = running != g_draft;
  if (g_saved.background != g_draft.background)
    HomeScreenSetBackground(g_draft.background);
  g_saved = g_draft;
  g_error.clear();
  return true;
}

void Text(ImVec2 pos,
          const char* text,
          ImU32 color,
          float size = 15,
          float wrap = 0) {
  ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), size, pos, color, text,
                                      nullptr, wrap);
}

void PrimaryBegin() {
  ImGui::PushStyleColor(ImGuiCol_Button, overlay_theme::kText);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(216, 229, 255, 255));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, overlay_theme::kAccent);
  ImGui::PushStyleColor(ImGuiCol_Text, overlay_theme::kCanvas);
}

bool Supported(const char* backend) {
  for (const auto& name : g_backends)
    if (name == backend)
      return true;
  return false;
}

void ApiCards() {
  const auto start = ImGui::GetCursorScreenPos();
  Text(start, "Graphics API", overlay_theme::kText);
  Text(ImVec2(start.x, start.y + 23),
       "Choose the renderer for your next session.", overlay_theme::kSecondary,
       12);
  ImGui::SetCursorScreenPos(ImVec2(start.x, start.y + 45));
  const float available = ImGui::GetContentRegionAvail().x;
  const bool compact = available < 450;
  const float card_width = compact ? available : (available - 24) / 3;
  const char* names[] = {"Vulkan", "OpenGL", "Direct3D 12"};
  const char* values[] = {"vulkan", "opengl", "d3d12"};
  const char* descriptions[] = {"Default renderer", "OpenGL 4.6",
                                "Via vkd3d on Linux"};
  for (int i = 0; i < 3; ++i) {
    if (i && !compact)
      ImGui::SameLine(0, 12);
    const bool supported = Supported(values[i]);
    const bool selected = g_draft.backend == values[i];
    ImGui::PushID(i);
    ImGui::BeginDisabled(!supported);
    if (ImGui::InvisibleButton("api", ImVec2(card_width, compact ? 60 : 80)))
      g_draft.backend = values[i];
    const auto tl = ImGui::GetItemRectMin();
    const auto br = ImGui::GetItemRectMax();
    auto* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(tl, br, overlay_theme::kRaised, 14);
    if (selected)
      dl->AddRectFilled(
          tl, br, overlay_theme::WithOpacity(overlay_theme::kAccent, 0.10f),
          14);
    dl->AddRect(tl, br,
                selected ? overlay_theme::kAccent : overlay_theme::kBorder, 14);
    if (ImGui::IsItemHovered() || ImGui::IsItemFocused())
      overlay_theme::DrawFocusHalo(dl, tl, br, 14, 0.6f);
    Text(ImVec2(tl.x + 16, tl.y + 14), names[i],
         supported ? overlay_theme::kText : overlay_theme::kMuted);
    Text(ImVec2(tl.x + 16, tl.y + (compact ? 36 : 55)),
         supported ? descriptions[i] : "Not in this build",
         overlay_theme::kSecondary, 12);
    if (selected) {
      const ImVec2 p(br.x - 23, tl.y + 23);
      dl->AddLine(ImVec2(p.x - 5, p.y), ImVec2(p.x - 1, p.y + 4),
                  overlay_theme::kAccent, 2);
      dl->AddLine(ImVec2(p.x - 1, p.y + 4), ImVec2(p.x + 6, p.y - 4),
                  overlay_theme::kAccent, 2);
    }
    ImGui::EndDisabled();
    ImGui::PopID();
  }
  ImGui::SetCursorScreenPos(
      ImVec2(start.x, start.y + 45 + (compact ? 204 : 86)));
}

float Row(const char* title, const char* help) {
  const auto pos = ImGui::GetCursorScreenPos();
  g_row_x = pos.x;
  const float width = ImGui::GetContentRegionAvail().x;
  const bool compact = width < 450;
  const float height = compact ? 94 : 56;
  Text(ImVec2(pos.x, pos.y + 10), title, overlay_theme::kText);
  Text(ImVec2(pos.x, pos.y + 33), help, overlay_theme::kSecondary, 12,
       compact ? width : width - 244);
  ImGui::GetWindowDrawList()->AddLine(ImVec2(pos.x, pos.y + height - 1),
                                      ImVec2(pos.x + width, pos.y + height - 1),
                                      overlay_theme::kBorder);
  ImGui::SetCursorScreenPos(ImVec2(compact ? pos.x : pos.x + width - 224,
                                   pos.y + (compact ? 55 : 12)));
  ImGui::SetNextItemWidth(compact ? width : 224);
  return pos.y + height;
}

void EndRow(float y) {
  ImGui::SetCursorScreenPos(ImVec2(g_row_x, y));
}

void Toggle(const char* id, bool* value) {
  ImGui::PushID(id);
  const auto pos = ImGui::GetCursorScreenPos();
  if (ImGui::InvisibleButton("toggle", ImVec2(80, 32)))
    *value = !*value;
  auto* dl = ImGui::GetWindowDrawList();
  const auto tl = ImVec2(pos.x, pos.y + 4);
  const auto br = ImVec2(pos.x + 44, pos.y + 28);
  dl->AddRectFilled(
      tl, br, *value ? overlay_theme::kAccent : overlay_theme::kRaised, 12);
  dl->AddCircleFilled(ImVec2(pos.x + (*value ? 32 : 12), pos.y + 16), 8,
                      overlay_theme::kText);
  Text(ImVec2(pos.x + 56, pos.y + 8), *value ? "On" : "Off",
       overlay_theme::kSecondary, 12);
  if (ImGui::IsItemFocused())
    dl->AddRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
                overlay_theme::kAccent, 8);
  ImGui::PopID();
}

void Graphics() {
  ApiCards();
  float y = Row("Graphics device", "Automatic selects the default device.");
  const char* preview = g_draft.gpu.empty() ? "Automatic" : g_draft.gpu.c_str();
  if (ImGui::BeginCombo("##device", preview)) {
    if (ImGui::Selectable("Automatic", g_draft.gpu.empty()))
      g_draft.gpu.clear();
    for (const auto& device : g_devices) {
      if (ImGui::Selectable(device.c_str(), g_draft.gpu == device))
        g_draft.gpu = device;
    }
    ImGui::EndCombo();
  }
  EndRow(y);
  y = Row("Vertical sync", "Automatic balances pacing and latency.");
  const char* modes[] = {"Automatic", "On", "Off"};
  int mode = g_draft.vsync == "1" ? 1 : (g_draft.vsync == "0" ? 2 : 0);
  if (ImGui::Combo("##vsync", &mode, modes, 3))
    g_draft.vsync = mode == 1 ? "1" : (mode == 2 ? "0" : "");
  EndRow(y);
  y = Row("Async presentation", "Keep the window responsive between frames.");
  Toggle("async", &g_draft.async_present);
  EndRow(y);
  ImGui::Dummy(ImVec2(0, 10));
  ImGui::TextWrapped(
      "Graphics changes take effect after restarting Prosperity.");
}

void Appearance() {
  float y =
      Row("Home background", "Backdrop when game artwork is unavailable.");
  const char* backgrounds[] = {"Random", "Current", "Liquid glass",
                               "Signal formation"};
  int background = static_cast<int>(base::Min(g_draft.background, 3u));
  if (ImGui::Combo("##background", &background, backgrounds, 4))
    g_draft.background = background;
  EndRow(y);
  y = Row("Performance overlay", "Show frame rate and GPU timing during play.");
  Toggle("performance", &g_draft.performance);
  EndRow(y);
  ImGui::Dummy(ImVec2(0, 16));
  ImGui::TextWrapped(
      "The background updates when saved. The performance "
      "overlay changes after restarting.");
}

void Controls() {
  float y = Row("Mouse look", "Move the mouse to control the camera.");
  Toggle("mouse_look", &g_draft.mouse_look);
  EndRow(y);
  ImGui::BeginDisabled(!g_draft.mouse_look);
  y = Row("Sensitivity", "Adjust how quickly the camera turns.");
  int sensitivity = static_cast<int>(g_draft.mouse_sensitivity);
  if (ImGui::SliderInt("##mouse_sensitivity", &sensitivity, 10, 300, "%d%%"))
    g_draft.mouse_sensitivity = static_cast<u32>(sensitivity);
  EndRow(y);
  y = Row("Invert vertical look", "Move the mouse up to look down.");
  Toggle("mouse_invert_y", &g_draft.mouse_invert_y);
  EndRow(y);
  ImGui::EndDisabled();
  ImGui::Dummy(ImVec2(0, 16));
  ImGui::TextWrapped(
      "During play: F3 toggles mouse look. Ctrl opens the pause "
      "menu and releases the cursor. Left / right click send "
      "R2 / L2. Capture also releases when you switch windows.");
  ImGui::Dummy(ImVec2(0, 8));
  ImGui::TextWrapped(
      "Saved controls apply immediately. Camera speed also "
      "depends on the game's controller settings.");
}

void Advanced() {
  const float y =
      Row("Vulkan validation", "Extra diagnostics for graphics issues.");
  Toggle("validation", &g_draft.validation);
  EndRow(y);
  ImGui::Dummy(ImVec2(0, 16));
  ImGui::TextWrapped(
      "Validation can slow down rendering. Leave it off for "
      "normal play. Changes require a restart.");
}
}  // namespace

void ConfigureSettings(const base::Vector<base::String>& backends,
                       const base::Vector<base::String>& devices) {
  g_backends = backends;
  g_devices = devices;
}

void SettingsOpen() {
  g_nav_flags = ImGui::GetIO().ConfigFlags;
  ImGui::GetIO().ConfigFlags |=
      ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
  if (!g_loaded) {
    g_saved = g_running = options::ReadSettings();
    g_loaded = true;
  }
  g_draft = g_saved;
  g_error.clear();
  g_open = g_focus = true;
  g_confirm = false;
  g_tab = 0;
  g_opened_ns = base::TickClock::NowNs();
}

bool SettingsVisible() {
  return g_open;
}

bool SettingsButton() {
  const bool clicked = ImGui::Button("##settings", ImVec2(40, 36));
  const auto tl = ImGui::GetItemRectMin();
  const ImVec2 center(tl.x + 20, tl.y + 18);
  auto* dl = ImGui::GetWindowDrawList();
  for (int i = 0; i < 8; ++i) {
    const float angle = i * 3.14159265f / 4;
    const ImVec2 direction(std::cos(angle), std::sin(angle));
    dl->AddLine(
        ImVec2(center.x + direction.x * 7, center.y + direction.y * 7),
        ImVec2(center.x + direction.x * 11, center.y + direction.y * 11),
        overlay_theme::kText, 3);
  }
  dl->AddCircle(center, 7, overlay_theme::kText, 24, 2);
  dl->AddCircle(center, 2.5f, overlay_theme::kText, 16, 1.5f);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Settings");
  return clicked;
}

void SettingsBuild(u32 width, u32 height) {
  if (!g_open)
    return;
  const bool confirming = g_confirm;
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  const float t = base::Min(
      float(base::TickClock::NowNs() - g_opened_ns) / 180'000'000, 1.0f);
  const float fade = t * t * (3 - 2 * t);
  const float panel_w = base::Min(820.0f, w - 32);
  const float panel_h = base::Min(610.0f, h - 96);
  ImGui::SetNextWindowPos(ImVec2(w * 0.5f, h * 0.5f + 24 + 10 * (1 - fade)),
                          ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(panel_w, panel_h));
  ImGui::SetNextWindowBgAlpha(fade);
  if (g_focus) {
    ImGui::SetNextWindowFocus();
    g_focus = false;
  }
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(28, 24));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 24);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, overlay_theme::kSurface);
  ImGui::Begin("##settings_lounge", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoSavedSettings);
  ImGui::BeginDisabled(g_confirm);
  const auto origin = ImGui::GetCursorPos();
  ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(overlay_theme::kMuted),
                     "MAKE YOURSELF AT HOME");
  ImGui::PushFont(overlay_theme::HeadingFont());
  ImGui::TextUnformatted("Settings");
  ImGui::PopFont();
  ImGui::SetCursorPos(ImVec2(panel_w - 66, origin.y + 8));
  if (ImGui::Button("##settings_close", ImVec2(38, 38)))
    RequestClose();
  const auto close_pos = ImGui::GetItemRectMin();
  auto* dl = ImGui::GetWindowDrawList();
  dl->AddLine(ImVec2(close_pos.x + 14, close_pos.y + 14),
              ImVec2(close_pos.x + 24, close_pos.y + 24),
              overlay_theme::kSecondary, 1.5f);
  dl->AddLine(ImVec2(close_pos.x + 24, close_pos.y + 14),
              ImVec2(close_pos.x + 14, close_pos.y + 24),
              overlay_theme::kSecondary, 1.5f);
  ImGui::SetCursorPos(ImVec2(origin.x, origin.y + 76));
  const char* tabs[] = {"Graphics", "Appearance", "Controls", "Advanced"};
  ImGui::SetWindowFontScale(panel_w < 450 ? 0.85f : 1.0f);
  const float tab_width = (ImGui::GetContentRegionAvail().x - 12) / 4;
  for (int i = 0; i < 4; ++i) {
    if (i)
      ImGui::SameLine(0, 4);
    ImGui::PushStyleColor(ImGuiCol_Button, i == g_tab ? overlay_theme::kRaised
                                                      : overlay_theme::kInset);
    if (ImGui::Button(tabs[i], ImVec2(tab_width, 36)))
      g_tab = i;
    ImGui::PopStyleColor();
  }
  ImGui::SetWindowFontScale(1);
  ImGui::Dummy(ImVec2(0, 14));
  const float footer_y =
      panel_h - (panel_w < 450 || !g_error.empty() ? 116 : 82);
  const float body_height =
      base::Max(40.0f, footer_y - ImGui::GetCursorPosY() - 12);
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
  ImGui::BeginChild("settings_body", ImVec2(0, body_height), false);
  if (g_tab == 0)
    Graphics();
  else if (g_tab == 1)
    Appearance();
  else if (g_tab == 2)
    Controls();
  else
    Advanced();
  ImGui::EndChild();
  ImGui::PopStyleColor();
  ImGui::SetCursorPos(ImVec2(origin.x, footer_y));
  ImGui::Separator();
  ImGui::Dummy(ImVec2(0, 6));
  const auto status_color =
      g_error.empty()
          ? (Dirty() ? overlay_theme::kWarning : overlay_theme::kSecondary)
          : overlay_theme::kError;
  const char* status = !g_error.empty() ? g_error.c_str()
                       : Dirty()        ? "Unsaved changes"
                       : g_restart
                           ? "Saved. Restart to apply graphics settings."
                           : "All changes saved";
  const float save_width = base::Min(160.0f, (panel_w - 68) * 0.5f);
  if (panel_w < 450) {
    ImGui::PushTextWrapPos(panel_w - 28);
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(status_color), "%s",
                       status);
    ImGui::PopTextWrapPos();
  } else {
    const auto pos = ImGui::GetCursorScreenPos();
    Text(ImVec2(pos.x, pos.y + 10), status, status_color, 12,
         panel_w - 80 - save_width * 2);
  }
  ImGui::SetCursorPosX(panel_w - 28 - save_width * 2 - 12);
  if (ImGui::Button(panel_w < 450 ? "Reset defaults" : "Restore defaults",
                    ImVec2(save_width, 36)))
    g_draft = options::Settings{};
  ImGui::SameLine(0, 12);
  ImGui::BeginDisabled(!Dirty());
  PrimaryBegin();
  if (ImGui::Button("Save changes", ImVec2(save_width, 36)))
    Save();
  ImGui::PopStyleColor(4);
  ImGui::EndDisabled();
  if (!g_confirm && ImGui::IsKeyPressed(ImGuiKey_Escape))
    RequestClose();
  ImGui::EndDisabled();
  if (g_confirm)
    ImGui::OpenPopup("##settings_unsaved");
  ImGui::SetNextWindowPos(ImVec2(w * 0.5f, h * 0.5f), ImGuiCond_Always,
                          ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(base::Min(480.0f, w - 48), 0));
  if (ImGui::BeginPopupModal(
          "##settings_unsaved", nullptr,
          ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextUnformatted("Keep your changes?");
    ImGui::Dummy(ImVec2(0, 10));
    ImGui::TextWrapped("You have settings that haven't been saved.");
    if (!g_error.empty())
      ImGui::TextWrapped("%s", g_error.c_str());
    ImGui::Dummy(ImVec2(0, 20));
    PrimaryBegin();
    if (ImGui::Button("Save and return") && Save()) {
      Close();
      ImGui::CloseCurrentPopup();
    }
    ImGui::PopStyleColor(4);
    if (w > 520)
      ImGui::SameLine();
    if (ImGui::Button("Discard")) {
      Close();
      ImGui::CloseCurrentPopup();
    }
    if (w > 520)
      ImGui::SameLine();
    if (ImGui::Button("Keep editing") ||
        (confirming && ImGui::IsKeyPressed(ImGuiKey_Escape))) {
      g_confirm = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }
  ImGui::End();
  ImGui::PopStyleColor();
  ImGui::PopStyleVar(2);
}

void SettingsReset() {
  g_open = g_confirm = false;
}
}  // namespace ui
#else
namespace ui {
void ConfigureSettings(const base::Vector<base::String>&,
                       const base::Vector<base::String>&) {}
bool SettingsButton() {
  return false;
}
void SettingsOpen() {}
bool SettingsVisible() {
  return false;
}
void SettingsBuild(u32, u32) {}
void SettingsReset() {}
}  // namespace ui
#endif
