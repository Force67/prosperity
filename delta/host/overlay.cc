/*
 * PS4Delta : PS4 emulation and research project
 *
 * On-screen overlay content (Dear ImGui): the keyboard->DualSense legend. This
 * file only builds the ImDrawData; overlay_vk.cc rasterises it through a
 * Vulkan pipeline. Frame timings and this process's CPU/GPU/RAM/VRAM use are
 * drawn by the GPU perf overlay instead (gpu/render/perf.cc).
 */

#include <cfloat>
#include "base/arch.h"

#include "base/math/value_bounds.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "host/overlay.h"
#include "host/overlay_log.h"
#include "host/overlay_theme.h"
#include "imgui.h"

namespace host {
namespace {

bool g_visible = true;
bool g_inited = false;

base::Mutex g_perf_mtx;
float g_fps = 0, g_gpu_ms = 0, g_frame_ms = 0;

struct Row {
  const char *key, *button;
};
const Row kRows[] = {
    {"WASD", "Left Stick / D-Pad  (move)"},
    {"Arrow Keys", "Right Stick  (aim)"},
    {"Space", "Cross  (confirm)"},
    {"Esc / Bksp", "Circle  (back)"},
    {"F", "Square"},
    {"R", "Triangle"},
    {"Q", "L1"},
    {"E", "R1"},
    {"Left Shift", "L2"},
    {"Right Shift", "R2"},
    {"Enter / P", "Options  (start)"},
    {"Tab", "Touchpad  (map)"},
};
const char* kTitle = "CONTROLS";

void BuildLegend() {
  ImDrawList* dl = ImGui::GetForegroundDrawList();
  ImFont* font = ImGui::GetFont();
  const float fs = ImGui::GetFontSize();
  const float pad = 16.0f, gap = 24.0f, lh = fs + 8.0f;
  float key_w = 0.0f;
  for (auto& r : kRows)
    key_w = base::Max(key_w, font->CalcTextSizeA(fs, FLT_MAX, 0.0f, r.key).x);
  float body_w = 0.0f;
  for (auto& r : kRows)
    body_w = base::Max(
        body_w,
        key_w + gap + font->CalcTextSizeA(fs, FLT_MAX, 0.0f, r.button).x);
  const float title_w =
      font->CalcTextSizeA(fs, FLT_MAX, 0.0f, kTitle).x + 96.0f;
  const ImVec2 o(12.0f, 12.0f);
  float panel_w = base::Max(title_w, body_w) + pad * 2.0f;
  float panel_h = pad * 2.0f + lh + 8.0f + lh * IM_ARRAYSIZE(kRows);
  overlay_theme::DrawPanel(dl, o, ImVec2(o.x + panel_w, o.y + panel_h));
  float x = o.x + pad, y = o.y + pad;
  overlay_theme::DrawMark(dl, ImVec2(x, y));
  dl->AddText(ImVec2(x + 20.0f, y), overlay_theme::kText, kTitle);
  ImFont* mono = overlay_theme::MonospaceFont();
  const float shortcut_w = mono->CalcTextSizeA(13, FLT_MAX, 0, "[F1]").x;
  dl->AddText(mono, 13.0f, ImVec2(o.x + panel_w - pad - shortcut_w, y + 1),
              overlay_theme::kAmber, "[F1]");
  const float separator_y = y + lh;
  dl->AddLine(ImVec2(x, separator_y), ImVec2(o.x + panel_w - pad, separator_y),
              overlay_theme::kBorder);
  y += lh + 8.0f;
  const float column_x = x + key_w + gap * 0.5f;
  dl->AddLine(ImVec2(column_x, y - 2),
              ImVec2(column_x, y + lh * IM_ARRAYSIZE(kRows) - 6),
              overlay_theme::kBorder);
  for (const auto& row : kRows) {
    dl->AddText(ImVec2(x, y), overlay_theme::kCyan, row.key);
    dl->AddText(ImVec2(x + key_w + gap, y), overlay_theme::kSecondary,
                row.button);
    y += lh;
  }
}

}  // namespace

void OverlayEnsureImGui() {
  if (g_inited)
    return;
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  overlay_theme::Apply();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.LogFilename = nullptr;
  g_inited = true;
}

void OverlaySetPerf(float fps, float gpu_ms, float frame_ms) {
  base::LockGuard<base::Mutex> lk(g_perf_mtx);
  g_fps = fps;
  g_gpu_ms = gpu_ms;
  g_frame_ms = frame_ms;
}

void OverlayBuildFrame(u32 w, u32 h, u64 vram_used, u64 vram_total) {
  OverlayEnsureImGui();
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2((float)w, (float)h);
  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  if (g_visible)
    BuildLegend();
  OverlayLogBuild(w, h);
  ImGui::Render();
}

void OverlayToggle() {
  g_visible = !g_visible;
}

}  // namespace host
