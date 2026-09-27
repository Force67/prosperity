/*
 * PS4Delta : PS4 emulation and research project
 *
 * On-screen overlay content (Dear ImGui): the keyboard->DualSense legend. This
 * file only builds the ImDrawData; overlay_vk.cc rasterises it through a
 * Vulkan pipeline. Frame timings and this process's CPU/GPU/RAM/VRAM use are
 * drawn by the GPU perf overlay instead (gpu/render/perf.cc).
 */

#include <unistd.h>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "base/arch.h"

#include "base/math/value_bounds.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "host/overlay.h"
#include "host/overlay_log.h"
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
const char* kTitle = "Controls  (F1 to toggle)";

// ---- drawing helpers (foreground draw list) --------------------------------
void PanelBg(ImDrawList* dl, ImVec2 tl, ImVec2 br) {
  dl->AddRectFilled(tl, br, IM_COL32(15, 15, 18, 205), 5.0f);
  dl->AddRect(tl, br, IM_COL32(255, 255, 255, 40), 5.0f);
}

void BuildLegend() {
  ImDrawList* dl = ImGui::GetForegroundDrawList();
  ImFont* font = ImGui::GetFont();
  const float fs = ImGui::GetFontSize();
  const float pad = 8.0f, gap = fs, lh = fs + 3.0f;
  float key_w = 0.0f;
  for (auto& r : kRows)
    key_w = base::Max(key_w, font->CalcTextSizeA(fs, FLT_MAX, 0.0f, r.key).x);
  float body_w = 0.0f;
  for (auto& r : kRows)
    body_w = base::Max(
        body_w,
        key_w + gap + font->CalcTextSizeA(fs, FLT_MAX, 0.0f, r.button).x);
  float title_w = font->CalcTextSizeA(fs, FLT_MAX, 0.0f, kTitle).x;
  const ImVec2 o(10.0f, 10.0f);
  float panel_w = base::Max(title_w, body_w) + pad * 2.0f;
  float panel_h = pad * 2.0f + lh + 4.0f + lh * IM_ARRAYSIZE(kRows);
  PanelBg(dl, o, ImVec2(o.x + panel_w, o.y + panel_h));
  float x = o.x + pad, y = o.y + pad;
  dl->AddText(ImVec2(x, y), IM_COL32(120, 200, 255, 255), kTitle);
  y += lh + 4.0f;
  for (auto& r : kRows) {
    dl->AddText(ImVec2(x, y), IM_COL32(255, 235, 150, 255), r.key);
    dl->AddText(ImVec2(x + key_w + gap, y), IM_COL32(230, 230, 230, 255),
                r.button);
    y += lh;
  }
}

}  // namespace

void OverlayEnsureImGui() {
  if (g_inited)
    return;
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
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
