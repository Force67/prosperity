/*
 * PS4Delta : PS4 emulation and research project
 *
 * On-screen overlay content (Dear ImGui): the keyboard->DualSense legend. This
 * file only builds the ImDrawData; overlay_vk.cpp rasterises it through a
 * Vulkan pipeline. Frame timings and this process's CPU/GPU/RAM/VRAM use are
 * drawn by the GPU perf overlay instead (gpu/vulkan/vk_perf.cc).
 */
#ifndef __ANDROID__

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <unistd.h>

#include "imgui.h"
#include "overlay.h"

namespace gfx {
namespace {

bool g_visible = true;
bool g_inited = false;

std::mutex g_perfMtx;
float g_fps = 0, g_gpuMs = 0, g_frameMs = 0;

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
const char *kTitle = "Controls  (F1 to toggle)";

// ---- drawing helpers (foreground draw list) --------------------------------
void panelBg(ImDrawList *dl, ImVec2 tl, ImVec2 br) {
  dl->AddRectFilled(tl, br, IM_COL32(15, 15, 18, 205), 5.0f);
  dl->AddRect(tl, br, IM_COL32(255, 255, 255, 40), 5.0f);
}

void buildLegend() {
  ImDrawList *dl = ImGui::GetForegroundDrawList();
  ImFont *font = ImGui::GetFont();
  const float fs = ImGui::GetFontSize();
  const float pad = 8.0f, gap = fs, lh = fs + 3.0f;
  float keyW = 0.0f;
  for (auto &r : kRows)
    keyW = std::max(keyW, font->CalcTextSizeA(fs, FLT_MAX, 0.0f, r.key).x);
  float bodyW = 0.0f;
  for (auto &r : kRows)
    bodyW = std::max(bodyW, keyW + gap + font->CalcTextSizeA(fs, FLT_MAX, 0.0f, r.button).x);
  float titleW = font->CalcTextSizeA(fs, FLT_MAX, 0.0f, kTitle).x;
  const ImVec2 o(10.0f, 10.0f);
  float panelW = std::max(titleW, bodyW) + pad * 2.0f;
  float panelH = pad * 2.0f + lh + 4.0f + lh * IM_ARRAYSIZE(kRows);
  panelBg(dl, o, ImVec2(o.x + panelW, o.y + panelH));
  float x = o.x + pad, y = o.y + pad;
  dl->AddText(ImVec2(x, y), IM_COL32(120, 200, 255, 255), kTitle);
  y += lh + 4.0f;
  for (auto &r : kRows) {
    dl->AddText(ImVec2(x, y), IM_COL32(255, 235, 150, 255), r.key);
    dl->AddText(ImVec2(x + keyW + gap, y), IM_COL32(230, 230, 230, 255), r.button);
    y += lh;
  }
}

}  // namespace

void overlayEnsureImGui() {
  if (g_inited)
    return;
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.LogFilename = nullptr;
  g_inited = true;
}

void overlaySetPerf(float fps, float gpuMs, float frameMs) {
  std::lock_guard<std::mutex> lk(g_perfMtx);
  g_fps = fps;
  g_gpuMs = gpuMs;
  g_frameMs = frameMs;
}

void overlayBuildFrame(uint32_t w, uint32_t h, uint64_t vramUsed,
                       uint64_t vramTotal) {
  overlayEnsureImGui();
  ImGuiIO &io = ImGui::GetIO();
  io.DisplaySize = ImVec2((float)w, (float)h);
  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  if (g_visible) {
    buildLegend();
  }
  ImGui::Render();
}

void overlayToggle() { g_visible = !g_visible; }

}  // namespace gfx

#endif  // !__ANDROID__
