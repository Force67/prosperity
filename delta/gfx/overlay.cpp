/*
 * PS4Delta : PS4 emulation and research project
 *
 * On-screen overlay content (Dear ImGui): the keyboard->DualSense legend, a
 * memory-pressure gauge (VRAM + system RAM) bottom-right, and a CPU/GPU
 * utilization gauge top-right. This file only builds the ImDrawData + gathers
 * host metrics; overlay_vk.cpp rasterises it through a Vulkan pipeline.
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

// ---- host metrics (Linux /proc), refreshed ~2x/second ----------------------
struct Metrics {
  float cpuPct = 0;    // whole-process CPU%, can exceed 100 (multi-threaded)
  float cpuMax = 100;  // ncpu*100, for the bar scale
  uint64_t ramUsed = 0, ramTotal = 0;  // bytes
} g_metrics;

double nowSec() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void refreshMetrics() {
  static double lastAt = 0;
  static uint64_t lastJiffies = 0;
  static double lastCpuAt = 0;
  const double t = nowSec();
  if (t - lastAt < 0.5)
    return;
  lastAt = t;

  // Process CPU time (utime+stime) from /proc/self/stat field 14/15.
  if (FILE *f = std::fopen("/proc/self/stat", "r")) {
    char buf[1024];
    size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    buf[n] = 0;
    const char *p = std::strrchr(buf, ')');  // comm may contain spaces/parens
    unsigned long ut = 0, st = 0;
    if (p) {
      // after ") ": field 3 (state) onward; utime=14, stime=15 -> skip 11 fields
      int field = 3;
      p += 2;
      while (*p && field < 14) {
        if (*p == ' ') field++;
        p++;
      }
      std::sscanf(p, "%lu %lu", &ut, &st);
    }
    const long hz = sysconf(_SC_CLK_TCK);
    const uint64_t jiffies = ut + st;
    if (lastCpuAt > 0 && hz > 0) {
      const double dt = t - lastCpuAt;
      const double dj = double(jiffies - lastJiffies) / hz;
      g_metrics.cpuPct = dt > 0 ? float(dj / dt * 100.0) : 0;
    }
    lastJiffies = jiffies;
    lastCpuAt = t;
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    g_metrics.cpuMax = ncpu > 0 ? float(ncpu * 100) : 100;
  }

  // System RAM from /proc/meminfo (kB).
  if (FILE *f = std::fopen("/proc/meminfo", "r")) {
    char line[256];
    uint64_t total = 0, avail = 0;
    while (std::fgets(line, sizeof(line), f)) {
      unsigned long kb = 0;
      if (std::sscanf(line, "MemTotal: %lu kB", &kb) == 1) total = kb * 1024ull;
      else if (std::sscanf(line, "MemAvailable: %lu kB", &kb) == 1) avail = kb * 1024ull;
    }
    std::fclose(f);
    g_metrics.ramTotal = total;
    g_metrics.ramUsed = total > avail ? total - avail : 0;
  }
}

// ---- drawing helpers (foreground draw list) --------------------------------
float gib(uint64_t b) { return float(b) / (1024.0f * 1024.0f * 1024.0f); }

// A labeled horizontal bar: title on top, "value" right-aligned, filled to frac.
// Returns the height consumed.
float bar(ImDrawList *dl, float x, float y, float w, const char *label,
          const char *value, float frac, ImU32 col) {
  const float fs = ImGui::GetFontSize();
  const float barH = fs * 0.55f;
  dl->AddText(ImVec2(x, y), IM_COL32(210, 210, 210, 255), label);
  ImFont *font = ImGui::GetFont();
  float vw = font->CalcTextSizeA(fs, FLT_MAX, 0.0f, value).x;
  dl->AddText(ImVec2(x + w - vw, y), IM_COL32(255, 255, 255, 255), value);
  const float by = y + fs + 2.0f;
  frac = std::clamp(frac, 0.0f, 1.0f);
  dl->AddRectFilled(ImVec2(x, by), ImVec2(x + w, by + barH),
                    IM_COL32(40, 40, 46, 220), 2.0f);
  if (frac > 0)
    dl->AddRectFilled(ImVec2(x, by), ImVec2(x + w * frac, by + barH), col, 2.0f);
  dl->AddRect(ImVec2(x, by), ImVec2(x + w, by + barH),
              IM_COL32(255, 255, 255, 30), 2.0f);
  return fs + 2.0f + barH + fs * 0.5f;
}

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

// CPU/GPU utilization, top-right.
void buildUtilGauge(float dispW) {
  ImDrawList *dl = ImGui::GetForegroundDrawList();
  const float fs = ImGui::GetFontSize(), pad = 8.0f, w = 190.0f;
  const float panelW = w + pad * 2.0f, panelH = pad * 2.0f + fs + (fs * 2.05f) * 2;
  const ImVec2 tl(dispW - panelW - 10.0f, 10.0f);
  panelBg(dl, tl, ImVec2(tl.x + panelW, tl.y + panelH));
  float x = tl.x + pad, y = tl.y + pad;
  dl->AddText(ImVec2(x, y), IM_COL32(120, 200, 255, 255), "Utilization");
  y += fs + 4.0f;
  float cpu, gpu;
  {
    std::lock_guard<std::mutex> lk(g_perfMtx);
    cpu = g_metrics.cpuPct;
    gpu = g_frameMs > 0.01f ? std::min(100.0f, g_gpuMs / g_frameMs * 100.0f) : 0;
  }
  char val[32];
  std::snprintf(val, sizeof val, "%.0f%%", cpu);
  y += bar(dl, x, y, w, "CPU", val, cpu / g_metrics.cpuMax, IM_COL32(90, 200, 120, 255));
  std::snprintf(val, sizeof val, "%.0f%%", gpu);
  bar(dl, x, y, w, "GPU", val, gpu / 100.0f, IM_COL32(230, 150, 90, 255));
}

// FPS + memory pressure (VRAM + system RAM), bottom-right (mirrors the fps meter).
void buildMemGauge(float dispW, float dispH, uint64_t vramUsed, uint64_t vramTotal) {
  ImDrawList *dl = ImGui::GetForegroundDrawList();
  const float fs = ImGui::GetFontSize(), pad = 8.0f, w = 190.0f;
  const float panelW = w + pad * 2.0f;
  const float panelH = pad * 2.0f + fs + fs + 2.0f + (fs * 2.05f) * 2;
  const ImVec2 tl(dispW - panelW - 10.0f, dispH - panelH - 10.0f);
  panelBg(dl, tl, ImVec2(tl.x + panelW, tl.y + panelH));
  float x = tl.x + pad, y = tl.y + pad;
  float fps, ms;
  {
    std::lock_guard<std::mutex> lk(g_perfMtx);
    fps = g_fps;
    ms = g_frameMs;
  }
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.0f FPS   %.1f ms", fps, ms);
  dl->AddText(ImVec2(x, y), IM_COL32(120, 200, 255, 255), buf);
  y += fs + 4.0f;
  char val[48];
  if (vramTotal) {
    std::snprintf(val, sizeof val, "%.1f / %.1f GB", gib(vramUsed), gib(vramTotal));
    y += bar(dl, x, y, w, "VRAM", val, float(vramUsed) / float(vramTotal),
             IM_COL32(200, 120, 220, 255));
  } else {
    y += bar(dl, x, y, w, "VRAM", "n/a", 0, IM_COL32(200, 120, 220, 255));
  }
  std::snprintf(val, sizeof val, "%.1f / %.1f GB", gib(g_metrics.ramUsed),
                gib(g_metrics.ramTotal));
  bar(dl, x, y, w, "RAM", val,
      g_metrics.ramTotal ? float(g_metrics.ramUsed) / float(g_metrics.ramTotal) : 0,
      IM_COL32(90, 160, 230, 255));
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
    refreshMetrics();
    buildLegend();
    buildUtilGauge((float)w);
    buildMemGauge((float)w, (float)h, vramUsed, vramTotal);
  }
  ImGui::Render();
}

void overlayToggle() { g_visible = !g_visible; }

}  // namespace gfx

#endif  // !__ANDROID__
