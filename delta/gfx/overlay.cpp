/*
 * PS4Delta : PS4 emulation and research project
 *
 * Keyboard->DualSense legend overlay (desktop/Linux). We let Dear ImGui build a
 * raw draw list (rounded panel + text) and rasterise the resulting ImDrawData
 * ourselves into the present framebuffer, so no extra Vulkan pipeline/render
 * pass is needed: it slots into the same CPU staging buffer that gfx_vk.cpp
 * uploads each frame. Android has its own touch overlay (gfx_android.cpp).
 */
#ifndef __ANDROID__

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include "imgui.h"
#include "overlay.h"

namespace gfx {
namespace {

bool g_visible = true;
bool g_inited = false;
const unsigned char *g_atlas = nullptr;  // alpha8 font pixels (owned by ImGui)
int g_atlasW = 0, g_atlasH = 0;

// Mirrors the keyboard map in gfx_vk.cpp::pollKeyboardPad. ProggyClean (the
// default font) is monospaced, so the two columns line up on the pixel grid.
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

void initImGui() {
  if (g_inited)
    return;
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  io.IniFilename = nullptr;  // headless overlay: never touch disk
  io.LogFilename = nullptr;
  unsigned char *px = nullptr;
  int w = 0, h = 0;
  io.Fonts->GetTexDataAsAlpha8(&px, &w, &h);  // also builds the atlas
  io.Fonts->SetTexID((ImTextureID)(intptr_t)1);  // we sample g_atlas ourselves
  g_atlas = px;
  g_atlasW = w;
  g_atlasH = h;
  g_inited = true;
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
    bodyW = std::max(bodyW, keyW + gap +
                                font->CalcTextSizeA(fs, FLT_MAX, 0.0f, r.button).x);
  float titleW = font->CalcTextSizeA(fs, FLT_MAX, 0.0f, kTitle).x;

  const ImVec2 o(10.0f, 10.0f);
  float panelW = std::max(titleW, bodyW) + pad * 2.0f;
  float panelH = pad * 2.0f + lh + 4.0f + lh * IM_ARRAYSIZE(kRows);
  ImVec2 br(o.x + panelW, o.y + panelH);
  dl->AddRectFilled(o, br, IM_COL32(15, 15, 18, 205), 5.0f);
  dl->AddRect(o, br, IM_COL32(255, 255, 255, 40), 5.0f);

  float x = o.x + pad, y = o.y + pad;
  dl->AddText(ImVec2(x, y), IM_COL32(120, 200, 255, 255), kTitle);
  y += lh + 4.0f;
  for (auto &r : kRows) {
    dl->AddText(ImVec2(x, y), IM_COL32(255, 235, 150, 255), r.key);
    dl->AddText(ImVec2(x + keyW + gap, y), IM_COL32(230, 230, 230, 255), r.button);
    y += lh;
  }
}

inline float edge(const ImVec2 &p0, const ImVec2 &p1, float px, float py) {
  return (px - p0.x) * (p1.y - p0.y) - (py - p0.y) * (p1.x - p0.x);
}

inline void blend(uint8_t *fb, int W, int H, bool bgra, int x, int y, float r,
                  float g, float b, float a) {
  if (a <= 0.0f || x < 0 || y < 0 || x >= W || y >= H)
    return;
  if (a > 1.0f)
    a = 1.0f;
  uint8_t *p = fb + (size_t)(y * W + x) * 4;
  int ri = bgra ? 2 : 0, bi = bgra ? 0 : 2;
  p[ri] = uint8_t(p[ri] * (1.0f - a) + r * a);
  p[1] = uint8_t(p[1] * (1.0f - a) + g * a);
  p[bi] = uint8_t(p[bi] * (1.0f - a) + b * a);
}

// Software-rasterise one ImGui draw list: textured triangles, the texture being
// the alpha8 font atlas (filled prims sample its opaque white pixel). Vertex
// colour and atlas alpha are interpolated, so ImGui's edge anti-aliasing works.
void rasterize(const ImDrawList *dl, uint8_t *fb, int W, int H, bool bgra) {
  const ImDrawVert *vtx = dl->VtxBuffer.Data;
  const ImDrawIdx *idx = dl->IdxBuffer.Data;
  for (const ImDrawCmd &cmd : dl->CmdBuffer) {
    if (cmd.UserCallback)
      continue;
    int cx0 = std::max(0, (int)cmd.ClipRect.x);
    int cy0 = std::max(0, (int)cmd.ClipRect.y);
    int cx1 = std::min(W, (int)cmd.ClipRect.z);
    int cy1 = std::min(H, (int)cmd.ClipRect.w);
    const ImDrawIdx *ii = idx + cmd.IdxOffset;
    for (unsigned n = 0; n + 3 <= cmd.ElemCount; n += 3) {
      const ImDrawVert &a = vtx[cmd.VtxOffset + ii[n + 0]];
      const ImDrawVert &b = vtx[cmd.VtxOffset + ii[n + 1]];
      const ImDrawVert &c = vtx[cmd.VtxOffset + ii[n + 2]];
      float area = edge(a.pos, b.pos, c.pos.x, c.pos.y);
      if (area == 0.0f)
        continue;
      float inv = 1.0f / area;
      int x0 = std::max(cx0, (int)std::floor(std::min({a.pos.x, b.pos.x, c.pos.x})));
      int y0 = std::max(cy0, (int)std::floor(std::min({a.pos.y, b.pos.y, c.pos.y})));
      int x1 = std::min(cx1, (int)std::ceil(std::max({a.pos.x, b.pos.x, c.pos.x})));
      int y1 = std::min(cy1, (int)std::ceil(std::max({a.pos.y, b.pos.y, c.pos.y})));

      auto chan = [](ImU32 col, int s) { return float((col >> s) & 0xFF); };
      for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++) {
          float px = x + 0.5f, py = y + 0.5f;
          // barycentrics (sign of `area` cancels, so winding doesn't matter)
          float w0 = edge(b.pos, c.pos, px, py) * inv;
          float w1 = edge(c.pos, a.pos, px, py) * inv;
          float w2 = edge(a.pos, b.pos, px, py) * inv;
          if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f)
            continue;
          float u = w0 * a.uv.x + w1 * b.uv.x + w2 * c.uv.x;
          float v = w0 * a.uv.y + w1 * b.uv.y + w2 * c.uv.y;
          int ax = std::clamp((int)(u * g_atlasW), 0, g_atlasW - 1);
          int ay = std::clamp((int)(v * g_atlasH), 0, g_atlasH - 1);
          float cov = g_atlas[ay * g_atlasW + ax] / 255.0f;
          float cr = w0 * chan(a.col, 0) + w1 * chan(b.col, 0) + w2 * chan(c.col, 0);
          float cg = w0 * chan(a.col, 8) + w1 * chan(b.col, 8) + w2 * chan(c.col, 8);
          float cb = w0 * chan(a.col, 16) + w1 * chan(b.col, 16) + w2 * chan(c.col, 16);
          float ca = w0 * chan(a.col, 24) + w1 * chan(b.col, 24) + w2 * chan(c.col, 24);
          blend(fb, W, H, bgra, x, y, cr, cg, cb, (ca / 255.0f) * cov);
        }
    }
  }
}

}  // namespace

void overlayDraw(uint8_t *fb, uint32_t w, uint32_t h, bool bgra) {
  if (!g_visible || !fb || !w || !h)
    return;
  initImGui();
  ImGuiIO &io = ImGui::GetIO();
  io.DisplaySize = ImVec2((float)w, (float)h);
  io.DeltaTime = 1.0f / 60.0f;
  ImGui::NewFrame();
  buildLegend();
  ImGui::Render();
  const ImDrawData *dd = ImGui::GetDrawData();
  if (!dd)
    return;
  for (int i = 0; i < dd->CmdListsCount; i++)
    rasterize(dd->CmdLists[i], fb, (int)w, (int)h, bgra);
}

void overlayToggle() { g_visible = !g_visible; }

}  // namespace gfx

#endif  // !__ANDROID__
