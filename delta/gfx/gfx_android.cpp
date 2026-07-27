/*
 * PS4Delta : PS4 emulation and research project
 *
 * On-screen Vulkan present for the Android app (DELTA_ANDROID_APP). Same scheme
 * as the desktop gfx_vk.cpp (CPU framebuffer -> staging buffer -> device image
 * -> blit into the acquired swapchain image -> present), but the window is an
 * ANativeWindow handed in by the NativeActivity loop (android_main) and the
 * surface comes from VK_KHR_android_surface. All Vulkan calls run on the guest
 * renderer thread (the only caller of ensure()/present()); android_main only
 * publishes the window handle and the touch-derived pad state.
 */
#if defined(__ANDROID__) && defined(DELTA_ANDROID_APP)

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#define VK_USE_PLATFORM_ANDROID_KHR
#include <android/native_window.h>
#include <vulkan/vulkan.h>

#include "gfx.h"
#include "gfx_android.h"
#include "overlay.h"

namespace gfx {
namespace {

#define VK_CHECK(expr)                                                         \
  do {                                                                         \
    VkResult _r = (expr);                                                      \
    if (_r != VK_SUCCESS) {                                                    \
      std::fprintf(stderr, "[gfx-android] %s failed: VkResult=%d\n", #expr,    \
                   _r);                                                        \
      return false;                                                            \
    }                                                                          \
  } while (0)

constexpr uint32_t kFrameSlotCount = 2;

struct FrameSlot {
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkSemaphore acquireSem = VK_NULL_HANDLE;
  VkFence acquireFence = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;

  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory stagingMem = VK_NULL_HANDLE;
  void *stagingMap = nullptr;
  VkImage frameImg = VK_NULL_HANDLE;
  VkDeviceMemory frameMem = VK_NULL_HANDLE;
};

struct State {
  ANativeWindow *window = nullptr;
  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  uint32_t queueFamily = 0;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;

  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkFormat swapFormat = VK_FORMAT_B8G8R8A8_UNORM;
  VkExtent2D swapExtent{};
  VkSurfaceTransformFlagBitsKHR preTransform =
      VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  std::vector<VkImage> swapImages;

  VkCommandPool cmdPool = VK_NULL_HANDLE;
  std::array<FrameSlot, kFrameSlotCount> slots;
  std::vector<VkSemaphore> renderSems;
  // Semaphores of a replaced swapchain; see retireRenderSemaphores.
  std::vector<VkSemaphore> retiredRenderSems;
  uint32_t nextSlot = 0;

  // Framebuffer dimensions shared by the per-slot upload resources.
  uint32_t fbW = 0, fbH = 0;
  VkFormat fbFormat = VK_FORMAT_R8G8B8A8_UNORM;

  bool needRecreate = false;
  bool ready = false;
};
State g;
std::atomic_bool g_presentFailed{false};
std::atomic_bool g_presentStopRequested{false};
constexpr uint64_t kPresentWaitSliceNs = 50'000'000;

void stopPresenting(const char *operation, VkResult result) {
  std::fprintf(stderr, "[gfx-android] %s failed: VkResult=%d\n", operation,
               result);
  g_presentFailed.store(true, std::memory_order_release);
}

bool waitForPresentFence(VkFence fence, const char *operation) {
  while (!g_presentFailed.load(std::memory_order_acquire) &&
         !g_presentStopRequested.load(std::memory_order_acquire)) {
    const VkResult result =
        vkWaitForFences(g.device, 1, &fence, VK_TRUE, kPresentWaitSliceNs);
    if (result == VK_SUCCESS)
      return true;
    if (result != VK_TIMEOUT) {
      stopPresenting(operation, result);
      return false;
    }
  }
  return false;
}

// Window handle + touch state published by android_main (other thread).
std::mutex g_inMutex;
ANativeWindow *g_pendingWindow = nullptr;
bool g_windowChanged = false;
constexpr int kMaxTouch = 8;
Touch g_touches[kMaxTouch];
int g_touchCount = 0;

// Rect the presented image occupies inside the surface (landscape px). The
// frame is letterboxed rather than stretched, so this is what maps a touch to
// a point in the image.
float g_viewX = 0.0f, g_viewY = 0.0f, g_viewW = 0.0f, g_viewH = 0.0f;

// On-screen virtual gamepad. Laid out in "height units": y runs 0..1 down the
// presented image and x runs 0..aspect across it, so the pad keeps its
// proportions on any panel and a circle stays a circle. Drawing and hit-testing
// share the units, so a control is exactly where it looks.
struct Pill {
  float cx, cy; // centre
  float r;      // half-height / corner radius
  float hw;     // half-width between the caps; 0 => circle
};

enum class Mark { Cross, Circle, Square, Triangle, Bars, Pad, L1, L2, R1, R2 };

struct Button {
  Pill p;
  bool PadKeys::*bit;
  Mark mark;
};

constexpr float kStickR = 0.155f;

struct Layout {
  Button btns[11];
  int n = 0;
  Pill lstick, rstick;
  float dpadCx, dpadCy, dpadArm;
};

// Thumbs sit at the bottom corners, so the sticks anchor there and the d-pad /
// face diamond stack above them; shoulders run along the top edge.
Layout buildLayout(float aspect) {
  const float A = aspect;
  Layout l;
  auto add = [&](float cx, float cy, float r, float hw, bool PadKeys::*bit,
                 Mark m) { l.btns[l.n++] = Button{{cx, cy, r, hw}, bit, m}; };

  const float faceCx = A - 0.24f, faceCy = 0.30f;
  const float spread = 0.105f, faceR = 0.058f;
  add(faceCx, faceCy - spread, faceR, 0.0f, &PadKeys::triangle, Mark::Triangle);
  add(faceCx + spread, faceCy, faceR, 0.0f, &PadKeys::circle, Mark::Circle);
  add(faceCx, faceCy + spread, faceR, 0.0f, &PadKeys::cross, Mark::Cross);
  add(faceCx - spread, faceCy, faceR, 0.0f, &PadKeys::square, Mark::Square);

  const float shY = 0.075f, shR = 0.042f, shHw = 0.080f;
  add(0.19f, shY, shR, shHw, &PadKeys::l1, Mark::L1);
  add(0.40f, shY, shR, shHw, &PadKeys::l2, Mark::L2);
  add(A - 0.19f, shY, shR, shHw, &PadKeys::r1, Mark::R1);
  add(A - 0.40f, shY, shR, shHw, &PadKeys::r2, Mark::R2);

  add(A * 0.5f, shY, shR, 0.130f, &PadKeys::touchpad, Mark::Pad);
  add(A * 0.5f + 0.26f, shY, 0.038f, 0.060f, &PadKeys::options, Mark::Bars);

  l.lstick = {0.26f, 0.74f, kStickR, 0.0f};
  l.rstick = {A - 0.26f, 0.74f, kStickR, 0.0f};
  l.dpadCx = 0.24f;
  l.dpadCy = 0.30f;
  l.dpadArm = 0.075f;
  return l;
}

// Distance test against a pill (a circle when hw == 0), grown by `grow` so the
// touch target is a little more forgiving than the drawn shape.
bool inPill(float hx, float hy, const Pill &p, float grow) {
  float dx = std::fabs(hx - p.cx) - p.hw;
  if (dx < 0.0f)
    dx = 0.0f;
  const float dy = hy - p.cy;
  const float r = p.r + grow;
  return dx * dx + dy * dy <= r * r;
}

// Surface pixel -> height units inside the presented image. False when the
// point falls in the letterbox margin.
bool toUnits(float sx, float sy, float &hx, float &hy) {
  hx = (sx - g_viewX) / g_viewH;
  hy = (sy - g_viewY) / g_viewH;
  return hx >= 0.0f && hy >= 0.0f && hx <= g_viewW / g_viewH && hy <= 1.0f;
}

// Map the down touches to a DS4 pad against the layout above.
PadKeys computePad() {
  PadKeys k; // neutral (sticks centred at 128)
  if (g_viewW <= 0.0f || g_viewH <= 0.0f)
    return k;
  const float A = g_viewW / g_viewH;
  const Layout l = buildLayout(A);
  for (int i = 0; i < g_touchCount; i++) {
    float hx, hy;
    if (!toUnits(g_touches[i].x, g_touches[i].y, hx, hy))
      continue;

    bool claimed = false;
    for (int b = 0; b < l.n && !claimed; b++)
      if (inPill(hx, hy, l.btns[b].p, 0.012f)) {
        k.*(l.btns[b].bit) = true;
        claimed = true;
      }
    if (claimed)
      continue;

    // D-pad: 8-way, from the offset within the cross.
    const float ddx = hx - l.dpadCx, ddy = hy - l.dpadCy;
    if (std::fabs(ddx) <= l.dpadArm * 1.5f && std::fabs(ddy) <= l.dpadArm * 1.5f) {
      const float dz = l.dpadArm * 0.32f;
      k.left = k.left || ddx < -dz;
      k.right = k.right || ddx > dz;
      k.up = k.up || ddy < -dz;
      k.down = k.down || ddy > dz;
      continue;
    }

    // Sticks: a generous zone around each ring, deflection from its centre.
    const bool leftSide = hx < A * 0.5f;
    const Pill &s = leftSide ? l.lstick : l.rstick;
    float dx = (hx - s.cx) / kStickR, dy = (hy - s.cy) / kStickR;
    if (std::fabs(dx) > 1.9f || std::fabs(dy) > 1.9f)
      continue;
    dx = std::clamp(dx, -1.0f, 1.0f);
    dy = std::clamp(dy, -1.0f, 1.0f);
    const uint8_t vx = uint8_t(std::clamp(128.0f + dx * 127.0f, 0.0f, 255.0f));
    const uint8_t vy = uint8_t(std::clamp(128.0f + dy * 127.0f, 0.0f, 255.0f));
    if (leftSide) {
      k.lx = vx;
      k.ly = vy;
      // Menus that only read the d-pad still follow the movement stick.
      k.left = k.left || dx < -0.4f;
      k.right = k.right || dx > 0.4f;
      k.up = k.up || dy < -0.4f;
      k.down = k.down || dy > 0.4f;
    } else {
      k.rx = vx;
      k.ry = vy;
    }
  }
  return k;
}

// --- control overlay (CPU alpha-blend into the present framebuffer) --------
// Everything is a signed distance field resolved with one pixel of coverage,
// so the pad has soft edges instead of the stair-stepped discs a hard test
// gives. Height units scale by the framebuffer height, which is exactly how
// the letterboxed blit maps them to the panel.

struct Paint {
  uint8_t r, g, b;
  float a;
};

constexpr Paint kNone{0, 0, 0, 0.0f};
constexpr Paint kGlass{12, 14, 20, 0.30f};   // resting button body
constexpr Paint kGlassOn{58, 66, 84, 0.58f}; // held
constexpr Paint kRim{224, 231, 245, 0.26f};
constexpr Paint kRimOn{255, 255, 255, 0.80f};
constexpr Paint kShadow{0, 0, 0, 0.20f};

inline float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

inline void blendPx(uint8_t *buf, int w, int h, bool bgra, int x, int y,
                    uint8_t r, uint8_t g, uint8_t b, float a) {
  if (x < 0 || y < 0 || x >= w || y >= h || a <= 0.0f)
    return;
  uint8_t *p = buf + (size_t)(y * w + x) * 4;
  int ri = bgra ? 2 : 0, bi = bgra ? 0 : 2;
  p[ri] = uint8_t(p[ri] * (1 - a) + r * a);
  p[1] = uint8_t(p[1] * (1 - a) + g * a);
  p[bi] = uint8_t(p[bi] * (1 - a) + b * a);
}

// Filled and/or stroked pill in framebuffer pixels. strokeW is the full width.
void drawPill(uint8_t *buf, int w, int h, bool bgra, float cx, float cy,
              float hw, float r, const Paint &fill, const Paint &stroke,
              float strokeW) {
  const float pad = r + strokeW + 2.0f;
  const int x0 = std::max(0, int(cx - hw - pad));
  const int x1 = std::min(w - 1, int(cx + hw + pad));
  const int y0 = std::max(0, int(cy - pad));
  const int y1 = std::min(h - 1, int(cy + pad));
  for (int y = y0; y <= y1; y++)
    for (int x = x0; x <= x1; x++) {
      float dx = std::fabs(x + 0.5f - cx) - hw;
      if (dx < 0.0f)
        dx = 0.0f;
      const float dy = y + 0.5f - cy;
      const float d = std::sqrt(dx * dx + dy * dy) - r;
      if (fill.a > 0.0f) {
        const float cov = clamp01(0.5f - d);
        if (cov > 0.0f)
          blendPx(buf, w, h, bgra, x, y, fill.r, fill.g, fill.b, fill.a * cov);
      }
      if (stroke.a > 0.0f && strokeW > 0.0f) {
        const float cov = clamp01(strokeW * 0.5f - std::fabs(d) + 0.5f);
        if (cov > 0.0f)
          blendPx(buf, w, h, bgra, x, y, stroke.r, stroke.g, stroke.b,
                  stroke.a * cov);
      }
    }
}

// Capsule segment; the building block for the ✕ / □ / △ marks and the d-pad.
void drawSeg(uint8_t *buf, int w, int h, bool bgra, float ax, float ay,
             float bx, float by, float halfW, const Paint &p) {
  const int x0 = std::max(0, int(std::min(ax, bx) - halfW - 2));
  const int x1 = std::min(w - 1, int(std::max(ax, bx) + halfW + 2));
  const int y0 = std::max(0, int(std::min(ay, by) - halfW - 2));
  const int y1 = std::min(h - 1, int(std::max(ay, by) + halfW + 2));
  const float vx = bx - ax, vy = by - ay;
  const float len2 = vx * vx + vy * vy;
  for (int y = y0; y <= y1; y++)
    for (int x = x0; x <= x1; x++) {
      const float px = x + 0.5f - ax, py = y + 0.5f - ay;
      const float t =
          len2 > 0.0f ? clamp01((px * vx + py * vy) / len2) : 0.0f;
      const float dx = px - vx * t, dy = py - vy * t;
      const float cov =
          clamp01(halfW - std::sqrt(dx * dx + dy * dy) + 0.5f);
      if (cov > 0.0f)
        blendPx(buf, w, h, bgra, x, y, p.r, p.g, p.b, p.a * cov);
    }
}

// 3x5 cells, MSB-first per row; just enough for the shoulder labels.
const uint8_t kGlyphL[5] = {4, 4, 4, 4, 7};
const uint8_t kGlyphR[5] = {6, 5, 6, 5, 5};
const uint8_t kGlyph1[5] = {2, 6, 2, 2, 7};
const uint8_t kGlyph2[5] = {7, 1, 7, 4, 7};

void drawCell(uint8_t *buf, int w, int h, bool bgra, float x0, float y0,
              float s, const Paint &p) {
  for (int y = int(y0); y <= int(y0 + s); y++)
    for (int x = int(x0); x <= int(x0 + s); x++) {
      const float cx = clamp01(std::min(x0 + s, x + 1.0f) - std::max(x0, float(x)));
      const float cy = clamp01(std::min(y0 + s, y + 1.0f) - std::max(y0, float(y)));
      const float cov = cx * cy;
      if (cov > 0.0f)
        blendPx(buf, w, h, bgra, x, y, p.r, p.g, p.b, p.a * cov);
    }
}

void drawChar(uint8_t *buf, int w, int h, bool bgra, const uint8_t rows[5],
              float x, float y, float s, const Paint &p) {
  for (int ry = 0; ry < 5; ry++)
    for (int rx = 0; rx < 3; rx++)
      if (rows[ry] & (4 >> rx))
        drawCell(buf, w, h, bgra, x + rx * s, y + ry * s, s, p);
}

void drawLabel(uint8_t *buf, int w, int h, bool bgra, const uint8_t a[5],
               const uint8_t b[5], float cx, float cy, float s,
               const Paint &p) {
  const float width = 7.0f * s; // 3 + gap + 3
  drawChar(buf, w, h, bgra, a, cx - width * 0.5f, cy - 2.5f * s, s, p);
  drawChar(buf, w, h, bgra, b, cx - width * 0.5f + 4.0f * s, cy - 2.5f * s, s,
           p);
}

void drawMark(uint8_t *buf, int w, int h, bool bgra, Mark m, float cx,
              float cy, float s, const Paint &p) {
  const float t = std::max(1.2f, s * 0.17f);
  switch (m) {
  case Mark::Cross: {
    const float d = s * 0.60f;
    drawSeg(buf, w, h, bgra, cx - d, cy - d, cx + d, cy + d, t, p);
    drawSeg(buf, w, h, bgra, cx - d, cy + d, cx + d, cy - d, t, p);
    break;
  }
  case Mark::Circle:
    drawPill(buf, w, h, bgra, cx, cy, 0.0f, s * 0.62f, kNone, p, t * 2.0f);
    break;
  case Mark::Square: {
    const float d = s * 0.55f;
    drawSeg(buf, w, h, bgra, cx - d, cy - d, cx + d, cy - d, t, p);
    drawSeg(buf, w, h, bgra, cx + d, cy - d, cx + d, cy + d, t, p);
    drawSeg(buf, w, h, bgra, cx + d, cy + d, cx - d, cy + d, t, p);
    drawSeg(buf, w, h, bgra, cx - d, cy + d, cx - d, cy - d, t, p);
    break;
  }
  case Mark::Triangle: {
    const float d = s * 0.70f;
    const float bx = d * 0.87f, by = d * 0.5f;
    drawSeg(buf, w, h, bgra, cx, cy - d, cx + bx, cy + by, t, p);
    drawSeg(buf, w, h, bgra, cx + bx, cy + by, cx - bx, cy + by, t, p);
    drawSeg(buf, w, h, bgra, cx - bx, cy + by, cx, cy - d, t, p);
    break;
  }
  case Mark::Bars:
    for (int i = -1; i <= 1; i++)
      drawSeg(buf, w, h, bgra, cx - s * 0.45f, cy + i * s * 0.42f,
              cx + s * 0.45f, cy + i * s * 0.42f, t * 0.7f, p);
    break;
  case Mark::Pad:
    drawPill(buf, w, h, bgra, cx, cy, s * 0.50f, s * 0.40f, kNone, p, t * 1.3f);
    break;
  case Mark::L1:
    drawLabel(buf, w, h, bgra, kGlyphL, kGlyph1, cx, cy, s * 0.30f, p);
    break;
  case Mark::L2:
    drawLabel(buf, w, h, bgra, kGlyphL, kGlyph2, cx, cy, s * 0.30f, p);
    break;
  case Mark::R1:
    drawLabel(buf, w, h, bgra, kGlyphR, kGlyph1, cx, cy, s * 0.30f, p);
    break;
  case Mark::R2:
    drawLabel(buf, w, h, bgra, kGlyphR, kGlyph2, cx, cy, s * 0.30f, p);
    break;
  }
}

Paint markPaint(Mark m, bool on) {
  const float a = on ? 1.0f : 0.62f;
  switch (m) {
  case Mark::Cross:
    return {108, 160, 255, a};
  case Mark::Circle:
    return {240, 106, 106, a};
  case Mark::Square:
    return {232, 130, 200, a};
  case Mark::Triangle:
    return {86, 210, 168, a};
  default:
    return {226, 232, 245, a};
  }
}

void drawOverlay(uint8_t *buf, uint32_t w, uint32_t h, bool bgra) {
  Touch t[kMaxTouch];
  int n;
  float vx, vy, vw, vh;
  {
    std::lock_guard<std::mutex> lk(g_inMutex);
    n = g_touchCount;
    std::memcpy(t, g_touches, sizeof(Touch) * (n < kMaxTouch ? n : kMaxTouch));
    vx = g_viewX;
    vy = g_viewY;
    vw = g_viewW;
    vh = g_viewH;
  }
  if (vw <= 0.0f || vh <= 0.0f)
    return;

  const float A = vw / vh;
  const Layout l = buildLayout(A);
  const float U = float(h); // height unit -> framebuffer pixels

  // Touches in height units, so press state is tested exactly as computePad
  // resolves it.
  float hx[kMaxTouch], hy[kMaxTouch];
  int m = 0;
  for (int i = 0; i < n; i++) {
    const float ux = (t[i].x - vx) / vh, uy = (t[i].y - vy) / vh;
    if (ux < 0.0f || uy < 0.0f || ux > A || uy > 1.0f)
      continue;
    hx[m] = ux;
    hy[m] = uy;
    m++;
  }
  auto held = [&](const Pill &p) {
    for (int i = 0; i < m; i++)
      if (inPill(hx[i], hy[i], p, 0.012f))
        return true;
    return false;
  };

  for (int b = 0; b < l.n; b++) {
    const Button &btn = l.btns[b];
    const bool on = held(btn.p);
    const float cx = btn.p.cx * U, cy = btn.p.cy * U;
    const float r = btn.p.r * U, hwp = btn.p.hw * U;
    drawPill(buf, w, h, bgra, cx, cy + r * 0.10f, hwp, r, kShadow, kNone, 0.0f);
    drawPill(buf, w, h, bgra, cx, cy, hwp, r, on ? kGlassOn : kGlass,
             on ? kRimOn : kRim, std::max(1.5f, r * 0.075f));
    drawMark(buf, w, h, bgra, btn.mark, cx, cy, r * 0.60f,
             markPaint(btn.mark, on));
  }

  // D-pad: a plus of capsules, each arm lit by its own direction.
  {
    const float cx = l.dpadCx * U, cy = l.dpadCy * U, arm = l.dpadArm * U;
    const float t2 = arm * 0.40f;
    float ddx = 0.0f, ddy = 0.0f;
    bool active = false;
    for (int i = 0; i < m; i++) {
      const float ax = hx[i] - l.dpadCx, ay = hy[i] - l.dpadCy;
      if (std::fabs(ax) <= l.dpadArm * 1.5f && std::fabs(ay) <= l.dpadArm * 1.5f) {
        ddx = ax;
        ddy = ay;
        active = true;
      }
    }
    const float dz = l.dpadArm * 0.32f;
    const bool left = active && ddx < -dz, right = active && ddx > dz;
    const bool up = active && ddy < -dz, down = active && ddy > dz;
    drawSeg(buf, w, h, bgra, cx - arm, cy, cx + arm, cy, t2, kShadow);
    drawSeg(buf, w, h, bgra, cx, cy - arm, cx, cy + arm, t2, kShadow);
    drawSeg(buf, w, h, bgra, cx - arm, cy, cx, cy, t2 * 0.92f,
            left ? kGlassOn : kGlass);
    drawSeg(buf, w, h, bgra, cx, cy, cx + arm, cy, t2 * 0.92f,
            right ? kGlassOn : kGlass);
    drawSeg(buf, w, h, bgra, cx, cy - arm, cx, cy, t2 * 0.92f,
            up ? kGlassOn : kGlass);
    drawSeg(buf, w, h, bgra, cx, cy, cx, cy + arm, t2 * 0.92f,
            down ? kGlassOn : kGlass);
    const Paint tip{226, 232, 245, 0.55f};
    const float ts = arm * 0.16f;
    drawSeg(buf, w, h, bgra, cx - arm * 0.70f, cy, cx - arm * 0.44f, cy, ts,
            left ? kRimOn : tip);
    drawSeg(buf, w, h, bgra, cx + arm * 0.44f, cy, cx + arm * 0.70f, cy, ts,
            right ? kRimOn : tip);
    drawSeg(buf, w, h, bgra, cx, cy - arm * 0.70f, cx, cy - arm * 0.44f, ts,
            up ? kRimOn : tip);
    drawSeg(buf, w, h, bgra, cx, cy + arm * 0.44f, cx, cy + arm * 0.70f, ts,
            down ? kRimOn : tip);
  }

  // Sticks: a recessed well with a knob that follows the thumb.
  const Pill *sticks[] = {&l.lstick, &l.rstick};
  for (const Pill *s : sticks) {
    const float cx = s->cx * U, cy = s->cy * U, r = s->r * U;
    float kx = cx, ky = cy;
    bool on = false;
    for (int i = 0; i < m; i++) {
      const float dx = (hx[i] - s->cx) / kStickR, dy = (hy[i] - s->cy) / kStickR;
      if (std::fabs(dx) > 1.9f || std::fabs(dy) > 1.9f)
        continue;
      const bool mine = (s == &l.lstick) ? (hx[i] < A * 0.5f) : (hx[i] >= A * 0.5f);
      if (!mine)
        continue;
      kx = cx + std::clamp(dx, -1.0f, 1.0f) * r;
      ky = cy + std::clamp(dy, -1.0f, 1.0f) * r;
      on = true;
    }
    drawPill(buf, w, h, bgra, cx, cy, 0.0f, r, Paint{8, 10, 14, 0.22f},
             Paint{224, 231, 245, on ? 0.40f : 0.22f}, std::max(1.5f, r * 0.035f));
    drawPill(buf, w, h, bgra, kx, ky + r * 0.05f, 0.0f, r * 0.42f, kShadow,
             kNone, 0.0f);
    drawPill(buf, w, h, bgra, kx, ky, 0.0f, r * 0.42f,
             on ? kGlassOn : Paint{18, 21, 28, 0.42f}, on ? kRimOn : kRim,
             std::max(1.5f, r * 0.05f));
  }
}

uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g.phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((typeBits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  return UINT32_MAX;
}

void imageBarrier(VkCommandBuffer c, VkImage img, VkImageLayout from,
                  VkImageLayout to, VkAccessFlags srcA, VkAccessFlags dstA,
                  VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  b.srcAccessMask = srcA;
  b.dstAccessMask = dstA;
  vkCmdPipelineBarrier(c, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
}

void destroyRenderSemaphores() {
  for (VkSemaphore sem : g.retiredRenderSems)
    vkDestroySemaphore(g.device, sem, nullptr);
  g.retiredRenderSems.clear();
  for (VkSemaphore sem : g.renderSems)
    vkDestroySemaphore(g.device, sem, nullptr);
  g.renderSems.clear();
}

// Park the current semaphores instead of destroying them: the presentation
// engine may still wait on one after vkDeviceWaitIdle returns (that guarantee
// needs VK_EXT_swapchain_maintenance1). Whatever the previous recreation
// parked is destroyed now -- a full swapchain generation later.
void retireRenderSemaphores() {
  for (VkSemaphore sem : g.retiredRenderSems)
    vkDestroySemaphore(g.device, sem, nullptr);
  g.retiredRenderSems = std::move(g.renderSems);
  g.renderSems.clear();
}

bool createRenderSemaphores(uint32_t count,
                            std::vector<VkSemaphore> &semaphores) {
  VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  semaphores.resize(count);
  for (VkSemaphore &sem : semaphores) {
    if (vkCreateSemaphore(g.device, &si, nullptr, &sem) != VK_SUCCESS) {
      for (VkSemaphore created : semaphores) {
        if (created)
          vkDestroySemaphore(g.device, created, nullptr);
      }
      semaphores.clear();
      return false;
    }
  }
  return true;
}

bool createSwapchain() {
  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g.phys, g.surface, &caps);

  uint32_t nfmt = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(g.phys, g.surface, &nfmt, nullptr);
  std::vector<VkSurfaceFormatKHR> fmts(nfmt);
  vkGetPhysicalDeviceSurfaceFormatsKHR(g.phys, g.surface, &nfmt, fmts.data());
  VkSurfaceFormatKHR chosen = fmts[0];
  for (auto &f : fmts)
    if (f.format == VK_FORMAT_R8G8B8A8_UNORM ||
        f.format == VK_FORMAT_B8G8R8A8_UNORM)
      chosen = f;
  // Opt out of Android pre-rotation: phones report currentTransform=ROTATE_90/
  // 270 for a landscape window (the panel is natively portrait) and expect the
  // app to render rotated. We present via a plain blit (can't rotate), so we
  // ask the compositor to do the rotation by choosing IDENTITY. currentExtent
  // is in the pre-rotated basis, so swap W/H to get the identity (display)
  // extent.
  const VkSurfaceTransformFlagsKHR rotated =
      VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR |
      VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR;
  bool preRotated = (caps.currentTransform & rotated) != 0;
  g.preTransform =
      (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
          ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
          : caps.currentTransform;
  std::printf(
      "[gfx-android] surface transform=%#x supported=%#x -> using %#x\n",
      caps.currentTransform, caps.supportedTransforms, g.preTransform);

  VkExtent2D ext = caps.currentExtent;
  if (ext.width == 0xFFFFFFFF) {
    ext.width = (uint32_t)ANativeWindow_getWidth(g.window);
    ext.height = (uint32_t)ANativeWindow_getHeight(g.window);
  }
  if (g.preTransform == VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR && preRotated)
    std::swap(ext.width, ext.height);
  if (ext.width == 0 || ext.height == 0)
    return false;
  g.swapExtent = ext;

  uint32_t imgCount = caps.minImageCount + 1;
  if (caps.maxImageCount && imgCount > caps.maxImageCount)
    imgCount = caps.maxImageCount;

  VkSwapchainCreateInfoKHR sc{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  sc.surface = g.surface;
  sc.minImageCount = imgCount;
  sc.imageFormat = chosen.format;
  sc.imageColorSpace = chosen.colorSpace;
  sc.imageExtent = ext;
  sc.imageArrayLayers = 1;
  sc.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  sc.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  sc.preTransform = g.preTransform;
  sc.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  sc.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  sc.clipped = VK_TRUE;
  const VkSwapchainKHR oldSwapchain = g.swapchain;
  sc.oldSwapchain = oldSwapchain;

  // Replacement is exceptional. Idle once so old images and their render-
  // finished semaphores can be destroyed together after the new set is ready.
  if (g.swapchain)
    vkDeviceWaitIdle(g.device);

  auto discardRetiredSwapchain = [&] {
    if (!oldSwapchain)
      return;
    retireRenderSemaphores();
    vkDestroySwapchainKHR(g.device, oldSwapchain, nullptr);
    g.swapchain = VK_NULL_HANDLE;
    g.swapImages.clear();
  };

  VkSwapchainKHR newSwap = VK_NULL_HANDLE;
  const VkResult createResult =
      vkCreateSwapchainKHR(g.device, &sc, nullptr, &newSwap);
  if (createResult != VK_SUCCESS) {
    discardRetiredSwapchain();
    std::fprintf(stderr,
                 "[gfx-android] vkCreateSwapchainKHR failed: VkResult=%d\n",
                 createResult);
    return false;
  }

  uint32_t n = 0;
  vkGetSwapchainImagesKHR(g.device, newSwap, &n, nullptr);
  std::vector<VkImage> newImages(n);
  vkGetSwapchainImagesKHR(g.device, newSwap, &n, newImages.data());
  std::vector<VkSemaphore> newRenderSems;
  if (!createRenderSemaphores(n, newRenderSems)) {
    discardRetiredSwapchain();
    vkDestroySwapchainKHR(g.device, newSwap, nullptr);
    return false;
  }

  retireRenderSemaphores();
  if (g.swapchain)
    vkDestroySwapchainKHR(g.device, g.swapchain, nullptr);

  g.swapchain = newSwap;
  g.swapFormat = chosen.format;
  g.swapImages.swap(newImages);
  g.renderSems.swap(newRenderSems);
  g.needRecreate = false;
  return true;
}

void destroyFrameResources(FrameSlot &slot) {
  if (slot.stagingMap) {
    vkUnmapMemory(g.device, slot.stagingMem);
    slot.stagingMap = nullptr;
  }
  if (slot.staging)
    vkDestroyBuffer(g.device, slot.staging, nullptr);
  if (slot.stagingMem)
    vkFreeMemory(g.device, slot.stagingMem, nullptr);
  if (slot.frameImg)
    vkDestroyImage(g.device, slot.frameImg, nullptr);
  if (slot.frameMem)
    vkFreeMemory(g.device, slot.frameMem, nullptr);
  slot.staging = VK_NULL_HANDLE;
  slot.stagingMem = VK_NULL_HANDLE;
  slot.frameImg = VK_NULL_HANDLE;
  slot.frameMem = VK_NULL_HANDLE;
}

bool createFrameResources(FrameSlot &slot, uint32_t w, uint32_t h,
                          VkFormat fmt) {
  VkDeviceSize size = (VkDeviceSize)w * h * 4;
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(g.device, &bi, nullptr, &slot.staging));
  VkMemoryRequirements br;
  vkGetBufferMemoryRequirements(g.device, slot.staging, &br);
  VkMemoryAllocateInfo ba{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ba.allocationSize = br.size;
  ba.memoryTypeIndex = findMemoryType(br.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(g.device, &ba, nullptr, &slot.stagingMem));
  VK_CHECK(vkBindBufferMemory(g.device, slot.staging, slot.stagingMem, 0));
  VK_CHECK(
      vkMapMemory(g.device, slot.stagingMem, 0, size, 0, &slot.stagingMap));

  VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ii.imageType = VK_IMAGE_TYPE_2D;
  ii.format = fmt;
  ii.extent = {w, h, 1};
  ii.mipLevels = 1;
  ii.arrayLayers = 1;
  ii.samples = VK_SAMPLE_COUNT_1_BIT;
  ii.tiling = VK_IMAGE_TILING_OPTIMAL;
  ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VK_CHECK(vkCreateImage(g.device, &ii, nullptr, &slot.frameImg));
  VkMemoryRequirements ir;
  vkGetImageMemoryRequirements(g.device, slot.frameImg, &ir);
  VkMemoryAllocateInfo ia{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ia.allocationSize = ir.size;
  ia.memoryTypeIndex =
      findMemoryType(ir.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VK_CHECK(vkAllocateMemory(g.device, &ia, nullptr, &slot.frameMem));
  VK_CHECK(vkBindImageMemory(g.device, slot.frameImg, slot.frameMem, 0));
  return true;
}

bool ensureFrameResources(uint32_t w, uint32_t h, VkFormat fmt) {
  bool ready = true;
  for (const FrameSlot &slot : g.slots)
    ready &= slot.staging != VK_NULL_HANDLE && slot.frameImg != VK_NULL_HANDLE;
  if (g.fbW == w && g.fbH == h && g.fbFormat == fmt && ready)
    return true;
  vkDeviceWaitIdle(g.device);
  for (FrameSlot &slot : g.slots)
    destroyFrameResources(slot);
  g.fbW = w;
  g.fbH = h;
  g.fbFormat = fmt;
  for (FrameSlot &slot : g.slots)
    if (!createFrameResources(slot, w, h, fmt))
      return false;
  return true;
}

// Full bring-up against the current g.window: instance, android surface,
// device, command/sync objects and the swapchain.
bool bringUp() {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "prosperity";
  app.apiVersion = VK_API_VERSION_1_1;
  const char *exts[] = {"VK_KHR_surface", "VK_KHR_android_surface"};
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  ici.enabledExtensionCount = 2;
  ici.ppEnabledExtensionNames = exts;
  VK_CHECK(vkCreateInstance(&ici, nullptr, &g.instance));

  VkAndroidSurfaceCreateInfoKHR si{
      VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
  si.window = g.window;
  VK_CHECK(vkCreateAndroidSurfaceKHR(g.instance, &si, nullptr, &g.surface));

  uint32_t nphys = 0;
  vkEnumeratePhysicalDevices(g.instance, &nphys, nullptr);
  if (!nphys) {
    std::fprintf(stderr, "[gfx-android] no Vulkan physical devices\n");
    return false;
  }
  std::vector<VkPhysicalDevice> phs(nphys);
  vkEnumeratePhysicalDevices(g.instance, &nphys, phs.data());
  bool found = false;
  for (auto pd : phs) {
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
    for (uint32_t i = 0; i < nq; i++) {
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, g.surface, &present);
      if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
        g.phys = pd;
        g.queueFamily = i;
        found = true;
        break;
      }
    }
    if (found)
      break;
  }
  if (!found) {
    std::fprintf(stderr, "[gfx-android] no graphics+present queue\n");
    return false;
  }
  {
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(g.phys, &pp);
    std::printf("[gfx-android] device: %s\n", pp.deviceName);
  }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = g.queueFamily;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  const char *devExts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = 1;
  dci.ppEnabledExtensionNames = devExts;
  VK_CHECK(vkCreateDevice(g.phys, &dci, nullptr, &g.device));
  vkGetDeviceQueue(g.device, g.queueFamily, 0, &g.queue);

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = g.queueFamily;
  VK_CHECK(vkCreateCommandPool(g.device, &pci, nullptr, &g.cmdPool));
  VkCommandBufferAllocateInfo cbi{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbi.commandPool = g.cmdPool;
  cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbi.commandBufferCount = kFrameSlotCount;
  std::array<VkCommandBuffer, kFrameSlotCount> commands{};
  VK_CHECK(vkAllocateCommandBuffers(g.device, &cbi, commands.data()));

  VkSemaphoreCreateInfo si2{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  VkFenceCreateInfo acquireFi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  for (uint32_t i = 0; i < kFrameSlotCount; i++) {
    g.slots[i].cmd = commands[i];
    VK_CHECK(
        vkCreateSemaphore(g.device, &si2, nullptr, &g.slots[i].acquireSem));
    VK_CHECK(
        vkCreateFence(g.device, &acquireFi, nullptr, &g.slots[i].acquireFence));
    VK_CHECK(vkCreateFence(g.device, &fi, nullptr, &g.slots[i].fence));
  }

  if (!createSwapchain())
    return false;
  std::printf("[gfx-android] swapchain %ux%u, %u images\n", g.swapExtent.width,
              g.swapExtent.height, (uint32_t)g.swapImages.size());
  g.ready = true;
  return true;
}

// Tear everything down (window lost). The guest GPU renderer has its own Vulkan
// device, so dropping ours only stops presentation; it resumes on re-init.
void teardown() {
  if (g.device)
    vkDeviceWaitIdle(g.device);
  for (FrameSlot &slot : g.slots) {
    destroyFrameResources(slot);
    if (slot.fence)
      vkDestroyFence(g.device, slot.fence, nullptr);
    if (slot.acquireFence)
      vkDestroyFence(g.device, slot.acquireFence, nullptr);
    if (slot.acquireSem)
      vkDestroySemaphore(g.device, slot.acquireSem, nullptr);
  }
  destroyRenderSemaphores();
  if (g.cmdPool)
    vkDestroyCommandPool(g.device, g.cmdPool, nullptr);
  if (g.swapchain)
    vkDestroySwapchainKHR(g.device, g.swapchain, nullptr);
  if (g.device)
    vkDestroyDevice(g.device, nullptr);
  if (g.surface)
    vkDestroySurfaceKHR(g.instance, g.surface, nullptr);
  if (g.instance)
    vkDestroyInstance(g.instance, nullptr);
  ANativeWindow *keep = g.window;
  g = State{};
  g.window = keep;
  g_presentFailed.store(false, std::memory_order_release);
}

} // namespace

// --- public gfx API ---------------------------------------------------------

bool init(const char *, uint32_t, uint32_t) {
  if (available())
    return true;
  if (g.ready)
    return createSwapchain();
  if (g.instance)
    teardown();
  return g.window && bringUp();
}

bool available() {
  return g.ready && g.window != nullptr && g.swapchain != VK_NULL_HANDLE;
}

bool canPresent() {
  std::lock_guard<std::mutex> lk(g_inMutex);
  return !g_presentStopRequested.load(std::memory_order_acquire) &&
         (g_windowChanged ||
          !g_presentFailed.load(std::memory_order_acquire)) &&
         (g_pendingWindow != nullptr || g.window != nullptr);
}

void requestPresentStop() {
  g_presentStopRequested.store(true, std::memory_order_release);
}

bool ensure(const char *, uint32_t, uint32_t) {
  // Adopt any window change published by android_main (this thread owns
  // Vulkan).
  {
    std::lock_guard<std::mutex> lk(g_inMutex);
    if (g_windowChanged) {
      g_windowChanged = false;
      if (g_pendingWindow != g.window ||
          g_presentFailed.load(std::memory_order_acquire)) {
        if (g.instance)
          teardown(); // resets g, preserves g.window
        else
          g_presentFailed.store(false, std::memory_order_release);
        g.window = g_pendingWindow;
      }
    }
  }
  if (g_presentFailed.load(std::memory_order_acquire) ||
      g_presentStopRequested.load(std::memory_order_acquire))
    return false;
  if (!g.window)
    return false;
  if (!g.instance)
    return bringUp();
  if (!g.ready) {
    teardown();
    return g.window && bringUp();
  }
  if (g.needRecreate)
    createSwapchain();
  return available();
}

void present(const void *pixels, uint32_t w, uint32_t h, uint32_t srcPitch,
             PixelFormat fmt) {
  if (g_presentFailed.load(std::memory_order_acquire) ||
      g_presentStopRequested.load(std::memory_order_acquire) || !g.device ||
      !g.swapchain || !pixels || !w || !h)
    return;
  if (g.needRecreate && !createSwapchain())
    return;
  if (srcPitch == 0)
    srcPitch = w * 4;
  VkFormat vkfmt = (fmt == PixelFormat::bgra8) ? VK_FORMAT_B8G8R8A8_UNORM
                                               : VK_FORMAT_R8G8B8A8_UNORM;
  if (!ensureFrameResources(w, h, vkfmt))
    return;

  FrameSlot &slot = g.slots[g.nextSlot];
  // The previous submission may still be reading this slot's mapped buffer.
  // Host writes and the CPU overlay must wait for that use to finish.
  if (!waitForPresentFence(slot.fence, "vkWaitForFences"))
    return;

  // Fit the frame into the panel without distorting it: phones are far wider
  // than 16:9, and stretching to fill made everything ~20% too wide.
  //
  // The swapchain image is portrait while the window is landscape (the
  // compositor scales each axis on its own rather than rotating), so the fit
  // has to be solved in the landscape space the player actually sees and the
  // resulting rect mapped back into image space for the blit.
  const int32_t iw = (int32_t)g.swapExtent.width;
  const int32_t ih = (int32_t)g.swapExtent.height;
  const int32_t scrW = std::max(iw, ih), scrH = std::min(iw, ih);
  int32_t vw = scrW, vh = (int32_t)((int64_t)scrW * h / w);
  if (vh > scrH) {
    vh = scrH;
    vw = (int32_t)((int64_t)scrH * w / h);
  }
  const int32_t vx = (scrW - vw) / 2, vy = (scrH - vh) / 2;
  {
    std::lock_guard<std::mutex> lk(g_inMutex);
    g_viewX = float(vx);
    g_viewY = float(vy);
    g_viewW = float(vw);
    g_viewH = float(vh);
  }
  const int32_t dx0 = (int32_t)((int64_t)vx * iw / scrW);
  const int32_t dx1 = (int32_t)((int64_t)(vx + vw) * iw / scrW);
  const int32_t dy0 = (int32_t)((int64_t)vy * ih / scrH);
  const int32_t dy1 = (int32_t)((int64_t)(vy + vh) * ih / scrH);

  auto *dst = static_cast<uint8_t *>(slot.stagingMap);
  auto *src = static_cast<const uint8_t *>(pixels);
  for (uint32_t y = 0; y < h; y++)
    std::memcpy(dst + (size_t)y * w * 4, src + (size_t)y * srcPitch, w * 4);

  // Composite the virtual-gamepad helper over the frame.
  drawOverlay(dst, w, h, fmt == PixelFormat::bgra8);

  uint32_t idx = 0;
  VkResult result = vkResetFences(g.device, 1, &slot.acquireFence);
  if (result != VK_SUCCESS) {
    stopPresenting("vkResetFences(acquire)", result);
    return;
  }
  VkResult ar;
  do {
    ar = vkAcquireNextImageKHR(g.device, g.swapchain, kPresentWaitSliceNs,
                               slot.acquireSem, slot.acquireFence, &idx);
  } while (ar == VK_TIMEOUT &&
           !g_presentFailed.load(std::memory_order_acquire) &&
           !g_presentStopRequested.load(std::memory_order_acquire));
  if (g_presentStopRequested.load(std::memory_order_acquire))
    return;
  if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
    g.needRecreate = true;
    return;
  }
  if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) {
    stopPresenting("vkAcquireNextImageKHR", ar);
    return;
  }
  if (!waitForPresentFence(slot.acquireFence, "vkWaitForFences(acquire)"))
    return;

  result = vkResetCommandBuffer(slot.cmd, 0);
  if (result != VK_SUCCESS) {
    stopPresenting("vkResetCommandBuffer", result);
    return;
  }
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  result = vkBeginCommandBuffer(slot.cmd, &bi);
  if (result != VK_SUCCESS) {
    stopPresenting("vkBeginCommandBuffer", result);
    return;
  }

  imageBarrier(slot.cmd, slot.frameImg, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
  VkBufferImageCopy cp{};
  cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  cp.imageExtent = {w, h, 1};
  vkCmdCopyBufferToImage(slot.cmd, slot.staging, slot.frameImg,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

  imageBarrier(slot.cmd, slot.frameImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
  imageBarrier(slot.cmd, g.swapImages[idx], VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);

  if (dx0 > 0 || dy0 > 0 || dx1 < iw || dy1 < ih) {
    const VkClearColorValue black{{0.0f, 0.0f, 0.0f, 1.0f}};
    const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(slot.cmd, g.swapImages[idx],
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &all);
  }

  VkImageBlit blit{};
  blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.srcOffsets[1] = {(int32_t)w, (int32_t)h, 1};
  blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blit.dstOffsets[0] = {dx0, dy0, 0};
  blit.dstOffsets[1] = {dx1, dy1, 1};
  vkCmdBlitImage(slot.cmd, slot.frameImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 g.swapImages[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                 &blit, VK_FILTER_LINEAR);

  imageBarrier(
      slot.cmd, g.swapImages[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  result = vkEndCommandBuffer(slot.cmd);
  if (result != VK_SUCCESS) {
    stopPresenting("vkEndCommandBuffer", result);
    return;
  }

  VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkSubmitInfo subi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  subi.waitSemaphoreCount = 1;
  subi.pWaitSemaphores = &slot.acquireSem;
  subi.pWaitDstStageMask = &waitStage;
  subi.commandBufferCount = 1;
  subi.pCommandBuffers = &slot.cmd;
  subi.signalSemaphoreCount = 1;
  subi.pSignalSemaphores = &g.renderSems[idx];
  result = vkResetFences(g.device, 1, &slot.fence);
  if (result != VK_SUCCESS) {
    stopPresenting("vkResetFences", result);
    return;
  }
  result = vkQueueSubmit(g.queue, 1, &subi, slot.fence);
  if (result != VK_SUCCESS) {
    stopPresenting("vkQueueSubmit", result);
    return;
  }
  g.nextSlot = (g.nextSlot + 1) % kFrameSlotCount;

  VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  pi.waitSemaphoreCount = 1;
  pi.pWaitSemaphores = &g.renderSems[idx];
  pi.swapchainCount = 1;
  pi.pSwapchains = &g.swapchain;
  pi.pImageIndices = &idx;
  VkResult pr = vkQueuePresentKHR(g.queue, &pi);
  if (pr == VK_ERROR_OUT_OF_DATE_KHR)
    g.needRecreate = true;
  else if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR)
    stopPresenting("vkQueuePresentKHR", pr);
}

// Lifecycle is driven by android_main's looper; nothing to pump here.
bool pumpEvents() { return true; }

bool pollKeyboardPad(PadKeys &out) {
  std::lock_guard<std::mutex> lk(g_inMutex);
  out = computePad();
  return true;
}

// Haptics on the touch-gamepad build would need a JNI call to the Vibrator
// service; no-op until that's wired through the NativeActivity.
void setRumble(uint8_t, uint8_t) {}

// The activity is fullscreen and has no title bar; the perf numbers already
// reach logcat through the renderer's [fps] line.
void setTitle(const char *) {}
void overlaySetPerf(float, float, float) {}

void shutdown() { teardown(); }

void setAndroidWindow(ANativeWindow *window) {
  std::lock_guard<std::mutex> lk(g_inMutex);
  g_pendingWindow = window;
  g_windowChanged = true;
}

void setAndroidTouches(const Touch *pts, int count) {
  std::lock_guard<std::mutex> lk(g_inMutex);
  g_touchCount = count < 0 ? 0 : (count > kMaxTouch ? kMaxTouch : count);
  for (int i = 0; i < g_touchCount; i++)
    g_touches[i] = pts[i];
}

} // namespace gfx

#endif // __ANDROID__ && DELTA_ANDROID_APP
