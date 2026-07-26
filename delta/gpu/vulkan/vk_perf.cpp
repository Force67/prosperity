/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "vulkan/vk_perf.h"

#include "gfx/overlay.h"

#include <cstdio>
#include <cstdlib>

namespace gpu::vk {

uint64_t g_nsDraw = 0, g_nsEnd = 0, g_nsReadback = 0, g_nsTexUp = 0;
uint64_t g_nsCs = 0, g_csBytes = 0;
uint64_t g_nsCsIn = 0, g_nsCsGpu = 0, g_nsCsOut = 0;
uint32_t g_csCount = 0;
uint64_t g_nsSubmit = 0, g_nsPresent = 0;
uint32_t g_texUps = 0;
uint32_t g_csStageN = 0, g_csFlushN = 0;
uint64_t g_csStageBytes = 0;
uint64_t g_frDraw = 0, g_frSubmit = 0, g_frWait = 0, g_frPresent = 0,
         g_frTexUp = 0;

namespace {

// Rolling per-frame stage history for the overlay graph (~4s at 60 fps).
struct StageSample {
  float rec, sub, gpu, prs, tex, oth, wall;  // ms
};

constexpr int kStageHistN = 240;
StageSample g_stageHist[kStageHistN];
int g_stageHistPos = 0, g_stageHistCount = 0;

// 3x5 bitmap font (rows top-down, bit 2 = left pixel). Uppercase + digits only.
const uint8_t *ovGlyph(char c) {
  struct Glyph { char c; uint8_t rows[5]; };
  static const Glyph f[] = {
      {'0', {7, 5, 5, 5, 7}}, {'1', {2, 6, 2, 2, 7}}, {'2', {7, 1, 7, 4, 7}},
      {'3', {7, 1, 7, 1, 7}}, {'4', {5, 5, 7, 1, 1}}, {'5', {7, 4, 7, 1, 7}},
      {'6', {7, 4, 7, 5, 7}}, {'7', {7, 1, 1, 2, 2}}, {'8', {7, 5, 7, 5, 7}},
      {'9', {7, 5, 7, 1, 7}}, {'.', {0, 0, 0, 0, 2}}, {'A', {7, 5, 7, 5, 5}},
      {'B', {6, 5, 6, 5, 6}}, {'C', {7, 4, 4, 4, 7}}, {'D', {6, 5, 5, 5, 6}},
      {'E', {7, 4, 7, 4, 7}}, {'F', {7, 4, 7, 4, 4}}, {'G', {7, 4, 5, 5, 7}},
      {'H', {5, 5, 7, 5, 5}}, {'I', {7, 2, 2, 2, 7}}, {'L', {4, 4, 4, 4, 7}},
      {'M', {5, 7, 7, 5, 5}}, {'N', {5, 7, 7, 7, 5}}, {'O', {7, 5, 5, 5, 7}},
      {'P', {7, 5, 7, 4, 4}}, {'R', {7, 5, 6, 5, 5}}, {'S', {7, 4, 7, 1, 7}},
      {'T', {7, 2, 2, 2, 2}}, {'U', {5, 5, 5, 5, 7}}, {'V', {5, 5, 5, 5, 2}},
      {'W', {5, 5, 7, 7, 5}}, {'X', {5, 5, 2, 5, 5}},
  };
  for (const Glyph &gl : f)
    if (gl.c == c) return gl.rows;
  return nullptr;  // unknown/space -> blank
}

inline void ovFill(uint8_t *b, uint32_t w, uint32_t h, int x, int y, int fw,
                   int fh, uint32_t bgra) {
  if (x < 0 || y < 0) return;
  for (int yy = y; yy < y + fh && yy < (int)h; yy++) {
    uint32_t *row = reinterpret_cast<uint32_t *>(b + (size_t)yy * w * 4);
    for (int xx = x; xx < x + fw && xx < (int)w; xx++) row[xx] = bgra;
  }
}

void ovText(uint8_t *b, uint32_t w, uint32_t h, int x, int y, int scale,
            uint32_t bgra, const char *s) {
  for (; *s; s++, x += 4 * scale) {
    const uint8_t *rows = ovGlyph(*s);
    if (!rows) continue;
    for (int ry = 0; ry < 5; ry++)
      for (int rx = 0; rx < 3; rx++)
        if (rows[ry] & (4 >> rx))
          ovFill(b, w, h, x + rx * scale, y + ry * scale, scale, scale, bgra);
  }
}

}  // namespace

// Stacked per-stage frame-time columns drawn over the presented image (default
// on; DELTA_GPU_OVERLAY=0 disables). One column per frame, 2 px per ms:
//   green  REC  command recording + per-draw analysis (rhi::draw)
//   yellow SUB  command-buffer end + queue submit
//   red    GPU  fence wait (the rasterizer)
//   blue   PRS  window present (SDL blit)
//   purple TEX  synchronous texture uploads
//   gray   OTH  everything else (guest emulation between frames)
// Drawn AFTER the PPM capture paths so dumps stay clean.

void pushStageSample() {
  static uint64_t prevNs = 0;
  const uint64_t now = nowNs();
  const float wall = prevNs ? (now - prevNs) / 1e6f : 0.0f;
  prevNs = now;
  StageSample s;
  s.rec = g_frDraw / 1e6f;
  s.sub = g_frSubmit / 1e6f;
  s.gpu = g_frWait / 1e6f;
  s.prs = g_frPresent / 1e6f;  // accrued after last frame's sample (1-frame lag)
  s.tex = g_frTexUp / 1e6f;
  const float known = s.rec + s.sub + s.gpu + s.prs + s.tex;
  s.oth = wall > known ? wall - known : 0.0f;
  s.wall = wall;
  g_frDraw = g_frSubmit = g_frWait = g_frPresent = g_frTexUp = 0;
  g_stageHist[g_stageHistPos] = s;
  g_stageHistPos = (g_stageHistPos + 1) % kStageHistN;
  if (g_stageHistCount < kStageHistN) g_stageHistCount++;
}

void drawPerfOverlay(uint8_t *bgra, uint32_t w, uint32_t h) {
  static const bool off = [] {
    const char *e = std::getenv("DELTA_GPU_OVERLAY");
    return e && e[0] == '0';
  }();
  if (off || !g_stageHistCount || w < 560 || h < 280) return;
  // BGRA little-endian constants (0xAARRGGBB written as a uint32).
  static constexpr uint32_t kCol[6] = {
      0xFF32C832,  // REC green
      0xFFC8C828,  // SUB yellow
      0xFFE63232,  // GPU red
      0xFF3288E6,  // PRS blue
      0xFFC832C8,  // TEX purple
      0xFF828282,  // OTH gray
  };
  static const char *kLabel[6] = {"REC", "SUB", "GPU", "PRS", "TEX", "OTH"};
  constexpr int colW = 2, graphH = 120;
  constexpr float pxPerMs = 2.0f;
  const int graphW = kStageHistN * colW;
  const int x0 = 10, y1 = (int)h - 10, y0 = y1 - graphH;
  const int legendH = 7 * 14 + 4;
  const int panelY0 = y0 - legendH - 4;
  // Darken the panel background (keeps the game visible underneath).
  for (int yy = panelY0 - 4; yy < y1 + 4 && yy < (int)h; yy++) {
    if (yy < 0) continue;
    uint32_t *row = reinterpret_cast<uint32_t *>(bgra + (size_t)yy * w * 4);
    for (int xx = x0 - 4; xx < x0 + graphW + 4 && xx < (int)w; xx++)
      row[xx] = (row[xx] >> 2) & 0x3F3F3F3F;
  }
  // Columns: oldest left, newest right; stages stacked bottom-up.
  for (int i = 0; i < kStageHistN; i++) {
    int idx = g_stageHistPos - kStageHistN + i;
    if (idx < g_stageHistPos - g_stageHistCount) continue;  // no sample yet
    idx = ((idx % kStageHistN) + kStageHistN) % kStageHistN;
    const StageSample &s = g_stageHist[idx];
    const float vals[6] = {s.rec, s.sub, s.gpu, s.prs, s.tex, s.oth};
    int x = x0 + i * colW, yBase = y1;
    for (int st = 0; st < 6; st++) {
      int hpx = (int)(vals[st] * pxPerMs + 0.5f);
      if (yBase - hpx < y0) hpx = yBase - y0;
      if (hpx > 0) ovFill(bgra, w, h, x, yBase - hpx, colW, hpx, kCol[st]);
      yBase -= hpx;
      if (yBase <= y0) break;
    }
  }
  // 60 / 30 fps reference lines.
  for (int xx = x0; xx < x0 + graphW; xx += 3) {
    ovFill(bgra, w, h, xx, y1 - (int)(16.7f * pxPerMs), 1, 1, 0xFFF0F0F0);
    ovFill(bgra, w, h, xx, y1 - (int)(33.3f * pxPerMs), 1, 1, 0xFFF0F0F0);
  }
  // Legend: averages over the last second's worth of samples.
  const int nAvg = g_stageHistCount < 60 ? g_stageHistCount : 60;
  float avg[6] = {}, avgWall = 0;
  for (int i = 1; i <= nAvg; i++) {
    const StageSample &s =
        g_stageHist[((g_stageHistPos - i) % kStageHistN + kStageHistN) %
                    kStageHistN];
    avg[0] += s.rec; avg[1] += s.sub; avg[2] += s.gpu;
    avg[3] += s.prs; avg[4] += s.tex; avg[5] += s.oth;
    avgWall += s.wall;
  }
  for (float &v : avg) v /= nAvg;
  avgWall /= nAvg;
  char buf[64];
  int ty = panelY0;
  std::snprintf(buf, sizeof buf, "FPS %.1f  %.1f MS", avgWall > 0.01f ? 1000.0f / avgWall : 0.0f, avgWall);
  ovText(bgra, w, h, x0, ty, 2, 0xFFFFFFFF, buf);
  ty += 14;
  for (int st = 0; st < 6; st++, ty += 14) {
    ovFill(bgra, w, h, x0, ty + 1, 8, 8, kCol[st]);
    std::snprintf(buf, sizeof buf, "%s %5.1f", kLabel[st], avg[st]);
    ovText(bgra, w, h, x0 + 14, ty, 2, 0xFFE6E6E6, buf);
  }
}

// Wall-clock FPS report. Always on (cheap): every ~2s of presented frames, log
// the average FPS over that window so perf changes can be measured empirically
// (DELTA_GPU_FPS=0 silences it).
void reportFps() {
  static const bool off = [] { const char *e = std::getenv("DELTA_GPU_FPS"); return e && e[0] == '0'; }();
  if (off) return;
  using clock = std::chrono::steady_clock;
  static auto last = clock::now();
  static int frames = 0;
  frames++;
  auto now = clock::now();
  double dt = std::chrono::duration<double>(now - last).count();
  if (dt >= 2.0) {
    double f = frames ? frames : 1;
    std::fprintf(stderr,
        "[fps] %.1f fps | per-frame gpu-code: draw=%.2fms end=%.2fms (wait=%.2fms "
        "submit=%.2fms present=%.2fms) texup=%.2fms x%.1f cs=%.2fms x%.1f "
        "(in=%.2f gpu=%.2f out=%.2f stage=%.1fx%.1fMB flush=%.1f)\n",
        frames / dt, g_nsDraw / f / 1e6, g_nsEnd / f / 1e6, g_nsReadback / f / 1e6,
        g_nsSubmit / f / 1e6, g_nsPresent / f / 1e6,
        g_nsTexUp / f / 1e6, g_texUps / f,
        g_nsCs / f / 1e6, g_csCount / f,
        g_nsCsIn / f / 1e6, g_nsCsGpu / f / 1e6, g_nsCsOut / f / 1e6,
        g_csStageN / f, g_csStageBytes / f / 1e6, g_csFlushN / f);
    // Feed the on-screen overlay gauge (gpuMs = GPU end/present-dominated cost).
    gfx::overlaySetPerf(float(frames / dt), float(g_nsEnd / f / 1e6),
                        float(1000.0 * dt / frames));
    last = now;
    frames = 0;
    g_nsDraw = g_nsEnd = g_nsReadback = g_nsTexUp = 0;
    g_nsSubmit = g_nsPresent = 0;
    g_texUps = 0;
    g_nsCs = g_csBytes = 0;
    g_nsCsIn = g_nsCsGpu = g_nsCsOut = 0;
    g_csCount = g_csStageN = g_csFlushN = 0;
    g_csStageBytes = 0;
  }
}

}  // namespace gpu::vk
