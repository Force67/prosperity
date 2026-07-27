/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_perf.h"

#include "gfx/overlay.h"

#include <cstdio>
#include <cstdlib>

namespace gpu::vk {

uint64_t g_ns_draw = 0, g_ns_end = 0, g_ns_readback = 0, g_ns_tex_up = 0;
uint64_t g_ns_cs = 0, g_cs_bytes = 0;
uint64_t g_ns_cs_in = 0, g_ns_cs_gpu = 0, g_ns_cs_out = 0;
uint32_t g_cs_count = 0;
uint64_t g_ns_submit = 0, g_ns_present = 0;
uint32_t g_tex_ups = 0;
uint32_t g_cs_stage_n = 0, g_cs_flush_n = 0;
uint64_t g_cs_stage_bytes = 0;
uint64_t g_fr_draw = 0, g_fr_submit = 0, g_fr_wait = 0, g_fr_present = 0,
         g_fr_tex_up = 0;

namespace {

// Rolling per-frame stage history for the overlay graph (~4s at 60 fps).
struct StageSample {
  float rec, sub, gpu, prs, tex, oth, wall;  // ms
};

constexpr int kStageHistN = 240;
StageSample g_stage_hist[kStageHistN];
int g_stage_hist_pos = 0, g_stage_hist_count = 0;

// 3x5 bitmap font (rows top-down, bit 2 = left pixel). Uppercase + digits only.
const uint8_t* OvGlyph(char c) {
  struct Glyph {
    char c;
    uint8_t rows[5];
  };
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
  for (const Glyph& gl : f)
    if (gl.c == c)
      return gl.rows;
  return nullptr;  // unknown/space -> blank
}

inline void OvFill(uint8_t* b,
                   uint32_t w,
                   uint32_t h,
                   int x,
                   int y,
                   int fw,
                   int fh,
                   uint32_t bgra) {
  if (x < 0 || y < 0)
    return;
  for (int yy = y; yy < y + fh && yy < (int)h; yy++) {
    uint32_t* row = reinterpret_cast<uint32_t*>(b + (size_t)yy * w * 4);
    for (int xx = x; xx < x + fw && xx < (int)w; xx++)
      row[xx] = bgra;
  }
}

void OvText(uint8_t* b,
            uint32_t w,
            uint32_t h,
            int x,
            int y,
            int scale,
            uint32_t bgra,
            const char* s) {
  for (; *s; s++, x += 4 * scale) {
    const uint8_t* rows = OvGlyph(*s);
    if (!rows)
      continue;
    for (int ry = 0; ry < 5; ry++)
      for (int rx = 0; rx < 3; rx++)
        if (rows[ry] & (4 >> rx))
          OvFill(b, w, h, x + rx * scale, y + ry * scale, scale, scale, bgra);
  }
}

}  // namespace

// Stacked per-stage frame-time columns drawn over the presented image (default
// on; DELTA_GPU_OVERLAY=0 disables). One column per frame, 2 px per ms:
//   green  REC  command recording + per-draw analysis (rhi::Draw)
//   yellow SUB  command-buffer end + queue submit
//   red    GPU  fence wait (the rasterizer)
//   blue   PRS  window present (SDL blit)
//   purple TEX  synchronous texture uploads
//   gray   OTH  everything else (guest emulation between frames)
// Drawn AFTER the PPM capture paths so dumps stay clean.

void PushStageSample() {
  static uint64_t prev_ns = 0;
  const uint64_t now = NowNs();
  const float wall = prev_ns ? (now - prev_ns) / 1e6f : 0.0f;
  prev_ns = now;
  StageSample s;
  s.rec = g_fr_draw / 1e6f;
  s.sub = g_fr_submit / 1e6f;
  s.gpu = g_fr_wait / 1e6f;
  s.prs =
      g_fr_present / 1e6f;  // accrued after last frame's sample (1-frame lag)
  s.tex = g_fr_tex_up / 1e6f;
  const float known = s.rec + s.sub + s.gpu + s.prs + s.tex;
  s.oth = wall > known ? wall - known : 0.0f;
  s.wall = wall;
  g_fr_draw = g_fr_submit = g_fr_wait = g_fr_present = g_fr_tex_up = 0;
  g_stage_hist[g_stage_hist_pos] = s;
  g_stage_hist_pos = (g_stage_hist_pos + 1) % kStageHistN;
  if (g_stage_hist_count < kStageHistN)
    g_stage_hist_count++;
}

void DrawPerfOverlay(uint8_t* bgra, uint32_t w, uint32_t h) {
  static const bool off = [] {
    const char* e = std::getenv("DELTA_GPU_OVERLAY");
    return e && e[0] == '0';
  }();
  if (off || !g_stage_hist_count || w < 560 || h < 280)
    return;
  // BGRA little-endian constants (0xAARRGGBB written as a uint32).
  static constexpr uint32_t kCol[6] = {
      0xFF32C832,  // REC green
      0xFFC8C828,  // SUB yellow
      0xFFE63232,  // GPU red
      0xFF3288E6,  // PRS blue
      0xFFC832C8,  // TEX purple
      0xFF828282,  // OTH gray
  };
  static const char* kLabel[6] = {"REC", "SUB", "GPU", "PRS", "TEX", "OTH"};
  constexpr int kColW = 2, kGraphH = 120;
  constexpr float kPxPerMs = 2.0f;
  const int graph_w = kStageHistN * kColW;
  const int x0 = 10, y1 = (int)h - 10, y0 = y1 - kGraphH;
  const int legend_h = 7 * 14 + 4;
  const int panel_y0 = y0 - legend_h - 4;
  // Darken the panel background (keeps the game visible underneath).
  for (int yy = panel_y0 - 4; yy < y1 + 4 && yy < (int)h; yy++) {
    if (yy < 0)
      continue;
    uint32_t* row = reinterpret_cast<uint32_t*>(bgra + (size_t)yy * w * 4);
    for (int xx = x0 - 4; xx < x0 + graph_w + 4 && xx < (int)w; xx++)
      row[xx] = (row[xx] >> 2) & 0x3F3F3F3F;
  }
  // Columns: oldest left, newest right; stages stacked bottom-up.
  for (int i = 0; i < kStageHistN; i++) {
    int idx = g_stage_hist_pos - kStageHistN + i;
    if (idx < g_stage_hist_pos - g_stage_hist_count)
      continue;  // no sample yet
    idx = ((idx % kStageHistN) + kStageHistN) % kStageHistN;
    const StageSample& s = g_stage_hist[idx];
    const float vals[6] = {s.rec, s.sub, s.gpu, s.prs, s.tex, s.oth};
    int x = x0 + i * kColW, y_base = y1;
    for (int st = 0; st < 6; st++) {
      int hpx = (int)(vals[st] * kPxPerMs + 0.5f);
      if (y_base - hpx < y0)
        hpx = y_base - y0;
      if (hpx > 0)
        OvFill(bgra, w, h, x, y_base - hpx, kColW, hpx, kCol[st]);
      y_base -= hpx;
      if (y_base <= y0)
        break;
    }
  }
  // 60 / 30 fps reference lines.
  for (int xx = x0; xx < x0 + graph_w; xx += 3) {
    OvFill(bgra, w, h, xx, y1 - (int)(16.7f * kPxPerMs), 1, 1, 0xFFF0F0F0);
    OvFill(bgra, w, h, xx, y1 - (int)(33.3f * kPxPerMs), 1, 1, 0xFFF0F0F0);
  }
  // Legend: averages over the last second's worth of samples.
  const int n_avg = g_stage_hist_count < 60 ? g_stage_hist_count : 60;
  float avg[6] = {}, avg_wall = 0;
  for (int i = 1; i <= n_avg; i++) {
    const StageSample& s =
        g_stage_hist[((g_stage_hist_pos - i) % kStageHistN + kStageHistN) %
                     kStageHistN];
    avg[0] += s.rec;
    avg[1] += s.sub;
    avg[2] += s.gpu;
    avg[3] += s.prs;
    avg[4] += s.tex;
    avg[5] += s.oth;
    avg_wall += s.wall;
  }
  for (float& v : avg)
    v /= n_avg;
  avg_wall /= n_avg;
  char buf[64];
  int ty = panel_y0;
  std::snprintf(buf, sizeof buf, "FPS %.1f  %.1f MS",
                avg_wall > 0.01f ? 1000.0f / avg_wall : 0.0f, avg_wall);
  OvText(bgra, w, h, x0, ty, 2, 0xFFFFFFFF, buf);
  ty += 14;
  for (int st = 0; st < 6; st++, ty += 14) {
    OvFill(bgra, w, h, x0, ty + 1, 8, 8, kCol[st]);
    std::snprintf(buf, sizeof buf, "%s %5.1f", kLabel[st], avg[st]);
    OvText(bgra, w, h, x0 + 14, ty, 2, 0xFFE6E6E6, buf);
  }
}

// Wall-clock FPS report. Always on (cheap): every ~2s of presented frames, log
// the average FPS over that window so perf changes can be measured empirically
// (DELTA_GPU_FPS=0 silences it).
void ReportFps() {
  static const bool off = [] {
    const char* e = std::getenv("DELTA_GPU_FPS");
    return e && e[0] == '0';
  }();
  if (off)
    return;
  using clock = std::chrono::steady_clock;
  static auto last = clock::now();
  static int frames = 0;
  frames++;
  auto now = clock::now();
  double dt = std::chrono::duration<double>(now - last).count();
  if (dt >= 2.0) {
    double f = frames ? frames : 1;
    std::fprintf(
        stderr,
        "[fps] %.1f fps | per-frame gpu-code: draw=%.2fms end=%.2fms "
        "(wait=%.2fms "
        "submit=%.2fms present=%.2fms) texup=%.2fms x%.1f cs=%.2fms x%.1f "
        "(in=%.2f gpu=%.2f out=%.2f stage=%.1fx%.1fMB flush=%.1f)\n",
        frames / dt, g_ns_draw / f / 1e6, g_ns_end / f / 1e6,
        g_ns_readback / f / 1e6, g_ns_submit / f / 1e6, g_ns_present / f / 1e6,
        g_ns_tex_up / f / 1e6, g_tex_ups / f, g_ns_cs / f / 1e6, g_cs_count / f,
        g_ns_cs_in / f / 1e6, g_ns_cs_gpu / f / 1e6, g_ns_cs_out / f / 1e6,
        g_cs_stage_n / f, g_cs_stage_bytes / f / 1e6, g_cs_flush_n / f);
    // Feed the on-screen overlay gauge (gpuMs = GPU end/present-dominated
    // cost).
    gfx::overlaySetPerf(float(frames / dt), float(g_ns_end / f / 1e6),
                        float(1000.0 * dt / frames));
    last = now;
    frames = 0;
    g_ns_draw = g_ns_end = g_ns_readback = g_ns_tex_up = 0;
    g_ns_submit = g_ns_present = 0;
    g_tex_ups = 0;
    g_ns_cs = g_cs_bytes = 0;
    g_ns_cs_in = g_ns_cs_gpu = g_ns_cs_out = 0;
    g_cs_count = g_cs_stage_n = g_cs_flush_n = 0;
    g_cs_stage_bytes = 0;
  }
}

}  // namespace gpu::vk
