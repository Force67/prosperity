/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/rhi/command.h"
#include "base/arch.h"
#include "gpu/vulkan/vk_perf.h"

#include "gfx/gfx.h"
#include "gfx/overlay.h"
#include "gpu/gcn/gcn_translate.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <base/logging.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kFpsReport, "DELTA_GPU_FPS", true);
DELTA_OPTION(bool, kOverlay, "DELTA_GPU_OVERLAY", true);
DELTA_OPTION(bool, kDrawProf, "DELTA_GPU_DRAWPROF", false);
// A frame longer than this counts as a stutter rather than a slow frame.
DELTA_OPTION(u32, kHitchMs, "DELTA_GPU_HITCH_MS", 40);
}  // namespace

namespace gpu::vk {

u64 g_ns_gpu_exec = 0;
u32 g_gpu_exec_samples = 0;

u64 g_ns_draw = 0, g_ns_end = 0, g_ns_readback = 0, g_ns_tex_up = 0;
u64 g_ns_cs = 0, g_cs_bytes = 0;
u64 g_ns_cs_in = 0, g_ns_cs_gpu = 0, g_ns_cs_out = 0;
u32 g_cs_count = 0;
u64 g_win_draws = 0, g_win_declines = 0;
u64 g_ns_dr_pre = 0, g_ns_dr_pipe = 0, g_ns_dr_tex = 0, g_ns_dr_bind = 0;
u64 g_ns_tex_hash = 0, g_tex_hash_bytes = 0, g_ns_tex_probe = 0;
u64 g_tex_hash_n = 0, g_tex_probe_n = 0, g_ns_tex_lookup = 0, g_tex_lookup_n = 0;
u64 g_ns_tex_set = 0, g_tex_set_n = 0, g_ns_region = 0, g_ns_cs_flush = 0;
u64 g_ns_build_draw = 0, g_build_draw_n = 0;
u64 g_ns_pipe_build = 0, g_pipe_build_n = 0;
float g_frame_worst_ms = 0;
u32 g_frame_hitch_n = 0;
u64 g_ns_gfx_present = 0, g_ns_borrow_wait = 0;
u64 g_ns_submit = 0, g_ns_present = 0;
u32 g_tex_ups = 0;
u32 g_cs_stage_n = 0, g_cs_flush_n = 0;
u64 g_cs_stage_bytes = 0;
u64 g_cs_wb_bytes_written = 0, g_cs_wb_bytes_total = 0;
u64 g_fr_draw = 0, g_fr_submit = 0, g_fr_wait = 0, g_fr_present = 0,
         g_fr_tex_up = 0;

namespace {

// Rolling per-frame stage history for the overlay graph (~4s at 60 fps).
struct StageSample {
  float rec, sub, gpu, prs, tex, oth, wall;  // ms
};

constexpr int kStageHistN = 240;
StageSample g_stage_hist[kStageHistN];
int g_stage_hist_pos = 0, g_stage_hist_count = 0;

// This process only, not the machine: when tuning the emulator the useful
// number is what it is itself using, and a system-wide gauge mostly reports
// whatever else happens to be running. CPU is utime+stime from
// /proc/self/stat, RSS the resident pages from /proc/self/statm.
struct ProcStats {
  float cpuPct = 0;  // whole process; exceeds 100 when several threads are busy
  u64 rss = 0;  // bytes
};
ProcStats g_proc;

void RefreshProcStats() {
  static u64 last_ns = 0;
  static u64 last_jiffies = 0;
  const u64 now = NowNs();
  if (last_ns && now - last_ns < 500000000ull)
    return;

  u64 jiffies = 0;
  if (FILE* f = std::fopen("/proc/self/stat", "r")) {
    char buf[1024];
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    buf[n] = 0;
    // comm (field 2) can contain spaces and parens, so start after the last ')'.
    if (const char* p = std::strrchr(buf, ')')) {
      unsigned long ut = 0, st = 0;
      int field = 3;
      p += 2;
      while (*p && field < 14) {
        if (*p == ' ')
          field++;
        p++;
      }
      if (std::sscanf(p, "%lu %lu", &ut, &st) == 2)
        jiffies = ut + st;
    }
  }
  const long hz = sysconf(_SC_CLK_TCK);
  if (last_ns && hz > 0 && jiffies >= last_jiffies) {
    const double dt = (now - last_ns) / 1e9;
    const double busy = double(jiffies - last_jiffies) / hz;
    g_proc.cpuPct = dt > 0 ? float(busy / dt * 100.0) : 0.0f;
  }
  last_jiffies = jiffies;
  last_ns = now;

  if (FILE* f = std::fopen("/proc/self/statm", "r")) {
    unsigned long total = 0, resident = 0;
    if (std::fscanf(f, "%lu %lu", &total, &resident) == 2)
      g_proc.rss = u64(resident) * u64(sysconf(_SC_PAGESIZE));
    std::fclose(f);
  }
}

// 3x5 bitmap font (rows top-down, bit 2 = left pixel). Uppercase + digits only.
const u8* OvGlyph(char c) {
  struct Glyph {
    char c;
    u8 rows[5];
  };
  static const Glyph f[] = {
      {'0', {7, 5, 5, 5, 7}}, {'1', {2, 6, 2, 2, 7}}, {'2', {7, 1, 7, 4, 7}},
      {'3', {7, 1, 7, 1, 7}}, {'4', {5, 5, 7, 1, 1}}, {'5', {7, 4, 7, 1, 7}},
      {'6', {7, 4, 7, 5, 7}}, {'7', {7, 1, 1, 2, 2}}, {'8', {7, 5, 7, 5, 7}},
      {'9', {7, 5, 7, 1, 7}}, {'.', {0, 0, 0, 0, 2}}, {'%', {5, 1, 2, 4, 5}},
      {'A', {7, 5, 7, 5, 5}},
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

// The overlay is written into the presented buffer, which carries the scanout's
// own component order. Its colours are authored BGRA, so a COMP_SWAP=STD frame
// wants them the other way round; every colour reaches the buffer through here.
bool g_overlay_rgba = false;

inline void OvFill(u8* b,
                   u32 w,
                   u32 h,
                   int x,
                   int y,
                   int fw,
                   int fh,
                   u32 bgra) {
  if (x < 0 || y < 0)
    return;
  const u32 col = g_overlay_rgba ? ((bgra & 0xFF00FF00u) |
                                    ((bgra & 0xFFu) << 16) |
                                    ((bgra >> 16) & 0xFFu))
                                 : bgra;
  for (int yy = y; yy < y + fh && yy < (int)h; yy++) {
    u32* row = reinterpret_cast<u32*>(b + (size_t)yy * w * 4);
    for (int xx = x; xx < x + fw && xx < (int)w; xx++)
      row[xx] = col;
  }
}

void OvText(u8* b,
            u32 w,
            u32 h,
            int x,
            int y,
            int scale,
            u32 bgra,
            const char* s) {
  for (; *s; s++, x += 4 * scale) {
    const u8* rows = OvGlyph(*s);
    if (!rows)
      continue;
    for (int ry = 0; ry < 5; ry++)
      for (int rx = 0; rx < 3; rx++)
        if (rows[ry] & (4 >> rx))
          OvFill(b, w, h, x + rx * scale, y + ry * scale, scale, scale, bgra);
  }
}

// What this process is using, top-right, in the same terse style as the stage
// legend. Deliberately its own panel: these are resource levels, not a
// breakdown of the frame, and reading them next to the graph invited the two to
// be compared. gpuMs/wallMs is our own GPU time as a share of the frame; VRAM
// is the driver's per-process estimate.
void DrawResourcePanel(u8* bgra,
                       u32 w,
                       u32 h,
                       float gpuMs,
                       float wallMs) {
  RefreshProcStats();
  u64 vramUsed = 0, vramTotal = 0;
  gfx::queryVram(vramUsed, vramTotal);
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  const float gpuPct =
      wallMs > 0.01f ? (gpuMs / wallMs * 100.0f > 100.0f ? 100.0f
                                                         : gpuMs / wallMs * 100.0f)
                     : 0.0f;
  const struct {
    u32 col;
    const char* fmt;
    double val;
  } kUse[4] = {
      {0xFF5AC8B4, "CPU %5.0f%%", g_proc.cpuPct},
      {0xFFE69632, "GPU %5.0f%%", gpuPct},
      {0xFF5AA0E6, "RAM %5.2f GB", g_proc.rss / kGiB},
      {0xFFD2A0F0, "VRAM %4.2f GB", vramUsed / kGiB},
  };
  constexpr int kRowH = 14, kSwatch = 8, kTextX = 14;
  constexpr int kPanelW = 14 + 13 * 8 + 6;  // swatch + widest row + padding
  const int x0 = (int)w - kPanelW - 10, y0 = 10;
  const int panelH = kRowH * 4 + 4;
  for (int yy = y0 - 4; yy < y0 + panelH && yy < (int)h; yy++) {
    if (yy < 0)
      continue;
    u32* row = reinterpret_cast<u32*>(bgra + (size_t)yy * w * 4);
    for (int xx = x0 - 4; xx < x0 + kPanelW && xx < (int)w; xx++)
      if (xx >= 0)
        row[xx] = (row[xx] >> 2) & 0x3F3F3F3F;
  }
  char buf[64];
  int ty = y0;
  for (const auto& u : kUse) {
    OvFill(bgra, w, h, x0, ty + 1, kSwatch, kSwatch, u.col);
    std::snprintf(buf, sizeof buf, u.fmt, u.val);
    OvText(bgra, w, h, x0 + kTextX, ty, 2, 0xFFE6E6E6, buf);
    ty += kRowH;
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
  static u64 prev_ns = 0;
  const u64 now = NowNs();
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
  if (wall > g_frame_worst_ms)
    g_frame_worst_ms = wall;
  if (wall > kHitchMs)
    g_frame_hitch_n++;
  g_fr_draw = g_fr_submit = g_fr_wait = g_fr_present = g_fr_tex_up = 0;
  g_stage_hist[g_stage_hist_pos] = s;
  g_stage_hist_pos = (g_stage_hist_pos + 1) % kStageHistN;
  if (g_stage_hist_count < kStageHistN)
    g_stage_hist_count++;
}

void DrawPerfOverlay(u8* pixels, u32 w, u32 h, bool rgba) {
  if (!kOverlay || !g_stage_hist_count || w < 560 || h < 280)
    return;
  g_overlay_rgba = rgba;
  u8* const bgra = pixels;
  // BGRA little-endian constants (0xAARRGGBB written as a uint32).
  static constexpr u32 kCol[6] = {
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
  const int legend_h = 7 * 14 + 4;  // fps + 6 stage rows
  const int panel_y0 = y0 - legend_h - 4;
  // Darken the panel background (keeps the game visible underneath).
  for (int yy = panel_y0 - 4; yy < y1 + 4 && yy < (int)h; yy++) {
    if (yy < 0)
      continue;
    u32* row = reinterpret_cast<u32*>(bgra + (size_t)yy * w * 4);
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

  DrawResourcePanel(bgra, w, h, avg[2], avg_wall);
}

// Wall-clock FPS report. Always on (cheap): every ~2s of presented frames, log
// the average FPS over that window so perf changes can be measured empirically
// (DELTA_GPU_FPS=0 silences it).
void ReportFps() {
  if (!kFpsReport)
    return;
  using clock = std::chrono::steady_clock;
  static auto last = clock::now();
  static int frames = 0;
  frames++;
  auto now = clock::now();
  double dt = std::chrono::duration<double>(now - last).count();
  if (dt >= 2.0) {
    double f = frames ? frames : 1;
    BASE_LOGI("fps", "{:.1f} fps | per-frame gpu-code: draw={:.2f}ms end={:.2f}ms "
                     "(wait={:.2f}ms exec={:.2f}ms "
                     "submit={:.2f}ms present={:.2f}ms) texup={:.2f}ms x{:.1f} "
                     "cs={:.2f}ms x{:.1f} "
                     "(in={:.2f} gpu={:.2f} out={:.2f} stage={:.1f}x{:.1f}MB "
                     "flush={:.1f}) "
                     "sh={:.2f}ms x{:.1f} dcb={:.2f}ms x{:.1f} (lock={:.2f}ms) "
                     "wb={:.0f}% draws={:.0f} (decl={:.0f})",
              frames / dt, g_ns_draw / f / 1e6, g_ns_end / f / 1e6,
              g_ns_readback / f / 1e6,
              g_gpu_exec_samples ? g_ns_gpu_exec / g_gpu_exec_samples / 1e6 : 0.0,
              g_ns_submit / f / 1e6, g_ns_present / f / 1e6, g_ns_tex_up / f / 1e6,
              g_tex_ups / f, g_ns_cs / f / 1e6, g_cs_count / f, g_ns_cs_in / f / 1e6,
              g_ns_cs_gpu / f / 1e6, g_ns_cs_out / f / 1e6, g_cs_stage_n / f,
              g_cs_stage_bytes / f / 1e6, g_cs_flush_n / f,
              gcn::g_ns_recomp / f / 1e6, gcn::g_recomp_n / f,
              rhi::g_ns_dcb / f / 1e6, rhi::g_dcb_n / f,
              rhi::g_ns_dcb_lock / f / 1e6,
              g_cs_wb_bytes_total ? 100.0 * double(g_cs_wb_bytes_written) /
                                        double(g_cs_wb_bytes_total)
                                  : 0.0,
              g_win_draws / f, g_win_declines / f);
    // Stutter, said out loud. Always on when it happens: the average above
    // reports a window that contained a 400 ms frame as merely slow, and the
    // whole point of the shader work is the frames this line counts.
    if (g_frame_hitch_n || g_pipe_build_n)
      BASE_LOGI("hitch",
                "worst={:.0f}ms over{}ms={} | pipe={:.0f}ms x{} recomp={:.0f}ms "
                "x{}",
                g_frame_worst_ms, (u32)kHitchMs, g_frame_hitch_n,
                g_ns_pipe_build / 1e6, g_pipe_build_n, gcn::g_ns_recomp / 1e6,
                gcn::g_recomp_n);
    if (kDrawProf)
      BASE_LOGI("drawprof",
                "per-frame pre={:.2f}ms pipe={:.2f}ms tex={:.2f}ms "
                "bind={:.2f}ms | hash={:.2f}ms x{:.0f} {:.1f}MB "
                "probe={:.2f}ms x{:.0f} lookup={:.2f}ms x{:.0f}",
                g_ns_dr_pre / f / 1e6, g_ns_dr_pipe / f / 1e6,
                g_ns_dr_tex / f / 1e6, g_ns_dr_bind / f / 1e6,
                g_ns_tex_hash / f / 1e6, g_tex_hash_n / f,
                g_tex_hash_bytes / f / 1e6, g_ns_tex_probe / f / 1e6,
                g_tex_probe_n / f, g_ns_tex_lookup / f / 1e6,
                g_tex_lookup_n / f);
    if (kDrawProf)
      BASE_LOGI("drawprof2",
                "per-frame texset={:.2f}ms x{:.0f} region={:.2f}ms "
                "csflush={:.2f}ms builddraw={:.2f}ms x{:.0f} "
                "gfxpres={:.2f}ms borrowwait={:.2f}ms",
                g_ns_tex_set / f / 1e6, g_tex_set_n / f, g_ns_region / f / 1e6,
                g_ns_cs_flush / f / 1e6, g_ns_build_draw / f / 1e6,
                g_build_draw_n / f, g_ns_gfx_present / f / 1e6,
                g_ns_borrow_wait / f / 1e6);
    CsSyncReport(f);
    // Feed the on-screen overlay gauge (gpuMs = GPU end/present-dominated
    // cost).
    gfx::overlaySetPerf(float(frames / dt), float(g_ns_end / f / 1e6),
                        float(1000.0 * dt / frames));
    last = now;
    frames = 0;
    g_ns_draw = g_ns_end = g_ns_readback = g_ns_tex_up = 0;
    g_ns_submit = g_ns_present = 0;
    g_ns_gpu_exec = 0;
    g_gpu_exec_samples = 0;
    g_tex_ups = 0;
    g_ns_cs = g_cs_bytes = 0;
    g_ns_cs_in = g_ns_cs_gpu = g_ns_cs_out = 0;
    g_cs_count = g_cs_stage_n = g_cs_flush_n = 0;
    g_win_draws = g_win_declines = 0;
    g_ns_dr_pre = g_ns_dr_pipe = g_ns_dr_tex = g_ns_dr_bind = 0;
    g_ns_tex_hash = g_tex_hash_bytes = g_ns_tex_probe = 0;
    g_tex_hash_n = g_tex_probe_n = g_ns_tex_lookup = g_tex_lookup_n = 0;
    g_ns_tex_set = g_tex_set_n = g_ns_region = g_ns_cs_flush = 0;
    g_ns_build_draw = g_build_draw_n = 0;
    g_ns_pipe_build = g_pipe_build_n = 0;
    g_frame_worst_ms = 0;
    g_frame_hitch_n = 0;
    g_ns_gfx_present = g_ns_borrow_wait = 0;
    g_cs_stage_bytes = 0;
    g_cs_wb_bytes_written = g_cs_wb_bytes_total = 0;
    gcn::g_ns_recomp = 0;
    gcn::g_recomp_n = 0;
    rhi::g_ns_dcb = 0;
    rhi::g_ns_dcb_lock = 0;
    rhi::g_dcb_n = 0;
  }
}

}  // namespace gpu::vk
