#include "ui/overlay_memory.h"

#if defined(DELTA_MEMORY_DEBUG) && !defined(__ANDROID__)
#include <unistd.h>
#include <cfloat>
#include <cstdio>

#include "base/algorithm.h"
#include "base/atomic.h"
#include "base/math/value_bounds.h"
#include "base/time/time.h"
#include "host_memory/host_memory.h"
#include "imgui.h"
#include "memory_debug/memory_debug.h"
#include "options/options.h"
#include "ui/overlay_theme.h"

namespace ui {
namespace {
DELTA_OPTION(u32,
             kMemoryOverlay,
             "DELTA_MEMORY_OVERLAY",
             0,
             "memory overlay: 0 off, 1 compact, 2 full (F4 cycles)");
constexpr u64 kSampleNs = 250'000'000;
constexpr u64 kHistoryNs = 60'000'000'000;
constexpr u32 kHistorySize = 256;
struct Sample {
  u64 ns = 0;
  memory_debug::Snapshot snapshot;
};
base::Atomic<i32> g_mode{-1};
base::Atomic<u32> g_escape_consumed{0};
Sample g_history[kHistorySize];
u32 g_next = 0;
u32 g_count = 0;
u64 g_sample_ns = 0;
u64 g_rss = 0;
host_memory::MappingStats g_guest;
memory_debug::Snapshot g_snapshot;
memory_debug::Snapshot g_baseline;
u64 g_baseline_ns = 0;

void FormatBytes(char* text, size_t length, u64 bytes) {
  constexpr double kMib = 1024.0 * 1024;
  if (bytes >= (1ull << 30)) {
    std::snprintf(text, length, "%.2f GiB", double(bytes) / (1ull << 30));
  } else if (bytes >= (1ull << 20)) {
    std::snprintf(text, length, "%.1f MiB", double(bytes) / kMib);
  } else if (bytes >= 1024) {
    std::snprintf(text, length, "%.1f KiB", double(bytes) / 1024);
  } else {
    std::snprintf(text, length, "%llu B",
                  static_cast<unsigned long long>(bytes));
  }
}

void SampleMemory() {
  const u64 now = base::TickClock::NowNs();
  if (g_sample_ns && now - g_sample_ns < kSampleNs)
    return;
  g_sample_ns = now;
  g_snapshot = memory_debug::ReadSnapshot();
  g_guest = host_memory::GetMappingStats();
  if (FILE* file = std::fopen("/proc/self/statm", "r")) {
    unsigned long total = 0, resident = 0;
    if (std::fscanf(file, "%lu %lu", &total, &resident) == 2)
      g_rss = resident * static_cast<u64>(sysconf(_SC_PAGESIZE));
    std::fclose(file);
  }
  g_history[g_next] = {now, g_snapshot};
  g_next = (g_next + 1) % kHistorySize;
  g_count = base::Min(g_count + 1, kHistorySize);
  const u32 oldest = (g_next + kHistorySize - g_count) % kHistorySize;
  g_baseline = g_history[oldest].snapshot;
  g_baseline_ns = g_history[oldest].ns;
  for (u32 i = 0; i < g_count; ++i) {
    const auto& sample = g_history[(oldest + i) % kHistorySize];
    if (now - sample.ns < kHistoryNs)
      break;
    g_baseline = sample.snapshot;
    g_baseline_ns = sample.ns;
  }
}

void Text(ImDrawList* dl,
          ImVec2 pos,
          float size,
          ImU32 color,
          const char* text) {
  dl->AddText(ImGui::GetIO().FontDefault, size, pos, color, text);
}

void RightText(ImDrawList* dl,
               ImVec2 pos,
               float size,
               ImU32 color,
               const char* text) {
  const auto* font = ImGui::GetIO().FontDefault;
  const float width = font->CalcTextSizeA(size, FLT_MAX, 0, text).x;
  Text(dl, ImVec2(pos.x - width, pos.y), size, color, text);
}

constexpr ImGuiWindowFlags kWindowFlags =
    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground |
    ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing;

void DrawDock(u32 width, u32 height) {
  const float scale = base::Min(1.0f, float(width) / 520);
  const float dock_width = 440 * scale;
  const float dock_height = 84 * scale;
  const ImVec2 tl((width - dock_width) * 0.5f,
                  height - dock_height - base::Max(16.0f, height * 0.08f));
  const ImVec2 br(tl.x + dock_width, tl.y + dock_height);
  ImGui::SetNextWindowPos(tl);
  ImGui::SetNextWindowSize(ImVec2(dock_width, dock_height));
  ImGui::Begin("##memory_dock", nullptr, kWindowFlags);
  auto* dl = ImGui::GetWindowDrawList();
  dl->AddRectFilled(tl, br,
                    overlay_theme::WithOpacity(overlay_theme::kSurface, 0.72f),
                    16 * scale);
  const char* labels[] = {"Process", "Host", "GPU"};
  const u64 values[] = {g_rss, g_snapshot.host_live, g_snapshot.gpu_live};
  for (u32 i = 0; i < 3; ++i) {
    const float x = tl.x + (28 + 110 * i) * scale;
    Text(dl, ImVec2(x, tl.y + 18 * scale), 14 * scale,
         overlay_theme::kSecondary, labels[i]);
    char text[32];
    FormatBytes(text, sizeof(text), values[i]);
    Text(dl, ImVec2(x, tl.y + 41 * scale), 20 * scale, overlay_theme::kText,
         text);
  }
  const ImVec2 button(tl.x + 370 * scale, tl.y + 22 * scale);
  ImGui::SetCursorScreenPos(button);
  if (ImGui::InvisibleButton("Expand memory", ImVec2(40 * scale, 40 * scale)))
    SetMemoryOverlayMode(MemoryOverlayMode::kFull);
  dl->AddCircleFilled(ImVec2(button.x + 20 * scale, button.y + 20 * scale),
                      20 * scale,
                      overlay_theme::WithOpacity(overlay_theme::kText, 0.08f));
  Text(dl, ImVec2(button.x + 13 * scale, button.y + 9 * scale), 20 * scale,
       overlay_theme::kText, ">");
  ImGui::End();
}

void DrawTile(ImDrawList* dl, ImVec2 tl, ImVec2 size, u32 index, float scale) {
  const auto bucket = static_cast<memory_debug::Bucket>(index);
  const auto& description = memory_debug::Describe(bucket);
  auto stats = g_snapshot.buckets[index];
  auto prior = g_baseline.buckets[index];
  if (bucket == memory_debug::Bucket::kImagePool) {
    stats.live = g_snapshot.buckets[0].live + g_snapshot.buckets[1].live;
    prior.live = g_baseline.buckets[0].live + g_baseline.buckets[1].live;
  }
  const u64 image_live =
      g_snapshot.buckets[0].live + g_snapshot.buckets[1].live;
  const u64 capacity =
      description.shared_backing
          ? g_snapshot
                .buckets[static_cast<u32>(memory_debug::Bucket::kImagePool)]
                .backing
          : stats.backing;
  const u64 occupancy = description.shared_backing ? image_live : stats.live;
  const bool has_backing = capacity != 0;
  const float fraction =
      has_backing
          ? base::Clamp(float(double(occupancy) / double(capacity)), 0.0f, 1.0f)
          : 0;
  const ImU32 color = fraction >= 0.95f  ? overlay_theme::kError
                      : fraction >= 0.8f ? overlay_theme::kWarning
                                         : overlay_theme::kAccent;
  const ImVec2 br(tl.x + size.x, tl.y + size.y);
  const float pad = 16 * scale;
  const float right = br.x - pad;
  dl->AddRectFilled(tl, br,
                    overlay_theme::WithOpacity(overlay_theme::kSurface, 0.62f),
                    12 * scale);
  if (has_backing) {
    dl->PushClipRect(tl, br, true);
    dl->AddRectFilled(ImVec2(tl.x, br.y - size.y * fraction), br,
                      overlay_theme::WithOpacity(color, 0.10f), 12 * scale);
    dl->PopClipRect();
  }
  Text(dl, ImVec2(tl.x + pad, tl.y + 12 * scale), 11 * scale,
       overlay_theme::kSecondary,
       description.domain == memory_debug::Domain::kGpu ? "GPU" : "Host");
  char text[64];
  if (has_backing) {
    std::snprintf(text, sizeof(text), "%s%.0f%%",
                  description.shared_backing ? "Pool " : "", fraction * 100);
    RightText(dl, ImVec2(right, tl.y + 12 * scale), 12 * scale,
              overlay_theme::kSecondary, text);
  }
  Text(dl, ImVec2(tl.x + pad, tl.y + 30 * scale), 16 * scale,
       overlay_theme::kText, description.name);
  FormatBytes(text, sizeof(text), stats.live);
  Text(dl, ImVec2(tl.x + pad, tl.y + 55 * scale), 23 * scale,
       overlay_theme::kText, text);
  if (has_backing) {
    char backing[32];
    FormatBytes(backing, sizeof(backing), capacity);
    std::snprintf(
        text, sizeof(text),
        description.shared_backing ? "Shared pool %s" : "of %s backing",
        backing);
    Text(dl, ImVec2(tl.x + pad, tl.y + 83 * scale), 12 * scale,
         overlay_theme::kSecondary, text);
  } else {
    Text(dl, ImVec2(tl.x + pad, tl.y + 83 * scale), 12 * scale,
         overlay_theme::kMuted, "No allocations");
  }
  const float bar_y = tl.y + 103 * scale;
  const float bar_width = size.x - pad * 2;
  dl->AddRectFilled(ImVec2(tl.x + pad, bar_y), ImVec2(right, bar_y + 4 * scale),
                    overlay_theme::WithOpacity(overlay_theme::kText, 0.09f),
                    2 * scale);
  if (has_backing && fraction > 0)
    dl->AddRectFilled(
        ImVec2(tl.x + pad, bar_y),
        ImVec2(tl.x + pad + bar_width * fraction, bar_y + 4 * scale), color,
        2 * scale);
  Text(dl, ImVec2(tl.x + pad, tl.y + 116 * scale), 12 * scale,
       overlay_theme::kSecondary,
       description.shared_backing ? "Pool free" : "Free");
  FormatBytes(text, sizeof(text),
              capacity > occupancy ? capacity - occupancy : 0);
  RightText(dl, ImVec2(right, tl.y + 116 * scale), 12 * scale,
            overlay_theme::kText, text);
  Text(dl, ImVec2(tl.x + pad, tl.y + 136 * scale), 12 * scale,
       overlay_theme::kSecondary,
       bucket == memory_debug::Bucket::kImagePool ? "Blocks" : "Allocations");
  std::snprintf(text, sizeof(text), "%llu",
                static_cast<unsigned long long>(stats.allocations));
  RightText(dl, ImVec2(right, tl.y + 136 * scale), 12 * scale,
            overlay_theme::kText, text);
  const i64 change =
      static_cast<i64>(stats.live) - static_cast<i64>(prior.live);
  const u64 seconds =
      base::Min<u64>(60, (g_sample_ns - g_baseline_ns) / 1'000'000'000);
  if (!seconds) {
    std::snprintf(text, sizeof(text), "Collecting history");
  } else if (!change) {
    std::snprintf(text, sizeof(text), "Stable / %llus",
                  static_cast<unsigned long long>(seconds));
  } else {
    char delta[32];
    FormatBytes(delta, sizeof(delta), change < 0 ? -change : change);
    std::snprintf(text, sizeof(text), "%c%s / %llus", change < 0 ? '-' : '+',
                  delta, static_cast<unsigned long long>(seconds));
  }
  Text(dl, ImVec2(tl.x + pad, tl.y + 156 * scale), 12 * scale,
       change > 0 ? overlay_theme::kWarning : overlay_theme::kSecondary, text);
}

void DrawFull(u32 width, u32 height) {
  if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    SetMemoryOverlayMode(MemoryOverlayMode::kCompact);
    return;
  }
  const float scale = base::Clamp(float(width) / 1280, 0.8f, 1.2f);
  const float margin = base::Max(24.0f, width * 0.05f);
  const float gap = 12 * scale;
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(width, height));
  ImGui::Begin("##memory_full", nullptr, kWindowFlags);
  auto* dl = ImGui::GetWindowDrawList();
  dl->AddRectFilled(ImVec2(0, 0), ImVec2(width, height),
                    IM_COL32(12, 13, 17, 130));
  const float header_y = 30 * scale;
  overlay_theme::DrawMark(dl, ImVec2(margin, header_y + 4));
  Text(dl, ImVec2(margin + 24, header_y), 15 * scale, overlay_theme::kText,
       "Prosperity");
  dl->AddRectFilled(ImVec2(margin + 132 * scale, header_y - 6 * scale),
                    ImVec2(margin + 214 * scale, header_y + 26 * scale),
                    overlay_theme::WithOpacity(overlay_theme::kText, 0.06f),
                    16 * scale);
  Text(dl, ImVec2(margin + 148 * scale, header_y), 15 * scale,
       overlay_theme::kText, "Memory");
  char text[512], rss[32], host[32], gpu[32], guest[32];
  FormatBytes(rss, sizeof(rss), g_rss);
  FormatBytes(host, sizeof(host), g_snapshot.host_live);
  FormatBytes(gpu, sizeof(gpu), g_snapshot.gpu_live);
  FormatBytes(guest, sizeof(guest), g_guest.mapped);
  std::snprintf(
      text, sizeof(text),
      "Process RSS %s     Host tagged %s     GPU live %s     Guest mapped %s",
      rss, host, gpu, guest);
  Text(dl, ImVec2(margin, 80 * scale), 13 * scale, overlay_theme::kSecondary,
       text);
  const float body_y = 110 * scale;
  const float body_width = width - margin * 2;
  const float body_height = base::Max(100.0f, height - body_y - 56 * scale);
  const u32 columns = base::Clamp<u32>(
      static_cast<u32>((body_width + gap) / (200 * scale + gap)), 1, 6);
  const u32 rows = (memory_debug::kVisibleBuckets + columns - 1) / columns;
  const float tile_height =
      base::Max(174 * scale, (body_height - gap * (rows - 1)) / rows);
  ImGui::SetCursorPos(ImVec2(margin, body_y));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::BeginChild("##memory_tiles", ImVec2(body_width, body_height), false,
                    ImGuiWindowFlags_NoBackground);
  ImGui::PopStyleVar();
  dl = ImGui::GetWindowDrawList();
  const float content_height = rows * tile_height + (rows - 1) * gap;
  const float content_width =
      body_width -
      (content_height > body_height ? ImGui::GetStyle().ScrollbarSize : 0);
  const float tile_width = (content_width - gap * (columns - 1)) / columns;
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  for (u32 i = 0; i < memory_debug::kVisibleBuckets; ++i) {
    const ImVec2 tl(origin.x + (i % columns) * (tile_width + gap),
                    origin.y + (i / columns) * (tile_height + gap));
    DrawTile(dl, tl, ImVec2(tile_width, tile_height), i, scale);
  }
  ImGui::Dummy(ImVec2(content_width, content_height));
  ImGui::EndChild();
  dl = ImGui::GetWindowDrawList();
  Text(dl, ImVec2(margin, height - 52 * scale), 12 * scale,
       overlay_theme::kSecondary, "Esc  Compact     F4  Cycle view");
  RightText(dl, ImVec2(width - margin, height - 52 * scale), 12 * scale,
            overlay_theme::kSecondary,
            "Used / backing: blue <80%   amber 80-94%   rose 95%+");
  char reserved[32], imported[32];
  FormatBytes(reserved, sizeof(reserved), g_guest.reserved);
  FormatBytes(imported, sizeof(imported), g_snapshot.imported);
  std::snprintf(text, sizeof(text),
                "Guest reserved %s   Imported aliases %s   C libraries / "
                "driver internals untracked%s",
                reserved, imported,
                g_snapshot.dropped_mappings ? "   Mapping table overflow" : "");
  Text(dl, ImVec2(margin, height - 28 * scale), 11 * scale,
       g_snapshot.dropped_mappings ? overlay_theme::kWarning
                                   : overlay_theme::kMuted,
       text);
  ImGui::End();
}
}  // namespace

MemoryOverlayMode GetMemoryOverlayMode() {
  i32 mode = g_mode.load(base::memory_order_relaxed);
  if (mode < 0) {
    const i32 initial = base::Min<u32>(kMemoryOverlay, 2);
    if (g_mode.compare_exchange_strong(mode, initial,
                                       base::memory_order_relaxed))
      mode = initial;
  }
  return static_cast<MemoryOverlayMode>(mode);
}

void SetMemoryOverlayMode(MemoryOverlayMode mode) {
  if (mode == MemoryOverlayMode::kOff) {
    g_sample_ns = g_count = g_next = 0;
  }
  g_mode.store(static_cast<i32>(mode), base::memory_order_relaxed);
}

void MemoryOverlayToggle() {
  const auto mode = GetMemoryOverlayMode();
  SetMemoryOverlayMode(
      mode == MemoryOverlayMode::kOff       ? MemoryOverlayMode::kCompact
      : mode == MemoryOverlayMode::kCompact ? MemoryOverlayMode::kFull
                                            : MemoryOverlayMode::kOff);
}

bool MemoryOverlayFull() {
  return GetMemoryOverlayMode() == MemoryOverlayMode::kFull;
}

bool MemoryOverlayBlocksInput() {
  return MemoryOverlayFull() ||
         g_escape_consumed.load(base::memory_order_relaxed) != 0;
}

bool MemoryOverlayEscape(bool pressed, bool gamepad) {
  const u32 mask = gamepad ? 2 : 1;
  if (pressed && MemoryOverlayFull()) {
    g_escape_consumed.fetch_or(mask, base::memory_order_relaxed);
    SetMemoryOverlayMode(MemoryOverlayMode::kCompact);
    return true;
  }
  if (!pressed &&
      (g_escape_consumed.fetch_and(~mask, base::memory_order_relaxed) & mask)) {
    return true;
  }
  return false;
}

void MemoryOverlayBuild(u32 width, u32 height) {
  const memory_debug::Scope scope(memory_debug::Bucket::kDebugger);
  if (GetMemoryOverlayMode() == MemoryOverlayMode::kOff)
    return;
  SampleMemory();
  if (GetMemoryOverlayMode() == MemoryOverlayMode::kCompact)
    DrawDock(width, height);
  else if (MemoryOverlayFull())
    DrawFull(width, height);
}

void MemoryOverlayReset() {
  g_mode.store(-1, base::memory_order_relaxed);
  g_escape_consumed.store(0, base::memory_order_relaxed);
  g_sample_ns = g_count = g_next = 0;
}
}  // namespace ui
#else
namespace ui {
MemoryOverlayMode GetMemoryOverlayMode() {
  return MemoryOverlayMode::kOff;
}
void SetMemoryOverlayMode(MemoryOverlayMode) {}
void MemoryOverlayToggle() {}
bool MemoryOverlayFull() {
  return false;
}
bool MemoryOverlayBlocksInput() {
  return false;
}
bool MemoryOverlayEscape(bool, bool) {
  return false;
}
void MemoryOverlayBuild(u32, u32) {}
void MemoryOverlayReset() {}
}  // namespace ui
#endif
