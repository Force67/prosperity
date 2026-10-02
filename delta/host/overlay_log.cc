/*
 * PS4Delta : PS4 emulation and research project
 *
 * The live log panel. See overlay_log.h.
 */

#include "base/arch.h"

#include "base/memory/mem_ops.h"
#include "base/memory/unique_pointer.h"
#include "logger/logger.h"

#include "base/math/value_bounds.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "host/overlay_log.h"
#include "host/overlay_theme.h"
#include "imgui.h"

namespace host {
namespace {

// The ring holds exactly what the panel shows. Lines are truncated at capture,
// so neither the copy in nor the text drawn out depends on the line a title
// decided to log.
constexpr u32 kLines = 14;
constexpr u32 kLineChars = 200;

struct Line {
  char text[kLineChars];
  u8 level;
};

// Held by the logger's backend thread for one memcpy per line, and by the
// render thread for one snapshot per frame.
base::Mutex g_mutex;
Line g_lines[kLines];
u32 g_next = 0;   // slot the next line goes in
u32 g_count = 0;  // filled slots, saturating at kLines
bool g_visible = true;
bool g_attached = false;

class LogPanelSink final : public logger::LogSink {
 public:
  const char* GetName() override { return "overlayLog"; }

  void Write(const logger::LogEntry& entry) override {
    const char* text = entry.message.c_str();
    const size_t length =
        base::Min<size_t>(entry.message.length(), kLineChars - 1);
    base::LockGuard<base::Mutex> lock(g_mutex);
    Line& line = g_lines[g_next];
    base::MemCopy(line.text, text, length);
    line.text[length] = '\0';
    line.level = static_cast<u8>(entry.log_level);
    g_next = (g_next + 1) % kLines;
    if (g_count < kLines)
      g_count++;
  }
};

struct LevelStyle {
  const char* label;
  ImU32 color;
};

LevelStyle GetLevelStyle(u8 level) {
  switch (static_cast<logger::LogLevel>(level)) {
    case logger::LogLevel::kTrace:
      return {"TRC", overlay_theme::kMuted};
    case logger::LogLevel::kDebug:
      return {"DBG", overlay_theme::kCyan};
    case logger::LogLevel::kWarning:
      return {"WRN", overlay_theme::kWarning};
    case logger::LogLevel::kError:
      return {"ERR", overlay_theme::kError};
    case logger::LogLevel::kCritical:
      return {"CRT", overlay_theme::kCritical};
    default:
      return {"INF", overlay_theme::kSecondary};
  }
}

}  // namespace

void OverlayLogAttach() {
  if (g_attached)
    return;
  g_attached = true;
  logger::AddLogSink(base::MakeUnique<LogPanelSink>());
}

void OverlayLogBuild(u32 w, u32 h) {
  if (!g_visible)
    return;

  Line lines[kLines];
  u32 count = 0;
  {
    base::LockGuard<base::Mutex> lock(g_mutex);
    count = g_count;
    // Oldest first: the newest line ends up at the bottom, nearest the corner.
    const u32 oldest = g_count == kLines ? g_next : 0;
    for (u32 i = 0; i < count; i++)
      lines[i] = g_lines[(oldest + i) % kLines];
  }
  if (!count)
    return;

  ImGui::PushFont(overlay_theme::MonospaceFont());
  ImDrawList* dl = ImGui::GetForegroundDrawList();
  const float fs = ImGui::GetFontSize();
  const float pad = 12.0f, lh = fs + 4.0f, margin = 12.0f;
  const float area_w = base::Min(float(w) * 0.42f, 760.0f);
  const float header_h = fs + 16.0f;
  const float area_h = pad * 2.0f + header_h + lh * count;
  const ImVec2 tl(float(w) - area_w - margin, float(h) - area_h - margin);
  const ImVec2 br(tl.x + area_w, tl.y + area_h);

  overlay_theme::DrawPanel(dl, tl, br);
  overlay_theme::DrawMark(dl, ImVec2(tl.x + pad, tl.y + pad));
  dl->AddText(ImGui::GetIO().FontDefault, 15.0f,
              ImVec2(tl.x + pad + 20, tl.y + pad), overlay_theme::kText,
              "SESSION LOG");
  const char* status = "LIVE / [F2]";
  const float status_w = ImGui::CalcTextSize(status).x;
  dl->AddText(ImVec2(br.x - pad - status_w, tl.y + pad), overlay_theme::kAmber,
              status);
  dl->AddLine(ImVec2(tl.x + pad, tl.y + pad + fs + 8),
              ImVec2(br.x - pad, tl.y + pad + fs + 8), overlay_theme::kBorder);

  // Clipping rather than measuring: a long line costs the same as a short one,
  // and no line can spill out of the dimmed area.
  dl->PushClipRect(ImVec2(tl.x + pad, tl.y), ImVec2(br.x - pad, br.y), true);
  float y = tl.y + pad + header_h;
  for (u32 i = 0; i < count; i++) {
    const auto level = GetLevelStyle(lines[i].level);
    dl->AddText(ImVec2(tl.x + pad, y), level.color, level.label);
    dl->AddText(ImVec2(tl.x + pad + 36, y), level.color, lines[i].text);
    y += lh;
  }
  dl->PopClipRect();
  ImGui::PopFont();
}

void OverlayLogToggle() {
  g_visible = !g_visible;
}

}  // namespace host
