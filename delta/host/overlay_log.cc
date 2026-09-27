/*
 * PS4Delta : PS4 emulation and research project
 *
 * The live log panel. See overlay_log.h.
 */

#include <cstring>
#include "base/arch.h"

#include "base/memory/unique_pointer.h"
#include "logger/logger.h"

#include "base/math/value_bounds.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "host/overlay_log.h"
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
    std::memcpy(line.text, text, length);
    line.text[length] = '\0';
    line.level = static_cast<u8>(entry.log_level);
    g_next = (g_next + 1) % kLines;
    if (g_count < kLines)
      g_count++;
  }
};

ImU32 LevelColour(u8 level) {
  switch (static_cast<logger::LogLevel>(level)) {
    case logger::LogLevel::kTrace:
      return IM_COL32(140, 140, 140, 255);
    case logger::LogLevel::kDebug:
      return IM_COL32(120, 200, 255, 255);
    case logger::LogLevel::kWarning:
      return IM_COL32(255, 210, 100, 255);
    case logger::LogLevel::kError:
      return IM_COL32(255, 110, 110, 255);
    case logger::LogLevel::kCritical:
      return IM_COL32(255, 130, 255, 255);
    default:
      return IM_COL32(225, 225, 225, 255);
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

  ImDrawList* dl = ImGui::GetForegroundDrawList();
  const float fs = ImGui::GetFontSize();
  const float pad = 6.0f, lh = fs + 2.0f, margin = 10.0f;
  const float area_w = base::Min(float(w) * 0.42f, 760.0f);
  const float area_h = pad * 2.0f + lh * count;
  const ImVec2 tl(float(w) - area_w - margin, float(h) - area_h - margin);
  const ImVec2 br(tl.x + area_w, tl.y + area_h);

  // Enough dim to read text over a bright frame, with no frame of its own.
  dl->AddRectFilled(tl, br, IM_COL32(0, 0, 0, 110));

  // Clipping rather than measuring: a long line costs the same as a short one,
  // and no line can spill out of the dimmed area.
  dl->PushClipRect(ImVec2(tl.x + pad, tl.y), ImVec2(br.x - pad, br.y), true);
  float y = tl.y + pad;
  for (u32 i = 0; i < count; i++) {
    dl->AddText(ImVec2(tl.x + pad, y), LevelColour(lines[i].level),
                lines[i].text);
    y += lh;
  }
  dl->PopClipRect();
}

void OverlayLogToggle() {
  g_visible = !g_visible;
}

}  // namespace host
