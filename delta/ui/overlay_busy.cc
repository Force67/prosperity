#include "ui/overlay_busy.h"

#include "base/algorithm.h"
#include "base/math/value_bounds.h"
#include "base/time/time.h"
#include "imgui.h"
#include "ui/overlay_theme.h"
#include "ui/shader_activity.h"

namespace ui {
namespace {
float g_visibility = 0;
u64 g_last_tick_ns = 0;
u64 g_last_busy_ns = 0;
bool g_compiling = false;

ImU32 Fade(ImU32 color, float opacity) {
  ImVec4 value = ImGui::ColorConvertU32ToFloat4(color);
  value.w *= opacity;
  return ImGui::ColorConvertFloat4ToU32(value);
}
}  // namespace

void OverlayBusyBuild(u32 width, u32 height, bool frame_stalled) {
  const u64 now = base::TickClock::NowNs();
  const ShaderActivity activity = GetShaderActivity();
  const bool compiling =
      (activity.active && now - activity.active_since_ns >= 120'000'000) ||
      (activity.last_slow_compile_ns &&
       now - activity.last_slow_compile_ns < 750'000'000);
  const bool busy = frame_stalled || compiling;
  if (busy) {
    g_last_busy_ns = now;
    g_compiling = compiling || (frame_stalled && activity.active);
  }
  // Hold briefly across short gaps between first-use pipelines.
  const bool visible =
      busy || (g_last_busy_ns && now - g_last_busy_ns < 350'000'000);
  const float dt = g_last_tick_ns
                       ? base::Min(float(now - g_last_tick_ns) / 1e9f, 0.1f)
                       : 0.0f;
  g_last_tick_ns = now;
  g_visibility =
      base::Clamp(g_visibility + dt / (visible ? 0.18f : -0.24f), 0.0f, 1.0f);
  if (g_visibility == 0 || width < 240 || height < 96)
    return;

  const float ease = 1 - (1 - g_visibility) * (1 - g_visibility);
  const float panel_w = base::Min(300.0f, float(width) - 24);
  const ImVec2 tl(float(width) - panel_w - 12 + (1 - ease) * 28, 12);
  const ImVec2 br(tl.x + panel_w, tl.y + 68);
  ImDrawList* dl = ImGui::GetForegroundDrawList();
  overlay_theme::DrawPanel(dl, tl, br, ease);

  const ImVec2 center(tl.x + 28, tl.y + 34);
  const float phase = float(now % 1'200'000'000) / 1'200'000'000 * 6.2831853f;
  dl->PathArcTo(center, 8, phase, phase + 4.712389f, 20);
  dl->PathStroke(Fade(overlay_theme::kText, ease), 0, 2);
  dl->AddText(ImVec2(tl.x + 48, tl.y + 16), Fade(overlay_theme::kText, ease),
              g_compiling ? "Compiling shaders" : "Frame busy");
  dl->AddText(
      ImVec2(tl.x + 48, tl.y + 38), Fade(overlay_theme::kSecondary, ease),
      g_compiling ? "Preparing graphics..." : "Waiting for the game...");
}

}  // namespace ui
