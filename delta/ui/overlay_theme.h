#pragma once

#include "imgui.h"

namespace ui::overlay_theme {

inline constexpr ImU32 kCanvas = IM_COL32(12, 13, 17, 255);
inline constexpr ImU32 kSurface = IM_COL32(24, 26, 32, 255);
inline constexpr ImU32 kRaised = IM_COL32(34, 36, 43, 255);
inline constexpr ImU32 kInset = IM_COL32(17, 18, 23, 255);
inline constexpr ImU32 kBorder = IM_COL32(220, 226, 238, 28);
inline constexpr ImU32 kText = IM_COL32(245, 247, 250, 255);
inline constexpr ImU32 kSecondary = IM_COL32(180, 185, 197, 255);
inline constexpr ImU32 kMuted = IM_COL32(125, 131, 145, 255);
inline constexpr ImU32 kAccent = IM_COL32(102, 159, 255, 255);
inline constexpr ImU32 kWarning = IM_COL32(230, 192, 113, 255);
inline constexpr ImU32 kError = IM_COL32(243, 139, 149, 255);
inline constexpr ImU32 kCritical = kError;

ImFont* AddSansFont(float size);
void Apply();
ImFont* MonospaceFont();
ImFont* HeadingFont();
ImU32 WithOpacity(ImU32 color, float opacity);
void DrawFocusHalo(ImDrawList* draw_list,
                   ImVec2 top_left,
                   ImVec2 bottom_right,
                   float rounding,
                   float opacity);
void DrawPanel(ImDrawList* draw_list,
               ImVec2 top_left,
               ImVec2 bottom_right,
               float opacity = 1.0f);
void DrawMark(ImDrawList* draw_list, ImVec2 position);

}  // namespace ui::overlay_theme
