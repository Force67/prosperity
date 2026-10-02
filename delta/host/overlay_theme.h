#pragma once

#include "imgui.h"

namespace host::overlay_theme {

inline constexpr ImU32 kCanvas = IM_COL32(10, 13, 18, 255);
inline constexpr ImU32 kSurface = IM_COL32(16, 22, 29, 255);
inline constexpr ImU32 kRaised = IM_COL32(25, 34, 43, 255);
inline constexpr ImU32 kInset = IM_COL32(10, 16, 22, 255);
inline constexpr ImU32 kBorder = IM_COL32(108, 128, 144, 64);
inline constexpr ImU32 kText = IM_COL32(231, 237, 249, 255);
inline constexpr ImU32 kSecondary = IM_COL32(178, 193, 207, 255);
inline constexpr ImU32 kMuted = IM_COL32(116, 137, 153, 255);
inline constexpr ImU32 kViolet = IM_COL32(132, 88, 245, 255);
inline constexpr ImU32 kBlue = IM_COL32(73, 146, 220, 255);
inline constexpr ImU32 kCyan = IM_COL32(116, 214, 232, 255);
inline constexpr ImU32 kAmber = IM_COL32(230, 174, 83, 255);
inline constexpr ImU32 kWarning = kAmber;
inline constexpr ImU32 kError = IM_COL32(247, 124, 143, 255);
inline constexpr ImU32 kCritical = IM_COL32(223, 155, 244, 255);

void Apply();
ImFont* MonospaceFont();
void DrawPanel(ImDrawList* draw_list, ImVec2 top_left, ImVec2 bottom_right);
void DrawMark(ImDrawList* draw_list, ImVec2 position);

}  // namespace host::overlay_theme
