#include "ui/overlay_theme.h"

#include "DroidSans.hpp"

namespace ui::overlay_theme {
namespace {

ImFont* g_monospace_font = nullptr;

ImVec4 Color(ImU32 value, float alpha = 1.0f) {
  ImVec4 color = ImGui::ColorConvertU32ToFloat4(value);
  color.w *= alpha;
  return color;
}

}  // namespace

ImFont* AddSansFont(float size) {
  static constexpr ImWchar kGlyphRanges[] = {0x0020, 0x00ff, 0x2122, 0x2122, 0};
  return ImGui::GetIO().Fonts->AddFontFromMemoryCompressedTTF(
      tracy::DroidSans_compressed_data, tracy::DroidSans_compressed_size, size,
      nullptr, kGlyphRanges);
}

void Apply() {
  ImGuiStyle& style = ImGui::GetStyle();
  style = ImGuiStyle();
  ImGuiIO& io = ImGui::GetIO();
  io.FontDefault = AddSansFont(15.0f);
  g_monospace_font = io.Fonts->AddFontDefault();

  style.WindowPadding = ImVec2(20.0f, 20.0f);
  style.FramePadding = ImVec2(12.0f, 8.0f);
  style.ButtonTextAlign = ImVec2(0.0f, 0.5f);
  style.ItemSpacing = ImVec2(12.0f, 12.0f);
  style.ItemInnerSpacing = ImVec2(8.0f, 4.0f);
  style.CellPadding = ImVec2(8.0f, 6.0f);
  style.IndentSpacing = 20.0f;
  style.WindowRounding = 12.0f;
  style.ChildRounding = 12.0f;
  style.FrameRounding = 6.0f;
  style.PopupRounding = 12.0f;
  style.ScrollbarRounding = 4.0f;
  style.GrabRounding = 4.0f;
  style.TabRounding = 6.0f;
  style.WindowBorderSize = 0.0f;
  style.ChildBorderSize = 0.0f;
  style.PopupBorderSize = 0.0f;
  style.FrameBorderSize = 0.0f;
  style.ScrollbarSize = 8.0f;
  style.GrabMinSize = 6.0f;
  style.DisabledAlpha = 0.5f;

  ImVec4* colors = style.Colors;
  colors[ImGuiCol_Text] = Color(kText);
  colors[ImGuiCol_TextDisabled] = Color(kMuted);
  colors[ImGuiCol_WindowBg] = Color(kSurface);
  colors[ImGuiCol_ChildBg] = Color(kCanvas);
  colors[ImGuiCol_PopupBg] = Color(kRaised);
  colors[ImGuiCol_Border] = Color(kBorder);
  colors[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_FrameBg] = Color(kInset);
  colors[ImGuiCol_FrameBgHovered] = Color(kAccent, 0.22f);
  colors[ImGuiCol_FrameBgActive] = Color(kAccent, 0.32f);
  colors[ImGuiCol_TitleBg] = Color(kSurface);
  colors[ImGuiCol_TitleBgActive] = Color(kSurface);
  colors[ImGuiCol_TitleBgCollapsed] = Color(kSurface, 0.9f);
  colors[ImGuiCol_MenuBarBg] = Color(kSurface);
  colors[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_ScrollbarGrab] = Color(kMuted, 0.45f);
  colors[ImGuiCol_ScrollbarGrabHovered] = Color(kAccent, 0.7f);
  colors[ImGuiCol_ScrollbarGrabActive] = Color(kAccent, 0.8f);
  colors[ImGuiCol_CheckMark] = Color(kText);
  colors[ImGuiCol_SliderGrab] = Color(kAccent);
  colors[ImGuiCol_SliderGrabActive] = Color(kAccent);
  colors[ImGuiCol_Button] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_ButtonHovered] = Color(kText, 0.08f);
  colors[ImGuiCol_ButtonActive] = Color(kAccent, 0.4f);
  colors[ImGuiCol_Header] = Color(kAccent, 0.2f);
  colors[ImGuiCol_HeaderHovered] = Color(kText, 0.08f);
  colors[ImGuiCol_HeaderActive] = Color(kAccent, 0.5f);
  colors[ImGuiCol_Separator] = Color(kBorder);
  colors[ImGuiCol_SeparatorHovered] = Color(kAccent, 0.7f);
  colors[ImGuiCol_SeparatorActive] = Color(kAccent);
  colors[ImGuiCol_ResizeGrip] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_ResizeGripHovered] = Color(kAccent, 0.65f);
  colors[ImGuiCol_ResizeGripActive] = Color(kAccent);
  colors[ImGuiCol_Tab] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_TabHovered] = Color(kText, 0.08f);
  colors[ImGuiCol_TabActive] = Color(kAccent, 0.18f);
  colors[ImGuiCol_TabUnfocused] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_TabUnfocusedActive] = Color(kRaised);
  colors[ImGuiCol_DockingPreview] = Color(kAccent, 0.35f);
  colors[ImGuiCol_DockingEmptyBg] = Color(kCanvas);
  colors[ImGuiCol_PlotLines] = Color(kAccent);
  colors[ImGuiCol_PlotLinesHovered] = Color(kAccent);
  colors[ImGuiCol_PlotHistogram] = Color(kAccent);
  colors[ImGuiCol_PlotHistogramHovered] = Color(kAccent);
  colors[ImGuiCol_TableHeaderBg] = Color(kRaised);
  colors[ImGuiCol_TableBorderStrong] = Color(kBorder);
  colors[ImGuiCol_TableBorderLight] = Color(kBorder, 0.5f);
  colors[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_TableRowBgAlt] = Color(kRaised, 0.35f);
  colors[ImGuiCol_TextSelectedBg] = Color(kAccent, 0.4f);
  colors[ImGuiCol_DragDropTarget] = Color(kAccent);
  colors[ImGuiCol_NavHighlight] = Color(kAccent, 0.9f);
  colors[ImGuiCol_NavWindowingHighlight] = Color(kAccent, 0.7f);
  colors[ImGuiCol_NavWindowingDimBg] = Color(kCanvas, 0.65f);
  colors[ImGuiCol_ModalWindowDimBg] = Color(kCanvas, 0.75f);
}

ImFont* MonospaceFont() {
  return g_monospace_font;
}

void DrawPanel(ImDrawList* draw_list,
               ImVec2 top_left,
               ImVec2 bottom_right,
               float opacity) {
  const float rounding = ImGui::GetStyle().WindowRounding;
  draw_list->AddRectFilled(
      top_left, bottom_right,
      ImGui::GetColorU32(ImGuiCol_WindowBg, 0.95f * opacity), rounding);
}

void DrawMark(ImDrawList* draw_list, ImVec2 position) {
  draw_list->AddTriangleFilled(ImVec2(position.x + 6, position.y),
                               ImVec2(position.x, position.y + 12),
                               ImVec2(position.x + 12, position.y + 12), kText);
  draw_list->AddTriangleFilled(ImVec2(position.x + 6, position.y + 4),
                               ImVec2(position.x + 3, position.y + 10),
                               ImVec2(position.x + 9, position.y + 10),
                               kSurface);
}

}  // namespace ui::overlay_theme
