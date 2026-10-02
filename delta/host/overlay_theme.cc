#include "host/overlay_theme.h"

#include "DroidSans.hpp"

namespace host::overlay_theme {
namespace {

ImFont* g_monospace_font = nullptr;

ImVec4 Color(ImU32 value, float alpha = 1.0f) {
  ImVec4 color = ImGui::ColorConvertU32ToFloat4(value);
  color.w *= alpha;
  return color;
}

}  // namespace

void Apply() {
  ImGuiStyle& style = ImGui::GetStyle();
  style = ImGuiStyle();
  ImGuiIO& io = ImGui::GetIO();
  io.FontDefault = io.Fonts->AddFontFromMemoryCompressedTTF(
      tracy::DroidSans_compressed_data, tracy::DroidSans_compressed_size,
      15.0f);
  g_monospace_font = io.Fonts->AddFontDefault();

  style.WindowPadding = ImVec2(16.0f, 16.0f);
  style.FramePadding = ImVec2(12.0f, 8.0f);
  style.ButtonTextAlign = ImVec2(0.0f, 0.5f);
  style.ItemSpacing = ImVec2(12.0f, 8.0f);
  style.ItemInnerSpacing = ImVec2(8.0f, 4.0f);
  style.CellPadding = ImVec2(8.0f, 6.0f);
  style.IndentSpacing = 20.0f;
  style.WindowRounding = 0.0f;
  style.ChildRounding = 0.0f;
  style.FrameRounding = 0.0f;
  style.PopupRounding = 0.0f;
  style.ScrollbarRounding = 0.0f;
  style.GrabRounding = 0.0f;
  style.TabRounding = 0.0f;
  style.WindowBorderSize = 0.0f;
  style.ChildBorderSize = 0.0f;
  style.PopupBorderSize = 1.0f;
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
  colors[ImGuiCol_FrameBgHovered] = Color(kBlue, 0.22f);
  colors[ImGuiCol_FrameBgActive] = Color(kViolet, 0.32f);
  colors[ImGuiCol_TitleBg] = Color(kSurface);
  colors[ImGuiCol_TitleBgActive] = Color(kSurface);
  colors[ImGuiCol_TitleBgCollapsed] = Color(kSurface, 0.9f);
  colors[ImGuiCol_MenuBarBg] = Color(kSurface);
  colors[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_ScrollbarGrab] = Color(kMuted, 0.45f);
  colors[ImGuiCol_ScrollbarGrabHovered] = Color(kBlue, 0.7f);
  colors[ImGuiCol_ScrollbarGrabActive] = Color(kCyan, 0.8f);
  colors[ImGuiCol_CheckMark] = Color(kCyan);
  colors[ImGuiCol_SliderGrab] = Color(kBlue);
  colors[ImGuiCol_SliderGrabActive] = Color(kCyan);
  colors[ImGuiCol_Button] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_ButtonHovered] = Color(kCyan, 0.14f);
  colors[ImGuiCol_ButtonActive] = Color(kViolet, 0.4f);
  colors[ImGuiCol_Header] = Color(kViolet, 0.2f);
  colors[ImGuiCol_HeaderHovered] = Color(kCyan, 0.16f);
  colors[ImGuiCol_HeaderActive] = Color(kViolet, 0.5f);
  colors[ImGuiCol_Separator] = Color(kBorder);
  colors[ImGuiCol_SeparatorHovered] = Color(kBlue, 0.7f);
  colors[ImGuiCol_SeparatorActive] = Color(kCyan);
  colors[ImGuiCol_ResizeGrip] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_ResizeGripHovered] = Color(kBlue, 0.65f);
  colors[ImGuiCol_ResizeGripActive] = Color(kCyan);
  colors[ImGuiCol_Tab] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_TabHovered] = Color(kCyan, 0.12f);
  colors[ImGuiCol_TabActive] = Color(kViolet, 0.18f);
  colors[ImGuiCol_TabUnfocused] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_TabUnfocusedActive] = Color(kRaised);
  colors[ImGuiCol_DockingPreview] = Color(kCyan, 0.35f);
  colors[ImGuiCol_DockingEmptyBg] = Color(kCanvas);
  colors[ImGuiCol_PlotLines] = Color(kBlue);
  colors[ImGuiCol_PlotLinesHovered] = Color(kCyan);
  colors[ImGuiCol_PlotHistogram] = Color(kViolet);
  colors[ImGuiCol_PlotHistogramHovered] = Color(kCyan);
  colors[ImGuiCol_TableHeaderBg] = Color(kRaised);
  colors[ImGuiCol_TableBorderStrong] = Color(kBorder);
  colors[ImGuiCol_TableBorderLight] = Color(kBorder, 0.5f);
  colors[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
  colors[ImGuiCol_TableRowBgAlt] = Color(kRaised, 0.35f);
  colors[ImGuiCol_TextSelectedBg] = Color(kViolet, 0.4f);
  colors[ImGuiCol_DragDropTarget] = Color(kCyan);
  colors[ImGuiCol_NavHighlight] = Color(kCyan, 0.9f);
  colors[ImGuiCol_NavWindowingHighlight] = Color(kCyan, 0.7f);
  colors[ImGuiCol_NavWindowingDimBg] = Color(kCanvas, 0.65f);
  colors[ImGuiCol_ModalWindowDimBg] = Color(kCanvas, 0.75f);
}

ImFont* MonospaceFont() {
  return g_monospace_font;
}

void DrawPanel(ImDrawList* draw_list, ImVec2 top_left, ImVec2 bottom_right) {
  const ImVec2 outline[] = {top_left, ImVec2(bottom_right.x - 8, top_left.y),
                            ImVec2(bottom_right.x, top_left.y + 8),
                            bottom_right, ImVec2(top_left.x, bottom_right.y)};
  draw_list->AddConvexPolyFilled(outline, IM_ARRAYSIZE(outline),
                                 ImGui::GetColorU32(ImGuiCol_WindowBg, 0.95f));
  draw_list->AddLine(top_left, ImVec2(top_left.x + 32, top_left.y), kViolet, 2);
  draw_list->AddLine(ImVec2(top_left.x + 32, top_left.y),
                     ImVec2(top_left.x + 64, top_left.y), kCyan, 2);
  draw_list->AddLine(ImVec2(top_left.x, bottom_right.y), bottom_right, kBorder);
}

void DrawMark(ImDrawList* draw_list, ImVec2 position) {
  draw_list->AddTriangleFilled(
      ImVec2(position.x + 6, position.y), ImVec2(position.x, position.y + 12),
      ImVec2(position.x + 12, position.y + 12), kViolet);
  draw_list->AddTriangleFilled(ImVec2(position.x + 6, position.y + 4),
                               ImVec2(position.x + 3, position.y + 10),
                               ImVec2(position.x + 9, position.y + 10), kCyan);
}

}  // namespace host::overlay_theme
