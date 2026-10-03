#include "ui/home_screen.h"

#if defined(__linux__) && !defined(__ANDROID__)
#include <SDL3/SDL.h>
#include <stb_image.h>
#include <sys/stat.h>

#include "base/algorithm.h"
#include "base/memory/mem_ops.h"
#include "base/random/random.h"
#include "base/strings/format.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/time/time.h"
#include "imgui.h"
#include "options/options.h"
#include "ui/home_background.h"
#include "ui/overlay.h"
#include "ui/overlay_theme.h"
#include "ui/settings.h"

namespace ui {
namespace {
DELTA_OPTION(
    u32,
    kBackgroundStyle,
    "DELTA_UI_BACKGROUND",
    0,
    "home fallback: 0 random, 1 current, 2 liquid glass, 3 signal formation");

struct Artwork {
  int icon = -1;
  int background = -1;
};
struct HomeMotion {
  base::Vector<float> card_focus;
  mem_size title_selection = 0;
  u64 title_ns = 0;
  u64 tick_ns = 0;
};
HomeMotion g_motion;

struct AtlasImage {
  int rect;
  u32 width, height;
  base::Vector<u8> pixels;
};

bool g_active = false;
u32 g_background_style = 1;
bool g_ps4_ready = false;
bool g_ps5_ready = false;
bool g_done = false;
const base::Vector<HomeGame>* g_games = nullptr;
base::Vector<Artwork> g_artwork;
mem_size g_selected = 0;
mem_size g_first = 0;
mem_size g_background_selection = 0;
mem_size g_previous_background = 0;
float g_background_transition = 1;
u64 g_started_ns = 0;
u64 g_selection_ns = 0;
u64 g_launch_ns = 0;
u64 g_game_frame_ns = 0;
u64 g_boot_ns = 0;
bool g_booting = false;
Artwork g_launch_artwork;
base::String g_launch_name;

float FadeProgress(u64 start, float seconds) {
  const float t = base::Min(
      float(base::TickClock::NowNs() - start) / (seconds * 1e9f), 1.0f);
  return t * t * (3 - 2 * t);
}

float LaunchOpacity() {
  return g_game_frame_ns ? 1 - FadeProgress(g_game_frame_ns, 0.55f) : 1;
}
ImFont* g_heading = nullptr;
base::String g_result;
base::String g_error;
base::String g_launch_error;
base::String (*g_check_game)(const base::String&) = nullptr;
base::Mutex g_dialog_mutex;
bool g_dialog_pending = false;
base::String g_dialog_path;
base::String g_add_path;
base::String g_dialog_error;

int AddArtwork(const base::Vector<u8>& png,
               u32 width,
               u32 height,
               base::Vector<AtlasImage>* images) {
  if (png.empty() || png.size() > (16u << 20))
    return -1;
  int w, h, channels;
  if (!stbi_info_from_memory(png.data(), static_cast<int>(png.size()), &w, &h,
                             &channels) ||
      w < 1 || h < 1 || w > 4096 || h > 4096)
    return -1;
  auto* pixels = stbi_load_from_memory(png.data(), static_cast<int>(png.size()),
                                       &w, &h, &channels, STBI_rgb_alpha);
  if (!pixels)
    return -1;
  AtlasImage image;
  image.rect = ImGui::GetIO().Fonts->AddCustomRectRegular(width, height);
  image.width = width;
  image.height = height;
  image.pixels.resize(width * height * 4);
  // Crop to fill, so square icons and wide backdrops keep their proportions.
  const float scale = base::Max(float(width) / w, float(height) / h);
  const float left = (w - width / scale) * 0.5f;
  const float top = (h - height / scale) * 0.5f;
  for (u32 y = 0; y < height; ++y) {
    const float sy = top + y / scale;
    const int y0 = base::Min(static_cast<int>(sy), h - 1);
    const int y1 = base::Min(y0 + 1, h - 1);
    const float fy = sy - y0;
    for (u32 x = 0; x < width; ++x) {
      const float sx = left + x / scale;
      const int x0 = base::Min(static_cast<int>(sx), w - 1);
      const int x1 = base::Min(x0 + 1, w - 1);
      const float fx = sx - x0;
      for (int c = 0; c < 4; ++c) {
        const float a = pixels[(y0 * w + x0) * 4 + c] * (1 - fx) +
                        pixels[(y0 * w + x1) * 4 + c] * fx;
        const float b = pixels[(y1 * w + x0) * 4 + c] * (1 - fx) +
                        pixels[(y1 * w + x1) * 4 + c] * fx;
        image.pixels[(y * width + x) * 4 + c] =
            static_cast<u8>(a * (1 - fy) + b * fy);
      }
    }
  }
  stbi_image_free(pixels);
  const int rect = image.rect;
  images->push_back(base::move(image));
  return rect;
}

void PrepareArtwork() {
  ImGuiIO& io = ImGui::GetIO();
  g_heading = overlay_theme::HeadingFont();
  io.Fonts->TexDesiredWidth = 2048;
  base::Vector<AtlasImage> images;
  g_artwork.clear();
  for (const auto& game : *g_games) {
    Artwork art;
    art.icon = AddArtwork(game.icon, 192, 192, &images);
    art.background = AddArtwork(game.artwork, 640, 360, &images);
    g_artwork.push_back(art);
  }
  u8* atlas;
  int width, height;
  io.Fonts->GetTexDataAsRGBA32(&atlas, &width, &height);
  for (const auto& image : images) {
    const auto* rect = io.Fonts->GetCustomRectByIndex(image.rect);
    for (u32 y = 0; y < image.height; ++y)
      base::MemCopy(atlas + ((rect->Y + y) * width + rect->X) * 4,
                    image.pixels.data() + y * image.width * 4, image.width * 4);
  }
}

void DrawArtwork(ImDrawList* dl,
                 int rect_index,
                 ImVec2 tl,
                 ImVec2 br,
                 float rounding = 0,
                 ImU32 tint = IM_COL32_WHITE) {
  if (rect_index < 0)
    return;
  ImVec2 uv0, uv1;
  auto* atlas = ImGui::GetIO().Fonts;
  atlas->CalcCustomRectUV(atlas->GetCustomRectByIndex(rect_index), &uv0, &uv1);
  const auto* rect = atlas->GetCustomRectByIndex(rect_index);
  const float source_ratio = float(rect->Width) / rect->Height;
  const float target_ratio = (br.x - tl.x) / (br.y - tl.y);
  if (target_ratio > source_ratio) {
    const float crop =
        (uv1.y - uv0.y) * (1 - source_ratio / target_ratio) * 0.5f;
    uv0.y += crop;
    uv1.y -= crop;
  } else {
    const float crop =
        (uv1.x - uv0.x) * (1 - target_ratio / source_ratio) * 0.5f;
    uv0.x += crop;
    uv1.x -= crop;
  }
  dl->AddImageRounded(atlas->TexID, tl, br, uv0, uv1, tint, rounding);
}

void Play(const base::String& path) {
  if (g_launch_ns)
    return;
  struct stat info{};
  if (::stat(path.c_str(), &info) != 0) {
    g_error = "This game is no longer at its saved location.";
    return;
  }
  if (g_check_game) {
    g_launch_error = g_check_game(path);
    if (!g_launch_error.empty())
      return;
  }
  g_result = path;
  g_launch_ns = base::TickClock::NowNs();
  g_launch_name = "Prosperity";
  g_launch_artwork = {};
  for (mem_size i = 0; i < g_games->size(); ++i) {
    if ((*g_games)[i].path == path) {
      g_selected = i;
      g_launch_name = (*g_games)[i].name;
      g_launch_artwork = g_artwork[i];
      break;
    }
  }
}

void SDLCALL DialogResult(void*, const char* const* files, int) {
  base::LockGuard<base::Mutex> lock(g_dialog_mutex);
  g_dialog_pending = false;
  if (!files)
    g_dialog_error =
        "Could not open the file picker. Check your desktop portal.";
  else if (files[0])
    g_dialog_path = files[0];
}

void OpenGame(bool folder) {
  {
    base::LockGuard<base::Mutex> lock(g_dialog_mutex);
    if (g_dialog_pending)
      return;
    g_dialog_pending = true;
  }
  auto* window = SDL_GetKeyboardFocus();
  static const SDL_DialogFileFilter kFilters[] = {
      {"Games (packages, archives, or eboot.bin)", "pkg;ffpkg;zip;rar;bin;elf"},
      {"All files", "*"}};
  if (folder)
    SDL_ShowOpenFolderDialog(DialogResult, nullptr, window, nullptr, false);
  else
    SDL_ShowOpenFileDialog(DialogResult, nullptr, window, kFilters, 2, nullptr,
                           false);
}

bool DialogPending() {
  base::LockGuard<base::Mutex> lock(g_dialog_mutex);
  return g_dialog_pending;
}

void ReadDialogResult() {
  base::String path;
  {
    base::LockGuard<base::Mutex> lock(g_dialog_mutex);
    path = base::move(g_dialog_path);
    if (!g_dialog_error.empty())
      g_error = base::move(g_dialog_error);
  }
  if (!path.empty())
    g_add_path = base::move(path);
}
float DrawShortcut(ImDrawList* dl,
                   float x,
                   float y,
                   const char* key,
                   const char* label) {
  const float key_width = ImGui::CalcTextSize(key).x + 16;
  dl->AddRectFilled(ImVec2(x, y), ImVec2(x + key_width, y + 24),
                    overlay_theme::kSurface, 6);
  dl->AddText(ImVec2(x + 8, y + 4), overlay_theme::kText, key);
  dl->AddText(ImVec2(x + key_width + 8, y + 4), overlay_theme::kSecondary,
              label);
  return x + key_width + ImGui::CalcTextSize(label).x + 32;
}

void DrawFirmwareWarning(ImDrawList* dl, float width, float margin) {
  const ImVec2 tl(margin, 72);
  const ImVec2 br(width - margin, 172);
  dl->AddRectFilled(tl, br, IM_COL32(37, 33, 26, 245), 50);
  dl->AddRect(tl, br, IM_COL32(230, 192, 113, 55), 50);
  const ImVec2 center(tl.x + 44, tl.y + 50);
  dl->AddCircleFilled(center, 18, overlay_theme::kWarning);
  dl->AddText(ImVec2(center.x - 2, center.y - 8), overlay_theme::kCanvas, "!");
  const char* title = !g_ps4_ready && !g_ps5_ready
                          ? "PS4 and PS5 firmware modules are missing"
                          : (!g_ps4_ready ? "PS4 firmware modules are missing"
                                          : "PS5 firmware modules are missing");
  const char* command =
      !g_ps4_ready && !g_ps5_ready
          ? "ps4delta --configure-ps4-fw <folder>   |   ps4delta "
            "--configure-ps5-fw <folders>"
          : (!g_ps4_ready ? "ps4delta --configure-ps4-fw <folder>"
                          : "ps4delta --configure-ps5-fw <folders>");
  dl->AddText(ImVec2(tl.x + 80, tl.y + 18), overlay_theme::kText, title);
  dl->AddText(ImVec2(tl.x + 80, tl.y + 42), overlay_theme::kSecondary,
              "Install your decrypted firmware modules before playing.");
  dl->AddText(overlay_theme::MonospaceFont(), 13, ImVec2(tl.x + 80, tl.y + 66),
              overlay_theme::kWarning, command);
}
}  // namespace

bool HomeScreenActive() {
  return g_active;
}

HomeBackground HomeScreenBackground() {
  HomeBackground background;
  if (!g_active && !LaunchTransitionActive())
    return background;
  background.visible =
      g_active ? g_games->empty() || g_artwork[g_selected].background < 0
               : g_launch_artwork.background < 0;
  background.opacity = g_active ? 1 : LaunchOpacity();
  background.style = g_background_style;
  const auto now = base::TickClock::NowNs();
  background.time = float(now - g_started_ns) / 1e9f;
  background.pulse = g_selection_ns ? float(now - g_selection_ns) / 1e9f : 20;
  return background;
}

void HomeScreenBuild(u32 width, u32 height) {
  if (!g_launch_ns)
    ReadDialogResult();
  const auto now = base::TickClock::NowNs();
  const float dt = base::Min(float(now - g_motion.tick_ns) / 1e9f, 0.05f);
  g_motion.tick_ns = now;
  const float w = float(width), h = float(height);
  const float margin = base::Clamp(w * 0.05f, 24.0f, 64.0f);
  const float tile = base::Clamp(w * 0.09f, 72.0f, 120.0f);
  const bool missing_firmware = !g_ps4_ready || !g_ps5_ready;
  const float row_y =
      base::Max(96.0f, h * 0.15f) + (missing_firmware ? 112.0f : 0.0f);
  const mem_size visible = base::Max(
      mem_size(1), static_cast<mem_size>((w - margin * 2) / (tile + 20)));
  const bool launch_error = !g_launch_error.empty();
  if (!SettingsVisible() && !g_launch_ns && !launch_error && !DialogPending() &&
      !g_games->empty()) {
    if ((ImGui::IsKeyPressed(ImGuiKey_LeftArrow) ||
         ImGui::GetIO().MouseWheel > 0) &&
        g_selected)
      --g_selected;
    if ((ImGui::IsKeyPressed(ImGuiKey_RightArrow) ||
         ImGui::GetIO().MouseWheel < 0) &&
        g_selected + 1 < g_games->size())
      ++g_selected;
    if (ImGui::IsKeyPressed(ImGuiKey_Enter) ||
        ImGui::IsKeyPressed(ImGuiKey_Space))
      Play((*g_games)[g_selected].path);
  }
  if (!SettingsVisible() && !g_launch_ns && g_launch_error.empty() &&
      !DialogPending()) {
    if (ImGui::IsKeyPressed(ImGuiKey_Escape))
      g_done = true;
    if (ImGui::IsKeyPressed(ImGuiKey_O))
      OpenGame(false);
    if (ImGui::IsKeyPressed(ImGuiKey_S))
      SettingsOpen();
  }
  if (g_selected < g_first)
    g_first = g_selected;
  if (g_selected >= g_first + visible)
    g_first = g_selected - visible + 1;

  ImDrawList* bg = ImGui::GetBackgroundDrawList();
  if (!HomeScreenBackground().visible)
    bg->AddRectFilled(ImVec2(0, 0), ImVec2(w, h), overlay_theme::kCanvas);
  if (!g_games->empty()) {
    if (g_selected != g_background_selection) {
      g_previous_background = g_background_selection;
      g_background_selection = g_selected;
      g_background_transition = 0;
      g_selection_ns = base::TickClock::NowNs();
    }
    g_background_transition = base::Min(
        1.0f, g_background_transition + ImGui::GetIO().DeltaTime / 0.2f);
    const float fade = g_background_transition * (2 - g_background_transition);
    if (fade < 1)
      DrawArtwork(bg, g_artwork[g_previous_background].background, ImVec2(0, 0),
                  ImVec2(w, h), 0,
                  IM_COL32(255, 255, 255, static_cast<int>(255 * (1 - fade))));
    DrawArtwork(bg, g_artwork[g_background_selection].background, ImVec2(0, 0),
                ImVec2(w, h), 0,
                IM_COL32(255, 255, 255, static_cast<int>(255 * fade)));
    if (!HomeScreenBackground().visible)
      bg->AddRectFilledMultiColor(
          ImVec2(0, 0), ImVec2(w, h), IM_COL32(12, 13, 17, 210),
          IM_COL32(12, 13, 17, 95), IM_COL32(12, 13, 17, 210),
          IM_COL32(12, 13, 17, 245));
  }
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(w, h));
  ImGui::Begin("Home", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoSavedSettings |
                   ImGuiWindowFlags_NoBackground);
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const int first_vertex = dl->VtxBuffer.Size;
  ImGui::BeginDisabled(SettingsVisible() || g_launch_ns != 0 ||
                       !g_launch_error.empty());
  overlay_theme::DrawMark(dl, ImVec2(margin, 34));
  dl->AddText(ImVec2(margin + 24, 32), overlay_theme::kText, "Prosperity");
  dl->AddRectFilled(ImVec2(margin + 132, 24), ImVec2(margin + 204, 56),
                    overlay_theme::WithOpacity(overlay_theme::kText, 0.06f),
                    16);
  dl->AddText(ImVec2(margin + 148, 32), overlay_theme::kText, "Games");
  ImGui::SetCursorPos(ImVec2(w - margin - 312, 24));
  ImGui::BeginDisabled(DialogPending());
  if (ImGui::Button("Add game", ImVec2(120, 36)))
    OpenGame(false);
  ImGui::SameLine(0, 16);
  if (ImGui::Button("Add folder", ImVec2(120, 36)))
    OpenGame(true);
  ImGui::SameLine(0, 16);
  if (SettingsButton())
    SettingsOpen();
  ImGui::EndDisabled();

  if (missing_firmware)
    DrawFirmwareWarning(dl, w, margin);

  if (g_games->empty()) {
    const ImVec2 tl(margin,
                    base::Max(missing_firmware ? 196.0f : 120.0f, h * 0.30f));
    const ImVec2 br(base::Min(w - margin, margin + 640), tl.y + 212);
    overlay_theme::DrawPanel(dl, tl, br);
    dl->AddText(g_heading, 32, ImVec2(tl.x + 24, tl.y + 24),
                overlay_theme::kText, "Your next game starts here");
    dl->AddText(ImVec2(tl.x + 24, tl.y + 80), overlay_theme::kSecondary,
                "Add a game file or an extracted game folder.");
    dl->AddText(ImVec2(tl.x + 24, tl.y + 104), overlay_theme::kSecondary,
                "Added games appear here. Select Play when you are ready.");
    ImGui::SetCursorPos(ImVec2(tl.x + 24, tl.y + 152));
    if (ImGui::Button("Add a game", ImVec2(176, 40)))
      OpenGame(false);
  } else {
    dl->AddText(ImVec2(margin, row_y - 28), overlay_theme::kSecondary,
                "Recently played");
    const auto count =
        base::Format("{:02} / {:02}", g_selected + 1, g_games->size());
    dl->AddText(overlay_theme::MonospaceFont(), 13,
                ImVec2(margin + 124, row_y - 27), overlay_theme::kMuted,
                count.c_str());
    if (g_games->size() > visible) {
      ImGui::SetCursorPos(ImVec2(w - margin - 84, row_y - 36));
      ImGui::BeginDisabled(g_selected == 0 || DialogPending());
      if (ImGui::Button("<", ImVec2(32, 28)))
        --g_selected;
      ImGui::EndDisabled();
      ImGui::SameLine(0, 12);
      ImGui::BeginDisabled(g_selected + 1 == g_games->size() ||
                           DialogPending());
      if (ImGui::Button(">", ImVec2(32, 28)))
        ++g_selected;
      ImGui::EndDisabled();
    }
    for (mem_size i = g_first;
         i < base::Min(g_games->size(), g_first + visible); ++i) {
      const auto& game = (*g_games)[i];
      const ImVec2 slot(margin + (i - g_first) * (tile + 20), row_y);
      ImGui::PushID(static_cast<int>(i));
      ImGui::SetCursorPos(ImVec2(slot.x - 4, slot.y - 8));
      if (ImGui::InvisibleButton("game", ImVec2(tile + 8, tile + 8)))
        g_selected = i;
      const bool hovered = ImGui::IsItemHovered();
      if (!DialogPending() && hovered && ImGui::IsMouseDoubleClicked(0))
        Play(game.path);
      const bool focused = i == g_selected;
      float& focus = g_motion.card_focus[i];
      const float target = focused ? 1 : (hovered ? 0.35f : 0);
      focus += (target - focus) * base::Min(dt * 14, 1.0f);
      const ImVec2 tl(slot.x - 4 * focus, slot.y - 8 * focus);
      const ImVec2 br(slot.x + tile + 4 * focus, slot.y + tile);
      overlay_theme::DrawFocusHalo(dl, tl, br, 12, focus);
      dl->AddRectFilled(tl, br, overlay_theme::kRaised, 12);
      if (g_artwork[i].icon >= 0)
        DrawArtwork(
            dl, g_artwork[i].icon, tl, br, 12,
            game.available ? IM_COL32_WHITE : IM_COL32(255, 255, 255, 110));
      else
        dl->AddText(g_heading, 32, ImVec2(tl.x + 28, tl.y + tile * 0.35f),
                    overlay_theme::kSecondary, game.is_ps5 ? "PS5" : "PS4");
      const ImVec2 chip(slot.x, br.y + 12);
      dl->AddRectFilled(chip, ImVec2(chip.x + 44, chip.y + 24),
                        overlay_theme::kSurface, 12);
      dl->AddRectFilled(
          chip, ImVec2(chip.x + 44, chip.y + 24),
          overlay_theme::WithOpacity(overlay_theme::kAccent, focus * 0.15f),
          12);
      dl->AddText(ImVec2(chip.x + 9, chip.y + 4), overlay_theme::kText,
                  game.is_ps5 ? "PS5" : "PS4");
      if (focus > 0.01f)
        dl->AddRect(ImVec2(tl.x - 4, tl.y - 4), ImVec2(br.x + 4, br.y + 4),
                    overlay_theme::WithOpacity(overlay_theme::kText, focus), 16,
                    0, 1.5f);
      ImGui::PopID();
    }
    const auto& game = (*g_games)[g_selected];
    if (g_motion.title_selection != g_selected) {
      g_motion.title_selection = g_selected;
      g_motion.title_ns = now;
    }
    const float reveal = FadeProgress(g_motion.title_ns, 0.24f);
    const float title_y =
        base::Max(row_y + tile + 56, h * 0.55f) + 8 * (1 - reveal);
    const float title_size = base::Min(36.0f, w / 28);
    const float title_width = w - margin * 2;
    const auto size =
        g_heading->CalcTextSizeA(title_size, title_width, 0, game.name.c_str());
    dl->AddText(g_heading, title_size, ImVec2(margin, title_y),
                overlay_theme::WithOpacity(overlay_theme::kText, reveal),
                game.name.c_str());
    const auto subtitle =
        base::String(game.is_ps5 ? "PS5" : "PS4") +
        (game.title_id.empty() ? "" : "  /  " + game.title_id);
    dl->AddText(ImVec2(margin, title_y + size.y + 10),
                overlay_theme::WithOpacity(overlay_theme::kSecondary, reveal),
                subtitle.c_str());
    ImGui::SetCursorPos(
        ImVec2(margin, base::Min(title_y + size.y + 40, h - 80)));
    ImGui::PushStyleColor(ImGuiCol_Button, overlay_theme::kText);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(216, 229, 255, 255));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, overlay_theme::kAccent);
    ImGui::PushStyleColor(ImGuiCol_Text, overlay_theme::kCanvas);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 22);
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.54f, 0.5f));
    ImGui::BeginDisabled(!game.available || DialogPending());
    if (ImGui::Button(game.available ? "Play" : "Game unavailable",
                      ImVec2(200, 44)))
      Play(game.path);
    if (game.available) {
      const auto tl = ImGui::GetItemRectMin();
      const auto br = ImGui::GetItemRectMax();
      if (ImGui::IsItemHovered())
        overlay_theme::DrawFocusHalo(dl, tl, br, 22, 1);
      dl->AddTriangleFilled(
          ImVec2(tl.x + 70, tl.y + 16), ImVec2(tl.x + 70, tl.y + 28),
          ImVec2(tl.x + 80, tl.y + 22), overlay_theme::kCanvas);
    }
    ImGui::EndDisabled();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(4);
  }
  float hint_x = margin;
  hint_x = DrawShortcut(dl, hint_x, h - 40, "< >", "Browse");
  hint_x = DrawShortcut(dl, hint_x, h - 40, "Enter", "Play");
  hint_x = DrawShortcut(dl, hint_x, h - 40, "O", "Add game");
  hint_x = DrawShortcut(dl, hint_x, h - 40, "S", "Settings");
  DrawShortcut(dl, hint_x, h - 40, "Esc", "Exit");
  if (!g_error.empty())
    dl->AddText(ImVec2(margin, h - 64), overlay_theme::kWarning,
                g_error.c_str());
  const auto background = HomeScreenBackground();
  if (background.visible && background.style == 4 && background.time > 6) {
    constexpr char kAnniversary[] = "EST 2019";
    ImFont* font = overlay_theme::MonospaceFont();
    const float opacity = base::Min((background.time - 6) / 2, 1.0f);
    const float text_width = font->CalcTextSizeA(15, w, 0, kAnniversary).x;
    const float x = w * 0.5f + h * 0.39f - text_width * 0.5f;
    const float y = base::Min(h * 0.81f, h - 72);
    dl->AddText(font, 15, ImVec2(x, y),
                IM_COL32(220, 202, 167, static_cast<int>(opacity * 175)),
                kAnniversary);
  }
  ImGui::EndDisabled();
  if (SettingsVisible())
    dl->AddRectFilled(
        ImVec2(0, 0), ImVec2(w, h),
        overlay_theme::WithOpacity(overlay_theme::kCanvas, 0.65f));
  if (g_launch_ns) {
    const float fade = FadeProgress(g_launch_ns, 0.35f);
    for (int i = first_vertex; i < dl->VtxBuffer.Size; ++i) {
      auto& vertex = dl->VtxBuffer[i];
      const u32 alpha = vertex.col >> IM_COL32_A_SHIFT;
      vertex.col = (vertex.col & ~IM_COL32_A_MASK) |
                   (u32(alpha * (1 - fade)) << IM_COL32_A_SHIFT);
      vertex.pos.y -= 12 * fade;
    }
  }
  ImGui::End();
  if (!g_launch_error.empty())
    ImGui::OpenPopup("##firmware_error");
  ImGui::SetNextWindowPos(ImVec2(w * 0.5f, h * 0.5f), ImGuiCond_Always,
                          ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(base::Min(600.0f, w - 48), 0));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(40, 32));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 16);
  if (ImGui::BeginPopupModal("##firmware_error", nullptr,
                             ImGuiWindowFlags_NoDecoration |
                                 ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoSavedSettings |
                                 ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::PushFont(g_heading);
    ImGui::TextUnformatted("Can't start this game");
    ImGui::PopFont();
    ImGui::Dummy(ImVec2(0, 16));
    ImGui::SetWindowFontScale(1.25f);
    ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x +
                           ImGui::GetCursorPosX());
    ImGui::TextUnformatted(g_launch_error.c_str());
    ImGui::PopTextWrapPos();
    ImGui::Dummy(ImVec2(0, 28));
    const float button_width =
        base::Min(240.0f, ImGui::GetContentRegionAvail().x);
    ImGui::SetCursorPosX((ImGui::GetWindowSize().x - button_width) * 0.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 22);
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.96f, 0.97f, 0.98f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          ImVec4(0.86f, 0.89f, 0.94f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                          ImVec4(0.74f, 0.80f, 0.90f, 1));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.05f, 0.05f, 0.07f, 1));
    if (ImGui::Button("OK", ImVec2(button_width, 44)) ||
        (launch_error && (ImGui::IsKeyPressed(ImGuiKey_Enter) ||
                          ImGui::IsKeyPressed(ImGuiKey_Space) ||
                          ImGui::IsKeyPressed(ImGuiKey_Escape)))) {
      g_launch_error.clear();
      ImGui::CloseCurrentPopup();
    }
    ImGui::SetItemDefaultFocus();
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(2);
    ImGui::SetWindowFontScale(1);
    ImGui::EndPopup();
  }
  ImGui::PopStyleVar(2);
  SettingsBuild(width, height);
}

bool LaunchTransitionActive() {
  return g_booting && LaunchOpacity() > 0;
}

void LaunchTransitionGameReady() {
  if (g_booting && !g_game_frame_ns)
    g_game_frame_ns = base::TickClock::NowNs();
}

void LaunchTransitionBuild(u32 width, u32 height) {
  const float w = float(width), h = float(height);
  const float opacity = LaunchOpacity();
  const auto tint = IM_COL32(255, 255, 255, int(255 * opacity));
  auto* bg = ImGui::GetBackgroundDrawList();
  const float zoom = 0.025f * FadeProgress(g_boot_ns, 6);
  DrawArtwork(bg, g_launch_artwork.background, ImVec2(-w * zoom, -h * zoom),
              ImVec2(w * (1 + zoom), h * (1 + zoom)), 0, tint);
  const float enter = FadeProgress(g_boot_ns, 0.45f);
  bg->AddRectFilledMultiColor(
      ImVec2(0, 0), ImVec2(w, h),
      IM_COL32(12, 13, 17, int(210 * (1 - enter) * opacity)),
      IM_COL32(12, 13, 17, int(95 * (1 - enter) * opacity)),
      IM_COL32(12, 13, 17, int((210 - 30 * enter) * opacity)),
      IM_COL32(12, 13, 17, int((245 - 65 * enter) * opacity)));
  const float x = base::Clamp(w * 0.05f, 24.0f, 64.0f);
  const float y = h - 100 + 16 * (1 - enter);
  const auto ink = IM_COL32(240, 244, 250, int(255 * opacity * enter));
  if (g_launch_artwork.icon >= 0)
    DrawArtwork(bg, g_launch_artwork.icon, ImVec2(x, y - 4),
                ImVec2(x + 64, y + 60), 14, ink);
  const float text_x = x + (g_launch_artwork.icon >= 0 ? 84 : 0);
  bg->AddText(g_heading, 26, ImVec2(text_x, y), ink, g_launch_name.c_str());
  const float angle = float(base::TickClock::NowNs() - g_launch_ns) / 1e9f * 4;
  bg->PathArcTo(ImVec2(text_x + 8, y + 44), 6, angle, angle + 4.5f, 24);
  bg->PathStroke(ink, 0, 1.5f);
  bg->AddText(ImVec2(text_x + 24, y + 36), ink, "Starting game");
}

void HomeScreenSetBackground(u32 style) {
  kBackgroundStyle.set(style);
  g_background_style = style >= 1 && style <= 3
                           ? style
                           : ChooseHomeBackground(base::RandomUint(0, 767));
  g_started_ns = base::TickClock::NowNs();
}

void BeginHomeScreen(const base::Vector<HomeGame>& games,
                     bool ps4_ready,
                     bool ps5_ready,
                     base::String (*check_game)(const base::String&)) {
  const u32 requested = kBackgroundStyle.get();
  g_background_style = requested >= 1 && requested <= 3
                           ? requested
                           : ChooseHomeBackground(base::RandomUint(0, 767));
  g_started_ns = base::TickClock::NowNs();
  g_selection_ns = 0;
  g_launch_ns = g_game_frame_ns = g_boot_ns = 0;
  g_booting = false;
  g_ps4_ready = ps4_ready;
  g_ps5_ready = ps5_ready;
  g_games = &games;
  g_selected = g_first = 0;
  g_motion = {};
  g_motion.card_focus.resize(games.size());
  g_motion.title_ns = g_motion.tick_ns = g_started_ns;
  g_background_selection = g_previous_background = 0;
  g_background_transition = 1;
  g_done = false;
  g_result.clear();
  g_error.clear();
  g_launch_error.clear();
  g_check_game = check_game;
  OverlayEnsureImGui();
  PrepareArtwork();
  g_active = true;
}

base::String TakeHomeScreenAddPath() {
  return base::move(g_add_path);
}

void HomeScreenSetError(const base::String& error) {
  g_error = error;
}

bool HomeScreenDone() {
  return g_done || (g_launch_ns && FadeProgress(g_launch_ns, 0.35f) >= 1);
}

base::String EndHomeScreen() {
  if (!HomeScreenDone())
    g_result.clear();
  g_booting = !g_result.empty();
  g_boot_ns = base::TickClock::NowNs();
  g_active = false;
  g_games = nullptr;
  g_artwork.clear();
  g_motion.card_focus.clear();
  ImGui::GetIO().ClearInputKeys();
  return g_result;
}

}  // namespace ui
#else
namespace ui {
void BeginHomeScreen(const base::Vector<HomeGame>&,
                     bool,
                     bool,
                     base::String (*)(const base::String&)) {}
base::String TakeHomeScreenAddPath() {
  return {};
}
void HomeScreenSetError(const base::String&) {}
bool HomeScreenDone() {
  return true;
}
base::String EndHomeScreen() {
  return {};
}
bool HomeScreenActive() {
  return false;
}
void HomeScreenSetBackground(u32) {}
HomeBackground HomeScreenBackground() {
  return {};
}
void HomeScreenBuild(u32, u32) {}
bool LaunchTransitionActive() {
  return false;
}
void LaunchTransitionGameReady() {}
void LaunchTransitionBuild(u32, u32) {}
}  // namespace ui
#endif
