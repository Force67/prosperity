#include "main/home_screen.h"

#include "main/firmware_config.h"
#include "main/game_firmware.h"
#include "main/recent_games.h"

#if defined(__linux__) && !defined(__ANDROID__)
#include "base/threading/thread.h"
#include "gpu/render/renderer.h"
#include "host/window.h"
#include "ui/overlay.h"
#include "ui/overlay_vk.h"
#include "ui/settings.h"

namespace cli {

base::String ShowHomeScreen(const base::Vector<ui::HomeGame>& games) {
  auto library = games;
  const bool retained_window = host::Available();
  if (retained_window) {
    ui::OverlayVkShutdown();
    ui::OverlayShutdownImGui();
  }
  ui::BeginHomeScreen(library, FirmwareModulesReady(false),
                      FirmwareModulesReady(true), CheckGameFirmware);
  if (host::Init("Prosperity", 1280, 720)) {
    if (retained_window)
      host::ReloadOverlay();
    ui::ConfigureSettings(gpu::render::GraphicsBackends(),
                          host::GraphicsDevices());
    const u32 pixel = 0xff110d0c;
    host::Present(&pixel, 1, 1);
    while (!ui::HomeScreenDone() && host::PumpEvents()) {
      const auto path = ui::TakeHomeScreenAddPath();
      if (!path.empty()) {
        const auto error = AddHomeGame(path);
        if (error.empty()) {
          ui::EndHomeScreen();
          ui::OverlayVkShutdown();
          ui::OverlayShutdownImGui();
          library = ReadRecentGames();
          ui::BeginHomeScreen(library, FirmwareModulesReady(false),
                              FirmwareModulesReady(true), CheckGameFirmware);
          host::ReloadOverlay();
        } else {
          ui::HomeScreenSetError(error);
        }
      }
      host::RefreshFrame(false);
      base::SleepForMilliseconds(16);
    }
  }
  auto selected = ui::EndHomeScreen();
  if (selected.empty())
    host::Shutdown();
  else
    host::ShowSplash({});
  return selected;
}

}  // namespace cli

#else
namespace cli {
base::String ShowHomeScreen(const base::Vector<ui::HomeGame>&) {
  return {};
}
}  // namespace cli
#endif
