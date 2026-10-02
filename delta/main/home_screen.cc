#include "main/home_screen.h"

#include "main/firmware_config.h"
#include "main/game_firmware.h"

#if defined(__linux__) && !defined(__ANDROID__)
#include "base/threading/thread.h"
#include "host/window.h"

namespace cli {

base::String ShowHomeScreen(const base::Vector<ui::HomeGame>& games) {
  ui::BeginHomeScreen(games, FirmwareModulesReady(false),
                      FirmwareModulesReady(true), CheckGameFirmware);
  if (host::Init("Prosperity", 1280, 720)) {
    const u32 pixel = 0xff110d0c;
    host::Present(&pixel, 1, 1);
    while (!ui::HomeScreenDone() && host::PumpEvents()) {
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
