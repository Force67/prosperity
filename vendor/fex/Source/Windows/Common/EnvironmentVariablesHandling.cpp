// SPDX-License-Identifier: MIT
#include <libloaderapi.h>
#include <stdlib.h>
#include <winternl.h>
#include <wine/unixlib.h>

namespace FEX::Windows {
void SetupEnvironmentVariableValues(HMODULE NtDll) {
  auto Sym = GetProcAddress(NtDll, "__wine_set_unix_env");

  // On an ARM system we are expecting uncached memory to have marked overhead due to atomic memory usage.
  // If `tu_override_uncached_as_cache_coherent` isn't already controlled by the user, then set it here.
  // This lets the `turnip` mesa driver allocate uncached memory types as cached coherent instead if the hardware supports it.
  //
  // The only downside for this approach is that the GPU /may/ need to snoop CPU cachelines which could add some additional load
  // on to the GPU side, but is unlikely to matter due to x86 emulation using excessive load-acquire/store-release semantic memory operations.
  //
  // Other UMA ARM platforms will run in to the same problem, with their own workarounds:
  // - NVIDIA Tegra/Spark: Their proprietary driver always maps uncached as cached-coherent.
  // - Windows Snapdragon: Same as Tegra
  // - Asahi,Mali,PowerVR,Exynos,etc: Unknown behaviour
  //
  // This `__wine_set_unix_env` path may also not be long-term viable.
  // Make sure to communicate with the Wine/Proton people if this interface is expected to change!
  // A /possible/ workaround for Proton would be to set the environment variable in the launch script.
  // - That would have the downside that arm64 native Windows applications would get workaround set, which isn't strictly necessary.
  if (Sym && getenv("tu_override_uncached_as_cache_coherent") == nullptr) {
    auto WineSetEnv = reinterpret_cast<decltype(__wine_set_unix_env)>(Sym);
    WineSetEnv("tu_override_uncached_as_cache_coherent", "true");
  }
}
} // namespace FEX::Windows
