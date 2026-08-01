#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * The service instances the IPMI manager can route to. Accessors rather than
 * self-registration: these live in a static library, so a service nobody names
 * here would be dropped by the linker.
 */

#include "ipmi.h"

namespace krnl::ipmi {

// The one running title's appId, as ShellCore would have assigned it. It leaks
// into guest-visible state in two places that must agree: the SceLncService
// GetAppStatus reply, and the SceShellCoreUtilAppFocus/CtrlFocus event-flag
// patterns (libSceSystemService's GetStatus reports the app backgrounded or
// overlaid unless the flag pattern equals its own appId).
constexpr uint32_t kForegroundAppId = 0x60000001;

Service &playGoService();
Service &npManagerService();
Service &npWebService();
Service &userService();
Service &lncService();

} // namespace krnl::ipmi
