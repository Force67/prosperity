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

Service &playGoService();
Service &npManagerService();
Service &npWebService();

} // namespace krnl::ipmi
