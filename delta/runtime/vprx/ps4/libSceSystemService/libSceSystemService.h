#pragma once

/*
 * Partial HLE for libSceSystemService: a single export override. The rest of the
 * module stays LLE (the real .sprx runs; vprx_get returns 0 for the NIDs not
 * listed here, so they resolve to the loaded module).
 *
 * sceSystemServiceReportAbnormalTermination is a crash-telemetry call the title
 * invokes from its own fatal-error path. The real .sprx asserts (int 0x44 -> ud2)
 * when handed a NULL argument, turning a guest-level abort into a hard process
 * crash. It is a no-op for us (no telemetry backend), so report success.
 */

#include "../../vprx.h"

#include <cstdint>

extern "C" {
int PS4ABI sceSystemServiceReportAbnormalTermination(void *param);
}  // extern "C"
