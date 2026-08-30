/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Shared between the two probe translation units: the generic guest-code
// probes in probe.cpp and the per-title bring-up patches in probe_title.cpp.
// Not part of the probe interface -- kern calls probe.h, nothing else.

#include "base/arch.h"

namespace krnl {
class proc;
class smodule;

namespace probe {

// probe_title.cpp, armed from onProcessCreated / onModuleLoaded / onBeforeStart.
void bringUpRebirthEbootRegistry(smodule &m);
void bringUpRebirthSurfaceRegistry(smodule &m);
void patchVideoOutDiag(smodule &m);
void investigateDcbGate(smodule &m);
void installAllocLock(smodule &m);
void installMatTrace(smodule &m);
void installJobTrace(smodule &m);
void applyBootPatches(proc &p);

}  // namespace probe
}  // namespace krnl
