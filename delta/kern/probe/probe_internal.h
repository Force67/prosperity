/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Shared between the two probe translation units: the generic guest-code
// probes in probe.cc and the per-title bring-up patches in probe_title.cc.
// Not part of the probe interface: kern calls probe.h, nothing else.

#include "base/arch.h"

#include <cstdint>

namespace kern {
class Process;
class Module;

namespace probe {

// probe_title.cc, armed from OnProcessCreated / OnModuleLoaded / OnBeforeStart.
void BringUpRebirthEbootRegistry(Module& m);
void BringUpRebirthSurfaceRegistry(Module& m);
void PatchVideoOutDiag(Module& m);
void InvestigateDcbGate(Module& m);
void InstallAllocLock(Module& m);
void InstallMatTrace(Module& m);
void InstallJobTrace(Module& m);
void ApplyBootPatches(Process& p);

// DELTA_IMPORT_TRACE: route an import through a recording stub (see
// import_trace.cc); returns real_addr when it is not traced.
uintptr_t MaybeTraceImport(const char* nid_name, uintptr_t real_addr);

}  // namespace probe
}  // namespace kern
