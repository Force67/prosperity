/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Shared between the two probe translation units: the generic guest-code
// probes in probe.cc and the per-title bring-up patches in probe_title.cc.
// Not part of the probe interface: kern calls probe.h, nothing else.

#include "base/arch.h"

namespace krnl {
class Proc;
class Smodule;

namespace probe {

// probe_title.cc, armed from OnProcessCreated / OnModuleLoaded / OnBeforeStart.
void BringUpRebirthEbootRegistry(Smodule& m);
void BringUpRebirthSurfaceRegistry(Smodule& m);
void PatchVideoOutDiag(Smodule& m);
void InvestigateDcbGate(Smodule& m);
void InstallAllocLock(Smodule& m);
void InstallMatTrace(Smodule& m);
void InstallJobTrace(Smodule& m);
void ApplyBootPatches(Proc& p);

}  // namespace probe
}  // namespace krnl
