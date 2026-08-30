/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Guest-code instrumentation: the traces, watches and per-title patches used to
// investigate what a title's own code is doing.
//
// This is research scaffolding, not part of the process model. It is armed
// entirely by DELTA_* options and does nothing when they are unset, which is
// every normal run. It lives behind these four calls so that the process model
// in kern/proc.cpp reads as the decisions it makes rather than as the probes
// that were needed to find them -- the same treatment gpu/cmd_trace gets.
//
// Everything here may read and write guest memory, plant int3, and depend on
// offsets in one specific build of one specific game. Nothing in kern may
// depend on it in return.

#include "base/arch.h"
#include <base/strings/string_ref.h>
#include <cstdint>

namespace krnl {
class proc;
class smodule;

namespace probe {

// The main module is loaded and its imports are resolved, but the guest has not
// run yet. Arms the generic probes and the PS5 bring-up patches.
void onProcessCreated(proc &p, smodule &main, bool ps5);

// A shared module finished loading. `name` is the requested module name, which
// is what the per-title patches key off.
void onModuleLoaded(smodule &m, base::StringRef name);

// About to enter the guest entry point.
void onBeforeStart(proc &p);

// Called from smodule::resolveImports for each PLT import. Returns a guest
// wrapper around `realAddr` that traces the libSceFios2 whole-file APIs when
// DELTA_FIOS_TRACE is set and `nidName` (encoded "NID#lib#mod") matches;
// otherwise returns realAddr unchanged.
uintptr_t wrapImport(const char *nidName, uintptr_t realAddr);

}  // namespace probe
}  // namespace krnl
