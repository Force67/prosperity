#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <guest_abi.h>
#include "base/arch.h"
#include <cstdint> // uintptr_t
#include <options/options.h>

namespace krnl {

// Syscall dispatch, shared by the Orbis (lv2/ps4) and Prospero (lv2/ps5)
// tables. A table only maps an id to a C handler; how that handler is entered,
// and how its return reaches the guest, lives here.

// The trampoline for `sid` under the active process's platform. Both CPU
// backends enter here.
uintptr_t lv2_lookup(u32 sid);

// The individual tables, for callers that already know the platform.
uintptr_t lv2_get(u32 sid);     // Orbis
uintptr_t lv2_get_ps5(u32 sid); // Prospero

// Wraps `handler` in the trampoline that applies the BSD carry/errno return
// convention. Cached per handler, or per id while tracing or counting so each
// id reports under its own number.
uintptr_t lv2_trampoline(const void *handler, u32 sid);

// A syscall a table names but does not implement.
int PS4ABI lv2_stub_syscall();

// An id no table row covers at all.
int PS4ABI lv2_unmapped_syscall();

const char *syscall_getname(u32 idx); // name_table.cc
void dumpSyscallHist();

// Classifies a raw handler return as an errno or a result; see the definition.
extern "C" u32 krnl_syscall_errno(u64 raw);

// DELTA_SCHIST per-id call counter. The native trampoline increments it, so a
// backend that emits no trampoline has to do so itself.
extern base::Option<bool> g_scHist;
} // namespace krnl

extern "C" u64 g_sysHist[1024];
