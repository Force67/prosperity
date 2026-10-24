#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstddef>
#include <cstdint>  // uintptr_t
#include "base/arch.h"
#include "guest_abi.h"
#include "options/options.h"

namespace kern {

// Syscall dispatch, shared by the Orbis (lv2/ps4) and Prospero (lv2/ps5)
// tables. A table only maps an id to a C handler; how that handler is entered,
// and how its return reaches the guest, lives here.

// The trampoline for `sid` under the active process's platform. Both CPU
// backends enter here.
uintptr_t Lv2Lookup(u32 sid);

// The individual tables, for callers that already know the platform.
uintptr_t Lv2Get(u32 sid);     // Orbis
uintptr_t Lv2GetPs5(u32 sid);  // Prospero

// Wraps `handler` in the trampoline that applies the BSD carry/errno return
// convention. Cached per handler, or per id while tracing or counting so each
// id reports under its own number.
uintptr_t Lv2Trampoline(const void* handler, u32 sid);

// The guest's registers at a context syscall (getcontext and friends), in the
// order the context trampoline pushes them. `rsp` arrives pointing at `rip`
// (the return address of the lifted call); the trampoline resumes the guest
// from whatever the frame and the XSAVE image hold once the handler returns,
// so a handler that rewrites them switches context.
struct GuestFrame {
  u64 r15, r14, r13, r12, r11, r10, r9, r8, rdi, rsi, rbp, rdx, rcx, rbx, rax;
  u64 rflags, rsp, rip;
};
// x87 + SSE + AVX, the layout the kernel's mcontext embeds.
inline constexpr size_t kGuestXsaveSize = 0x340;

struct GuestContextImage {
  GuestFrame frame;
  alignas(64) u8 xsave[kGuestXsaveSize];
};

using ContextHandler = void(PS4ABI*)(u32 sid, GuestFrame* frame, u8* xsave);

// The trampoline that hands `handler` the full register state, or 0 on a
// backend that captures that state itself.
uintptr_t Lv2ContextTrampoline(ContextHandler handler, u32 sid);

// A guest thread entry that starts from a heap GuestContextImage passed as its
// argument (and frees it).
uintptr_t Lv2ContextEntry();

// The handler that needs the full register state for `sid` on the active
// process's platform, or null for an ordinary syscall. Backends without the
// context trampoline call it around their own capture of the state.
ContextHandler Lv2ContextHandler(u32 sid);

// The private id the FEX context-entry stub issues with the image in rdi.
inline constexpr u32 kContextEntrySyscall = 0x40000002u;

// A syscall a table names but does not implement.
int PS4ABI Lv2StubSyscall();

// An id no table row covers at all.
int PS4ABI Lv2UnmappedSyscall();

const char* SyscallGetname(u32 idx);  // name_table.cc
void DumpSyscallHist();

// Classifies a raw handler return as an errno or a result; see the definition.
extern "C" u32 SyscallErrno(u64 raw);

// DELTA_SCHIST per-id call counter. The native trampoline increments it, so a
// backend that emits no trampoline has to do so itself.
extern base::Option<bool> g_sc_hist;

struct ModuleInfo;

// The loaded module whose image contains `addr`, or null.
ModuleInfo* CalledIn(void* addr);

}  // namespace kern

extern "C" u64 g_sys_hist[1024];
