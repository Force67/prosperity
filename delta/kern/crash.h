#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

namespace krnl {
// Install signal handlers that symbolize a guest fault (rip + frame-pointer
// backtrace) to <module>+offset using the loaded module table, then exit.
void installCrashHandler();

// Enable the DELTA_ALLOC_TRACE allocator-entry tracer: addr's first byte must be
// `push rbp` (0x55); the caller plants an int3 there and this records addr so the
// fatal handler logs each allocation (size in rsi) >= minSize and resumes.
void setAllocTrace(uintptr_t addr, uint64_t minSize);

// DELTA_HEAP_PROF: int3 at a guest allocator entry (push rbp) whose size arg is
// in rdi (operator new / malloc). Each hit aggregates bytes+count keyed by the
// guest caller ([rsp]); SIGUSR1 dumps the top sites. Finds the heap's dominant
// consumer/leaker without a per-call log flood.
void setHeapProf(uintptr_t addr);

// DELTA_CNT_TRACE: like setAllocTrace but logs the archive entry-count [rdi+0x30]
// and the inline name [rdi+0x5c] at the hooked entry (push rbp -> int3).
void setCntTrace(uintptr_t addr);

// DELTA_FATAL_TRACE: int3 at a printf-style fatal handler entry; logs rdi (the
// format string) + caller + varargs so we learn why a worker thread bailed.
void setFatalTrace(uintptr_t addr);
void setHdrTrace(uintptr_t addr);

// DELTA_RDOFF_FIX: hook the file-read-request setter to force a manifest read's
// offset to 0; markManifestFd flags which fds are .manifest.bin (set in sys_open).
void setRdoffFix(uintptr_t addr);
void setSkipFn(uintptr_t addr);
void markManifestFd(uint32_t fd, bool v);

// Give the calling thread a dedicated signal-handler stack (SA_ONSTACK). The
// fatal handler then runs even when the guest's own RSP is corrupt or blown
// (a stack-overflow fault would otherwise be undeliverable -> silent core).
// Must be called on every guest thread before it enters the JIT.
void installSigAltStack();
}  // namespace krnl
