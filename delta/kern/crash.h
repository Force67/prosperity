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

// Resolve a host/guest address to "<module>+offset (.text/.data)" via the loaded
// module table (or "0x.. (??)" if unmapped). Exposed for targeted mmap/loop tracing.
void symbolize(uintptr_t addr, char *out, size_t n);

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

// DELTA_PS5_GLYPHGUARD: recover a null-object deref in the first-frame UI/text
// renderer (fonts unbound). On a SIGSEGV at `addr`, zero the destination register
// `reg` and advance rip past the `insnLen`-byte faulting instruction, so the code
// continues with a benign value (unbound-font text renders empty) instead of
// crashing before the game queues real draws. Diagnostic; the real fix binds the
// font before rendererFrame.
enum class GuardReg { rax, rsi };
void setNullGuard(uintptr_t addr, GuardReg reg, int insnLen);

// DELTA_PS5_GLYPHGUARD: skip a guest CALL that blocks (a render-context getter
// vtable dispatch that never returns in our env). Plants an int3 at `addr`; when
// hit, sets rax=`raxVal` and advances rip past the `insnLen`-byte call, so the
// caller continues with the injected return value instead of blocking forever.
void setCallSkip(uintptr_t addr, long raxVal, int insnLen);

// DELTA_PS5_DCBWATCH call-order trace: int3 at an engine entry (push rbp); on
// each hit log "[order] <label> <- caller" with a timestamp, emulate the push,
// and resume. Used to dump the actual call order of the renderer/DCB-creation
// path (which functions run, in what order, before the null-DCB crash).
void setOrderTrace(uintptr_t addr, const char *label);

// Return-value trace: int3 at a `mov ebx,eax` (89 c3) site right after a call;
// logs eax, emulates the mov (ebx=eax), and resumes. Pins which sub-call in a
// run-once init returns the non-zero error that blocks the graphics bring-up.
void setRetTrace(uintptr_t addr, const char *label);

// Give the calling thread a dedicated signal-handler stack (SA_ONSTACK). The
// fatal handler then runs even when the guest's own RSP is corrupt or blown
// (a stack-overflow fault would otherwise be undeliverable -> silent core).
// Must be called on every guest thread before it enters the JIT.
void installSigAltStack();
}  // namespace krnl
