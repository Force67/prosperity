#pragma once

#include <cstddef>
#include <cstdint>
#include "base/arch.h"

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

namespace krnl {
// Describes the compute-shader range covering a guest address, when the crash
// dump is walking an allocator structure the GPU also touches. Registered by
// the composition root: the answer lives in delta_gpu and kern may not name it.
using CsRangeDescriber = bool (*)(u64 addr, char *out, size_t outSize);
void setCsRangeDescriber(CsRangeDescriber fn);

// Install signal handlers that symbolize a guest fault (rip + frame-pointer
// backtrace) to <module>+offset using the loaded module table, then exit.
void installCrashHandler();

// Resolve a host/guest address to "<module>+offset (.text/.data)" via the loaded
// module table (or "0x.. (??)" if unmapped). Exposed for targeted mmap/loop tracing.
void symbolize(uintptr_t addr, char *out, size_t n);

// Print the calling thread's guest call sites by scanning its stack for return
// addresses inside a loaded module (see crash.cpp).
void guestStackTrace(const char *tag, int maxFrames);

// Walk a frame-pointer chain and symbolize each return address. Only works on
// code built with frame pointers, which optimised guest code is not -- prefer
// guestStackTrace for those; this is the cheap path when rbp is trustworthy.
void backtrace(uintptr_t rbp);

// Where the guest's stack pointer was when this thread last entered a syscall
// handler. The handler runs on its own stack (see krnl_kstack_top), so a scan
// has to start from the guest side of the switch. 0 = no switch happened.
void setGuestStackScanBase(uintptr_t sp);
uintptr_t guestStackScanBase();

// The same scan as guestStackTrace, but from a base captured earlier and for a
// thread other than the caller: a parked thread cannot report itself, so the
// wait probe records its scan base on the way in and walks it from outside.
void guestStackTraceFrom(uintptr_t base, const char *tag, int maxFrames,
                         long tid);

// DELTA_PS5_GLYPHGUARD: recover a null-object deref in the first-frame UI/text
// renderer (fonts unbound). On a SIGSEGV at `addr`, zero the destination register
// `reg` and advance rip past the `insnLen`-byte faulting instruction, so the code
// continues with a benign value (unbound-font text renders empty) instead of
// crashing before the game queues real draws. Diagnostic; the real fix binds the
// font before rendererFrame.
enum class GuardReg { rax, rsi };
void setNullGuard(uintptr_t addr, GuardReg reg, int insnLen);

// Give the calling thread a dedicated signal-handler stack (SA_ONSTACK). The
// fatal handler then runs even when the guest's own RSP is corrupt or blown
// (a stack-overflow fault would otherwise be undeliverable -> silent core).
// Must be called on every guest thread before it enters the JIT.
void installSigAltStack();
}  // namespace krnl
