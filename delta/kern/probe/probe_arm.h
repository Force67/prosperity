/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Arming the guest-code trap probes, and the seam the crash handler offers them
// every signal through (see probe_trap.cpp). Formerly in crash.h, whose other 23
// functions have nothing to do with crashing. Inert until a DELTA_* option arms it.

#include "base/arch.h"
#include <csignal>
#include <cstddef>
#include <cstdint>

namespace krnl::probe {

// Offered every signal before the crash reporter looks at it. True means a
// probe owned this trap and the guest must be resumed rather than dumped.
bool onSignal(int sig, siginfo_t *si, void *ucv);

// The reporter is about to print a fatal dump; flush what a probe has
// accumulated that would otherwise die with the process.
void onFatal();

// Points `pa` at the SIGUSR1 handler that dumps a thread's guest backtrace.
void installThreadProbeHandler(struct sigaction &pa);

// Enable the DELTA_ALLOC_TRACE allocator-entry tracer: addr's first byte must be
// `push rbp` (0x55); the caller plants an int3 there and this records addr so the
// fatal handler logs each allocation (size in rsi) >= minSize and resumes.
void setAllocTrace(uintptr_t addr, u64 minSize);

// DELTA_HEAP_PROF: int3 at a `push rbp` allocator entry (size in rdi); aggregates
// bytes+count by guest caller, SIGUSR1 dumps the top sites. countOnly for an entry
// whose first arg is a pointer (a deallocator): ranked by call count instead.
void setHeapProf(uintptr_t addr, bool countOnly = false);

// DELTA_HEAP_PROF_SCOPE=<tls-slot-global>:<depth-offset>: also record per site
// whether a scoped allocator was live (scopes free wholesale, the process-heap
// fallback does not, so "no scope" is a leak a caller-keyed profile cannot show).
void setHeapProfScope(uintptr_t tlsSlotGlobal, u64 depthOffset);

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
void markManifestFd(u32 fd, bool v);

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

// Return-value trace: int3 at a `mov ebx,eax` (89 c3) or `test eax,eax` (85 c0)
// site right after a call; logs eax, emulates the instruction, and resumes. Pins
// which sub-call in a run-once init returns the non-zero error that blocks the
// bring-up. DELTA_RETTRACE="hexoff:label,..." arms it by eboot offset.
void setRetTrace(uintptr_t addr, const char *label, bool isTest = false);

// DELTA_FNWATCH="off:label,...": int3 hit-COUNTER at function entries, totals
// printed every 2s, safe at any call frequency (unlike setOrderTrace's per-hit log).
void setFnWatch(uintptr_t addr, const char *label);
void startFnWatchPrinter();

// DELTA_GUEST_POPCNT=<hex addr>:<hex bytes>[:<ms>]: report a guest bitmap's
// population count, and its first/last set bit, on an interval (see crash.cpp).
// Tells a map that drains from one that was never filled, which a single dump
// at the crash cannot.
void startPopcntPrinter(uintptr_t addr, size_t bytes, unsigned everyMs);

// DELTA_GUEST_WPROT=<addr>:<bytes>[:<ms>]: write-protect a guest range once mapped
// and report the instruction behind every write (see crash.cpp). DELTA_GUEST_RPROT
// traps reads too, naming the consumer rather than the producer.
void startWriteWatch(uintptr_t addr, size_t bytes, unsigned everyMs,
                     bool trapReads = false, bool singleStep = false);

// DELTA_GUEST_WHIST=<hex addr>:<hex bytes>[:<ms>]: the same trap re-armed on an
// interval and reduced to counts, so a multi-GB pool can be surveyed for the
// length of a run: which 16 MiB slices the guest writes, and from where.
void startWriteHist(uintptr_t addr, size_t bytes, unsigned everyMs);

// DELTA_GUEST_SUMWATCH=<slot>:<off>:<stride>:<count>[:<ms>]: watch a set of u32
// counters behind a guest pointer slot (see crash.cpp).
void startSumWatchPrinter(uintptr_t slot, size_t off, size_t stride, int count,
                          unsigned everyMs);

// DELTA_POOLMAP=<hex addr>:<hex bytes>[:<ms>]: occupancy map of a large guest
// pool, by resident page rather than by read, so a multi-GB video pool can be
// surveyed without faulting it in (see crash.cpp).
void startPoolMap(uintptr_t addr, size_t bytes, unsigned everyMs);

// DELTA_POOLMAP=all[:<ms>]: the same survey over every guest mapping.
void startPoolCensus(unsigned everyMs);

// DELTA_MEMDUMP=<hex addr>:<hex bytes>:<ms>:<path>[,...]: write a guest range to
// a host file once the title has settled, so its contents can be identified
// offline (see crash.cpp).
void startMemDump(uintptr_t addr, size_t bytes, unsigned afterMs,
                  const char *path);

// DELTA_FNARGS="off+o1+o2...:label,...": int3 at an entry that logs rdi then walks
// the offset chain, printing every intermediate pointer and the final qword, i.e.
// which address the function polls/stores (a backtrace cannot: by the time a wedged
// thread is inside the emulator its callee-saved regs are gone).
void setFnArgs(uintptr_t addr, const char *label, const u64 *offsets,
               int noffsets);
}  // namespace krnl::probe
