/*
 * PS4Delta : PS4 emulation and research project
 *
 * CPU execution backend abstraction.
 *
 * The emulator runs guest PS4 x86-64 code one of two ways depending on the host:
 *
 *  - NativeBackend (x86-64 host, DELTA_BACKEND_NATIVE): guest code runs directly
 *    on the host CPU. The lifter (runtime/code_lift) rewrites guest `syscall`
 *    and `fs:`-relative TLS reads in-place so they trap to host HLE; the guest
 *    entry is just called as a host function and guest threads are host threads.
 *
 *  - FexBackend (aarch64 host, DELTA_BACKEND_FEX): guest code runs inside an
 *    embedded FEXCore JIT. Nothing is rewritten; the JIT decodes `syscall`
 *    itself and hands it to us via a callback (dispatched to lv2), and emulates
 *    fs/gs from the guest CPUState. TLS is the guest fs base in CPUState; each
 *    guest thread is a FEXCore thread pinned to one host thread.
 *
 * Everything else (loader, lv2 HLE, devices) is host-native on both paths.
 */
#pragma once

#include <cstdint>
#include "base/arch.h"
#include <vector>

namespace krnl {
struct moduleInfo;
}

namespace cpu {

// Magic syscall on the FEX path bridging the guest dynamic-TLS resolver to host HLE
// (the native path patches __tls_get_addr directly; a host jump is invalid inside
// the x86 JIT, so FEX patches a `mov eax,<this>; syscall; ret` stub dispatched in
// HandleSyscall -> krnl::guest_tls_get_addr). Above any real PS4 syscall number.
constexpr u32 kTlsGetAddrSyscall = 0x40000001u;

// Magic syscall range for guest->native HLE calls (vprx PRX overrides): an import
// slot can't point at a host ARM function, so it gets a guest trampoline issuing
// `mov eax,<base|idx>; syscall`, dispatched by index in HandleSyscall. Top byte
// tags the range, low 24 bits are the thunk index.
constexpr u32 kHostThunkSyscallBase = 0x42000000u;

class ICpuBackend {
public:
  virtual ~ICpuBackend() = default;

  // Once per module, after segments are copied and before protections finalize.
  // Native: no-op. FEX: register the executable range with the JIT.
  virtual void onImageMapped(krnl::moduleInfo &info) = 0;

  // Create a guest thread object without running it. MUST be called on the PARENT
  // thread: FEX serializes creation against running threads, and creating on the
  // fresh worker races on shared JIT/context state. Returns an opaque handle.
  virtual void *createGuestThread(uintptr_t entry, void *arg, u64 fsbase) = 0;

  // Run a previously-created guest thread to completion on the CURRENT host
  // thread (does the FEX per-thread registration first), then destroys it.
  virtual void runGuestThread(void *handle) = 0;

  // Synchronously call guest `fn(a0..a3)` (SysV) and return its eax: how the kernel
  // runs a module's DT_INIT so it can self-register (libSceVideoOut's display
  // driver). Native: direct call. FEX: on a fresh JIT thread unwinding via
  // exitGuestThread. Guest memory + deps must already be mapped and relocated.
  virtual u64 runGuestFunction(uintptr_t fn, u64 a0, u64 a1,
                                    u64 a2, u64 a3 = 0) = 0;

  // Convenience for the main thread: create + run on this thread.
  void enterGuest(uintptr_t entry, void *arg, u64 fsbase) {
    runGuestThread(createGuestThread(entry, arg, fsbase));
  }
};

// Terminate the guest thread on this host thread without returning to guest code.
// thr_exit must never return (libkernel's trampoline treats that as fatal); FEX
// longjmps out of the JIT to runGuestThread, native is a no-op (entry returns
// naturally). Only from a guest thread context.
void exitGuestThread();

// Guest-callable address invoking host `hostFn` with the guest's integer args (up to
// 14; args 7+ from the guest stack). Native: just `hostFn`. FEX: a small x86
// trampoline through the magic syscall; used to bind vprx HLE exports into import
// slots. Thread-safe.
uintptr_t makeHostThunk(void *hostFn, const char *name = nullptr);

// If `addr` is in the host-thunk pool: the "libname!NID" of the export whose
// trampoline lives there ("" if unnamed), else null; names a fault through a
// bound-but-unserviced import slot. FEX only.
const char *hostThunkNameForAddr(uintptr_t addr, u32 *idxOut = nullptr);

// Wrap resolved guest `realTarget` with a return-capturing trampoline: calls it,
// then loggerFn(hookId, a0..a3, ret), and returns its value. Install by overwriting
// the GOT slot. ARM-safe hook facility (int3 is x86-host-only); native no-op
// returning realTarget. Guest addr, 0 on failure.
uintptr_t makeGuestReturnHook(void *realTarget, u32 hookId, void *loggerFn,
                              const char *name = nullptr);

// Callable copy of an internal guest function whose prologue an entry detour will
// overwrite (realTarget for makeGuestReturnHook; native returns the entry unchanged;
// constraints in fex_backend.cpp). Also: wrap a guest function so a native lock is
// held across the call, serialising a guest critical section from the host; a failed
// try_lock deterministically names a second thread inside.
uintptr_t makeGuestLockWrapper(void *realTarget, void *lockFn, void *unlockFn,
                               const char *name);

uintptr_t makeGuestTrampoline(const void *fnBytes, u32 prologueLen,
                              const void *continueAt);

// Once at process start, before any large allocation or guest mapping: FEX reserves
// address space so guest memory can't collide with FEXCore's JIT internals
// (Setup48BitAllocator + SetupHooks, FEX's order). Native: no-op.
void earlyInit();

// The process-wide backend, selected at build time by the host arch.
ICpuBackend &backend();

// Guest RIP of the guest thread currently running on this host thread, or 0.
// On FEX this is the live CPUState.rip; on native the host RIP is the guest RIP
// already (the crash handler reads it from the signal context), so this is 0.
u64 currentGuestRip();

// Guest fs-segment base (TLS pointer) of the guest thread running on this host
// thread, or 0. Used by hook loggers to read guest thread-local state.
u64 currentGuestFsBase();

// Every live guest thread's fs base. Lets a host thread survey guest TLS
// across the whole process (e.g. which BPE JobSystem worker ordinals exist)
// without instrumenting a hot guest path. FEX only; empty on native.
void guestThreadFsBases(std::vector<u64> &out);

// The 16 guest GPRs (FEXCore::X86State::REG_* order) of the guest thread on this
// host thread, or nullptr. FEX only; lets the crash handler dump guest regs and
// walk the guest rbp chain. Native reads regs from the host signal context, so 0.
const u64 *currentGuestGregs();

// The 16 guest GPRs from a SIGNAL CONTEXT inside JIT'd code, FEXCore REG_* order.
// Exact at the faulting instruction (FEX pins guest GPRs to fixed host registers),
// unlike currentGuestGregs(), accurate only at block boundaries. False when the
// host PC is not in a JIT buffer or the backend has no such mapping.
bool guestGregsFromSignal(const void *ucontext, u64 out[16]);

// If this host thread faulted while inside a guest syscall handler, the syscall
// number; otherwise -1. Lets the crash handler name the culprit HLE call.
int faultingSyscall();

// Dump this thread's recent guest-boundary events (syscalls + HLE thunks), oldest
// first; the crash handler uses it because failing init makes its last HLE call
// right before a downstream null-deref. No-op on native.
void dumpThreadTrace(void *fileStar);

// Precise guest RIP from a host PC captured in a signal (FEX only). CPUState.rip is
// block-accurate while the JIT runs and multiblock hides the fault site; this maps
// the host JIT PC back to the exact instruction. 0 if not in a JIT buffer.
u64 reconstructGuestRip(u64 hostPC);

// Let the backend handle a host signal raised inside JIT'd code before it becomes a
// fatal guest fault (FEX only; currently SIGBUS from unaligned atomics, which the ARM
// JIT raises and FEX backpatches). True = handled, resume the adjusted context.
bool tryHandleJitSignal(int sig, void *info, void *ucontext);

} // namespace cpu
