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
#include <vector>

namespace krnl {
struct moduleInfo;
}

namespace cpu {

// Magic syscall number used on the FEX path to bridge the guest's dynamic-TLS
// resolver to the host HLE. The native path patches libkernel's __tls_get_addr
// to `jmp` a host function pointer directly, but a host (ARM) jump is invalid
// inside the x86 JIT; instead we patch it to a tiny `mov eax, <this>; syscall;
// ret` stub and dispatch it in HandleSyscall -> krnl::guest_tls_get_addr. Well
// above any real PS4 syscall number so it can't collide.
constexpr uint32_t kTlsGetAddrSyscall = 0x40000001u;

// Magic syscall range used on the FEX path to call native HLE functions from
// guest code (vprx PRX overrides). A guest import slot can't point straight at a
// host (ARM) function (the x86 JIT would try to decode ARM bytes), so we hand
// the slot a tiny guest trampoline that issues `mov eax,<base|idx>; syscall` and
// dispatch index `idx` to the registered host function in HandleSyscall. The top
// byte tags the range; the low 24 bits are the thunk index.
constexpr uint32_t kHostThunkSyscallBase = 0x42000000u;

class ICpuBackend {
public:
  virtual ~ICpuBackend() = default;

  // Called once per module after its PT_LOAD/PT_SCE_RELRO segments have been
  // copied into the guest address space and before page protections are
  // finalized. Native: no-op (the lifter runs inline in the loader). FEX:
  // register the module's executable range so the JIT knows what to decode.
  virtual void onImageMapped(krnl::moduleInfo &info) = 0;

  // Create a guest thread object (CPU state, stack, TLS) without running it.
  // MUST be called on the PARENT thread before the worker host thread is spawned.
  // FEX serializes thread creation against running threads, so creating on the
  // freshly-spawned worker (while other guest threads run in the JIT) races on
  // shared JIT/context state and corrupts it. Returns an opaque handle.
  virtual void *createGuestThread(uintptr_t entry, void *arg, uint64_t fsbase) = 0;

  // Run a previously-created guest thread to completion on the CURRENT host
  // thread (does the FEX per-thread registration first), then destroys it.
  virtual void runGuestThread(void *handle) = 0;

  // Synchronously call a guest function `fn(a0..a3)` (PS4 SysV ABI) from host
  // code and return when it returns. This is how the real kernel runs a loaded
  // module's DT_INIT (module_start) so the module can self-register (e.g. the
  // real libSceVideoOut registering its display driver). Native: a direct
  // function-pointer call. FEX: runs the function on a fresh JIT thread whose
  // return address unwinds out via exitGuestThread.
  // Returns the function's eax. Must be called from a context where guest memory
  // and the module's dependencies are already mapped + relocated.
  virtual uint64_t runGuestFunction(uintptr_t fn, uint64_t a0, uint64_t a1,
                                    uint64_t a2, uint64_t a3 = 0) = 0;

  // Convenience for the main thread: create + run on this thread.
  void enterGuest(uintptr_t entry, void *arg, uint64_t fsbase) {
    runGuestThread(createGuestThread(entry, arg, fsbase));
  }
};

// Terminate the guest thread currently running on this host thread immediately,
// without returning to guest code. The PS4 thr_exit syscall must never return
// to its caller (FreeBSD destroys the thread in-kernel); libkernel's pthread
// trampoline treats a return as a fatal error. On FEX this unwinds out of the
// JIT (longjmp) back to runGuestThread so the thread is destroyed; on
// the native backend it's a no-op (the guest entry returns naturally). Only call
// from a guest thread context (i.e. from inside a syscall handler).
void exitGuestThread();

// Return a guest-callable address that invokes the host (PS4ABI / sysv) function
// `hostFn` with the guest's integer arguments (up to 14; args 7+ read from the
// guest stack). On the native x86 backend this is just `hostFn`; the guest
// calls it directly. On FEX it returns a small guest x86 trampoline that bounces
// through the kHostThunkSyscallBase magic syscall. Used by the loader to bind
// vprx HLE exports (e.g. libSceVideoOut) into guest import slots. Thread-safe.
uintptr_t makeHostThunk(void *hostFn, const char *name = nullptr);

// If `addr` lands in the host-thunk pool, the "libname!NID" of the HLE export
// whose trampoline lives there ("" if it was bound without a name), else null.
// A guest fault inside the pool is a call through an import slot we bound but
// cannot service, and this is what names it. FEX backend only; the native
// backend has no trampolines and returns null.
const char *hostThunkNameForAddr(uintptr_t addr, uint32_t *idxOut = nullptr);

// Wrap an already-resolved guest function `realTarget` with a return-capturing
// guest trampoline: it calls realTarget, then invokes native
// `loggerFn(hookId, a0,a1,a2,a3, ret)` (a0..a3 = original rdi/rsi/rdx/rcx, ret =
// realTarget's rax), and returns realTarget's value. Install by overwriting the
// GOT slot that held realTarget. ARM-safe guest-function trace/hook facility
// (int3 hooks are x86-host-only). On the native x86 backend this is a no-op that
// returns realTarget unchanged (int3 works there). Returns guest addr, 0 on fail.
uintptr_t makeGuestReturnHook(void *realTarget, uint32_t hookId, void *loggerFn,
                              const char *name = nullptr);

// Build a callable copy (trampoline) of an internal guest function whose first
// `prologueLen` bytes will be overwritten by an entry detour. Returns a
// guest-executable address that runs the original from the top and continues
// into its body; pass it as `realTarget` to makeGuestReturnHook to wrap an
// eboot-internal (non-import) function. Native backend: returns the original
// entry unchanged. See makeGuestTrampoline in fex_backend.cpp for constraints.
uintptr_t makeGuestTrampoline(const void *fnBytes, uint32_t prologueLen,
                              const void *continueAt);

// Called once at process start, before any large allocation or guest mapping.
// FEX: reserves/segregates the address space so guest memory can't collide with
// FEXCore's own JIT/internal allocations (Setup48BitAllocator + SetupHooks, in
// FEX's order). Native: no-op. Must run as early as possible.
void earlyInit();

// The process-wide backend, selected at build time by the host arch.
ICpuBackend &backend();

// Guest RIP of the guest thread currently running on this host thread, or 0.
// On FEX this is the live CPUState.rip; on native the host RIP is the guest RIP
// already (the crash handler reads it from the signal context), so this is 0.
uint64_t currentGuestRip();

// Guest fs-segment base (TLS pointer) of the guest thread running on this host
// thread, or 0. Used by hook loggers to read guest thread-local state.
uint64_t currentGuestFsBase();

// Every live guest thread's fs base. Lets a host thread survey guest TLS
// across the whole process (e.g. which BPE JobSystem worker ordinals exist)
// without instrumenting a hot guest path. FEX only; empty on native.
void guestThreadFsBases(std::vector<uint64_t> &out);

// The 16 guest GPRs (FEXCore::X86State::REG_* order) of the guest thread on this
// host thread, or nullptr. FEX only; lets the crash handler dump guest regs and
// walk the guest rbp chain. Native reads regs from the host signal context, so 0.
const uint64_t *currentGuestGregs();

// If this host thread faulted while inside a guest syscall handler, the syscall
// number; otherwise -1. Lets the crash handler name the culprit HLE call.
int faultingSyscall();

// Dump this host thread's most-recent guest-boundary events (syscalls + HLE
// thunk calls) oldest-first to the given stream. The crash handler uses it
// because failing init logic often makes its last HLE/Gnm/kernel call right
// before a downstream null-deref. No-op on the native backend.
void dumpThreadTrace(void *fileStar);

// Reconstruct the precise guest RIP from a host PC captured in a signal (FEX
// only). CPUState.rip is only block-accurate while the JIT is running and
// multiblock compilation hides the real fault site; this asks FEX to map the
// host JIT PC back to the exact guest instruction. Returns 0 if the host PC
// isn't in a JIT code buffer (or on the native backend).
uint64_t reconstructGuestRip(uint64_t hostPC);

// Give the active backend a chance to handle a host signal raised inside JIT'd
// code before it's treated as a fatal guest fault (FEX only). Currently handles
// SIGBUS from unaligned atomic accesses (which the ARM JIT raises and FEX
// backpatches). Returns true if handled, so the signal handler should return
// to resume the (possibly PC-adjusted) context. Native: always false.
bool tryHandleJitSignal(int sig, void *info, void *ucontext);

} // namespace cpu
