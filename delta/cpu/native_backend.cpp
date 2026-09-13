/*
 * PS4Delta : PS4 emulation and research project
 *
 * NativeBackend (x86-64 host). Guest code runs directly on the host CPU; the
 * lifter has already rewritten syscalls/fs-TLS in-place, so "entering the guest"
 * is just a host function call and the lifted FS stubs read the guest FS base
 * from host thread-local storage.
 */
#if defined(DELTA_BACKEND_NATIVE)

#include <base.h>
#include "base/arch.h"
#include <csetjmp>
#include "cpu_backend.h"
#include "kern/crash.h"
#include "kern/proc.h"

namespace krnl {

// Per-thread guest fs base and spill slot used by stackless FS lift stubs.
__attribute__((tls_model("initial-exec"))) static thread_local u64 t_fsbase = 0;
__attribute__((tls_model("initial-exec"))) static thread_local u64 t_fs_scratch = 0;

static i32 hostTlsOffset(const void *address) {
  uintptr_t thread_pointer;
  asm("mov %%fs:0, %0" : "=r"(thread_pointer));
  return static_cast<i32>(reinterpret_cast<uintptr_t>(address) -
                              thread_pointer);
}

i32 hostGuestFsOffset() { return hostTlsOffset(&t_fsbase); }
u64 threadFsBase() { return t_fsbase; }
i32 hostFsScratchOffset() { return hostTlsOffset(&t_fs_scratch); }
void setThreadFsBase(u64 v) { t_fsbase = v; }

} // namespace krnl

namespace cpu {

// Per-thread unwind target for thr_exit. The guest pthread trampoline issues
// thr_exit as its last act and treats a return as fatal ("thr_exit() returned").
// On native the trampoline is just a host call chain (runGuestThread -> guest
// frames -> the syscall handler), and those guest frames carry no C++ unwind
// info, so we can't throw past them. longjmp restores SP/IP directly, so it
// unwinds straight back to runGuestThread without touching the guest frames.
static thread_local std::jmp_buf *t_exitJmp = nullptr;

class NativeBackend final : public ICpuBackend {
public:
  void onImageMapped(krnl::moduleInfo &) override {
    // Nothing to do: the loader runs the lifter inline (runtime/code_lift),
    // rewriting syscall/int/fs in the executable segment in place.
  }

  // Native execution shares the host CPU, so there's no guest CPU state to
  // build ahead of time: just capture the entry parameters.
  struct NativeThread {
    uintptr_t entry;
    void *arg;
    u64 fsbase;
  };

  void *createGuestThread(uintptr_t entry, void *arg, u64 fsbase) override {
    return new NativeThread{entry, arg, fsbase};
  }

  void runGuestThread(void *handle) override {
    auto *t = static_cast<NativeThread *>(handle);
    // Give this guest thread a signal alt-stack so the fatal handler still runs
    // (and dumps the guest RIP) when the guest blows or corrupts its own RSP;
    // otherwise the kernel can't deliver SIGSEGV and silently core-dumps.
    krnl::installSigAltStack();
    krnl::setThreadFsBase(t->fsbase);
    auto entry = t->entry;
    auto arg = t->arg;
    delete t;
    // thr_exit longjmps back here instead of returning into the guest. The
    // guest entry may also just return naturally (setjmp returns 0 first time).
    std::jmp_buf jb;
    t_exitJmp = &jb;
    if (setjmp(jb) == 0)
      reinterpret_cast<void(PS4ABI *)(void *)>(entry)(arg);
    t_exitJmp = nullptr;
  }

  u64 runGuestFunction(uintptr_t fn, u64 a0, u64 a1,
                            u64 a2, u64 a3) override {
    // Guest code runs natively on x86-64: a direct function-pointer call.
    return reinterpret_cast<u64(PS4ABI *)(u64, u64, u64,
                                               u64)>(fn)(a0, a1, a2, a3);
  }
};

// Native: unwind out of the guest call chain back to runGuestThread so the
// host thread ends, instead of returning into the guest pthread trampoline.
void exitGuestThread() {
  if (t_exitJmp)
    std::longjmp(*t_exitJmp, 1);
}

// Native host is x86-64: the guest can call the host function directly.
uintptr_t makeHostThunk(void *hostFn, const char * /*name*/) { return reinterpret_cast<uintptr_t>(hostFn); }
// No trampoline pool on native: an HLE import IS the host function pointer.
const char *hostThunkNameForAddr(uintptr_t, u32 *) { return nullptr; }

// Native x86 host: int3 return hooks work directly, so guest-fn return hooking
// isn't needed; hand back the real target unchanged (no wrap).
uintptr_t makeGuestReturnHook(void *realTarget, u32 /*hookId*/,
                              void * /*loggerFn*/, const char * /*name*/) {
  return reinterpret_cast<uintptr_t>(realTarget);
}

uintptr_t makeGuestLockWrapper(void *, void *, void *, const char *) { return 0; }

uintptr_t makeGuestTrampoline(const void *fnBytes, u32 /*prologueLen*/,
                              const void * /*continueAt*/) {
  return reinterpret_cast<uintptr_t>(const_cast<void *>(fnBytes));
}

void earlyInit() {} // native: nothing to segregate

ICpuBackend &backend() {
  static NativeBackend instance;
  return instance;
}

u64 currentGuestRip() { return 0; }
// The guest fs base lives in host TLS on this backend (the lifter's fs stubs
// read it from there), so it is available to host code without a segment read.
u64 currentGuestFsBase() { return krnl::threadFsBase(); }
void guestThreadFsBases(std::vector<u64> & /*out*/) {} // FEX only
const u64 *currentGuestGregs() { return nullptr; }

bool guestGregsFromSignal(const void *, u64[16]) { return false; }
int faultingSyscall() { return -1; }
u64 reconstructGuestRip(u64) { return 0; }
bool tryHandleJitSignal(int, void *, void *) { return false; }

} // namespace cpu

#endif // DELTA_BACKEND_NATIVE
