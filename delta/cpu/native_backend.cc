/*
 * PS4Delta : PS4 emulation and research project
 *
 * NativeBackend (x86-64 host). Guest code runs directly on the host CPU; the
 * lifter has already rewritten syscalls/fs-TLS in-place, so "entering the
 * guest" is just a host function call and the lifted FS stubs read the guest FS
 * base from host thread-local storage.
 */

#include <csetjmp>
#include "base/arch.h"
#include "base/containers/vector.h"
#include "cpu/backend.h"
#include "guest_abi.h"
#include "kern/crash.h"

namespace cpu {

// Per-thread guest fs base and spill slot used by stackless FS lift stubs.
__attribute__((tls_model("initial-exec"))) static thread_local u64 t_fsbase = 0;
__attribute__((
    tls_model("initial-exec"))) static thread_local u64 t_fs_scratch = 0;

static i32 HostTlsOffset(const void* address) {
  uintptr_t thread_pointer;
  asm("mov %%fs:0, %0" : "=r"(thread_pointer));
  return static_cast<i32>(reinterpret_cast<uintptr_t>(address) -
                          thread_pointer);
}


i32 HostGuestFsOffset() {
  return HostTlsOffset(&t_fsbase);
}
u64 ThreadFsBase() {
  return t_fsbase;
}
i32 HostFsScratchOffset() {
  return HostTlsOffset(&t_fs_scratch);
}
void SetThreadFsBase(u64 v) {
  t_fsbase = v;
}

}  // namespace cpu

namespace cpu {

// Per-thread unwind target for thr_exit. The guest pthread trampoline issues
// thr_exit as its last act and treats a return as fatal ("thr_exit()
// returned"). On native the trampoline is just a host call chain
// (RunGuestThread -> guest frames -> the syscall handler), and those guest
// frames carry no C++ unwind info, so we can't throw past them. longjmp
// restores SP/IP directly, so it unwinds straight back to RunGuestThread
// without touching the guest frames.
static thread_local std::jmp_buf* t_exit_jmp = nullptr;

class NativeBackend final : public Backend {
 public:
  void OnImageMapped(krnl::moduleInfo&) override {
    // Nothing to do: the loader runs the lifter inline (runtime/code_lift),
    // rewriting syscall/int/fs in the executable segment in place.
  }

  // Native execution shares the host CPU, so there's no guest CPU state to
  // build ahead of time: just capture the entry parameters.
  struct NativeThread {
    uintptr_t entry;
    void* arg;
    u64 fsbase;
  };

  void* CreateGuestThread(uintptr_t entry, void* arg, u64 fsbase) override {
    return new NativeThread{entry, arg, fsbase};
  }

  void RunGuestThread(void* handle) override {
    auto* t = static_cast<NativeThread*>(handle);
    // Give this guest thread a signal alt-stack so the fatal handler still runs
    // (and dumps the guest RIP) when the guest blows or corrupts its own RSP;
    // otherwise the kernel can't deliver SIGSEGV and silently core-dumps.
    krnl::installSigAltStack();
    SetThreadFsBase(t->fsbase);
    auto entry = t->entry;
    auto arg = t->arg;
    delete t;
    // thr_exit longjmps back here instead of returning into the guest. The
    // guest entry may also just return naturally (setjmp returns 0 first time).
    std::jmp_buf jb;
    t_exit_jmp = &jb;
    if (setjmp(jb) == 0)
      reinterpret_cast<void(PS4ABI*)(void*)>(entry)(arg);
    t_exit_jmp = nullptr;
  }

  u64 RunGuestFunction(uintptr_t fn, u64 a0, u64 a1, u64 a2, u64 a3) override {
    // Guest code runs natively on x86-64: a direct function-pointer call.
    return reinterpret_cast<u64(PS4ABI*)(u64, u64, u64, u64)>(fn)(a0, a1, a2,
                                                                  a3);
  }
};

// Native: unwind out of the guest call chain back to RunGuestThread so the
// host thread ends, instead of returning into the guest pthread trampoline.
void ExitGuestThread() {
  if (t_exit_jmp)
    std::longjmp(*t_exit_jmp, 1);
}

// Native host is x86-64: the guest can call the host function directly.
uintptr_t MakeHostThunk(void* host_fn, const char* /*name*/) {
  return reinterpret_cast<uintptr_t>(host_fn);
}
// No trampoline pool on native: an HLE import IS the host function pointer.
const char* HostThunkNameForAddr(uintptr_t, u32*) {
  return nullptr;
}

// Native x86 host: int3 return hooks work directly, so guest-fn return hooking
// isn't needed; hand back the real target unchanged (no wrap).
uintptr_t MakeGuestReturnHook(void* real_target,
                              u32 /*hookId*/,
                              void* /*loggerFn*/,
                              const char* /*name*/) {
  return reinterpret_cast<uintptr_t>(real_target);
}

uintptr_t MakeGuestLockWrapper(void*, void*, void*, const char*) {
  return 0;
}

uintptr_t MakeGuestTrampoline(const void* fn_bytes,
                              u32 /*prologueLen*/,
                              const void* /*continueAt*/) {
  return reinterpret_cast<uintptr_t>(const_cast<void*>(fn_bytes));
}

void EarlyInit() {}  // native: nothing to segregate

Backend& GetBackend() {
  static NativeBackend instance;
  return instance;
}

u64 CurrentGuestRip() {
  return 0;
}
// The guest fs base lives in host TLS on this backend (the lifter's fs stubs
// read it from there), so it is available to host code without a segment read.
u64 CurrentGuestFsBase() {
  return ThreadFsBase();
}
void GuestThreadFsBases(base::Vector<u64>& /*out*/) {}  // FEX only
const u64* CurrentGuestGregs() {
  return nullptr;
}

bool GuestGregsFromSignal(const void*, u64[16]) {
  return false;
}
int FaultingSyscall() {
  return -1;
}
u64 ReconstructGuestRip(u64) {
  return 0;
}
bool TryHandleJitSignal(int, void*, void*) {
  return false;
}

}  // namespace cpu
