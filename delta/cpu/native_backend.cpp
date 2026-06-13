/*
 * PS4Delta : PS4 emulation and research project
 *
 * NativeBackend (x86-64 host). Guest code runs directly on the host CPU; the
 * lifter has already rewritten syscalls/fs-TLS in-place, so "entering the guest"
 * is just a host function call and the guest fs base is a host thread_local that
 * the lifted fs stub reads back via krnl_current_fsbase().
 */
#if defined(DELTA_BACKEND_NATIVE)

#include <base.h>
#include <csetjmp>
#include "cpu_backend.h"
#include "kern/crash.h"
#include "kern/proc.h"

namespace krnl {

// Per-thread guest fs base (TLS). The lifter rewrites guest `fs:[disp]` reads to
// call krnl_current_fsbase() so each host thread running guest code reads its
// own thread's TLS. initial-exec model => the access is a single
// `mov rax, fs:[off]; ret` that clobbers only rax, which the lifter stub relies
// on (it preserves only rax around the call).
__attribute__((tls_model("initial-exec"))) static thread_local uint64_t t_fsbase = 0;
extern "C" uint64_t krnl_current_fsbase() { return t_fsbase; }
void setThreadFsBase(uint64_t v) { t_fsbase = v; }

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
    uint64_t fsbase;
  };

  void *createGuestThread(uintptr_t entry, void *arg, uint64_t fsbase) override {
    return new NativeThread{entry, arg, fsbase};
  }

  void runGuestThread(void *handle) override {
    auto *t = static_cast<NativeThread *>(handle);
    // Give this guest thread a signal alt-stack so the fatal handler still runs
    // (and dumps the guest RIP) when the guest blows or corrupts its own RSP --
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

  uint64_t runGuestFunction(uintptr_t fn, uint64_t a0, uint64_t a1,
                            uint64_t a2) override {
    // Guest code runs natively on x86-64: a direct function-pointer call.
    return reinterpret_cast<uint64_t(PS4ABI *)(uint64_t, uint64_t, uint64_t)>(fn)(
        a0, a1, a2);
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

void earlyInit() {} // native: nothing to segregate

ICpuBackend &backend() {
  static NativeBackend instance;
  return instance;
}

uint64_t currentGuestRip() { return 0; }
const uint64_t *currentGuestGregs() { return nullptr; }
int faultingSyscall() { return -1; }
uint64_t reconstructGuestRip(uint64_t) { return 0; }
bool tryHandleJitSignal(int, void *, void *) { return false; }

} // namespace cpu

#endif // DELTA_BACKEND_NATIVE
