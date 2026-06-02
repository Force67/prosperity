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
#include "cpu_backend.h"
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
    krnl::setThreadFsBase(t->fsbase);
    auto entry = t->entry;
    auto arg = t->arg;
    delete t;
    reinterpret_cast<void(PS4ABI *)(void *)>(entry)(arg);
  }
};

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
