
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <unordered_map>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#if defined(DELTA_BACKEND_NATIVE)
#include <xbyak.h>
#endif

#include "dispatch.h"
#include "kern/crash.h"
#include "kern/module.h"
#include "kern/proc.h"

namespace {
DELTA_OPTION(bool, kScerrTrace, "DELTA_SCERR_TRACE", false);
} // namespace

namespace krnl {

// Read by the stub handlers and the trampoline alike, so it is a krnl symbol
// rather than a file-local one.
DELTA_OPTION(bool, g_scHist, "DELTA_SCHIST", false);

moduleInfo *called_in(void *addr) {
  uintptr_t addrsafe = (uintptr_t)addr;

  for (auto &mod : proc::getActive()->getModuleList()) {
    auto &info = mod->getInfo();

    if (addrsafe <= (uintptr_t)(info.base + info.codeSize) &&
        (addrsafe >= (uintptr_t)info.base)) {
      BASE_LOGI("nullhandler", "{:p} called in {}", addr, info.name.c_str());
      return &info;
    }
  }
  return nullptr;
}

int PS4ABI lv2_stub_syscall() {
#ifdef _MSC_VER
  void *ret = _ReturnAddress();
#else
  void *ret = __builtin_return_address(0);
#endif
  called_in(ret);

  static std::atomic<u64> nulls{0};
  const u64 n = ++nulls;
  if (n == 1 || (g_scHist && n % 400 == 0)) {
    BASE_LOGI("nullhandler", ">>>>>>>>>>>>> NULL HANDLER called {} times",
              (unsigned long long)n);
    if (g_scHist)
      dumpSyscallHist();
  }
  return 0;
}

int PS4ABI lv2_unmapped_syscall() {
#ifdef _MSC_VER
  void *ret = _ReturnAddress();
#else
  void *ret = __builtin_return_address(0);
#endif
  called_in(ret);

  BASE_LOGI("nullhandler", ">>>>>>>>>>>>> NULL HANDLER NULLTABLE CALLED BY {:p}", ret);
  return 0;
}

// BSD/PS4 syscall return convention: on failure the kernel returns the positive
// errno in rax with the carry flag SET; on success it clears carry and rax holds
// the result. Our C handlers use the Linux-style convention instead (a negative
// errno, or the result, in rax). Classify a raw handler return for both the
// native trampoline and the FEX syscall bridge.
//
// An `int` handler zero-extends its 32-bit result into rax, an `i64` handler
// fills all 64 bits, so a negative errno arrives as either 0x00000000_FFFFFFxx or
// 0xFFFFFFFF_FFFFFFxx. Guest pointers live at >= 64 GiB and cannot match this.
extern "C" u32 krnl_syscall_errno(u64 raw) {
  i32 lo = static_cast<i32>(static_cast<u32>(raw));
  u32 hi = static_cast<u32>(raw >> 32);
  if (lo < 0 && lo >= -static_cast<i32>(SysError::eLAST) &&
      (hi == 0 || hi == 0xFFFFFFFFu))
    return static_cast<u32>(-lo);
  return 0;
}

#if defined(DELTA_BACKEND_NATIVE)
// Per-thread stack for syscall handlers -- the emulator's equivalent of a
// kernel stack. The native backend runs guest code on the host thread directly,
// so a handler would otherwise execute on whatever stack the guest is using,
// and a title that runs jobs on FIBERS gives those a stack of its own choosing:
// SotC's are 16 KiB with the fiber's saved context sitting at the bottom, which
// a handler's host frames (a std::mutex wait, a printf, an allocation) walk
// straight through. The corruption showed up as a fiber resuming into the
// middle of glibc's free().
//
// Returns 0 when no switch is wanted, which is also how nesting is handled: a
// guest callback invoked from a handler makes its syscalls on the stack we
// already switched to, and it just grows further down.
extern "C" u64 krnl_kstack_top() {
  constexpr size_t kSize = 1u << 20;  // 1 MiB, lazily backed
  static thread_local u8 *base = nullptr;
  const auto here = reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
  if (base) {
    if (here > reinterpret_cast<uintptr_t>(base) &&
        here < reinterpret_cast<uintptr_t>(base) + kSize)
      return 0;  // already on it
  } else {
    void *p = ::mmap(nullptr, kSize, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED)
      return 0;
    base = static_cast<u8 *>(p);
  }
  // Where the guest's own stack was, so a handler-side stack scan (see
  // crash.cpp guestStackTrace) still finds the guest frames it is after.
  setGuestStackScanBase(here);
  return reinterpret_cast<u64>(base + kSize);
}

// DELTA_SCERR_TRACE: emitted (when set) on the trampoline's error path so we can
// see exactly which syscall returned which errno just before a guest abort. The
// errno is the BSD-style positive value the guest's libkernel turns into an SCE
// error (0x80020000 | errno), e.g. errno 1 -> 0x80020001.
static void PS4ABI trace_syscall_err(u32 sid, u32 err) {
  const char *name = syscall_getname(sid);
  BASE_LOGI("syscallerr", "{} {} -> errno {} (SCE {:#010x})", sid,
            name ? name : "?", err, 0x80020000u | err);
}
#endif // DELTA_BACKEND_NATIVE

// Per-syscall-id call counter (DELTA_SCHIST). The only profiler available on this
// build: perf/strace/`/proc/PID/mem` are all yama-blocked, so to find what a
// wedged/slow title hammers, count every syscall in the trampoline and dump the
// histogram from the SIGUSR1 probe (crash.cpp). Racy increments are fine here.
// Deliberately outside the DELTA_BACKEND_NATIVE guard: the stub handlers read
// g_scHist/dumpSyscallHist unconditionally, so the FEX build must link them too
// (only the trampoline that increments g_sysHist is native-only).
extern "C" u64 g_sysHist[1024] = {};

// The histogram was only ever printed by the crash reporter, so a clean run
// discarded it. "Which syscall is this title hammering" is the question it
// exists to answer, and most runs neither crash nor exit gracefully (the
// emulator is normally SIGKILLed), so it is also dumped from the unimplemented
// -syscall stub, which is where the question usually comes up.
void dumpSyscallHist() {
  BASE_LOGI("schist", "syscall call counts:");
  for (int i = 0; i < 1024; i++) {
    if (!g_sysHist[i])
      continue;
    const char *n = syscall_getname(i);
    BASE_LOGI("schist", "{:4} {:<24} {}", i, n ? n : "?",
              (unsigned long long)g_sysHist[i]);
  }
}

static const bool g_scHistDump = [] {
  if (!g_scHist)
    return false;
  std::atexit(&dumpSyscallHist);
  // The emulator is normally SIGKILLed, so also sample on a timer: what a title
  // is doing in STEADY STATE is a different question from what it did at boot,
  // and only a periodic dump answers it.
  std::thread([] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(20));
      dumpSyscallHist();
    }
  }).detach();
  return true;
}();

#if defined(DELTA_BACKEND_NATIVE)
// One-time trampoline per handler: call it with the guest's arg registers
// untouched, then set/clear the carry flag and normalise rax per the convention
// above. Replaces the bare `call handler` so the guest sees faithful errors.
static uintptr_t emit_bsd_trampoline(const void *handler, u32 sid,
                                     bool trace, bool count) {
  // The two handlers that never return -- they longjmp out of the guest call
  // chain (cpu::exitGuestThread) -- must stay on the guest stack: glibc's
  // longjmp check rejects a jump to a frame that is not on the current stack.
  // Neither needs the room anyway.
  const bool ownStack = sid != 1 /*exit*/ && sid != 431 /*thr_exit*/;
  struct bsdRet : Xbyak::CodeGenerator {
    bsdRet(uintptr_t handler, u32 sid, bool trace, bool count,
           bool ownStack) {
      if (count) {          // DELTA_SCHIST: ++g_sysHist[sid] (rax is caller-saved)
        mov(rax, reinterpret_cast<uintptr_t>(&g_sysHist[sid & 1023]));
        inc(qword[rax]);
      }
      // The lifted site enters with `call`, so rsp is 16-aligned here.
      push(rbx);            // save guest rbx (callee-saved: survives the calls)
      mov(rbx, rsp);        // remember the guest stack
      if (ownStack) {
        // Ask for this thread's handler stack without disturbing the args.
        push(rdi); push(rsi); push(rdx); push(rcx); push(r8); push(r9);
        sub(rsp, 8);
        mov(rax, reinterpret_cast<uintptr_t>(&krnl_kstack_top));
        call(rax);
        add(rsp, 8);
        mov(r11, rax);      // r11/rcx are scratch under the syscall convention
        pop(r9); pop(r8); pop(rcx); pop(rdx); pop(rsi); pop(rdi);
        Xbyak::Label keep;
        test(r11, r11);
        jz(keep);
        mov(rsp, r11);      // run the handler on its own stack
        L(keep);
      }
      and_(rsp, -16);       // the guest rsp is safe in rbx either way
      mov(rax, handler);
      call(rax);            // handler(rdi,rsi,rdx,rcx,r8,r9) -> rax (args intact)
      mov(rsp, rbx);        // back onto the guest stack
      push(rax);            // stash the raw return across the helper call
      mov(rdi, rax);
      mov(rax, reinterpret_cast<uintptr_t>(&krnl_syscall_errno));
      call(rax);            // eax = errno, or 0 when the return is a result
      pop(r11);             // r11 = raw return
      test(eax, eax);
      Xbyak::Label ok;
      jz(ok);
      // error path: eax = positive errno, rsp 8 mod 16 (at the saved rbx).
      if (trace) {
        push(rax);          // save errno; rsp is now 16-aligned for the call
        mov(esi, eax);      // arg2 = errno
        mov(edi, sid);      // arg1 = syscall id
        mov(rax, reinterpret_cast<uintptr_t>(&trace_syscall_err));
        call(rax);
        pop(rax);           // restore errno
      }
      pop(rbx);
      stc();                // error: rax already = positive errno
      ret();
      L(ok);
      mov(rax, r11);        // success: restore the raw result
      pop(rbx);
      clc();
      ret();
    }
  };
  auto *gen = new bsdRet(reinterpret_cast<uintptr_t>(handler), sid, trace,
                         count, ownStack);
  return reinterpret_cast<uintptr_t>(gen->getCode());
}
#endif // DELTA_BACKEND_NATIVE

uintptr_t lv2_trampoline(const void *handler, u32 sid) {
#if defined(DELTA_BACKEND_NATIVE)
  // Wrap each handler once. Cache by handler so syscall ids that share a handler
  // (and the many duplicate sites in the guest) reuse a single stub. When
  // DELTA_SCERR_TRACE or DELTA_SCHIST is set the stub also reports under its
  // own number, so key the cache by sid instead.
  static std::mutex trMutex;
  static std::unordered_map<u64, uintptr_t> trCache;
  std::lock_guard<std::mutex> lk(trMutex);
  u64 key = (kScerrTrace || g_scHist) ? static_cast<u64>(sid)
                                      : reinterpret_cast<u64>(handler);
  auto it = trCache.find(key);
  if (it != trCache.end())
    return it->second;
  uintptr_t tr = emit_bsd_trampoline(handler, sid, kScerrTrace, g_scHist);
  trCache.emplace(key, tr);
  return tr;
#else
  return reinterpret_cast<uintptr_t>(handler);
#endif
}

// APR (0x2bc..0x2c0, the PS5 async-package-read interface) must FAIL, not
// empty-succeed: the Prospero table stubs it as success without filling any
// output, and a title that routes its /app0 asset opens through
// sceKernelAprResolve then trusts garbage resolve results instead of taking
// its plain-open fallback (Demon's Souls reported every asset as FileNotFound
// this way). ENOSYS makes the libkernel wrapper return a negative SCE error
// and the engine falls back to open/read, which the VFS serves fine.
// TODO: fold into lv2/ps5/table.cpp once that file is free to edit.
static int PS4ABI sys_apr_unavailable() { return -SysError::eNOSYS; }

uintptr_t lv2_lookup(u32 sid) {
  // PS5 titles route to the Prospero table (FreeBSD 11 ABI); never the Orbis one.
  auto *pr = proc::getActive();
  if (pr && pr->getPlatform() == proc::platform::ps5) {
    if (sid >= 0x2bc && sid <= 0x2c0) // apr_submit..apr_ctrl
      return lv2_trampoline(reinterpret_cast<const void *>(&sys_apr_unavailable),
                            sid);
    return lv2_get_ps5(sid);
  }
  return lv2_get(sid);
}
} // namespace krnl
