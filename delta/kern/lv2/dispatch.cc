/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <sys/mman.h>
#include <cstdlib>
#include "base/arch.h"
#include "base/logging.h"
#include "guest/session.h"
#include "guest_abi.h"
#ifdef _MSC_VER
#include <intrin.h>
#endif
#if defined(DELTA_BACKEND_NATIVE)
#include <xbyak.h>
#endif

#include "base/atomic.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "cpu/backend.h"
#include "kern/crash.h"
#include "kern/lv2/dispatch.h"
#include "kern/module.h"
#include "kern/process.h"

#if defined(DELTA_TRACY) && defined(DELTA_BACKEND_NATIVE)
#include <cstring>

#include "base/strings/format.h"
#include "tracy/TracyC.h"
#define DELTA_SYSCALL_ZONES 1
#endif

namespace {
DELTA_OPTION(bool, kScerrTrace, "DELTA_SCERR_TRACE", false);
}  // namespace

namespace kern {

// Read by the stub handlers and the trampoline alike, so it is a kern symbol
// rather than a file-local one.
DELTA_OPTION(bool, g_sc_hist, "DELTA_SCHIST", false);

ModuleInfo* CalledIn(void* addr) {
  uintptr_t addrsafe = (uintptr_t)addr;

  for (auto& mod : Process::GetActive()->GetModuleList()) {
    auto& info = mod->GetInfo();

    if (addrsafe <= (uintptr_t)(info.base + info.code_size) &&
        (addrsafe >= (uintptr_t)info.base)) {
      BASE_LOGI("nullhandler", "{:p} called in {}", addr, info.name.c_str());
      return &info;
    }
  }
  return nullptr;
}

int PS4ABI Lv2StubSyscall() {
#ifdef _MSC_VER
  void* ret = __builtin_return_address(0);
#else
  void* ret = __builtin_return_address(0);
#endif
  CalledIn(ret);

  static base::Atomic<u64> nulls{0};
  const u64 n = ++nulls;
  if (n == 1 || (g_sc_hist && n % 400 == 0)) {
    BASE_LOGI("nullhandler", ">>>>>>>>>>>>> NULL HANDLER called {} times",
              (unsigned long long)n);
    if (g_sc_hist)
      DumpSyscallHist();
  }
  return 0;
}

int PS4ABI Lv2UnmappedSyscall() {
#ifdef _MSC_VER
  void* ret = __builtin_return_address(0);
#else
  void* ret = __builtin_return_address(0);
#endif
  CalledIn(ret);

  BASE_LOGI("nullhandler",
            ">>>>>>>>>>>>> NULL HANDLER NULLTABLE CALLED BY {:p}", ret);
  return 0;
}

// BSD/PS4 syscall return convention: failure = positive errno in rax with carry
// SET; success = carry clear, result in rax. Our C handlers use Linux-style
// negative errno, zero-extended by `int` handlers and full-width by `i64` ones
// (0x00000000_FFFFFFxx or 0xFFFFFFFF_FFFFFFxx; guest pointers live >= 64 GiB
// and never match). Classify for both the native trampoline and the FEX bridge.
extern "C" u32 SyscallErrno(u64 raw) {
  i32 lo = static_cast<i32>(static_cast<u32>(raw));
  u32 hi = static_cast<u32>(raw >> 32);
  if (lo < 0 && lo >= -static_cast<i32>(SysError::eLAST) &&
      (hi == 0 || hi == 0xFFFFFFFFu))
    return static_cast<u32>(-lo);
  return 0;
}

#if defined(DELTA_BACKEND_NATIVE)
// Per-thread stack for syscall handlers (the kernel-stack equivalent). The
// native backend runs guest code on the host thread, so a handler would
// otherwise run on the guest's stack: SotC's fibers are 16 KiB with saved
// context at the bottom, which a handler's host frames walk straight through (a
// fiber resumed inside glibc free()). 0 = no switch; nesting works because a
// guest callback's syscalls land on the already-switched stack.
extern "C" u64 KernelStackTop() {
  constexpr size_t kSize = 1u << 20;  // 1 MiB, lazily backed
  struct Stack {
    u8* base = nullptr;
    ~Stack() {
      if (base)
        munmap(base, 1u << 20);
    }
  };
  static thread_local Stack stack;
  auto& base = stack.base;
  const auto here = reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
  if (base) {
    if (here > reinterpret_cast<uintptr_t>(base) &&
        here < reinterpret_cast<uintptr_t>(base) + kSize)
      return 0;  // already on it
  } else {
    void* p = ::mmap(nullptr, kSize, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED)
      return 0;
    base = static_cast<u8*>(p);
  }
  // Where the guest's own stack was, so a handler-side stack scan (see
  // crash.cc GuestStackTrace) still finds the guest frames it is after.
  SetGuestStackScanBase(here);
  return reinterpret_cast<u64>(base + kSize);
}

// DELTA_SCERR_TRACE: emitted (when set) on the trampoline's error path so we
// can see exactly which syscall returned which errno just before a guest abort.
// The errno is the BSD-style positive value the guest's libkernel turns into an
// SCE error (0x80020000 | errno), e.g. errno 1 -> 0x80020001.
static void PS4ABI TraceSyscallErr(u32 sid, u32 err) {
  const char* name = SyscallGetname(sid);
  BASE_LOGI("syscallerr", "{} {} -> errno {} (SCE {:#010x})", sid,
            name ? name : "?", err, 0x80020000u | err);
}
#endif  // DELTA_BACKEND_NATIVE

// DELTA_SCHIST per-syscall-id counter, the only profiler on this build
// (perf/strace/ proc-mem are yama-blocked); dumped from the SIGUSR1 probe.
// Outside the NATIVE guard: the stub handlers read these unconditionally, so
// the FEX build must link them too.
extern "C" u64 g_sys_hist[1024] = {};

// Also dump from the unimplemented-syscall stub: most runs neither crash nor
// exit gracefully (SIGKILL), and "which syscall is this title hammering"
// usually comes up there.
void DumpSyscallHist() {
  BASE_LOGI("schist", "syscall call counts:");
  for (int i = 0; i < 1024; i++) {
    if (!g_sys_hist[i])
      continue;
    const char* n = SyscallGetname(i);
    BASE_LOGI("schist", "{:4} {:<24} {}", i, n ? n : "?",
              (unsigned long long)g_sys_hist[i]);
  }
}

static const bool kScHistDump = [] {
  if (!g_sc_hist)
    return false;
  std::atexit(&DumpSyscallHist);
  // The emulator is normally SIGKILLed, so also sample on a timer: what a title
  // is doing in STEADY STATE is a different question from what it did at boot,
  // and only a periodic dump answers it.
  guest::SpawnThread("dispatch", [] {
    for (;;) {
      guest::SleepForMilliseconds((20) * 1000);
      DumpSyscallHist();
    }
  });
  return true;
}();

#if defined(DELTA_SYSCALL_ZONES)
// Each syscall is a Tracy zone named after it, on the guest thread that made
// it: where a guest thread blocks, and which calls cost, is on the timeline.
extern "C" u64 SyscallZoneBegin(const ___tracy_source_location_data* where) {
  const TracyCZoneCtx ctx = ___tracy_emit_zone_begin(where, 1);
  u64 packed;
  std::memcpy(&packed, &ctx, sizeof(packed));
  return packed;
}

extern "C" void SyscallZoneEnd(u64 packed) {
  TracyCZoneCtx ctx;
  std::memcpy(&ctx, &packed, sizeof(ctx));
  ___tracy_emit_zone_end(ctx);
}

// Source locations must outlive every zone that names them; one per syscall id.
// The clock and yield calls are left out: a title spinning on them makes
// millions a second, and zoning each one costs more than the call and swamps
// the capture (DELTA_SCHIST counts them instead).
static const ___tracy_source_location_data* SyscallZoneLocation(u32 sid) {
  const char* name = SyscallGetname(sid);
  static const char* const kTooHot[] = {"sys_gettimeofday", "sys_clock_gettime",
                                        "sys_clock_getres", "sys_sched_yield",
                                        "sys_thr_self",     "sys_getpid"};
  for (const char* hot : kTooHot)
    if (name && std::strcmp(name, hot) == 0)
      return nullptr;
  const base::String label =
      name ? base::String("sys.") + name : base::Format("sys.{}", sid);
  static base::Map<u32, ___tracy_source_location_data*> locations;
  auto it = locations.find(sid);
  if (it != locations.end())
    return it->second;
  auto* where = new ___tracy_source_location_data{};
  locations[sid] = where;
  where->name = strdup(label.c_str());
  where->function = where->name;
  where->file = __FILE__;
  where->line = sid;
  return where;
}
#endif

#if defined(DELTA_BACKEND_NATIVE)
// One-time trampoline per handler: call it with the guest's arg registers
// untouched, then set/clear the carry flag and normalise rax per the convention
// above. Replaces the bare `call handler` so the guest sees faithful errors.
static uintptr_t EmitBsdTrampoline(const void* handler,
                                   u32 sid,
                                   bool trace,
                                   bool count,
                                   const void* zone) {
  // The two handlers that never return (they longjmp out of the guest call
  // chain, cpu::ExitGuestThread) must stay on the guest stack: glibc's
  // longjmp check rejects a jump to a frame that is not on the current stack.
  // Neither needs the room anyway.
  const bool own_stack = sid != 1 /*exit*/ && sid != 431 /*thr_exit*/;
  struct BsdRet : Xbyak::CodeGenerator {
    BsdRet(uintptr_t handler,
           u32 sid,
           bool trace,
           bool count,
           bool own_stack,
           const void* zone) {
      if (count) {  // DELTA_SCHIST: ++g_sys_hist[sid] (rax is caller-saved)
        mov(rax, reinterpret_cast<uintptr_t>(&g_sys_hist[sid & 1023]));
        inc(qword[rax]);
      }
      // The lifted site enters with `call`, so rsp is 16-aligned here.
      push(rbx);      // save guest rbx (callee-saved: survives the calls)
      mov(rbx, rsp);  // remember the guest stack
      if (own_stack) {
        // Ask for this thread's handler stack without disturbing the args.
        push(rdi);
        push(rsi);
        push(rdx);
        push(rcx);
        push(r8);
        push(r9);
        sub(rsp, 8);
        mov(rax, reinterpret_cast<uintptr_t>(&KernelStackTop));
        call(rax);
        add(rsp, 8);
        mov(r11, rax);  // r11/rcx are scratch under the syscall convention
        pop(r9);
        pop(r8);
        pop(rcx);
        pop(rdx);
        pop(rsi);
        pop(rdi);
        Xbyak::Label keep;
        test(r11, r11);
        jz(keep);
        mov(rsp, r11);  // run the handler on its own stack
        L(keep);
      }
      and_(rsp, -16);  // the guest rsp is safe in rbx either way
      if (zone) {
        // [rsp, +48) the six arg registers, +48 the zone, +56 the result.
        const Xbyak::Reg64 args[] = {rdi, rsi, rdx, rcx, r8, r9};
        sub(rsp, 64);
        for (int i = 0; i < 6; i++)
          mov(qword[rsp + i * 8], args[i]);
        mov(rdi, reinterpret_cast<uintptr_t>(zone));
        mov(rax, reinterpret_cast<uintptr_t>(&SyscallZoneBegin));
        call(rax);
        mov(qword[rsp + 48], rax);
        for (int i = 0; i < 6; i++)
          mov(args[i], qword[rsp + i * 8]);
        mov(rax, handler);
        call(rax);
        mov(qword[rsp + 56], rax);
        mov(rdi, qword[rsp + 48]);
        mov(rax, reinterpret_cast<uintptr_t>(&SyscallZoneEnd));
        call(rax);
        mov(rax, qword[rsp + 56]);
      } else {
        mov(rax, handler);
        call(rax);  // handler(rdi,rsi,rdx,rcx,r8,r9) -> rax (args intact)
      }
      mov(rsp, rbx);  // back onto the guest stack
      push(rax);      // stash the raw return across the helper call
      mov(rdi, rax);
      mov(rax, reinterpret_cast<uintptr_t>(&SyscallErrno));
      call(rax);  // eax = errno, or 0 when the return is a result
      pop(r11);   // r11 = raw return
      test(eax, eax);
      Xbyak::Label ok;
      jz(ok);
      // error path: eax = positive errno, rsp 8 mod 16 (at the saved rbx).
      if (trace) {
        push(rax);      // save errno; rsp is now 16-aligned for the call
        mov(esi, eax);  // arg2 = errno
        mov(edi, sid);  // arg1 = syscall id
        mov(rax, reinterpret_cast<uintptr_t>(&TraceSyscallErr));
        call(rax);
        pop(rax);  // restore errno
      }
      pop(rbx);
      stc();  // error: rax already = positive errno
      ret();
      L(ok);
      mov(rax, r11);  // success: restore the raw result
      pop(rbx);
      clc();
      ret();
    }
  };
  static base::Vector<Xbyak::CodeGenerator*> generators;
  static const guest::SessionReset reset_generators([] {
    for (auto* generator : generators)
      delete generator;
    guest::ResetResource(generators);
  });
  auto* gen = new BsdRet(cpu::MakeSyscallPauseThunk(handler), sid, trace, count,
                         own_stack, zone);
  generators.push_back(gen);
  return reinterpret_cast<uintptr_t>(gen->getCode());
}
#endif  // DELTA_BACKEND_NATIVE

uintptr_t Lv2Trampoline(const void* handler, u32 sid) {
#if defined(DELTA_BACKEND_NATIVE)
  // Wrap each handler once. Cache by handler so syscall ids that share a
  // handler (and the many duplicate sites in the guest) reuse a single stub.
  // When DELTA_SCERR_TRACE or DELTA_SCHIST is set the stub also reports under
  // its own number, so key the cache by sid instead.
  static base::Mutex tr_mutex;
  static base::HashMap<u64, uintptr_t> tr_cache;
  static const guest::SessionReset reset_tr_cache(
      [] { guest::ResetResource(tr_cache); });
  base::LockGuard<base::Mutex> lk(tr_mutex);
#if defined(DELTA_SYSCALL_ZONES)
  constexpr bool kPerSyscall = true;  // each zone names its own syscall
#else
  constexpr bool kPerSyscall = false;
#endif
  u64 key = (kPerSyscall || kScerrTrace || g_sc_hist)
                ? static_cast<u64>(sid)
                : reinterpret_cast<u64>(handler);
  auto it = tr_cache.find(key);
  if (it != tr_cache.end())
    return it->second;
  const void* zone = nullptr;
#if defined(DELTA_SYSCALL_ZONES)
  // exit and thr_exit longjmp out of the handler and never close a zone.
  if (sid != 1 && sid != 431)
    zone = SyscallZoneLocation(sid);
#endif
  uintptr_t tr = EmitBsdTrampoline(handler, sid, kScerrTrace, g_sc_hist, zone);
  tr_cache.emplace(key, tr);
  return tr;
#else
  return reinterpret_cast<uintptr_t>(handler);
#endif
}

// APR (0x2bc..0x2c0, PS5 async-package-read) must FAIL, not empty-succeed: the
// Prospero stub's success-with-garbage made Demon's Souls trust bogus resolve
// results (every asset FileNotFound) instead of its plain-open fallback. ENOSYS
// sends the wrapper negative and the engine falls back to VFS open/read.
static int PS4ABI sys_apr_unavailable() {
  return -SysError::eNOSYS;
}

uintptr_t Lv2Lookup(u32 sid) {
  // PS5 titles route to the Prospero table (FreeBSD 11 ABI); never the Orbis
  // one.
  auto* pr = Process::GetActive();
  if (pr && pr->GetPlatform() == Process::Platform::kPs5) {
    if (sid >= 0x2bc && sid <= 0x2c0)  // apr_submit..apr_ctrl
      return Lv2Trampoline(reinterpret_cast<const void*>(&sys_apr_unavailable),
                           sid);
    return Lv2GetPs5(sid);
  }
  return Lv2Get(sid);
}
}  // namespace kern
