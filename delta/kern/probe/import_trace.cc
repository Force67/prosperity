/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * DELTA_IMPORT_TRACE=<module or NID>[,...]|all[:<seconds>]: name the system
 * library calls a title makes. Every import that binds into a matching module
 * goes through a stub that records (import, caller) in a ring and jumps on;
 * a reporter thread prints the first calls of each import and, every few
 * seconds, which ones are hot. Answers "what does the title do after this
 * input" without a debugger. x86-64 native backend only.
 */

#include "kern/probe/probe_internal.h"

#include <sys/syscall.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>

#include "base/arch.h"
#include "base/atomic.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "base/time/time.h"
#include "host_memory/host_memory.h"
#include "kern/crash.h"
#include "options/options.h"

#if defined(DELTA_BACKEND_NATIVE)
#include <xbyak.h>
#endif

namespace {
DELTA_OPTION(const char*, kImportTrace, "DELTA_IMPORT_TRACE", nullptr);
}  // namespace

namespace kern::probe {
namespace {

struct Traced {
  char name[48];     // the import as the title names it (NID#lib#mod)
  char target[64];   // where it binds, "<module>+off"
  base::Atomic<u64> calls{0};
  base::Atomic<u64> named{0};
  u64 reported = 0;  // calls already covered by a hot-list line
};

constexpr u32 kMaxTraced = 4096;
constexpr u64 kNamedCalls = 6;
Traced g_traced[kMaxTraced];
base::Atomic<u32> g_traced_n{0};

struct Call {
  u32 index;
  u32 tid;
  uintptr_t caller;
  u64 args[3];
};
constexpr u32 kRing = 1 << 14;
Call g_ring[kRing];
base::Atomic<u64> g_ring_head{0};

// Runs on the guest's stack from the stub: no formatting, no allocation.
// Set once the start time has passed: calls before it are only counted, so
// an import the title used all along is still named when it matters.
base::Atomic<bool> g_armed{false};

void Record(u64 index, uintptr_t caller, const u64* args) {
  Traced& t = g_traced[index];
  t.calls.fetch_add(1, base::memory_order_relaxed);
  if (!g_armed.load(base::memory_order_relaxed))
    return;
  // Only the first calls since arming are named; the rest are counted.
  if (t.named.fetch_add(1, base::memory_order_relaxed) >= kNamedCalls)
    return;
  const u64 slot = g_ring_head.fetch_add(1, base::memory_order_relaxed);
  g_ring[slot % kRing] = {static_cast<u32>(index),
                          static_cast<u32>(::syscall(SYS_gettid)),
                          caller,
                          {args[2], args[1], args[0]}};  // pushed rdi first
}

// A value as a number, or as the string it points at when that reads as
// text (paths, names): the argument a trace is usually after.
void FormatArg(u64 value, char* out, size_t n) {
  const auto* p = reinterpret_cast<const char*>(value);
  if (value > 0x10000 && host_memory::IsMemoryRangeMapped(p, 32)) {
    size_t len = 0;
    while (len < 48 && p[len] >= 0x20 && p[len] < 0x7f)
      len++;
    if (len >= 4 && (len == 48 || p[len] == '\0')) {
      std::snprintf(out, n, "\"%.*s\"", static_cast<int>(len), p);
      return;
    }
  }
  std::snprintf(out, n, "%#llx", static_cast<unsigned long long>(value));
}

void Reporter(double start_s) {
  const base::TimeTicks t0 = base::TimeTicks::Now();
  u64 shown = 0;
  base::TimeTicks last_hot = t0;
  for (;;) {
    base::SleepForMilliseconds(250);
    const double now_s = (base::TimeTicks::Now() - t0).InSecondsF();
    if (now_s >= start_s)
      g_armed.store(true);
    const u64 head = g_ring_head.load(base::memory_order_relaxed);
    for (; shown < head; shown++) {
      const Call& c = g_ring[shown % kRing];
      char caller[128];
      Symbolize(c.caller, caller, sizeof(caller));
      char args[3][72];
      for (int i = 0; i < 3; i++)
        FormatArg(c.args[i], args[i], sizeof(args[i]));
      BASE_LOGI("imptrace", "{} -> {} ({}, {}, {}) from {} tid={}",
                g_traced[c.index].name, g_traced[c.index].target, args[0],
                args[1], args[2], caller, c.tid);
    }
    if (now_s < start_s ||
        base::TimeTicks::Now() - last_hot < base::Seconds(5))
      continue;
    last_hot = base::TimeTicks::Now();
    const u32 n = g_traced_n.load();
    for (u32 i = 0; i < n; i++) {
      Traced& t = g_traced[i];
      const u64 calls = t.calls.load(base::memory_order_relaxed);
      if (calls - t.reported >= 50)
        BASE_LOGI("imptrace", "hot: {} -> {} x{} in 5s", t.name, t.target,
                  calls - t.reported);
      t.reported = calls;
    }
  }
}

}  // namespace

uintptr_t MaybeTraceImport(const char* nid_name, uintptr_t real_addr) {
#if defined(DELTA_BACKEND_NATIVE)
  const char* spec = kImportTrace;
  if (!spec || !real_addr)
    return real_addr;
  char want[64] = {};
  double start_s = 0;
  if (const char* colon = std::strchr(spec, ':')) {
    std::strncpy(want, spec,
                 base::Min<size_t>(sizeof(want) - 1, colon - spec));
    start_s = std::strtod(colon + 1, nullptr);
  } else {
    std::strncpy(want, spec, sizeof(want) - 1);
  }
  char target[64];
  Symbolize(real_addr, target, sizeof(target));
  // "all" skips the libraries every frame lives in.
  // Tokens match the target module or the import's NID, which is how an
  // import bound to one of our HLE functions is picked.
  const bool all = std::strcmp(want, "all") == 0;
  bool match = false;
  for (char* tok = std::strtok(want, ","); tok && !match;
       tok = std::strtok(nullptr, ","))
    match = std::strstr(target, tok) || std::strstr(nid_name, tok);
  if (all ? (std::strstr(target, "libkernel") || std::strstr(target, "libc+") ||
             std::strstr(target, "LibcInternal"))
          : !match)
    return real_addr;
  const u32 index = g_traced_n.fetch_add(1);
  if (index >= kMaxTraced)
    return real_addr;
  Traced& t = g_traced[index];
  std::strncpy(t.name, nid_name, sizeof(t.name) - 1);
  std::strncpy(t.target, target, sizeof(t.target) - 1);
  if (char* sp = std::strchr(t.target, ' '))
    *sp = '\0';
  static const bool started = [start_s] {
    base::SpawnDetachedThread("imptrace", [start_s] { Reporter(start_s); });
    return true;
  }();
  (void)started;

  // Entered by the PLT's jump, so [rsp] is the guest's return address and rsp
  // is 8 mod 16. Keep every argument register, rax (a variadic call's vector
  // count) and xmm0-7, then leave through r11, the one register the ABI lets
  // a PLT stub clobber.
  struct Stub : Xbyak::CodeGenerator {
    Stub(u64 index, uintptr_t target) : Xbyak::CodeGenerator(256) {
      push(rbp);
      mov(rbp, rsp);
      for (const auto& r : {rdi, rsi, rdx, rcx, r8, r9, rax, r10})
        push(r);
      sub(rsp, 128);
      for (int i = 0; i < 8; i++)
        movdqu(ptr[rsp + i * 16], Xbyak::Xmm(i));
      // The saved rdi, rsi, rdx sit at rbp-8, -16, -24: pass their address.
      lea(rdx, ptr[rbp - 24]);
      mov(rdi, index);
      mov(rsi, ptr[rbp + 8]);
      mov(rax, reinterpret_cast<uintptr_t>(&Record));
      call(rax);
      for (int i = 0; i < 8; i++)
        movdqu(Xbyak::Xmm(i), ptr[rsp + i * 16]);
      add(rsp, 128);
      for (const auto& r : {r10, rax, r9, r8, rcx, rdx, rsi, rdi})
        pop(r);
      pop(rbp);
      mov(r11, target);
      jmp(r11);
    }
  };
  auto* stub = new Stub(index, real_addr);
  return reinterpret_cast<uintptr_t>(stub->getCode());
#else
  (void)nid_name;
  return real_addr;
#endif
}

}  // namespace kern::probe
