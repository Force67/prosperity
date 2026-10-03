/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The guest-code trap engine: the int3/trap probes a research session arms, the
 * signal dispatch that services them, and the threads that report them.
 *
 * This used to live inside kern/crash.cc, which meant the fatal crash reporter
 * and two dozen unrelated debug probes shared one file and one header. They do
 * share the signal handler (a probe trap and a real fault arrive the same
 * way), so the seam is OnSignal(): crash.cc offers every signal here first and
 * only reports a fault the probes did not claim. Ordering matters and is
 * preserved: the probes run BEFORE the CPU backend's JIT-signal hand-off,
 * because a trap we planted is ours whatever the JIT would make of it.
 *
 * The fault-recovery policies (null-guard, SDK assert skip) deliberately stay
 * in crash.cc: they run after the JIT hand-off and are part of deciding
 * whether a fault is fatal at all, not part of tracing guest code.
 */

#include "cpu/backend.h"
#include "guest/session.h"
#include "kern/probe/probe_arm.h"

#include "base/arch.h"
#include "kern/crash.h"
#include "kern/lv2/dispatch.h"
#include "kern/process.h"
#include "kern/vfs.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <ucontext.h>
#include <unistd.h>

#include "base/atomic.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "base/threading/thread.h"
#include "host_memory/host_memory.h"
#include "kern/lv2/sys_thread.h"
#include "options/options.h"
#include "write_watch/write_watch.h"

namespace {
DELTA_OPTION(uintptr_t, kBrkTrace, "DELTA_GUEST_BRK_TRACE", 0);
DELTA_OPTION(const char*, kBrkDump, "DELTA_GUEST_BRK_DUMP", nullptr);
DELTA_OPTION(uintptr_t, kBrkArm, "DELTA_GUEST_BRK_ARM", 0);
DELTA_OPTION(u32, kBrkArmPos, "DELTA_GUEST_BRK_ARM_POS", 0);
DELTA_OPTION(const char*, kBrkPeek, "DELTA_GUEST_BRK_PEEK", nullptr);
DELTA_OPTION(const char*, kBrkWprot, "DELTA_GUEST_BRK_WPROT", nullptr);
DELTA_OPTION(const char*, kCrashPeek, "DELTA_CRASH_PEEK", nullptr);
DELTA_OPTION(bool, kCntClamp, "DELTA_CNT_CLAMP", false);
DELTA_OPTION(bool, kHdrFill, "DELTA_HDR_FILL", false);
DELTA_OPTION(bool, kHdrWait, "DELTA_HDR_WAIT", false);
DELTA_OPTION(bool, kPs5Dcbforce, "DELTA_PS5_DCBFORCE", false);
DELTA_OPTION(bool, kRdoffNofix, "DELTA_RDOFF_NOFIX", false);
DELTA_OPTION(bool, kRdoffTrace, "DELTA_RDOFF_TRACE", false);
}  // namespace

namespace kern {}

namespace kern::probe {

// DELTA_HEAP_PROF: dump the top allocation sites (defined below).
extern uintptr_t g_heap_prof_addr;
static void HeapProfDumpOnce();

// SIGUSR1 probe: dump the receiving thread's guest RIP + a module-stack scan,
// one per /proc task, to find what a wedged title's threads are blocked on.
#if defined(__x86_64__)
static void ProbeHandler(int, siginfo_t*, void* ucv) {
  auto* uc = static_cast<ucontext_t*>(ucv);
  auto* gr = uc->uc_mcontext.gregs;
  char rip[256];
  Symbolize(gr[REG_RIP], rip, sizeof(rip));
  // The GUEST tid too: the host tid says nothing about which of the title's
  // threads this is, the first thing to know when it stops.
  BASE_LOGI("probe", "tid={} gtid={} rip={:016x} {}", (long)gettid(),
            *CurrentGuestTidPtr(), (unsigned long long)gr[REG_RIP], rip);
  // GPRs too: a thread caught in a busy-wait only makes sense with the address
  // and value it is polling.
  BASE_LOGI("probe",
            "  rax={:016x} rbx={:016x} rcx={:016x} rdx={:016x}\n"
            "  rsi={:016x} rdi={:016x} r8 ={:016x} r9 ={:016x}\n"
            "  r12={:016x} r13={:016x} r14={:016x} r15={:016x}",
            (unsigned long long)gr[REG_RAX], (unsigned long long)gr[REG_RBX],
            (unsigned long long)gr[REG_RCX], (unsigned long long)gr[REG_RDX],
            (unsigned long long)gr[REG_RSI], (unsigned long long)gr[REG_RDI],
            (unsigned long long)gr[REG_R8], (unsigned long long)gr[REG_R9],
            (unsigned long long)gr[REG_R12], (unsigned long long)gr[REG_R13],
            (unsigned long long)gr[REG_R14], (unsigned long long)gr[REG_R15]);
  // The frame chain first: a stack scan finds stale return addresses too, which
  // is misleading when the question is "what is this thread blocked in".
  Backtrace(gr[REG_RBP]);
  // The host frames too, raw: which of our waits a parked thread is in.
  // tools/drun.py symbolizes them against the binary.
  {
    void* frames[24];
    const int n = ::backtrace(frames, 24);
    char line[24 * 17 + 1];
    int at = 0;
    for (int i = 0; i < n; i++)
      at += std::snprintf(line + at, sizeof(line) - at, " %lx",
                          reinterpret_cast<unsigned long>(frames[i]));
    line[at] = '\0';
    BASE_LOGI("probe", "  hostbt:{}", line);
  }
  // A thread parked in a wait is parked inside a SYSCALL, so its rsp is our own
  // handler stack; the guest stack it came off is recorded on syscall entry,
  // copied out with process_vm_readv, and is the one that names the waiter.
  GuestStackTrace("probe", 8);
  // DELTA_SCHIST histogram dump: the only profiler available (perf/strace/
  // proc-mem are yama-blocked here). Only the first responder of a burst
  // prints it; forty interleaved copies are unreadable.
  static base::Atomic<u64> last_hist{0};
  const u64 now_s = (u64)::time(nullptr);
  u64 prev = last_hist.load();
  if (now_s - prev < 2 || !last_hist.compare_exchange_strong(prev, now_s)) {
    std::fflush(stderr);
    return;
  }
  bool any = false;
  for (int i = 0; i < 1024; i++) {
    if (g_sys_hist[i] > 100) {  // skip noise
      if (!any) {
        BASE_LOGI("schist", "syscall counts:");
        any = true;
      }
      BASE_LOGI("schist", "  {:4} {:<28} {}", i, SyscallGetname(i),
                (unsigned long long)g_sys_hist[i]);
    }
  }
  if (g_heap_prof_addr)
    HeapProfDumpOnce();
  std::fflush(stderr);
}
#endif

// DELTA_ALLOC_TRACE: int3 at a guest allocator entry (push rbp) to log each
// large allocation's size: the handler logs rsi when big, emulates the push,
// resumes. One trap per call, no single-stepping; gdb is too slow here.
uintptr_t g_alloc_trace_addr = 0;
u64 g_alloc_trace_min = 0x1000000;  // 16 MiB
// DELTA_HEAP_PROF: aggregate operator-new/malloc by guest caller in a fixed
// lock-free open-addressing table (called from every guest thread).
uintptr_t g_heap_prof_addr = 0;  // non-zero once any hook is armed
static constexpr int kHeapProfMaxHooks = 24;
static uintptr_t g_heap_prof_hooks[kHeapProfMaxHooks];
static int g_heap_prof_hook_count = 0;
static base::Atomic<u64> g_heap_prof_hook_bytes[kHeapProfMaxHooks];
static base::Atomic<u64> g_heap_prof_hook_calls[kHeapProfMaxHooks];
static bool g_heap_prof_count_only[kHeapProfMaxHooks];
namespace {
constexpr u32 kHeapProfSlots = 16384;
struct HeapProfSlot {
  base::Atomic<uintptr_t> caller{0};
  base::Atomic<u64> bytes{0};
  base::Atomic<u64> count{0};
  base::Atomic<u64> unscoped{0};  // of `bytes`, taken with no scope live
};
HeapProfSlot g_heap_prof[kHeapProfSlots];
base::Atomic<u64> g_heap_prof_total{0};

// DELTA_HEAP_PROF_SCOPE=<tls-slot-global>:<depth-offset>: engines route
// allocations through a THREAD-LOCAL stack of scoped allocators, falling back
// to the process heap when empty (scope memory frees wholesale, fallback does
// not), so a leak that is really "this thread had no scope" is invisible to a
// call-site profile. Record the guest's own indirection:
//   slot = fs_base + *(u64*)<tls-slot-global>; block = *(u64*)slot;
//   depth = block ? *(u32*)(block + <depth-offset>) : 0
uintptr_t g_heap_prof_scope_slot = 0;
u64 g_heap_prof_scope_depth_off = 0;
base::Atomic<u64> g_heap_prof_scoped{0}, g_heap_prof_unscoped{0};

// Reads through process_vm_readv so a mis-specified address reports nothing
// instead of taking the run down from inside the trap handler.
bool HeapProfPeek(uintptr_t addr, void* out, size_t n) {
  if (addr < 0x10000)
    return false;
  iovec l{out, n}, r{reinterpret_cast<void*>(addr), n};
  return ::process_vm_readv(::getpid(), &l, 1, &r, 1, 0) == (ssize_t)n;
}

u32 HeapProfScopeDepth(uintptr_t fs_base) {
  if (!g_heap_prof_scope_slot || !fs_base)
    return 0;
  u64 off = 0, block = 0;
  u32 depth = 0;
  if (!HeapProfPeek(g_heap_prof_scope_slot, &off, sizeof(off)))
    return 0;
  if (!HeapProfPeek(fs_base + off, &block, sizeof(block)))
    return 0;
  if (!HeapProfPeek(block + g_heap_prof_scope_depth_off, &depth, sizeof(depth)))
    return 0;
  return depth;
}

void HeapProfRecord(uintptr_t caller, u64 size, bool unscoped) {
  u32 h = static_cast<u32>((caller * 2654435761u) >> 13) & (kHeapProfSlots - 1);
  const auto add = [&](u32 s) {
    g_heap_prof[s].bytes.fetch_add(size, base::memory_order_relaxed);
    g_heap_prof[s].count.fetch_add(1, base::memory_order_relaxed);
    if (unscoped)
      g_heap_prof[s].unscoped.fetch_add(size, base::memory_order_relaxed);
  };
  for (u32 i = 0; i < kHeapProfSlots; i++) {
    u32 s = (h + i) & (kHeapProfSlots - 1);
    uintptr_t c = g_heap_prof[s].caller.load(base::memory_order_relaxed);
    if (c == caller) {
      add(s);
      break;
    }
    if (c == 0) {
      uintptr_t expected = 0;
      if (g_heap_prof[s].caller.compare_exchange_strong(
              expected, caller, base::memory_order_relaxed)) {
        add(s);
        break;
      }
      if (expected == caller) {
        add(s);
        break;
      }
    }
  }
  g_heap_prof_total.fetch_add(size, base::memory_order_relaxed);
  (unscoped ? g_heap_prof_unscoped : g_heap_prof_scoped)
      .fetch_add(size, base::memory_order_relaxed);
}
void HeapProfDump() {
  BASE_LOGI("heapprof",
            "total={} bytes ({:.1f} MB) across sites; top by bytes:",
            (unsigned long long)g_heap_prof_total.load(),
            g_heap_prof_total.load() / 1048576.0);
  if (g_heap_prof_scope_slot)
    BASE_LOGI("heapprof",
              "scoped={:.1f} MB  unscoped={:.1f} MB (no scoped "
              "allocator live on the calling thread)",
              g_heap_prof_scoped.load() / 1048576.0,
              g_heap_prof_unscoped.load() / 1048576.0);
  for (int i = 0; i < g_heap_prof_hook_count; i++) {
    u64 b = g_heap_prof_hook_bytes[i].load(base::memory_order_relaxed);
    BASE_LOGI("heapprof", "  hook[{}] {:#x}: {:8.1f} MB  {:8} calls", i,
              (unsigned long)g_heap_prof_hooks[i], b / 1048576.0,
              (unsigned long long)g_heap_prof_hook_calls[i].load(
                  base::memory_order_relaxed));
  }
  // Select top 20 by bytes without allocating (linear passes).
  u64 prev_bytes = ~0ull;
  uintptr_t prev_caller = 0;
  for (int rank = 0; rank < 20; rank++) {
    u64 best_b = 0;
    u32 best_s = kHeapProfSlots;
    for (u32 s = 0; s < kHeapProfSlots; s++) {
      u64 b = g_heap_prof[s].bytes.load(base::memory_order_relaxed);
      if (b == 0)
        continue;
      uintptr_t c = g_heap_prof[s].caller.load(base::memory_order_relaxed);
      bool below = b < prev_bytes || (b == prev_bytes && c > prev_caller);
      if (below && b >= best_b) {
        best_b = b;
        best_s = s;
      }
    }
    if (best_s == kHeapProfSlots)
      break;
    uintptr_t c = g_heap_prof[best_s].caller.load(base::memory_order_relaxed);
    u64 cnt = g_heap_prof[best_s].count.load(base::memory_order_relaxed);
    u64 uns = g_heap_prof[best_s].unscoped.load(base::memory_order_relaxed);
    char sym[200];
    Symbolize(c, sym, sizeof(sym));
    BASE_LOGI("heapprof", "  {:8.1f} MB  {:8} calls  {}{}", best_b / 1048576.0,
              (unsigned long long)cnt, sym,
              g_heap_prof_scope_slot ? (uns == best_b ? "  [all unscoped]"
                                        : uns         ? "  [part unscoped]"
                                                      : "  [scoped]")
                                     : "");
    prev_bytes = best_b;
    prev_caller = c;
  }
  std::fflush(stderr);
}
}  // namespace
void SetHeapProfScope(uintptr_t tls_slot_global, u64 depth_offset) {
  g_heap_prof_scope_slot = tls_slot_global;
  g_heap_prof_scope_depth_off = depth_offset;
}
void SetHeapProf(uintptr_t addr, bool count_only) {
  if (g_heap_prof_hook_count < kHeapProfMaxHooks) {
    g_heap_prof_count_only[g_heap_prof_hook_count] = count_only;
    g_heap_prof_hooks[g_heap_prof_hook_count++] = addr;
  }
  g_heap_prof_addr = addr;  // any non-zero arms the SIGTRAP path
}
// Throttle to one dump per SIGUSR1 burst (every thread gets the signal).
static void HeapProfDumpOnce() {
  static base::Atomic<u64> last{0};
  timespec t{};
  clock_gettime(CLOCK_MONOTONIC, &t);
  u64 now = (u64)t.tv_sec * 1000 + t.tv_nsec / 1000000;
  u64 prev = last.load(base::memory_order_relaxed);
  if (now - prev < 300)
    return;
  if (!last.compare_exchange_strong(prev, now, base::memory_order_relaxed))
    return;
  HeapProfDump();
}
// DELTA_CNT_TRACE: same int3-emulate trick at an entry whose 1st byte is push
// rbp, but logs the per-archive entry-count [rdi+0x30] and the inline name at
// [rdi+0x5c].
uintptr_t g_cnt_trace_addr = 0;
// DELTA_FATAL_TRACE: int3 at a printf-style fatal handler entry (push rbp); log
// rdi (the format string) + caller + the first varargs, then resume.
uintptr_t g_fatal_trace_addr = 0;
// DELTA_HDR_TRACE: int3 at each manifest consumer (push rbp, rdi=parent); the
// header is [parent+0x8], the archive name [[parent+0x10]+0x5c]. Comma-
// separated addresses cover every consumer that reads the header.
uintptr_t g_hdr_trace_addrs[8] = {0};
int g_hdr_trace_count = 0;
// DELTA_RDOFF_FIX: int3 at the file-read-request setter 0x60b510 (push rbp;
// esi=fd edx=off ecx=n r8=buf); force offset 0 for .manifest.bin fds,
// which SOTTR passes as garbage.
uintptr_t g_rdoff_addr = 0;
bool g_manifest_fd[8192] = {false};
void MarkManifestFd(u32 fd, bool v) {
  if (fd < 8192)
    g_manifest_fd[fd] = v;
}
// DELTA_SKIP_FN: int3 at a function entry (push rbp); emulate an immediate ret
// (the push hasn't run, so [rsp] is the return addr) with rax=0, skipping the
// whole function (e.g. the localization loader) to reach the next boot stage.
uintptr_t g_skip_fn_addrs[8] = {0};
int g_skip_fn_count = 0;

// DELTA_PS5_GLYPHGUARD: recover the first-frame unbound-font null derefs in the
// UI/text renderer. Per entry: faulting rip, the GP register the instruction
// writes (zeroed so the code proceeds), and the instruction length. Fires only
// when the base register is null, so bound-font calls are untouched.

// DELTA_PS5_DCBWATCH call-order trace (see crash.h).
static constexpr int kOrderMax = 12;
static uintptr_t g_order_addrs[kOrderMax];
static const char* g_order_labels[kOrderMax];
static int g_order_count = 0;
static timespec g_order_start;
static uintptr_t g_ret_addrs[8];
static const char* g_ret_labels[8];
static bool g_ret_is_test[8];  // site was `test eax,eax`, not `mov ebx,eax`
static int g_ret_count = 0;

// DELTA_PS5_GLYPHGUARD call-skip (see crash.h): int3 planted over a blocking
// vtable-dispatch call; on hit, inject rax and step past the whole call insn.
static uintptr_t g_call_skip_addrs[8];
static long g_call_skip_vals[8];
static int g_call_skip_lens[8];
static int g_call_skip_count = 0;

// DELTA_FNWATCH hit counter (see crash.h).
static constexpr int kFnWatchMax = 16;
static uintptr_t g_fn_watch_addrs[kFnWatchMax];
static const char* g_fn_watch_labels[kFnWatchMax];
static base::Atomic<u64> g_fn_watch_hits[kFnWatchMax];
static int g_fn_watch_count = 0;

// DELTA_FNARGS pointer-chain probe (see crash.h).
static constexpr int kFnArgsMax = 8;
static constexpr int kFnArgsOffsMax = 6;
static constexpr int kFnArgsLogs = 8;  // logs per site; these sites are hot
static uintptr_t g_fn_args_addrs[kFnArgsMax];
static const char* g_fn_args_labels[kFnArgsMax];
static u64 g_fn_args_offs[kFnArgsMax][kFnArgsOffsMax];
static int g_fn_args_noffs[kFnArgsMax];
static base::Atomic<u64> g_fn_args_hits[kFnArgsMax];
static int g_fn_args_count = 0;

// Range currently write-protected for the write watch, armed either from the
// DELTA_GUEST_BRK_TRACE handler or standalone by DELTA_GUEST_WPROT.
static base::Atomic<uintptr_t> g_wprot_base{0};
static base::Atomic<size_t> g_wprot_len{0};
static base::Atomic<bool> g_wprot_regs{false};
static base::Atomic<bool> g_wprot_step{false};
static base::Atomic<uintptr_t> g_wprot_report_base{0};
static base::Atomic<size_t> g_wprot_report_len{0};
#if defined(__x86_64__)
static thread_local uintptr_t g_wprot_step_page = 0;
#endif

// DELTA_GUEST_WHIST census state (see StartWriteHist): re-arming watch, so a
// pool too large to log per write still yields who writes which parts over
// the whole run.
static constexpr size_t kWhistGranule = 16u << 20;
static constexpr int kWhistBuckets = 512;
static constexpr int kWhistSites = 32;
static base::Atomic<uintptr_t> g_whist_base{0};
static base::Atomic<size_t> g_whist_len{0};
static base::Atomic<u32> g_whist_bucket[kWhistBuckets];
static base::Atomic<uintptr_t> g_whist_site[kWhistSites];
static base::Atomic<uintptr_t> g_whist_site_caller[kWhistSites];
static base::Atomic<u32> g_whist_site_hits[kWhistSites];
// Faults whose guest instruction could not be established; reported alongside
// the sites so a census never reads as complete when some writers went unnamed.
static base::Atomic<u64> g_whist_unattributed{0};

// The guest instruction behind a memory-watch fault; false (rip 0) when it
// cannot be established. On x86 the context RIP already IS the guest RIP;
// under FEX on ARM map the host pc back through FEX. cpu::CurrentGuestRip()
// is NOT a fallback: block-accurate only, it attributed a title's store to
// libc memcpy and sent an investigation down a dead end. Unattributable
// faults must say so.
static bool WatchFaultGuestRip(void* ucv, uintptr_t& rip) {
  rip = 0;
#if defined(__x86_64__)
  rip = (uintptr_t)static_cast<ucontext_t*>(ucv)->uc_mcontext.gregs[REG_RIP];
  return rip != 0;
#elif defined(__aarch64__)
  rip = (uintptr_t)cpu::ReconstructGuestRip(
      static_cast<ucontext_t*>(ucv)->uc_mcontext.pc);
  return rip != 0;
#else
  (void)ucv;
  return false;
#endif
}

// Guest stack pointer at the fault, or 0. Lets a leaf writer (libc memcpy names
// no subsystem) be attributed to its caller.
static uintptr_t WatchFaultGuestRsp(void* ucv) {
#if defined(__x86_64__)
  return (uintptr_t)static_cast<ucontext_t*>(ucv)->uc_mcontext.gregs[REG_RSP];
#else
  (void)ucv;
  if (const u64* g = cpu::CurrentGuestGregs()) {
    enum { RAX, RCX, RDX, RBX, RSP };  // NOLINT(readability-identifier-naming)
    return (uintptr_t)g[RSP];
  }
  return 0;
#endif
}

#if defined(__aarch64__)
static void probeHandler(int, siginfo_t*, void* ucv) {
  uintptr_t rip = 0;
  const bool attributed = watchFaultGuestRip(ucv, rip);
  char sym[256];
  if (attributed)
    symbolize(rip, sym, sizeof(sym));
  else
    std::snprintf(sym, sizeof(sym), "<unattributed FEX host pc>");
  const auto host_pc = static_cast<ucontext_t*>(ucv)->uc_mcontext.pc;
  BASE_LOGI("probe", "tid={} host_pc={:#x} guest_pc={:#x} {}", (long)gettid(),
            (unsigned long long)host_pc, (unsigned long)rip, sym);
  if (const uintptr_t sp = watchFaultGuestRsp(ucv))
    GuestStackTraceFrom(sp, "probe", 8, (long)syscall(SYS_gettid));
  std::fflush(stderr);
}
#endif

// Reopen the one page a watch fault landed on so the guest retries: returning
// re-executes the faulting instruction, turning a one-shot trap into a trace.
// Without this the watch is not merely blind, it is FATAL on ARM.
static void ReopenWatchPage(uintptr_t at) {
  const long pgsz = sysconf(_SC_PAGESIZE);
  ::mprotect(reinterpret_cast<void*>(at & ~((uintptr_t)pgsz - 1)), (size_t)pgsz,
             PROT_READ | PROT_WRITE | PROT_EXEC);
}

static void ResumeWatchedWrite(uintptr_t at, void* ucv) {
  ReopenWatchPage(at);
#if defined(__x86_64__)
  if (g_wprot_step.load(base::memory_order_relaxed)) {
    const uintptr_t pgsz = (uintptr_t)sysconf(_SC_PAGESIZE);
    g_wprot_step_page = at & ~(pgsz - 1);
    static_cast<ucontext_t*>(ucv)->uc_mcontext.gregs[REG_EFL] |= 0x100;
  }
#else
  (void)ucv;
#endif
}

// Offered every signal before the crash reporter looks at it. True means a
// probe owned this trap and the handler must resume the guest.
bool OnSignal(int sig, siginfo_t* si, void* ucv) {
#if defined(__x86_64__)
  if (sig == SIGTRAP && ucv && g_wprot_step_page) {
    const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    ::mprotect(reinterpret_cast<void*>(g_wprot_step_page), pgsz, PROT_READ);
    static_cast<ucontext_t*>(ucv)->uc_mcontext.gregs[REG_EFL] &= ~0x100;
    g_wprot_step_page = 0;
    return true;
  }
#endif
  // Write census: same trap as the write watch, but only counts (per 16 MiB
  // bucket and per instruction) and reopens the page.
  if (sig == SIGSEGV && si && ucv && g_whist_len.load()) {
    const uintptr_t base = g_whist_base.load();
    const uintptr_t at = reinterpret_cast<uintptr_t>(si->si_addr);
    if (at >= base && at < base + g_whist_len.load()) {
      uintptr_t rip = 0;
      const bool attributed = WatchFaultGuestRip(ucv, rip);
      const size_t b = (at - base) / kWhistGranule;
      if (b < kWhistBuckets)
        g_whist_bucket[b].fetch_add(1, base::memory_order_relaxed);
      // Slot 0 doubles as "empty" in the site table, so an unattributable fault
      // is counted apart, never filed as a writer at rip 0.
      if (!attributed) {
        g_whist_unattributed.fetch_add(1, base::memory_order_relaxed);
      } else {
        for (int i = 0; i < kWhistSites; i++) {
          uintptr_t cur = g_whist_site[i].load(base::memory_order_relaxed);
          if (cur == rip) {
            g_whist_site_hits[i].fetch_add(1, base::memory_order_relaxed);
            break;
          }
          if (!cur && g_whist_site[i].compare_exchange_strong(cur, rip)) {
            // A leaf writer (libc memcpy) names no subsystem; keep one sample
            // of its return address so the report can name the caller too.
            if (const uintptr_t sp = WatchFaultGuestRsp(ucv))
              g_whist_site_caller[i].store(
                  *reinterpret_cast<const uintptr_t*>(sp),
                  base::memory_order_relaxed);
            g_whist_site_hits[i].fetch_add(1, base::memory_order_relaxed);
            break;
          }
        }
      }
      ReopenWatchPage(at);
      return true;
    }
  }
  // Write watch: the range is read-only, so a write faults here; name the
  // instruction, reopen that page and resume.
  if (sig == SIGSEGV && si && ucv && g_wprot_len.load()) {
    const uintptr_t base = g_wprot_base.load();
    const size_t len = g_wprot_len.load();
    const uintptr_t at = reinterpret_cast<uintptr_t>(si->si_addr);
    if (at >= base && at < base + len) {
      const uintptr_t report_base = g_wprot_report_base.load();
      const size_t report_len = g_wprot_report_len.load();
      const bool report = !g_wprot_step.load() ||
                          (at >= report_base && at < report_base + report_len);
      if (!report) {
        ResumeWatchedWrite(at, ucv);
        return true;
      }
      uintptr_t rip = 0;
      const bool attributed = WatchFaultGuestRip(ucv, rip);
      char sym[192];
      if (attributed)
        Symbolize(rip, sym, sizeof(sym));
      else {
        // Not guest code, so it is OUR code writing into the guest (HLE,
        // kernel, GPU readback); naming the host module is the point. Only the
        // leaf is named: backtrace() cannot unwind past the signal trampoline.
#if defined(__x86_64__)
        const uintptr_t host_pc = (uintptr_t)static_cast<ucontext_t*>(ucv)
                                      ->uc_mcontext.gregs[REG_RIP];
#elif defined(__aarch64__)
        const uintptr_t host_pc =
            (uintptr_t)static_cast<ucontext_t*>(ucv)->uc_mcontext.pc;
#else
        const uintptr_t host_pc = 0;
#endif
        Dl_info di{};
        const char* base = nullptr;
        if (dladdr(reinterpret_cast<void*>(host_pc), &di) && di.dli_fname)
          base = std::strrchr(di.dli_fname, '/');
        if (di.dli_sname)
          std::snprintf(sym, sizeof(sym), "HOST %s+%#lx", di.dli_sname,
                        (unsigned long)(host_pc - reinterpret_cast<uintptr_t>(
                                                      di.dli_saddr)));
        else if (di.dli_fname)
          std::snprintf(
              sym, sizeof(sym), "HOST %s+%#lx", base ? base + 1 : di.dli_fname,
              (unsigned long)(host_pc -
                              reinterpret_cast<uintptr_t>(di.dli_fbase)));
        else
          std::snprintf(sym, sizeof(sym), "HOST pc=%#lx (no symbol)",
                        (unsigned long)host_pc);
      }
      // Only a write-only watch can name the access from the protection alone;
      // a reads-too watch on ARM (no x86 error code) says "access".
#if defined(__x86_64__)
      const char* kind =
          (static_cast<ucontext_t*>(ucv)->uc_mcontext.gregs[REG_ERR] & 2)
              ? "write"
              : "read";
#else
      const char* kind = g_wprotRegs.load() ? "access" : "write";
#endif
      if (const uintptr_t probe = write_watch::ValueProbe();
          probe &&
          host_memory::IsMemoryRangeMapped(reinterpret_cast<void*>(probe), 8))
        BASE_LOGI("wprot", "{} {:#x} from {} | probe {:#x} = {:#x}", kind,
                  (unsigned long long)at, sym, (unsigned long)probe,
                  (unsigned long long)*reinterpret_cast<u64*>(probe));
      else
        BASE_LOGI("wprot", "{} {:#x} from {}", kind, (unsigned long long)at,
                  sym);
      // The writer of a descriptor/command ring is nearly always libc memcpy,
      // which names no subsystem, so the CALLER is the whole point. The guest
      // rsp qword is mid-body by fault time; scan the stack window for module
      // .text values, like the fatal reporter does.
      if (attributed) {
        if (const uintptr_t sp = WatchFaultGuestRsp(ucv))
          GuestStackTraceFrom(sp, "wprot", 4, (long)syscall(SYS_gettid));
      }
      // A consumer's other pointer (where it puts what it just read) is only
      // visible in its registers at the access; a value probe follows one word,
      // and the word's SOURCE (a memcpy's rsi) is the next hop.
      if (g_wprot_regs.load() || write_watch::ValueProbe()) {
        const u64* g = nullptr;
#if defined(__x86_64__)
        u64 xg[16];
        auto* hg = static_cast<ucontext_t*>(ucv)->uc_mcontext.gregs;
        const int kOrder[16] = {REG_RAX, REG_RCX, REG_RDX, REG_RBX,
                                REG_RSP, REG_RBP, REG_RSI, REG_RDI,
                                REG_R8,  REG_R9,  REG_R10, REG_R11,
                                REG_R12, REG_R13, REG_R14, REG_R15};
        for (int i = 0; i < 16; i++)
          xg[i] = (u64)hg[kOrder[i]];
        g = xg;
#else
        // FEX pins every guest GPR to a fixed host register, so the signal
        // context holds the exact values at the fault; the in-memory state lags
        // (written at block boundaries) and is only a fallback for non-JIT
        // faults.
        u64 sig_gregs[16];
        bool exact = cpu::GuestGregsFromSignal(ucv, sig_gregs);
        g = exact ? sig_gregs : cpu::CurrentGuestGregs();
#endif
        if (g) {
          // NOLINTBEGIN(readability-identifier-naming)
          enum {
            RAX,
            RCX,
            RDX,
            RBX,
            RSP,
            RBP,
            RSI,
            RDI,
            R8,
            R9,
            R10,
            R11,
            R12,
            R13,
            R14,
            R15
          };
          // NOLINTEND(readability-identifier-naming)
          BASE_LOGI("wprot",
                    "  ax={:x} bx={:x} cx={:x} dx={:x} si={:x} di={:x} "
                    "bp={:x} sp={:x} r8={:x} r9={:x} r10={:x} r11={:x} "
                    "r12={:x} r13={:x} r14={:x} r15={:x}{}",
                    (long)g[RAX], (long)g[RBX], (long)g[RCX], (long)g[RDX],
                    (long)g[RSI], (long)g[RDI], (long)g[RBP], (long)g[RSP],
                    (long)g[R8], (long)g[R9], (long)g[R10], (long)g[R11],
                    (long)g[R12], (long)g[R13], (long)g[R14], (long)g[R15],
#if defined(__x86_64__)
                    "");
#else
                    exact ? "" : "  (guest regs may lag the faulting insn)");
#endif
#if !defined(__x86_64__)
          // Chase the probed word upstream: with exact registers a block copy
          // reads as rdi=dest/rsi=src/rcx=len, so the same word in the SOURCE
          // is rsi + (probe - rdi). Re-aim there: one hop towards the producer.
          enum { C_RCX = 1, C_RSI = 6, C_RDI = 7 };
          const uintptr_t probe = write_watch::ValueProbe();
          if (exact && probe && write_watch::ChaseLeft()) {
            const u64 rdi = g[C_RDI], rsi = g[C_RSI], rcx = g[C_RCX];
            const bool looks_like_copy =
                rdi && rsi && rcx && probe >= rdi && probe - rdi < rcx;
            const uintptr_t src = rsi + (probe - rdi);
            if (looks_like_copy && host_memory::IsMemoryRangeMapped(
                                       reinterpret_cast<void*>(src), 8)) {
              write_watch::ChaseTook();
              BASE_LOGI("wprot",
                        "chase: probe {:#x} came from {:#x} (copy {:#x} <- "
                        "{:#x} len {:#x}); watching the source",
                        (unsigned long)probe, (unsigned long)src,
                        (unsigned long)rdi, (unsigned long)rsi,
                        (unsigned long)rcx);
              write_watch::SetValueProbe(src);
              write_watch::Arm(src, 8, 200);
            }
          }
#endif
        }
      }
      std::fflush(stderr);
      ResumeWatchedWrite(at, ucv);
      return true;
    }
  }
#if defined(__x86_64__)
  // DELTA_GUEST_BRK_TRACE: a RESUMABLE planted breakpoint (ud2 over a known
  // 3-byte instruction, e.g. V8 snapshot dispatch `mov %esi,%r12d`): emulate
  // it, log the bytecode in esi and the stream position at rdi+0x3c, and
  // return.
  if (sig == SIGILL && ucv) {
    const uintptr_t site = kBrkTrace;
    if (site) {
      auto* uc = static_cast<ucontext_t*>(ucv);
      auto* gr = uc->uc_mcontext.gregs;
      if ((uintptr_t)gr[REG_RIP] == site) {
        const u32 code = (u32)gr[REG_RSI] & 0xFF;
        const uintptr_t self = (uintptr_t)gr[REG_RDI];
        u32 pos = 0;
        if (self > 0x10000)
          pos = *reinterpret_cast<const u32*>(self + 0x3c);
        // One open()+write() per record to a dedicated fd: fprintf to stderr
        // loses records to mid-line interleaving from other threads.
        static const int kFd = ::open(
            "/tmp/bc_trace.txt", O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
        static u64 n = 0;
        if (kFd >= 0) {
          char buf[64];
          const int len = std::snprintf(buf, sizeof(buf), "%llu %02x %u\n",
                                        (unsigned long long)n++, code, pos);
          ssize_t ignored = ::write(kFd, buf, len);
          (void)ignored;
        }
        // DELTA_GUEST_BRK_ARM=<addr>: plant a ud2 when the traced stream
        // RESTARTS (a hot-path site always traps on the first of two
        // back-to-back deserializations; this reaches the second).
        // DELTA_GUEST_BRK_ARM_POS=<n>: wait until the restarted stream reaches
        // that position.
        static u32 last_pos = 0;
        static bool armed = false, restarted = false;
        if (pos < last_pos)
          restarted = true;
        if (kBrkArm && !armed && restarted && pos >= kBrkArmPos) {
          auto* at = reinterpret_cast<u8*>((uintptr_t)kBrkArm);
          at[0] = 0x0F;
          at[1] = 0x0B;
          armed = true;
        }
        // DELTA_GUEST_BRK_WPROT=<hex addr>:<hex size>: write-protect the range
        // at the same moment; a writer makes the write fault and the report
        // names it.
        static bool wprot_done = false;
        if (const char* wp = kBrkWprot;
            wp && !wprot_done && pos >= kBrkArmPos) {
          wprot_done = true;
          const uintptr_t a = std::strtoull(wp, nullptr, 16);
          const char* c = std::strchr(wp, ':');
          const size_t n = c ? std::strtoull(c + 1, nullptr, 16) : 0x40000;
          if (::mprotect(reinterpret_cast<void*>(a), n, PROT_READ) == 0) {
            g_wprot_base = a;
            g_wprot_len = n;
          }
        }
        last_pos = pos;
        gr[REG_R12] = (u32)gr[REG_RSI];  // mov %esi,%r12d (zero-extends)
        gr[REG_RIP] = site + 3;
        return true;
      }
    }
  }
  if (sig == SIGTRAP && g_ret_count && ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    auto* gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_ret_count; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_ret_addrs[i] + 1)
        continue;
      u32 eax = (u32)gr[REG_RAX];
      // DELTA_PS5_DCBFORCE: force a failing graphics-init sub-call to report
      // SCE_OK so init 0x69e720 completes and the engine creates its
      // DrawCommandBuffer, measuring how far boot gets.
      static const int kForce = kPs5Dcbforce;
      if (kForce && eax) {
        gr[REG_RAX] = 0;
        eax = 0;
      }
      char m[128];
      int n = std::snprintf(m, sizeof(m), "[ret] %s eax=%#x\n", g_ret_labels[i],
                            eax);
      if (n > 0) {
        ssize_t w = write(2, m, (size_t)n);
        (void)w;
      }
      if (g_ret_is_test[i]) {
        // emulate `test eax,eax`: CF/OF cleared, ZF/SF/PF from the result
        greg_t fl = gr[REG_EFL] & ~(greg_t)(0x8d5);  // CF PF AF ZF SF OF
        if (eax == 0)
          fl |= 0x40;
        if (eax & 0x80000000u)
          fl |= 0x80;
        if (__builtin_parity(eax & 0xff) == 0)
          fl |= 0x4;
        gr[REG_EFL] = fl;
      } else {
        gr[REG_RBX] = eax;  // emulate `mov ebx,eax` (zero-extends)
      }
      gr[REG_RIP] = g_ret_addrs[i] + 2;  // resume past the 2-byte insn
      return true;
    }
  }
  if (sig == SIGTRAP && g_call_skip_count && ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    auto* gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_call_skip_count; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_call_skip_addrs[i] + 1)
        continue;
      static bool s_seen[8] = {};
      if (!s_seen[i]) {
        s_seen[i] = true;
        char m[64];
        int n = std::snprintf(m, sizeof(m), "[callskip] #%d fired -> rax=%ld\n",
                              i, g_call_skip_vals[i]);
        if (n > 0) {
          ssize_t w = write(2, m, (size_t)n);
          (void)w;
        }
      }
      gr[REG_RAX] = g_call_skip_vals[i];  // inject the blocked call's return
      gr[REG_RIP] =
          g_call_skip_addrs[i] + g_call_skip_lens[i];  // step past the call
      return true;
    }
  }
  if (sig == SIGTRAP && g_order_count && ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    auto* gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_order_count; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_order_addrs[i] + 1)
        continue;
      timespec t{};
      clock_gettime(CLOCK_MONOTONIC, &t);
      long ms = (t.tv_sec - g_order_start.tv_sec) * 1000 +
                (t.tv_nsec - g_order_start.tv_nsec) / 1000000;
      uintptr_t rsp = (uintptr_t)gr[REG_RSP];
      uintptr_t caller = rsp >= 0x10000 ? *reinterpret_cast<u64*>(rsp) : 0;
      char csym[200];
      Symbolize(caller, csym, sizeof(csym));
      char m[320];
      int n = std::snprintf(m, sizeof(m), "[order t=%ldms tid=%ld] %s  <- %s\n",
                            ms, (long)gettid(), g_order_labels[i], csym);
      if (n > 0) {
        ssize_t w = write(2, m, (size_t)n);
        (void)w;
      }
      gr[REG_RSP] -= 8;  // emulate the displaced `push rbp`
      *reinterpret_cast<u64*>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_fn_args_count && ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    auto* gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_fn_args_count; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_fn_args_addrs[i] + 1)
        continue;
      if (g_fn_args_hits[i].fetch_add(1, base::memory_order_relaxed) <
          kFnArgsLogs) {
        char m[512];
        int n = std::snprintf(
            m, sizeof(m), "[fnargs] %s rdi=%#lx rsi=%#lx rdx=%#lx rcx=%#lx",
            g_fn_args_labels[i], (unsigned long)gr[REG_RDI],
            (unsigned long)gr[REG_RSI], (unsigned long)gr[REG_RDX],
            (unsigned long)gr[REG_RCX]);
        uintptr_t p = (uintptr_t)gr[REG_RDI];
        for (int o = 0; o < g_fn_args_noffs[i] && n > 0; o++) {
          uintptr_t at = p + g_fn_args_offs[i][o];
          if (!host_memory::IsMemoryRangeMapped(reinterpret_cast<void*>(at),
                                                8)) {
            n += std::snprintf(m + n, sizeof(m) - n, " +%#lx=<unmapped>",
                               (unsigned long)g_fn_args_offs[i][o]);
            p = 0;
            break;
          }
          p = *reinterpret_cast<uintptr_t*>(at);
          n += std::snprintf(m + n, sizeof(m) - n, " +%#lx->%#lx",
                             (unsigned long)g_fn_args_offs[i][o],
                             (unsigned long)p);
        }
        if (n > 0 && p &&
            host_memory::IsMemoryRangeMapped(reinterpret_cast<void*>(p), 64)) {
          n += std::snprintf(m + n, sizeof(m) - n, " *=");
          const auto* w = reinterpret_cast<const u32*>(p);
          for (int j = 0; j < 12 && n > 0 && n < (int)sizeof(m) - 12; j++)
            n += std::snprintf(m + n, sizeof(m) - n, " %08x", w[j]);
        }
        if (n > 0 && n < (int)sizeof(m) - 1) {
          m[n++] = '\n';
          ssize_t w = write(2, m, (size_t)n);
          (void)w;
        }
      }
      gr[REG_RSP] -= 8;  // emulate the displaced `push rbp`
      *reinterpret_cast<u64*>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_fn_watch_count && ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    auto* gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_fn_watch_count; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_fn_watch_addrs[i] + 1)
        continue;
      g_fn_watch_hits[i].fetch_add(1, base::memory_order_relaxed);
      // Emulate the displaced `push rbp`; RIP stays at addr+1 (the `mov
      // rbp,rsp`).
      gr[REG_RSP] -= 8;
      *reinterpret_cast<u64*>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_skip_fn_count && ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    auto* gr = uc->uc_mcontext.gregs;
    for (int si2 = 0; si2 < g_skip_fn_count; si2++) {
      if ((uintptr_t)gr[REG_RIP] == g_skip_fn_addrs[si2] + 1) {
        uintptr_t rsp = (uintptr_t)gr[REG_RSP];
        gr[REG_RIP] = *reinterpret_cast<u64*>(rsp);  // return addr
        gr[REG_RSP] = rsp + 8;
        gr[REG_RAX] = 0;
        return true;
      }
    }
  }
  if (sig == SIGTRAP && g_rdoff_addr && ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    auto* gr = uc->uc_mcontext.gregs;
    if ((uintptr_t)gr[REG_RIP] == g_rdoff_addr + 1) {
      u32 fd = (u32)gr[REG_RSI];
      bool mf = (fd < 8192 && g_manifest_fd[fd]);
      if (kRdoffTrace) {
        char m[128];
        int n = std::snprintf(
            m, sizeof(m),
            "[rdoff] fd=%u off=%lld nbytes=%lld buf=%llx manifest=%d\n", fd,
            (long long)(i32)gr[REG_RDX], (long long)(i32)gr[REG_RCX],
            (unsigned long long)gr[REG_R8], mf);
        if (n > 0) {
          ssize_t w = write(2, m, (size_t)n);
          (void)w;
        }
      }
      // DELTA_RDOFF_NOFIX: observe-only (log requests, don't rewrite offsets).
      if (mf && !kRdoffNofix)
        gr[REG_RDX] = 0;  // force manifest read offset to 0 (read from start)
      gr[REG_RSP] -= 8;   // emulate push rbp
      *reinterpret_cast<u64*>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_hdr_trace_count && ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    auto* gr = uc->uc_mcontext.gregs;
    bool hit = false;
    for (int hi = 0; hi < g_hdr_trace_count; hi++)
      if ((uintptr_t)gr[REG_RIP] == g_hdr_trace_addrs[hi] + 1) {
        hit = true;
        break;
      }
    if (hit) {
      u64 parent = (u64)gr[REG_RDI];
      u64 hdr = 0, obj = 0;
      u32 magic = 0, cnt = 0;
      char nm[48] = {0};
      if (parent >= 0x10000) {
        hdr = *reinterpret_cast<u64*>(parent + 0x8);
        obj = *reinterpret_cast<u64*>(parent + 0x10);
        // DELTA_HDR_WAIT: if the manifest header isn't filled yet (magic !=
        // "TAFS"), block this consumer thread to let the worker's read+copy
        // complete.
        if (kHdrWait && hdr >= 0x10000) {
          for (int i = 0; i < 2000; i++) {
            if (*reinterpret_cast<volatile u32*>(hdr) == 0x53464154u)
              break;
            timespec ts{0, 200000};  // 0.2ms
            nanosleep(&ts, nullptr);
          }
        }
        if (hdr >= 0x10000) {
          magic = *reinterpret_cast<u32*>(hdr);
          cnt = *reinterpret_cast<u32*>(hdr + 0xc);
        }
        if (obj >= 0x10000) {
          const char* s = reinterpret_cast<const char*>(obj + 0x5c);
          int j = 0;
          for (; j < 47 && s[j] >= 0x20 && s[j] <= 0x7e; j++)
            nm[j] = s[j];
          nm[j] = 0;
        }
        // DELTA_HDR_FILL: bypass the racy async manifest reader by copying the
        // cached manifest bytes straight into the header buffer.
        if (kHdrFill && hdr >= 0x10000 && nm[0]) {
          auto* h = reinterpret_cast<u8*>(hdr);
          // The header buffer [parent+0x8] is filesize-sized (0x605e30), so
          // fill the WHOLE manifest at every consumer hook: header AND entry
          // table must be correct or downstream processing reads garbage.
          if (const auto* mf = vfs::GetCachedFile(nm)) {
            std::memcpy(h, mf->data(), mf->size());
          } else {
            // Missing archive: write a valid empty TAFS header (count=0)
            // instead of a garbage count -> OOM.
            std::memset(h, 0, 0x34);
            h[0] = 'T';
            h[1] = 'A';
            h[2] = 'F';
            h[3] = 'S';
            *reinterpret_cast<u32*>(h + 4) = 3;     // version
            *reinterpret_cast<u32*>(h + 0x10) = 7;  // strlen("orbis-w")
            std::memcpy(h + 0x14, "orbis-w", 7);
          }
          magic = *reinterpret_cast<u32*>(h);
          cnt = *reinterpret_cast<u32*>(h + 0xc);
        }
      }
      char m[176];
      int n = std::snprintf(
          m, sizeof(m),
          "[hdr] t=%ld name=\"%s\" hdr=%#llx magic=%08x count=%u\n",
          (long)gettid(), nm, (unsigned long long)hdr, magic, cnt);
      if (n > 0) {
        ssize_t w = write(2, m, (size_t)n);
        (void)w;
      }
      static bool once = false;
      if (!once && obj >= 0x10000) {
        once = true;
        u64 vt = *reinterpret_cast<u64*>(obj);
        u64 m58 = (vt >= 0x10000) ? *reinterpret_cast<u64*>(vt + 0x58) : 0;
        // 0x608390-style forward: inner obj = [obj+0x8], real method =
        // inner.vt[0x58].
        u64 inner = *reinterpret_cast<u64*>(obj + 0x8);
        u64 ivt = (inner >= 0x10000) ? *reinterpret_cast<u64*>(inner) : 0;
        u64 im58 = (ivt >= 0x10000) ? *reinterpret_cast<u64*>(ivt + 0x58) : 0;
        u64 im30 = (ivt >= 0x10000) ? *reinterpret_cast<u64*>(ivt + 0x30) : 0;
        char v[224];
        int vn =
            std::snprintf(v, sizeof(v),
                          "[hdr] obj=%#llx vt=%#llx m58=%#llx | inner=%#llx "
                          "ivt=%#llx im30=%#llx im58=%#llx\n",
                          (unsigned long long)obj, (unsigned long long)vt,
                          (unsigned long long)m58, (unsigned long long)inner,
                          (unsigned long long)ivt, (unsigned long long)im30,
                          (unsigned long long)im58);
        if (vn > 0) {
          ssize_t w = write(2, v, (size_t)vn);
          (void)w;
        }
      }
      gr[REG_RSP] -= 8;
      *reinterpret_cast<u64*>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_fatal_trace_addr && ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    auto* gr = uc->uc_mcontext.gregs;
    if ((uintptr_t)gr[REG_RIP] == g_fatal_trace_addr + 1) {
      u64 fmt = (u64)gr[REG_RDI];
      u64 caller = 0;
      uintptr_t rsp = (uintptr_t)gr[REG_RSP];
      if (rsp >= 0x10000)
        caller = *reinterpret_cast<u64*>(rsp);
      char msg[256] = {0};
      if (fmt >= 0x10000) {
        const char* s = reinterpret_cast<const char*>(fmt);
        int j = 0;
        for (; j < 255 && s[j]; j++)
          msg[j] = (s[j] >= 0x20 || s[j] == '\n') ? s[j] : '.';
        msg[j] = 0;
      }
      char out[480];
      int n = std::snprintf(out, sizeof(out),
                            "[FATAL] caller=%#llx rsi=%#llx rdx=%#llx "
                            "rcx=%#llx\n        fmt=\"%s\"\n",
                            (unsigned long long)caller,
                            (unsigned long long)gr[REG_RSI],
                            (unsigned long long)gr[REG_RDX],
                            (unsigned long long)gr[REG_RCX], msg);
      if (n > 0) {
        ssize_t w = write(2, out, (size_t)n);
        (void)w;
      }
      gr[REG_RSP] -= 8;
      *reinterpret_cast<u64*>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_cnt_trace_addr && ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    auto* gr = uc->uc_mcontext.gregs;
    if ((uintptr_t)gr[REG_RIP] == g_cnt_trace_addr + 1) {
      u64 obj = (u64)gr[REG_RDI];
      u32 cnt = 0;
      char nm[48] = {0};
      if (obj >= 0x10000) {
        cnt = *reinterpret_cast<u32*>(obj + 0x30);
        const char* s = reinterpret_cast<const char*>(obj + 0x5c);
        int j = 0;
        for (; j < 47 && s[j] >= 0x20 && s[j] <= 0x7e; j++)
          nm[j] = s[j];
        nm[j] = 0;
      }
      char m[128];
      int n = std::snprintf(m, sizeof(m),
                            "[cnt] obj=%llx count=%u (%#x) name=\"%s\"\n",
                            (unsigned long long)obj, cnt, cnt, nm);
      if (n > 0) {
        ssize_t w = write(2, m, (size_t)n);
        (void)w;
      }
      // DELTA_CNT_CLAMP: force an absurd (uninitialised) entry count to 0 so
      // the entry-table alloc is tiny and boot proceeds past the OOM.
      if (kCntClamp && obj >= 0x10000 && cnt > 0x100000)
        *reinterpret_cast<u32*>(obj + 0x30) = 0;
      gr[REG_RSP] -= 8;
      *reinterpret_cast<u64*>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  // Allocator-trace trap: handle first so it neither marks s_dumping nor floods
  // the entry marker. After int3 the RIP sits one byte past the hooked entry.
  if (sig == SIGTRAP && g_heap_prof_addr && ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    auto* gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_heap_prof_hook_count; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_heap_prof_hooks[i] + 1)
        continue;
      uintptr_t rsp = (uintptr_t)gr[REG_RSP];
      uintptr_t caller = rsp >= 0x10000 ? *reinterpret_cast<u64*>(rsp) : 0;
      const u64 size = g_heap_prof_count_only[i] ? 1u : (u64)gr[REG_RDI];
      // The guest's fs base is NOT the thread's real fs (the lifter rewrites
      // guest fs accesses); ask the backend for the base the guest's fs:0
      // resolves to.
      HeapProfRecord(caller, size,
                     g_heap_prof_scope_slot &&
                         HeapProfScopeDepth(cpu::ThreadFsBase()) == 0);
      g_heap_prof_hook_bytes[i].fetch_add(size, base::memory_order_relaxed);
      g_heap_prof_hook_calls[i].fetch_add(1, base::memory_order_relaxed);
      gr[REG_RSP] -= 8;  // emulate the displaced `push rbp`
      *reinterpret_cast<u64*>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_alloc_trace_addr && ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    auto* gr = uc->uc_mcontext.gregs;
    if ((uintptr_t)gr[REG_RIP] == g_alloc_trace_addr + 1) {
      u64 size = (u64)gr[REG_RSI];
      if (size >= g_alloc_trace_min) {
        char m[96];
        int n = std::snprintf(m, sizeof(m),
                              "[alloc] %llu bytes (%.1f MB) heap=%llx\n",
                              (unsigned long long)size, size / 1048576.0,
                              (unsigned long long)gr[REG_RDI]);
        if (n > 0) {
          ssize_t w = write(2, m, (size_t)n);
          (void)w;
        }
        // Scan the guest stack for return addresses in a module .text to show
        // which code computed this (garbage) size.
        uintptr_t rsp = (uintptr_t)gr[REG_RSP];
        if (rsp >= 0x10000) {
          auto* sp = reinterpret_cast<uintptr_t*>(rsp);
          int shown = 0;
          for (int i = 0; i < 256 && shown < 8; i++) {
            char sym[200];
            Symbolize(sp[i], sym, sizeof(sym));
            if (std::strstr(sym, "(.text)")) {
              char l[256];
              int ln =
                  std::snprintf(l, sizeof(l), "  sp+%-4x %s\n", i * 8, sym);
              if (ln > 0) {
                ssize_t w = write(2, l, (size_t)ln);
                (void)w;
              }
              shown++;
            }
          }
        }
      }
      gr[REG_RSP] -= 8;  // emulate the displaced `push rbp`
      *reinterpret_cast<u64*>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;  // resume at addr+1 (the mov rbp,rsp that follows)
    }
  }
#endif
  return false;
}

// The crash reporter is about to print a fatal dump; flush anything a probe has
// accumulated that would otherwise be lost with the process.
void OnFatal() {
  if (g_heap_prof_addr)
    HeapProfDump();
}

void InstallThreadProbeHandler(struct sigaction& pa) {
#if defined(__x86_64__) || defined(__aarch64__)
  pa.sa_sigaction = ProbeHandler;
#else
  (void)pa;
#endif
}

void SetAllocTrace(uintptr_t addr, u64 min_size) {
  g_alloc_trace_addr = addr;
  if (min_size)
    g_alloc_trace_min = min_size;
}

void SetCntTrace(uintptr_t addr) {
  g_cnt_trace_addr = addr;
}
void SetFatalTrace(uintptr_t addr) {
  g_fatal_trace_addr = addr;
}
void SetHdrTrace(uintptr_t addr) {
  if (g_hdr_trace_count < 8)
    g_hdr_trace_addrs[g_hdr_trace_count++] = addr;
}
void SetRdoffFix(uintptr_t addr) {
  g_rdoff_addr = addr;
}
void SetSkipFn(uintptr_t addr) {
  if (g_skip_fn_count < 8)
    g_skip_fn_addrs[g_skip_fn_count++] = addr;
}
void SetCallSkip(uintptr_t addr, long rax_val, int insn_len) {
#if defined(__x86_64__)
  if (g_call_skip_count >= 8)
    return;
  g_call_skip_addrs[g_call_skip_count] = addr;
  g_call_skip_vals[g_call_skip_count] = rax_val;
  g_call_skip_lens[g_call_skip_count] = insn_len;
  g_call_skip_count++;
#else
  (void)addr;
  (void)raxVal;
  (void)insnLen;
#endif
}
void SetOrderTrace(uintptr_t addr, const char* label) {
  if (g_order_count >= kOrderMax)
    return;
  if (g_order_count == 0)
    clock_gettime(CLOCK_MONOTONIC, &g_order_start);
  g_order_addrs[g_order_count] = addr;
  g_order_labels[g_order_count] = label;
  g_order_count++;
}
void SetRetTrace(uintptr_t addr, const char* label, bool is_test) {
  if (g_ret_count < 8) {
    g_ret_addrs[g_ret_count] = addr;
    g_ret_labels[g_ret_count] = label;
    g_ret_is_test[g_ret_count] = is_test;
    g_ret_count++;
  }
}

void SetFnWatch(uintptr_t addr, const char* label) {
  if (g_fn_watch_count >= kFnWatchMax)
    return;
  g_fn_watch_addrs[g_fn_watch_count] = addr;
  g_fn_watch_labels[g_fn_watch_count] = label;
  g_fn_watch_hits[g_fn_watch_count].store(0, base::memory_order_relaxed);
  g_fn_watch_count++;
}

void SetFnArgs(uintptr_t addr,
               const char* label,
               const u64* offsets,
               int noffsets) {
  if (g_fn_args_count >= kFnArgsMax)
    return;
  if (noffsets > kFnArgsOffsMax)
    noffsets = kFnArgsOffsMax;
  g_fn_args_addrs[g_fn_args_count] = addr;
  g_fn_args_labels[g_fn_args_count] = label;
  g_fn_args_noffs[g_fn_args_count] = noffsets;
  for (int i = 0; i < noffsets; i++)
    g_fn_args_offs[g_fn_args_count][i] = offsets[i];
  g_fn_args_hits[g_fn_args_count].store(0, base::memory_order_relaxed);
  g_fn_args_count++;
}

// DELTA_GUEST_WPROT=<hex addr>:<hex bytes>[:<ms>]: name every writer of a guest
// range. Waits until mapped, makes it read-only, and the SIGSEGV path reports
// the faulting instruction and reopens the page. The trap RE-ARMS every `ms`: a
// single report cannot tell a one-time initialiser from a per-frame producer,
// and re-arming costs about one fault per page per interval.
void StartWriteWatch(uintptr_t addr,
                     size_t bytes,
                     unsigned every_ms,
                     bool trap_reads,
                     bool single_step) {
  if (!addr || !bytes)
    return;
#if !defined(__x86_64__)
  singleStep = false;
#endif
  const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
  const uintptr_t base = addr & ~((uintptr_t)pgsz - 1);
  const size_t span = (addr + bytes - base + pgsz - 1) & ~(pgsz - 1);
  if (single_step) {
    g_wprot_base = base;
    g_wprot_len = span;
    g_wprot_regs = trap_reads;
    g_wprot_step = true;
    g_wprot_report_base = addr;
    g_wprot_report_len = bytes;
    if (::mprotect(reinterpret_cast<void*>(base), span,
                   trap_reads ? PROT_NONE : PROT_READ) == 0) {
      BASE_LOGI("wprot",
                "single-stepping {:#x}+{:#x}, reporting {:#x}+{:#x} ({})",
                (unsigned long)base, (unsigned long)span, (unsigned long)addr,
                (unsigned long)bytes, trap_reads ? "reads+writes" : "writes");
    }
    return;
  }
  guest::SpawnThread("probe_trap", [addr, bytes, every_ms, trap_reads] {
    const long pgsz = sysconf(_SC_PAGESIZE);
    const uintptr_t base = addr & ~((uintptr_t)pgsz - 1);
    const size_t span =
        (addr + bytes - base + (size_t)pgsz - 1) & ~((size_t)pgsz - 1);
    bool announced = false;
    for (;;) {
      guest::SleepForMilliseconds(every_ms);
      unsigned char vec = 0;
      if (mincore(reinterpret_cast<void*>(base), 1, &vec) != 0)
        continue;  // not mapped yet
      if (::mprotect(reinterpret_cast<void*>(base), span,
                     trap_reads ? PROT_NONE : PROT_READ) != 0)
        continue;
      g_wprot_base = base;
      g_wprot_len = span;
      g_wprot_regs = trap_reads;
      g_wprot_step = false;
      g_wprot_report_base = base;
      g_wprot_report_len = span;
      if (!announced) {
        announced = true;
        BASE_LOGI("wprot", "watching {:#x}+{:#x} ({}), re-armed every {}ms",
                  (unsigned long)base, (unsigned long)span,
                  trap_reads ? "reads+writes" : "writes", every_ms);
      }
    }
  });
}

// DELTA_GUEST_WHIST=<hex addr>:<hex bytes>[:<ms>]: write census over a pool too
// big to watch per write; re-arms read-only every `ms`, so each report says
// which 16 MiB slices were written in that window and by which instructions.
void StartWriteHist(uintptr_t addr, size_t bytes, unsigned every_ms) {
  if (!addr || !bytes)
    return;
  guest::SpawnThread("probe_trap", [addr, bytes, every_ms] {
    const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t base = addr & ~(pgsz - 1);
    const size_t span = (bytes + pgsz - 1) & ~(pgsz - 1);
    for (;;) {
      guest::SleepForMilliseconds(200);
      unsigned char vec = 0;
      if (mincore(reinterpret_cast<void*>(base), 1, &vec) == 0)
        break;
    }
    g_whist_base = base;
    g_whist_len = span;
    BASE_LOGI("whist", "census on {:#x}+{:#x} every {}ms", (unsigned long)base,
              (unsigned long)span, every_ms);
    unsigned since_report = 0;
    for (;;) {
      ::mprotect(reinterpret_cast<void*>(base), span, PROT_READ);
      guest::SleepForMilliseconds(every_ms);
      if ((since_report += every_ms) < 4000)
        continue;
      since_report = 0;
      base::String map;
      for (size_t off = 0; off < span; off += kWhistGranule) {
        const u32 n = g_whist_bucket[off / kWhistGranule].load();
        map += n == 0 ? '_' : n < 10 ? '.' : n < 100 ? '+' : '#';
      }
      BASE_LOGI("whist", "{}", map.c_str());
      for (int i = 0; i < kWhistSites; i++) {
        const uintptr_t rip = g_whist_site[i].load();
        if (!rip)
          break;
        char sym[192], csym[192];
        Symbolize(rip, sym, sizeof(sym));
        Symbolize(g_whist_site_caller[i].load(), csym, sizeof(csym));
        BASE_LOGI("whist", "  {:8} {} <- {}", g_whist_site_hits[i].load(), sym,
                  csym);
      }
      // Never let the site list read as the complete set of writers when some
      // faults could not be attributed to a guest instruction.
      if (const u64 unattributed = g_whist_unattributed.load())
        BASE_LOGI("whist", "  {:8} <unattributed>",
                  (unsigned long long)unattributed);
    }
  });
}

// DELTA_GUEST_POPCNT=<hex addr>:<hex bytes>: population count of a guest bitmap
// every 2s (a title allocator's free/used map); "does it drain, or was it never
// filled" needs the interval, not one dump.
void StartPopcntPrinter(uintptr_t addr, size_t bytes, unsigned every_ms) {
  if (!addr || !bytes)
    return;
  guest::SpawnThread("probe_trap", [addr, bytes, every_ms] {
    const long pgsz = sysconf(_SC_PAGESIZE);
    for (;;) {
      guest::SleepForMilliseconds(every_ms);
      unsigned char vec = 0;
      if (mincore(reinterpret_cast<void*>(addr & ~((uintptr_t)pgsz - 1)), 1,
                  &vec) != 0)
        continue;
      const auto* w = reinterpret_cast<const u64*>(addr);
      u64 set = 0;
      long first = -1, last = -1;
      for (size_t i = 0; i < bytes / 8; i++) {
        if (!w[i])
          continue;
        set += (u64)__builtin_popcountll(w[i]);
        if (first < 0)
          first = (long)(i * 64 + __builtin_ctzll(w[i]));
        last = (long)(i * 64 + 63 - __builtin_clzll(w[i]));
      }
      BASE_LOGI("popcnt", "{:#x}: {} / {} set, first={} last={}",
                (unsigned long)addr, (unsigned long long)set, bytes * 8, first,
                last);
    }
  });
}

// DELTA_GUEST_SUMWATCH=<slot>:<off>:<stride>:<count>[:<ms>] (hex but count):
// dereference guest pointer SLOT and report the u32 counters obj+off+i*stride
// and their sum; an engine's "work remaining" is usually such a counter set.
void StartSumWatchPrinter(uintptr_t slot,
                          size_t off,
                          size_t stride,
                          int count,
                          unsigned every_ms) {
  if (!slot || count <= 0 || count > 32)
    return;
  guest::SpawnThread("probe_trap", [slot, off, stride, count, every_ms] {
    const long pgsz = sysconf(_SC_PAGESIZE);
    auto readable = [pgsz](uintptr_t a) {
      unsigned char v = 0;
      return a >= 0x10000 &&
             mincore(reinterpret_cast<void*>(a & ~((uintptr_t)pgsz - 1)), 1,
                     &v) == 0;
    };
    for (;;) {
      guest::SleepForMilliseconds(every_ms);
      if (!readable(slot))
        continue;
      const uintptr_t obj = *reinterpret_cast<const uintptr_t*>(slot);
      if (!readable(obj))
        continue;
      base::String line;
      base::FormatTo(line, "{:#x}:", (unsigned long)obj);
      u64 sum = 0;
      for (int i = 0; i < count; i++) {
        const u32 v =
            *reinterpret_cast<const u32*>(obj + off + (size_t)i * stride);
        sum += v;
        base::FormatTo(line, " {}", v);
      }
      BASE_LOGI("sumwatch", "{} = {}", line.c_str(), (unsigned long long)sum);
    }
  });
}

// DELTA_POOLMAP=<hex addr>:<hex bytes>[:<ms>]: survey a multi-GB pool without
// touching it: mincore shows which pages the guest actually faulted in, so an
// unfilled region shows as a hole; only resident pages get the non-zero test.
void StartPoolMap(uintptr_t addr, size_t bytes, unsigned every_ms) {
  if (!addr || !bytes)
    return;
  guest::SpawnThread("probe_trap", [addr, bytes, every_ms] {
    const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t base = addr & ~(pgsz - 1);
    const size_t span = (bytes + pgsz - 1) & ~(pgsz - 1);
    const size_t npages = span / pgsz;
    const size_t granule = 16u << 20;  // one report column
    const size_t pages_per_granule = granule / pgsz;
    base::Vector<unsigned char> vec(npages);
    for (;;) {
      guest::SleepForMilliseconds(every_ms);
      if (mincore(reinterpret_cast<void*>(base), span, vec.data()) != 0)
        continue;
      size_t resident = 0, nonzero = 0;
      base::String map;
      for (size_t g = 0; g * pages_per_granule < npages; g++) {
        size_t res = 0, nz = 0;
        for (size_t i = g * pages_per_granule;
             i < npages && i < (g + 1) * pages_per_granule; i++) {
          if (!(vec[i] & 1))
            continue;
          res++;
          const auto* w = reinterpret_cast<const u64*>(base + i * pgsz);
          for (size_t j = 0; j < pgsz / 8; j++)
            if (w[j]) {
              nz++;
              break;
            }
        }
        resident += res;
        nonzero += nz;
        map += nz ? '#' : res ? '.' : '_';
      }
      BASE_LOGI("poolmap", "{:#x}+{:#x} resident={}/{} pages nonzero={}",
                (unsigned long)base, (unsigned long)span, resident, npages,
                nonzero);
      BASE_LOGI("poolmap", "{}", map.c_str());
    }
  });
}

// DELTA_POOLMAP=all[:<ms>]: the same survey over every guest mapping; a
// "wrote a gigabyte somewhere" question needs the whole address space.
void StartPoolCensus(unsigned every_ms) {
  guest::SpawnThread("probe_trap", [every_ms] {
    const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    for (;;) {
      guest::SleepForMilliseconds(every_ms);
      FILE* f = std::fopen("/proc/self/maps", "r");
      if (!f)
        return;
      char line[512];
      base::Vector<unsigned char> vec;
      BASE_LOGI("census", "---- guest mappings ----");
      while (std::fgets(line, sizeof line, f)) {
        unsigned long lo = 0, hi = 0;
        char perm[8] = {0};
        if (std::sscanf(line, "%lx-%lx %4s", &lo, &hi, perm) != 3)
          continue;
        if (lo < 0x8000000000ull || perm[0] != 'r')
          continue;
        const size_t span = hi - lo;
        if (span < (1u << 20))
          continue;
        vec.assign(span / pgsz, 0);
        if (mincore(reinterpret_cast<void*>(lo), span, vec.data()) != 0)
          continue;
        size_t res = 0, nz = 0;
        for (size_t i = 0; i < vec.size(); i++) {
          if (!(vec[i] & 1))
            continue;
          res++;
          const auto* w = reinterpret_cast<const u64*>(lo + i * pgsz);
          for (size_t j = 0; j < pgsz / 8; j++)
            if (w[j]) {
              nz++;
              break;
            }
        }
        BASE_LOGI("census",
                  "{:012x}+{:09x} {:.1f} MB resident={:.1f} MB "
                  "nonzero={:.1f} MB",
                  lo, span, span / 1048576.0, res * pgsz / 1048576.0,
                  nz * pgsz / 1048576.0);
      }
      std::fclose(f);
    }
  });
}

// DELTA_MEMDUMP=<hex addr>:<hex bytes>:<ms>:<path>[,...]: snapshot a guest
// range to a file; only resident pages are read, holes come out as zeros.
void StartMemDump(uintptr_t addr,
                  size_t bytes,
                  unsigned after_ms,
                  const char* path) {
  if (!addr || !bytes || !path)
    return;
  base::String out(path);
  guest::SpawnThread("probe_trap", [addr, bytes, after_ms, out] {
    guest::SleepForMilliseconds(after_ms);
    const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t base = addr & ~(pgsz - 1);
    const size_t span = (bytes + pgsz - 1) & ~(pgsz - 1);
    base::Vector<unsigned char> vec(span / pgsz);
    if (mincore(reinterpret_cast<void*>(base), span, vec.data()) != 0) {
      BASE_LOGI("memdump", "{:#x} not mapped", (unsigned long)base);
      return;
    }
    FILE* f = std::fopen(out.c_str(), "wb");
    if (!f)
      return;
    base::Vector<unsigned char> zero(pgsz, 0);
    size_t resident = 0;
    for (size_t i = 0; i < vec.size(); i++) {
      if (vec[i] & 1) {
        resident++;
        std::fwrite(reinterpret_cast<const void*>(base + i * pgsz), 1, pgsz, f);
      } else {
        std::fwrite(zero.data(), 1, pgsz, f);
      }
    }
    std::fclose(f);
    BASE_LOGI("memdump", "{:#x}+{:#x} -> {} ({}/{} resident pages)",
              (unsigned long)base, (unsigned long)span, out.c_str(), resident,
              vec.size());
  });
}

void StartFnWatchPrinter() {
  static base::Atomic<bool> started{false};
  bool exp = false;
  if (!started.compare_exchange_strong(exp, true))
    return;
  guest::SpawnThread("probe_trap", [] {
    u64 last[kFnWatchMax] = {0};
    for (;;) {
      guest::SleepForMilliseconds((2) * 1000);
      base::String line;
      base::FormatTo(line, "[fnwatch]");
      for (int i = 0; i < g_fn_watch_count; i++) {
        u64 h = g_fn_watch_hits[i].load(base::memory_order_relaxed);
        base::FormatTo(line, " {}={}(+{})", g_fn_watch_labels[i],
                       (unsigned long long)h,
                       (unsigned long long)(h - last[i]));
        last[i] = h;
      }
      BASE_LOGI("fnwatch", "{}", line.c_str());
    }
  });
}

namespace {
const guest::SessionReset g_session_reset([] {
  guest::ResetResource(g_alloc_trace_addr);
  guest::ResetResource(g_heap_prof_addr);
  guest::ResetResource(g_heap_prof_hook_count);
  guest::ResetResource(g_heap_prof_scope_slot);
  guest::ResetResource(g_cnt_trace_addr);
  guest::ResetResource(g_fatal_trace_addr);
  guest::ResetResource(g_hdr_trace_addrs);
  guest::ResetResource(g_hdr_trace_count);
  guest::ResetResource(g_rdoff_addr);
  guest::ResetResource(g_manifest_fd);
  guest::ResetResource(g_skip_fn_addrs);
  guest::ResetResource(g_skip_fn_count);
  guest::ResetResource(g_order_count);
  guest::ResetResource(g_ret_count);
  guest::ResetResource(g_call_skip_count);
  guest::ResetResource(g_fn_watch_count);
  guest::ResetResource(g_fn_args_count);
  guest::ResetResource(g_wprot_base);
  guest::ResetResource(g_wprot_len);
  guest::ResetResource(g_wprot_report_base);
  guest::ResetResource(g_wprot_report_len);
  guest::ResetResource(g_whist_base);
  guest::ResetResource(g_whist_len);
});
}  // namespace

}  // namespace kern::probe
