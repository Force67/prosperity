/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The guest-code trap engine: the int3/trap probes a research session arms, the
 * signal dispatch that services them, and the threads that report them.
 *
 * This used to live inside kern/crash.cpp, which meant the fatal crash reporter
 * and two dozen unrelated debug probes shared one file and one header. They do
 * share the signal handler -- a probe trap and a real fault arrive the same way
 * -- so the seam is onSignal(): crash.cpp offers every signal here first and
 * only reports a fault the probes did not claim. Ordering matters and is
 * preserved: the probes run BEFORE the CPU backend's JIT-signal hand-off,
 * because a trap we planted is ours whatever the JIT would make of it.
 *
 * The fault-recovery policies (null-guard, SDK assert skip) deliberately stay
 * in crash.cpp: they run after the JIT hand-off and are part of deciding
 * whether a fault is fatal at all, not part of tracing guest code.
 */

#include "kern/probe/probe_arm.h"

#include "base/arch.h"
#include "kern/crash.h"
#include "kern/lv2/dispatch.h"
#include "kern/proc.h"
#include "kern/vfs.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <vector>
#include <unordered_map>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <dlfcn.h>
#include <unistd.h>
#include <ucontext.h>

#include <base.h>
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <utl/mem.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(uintptr_t, kBrkTrace, "DELTA_GUEST_BRK_TRACE", 0);
DELTA_OPTION(const char *, kBrkDump, "DELTA_GUEST_BRK_DUMP", nullptr);
DELTA_OPTION(uintptr_t, kBrkArm, "DELTA_GUEST_BRK_ARM", 0);
DELTA_OPTION(u32, kBrkArmPos, "DELTA_GUEST_BRK_ARM_POS", 0);
DELTA_OPTION(const char *, kBrkPeek, "DELTA_GUEST_BRK_PEEK", nullptr);
DELTA_OPTION(const char *, kBrkWprot, "DELTA_GUEST_BRK_WPROT", nullptr);
DELTA_OPTION(const char *, kCrashPeek, "DELTA_CRASH_PEEK", nullptr);
DELTA_OPTION(bool, kCntClamp, "DELTA_CNT_CLAMP", false);
DELTA_OPTION(bool, kHdrFill, "DELTA_HDR_FILL", false);
DELTA_OPTION(bool, kHdrWait, "DELTA_HDR_WAIT", false);
DELTA_OPTION(bool, kPs5Dcbforce, "DELTA_PS5_DCBFORCE", false);
DELTA_OPTION(bool, kRdoffNofix, "DELTA_RDOFF_NOFIX", false);
DELTA_OPTION(bool, kRdoffTrace, "DELTA_RDOFF_TRACE", false);
}  // namespace

namespace krnl {
const u32 *currentGuestTidPtr();  // sys_thread.cpp: this thread's guest tid
}

namespace krnl::probe {

// DELTA_HEAP_PROF: dump the top allocation sites (defined below).
extern uintptr_t g_heapProfAddr;
static void heapProfDumpOnce();

// SIGUSR1 probe: dump the receiving thread's current guest RIP + a stack scan of
// return addresses in loaded modules. Sent to every thread (one per /proc task)
// to find what a wedged title's threads are blocked on. x86-native only.
#if defined(__x86_64__)
static void probeHandler(int, siginfo_t *, void *ucv) {
  auto *uc = static_cast<ucontext_t *>(ucv);
  auto *gr = uc->uc_mcontext.gregs;
  char rip[256];
  symbolize(gr[REG_RIP], rip, sizeof(rip));
  // The GUEST tid too: the host tid says nothing about which of the title's
  // threads this is, and "which thread is the title's main one" is the first
  // thing to know when it stops.
  BASE_LOGI("probe", "tid={} gtid={} rip={:016x} {}", (long)gettid(),
            *currentGuestTidPtr(), (unsigned long long)gr[REG_RIP], rip);
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
  backtrace(gr[REG_RBP]);
  // A thread parked in a wait is parked inside a SYSCALL, so its rsp is our own
  // handler stack and scanning it finds host frames and nothing else -- which is
  // what the probe used to print, and it also walked off the end of the mapping
  // and took the run down. The guest stack the thread came off is recorded on
  // syscall entry, it is copied out with process_vm_readv rather than read
  // directly, and it is the one that says which guest function is waiting.
  guestStackTrace("probe", 8);
  // DELTA_SCHIST syscall histogram (lv2.cpp counts each syscall in its trampoline).
  // Dump the non-zero counts so a slow/wedged title's hammered syscalls are visible
  // -- the only profiler available (perf/strace/proc-mem are yama-blocked here).
  // The probe is sent to every thread, so only the first responder of a burst
  // prints it: forty copies interleaved with forty stack dumps is unreadable,
  // and the stacks are the reason for the burst.
  static std::atomic<u64> lastHist{0};
  const u64 nowS = (u64)::time(nullptr);
  u64 prev = lastHist.load();
  if (nowS - prev < 2 || !lastHist.compare_exchange_strong(prev, nowS)) {
    std::fflush(stderr);
    return;
  }
  bool any = false;
  for (int i = 0; i < 1024; i++) {
    if (g_sysHist[i] > 100) {  // skip noise
      if (!any) { BASE_LOGI("schist", "syscall counts:"); any = true; }
      BASE_LOGI("schist", "  {:4} {:<28} {}", i, syscall_getname(i),
                (unsigned long long)g_sysHist[i]);
    }
  }
  if (g_heapProfAddr)
    heapProfDumpOnce();
  std::fflush(stderr);
}
#endif

// DELTA_ALLOC_TRACE: a guest allocator-entry vaddr whose first byte is `push rbp`
// (0x55), replaced with int3 so we log each large allocation's size without gdb
// (gdb conditional breakpoints are far too slow on this hot path). The handler
// logs rsi (the size arg) when big, emulates the push rbp, and resumes -- one
// trap per call, no single-stepping. x86-native only.
uintptr_t g_allocTraceAddr = 0;
u64 g_allocTraceMin = 0x1000000;  // 16 MiB
// DELTA_HEAP_PROF: aggregate operator-new/malloc (size in rdi) by guest caller.
// Fixed open-addressing table, claimed lock-free from the trap handler (called
// concurrently from every guest thread). SIGUSR1 dumps the top sites by bytes.
uintptr_t g_heapProfAddr = 0;  // non-zero once any hook is armed
static constexpr int kHeapProfMaxHooks = 24;
static uintptr_t g_heapProfHooks[kHeapProfMaxHooks];
static int g_heapProfHookCount = 0;
static std::atomic<u64> g_heapProfHookBytes[kHeapProfMaxHooks];
static std::atomic<u64> g_heapProfHookCalls[kHeapProfMaxHooks];
static bool g_heapProfCountOnly[kHeapProfMaxHooks];
namespace {
constexpr u32 kHeapProfSlots = 16384;
struct HeapProfSlot {
  std::atomic<uintptr_t> caller{0};
  std::atomic<u64> bytes{0};
  std::atomic<u64> count{0};
  std::atomic<u64> unscoped{0};  // of `bytes`, taken with no scope live
};
HeapProfSlot g_heapProf[kHeapProfSlots];
std::atomic<u64> g_heapProfTotal{0};

// DELTA_HEAP_PROF_SCOPE=<tls-slot-global>:<depth-offset>. Engines route an
// allocation through a THREAD-LOCAL stack of scoped allocators and fall back
// to the process heap when that stack is empty; memory taken from a scope is
// released wholesale when the scope resets, memory from the fallback is not.
// A leak that is really "this thread had no scope" is invisible in a profile
// keyed by call site alone, so record the depth the guest would have read:
//   slot  = fs_base + *(u64*)<tls-slot-global>   (the guest's own indirection)
//   block = *(u64*)slot
//   depth = block ? *(u32*)(block + <depth-offset>) : 0
uintptr_t g_heapProfScopeSlot = 0;
u64 g_heapProfScopeDepthOff = 0;
std::atomic<u64> g_heapProfScoped{0}, g_heapProfUnscoped{0};

// Reads through process_vm_readv so a mis-specified address reports nothing
// instead of taking the run down from inside the trap handler.
bool heapProfPeek(uintptr_t addr, void *out, size_t n) {
  if (addr < 0x10000)
    return false;
  iovec l{out, n}, r{reinterpret_cast<void *>(addr), n};
  return ::process_vm_readv(::getpid(), &l, 1, &r, 1, 0) == (ssize_t)n;
}

u32 heapProfScopeDepth(uintptr_t fs_base) {
  if (!g_heapProfScopeSlot || !fs_base)
    return 0;
  u64 off = 0, block = 0;
  u32 depth = 0;
  if (!heapProfPeek(g_heapProfScopeSlot, &off, sizeof(off)))
    return 0;
  if (!heapProfPeek(fs_base + off, &block, sizeof(block)))
    return 0;
  if (!heapProfPeek(block + g_heapProfScopeDepthOff, &depth, sizeof(depth)))
    return 0;
  return depth;
}

void heapProfRecord(uintptr_t caller, u64 size, bool unscoped) {
  u32 h = static_cast<u32>((caller * 2654435761u) >> 13) & (kHeapProfSlots - 1);
  const auto add = [&](u32 s) {
    g_heapProf[s].bytes.fetch_add(size, std::memory_order_relaxed);
    g_heapProf[s].count.fetch_add(1, std::memory_order_relaxed);
    if (unscoped)
      g_heapProf[s].unscoped.fetch_add(size, std::memory_order_relaxed);
  };
  for (u32 i = 0; i < kHeapProfSlots; i++) {
    u32 s = (h + i) & (kHeapProfSlots - 1);
    uintptr_t c = g_heapProf[s].caller.load(std::memory_order_relaxed);
    if (c == caller) {
      add(s);
      break;
    }
    if (c == 0) {
      uintptr_t expected = 0;
      if (g_heapProf[s].caller.compare_exchange_strong(expected, caller,
                                                       std::memory_order_relaxed)) {
        add(s);
        break;
      }
      if (expected == caller) {
        add(s);
        break;
      }
    }
  }
  g_heapProfTotal.fetch_add(size, std::memory_order_relaxed);
  (unscoped ? g_heapProfUnscoped : g_heapProfScoped)
      .fetch_add(size, std::memory_order_relaxed);
}
void heapProfDump() {
  BASE_LOGI("heapprof",
            "total={} bytes ({:.1f} MB) across sites; top by bytes:",
            (unsigned long long)g_heapProfTotal.load(),
            g_heapProfTotal.load() / 1048576.0);
  if (g_heapProfScopeSlot)
    BASE_LOGI("heapprof",
              "scoped={:.1f} MB  unscoped={:.1f} MB (no scoped "
              "allocator live on the calling thread)",
              g_heapProfScoped.load() / 1048576.0,
              g_heapProfUnscoped.load() / 1048576.0);
  for (int i = 0; i < g_heapProfHookCount; i++) {
    u64 b = g_heapProfHookBytes[i].load(std::memory_order_relaxed);
    BASE_LOGI("heapprof", "  hook[{}] {:#x}: {:8.1f} MB  {:8} calls", i,
              (unsigned long)g_heapProfHooks[i], b / 1048576.0,
              (unsigned long long)g_heapProfHookCalls[i].load(std::memory_order_relaxed));
  }
  // Select top 20 by bytes without allocating (linear passes).
  u64 prevBytes = ~0ull;
  uintptr_t prevCaller = 0;
  for (int rank = 0; rank < 20; rank++) {
    u64 bestB = 0; u32 bestS = kHeapProfSlots;
    for (u32 s = 0; s < kHeapProfSlots; s++) {
      u64 b = g_heapProf[s].bytes.load(std::memory_order_relaxed);
      if (b == 0) continue;
      uintptr_t c = g_heapProf[s].caller.load(std::memory_order_relaxed);
      bool below = b < prevBytes || (b == prevBytes && c > prevCaller);
      if (below && b >= bestB) { bestB = b; bestS = s; }
    }
    if (bestS == kHeapProfSlots) break;
    uintptr_t c = g_heapProf[bestS].caller.load(std::memory_order_relaxed);
    u64 cnt = g_heapProf[bestS].count.load(std::memory_order_relaxed);
    u64 uns = g_heapProf[bestS].unscoped.load(std::memory_order_relaxed);
    char sym[200];
    symbolize(c, sym, sizeof(sym));
    BASE_LOGI("heapprof", "  {:8.1f} MB  {:8} calls  {}{}",
              bestB / 1048576.0, (unsigned long long)cnt, sym,
              g_heapProfScopeSlot
                  ? (uns == bestB ? "  [all unscoped]"
                                  : uns ? "  [part unscoped]" : "  [scoped]")
                  : "");
    prevBytes = bestB; prevCaller = c;
  }
  std::fflush(stderr);
}
}  // namespace
void setHeapProfScope(uintptr_t tlsSlotGlobal, u64 depthOffset) {
  g_heapProfScopeSlot = tlsSlotGlobal;
  g_heapProfScopeDepthOff = depthOffset;
}
void setHeapProf(uintptr_t addr, bool countOnly) {
  if (g_heapProfHookCount < kHeapProfMaxHooks) {
    g_heapProfCountOnly[g_heapProfHookCount] = countOnly;
    g_heapProfHooks[g_heapProfHookCount++] = addr;
  }
  g_heapProfAddr = addr;  // any non-zero arms the SIGTRAP path
}
// Throttle to one dump per SIGUSR1 burst (every thread gets the signal).
static void heapProfDumpOnce() {
  static std::atomic<u64> last{0};
  timespec t{};
  clock_gettime(CLOCK_MONOTONIC, &t);
  u64 now = (u64)t.tv_sec * 1000 + t.tv_nsec / 1000000;
  u64 prev = last.load(std::memory_order_relaxed);
  if (now - prev < 300)
    return;
  if (!last.compare_exchange_strong(prev, now, std::memory_order_relaxed))
    return;
  heapProfDump();
}
// DELTA_CNT_TRACE: same int3-emulate trick at an entry whose 1st byte is push rbp,
// but logs the per-archive entry-count [rdi+0x30] and the inline name at [rdi+0x5c].
uintptr_t g_cntTraceAddr = 0;
// DELTA_FATAL_TRACE: int3 at a printf-style fatal handler entry (push rbp); log
// rdi (the format string) + caller + the first varargs, then resume.
uintptr_t g_fatalTraceAddr = 0;
// DELTA_HDR_TRACE: int3 at each manifest consumer (push rbp) where rdi=parent;
// the manifest header is [parent+0x8] and the archive name is [[parent+0x10]+0x5c].
// Supports several addresses (comma-separated env) so every consumer that reads
// the header (count-setter 0x606150, segcount-reader 0x6063a0, ...) gets it filled.
uintptr_t g_hdrTraceAddrs[8] = {0};
int g_hdrTraceCount = 0;
// DELTA_RDOFF_FIX: int3 at the file-read-request setter 0x60b510 (push rbp), args
// esi=fd edx=offset ecx=nbytes r8=buf. SOTTR passes a garbage offset for manifest
// reads; force it to 0 (read from the start) when the fd is a .manifest.bin fd.
uintptr_t g_rdoffAddr = 0;
bool g_manifestFd[8192] = {false};
void markManifestFd(u32 fd, bool v) { if (fd < 8192) g_manifestFd[fd] = v; }
// DELTA_SKIP_FN: int3 at a function entry (push rbp); emulate an immediate
// `ret` (the push rbp hasn't run, so [rsp] is the return addr) with rax=0. Skips
// the whole function -- used to step past a guest function that crashes/wedges
// (e.g. the localization loader 0x666410) to reach the next boot stage.
uintptr_t g_skipFnAddrs[8] = {0};
int g_skipFnCount = 0;

// DELTA_PS5_GLYPHGUARD: recover the first-frame unbound-font null derefs in the
// game's UI/text renderer. Each entry: the faulting rip, the GP register the
// faulting instruction writes (zeroed so the code proceeds with a benign value),
// and the instruction length (rip is advanced past it). The fault only fires when
// the base register is null, so normal (bound-font) calls are untouched.

// DELTA_PS5_DCBWATCH call-order trace (see crash.h).
static constexpr int kOrderMax = 12;
static uintptr_t g_orderAddrs[kOrderMax];
static const char *g_orderLabels[kOrderMax];
static int g_orderCount = 0;
static timespec g_orderStart;
static uintptr_t g_retAddrs[8];
static const char *g_retLabels[8];
static bool g_retIsTest[8];  // site was `test eax,eax`, not `mov ebx,eax`
static int g_retCount = 0;

// DELTA_PS5_GLYPHGUARD call-skip (see crash.h): int3 planted over a blocking
// vtable-dispatch call; on hit, inject rax and step past the whole call insn.
static uintptr_t g_callSkipAddrs[8];
static long g_callSkipVals[8];
static int g_callSkipLens[8];
static int g_callSkipCount = 0;

// DELTA_FNWATCH hit counter (see crash.h).
static constexpr int kFnWatchMax = 16;
static uintptr_t g_fnWatchAddrs[kFnWatchMax];
static const char *g_fnWatchLabels[kFnWatchMax];
static std::atomic<u64> g_fnWatchHits[kFnWatchMax];
static int g_fnWatchCount = 0;

// DELTA_FNARGS pointer-chain probe (see crash.h).
static constexpr int kFnArgsMax = 8;
static constexpr int kFnArgsOffsMax = 6;
static constexpr int kFnArgsLogs = 8;  // logs per site; these sites are hot
static uintptr_t g_fnArgsAddrs[kFnArgsMax];
static const char *g_fnArgsLabels[kFnArgsMax];
static u64 g_fnArgsOffs[kFnArgsMax][kFnArgsOffsMax];
static int g_fnArgsNoffs[kFnArgsMax];
static std::atomic<u64> g_fnArgsHits[kFnArgsMax];
static int g_fnArgsCount = 0;

// Range currently write-protected for the write watch, armed either from the
// DELTA_GUEST_BRK_TRACE handler or standalone by DELTA_GUEST_WPROT.
static std::atomic<uintptr_t> g_wprotBase{0};
static std::atomic<size_t> g_wprotLen{0};
static std::atomic<bool> g_wprotRegs{false};
static std::atomic<bool> g_wprotStep{false};
static std::atomic<uintptr_t> g_wprotReportBase{0};
static std::atomic<size_t> g_wprotReportLen{0};
#if defined(__x86_64__)
static thread_local uintptr_t g_wprotStepPage = 0;
#endif

// DELTA_GUEST_WHIST census state (see startWriteHist). A one-shot watch names
// the first writer of a page and then goes quiet; this one re-arms, so a pool
// too large to log per write still yields "which parts of it are written, by
// whom, over the whole run".
static constexpr size_t kWhistGranule = 16u << 20;
static constexpr int kWhistBuckets = 512;
static constexpr int kWhistSites = 32;
static std::atomic<uintptr_t> g_whistBase{0};
static std::atomic<size_t> g_whistLen{0};
static std::atomic<u32> g_whistBucket[kWhistBuckets];
static std::atomic<uintptr_t> g_whistSite[kWhistSites];
static std::atomic<uintptr_t> g_whistSiteCaller[kWhistSites];
static std::atomic<u32> g_whistSiteHits[kWhistSites];
// Faults whose guest instruction could not be established (see
// watchFaultGuestRip). Reported alongside the sites so a census can never read
// as "these are all the writers" when some of them went unnamed.
static std::atomic<u64> g_whistUnattributed{0};

// The guest instruction behind a memory-watch fault. Returns false when it
// cannot be established, and then `rip` is 0.
//
// On an x86 host the signal context's RIP already IS the guest RIP. Under FEX on
// ARM the fault is raised inside JIT'd code, so the host pc must be mapped back
// through FEX. `cpu::currentGuestRip()` must NOT serve as a fallback here: it is
// only block-accurate (see cpu_backend.h), so under multiblock compilation it
// names some earlier instruction of whatever block is running. That reads as an
// authoritative answer while being wrong -- it attributed a write to one of
// SotC's descriptor pages to libc's memcpy, a store the title never made, and
// sent a whole investigation down a dead end. An unattributable fault must SAY
// it is unattributable.
static bool watchFaultGuestRip(void *ucv, uintptr_t &rip) {
  rip = 0;
#if defined(__x86_64__)
  rip = (uintptr_t)static_cast<ucontext_t *>(ucv)->uc_mcontext.gregs[REG_RIP];
  return rip != 0;
#elif defined(__aarch64__)
  rip = (uintptr_t)cpu::reconstructGuestRip(
      static_cast<ucontext_t *>(ucv)->uc_mcontext.pc);
  return rip != 0;
#else
  (void)ucv;
  return false;
#endif
}

// Guest stack pointer at the fault, or 0. Lets a leaf writer (libc memcpy names
// no subsystem) be attributed to its caller.
static uintptr_t watchFaultGuestRsp(void *ucv) {
#if defined(__x86_64__)
  return (uintptr_t)static_cast<ucontext_t *>(ucv)->uc_mcontext.gregs[REG_RSP];
#else
  (void)ucv;
  if (const u64 *g = cpu::currentGuestGregs()) {
    enum { RAX, RCX, RDX, RBX, RSP };
    return (uintptr_t)g[RSP];
  }
  return 0;
#endif
}

#if defined(__aarch64__)
static void probeHandler(int, siginfo_t *, void *ucv) {
  uintptr_t rip = 0;
  const bool attributed = watchFaultGuestRip(ucv, rip);
  char sym[256];
  if (attributed)
    symbolize(rip, sym, sizeof(sym));
  else
    std::snprintf(sym, sizeof(sym), "<unattributed FEX host pc>");
  const auto host_pc = static_cast<ucontext_t *>(ucv)->uc_mcontext.pc;
  BASE_LOGI("probe", "tid={} host_pc={:#x} guest_pc={:#x} {}",
            (long)gettid(), (unsigned long long)host_pc,
            (unsigned long)rip, sym);
  if (const uintptr_t sp = watchFaultGuestRsp(ucv))
    guestStackTraceFrom(sp, "probe", 8, (long)syscall(SYS_gettid));
  std::fflush(stderr);
}
#endif

// Reopen the one page a watch fault landed on so the guest can retry the access.
// Arch-independent: returning from the handler re-executes the faulting
// instruction, which is what turns a one-shot trap into a running trace. Without
// this the watch is not merely blind, it is FATAL -- on ARM both watches used to
// fall through to the crash reporter and kill the title.
static void reopenWatchPage(uintptr_t at) {
  const long pgsz = sysconf(_SC_PAGESIZE);
  ::mprotect(reinterpret_cast<void *>(at & ~((uintptr_t)pgsz - 1)),
             (size_t)pgsz, PROT_READ | PROT_WRITE | PROT_EXEC);
}

static void resumeWatchedWrite(uintptr_t at, void *ucv) {
  reopenWatchPage(at);
#if defined(__x86_64__)
  if (g_wprotStep.load(std::memory_order_relaxed)) {
    const uintptr_t pgsz = (uintptr_t)sysconf(_SC_PAGESIZE);
    g_wprotStepPage = at & ~(pgsz - 1);
    static_cast<ucontext_t *>(ucv)->uc_mcontext.gregs[REG_EFL] |= 0x100;
  }
#else
  (void)ucv;
#endif
}


// Offered every signal before the crash reporter looks at it. True means a
// probe owned this trap and the handler must resume the guest.
bool onSignal(int sig, siginfo_t *si, void *ucv) {
#if defined(__x86_64__)
  if (sig == SIGTRAP && ucv && g_wprotStepPage) {
    const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    ::mprotect(reinterpret_cast<void *>(g_wprotStepPage), pgsz, PROT_READ);
    static_cast<ucontext_t *>(ucv)->uc_mcontext.gregs[REG_EFL] &= ~0x100;
    g_wprotStepPage = 0;
    return true;
  }
#endif
  // Write census: same trap as the write watch, but it only counts (per 16 MiB
  // bucket and per faulting instruction) and reopens the page, so it survives a
  // multi-GB range being re-armed for the length of a run.
  if (sig == SIGSEGV && si && ucv && g_whistLen.load()) {
    const uintptr_t base = g_whistBase.load();
    const uintptr_t at = reinterpret_cast<uintptr_t>(si->si_addr);
    if (at >= base && at < base + g_whistLen.load()) {
      uintptr_t rip = 0;
      const bool attributed = watchFaultGuestRip(ucv, rip);
      const size_t b = (at - base) / kWhistGranule;
      if (b < kWhistBuckets)
        g_whistBucket[b].fetch_add(1, std::memory_order_relaxed);
      // Slot 0 doubles as "empty" in the site table, so an unattributable
      // fault must not be filed as a writer at rip 0 -- it is counted apart and
      // the report says how many there were.
      if (!attributed) {
        g_whistUnattributed.fetch_add(1, std::memory_order_relaxed);
      } else {
        for (int i = 0; i < kWhistSites; i++) {
          uintptr_t cur = g_whistSite[i].load(std::memory_order_relaxed);
          if (cur == rip) {
            g_whistSiteHits[i].fetch_add(1, std::memory_order_relaxed);
            break;
          }
          if (!cur && g_whistSite[i].compare_exchange_strong(cur, rip)) {
            // A leaf writer (libc memcpy) names no subsystem; keep one sample
            // of its return address so the report can name the caller too.
            if (const uintptr_t sp = watchFaultGuestRsp(ucv))
              g_whistSiteCaller[i].store(
                  *reinterpret_cast<const uintptr_t *>(sp),
                  std::memory_order_relaxed);
            g_whistSiteHits[i].fetch_add(1, std::memory_order_relaxed);
            break;
          }
        }
      }
      reopenWatchPage(at);
      return true;
    }
  }
  // Write watch: the range was made read-only, so a write faults here. Name the
  // instruction, open that one page and resume, which turns the trap into a
  // list of everything that writes the range rather than just the first thing.
  if (sig == SIGSEGV && si && ucv && g_wprotLen.load()) {
    const uintptr_t base = g_wprotBase.load();
    const size_t len = g_wprotLen.load();
    const uintptr_t at = reinterpret_cast<uintptr_t>(si->si_addr);
    if (at >= base && at < base + len) {
      const uintptr_t report_base = g_wprotReportBase.load();
      const size_t report_len = g_wprotReportLen.load();
      const bool report = !g_wprotStep.load() ||
                          (at >= report_base && at < report_base + report_len);
      if (!report) {
        resumeWatchedWrite(at, ucv);
        return true;
      }
      uintptr_t rip = 0;
      const bool attributed = watchFaultGuestRip(ucv, rip);
      char sym[192];
      if (attributed)
        symbolize(rip, sym, sizeof(sym));
      else {
        // Not guest code, so it is OUR code writing into the guest's memory
        // (an HLE call, the kernel, a GPU readback). Naming the host module is
        // the point: "unattributed" alone reads as noise when it may be the
        // emulator scribbling on the title's heap. Only the leaf is named --
        // backtrace() cannot unwind past the signal trampoline, and a hand
        // walk of the interrupted frame chain yields addresses dladdr cannot
        // symbolise in our own binary, so resolving our own frames would need
        // the project's symbolize() driven from uc_mcontext.
#if defined(__x86_64__)
        const uintptr_t host_pc = (uintptr_t)static_cast<ucontext_t *>(ucv)
                                      ->uc_mcontext.gregs[REG_RIP];
#elif defined(__aarch64__)
        const uintptr_t host_pc =
            (uintptr_t)static_cast<ucontext_t *>(ucv)->uc_mcontext.pc;
#else
        const uintptr_t host_pc = 0;
#endif
        Dl_info di{};
        const char *base = nullptr;
        if (dladdr(reinterpret_cast<void *>(host_pc), &di) && di.dli_fname)
          base = std::strrchr(di.dli_fname, '/');
        if (di.dli_sname)
          std::snprintf(sym, sizeof(sym), "HOST %s+%#lx", di.dli_sname,
                        (unsigned long)(host_pc -
                                        reinterpret_cast<uintptr_t>(di.dli_saddr)));
        else if (di.dli_fname)
          std::snprintf(sym, sizeof(sym), "HOST %s+%#lx",
                        base ? base + 1 : di.dli_fname,
                        (unsigned long)(host_pc -
                                        reinterpret_cast<uintptr_t>(di.dli_fbase)));
        else
          std::snprintf(sym, sizeof(sym), "HOST pc=%#lx (no symbol)",
                        (unsigned long)host_pc);
      }
      // Only a write-only watch can name the access from the protection alone.
      // A reads-too watch (PROT_NONE) cannot on ARM, where there is no x86
      // page-fault error code -- so it says "access" rather than guessing.
#if defined(__x86_64__)
      const char *kind =
          (static_cast<ucontext_t *>(ucv)->uc_mcontext.gregs[REG_ERR] & 2)
              ? "write"
              : "read";
#else
      const char *kind = g_wprotRegs.load() ? "access" : "write";
#endif
      if (const uintptr_t probe = utl::writeWatchValueProbe();
          probe && utl::isMemoryRangeMapped(reinterpret_cast<void *>(probe), 8))
        BASE_LOGI("wprot", "{} {:#x} from {} | probe {:#x} = {:#x}", kind,
                  (unsigned long long)at, sym, (unsigned long)probe,
                  (unsigned long long)*reinterpret_cast<u64 *>(probe));
      else
        BASE_LOGI("wprot", "{} {:#x} from {}", kind, (unsigned long long)at,
                  sym);
      // The writer of a descriptor/command ring is nearly always libc memcpy,
      // which names no subsystem -- so the CALLER is the whole point of the
      // report, not an extra for the reads-too mode. Reading the single qword at
      // the guest rsp does not find it: the leaf is mid-body by the time it
      // faults (and on ARM the guest rsp snapshot can lag), which yields a stack
      // address rather than a return address. Scan the stack window for values
      // that land in a loaded module's .text, the same way the fatal reporter
      // recovers a call chain the frame pointer misses.
      if (attributed) {
        if (const uintptr_t sp = watchFaultGuestRsp(ucv))
          guestStackTraceFrom(sp, "wprot", 4, (long)syscall(SYS_gettid));
      }
      // A consumer's other pointer (where it puts what it just read) is only
      // visible in its registers at the access. A value probe is an explicit
      // request to follow one word, and the word's SOURCE (a memcpy's rsi) is
      // the next hop, so dump registers for that case too.
      if (g_wprotRegs.load() || utl::writeWatchValueProbe()) {
        const u64 *g = nullptr;
#if defined(__x86_64__)
        u64 xg[16];
        auto *hg = static_cast<ucontext_t *>(ucv)->uc_mcontext.gregs;
        const int kOrder[16] = {REG_RAX, REG_RCX, REG_RDX, REG_RBX,
                                REG_RSP, REG_RBP, REG_RSI, REG_RDI,
                                REG_R8,  REG_R9,  REG_R10, REG_R11,
                                REG_R12, REG_R13, REG_R14, REG_R15};
        for (int i = 0; i < 16; i++)
          xg[i] = (u64)hg[kOrder[i]];
        g = xg;
#else
        // FEX pins every guest GPR to a fixed host register, so the signal
        // context holds the exact values at the faulting instruction. Only fall
        // back to the in-memory thread state -- which is written back at block
        // boundaries and therefore lags -- when the fault was not in JIT code.
        u64 sig_gregs[16];
        bool exact = cpu::guestGregsFromSignal(ucv, sig_gregs);
        g = exact ? sig_gregs : cpu::currentGuestGregs();
#endif
        if (g) {
          enum { RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
                 R8, R9, R10, R11, R12, R13, R14, R15 };
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
          // Chase the probed word upstream. With exact registers a block copy
          // reads as rdi=dest, rsi=source, rcx=length; the word we are watching
          // sits at (probe - rdi) into the destination, so the same word in the
          // SOURCE is rsi + that delta. Re-aim there and watch it: that is one
          // hop back towards wherever the value was first produced.
          enum { C_RCX = 1, C_RSI = 6, C_RDI = 7 };
          const uintptr_t probe = utl::writeWatchValueProbe();
          if (exact && probe && utl::writeWatchChaseLeft()) {
            const u64 rdi = g[C_RDI], rsi = g[C_RSI], rcx = g[C_RCX];
            const bool looks_like_copy =
                rdi && rsi && rcx && probe >= rdi && probe - rdi < rcx;
            const uintptr_t src = rsi + (probe - rdi);
            if (looks_like_copy &&
                utl::isMemoryRangeMapped(reinterpret_cast<void *>(src), 8)) {
              utl::writeWatchChaseTook();
              BASE_LOGI("wprot",
                        "chase: probe {:#x} came from {:#x} (copy {:#x} <- "
                        "{:#x} len {:#x}); watching the source",
                        (unsigned long)probe, (unsigned long)src,
                        (unsigned long)rdi, (unsigned long)rsi,
                        (unsigned long)rcx);
              utl::setWriteWatchValueProbe(src);
              utl::armWriteWatch(src, 8, 200);
            }
          }
#endif
        }
      }
      std::fflush(stderr);
      resumeWatchedWrite(at, ucv);
      return true;
    }
  }
#if defined(__x86_64__)
  // DELTA_GUEST_BRK_TRACE: a RESUMABLE planted breakpoint. The ud2 replaced the
  // first bytes of a known instruction, so the handler emulates that
  // instruction, logs what we came for, and returns -- turning a one-shot trap
  // into a trace. Shaped for the V8 snapshot dispatch
  // (libcohtml `mov %esi,%r12d`, 3 bytes): logs the bytecode in esi and the
  // SnapshotByteSource position at rdi+0x3c, so each record says which bytecode
  // ran and how many stream bytes the previous one consumed.
  if (sig == SIGILL && ucv) {
    const uintptr_t site = kBrkTrace;
    if (site) {
      auto *uc = static_cast<ucontext_t *>(ucv);
      auto *gr = uc->uc_mcontext.gregs;
      if ((uintptr_t)gr[REG_RIP] == site) {
        const u32 code = (u32)gr[REG_RSI] & 0xFF;
        const uintptr_t self = (uintptr_t)gr[REG_RDI];
        u32 pos = 0;
        if (self > 0x10000)
          pos = *reinterpret_cast<const u32 *>(self + 0x3c);
        // One open() and one write() per record to a dedicated fd: fprintf to
        // stderr loses most records here, because other threads interleave
        // mid-line and the trace is the whole point.
        static const int fd = ::open("/tmp/bc_trace.txt",
                                     O_WRONLY | O_CREAT | O_TRUNC | O_APPEND,
                                     0644);
        static u64 n = 0;
        if (fd >= 0) {
          char buf[64];
          const int len = std::snprintf(buf, sizeof(buf), "%llu %02x %u\n",
                                        (unsigned long long)n++, code, pos);
          ssize_t ignored = ::write(fd, buf, len);
          (void)ignored;
        }
        // DELTA_GUEST_BRK_ARM=<addr>: plant a ud2 there the moment the traced
        // stream RESTARTS. A site on a hot path traps on its first execution,
        // which for two back-to-back deserializations is always the first one;
        // this reaches the second.
        // DELTA_GUEST_BRK_ARM_POS=<n>: wait until the restarted stream reaches
        // this position, so the trap lands on one named record rather than the
        // first of its kind.
        static u32 last_pos = 0;
        static bool armed = false, restarted = false;
        if (pos < last_pos)
          restarted = true;
        if (kBrkArm && !armed && restarted && pos >= kBrkArmPos) {
          auto *at = reinterpret_cast<u8 *>((uintptr_t)kBrkArm);
          at[0] = 0x0F;
          at[1] = 0x0B;
          armed = true;
        }
        // DELTA_GUEST_BRK_WPROT=<hex addr>:<hex size>: write-protect a guest
        // range at the same moment. Memory that holds live data in one pass and
        // reads empty in the next has a writer; this makes the write fault, so
        // the crash report names the instruction instead of the symptom.
        static bool wprot_done = false;
        if (const char *wp = kBrkWprot; wp && !wprot_done &&
                                        pos >= kBrkArmPos) {
          wprot_done = true;
          const uintptr_t a = std::strtoull(wp, nullptr, 16);
          const char *c = std::strchr(wp, ':');
          const size_t n = c ? std::strtoull(c + 1, nullptr, 16) : 0x40000;
          if (::mprotect(reinterpret_cast<void *>(a), n, PROT_READ) == 0) {
            g_wprotBase = a;
            g_wprotLen = n;
          }
        }
        last_pos = pos;
        gr[REG_R12] = (u32)gr[REG_RSI];  // mov %esi,%r12d (zero-extends)
        gr[REG_RIP] = site + 3;
        return true;
      }
    }
  }
  if (sig == SIGTRAP && g_retCount && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_retCount; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_retAddrs[i] + 1)
        continue;
      u32 eax = (u32)gr[REG_RAX];
      // DELTA_PS5_DCBFORCE: force a failing graphics-init sub-call to report
      // success (SCE_OK) so the run-once init 0x69e720 completes and the engine
      // creates its DrawCommandBuffer -- lets us measure how far the boot gets
      // when the (obfuscated) libSceAgc call is treated as succeeding.
      static const int force = kPs5Dcbforce;
      if (force && eax) {
        gr[REG_RAX] = 0;
        eax = 0;
      }
      char m[128];
      int n = std::snprintf(m, sizeof(m), "[ret] %s eax=%#x\n", g_retLabels[i], eax);
      if (n > 0) { ssize_t w = write(2, m, (size_t)n); (void)w; }
      if (g_retIsTest[i]) {
        // emulate `test eax,eax`: CF/OF cleared, ZF/SF/PF from the result
        greg_t fl = gr[REG_EFL] & ~(greg_t)(0x8d5);  // CF PF AF ZF SF OF
        if (eax == 0) fl |= 0x40;
        if (eax & 0x80000000u) fl |= 0x80;
        if (__builtin_parity(eax & 0xff) == 0) fl |= 0x4;
        gr[REG_EFL] = fl;
      } else {
        gr[REG_RBX] = eax;                // emulate `mov ebx,eax` (zero-extends)
      }
      gr[REG_RIP] = g_retAddrs[i] + 2;    // resume past the 2-byte insn
      return true;
    }
  }
  if (sig == SIGTRAP && g_callSkipCount && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_callSkipCount; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_callSkipAddrs[i] + 1)
        continue;
      static bool s_seen[8] = {};
      if (!s_seen[i]) {
        s_seen[i] = true;
        char m[64];
        int n = std::snprintf(m, sizeof(m), "[callskip] #%d fired -> rax=%ld\n",
                              i, g_callSkipVals[i]);
        if (n > 0) { ssize_t w = write(2, m, (size_t)n); (void)w; }
      }
      gr[REG_RAX] = g_callSkipVals[i];         // inject the blocked call's return
      gr[REG_RIP] = g_callSkipAddrs[i] + g_callSkipLens[i];  // step past the call
      return true;
    }
  }
  if (sig == SIGTRAP && g_orderCount && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_orderCount; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_orderAddrs[i] + 1)
        continue;
      timespec t{};
      clock_gettime(CLOCK_MONOTONIC, &t);
      long ms = (t.tv_sec - g_orderStart.tv_sec) * 1000 +
                (t.tv_nsec - g_orderStart.tv_nsec) / 1000000;
      uintptr_t rsp = (uintptr_t)gr[REG_RSP];
      uintptr_t caller = rsp >= 0x10000 ? *reinterpret_cast<u64 *>(rsp) : 0;
      char csym[200];
      symbolize(caller, csym, sizeof(csym));
      char m[320];
      int n = std::snprintf(m, sizeof(m), "[order t=%ldms tid=%ld] %s  <- %s\n",
                            ms, (long)gettid(), g_orderLabels[i], csym);
      if (n > 0) { ssize_t w = write(2, m, (size_t)n); (void)w; }
      gr[REG_RSP] -= 8;  // emulate the displaced `push rbp`
      *reinterpret_cast<u64 *>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_fnArgsCount && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_fnArgsCount; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_fnArgsAddrs[i] + 1)
        continue;
      if (g_fnArgsHits[i].fetch_add(1, std::memory_order_relaxed) < kFnArgsLogs) {
        char m[512];
        int n = std::snprintf(m, sizeof(m),
                              "[fnargs] %s rdi=%#lx rsi=%#lx rdx=%#lx rcx=%#lx",
                              g_fnArgsLabels[i], (unsigned long)gr[REG_RDI],
                              (unsigned long)gr[REG_RSI],
                              (unsigned long)gr[REG_RDX],
                              (unsigned long)gr[REG_RCX]);
        uintptr_t p = (uintptr_t)gr[REG_RDI];
        for (int o = 0; o < g_fnArgsNoffs[i] && n > 0; o++) {
          uintptr_t at = p + g_fnArgsOffs[i][o];
          if (!utl::isMemoryRangeMapped(reinterpret_cast<void *>(at), 8)) {
            n += std::snprintf(m + n, sizeof(m) - n, " +%#lx=<unmapped>",
                               (unsigned long)g_fnArgsOffs[i][o]);
            p = 0;
            break;
          }
          p = *reinterpret_cast<uintptr_t *>(at);
          n += std::snprintf(m + n, sizeof(m) - n, " +%#lx->%#lx",
                             (unsigned long)g_fnArgsOffs[i][o], (unsigned long)p);
        }
        if (n > 0 && p && utl::isMemoryRangeMapped(reinterpret_cast<void *>(p), 64)) {
          n += std::snprintf(m + n, sizeof(m) - n, " *=");
          const auto *w = reinterpret_cast<const u32 *>(p);
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
      *reinterpret_cast<u64 *>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_fnWatchCount && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_fnWatchCount; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_fnWatchAddrs[i] + 1)
        continue;
      g_fnWatchHits[i].fetch_add(1, std::memory_order_relaxed);
      // Emulate the displaced `push rbp`; RIP stays at addr+1 (the `mov rbp,rsp`).
      gr[REG_RSP] -= 8;
      *reinterpret_cast<u64 *>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_skipFnCount && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    for (int si2 = 0; si2 < g_skipFnCount; si2++) {
      if ((uintptr_t)gr[REG_RIP] == g_skipFnAddrs[si2] + 1) {
        uintptr_t rsp = (uintptr_t)gr[REG_RSP];
        gr[REG_RIP] = *reinterpret_cast<u64 *>(rsp);  // return addr
        gr[REG_RSP] = rsp + 8;
        gr[REG_RAX] = 0;
        return true;
      }
    }
  }
  if (sig == SIGTRAP && g_rdoffAddr && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    if ((uintptr_t)gr[REG_RIP] == g_rdoffAddr + 1) {
      u32 fd = (u32)gr[REG_RSI];
      bool mf = (fd < 8192 && g_manifestFd[fd]);
      if (kRdoffTrace) {
        char m[128];
        int n = std::snprintf(m, sizeof(m),
                              "[rdoff] fd=%u off=%lld nbytes=%lld buf=%llx manifest=%d\n",
                              fd, (long long)(i32)gr[REG_RDX],
                              (long long)(i32)gr[REG_RCX],
                              (unsigned long long)gr[REG_R8], mf);
        if (n > 0) { ssize_t w = write(2, m, (size_t)n); (void)w; }
      }
      // DELTA_RDOFF_NOFIX: observe-only (log requests, don't rewrite offsets).
      if (mf && !kRdoffNofix)
        gr[REG_RDX] = 0;  // force manifest read offset to 0 (read from start)
      gr[REG_RSP] -= 8;   // emulate push rbp
      *reinterpret_cast<u64 *>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_hdrTraceCount && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    bool hit = false;
    for (int hi = 0; hi < g_hdrTraceCount; hi++)
      if ((uintptr_t)gr[REG_RIP] == g_hdrTraceAddrs[hi] + 1) { hit = true; break; }
    if (hit) {
      u64 parent = (u64)gr[REG_RDI];
      u64 hdr = 0, obj = 0;
      u32 magic = 0, cnt = 0;
      char nm[48] = {0};
      if (parent >= 0x10000) {
        hdr = *reinterpret_cast<u64 *>(parent + 0x8);
        obj = *reinterpret_cast<u64 *>(parent + 0x10);
        // DELTA_HDR_WAIT: test the producer-consumer-race hypothesis. If the
        // manifest header buffer isn't filled yet (magic != "TAFS"), block this
        // (consumer) thread to let the worker thread's read+copy complete.
        if (kHdrWait && hdr >= 0x10000) {
          for (int i = 0; i < 2000; i++) {
            if (*reinterpret_cast<volatile u32 *>(hdr) == 0x53464154u)
              break;
            timespec ts{0, 200000};  // 0.2ms
            nanosleep(&ts, nullptr);
          }
        }
        if (hdr >= 0x10000) {
          magic = *reinterpret_cast<u32 *>(hdr);
          cnt = *reinterpret_cast<u32 *>(hdr + 0xc);
        }
        if (obj >= 0x10000) {
          const char *s = reinterpret_cast<const char *>(obj + 0x5c);
          int j = 0; for (; j < 47 && s[j] >= 0x20 && s[j] <= 0x7e; j++) nm[j] = s[j];
          nm[j] = 0;
        }
        // DELTA_HDR_FILL: bypass the racy async manifest reader by copying the
        // real (cached) manifest bytes straight into the header buffer, so the
        // count-setter reads the correct count + entry table for THIS archive.
        if (kHdrFill && hdr >= 0x10000 && nm[0]) {
          auto *h = reinterpret_cast<u8 *>(hdr);
          // The header buffer [parent+0x8] is allocated filesize (at 0x605e30),
          // so fill the WHOLE manifest at every consumer hook -- both the header
          // (count) and the entry table must be correct for the downstream
          // segment/entry processing (0x666xxx) not to read garbage.
          if (const auto *mf = vfs::getCachedFile(nm)) {
            std::memcpy(h, mf->data(), mf->size());
          } else {
            // Missing archive (e.g. JAPANESE not in this pkg): write a valid
            // empty TAFS header (count=0) so the entry-table alloc is tiny and
            // the archive is empty, instead of reading a garbage count -> OOM.
            std::memset(h, 0, 0x34);
            h[0] = 'T'; h[1] = 'A'; h[2] = 'F'; h[3] = 'S';
            *reinterpret_cast<u32 *>(h + 4) = 3;     // version
            *reinterpret_cast<u32 *>(h + 0x10) = 7;  // strlen("orbis-w")
            std::memcpy(h + 0x14, "orbis-w", 7);
          }
          magic = *reinterpret_cast<u32 *>(h);
          cnt = *reinterpret_cast<u32 *>(h + 0xc);
        }
      }
      char m[176];
      int n = std::snprintf(m, sizeof(m),
                            "[hdr] t=%ld name=\"%s\" hdr=%#llx magic=%08x count=%u\n",
                            (long)gettid(), nm, (unsigned long long)hdr, magic, cnt);
      if (n > 0) { ssize_t w = write(2, m, (size_t)n); (void)w; }
      static bool once = false;
      if (!once && obj >= 0x10000) {
        once = true;
        u64 vt = *reinterpret_cast<u64 *>(obj);
        u64 m58 = (vt >= 0x10000) ? *reinterpret_cast<u64 *>(vt + 0x58) : 0;
        // 0x608390-style forward: inner obj = [obj+0x8], real method = inner.vt[0x58].
        u64 inner = *reinterpret_cast<u64 *>(obj + 0x8);
        u64 ivt = (inner >= 0x10000) ? *reinterpret_cast<u64 *>(inner) : 0;
        u64 im58 = (ivt >= 0x10000) ? *reinterpret_cast<u64 *>(ivt + 0x58) : 0;
        u64 im30 = (ivt >= 0x10000) ? *reinterpret_cast<u64 *>(ivt + 0x30) : 0;
        char v[224];
        int vn = std::snprintf(v, sizeof(v),
                               "[hdr] obj=%#llx vt=%#llx m58=%#llx | inner=%#llx ivt=%#llx im30=%#llx im58=%#llx\n",
                               (unsigned long long)obj, (unsigned long long)vt,
                               (unsigned long long)m58, (unsigned long long)inner,
                               (unsigned long long)ivt, (unsigned long long)im30,
                               (unsigned long long)im58);
        if (vn > 0) { ssize_t w = write(2, v, (size_t)vn); (void)w; }
      }
      gr[REG_RSP] -= 8;
      *reinterpret_cast<u64 *>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_fatalTraceAddr && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    if ((uintptr_t)gr[REG_RIP] == g_fatalTraceAddr + 1) {
      u64 fmt = (u64)gr[REG_RDI];
      u64 caller = 0;
      uintptr_t rsp = (uintptr_t)gr[REG_RSP];
      if (rsp >= 0x10000) caller = *reinterpret_cast<u64 *>(rsp);
      char msg[256] = {0};
      if (fmt >= 0x10000) {
        const char *s = reinterpret_cast<const char *>(fmt);
        int j = 0;
        for (; j < 255 && s[j]; j++) msg[j] = (s[j] >= 0x20 || s[j] == '\n') ? s[j] : '.';
        msg[j] = 0;
      }
      char out[480];
      int n = std::snprintf(out, sizeof(out),
                            "[FATAL] caller=%#llx rsi=%#llx rdx=%#llx rcx=%#llx\n        fmt=\"%s\"\n",
                            (unsigned long long)caller, (unsigned long long)gr[REG_RSI],
                            (unsigned long long)gr[REG_RDX], (unsigned long long)gr[REG_RCX], msg);
      if (n > 0) { ssize_t w = write(2, out, (size_t)n); (void)w; }
      gr[REG_RSP] -= 8;
      *reinterpret_cast<u64 *>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_cntTraceAddr && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    if ((uintptr_t)gr[REG_RIP] == g_cntTraceAddr + 1) {
      u64 obj = (u64)gr[REG_RDI];
      u32 cnt = 0; char nm[48] = {0};
      if (obj >= 0x10000) {
        cnt = *reinterpret_cast<u32 *>(obj + 0x30);
        const char *s = reinterpret_cast<const char *>(obj + 0x5c);
        int j = 0; for (; j < 47 && s[j] >= 0x20 && s[j] <= 0x7e; j++) nm[j] = s[j];
        nm[j] = 0;
      }
      char m[128];
      int n = std::snprintf(m, sizeof(m), "[cnt] obj=%llx count=%u (%#x) name=\"%s\"\n",
                            (unsigned long long)obj, cnt, cnt, nm);
      if (n > 0) { ssize_t w = write(2, m, (size_t)n); (void)w; }
      // DELTA_CNT_CLAMP: experiment - if the entry count is absurd (uninitialised
      // garbage), force it to 0 so the entry-table alloc is tiny and the boot can
      // proceed past the OOM to reveal the next blocker.
      if (kCntClamp && obj >= 0x10000 && cnt > 0x100000)
        *reinterpret_cast<u32 *>(obj + 0x30) = 0;
      gr[REG_RSP] -= 8;
      *reinterpret_cast<u64 *>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  // Allocator-trace trap: handle first so it neither marks s_dumping nor floods
  // the entry marker. After int3 the RIP sits one byte past the hooked entry.
  if (sig == SIGTRAP && g_heapProfAddr && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_heapProfHookCount; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_heapProfHooks[i] + 1)
        continue;
      uintptr_t rsp = (uintptr_t)gr[REG_RSP];
      uintptr_t caller = rsp >= 0x10000 ? *reinterpret_cast<u64 *>(rsp) : 0;
      const u64 size = g_heapProfCountOnly[i] ? 1u : (u64)gr[REG_RDI];
      // The guest's fs base is NOT the thread's real fs (the lifter rewrites
      // guest fs accesses and the host keeps its own TLS there), so ask the
      // backend for the base the guest's own `fs:0` resolves to.
      heapProfRecord(
          caller, size,
          g_heapProfScopeSlot && heapProfScopeDepth(threadFsBase()) == 0);
      g_heapProfHookBytes[i].fetch_add(size, std::memory_order_relaxed);
      g_heapProfHookCalls[i].fetch_add(1, std::memory_order_relaxed);
      gr[REG_RSP] -= 8;  // emulate the displaced `push rbp`
      *reinterpret_cast<u64 *>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;
    }
  }
  if (sig == SIGTRAP && g_allocTraceAddr && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    if ((uintptr_t)gr[REG_RIP] == g_allocTraceAddr + 1) {
      u64 size = (u64)gr[REG_RSI];
      if (size >= g_allocTraceMin) {
        char m[96];
        int n = std::snprintf(m, sizeof(m), "[alloc] %llu bytes (%.1f MB) heap=%llx\n",
                              (unsigned long long)size, size / 1048576.0,
                              (unsigned long long)gr[REG_RDI]);
        if (n > 0) { ssize_t w = write(2, m, (size_t)n); (void)w; }
        // Scan the guest stack for return addresses in a module .text to show
        // which code computed this (garbage) size.
        uintptr_t rsp = (uintptr_t)gr[REG_RSP];
        if (rsp >= 0x10000) {
          auto *sp = reinterpret_cast<uintptr_t *>(rsp);
          int shown = 0;
          for (int i = 0; i < 256 && shown < 8; i++) {
            char sym[200];
            symbolize(sp[i], sym, sizeof(sym));
            if (std::strstr(sym, "(.text)")) {
              char l[256];
              int ln = std::snprintf(l, sizeof(l), "  sp+%-4x %s\n", i * 8, sym);
              if (ln > 0) { ssize_t w = write(2, l, (size_t)ln); (void)w; }
              shown++;
            }
          }
        }
      }
      gr[REG_RSP] -= 8;  // emulate the displaced `push rbp`
      *reinterpret_cast<u64 *>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return true;            // resume at addr+1 (the mov rbp,rsp that follows)
    }
  }
#endif
  return false;
}

// The crash reporter is about to print a fatal dump; flush anything a probe has
// accumulated that would otherwise be lost with the process.
void onFatal() {
  if (g_heapProfAddr)
    heapProfDump();
}

void installThreadProbeHandler(struct sigaction &pa) {
#if defined(__x86_64__) || defined(__aarch64__)
  pa.sa_sigaction = probeHandler;
#else
  (void)pa;
#endif
}

void setAllocTrace(uintptr_t addr, u64 minSize) {
  g_allocTraceAddr = addr;
  if (minSize)
    g_allocTraceMin = minSize;
}

void setCntTrace(uintptr_t addr) { g_cntTraceAddr = addr; }
void setFatalTrace(uintptr_t addr) { g_fatalTraceAddr = addr; }
void setHdrTrace(uintptr_t addr) {
  if (g_hdrTraceCount < 8) g_hdrTraceAddrs[g_hdrTraceCount++] = addr;
}
void setRdoffFix(uintptr_t addr) { g_rdoffAddr = addr; }
void setSkipFn(uintptr_t addr) { if (g_skipFnCount < 8) g_skipFnAddrs[g_skipFnCount++] = addr; }
void setCallSkip(uintptr_t addr, long raxVal, int insnLen) {
#if defined(__x86_64__)
  if (g_callSkipCount >= 8) return;
  g_callSkipAddrs[g_callSkipCount] = addr;
  g_callSkipVals[g_callSkipCount] = raxVal;
  g_callSkipLens[g_callSkipCount] = insnLen;
  g_callSkipCount++;
#else
  (void)addr; (void)raxVal; (void)insnLen;
#endif
}
void setOrderTrace(uintptr_t addr, const char *label) {
  if (g_orderCount >= kOrderMax)
    return;
  if (g_orderCount == 0)
    clock_gettime(CLOCK_MONOTONIC, &g_orderStart);
  g_orderAddrs[g_orderCount] = addr;
  g_orderLabels[g_orderCount] = label;
  g_orderCount++;
}
void setRetTrace(uintptr_t addr, const char *label, bool isTest) {
  if (g_retCount < 8) {
    g_retAddrs[g_retCount] = addr;
    g_retLabels[g_retCount] = label;
    g_retIsTest[g_retCount] = isTest;
    g_retCount++;
  }
}

void setFnWatch(uintptr_t addr, const char *label) {
  if (g_fnWatchCount >= kFnWatchMax)
    return;
  g_fnWatchAddrs[g_fnWatchCount] = addr;
  g_fnWatchLabels[g_fnWatchCount] = label;
  g_fnWatchHits[g_fnWatchCount].store(0, std::memory_order_relaxed);
  g_fnWatchCount++;
}

void setFnArgs(uintptr_t addr, const char *label, const u64 *offsets,
               int noffsets) {
  if (g_fnArgsCount >= kFnArgsMax)
    return;
  if (noffsets > kFnArgsOffsMax)
    noffsets = kFnArgsOffsMax;
  g_fnArgsAddrs[g_fnArgsCount] = addr;
  g_fnArgsLabels[g_fnArgsCount] = label;
  g_fnArgsNoffs[g_fnArgsCount] = noffsets;
  for (int i = 0; i < noffsets; i++)
    g_fnArgsOffs[g_fnArgsCount][i] = offsets[i];
  g_fnArgsHits[g_fnArgsCount].store(0, std::memory_order_relaxed);
  g_fnArgsCount++;
}

// DELTA_GUEST_WPROT=<hex addr>:<hex bytes>[:<ms>]: name every writer of a guest
// range. Waits until the range is mapped (a title allocates its pools well after
// startup), makes it read-only, and lets the SIGSEGV path report the faulting
// instruction and reopen that page. Memory that stays empty while the title
// behaves as if it filled it either has no writer at all or one writing
// elsewhere, and only the fault distinguishes those.
//
// The trap RE-ARMS every `ms` instead of firing once. Each fault reopens its own
// page so the guest makes progress, which used to end the watch after a single
// report for a range the title writes continuously -- one sample cannot tell a
// one-time initialiser from a per-frame producer, and it certainly cannot follow
// a value from one buffer to the next. Re-arming keeps the cost at roughly one
// fault per page per interval while turning the watch into a stream.
void startWriteWatch(uintptr_t addr, size_t bytes, unsigned everyMs,
                     bool trapReads, bool singleStep) {
  if (!addr || !bytes)
    return;
#if !defined(__x86_64__)
  singleStep = false;
#endif
  const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
  const uintptr_t base = addr & ~((uintptr_t)pgsz - 1);
  const size_t span = (addr + bytes - base + pgsz - 1) & ~(pgsz - 1);
  if (singleStep) {
    g_wprotBase = base;
    g_wprotLen = span;
    g_wprotRegs = trapReads;
    g_wprotStep = true;
    g_wprotReportBase = addr;
    g_wprotReportLen = bytes;
    if (::mprotect(reinterpret_cast<void *>(base), span,
                   trapReads ? PROT_NONE : PROT_READ) == 0) {
      BASE_LOGI("wprot", "single-stepping {:#x}+{:#x}, reporting {:#x}+{:#x} ({})",
                (unsigned long)base, (unsigned long)span, (unsigned long)addr,
                (unsigned long)bytes, trapReads ? "reads+writes" : "writes");
    }
    return;
  }
  std::thread([addr, bytes, everyMs, trapReads] {
    const long pgsz = sysconf(_SC_PAGESIZE);
    const uintptr_t base = addr & ~((uintptr_t)pgsz - 1);
    const size_t span =
        (addr + bytes - base + (size_t)pgsz - 1) & ~((size_t)pgsz - 1);
    bool announced = false;
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(everyMs));
      unsigned char vec = 0;
      if (mincore(reinterpret_cast<void *>(base), 1, &vec) != 0)
        continue;  // not mapped yet
      if (::mprotect(reinterpret_cast<void *>(base), span,
                     trapReads ? PROT_NONE : PROT_READ) != 0)
        continue;
      g_wprotBase = base;
      g_wprotLen = span;
      g_wprotRegs = trapReads;
      g_wprotStep = false;
      g_wprotReportBase = base;
      g_wprotReportLen = span;
      if (!announced) {
        announced = true;
        BASE_LOGI("wprot", "watching {:#x}+{:#x} ({}), re-armed every {}ms",
                  (unsigned long)base, (unsigned long)span,
                  trapReads ? "reads+writes" : "writes", everyMs);
      }
    }
  }).detach();
}

// DELTA_GUEST_WHIST=<hex addr>:<hex bytes>[:<ms>]: write census over a pool too
// big to watch one write at a time. Re-arms the whole range read-only every
// `ms`, so each report says which 16 MiB slices the guest wrote in that window
// and which instructions wrote them. Answers "the title fills its video pool
// somewhere, but where, and from what code" in one run.
void startWriteHist(uintptr_t addr, size_t bytes, unsigned everyMs) {
  if (!addr || !bytes)
    return;
  std::thread([addr, bytes, everyMs] {
    const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t base = addr & ~(pgsz - 1);
    const size_t span = (bytes + pgsz - 1) & ~(pgsz - 1);
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      unsigned char vec = 0;
      if (mincore(reinterpret_cast<void *>(base), 1, &vec) == 0)
        break;
    }
    g_whistBase = base;
    g_whistLen = span;
    BASE_LOGI("whist", "census on {:#x}+{:#x} every {}ms", (unsigned long)base,
              (unsigned long)span, everyMs);
    unsigned sinceReport = 0;
    for (;;) {
      ::mprotect(reinterpret_cast<void *>(base), span, PROT_READ);
      std::this_thread::sleep_for(std::chrono::milliseconds(everyMs));
      if ((sinceReport += everyMs) < 4000)
        continue;
      sinceReport = 0;
      std::string map;
      for (size_t off = 0; off < span; off += kWhistGranule) {
        const u32 n = g_whistBucket[off / kWhistGranule].load();
        map += n == 0 ? '_' : n < 10 ? '.' : n < 100 ? '+' : '#';
      }
      BASE_LOGI("whist", "{}", map.c_str());
      for (int i = 0; i < kWhistSites; i++) {
        const uintptr_t rip = g_whistSite[i].load();
        if (!rip)
          break;
        char sym[192], csym[192];
        symbolize(rip, sym, sizeof(sym));
        symbolize(g_whistSiteCaller[i].load(), csym, sizeof(csym));
        BASE_LOGI("whist", "  {:8} {} <- {}", g_whistSiteHits[i].load(), sym,
                  csym);
      }
      // Never let the site list read as the complete set of writers when some
      // faults could not be attributed to a guest instruction.
      if (const u64 unattributed = g_whistUnattributed.load())
        BASE_LOGI("whist", "  {:8} <unattributed>",
                  (unsigned long long)unattributed);
    }
  }).detach();
}

// DELTA_GUEST_POPCNT=<hex addr>:<hex bytes>: report the population count of a
// guest bitmap every 2s. A title's own allocator keeps its free/used map as a
// bitmap, and "does it drain, or was it never filled" is the question a single
// dump at the crash cannot answer.
void startPopcntPrinter(uintptr_t addr, size_t bytes, unsigned everyMs) {
  if (!addr || !bytes)
    return;
  std::thread([addr, bytes, everyMs] {
    const long pgsz = sysconf(_SC_PAGESIZE);
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(everyMs));
      unsigned char vec = 0;
      if (mincore(reinterpret_cast<void *>(addr & ~((uintptr_t)pgsz - 1)), 1,
                  &vec) != 0)
        continue;
      const auto *w = reinterpret_cast<const u64 *>(addr);
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
  }).detach();
}

// DELTA_GUEST_SUMWATCH=<slot>:<off>:<stride>:<count>[:<ms>] (all hex but count):
// dereference a guest pointer SLOT, then report the individual u32 counters at
// obj+off+i*stride and their sum, on an interval. An engine's "work remaining"
// is usually a set of per-queue counters behind a singleton pointer, and whether
// it moves is the difference between "stalled" and "just slow".
void startSumWatchPrinter(uintptr_t slot, size_t off, size_t stride, int count,
                          unsigned everyMs) {
  if (!slot || count <= 0 || count > 32)
    return;
  std::thread([slot, off, stride, count, everyMs] {
    const long pgsz = sysconf(_SC_PAGESIZE);
    auto readable = [pgsz](uintptr_t a) {
      unsigned char v = 0;
      return a >= 0x10000 &&
             mincore(reinterpret_cast<void *>(a & ~((uintptr_t)pgsz - 1)), 1,
                     &v) == 0;
    };
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(everyMs));
      if (!readable(slot))
        continue;
      const uintptr_t obj = *reinterpret_cast<const uintptr_t *>(slot);
      if (!readable(obj))
        continue;
      base::String line;
      base::FormatTo(line, "{:#x}:", (unsigned long)obj);
      u64 sum = 0;
      for (int i = 0; i < count; i++) {
        const u32 v =
            *reinterpret_cast<const u32 *>(obj + off + (size_t)i * stride);
        sum += v;
        base::FormatTo(line, " {}", v);
      }
      BASE_LOGI("sumwatch", "{} = {}", line.c_str(), (unsigned long long)sum);
    }
  }).detach();
}

// DELTA_POOLMAP=<hex addr>:<hex bytes>[:<ms>]: survey a multi-GB guest pool
// without touching it. mincore reports which pages the guest has actually
// faulted in, so a region the title claims to have filled but never wrote is
// visible as a hole; only resident pages are then read for a non-zero test.
void startPoolMap(uintptr_t addr, size_t bytes, unsigned everyMs) {
  if (!addr || !bytes)
    return;
  std::thread([addr, bytes, everyMs] {
    const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t base = addr & ~(pgsz - 1);
    const size_t span = (bytes + pgsz - 1) & ~(pgsz - 1);
    const size_t npages = span / pgsz;
    const size_t granule = 16u << 20;              // one report column
    const size_t pagesPerGranule = granule / pgsz;
    std::vector<unsigned char> vec(npages);
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(everyMs));
      if (mincore(reinterpret_cast<void *>(base), span, vec.data()) != 0)
        continue;
      size_t resident = 0, nonzero = 0;
      std::string map;
      for (size_t g = 0; g * pagesPerGranule < npages; g++) {
        size_t res = 0, nz = 0;
        for (size_t i = g * pagesPerGranule;
             i < npages && i < (g + 1) * pagesPerGranule; i++) {
          if (!(vec[i] & 1))
            continue;
          res++;
          const auto *w = reinterpret_cast<const u64 *>(base + i * pgsz);
          for (size_t j = 0; j < pgsz / 8; j++)
            if (w[j]) { nz++; break; }
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
  }).detach();
}

// DELTA_POOLMAP=all[:<ms>]: the same survey over every guest mapping, read out
// of /proc/self/maps. "The title wrote a gigabyte somewhere, but not where the
// GPU reads" is only answerable with the whole address space in one view.
void startPoolCensus(unsigned everyMs) {
  std::thread([everyMs] {
    const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(everyMs));
      FILE *f = std::fopen("/proc/self/maps", "r");
      if (!f)
        return;
      char line[512];
      std::vector<unsigned char> vec;
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
        if (mincore(reinterpret_cast<void *>(lo), span, vec.data()) != 0)
          continue;
        size_t res = 0, nz = 0;
        for (size_t i = 0; i < vec.size(); i++) {
          if (!(vec[i] & 1))
            continue;
          res++;
          const auto *w = reinterpret_cast<const u64 *>(lo + i * pgsz);
          for (size_t j = 0; j < pgsz / 8; j++)
            if (w[j]) { nz++; break; }
        }
        BASE_LOGI("census", "{:012x}+{:09x} {:.1f} MB resident={:.1f} MB "
                            "nonzero={:.1f} MB",
                  lo, span, span / 1048576.0, res * pgsz / 1048576.0,
                  nz * pgsz / 1048576.0);
      }
      std::fclose(f);
    }
  }).detach();
}

// DELTA_MEMDUMP=<hex addr>:<hex bytes>:<ms>:<path>[,...]: snapshot a guest range
// to a file. Only resident pages are read, so dumping a sparse pool does not
// commit it; the holes come out as zeros.
void startMemDump(uintptr_t addr, size_t bytes, unsigned afterMs,
                  const char *path) {
  if (!addr || !bytes || !path)
    return;
  std::string out(path);
  std::thread([addr, bytes, afterMs, out] {
    std::this_thread::sleep_for(std::chrono::milliseconds(afterMs));
    const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t base = addr & ~(pgsz - 1);
    const size_t span = (bytes + pgsz - 1) & ~(pgsz - 1);
    std::vector<unsigned char> vec(span / pgsz);
    if (mincore(reinterpret_cast<void *>(base), span, vec.data()) != 0) {
      BASE_LOGI("memdump", "{:#x} not mapped", (unsigned long)base);
      return;
    }
    FILE *f = std::fopen(out.c_str(), "wb");
    if (!f)
      return;
    std::vector<unsigned char> zero(pgsz, 0);
    size_t resident = 0;
    for (size_t i = 0; i < vec.size(); i++) {
      if (vec[i] & 1) {
        resident++;
        std::fwrite(reinterpret_cast<const void *>(base + i * pgsz), 1, pgsz, f);
      } else {
        std::fwrite(zero.data(), 1, pgsz, f);
      }
    }
    std::fclose(f);
    BASE_LOGI("memdump", "{:#x}+{:#x} -> {} ({}/{} resident pages)",
              (unsigned long)base, (unsigned long)span, out.c_str(), resident,
              vec.size());
  }).detach();
}

void startFnWatchPrinter() {
  static std::atomic<bool> started{false};
  bool exp = false;
  if (!started.compare_exchange_strong(exp, true))
    return;
  std::thread([] {
    u64 last[kFnWatchMax] = {0};
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      base::String line;
      base::FormatTo(line, "[fnwatch]");
      for (int i = 0; i < g_fnWatchCount; i++) {
        u64 h = g_fnWatchHits[i].load(std::memory_order_relaxed);
        base::FormatTo(line, " {}={}(+{})", g_fnWatchLabels[i],
                       (unsigned long long)h, (unsigned long long)(h - last[i]));
        last[i] = h;
      }
      BASE_LOGI("fnwatch", "{}", line.c_str());
    }
  }).detach();
}


}  // namespace krnl::probe
