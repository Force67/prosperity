/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#define _GNU_SOURCE
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <ucontext.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <pthread.h>

#include "crash.h"
#include "module.h"
#include "proc.h"
#include "vfs.h"
#include "cpu/cpu_backend.h"
#include <logger/logger.h>

namespace krnl {
const char *syscall_getname(uint32_t idx); // name_table.cpp

// Resolve a host address to "<module>+0x<off> (<seg>)" by scanning loaded module
// images, so a guest fault points straight at a guest module offset.
void symbolize(uintptr_t addr, char *out, size_t n) {
  if (auto *proc = proc::getActive()) {
    for (auto &mod : proc->getModuleList()) {
      auto &mi = mod->getInfo();
      auto *t = mi.textSeg.addr;
      auto *d = mi.dataSeg.addr;
      if (t && addr >= (uintptr_t)t && addr < (uintptr_t)t + mi.textSeg.size) {
        std::snprintf(out, n, "%s+%#lx (.text)", mi.name.c_str(),
                      addr - (uintptr_t)t);
        return;
      }
      if (d && addr >= (uintptr_t)d && addr < (uintptr_t)d + mi.dataSeg.size) {
        std::snprintf(out, n, "%s+%#lx (.data)", mi.name.c_str(),
                      addr - (uintptr_t)d);
        return;
      }
    }
  }
  std::snprintf(out, n, "%#lx (??)", addr);
}

// Walk the rbp frame chain (guest code keeps frame pointers) and symbolize each
// return address. Bounded and range-checked so a bad frame can't loop or fault.
// Works for both backends: guest frames are x86-64 frames in identity-mapped
// memory, so the walk is plain pointer reads on either host arch.
static void backtrace(uintptr_t rbp) {
  std::fprintf(stderr, "  --- backtrace ---\n");
  for (int i = 0; i < 32; i++) {
    if (rbp < 0x10000 || (rbp & 7))
      break;
    auto *frame = reinterpret_cast<uintptr_t *>(rbp);
    uintptr_t next = frame[0];
    uintptr_t ret = frame[1];
    if (!ret)
      break;
    char sym[256];
    symbolize(ret, sym, sizeof(sym));
    std::fprintf(stderr, "  #%-2d %016lx  %s\n", i, ret, sym);
    if (next <= rbp)  // frames grow upward; stop if it doesn't
      break;
    rbp = next;
  }
}

// ---------------------------------------------------------------------------
// SOTC AllocationTracker walk (crash-scoped diagnostic; only runs on faults
// inside that title's tracker code). On a fault inside the Shadow_Shipping
// eboot's slot-21 "untrack on
// free" methods (fn @+0x18920 CPU tracker, +0x8d930 GPU/renderer tracker), the
// record lookup @+0x47ab0 returned NULL and the caller unconditionally read
// [rec+0x58]/[rec+0x50] -> #GP. We walk the tracker's own record list to answer:
// is the freed KEY absent, present at a different base, or in the other tracker?
//
// Object layout (verified from disasm of +0x18920 / +0x47ab0 / +0x476f0):
//   tracker+0x28  listener; +0x38  embedded list SENTINEL node;
//   tracker+0x48  == sentinel.next (first record); +0x70 spinlock;
//   +0x80 total bytes; +0x90 count.
//   record node:  +0x08 prev, +0x10 next (circular list threaded here),
//                 +0x58 size (interval-end field used by the lookup),
//                 +0x60 base/key. (+0x50 is a second size copy on the GPU var.)
// Recovery of (tracker,key) at fault time is done from RSP, NOT the callee-saved
// regs (FEX reconstruction of those is unreliable): both fns push
// rbp;r15;r14;r13;r12;rbx;rax with NO further stack alloc before the fault, so
//   [rsp+0x18]=saved r13=TRACKER   [rsp+0x20]=saved r14=KEY.
namespace {
inline bool trkMincore(uint64_t va) {
  if (va < 0x10000) return false;
  long pg = sysconf(_SC_PAGESIZE);
  unsigned char vec = 0;
  void *pa = reinterpret_cast<void *>(va & ~((uint64_t)pg - 1));
  return mincore(pa, 1, &vec) == 0;
}
inline bool trkRd64(uint64_t va, uint64_t &out) {
  if (!trkMincore(va) || !trkMincore(va + 7)) return false;
  out = *reinterpret_cast<const uint64_t *>(va);
  return true;
}
// Walk one tracker's circular record list; report count/bytes, whether `key`
// is covered by a record, and the 8 records nearest to `key` by |base-key|.
// Returns true if `key` fell inside some record's [base,base+size).
bool sotcWalkTracker(uint64_t tracker, uint64_t key, const char *tag) {
  std::fprintf(stderr, "  [trkwalk:%s] tracker=%#llx key=%#llx\n", tag,
               (unsigned long long)tracker, (unsigned long long)key);
  if (!trkMincore(tracker) || !trkMincore(tracker + 0x98)) {
    std::fprintf(stderr, "  [trkwalk:%s]   tracker not mapped -- skip\n", tag);
    return false;
  }
  uint64_t sentinel = tracker + 0x38;
  uint64_t first = 0, hdrCount = 0, hdrBytes = 0, listener = 0;
  trkRd64(tracker + 0x48, first);
  trkRd64(tracker + 0x90, hdrCount);
  trkRd64(tracker + 0x80, hdrBytes);
  trkRd64(tracker + 0x28, listener);
  std::fprintf(stderr,
               "  [trkwalk:%s]   listener=%#llx first=%#llx count(+0x90)=%llu "
               "bytes(+0x80)=%#llx\n",
               tag, (unsigned long long)listener, (unsigned long long)first,
               (unsigned long long)hdrCount, (unsigned long long)hdrBytes);
  // Nearest-8 online selection by absolute distance from key.
  uint64_t nb[8], ns[8], nd[8];
  for (int i = 0; i < 8; i++) { nb[i] = ns[i] = 0; nd[i] = ~0ull; }
  uint64_t node = first, walked = 0, sumSize = 0;
  bool covered = false, coverPrinted = false;
  for (; walked < 200000; walked++) {
    if (node == sentinel || node == 0) break;
    if (!trkMincore(node) || !trkMincore(node + 0x60 + 7)) {
      std::fprintf(stderr, "  [trkwalk:%s]   node %#llx unmapped -- stop\n", tag,
                   (unsigned long long)node);
      break;
    }
    uint64_t base = 0, size = 0, next = 0;
    trkRd64(node + 0x60, base);
    trkRd64(node + 0x58, size);
    trkRd64(node + 0x10, next);
    sumSize += size;
    if (base <= key && key < base + size) {
      covered = true;
      if (!coverPrinted) {
        std::fprintf(stderr,
                     "  [trkwalk:%s]   *** COVER: rec %#llx base=%#llx size=%#llx "
                     "end=%#llx contains key ***\n",
                     tag, (unsigned long long)node, (unsigned long long)base,
                     (unsigned long long)size, (unsigned long long)(base + size));
        coverPrinted = true;
      }
    }
    uint64_t d = base > key ? base - key : key - base;
    // insert into nearest-8 if closer than the current worst
    int worst = 0;
    for (int i = 1; i < 8; i++) if (nd[i] > nd[worst]) worst = i;
    if (d < nd[worst]) { nd[worst] = d; nb[worst] = base; ns[worst] = size; }
    node = next;
  }
  std::fprintf(stderr,
               "  [trkwalk:%s]   walked %llu records, sum(size)=%#llx, key %s\n",
               tag, (unsigned long long)walked, (unsigned long long)sumSize,
               covered ? "IS COVERED" : "is NOT covered by any record");
  // sort nearest-8 by distance (tiny insertion sort)
  for (int i = 0; i < 8; i++)
    for (int j = i + 1; j < 8; j++)
      if (nd[j] < nd[i]) {
        uint64_t t;
        t = nd[i]; nd[i] = nd[j]; nd[j] = t;
        t = nb[i]; nb[i] = nb[j]; nb[j] = t;
        t = ns[i]; ns[i] = ns[j]; ns[j] = t;
      }
  std::fprintf(stderr, "  [trkwalk:%s]   8 nearest records to key (by |base-key|):\n", tag);
  for (int i = 0; i < 8; i++) {
    if (nd[i] == ~0ull) break;
    long long signedDelta = (long long)(nb[i] - key);
    std::fprintf(stderr,
                 "  [trkwalk:%s]     base=%#llx size=%#llx end=%#llx  base-key=%+lld (%#llx)\n",
                 tag, (unsigned long long)nb[i], (unsigned long long)ns[i],
                 (unsigned long long)(nb[i] + ns[i]), signedDelta,
                 (unsigned long long)nd[i]);
  }
  return covered;
}
}  // namespace

// Per-syscall call counter, filled by the lv2 trampoline under DELTA_SCHIST.
extern "C" uint64_t g_sysHist[1024];

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
  std::fprintf(stderr, "[probe] tid=%ld rip=%016llx %s\n", (long)gettid(),
               (unsigned long long)gr[REG_RIP], rip);
  // GPRs too: a thread caught in a busy-wait only makes sense with the address
  // and value it is polling.
  std::fprintf(stderr,
               "[probe]   rax=%016llx rbx=%016llx rcx=%016llx rdx=%016llx\n"
               "[probe]   rsi=%016llx rdi=%016llx r8 =%016llx r9 =%016llx\n"
               "[probe]   r12=%016llx r13=%016llx r14=%016llx r15=%016llx\n",
               (unsigned long long)gr[REG_RAX], (unsigned long long)gr[REG_RBX],
               (unsigned long long)gr[REG_RCX], (unsigned long long)gr[REG_RDX],
               (unsigned long long)gr[REG_RSI], (unsigned long long)gr[REG_RDI],
               (unsigned long long)gr[REG_R8], (unsigned long long)gr[REG_R9],
               (unsigned long long)gr[REG_R12], (unsigned long long)gr[REG_R13],
               (unsigned long long)gr[REG_R14], (unsigned long long)gr[REG_R15]);
  uintptr_t rsp = gr[REG_RSP];
  if (rsp >= 0x10000) {
    auto *sp = reinterpret_cast<uintptr_t *>(rsp);
    int printed = 0;
    for (int i = 0; i < 512 && printed < 6; i++) {
      char sym[256];
      symbolize(sp[i], sym, sizeof(sym));
      if (std::strstr(sym, "(.text)")) {
        std::fprintf(stderr, "[probe]   sp+%-4x %s\n", i * 8, sym);
        printed++;
      }
    }
  }
  // DELTA_SCHIST syscall histogram (lv2.cpp counts each syscall in its trampoline).
  // Dump the non-zero counts so a slow/wedged title's hammered syscalls are visible
  // -- the only profiler available (perf/strace/proc-mem are yama-blocked here).
  // Signal ONE thread to avoid interleaved output from concurrent handlers.
  bool any = false;
  for (int i = 0; i < 1024; i++) {
    if (g_sysHist[i] > 100) {  // skip noise
      if (!any) { std::fprintf(stderr, "[schist] syscall counts:\n"); any = true; }
      std::fprintf(stderr, "[schist]   %4d %-28s %llu\n", i, syscall_getname(i),
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
uint64_t g_allocTraceMin = 0x1000000;  // 16 MiB
// DELTA_HEAP_PROF: aggregate operator-new/malloc (size in rdi) by guest caller.
// Fixed open-addressing table, claimed lock-free from the trap handler (called
// concurrently from every guest thread). SIGUSR1 dumps the top sites by bytes.
uintptr_t g_heapProfAddr = 0;  // non-zero once any hook is armed
static constexpr int kHeapProfMaxHooks = 24;
static uintptr_t g_heapProfHooks[kHeapProfMaxHooks];
static int g_heapProfHookCount = 0;
static std::atomic<uint64_t> g_heapProfHookBytes[kHeapProfMaxHooks];
static std::atomic<uint64_t> g_heapProfHookCalls[kHeapProfMaxHooks];
namespace {
constexpr uint32_t kHeapProfSlots = 16384;
struct HeapProfSlot {
  std::atomic<uintptr_t> caller{0};
  std::atomic<uint64_t> bytes{0};
  std::atomic<uint64_t> count{0};
};
HeapProfSlot g_heapProf[kHeapProfSlots];
std::atomic<uint64_t> g_heapProfTotal{0};
void heapProfRecord(uintptr_t caller, uint64_t size) {
  uint32_t h = static_cast<uint32_t>((caller * 2654435761u) >> 13) & (kHeapProfSlots - 1);
  for (uint32_t i = 0; i < kHeapProfSlots; i++) {
    uint32_t s = (h + i) & (kHeapProfSlots - 1);
    uintptr_t c = g_heapProf[s].caller.load(std::memory_order_relaxed);
    if (c == caller) {
      g_heapProf[s].bytes.fetch_add(size, std::memory_order_relaxed);
      g_heapProf[s].count.fetch_add(1, std::memory_order_relaxed);
      break;
    }
    if (c == 0) {
      uintptr_t expected = 0;
      if (g_heapProf[s].caller.compare_exchange_strong(expected, caller,
                                                       std::memory_order_relaxed)) {
        g_heapProf[s].bytes.fetch_add(size, std::memory_order_relaxed);
        g_heapProf[s].count.fetch_add(1, std::memory_order_relaxed);
        break;
      }
      if (expected == caller) {
        g_heapProf[s].bytes.fetch_add(size, std::memory_order_relaxed);
        g_heapProf[s].count.fetch_add(1, std::memory_order_relaxed);
        break;
      }
    }
  }
  g_heapProfTotal.fetch_add(size, std::memory_order_relaxed);
}
void heapProfDump() {
  std::fprintf(stderr, "[heapprof] total=%llu bytes (%.1f MB) across sites; top by bytes:\n",
               (unsigned long long)g_heapProfTotal.load(),
               g_heapProfTotal.load() / 1048576.0);
  for (int i = 0; i < g_heapProfHookCount; i++) {
    uint64_t b = g_heapProfHookBytes[i].load(std::memory_order_relaxed);
    std::fprintf(stderr, "[heapprof]  hook[%d] %#lx: %8.1f MB  %8llu calls\n", i,
                 (unsigned long)g_heapProfHooks[i], b / 1048576.0,
                 (unsigned long long)g_heapProfHookCalls[i].load(std::memory_order_relaxed));
  }
  // Select top 20 by bytes without allocating (linear passes).
  uint64_t prevBytes = ~0ull;
  uintptr_t prevCaller = 0;
  for (int rank = 0; rank < 20; rank++) {
    uint64_t bestB = 0; uint32_t bestS = kHeapProfSlots;
    for (uint32_t s = 0; s < kHeapProfSlots; s++) {
      uint64_t b = g_heapProf[s].bytes.load(std::memory_order_relaxed);
      if (b == 0) continue;
      uintptr_t c = g_heapProf[s].caller.load(std::memory_order_relaxed);
      bool below = b < prevBytes || (b == prevBytes && c > prevCaller);
      if (below && b >= bestB) { bestB = b; bestS = s; }
    }
    if (bestS == kHeapProfSlots) break;
    uintptr_t c = g_heapProf[bestS].caller.load(std::memory_order_relaxed);
    uint64_t cnt = g_heapProf[bestS].count.load(std::memory_order_relaxed);
    char sym[200];
    symbolize(c, sym, sizeof(sym));
    std::fprintf(stderr, "[heapprof]  %8.1f MB  %8llu calls  %s\n",
                 bestB / 1048576.0, (unsigned long long)cnt, sym);
    prevBytes = bestB; prevCaller = c;
  }
  std::fflush(stderr);
}
}  // namespace
void setHeapProf(uintptr_t addr) {
  if (g_heapProfHookCount < kHeapProfMaxHooks)
    g_heapProfHooks[g_heapProfHookCount++] = addr;
  g_heapProfAddr = addr;  // any non-zero arms the SIGTRAP path
}
// Throttle to one dump per SIGUSR1 burst (every thread gets the signal).
static void heapProfDumpOnce() {
  static std::atomic<uint64_t> last{0};
  timespec t{};
  clock_gettime(CLOCK_MONOTONIC, &t);
  uint64_t now = (uint64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
  uint64_t prev = last.load(std::memory_order_relaxed);
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
void markManifestFd(uint32_t fd, bool v) { if (fd < 8192) g_manifestFd[fd] = v; }
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
struct NullGuard {
  uintptr_t addr;
  int greg;  // REG_* index to zero
  int len;   // faulting instruction length
};
NullGuard g_nullGuards[16] = {};
int g_nullGuardCount = 0;

// DELTA_PS5_DCBWATCH call-order trace (see crash.h).
static constexpr int kOrderMax = 12;
static uintptr_t g_orderAddrs[kOrderMax];
static const char *g_orderLabels[kOrderMax];
static int g_orderCount = 0;
static timespec g_orderStart;
static uintptr_t g_retAddrs[8];
static const char *g_retLabels[8];
static int g_retCount = 0;

// DELTA_PS5_GLYPHGUARD call-skip (see crash.h): int3 planted over a blocking
// vtable-dispatch call; on hit, inject rax and step past the whole call insn.
static uintptr_t g_callSkipAddrs[8];
static long g_callSkipVals[8];
static int g_callSkipLens[8];
static int g_callSkipCount = 0;

static void crashHandler(int sig, siginfo_t *si, void *ucv) {
#if defined(__x86_64__)
  if (sig == SIGTRAP && g_retCount && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_retCount; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_retAddrs[i] + 1)
        continue;
      uint32_t eax = (uint32_t)gr[REG_RAX];
      // DELTA_PS5_DCBFORCE: force a failing graphics-init sub-call to report
      // success (SCE_OK) so the run-once init 0x69e720 completes and the engine
      // creates its DrawCommandBuffer -- lets us measure how far the boot gets
      // when the (obfuscated) libSceAgc call is treated as succeeding.
      static const int force = std::getenv("DELTA_PS5_DCBFORCE") != nullptr;
      if (force && eax) {
        gr[REG_RAX] = 0;
        eax = 0;
      }
      char m[128];
      int n = std::snprintf(m, sizeof(m), "[ret] %s eax=%#x\n", g_retLabels[i], eax);
      if (n > 0) { ssize_t w = write(2, m, (size_t)n); (void)w; }
      gr[REG_RBX] = eax;                  // emulate `mov ebx,eax` (zero-extends)
      gr[REG_RIP] = g_retAddrs[i] + 2;    // resume past the 2-byte `89 c3`
      return;
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
      return;
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
      uintptr_t caller = rsp >= 0x10000 ? *reinterpret_cast<uint64_t *>(rsp) : 0;
      char csym[200];
      symbolize(caller, csym, sizeof(csym));
      char m[320];
      int n = std::snprintf(m, sizeof(m), "[order t=%ldms tid=%ld] %s  <- %s\n",
                            ms, (long)gettid(), g_orderLabels[i], csym);
      if (n > 0) { ssize_t w = write(2, m, (size_t)n); (void)w; }
      gr[REG_RSP] -= 8;  // emulate the displaced `push rbp`
      *reinterpret_cast<uint64_t *>(gr[REG_RSP]) = (uint64_t)gr[REG_RBP];
      return;
    }
  }
  if (sig == SIGTRAP && g_skipFnCount && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    for (int si2 = 0; si2 < g_skipFnCount; si2++) {
      if ((uintptr_t)gr[REG_RIP] == g_skipFnAddrs[si2] + 1) {
        uintptr_t rsp = (uintptr_t)gr[REG_RSP];
        gr[REG_RIP] = *reinterpret_cast<uint64_t *>(rsp);  // return addr
        gr[REG_RSP] = rsp + 8;
        gr[REG_RAX] = 0;
        return;
      }
    }
  }
  if (sig == SIGTRAP && g_rdoffAddr && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    if ((uintptr_t)gr[REG_RIP] == g_rdoffAddr + 1) {
      uint32_t fd = (uint32_t)gr[REG_RSI];
      bool mf = (fd < 8192 && g_manifestFd[fd]);
      if (std::getenv("DELTA_RDOFF_TRACE")) {
        char m[128];
        int n = std::snprintf(m, sizeof(m),
                              "[rdoff] fd=%u off=%lld nbytes=%lld buf=%llx manifest=%d\n",
                              fd, (long long)(int32_t)gr[REG_RDX],
                              (long long)(int32_t)gr[REG_RCX],
                              (unsigned long long)gr[REG_R8], mf);
        if (n > 0) { ssize_t w = write(2, m, (size_t)n); (void)w; }
      }
      // DELTA_RDOFF_NOFIX: observe-only (log requests, don't rewrite offsets).
      static const bool nofix = std::getenv("DELTA_RDOFF_NOFIX") != nullptr;
      if (mf && !nofix)
        gr[REG_RDX] = 0;  // force manifest read offset to 0 (read from start)
      gr[REG_RSP] -= 8;   // emulate push rbp
      *reinterpret_cast<uint64_t *>(gr[REG_RSP]) = (uint64_t)gr[REG_RBP];
      return;
    }
  }
  if (sig == SIGTRAP && g_hdrTraceCount && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    bool hit = false;
    for (int hi = 0; hi < g_hdrTraceCount; hi++)
      if ((uintptr_t)gr[REG_RIP] == g_hdrTraceAddrs[hi] + 1) { hit = true; break; }
    if (hit) {
      uint64_t parent = (uint64_t)gr[REG_RDI];
      uint64_t hdr = 0, obj = 0;
      uint32_t magic = 0, cnt = 0;
      char nm[48] = {0};
      if (parent >= 0x10000) {
        hdr = *reinterpret_cast<uint64_t *>(parent + 0x8);
        obj = *reinterpret_cast<uint64_t *>(parent + 0x10);
        // DELTA_HDR_WAIT: test the producer-consumer-race hypothesis. If the
        // manifest header buffer isn't filled yet (magic != "TAFS"), block this
        // (consumer) thread to let the worker thread's read+copy complete.
        static const bool waitMode = std::getenv("DELTA_HDR_WAIT") != nullptr;
        if (waitMode && hdr >= 0x10000) {
          for (int i = 0; i < 2000; i++) {
            if (*reinterpret_cast<volatile uint32_t *>(hdr) == 0x53464154u)
              break;
            timespec ts{0, 200000};  // 0.2ms
            nanosleep(&ts, nullptr);
          }
        }
        if (hdr >= 0x10000) {
          magic = *reinterpret_cast<uint32_t *>(hdr);
          cnt = *reinterpret_cast<uint32_t *>(hdr + 0xc);
        }
        if (obj >= 0x10000) {
          const char *s = reinterpret_cast<const char *>(obj + 0x5c);
          int j = 0; for (; j < 47 && s[j] >= 0x20 && s[j] <= 0x7e; j++) nm[j] = s[j];
          nm[j] = 0;
        }
        // DELTA_HDR_FILL: bypass the racy async manifest reader by copying the
        // real (cached) manifest bytes straight into the header buffer, so the
        // count-setter reads the correct count + entry table for THIS archive.
        static const bool fill = std::getenv("DELTA_HDR_FILL") != nullptr;
        if (fill && hdr >= 0x10000 && nm[0]) {
          auto *h = reinterpret_cast<uint8_t *>(hdr);
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
            *reinterpret_cast<uint32_t *>(h + 4) = 3;     // version
            *reinterpret_cast<uint32_t *>(h + 0x10) = 7;  // strlen("orbis-w")
            std::memcpy(h + 0x14, "orbis-w", 7);
          }
          magic = *reinterpret_cast<uint32_t *>(h);
          cnt = *reinterpret_cast<uint32_t *>(h + 0xc);
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
        uint64_t vt = *reinterpret_cast<uint64_t *>(obj);
        uint64_t m58 = (vt >= 0x10000) ? *reinterpret_cast<uint64_t *>(vt + 0x58) : 0;
        // 0x608390-style forward: inner obj = [obj+0x8], real method = inner.vt[0x58].
        uint64_t inner = *reinterpret_cast<uint64_t *>(obj + 0x8);
        uint64_t ivt = (inner >= 0x10000) ? *reinterpret_cast<uint64_t *>(inner) : 0;
        uint64_t im58 = (ivt >= 0x10000) ? *reinterpret_cast<uint64_t *>(ivt + 0x58) : 0;
        uint64_t im30 = (ivt >= 0x10000) ? *reinterpret_cast<uint64_t *>(ivt + 0x30) : 0;
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
      *reinterpret_cast<uint64_t *>(gr[REG_RSP]) = (uint64_t)gr[REG_RBP];
      return;
    }
  }
  if (sig == SIGTRAP && g_fatalTraceAddr && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    if ((uintptr_t)gr[REG_RIP] == g_fatalTraceAddr + 1) {
      uint64_t fmt = (uint64_t)gr[REG_RDI];
      uint64_t caller = 0;
      uintptr_t rsp = (uintptr_t)gr[REG_RSP];
      if (rsp >= 0x10000) caller = *reinterpret_cast<uint64_t *>(rsp);
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
      *reinterpret_cast<uint64_t *>(gr[REG_RSP]) = (uint64_t)gr[REG_RBP];
      return;
    }
  }
  if (sig == SIGTRAP && g_cntTraceAddr && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    if ((uintptr_t)gr[REG_RIP] == g_cntTraceAddr + 1) {
      uint64_t obj = (uint64_t)gr[REG_RDI];
      uint32_t cnt = 0; char nm[48] = {0};
      if (obj >= 0x10000) {
        cnt = *reinterpret_cast<uint32_t *>(obj + 0x30);
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
      static const bool clamp = std::getenv("DELTA_CNT_CLAMP") != nullptr;
      if (clamp && obj >= 0x10000 && cnt > 0x100000)
        *reinterpret_cast<uint32_t *>(obj + 0x30) = 0;
      gr[REG_RSP] -= 8;
      *reinterpret_cast<uint64_t *>(gr[REG_RSP]) = (uint64_t)gr[REG_RBP];
      return;
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
      uintptr_t caller = rsp >= 0x10000 ? *reinterpret_cast<uint64_t *>(rsp) : 0;
      uint64_t size = (uint64_t)gr[REG_RDI];
      heapProfRecord(caller, size);
      g_heapProfHookBytes[i].fetch_add(size, std::memory_order_relaxed);
      g_heapProfHookCalls[i].fetch_add(1, std::memory_order_relaxed);
      gr[REG_RSP] -= 8;  // emulate the displaced `push rbp`
      *reinterpret_cast<uint64_t *>(gr[REG_RSP]) = (uint64_t)gr[REG_RBP];
      return;
    }
  }
  if (sig == SIGTRAP && g_allocTraceAddr && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    if ((uintptr_t)gr[REG_RIP] == g_allocTraceAddr + 1) {
      uint64_t size = (uint64_t)gr[REG_RSI];
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
      *reinterpret_cast<uint64_t *>(gr[REG_RSP]) = (uint64_t)gr[REG_RBP];
      return;            // resume at addr+1 (the mov rbp,rsp that follows)
    }
  }
#endif
  // Let the CPU backend handle JIT-internal signals (e.g. FEX unaligned-atomic
  // SIGBUS) and resume; only a genuinely fatal fault falls through to the dump.
  if (cpu::tryHandleJitSignal(sig, si, ucv))
    return;

#if defined(__x86_64__)
  // DELTA_PS5_GLYPHGUARD: recover a registered null-object deref in the UI/text
  // renderer -- zero the destination register and step past the faulting load so
  // the code continues with a benign value (unbound-font text renders empty).
  if (sig == SIGSEGV && g_nullGuardCount && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *gr = uc->uc_mcontext.gregs;
    for (int i = 0; i < g_nullGuardCount; i++) {
      if ((uintptr_t)gr[REG_RIP] != g_nullGuards[i].addr) continue;
      gr[g_nullGuards[i].greg] = 0;
      gr[REG_RIP] += g_nullGuards[i].len;
      return;
    }
  }
#endif

  // Async-signal-safe entry marker: proves the handler actually ran even if a
  // later step (symbolize / backtrace) re-faults. Without it a re-fault inside
  // the handler is indistinguishable from the handler never being entered.
  { char m[48];
    int n = std::snprintf(m, sizeof(m), "\n[crashHandler] entered sig=%d\n", sig);
    if (n > 0) { ssize_t w = write(2, m, (size_t)n); (void)w; } }

  // Only the first faulting thread prints. A second concurrent fault (common at
  // teardown) would interleave the dump and can itself core-dump, truncating it.
  static std::atomic<bool> s_dumping{false};
  if (s_dumping.exchange(true)) {
    for (;;) pause();  // park until the first thread's _Exit ends the process
  }
  utl::silenceLogging();  // stop the async log thread racing us on stderr

  if (g_heapProfAddr)
    heapProfDump();

#if defined(__x86_64__)
  // Guest SDK assert/__debugbreak is a software interrupt (`int 0xNN`, cd NN);
  // in userspace it raises SIGSEGV (#GP). Real hardware with no kernel debugger
  // attached just steps over it and the assert handler's caller continues, so do
  // the same: skip the 2-byte instruction and resume. Otherwise every guest
  // assertion would kill the boot. PS4 uses int 0x41; PS5 (Prospero) also uses
  // int 0x44/0x45 (e.g. libkernel's `int 0x45; xor eax,eax; ret` assert stub).
  if (sig == SIGSEGV && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *ip = reinterpret_cast<const uint8_t *>(uc->uc_mcontext.gregs[REG_RIP]);
    if (ip && ip[0] == 0xcd &&
        (ip[1] == 0x41 || ip[1] == 0x44 || ip[1] == 0x45)) {
      static std::atomic<int> n{0};
      if (n.fetch_add(1) < 20) {
        char sym[256];
        symbolize(uc->uc_mcontext.gregs[REG_RIP], sym, sizeof(sym));
        auto *g = uc->uc_mcontext.gregs;
        std::fprintf(stderr,
                     "[assert] skipped guest int 0x%02x @ %s "
                     "rsi=%lld rdx=%lld r15=%lld rax=%lld rcx=%lld\n",
                     ip[1], sym, (long long)g[REG_RSI], (long long)g[REG_RDX],
                     (long long)g[REG_R15], (long long)g[REG_RAX],
                     (long long)g[REG_RCX]);
      }
      uc->uc_mcontext.gregs[REG_RIP] += 2;
      return;
    }
  }
#endif

  char fault[256];
  symbolize((uintptr_t)si->si_addr, fault, sizeof(fault));
  std::fprintf(stderr, "\n=== GUEST FAULT: %s (signal %d) ===\n",
               strsignal(sig), sig);
  if (int sc = cpu::faultingSyscall(); sc >= 0)
    std::fprintf(stderr, "  in syscall %d (%s)\n", sc, syscall_getname((uint32_t)sc));
  std::fprintf(stderr, "  fault = %016llx  %s\n",
               (unsigned long long)si->si_addr, fault);
  // Show what the host VA space holds around the fault: which mapping it hit,
  // or which two mappings it fell between. Async-signal-safe (read+write only).
  if (si->si_addr) {
    const uint64_t fa = (uint64_t)si->si_addr;
    int mf = open("/proc/self/maps", O_RDONLY);
    if (mf >= 0) {
      static char mbuf[1 << 20];
      ssize_t n = 0, off = 0, r;
      while ((r = read(mf, mbuf + off, sizeof(mbuf) - 1 - off)) > 0)
        off += r;
      n = off;
      close(mf);
      mbuf[n] = 0;
      char *prev = nullptr, *line = mbuf;
      while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        uint64_t lo = strtoull(line, nullptr, 16);
        const char *dash = strchr(line, '-');
        uint64_t hi = dash ? strtoull(dash + 1, nullptr, 16) : 0;
        if (fa < hi || !nl) {
          if (prev)
            std::fprintf(stderr, "  maps prev: %s\n", prev);
          std::fprintf(stderr, "  maps %s : %s\n",
                       (fa >= lo && fa < hi) ? "HIT " : "next", line);
          // A couple of following lines: what the faulting pointer sits under.
          char *after = nl ? nl + 1 : nullptr;
          for (int k = 0; k < 3 && after && *after; k++) {
            char *anl = strchr(after, '\n');
            if (anl) *anl = 0;
            std::fprintf(stderr, "  maps  +%d : %s\n", k + 1, after);
            after = anl ? anl + 1 : nullptr;
          }
          break;
        }
        prev = line;
        line = nl ? nl + 1 : nullptr;
      }
    }
  }
#if defined(__x86_64__)
  // Native x86 host: the host signal context IS the guest context.
  auto *uc = static_cast<ucontext_t *>(ucv);
  auto *gr = uc->uc_mcontext.gregs;
  char rip[256];
  symbolize(gr[REG_RIP], rip, sizeof(rip));
  std::fprintf(stderr, "  rip   = %016llx  %s\n",
               (unsigned long long)gr[REG_RIP], rip);
  std::fprintf(stderr, "  rax=%016llx rbx=%016llx rcx=%016llx rdx=%016llx\n",
               (unsigned long long)gr[REG_RAX], (unsigned long long)gr[REG_RBX],
               (unsigned long long)gr[REG_RCX], (unsigned long long)gr[REG_RDX]);
  std::fprintf(stderr, "  rsi=%016llx rdi=%016llx rbp=%016llx rsp=%016llx\n",
               (unsigned long long)gr[REG_RSI], (unsigned long long)gr[REG_RDI],
               (unsigned long long)gr[REG_RBP], (unsigned long long)gr[REG_RSP]);
  std::fprintf(stderr, "  r8 =%016llx r9 =%016llx r10=%016llx r11=%016llx\n",
               (unsigned long long)gr[REG_R8], (unsigned long long)gr[REG_R9],
               (unsigned long long)gr[REG_R10], (unsigned long long)gr[REG_R11]);
  std::fprintf(stderr, "  r12=%016llx r13=%016llx r14=%016llx r15=%016llx\n",
               (unsigned long long)gr[REG_R12], (unsigned long long)gr[REG_R13],
               (unsigned long long)gr[REG_R14], (unsigned long long)gr[REG_R15]);
  if (gr[REG_RIP]) {
    auto *b = reinterpret_cast<const uint8_t *>(gr[REG_RIP]);
    std::fprintf(stderr, "  insn bytes:");
    for (int i = 0; i < 16; i++)
      std::fprintf(stderr, " %02x", b[i]);
    std::fprintf(stderr, "\n");
  }
  backtrace(gr[REG_RBP]);
  // Raw stack scan: optimised guest code omits frame pointers, so the rbp chain
  // above misses frames. Scan the guest stack for values that land in a loaded
  // module's .text (the real call chain) and for pointers into the guest heap
  // arena that hold a printable ASCII string (an asset filename / tag the
  // faulting code was handling). The arena (0x40_0000_0000..0x41_0000_0000) is
  // always mapped, so reading a string there can't fault.
  std::fprintf(stderr, "  --- stack scan ---\n");
  if (uintptr_t rsp = gr[REG_RSP]; rsp >= 0x10000) {
    auto *sp = reinterpret_cast<uintptr_t *>(rsp);
    for (int i = 0; i < 512; i++) {
      uintptr_t v = sp[i];
      char sym[256];
      symbolize(v, sym, sizeof(sym));
      if (std::strstr(sym, "(.text)")) {
        std::fprintf(stderr, "  sp+%-5x %016lx  %s\n", i * 8, v, sym);
      } else if (v >= 0x4000000000ull && v < 0x4100000000ull) {
        auto *s = reinterpret_cast<const char *>(v);
        int n = 0;
        while (n < 40 && s[n] >= 0x20 && s[n] <= 0x7e) n++;
        if (n >= 5 && s[n] == 0)
          std::fprintf(stderr, "  sp+%-5x %016lx  str=\"%.40s\"\n", i * 8, v, s);
      }
    }
  }
#else
  // aarch64 host: guest x86 state lives in the FEXCore CPUState, not the host
  // ARM signal context. Reconstruct the precise guest RIP from the host JIT PC
  // (CPUState.rip alone is only block-accurate and multiblock hides the site).
  uint64_t hostpc = 0;
#if defined(__aarch64__)
  if (ucv)
    hostpc = static_cast<ucontext_t *>(ucv)->uc_mcontext.pc;
#endif
  uint64_t recon = cpu::reconstructGuestRip(hostpc);
  uint64_t grip = recon ? recon : cpu::currentGuestRip(); // fall back to block rip
  std::fprintf(stderr, "  host pc in JIT: %s\n", recon ? "yes" : "no (FEX/HLE C++)");
  char ripsym[256];
  symbolize(grip, ripsym, sizeof(ripsym));
  std::fprintf(stderr, "  host pc   = %016llx\n", (unsigned long long)hostpc);
  std::fprintf(stderr, "  guest rip = %016llx  %s\n",
               (unsigned long long)grip, ripsym);
  if (grip) {
    auto *b = reinterpret_cast<const uint8_t *>(grip);
    std::fprintf(stderr, "  insn bytes:");
    for (int i = 0; i < 16; i++)
      std::fprintf(stderr, " %02x", b[i]);
    std::fprintf(stderr, "\n");
  }
  // Guest GPR dump + rbp backtrace (parity with the native x86 dump above).
  // gregs order is FEXCore::X86State::REG_* (RAX,RCX,RDX,RBX,RSP,RBP,RSI,RDI,
  // R8..R15); mirror it locally so this TU needs no FEXCore headers.
  if (const uint64_t *g = cpu::currentGuestGregs()) {
    enum { RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15 };
    std::fprintf(stderr, "  rax=%016llx rbx=%016llx rcx=%016llx rdx=%016llx\n",
                 (unsigned long long)g[RAX], (unsigned long long)g[RBX],
                 (unsigned long long)g[RCX], (unsigned long long)g[RDX]);
    std::fprintf(stderr, "  rsi=%016llx rdi=%016llx rbp=%016llx rsp=%016llx\n",
                 (unsigned long long)g[RSI], (unsigned long long)g[RDI],
                 (unsigned long long)g[RBP], (unsigned long long)g[RSP]);
    std::fprintf(stderr, "  r8 =%016llx r9 =%016llx r10=%016llx r11=%016llx\n",
                 (unsigned long long)g[R8], (unsigned long long)g[R9],
                 (unsigned long long)g[R10], (unsigned long long)g[R11]);
    std::fprintf(stderr, "  r12=%016llx r13=%016llx r14=%016llx r15=%016llx\n",
                 (unsigned long long)g[R12], (unsigned long long)g[R13],
                 (unsigned long long)g[R14], (unsigned long long)g[R15]);

    // ---- SOTC AllocationTracker walk (diagnostic; see helper above) ----
    // Fire only when the fault is inside the eboot's slot-21 untrack-on-free
    // methods: fn @+0x18920 (CPU tracker) / +0x8d930 (GPU tracker). Resolve the
    // eboot base from the module whose .text contains grip (== textSeg.addr, so
    // grip-base is the module-relative ELF vaddr), like symbolize() does.
    {
      uint64_t ebase = 0;
      if (auto *proc = proc::getActive()) {
        for (auto &mod : proc->getModuleList()) {
          auto &mi = mod->getInfo();
          auto *t = mi.textSeg.addr;
          if (t && grip >= (uintptr_t)t && grip < (uintptr_t)t + mi.textSeg.size) {
            ebase = (uint64_t)t;
            break;
          }
        }
      }
      uint64_t off = ebase ? grip - ebase : 0;
      bool inTrk = ebase && ((off >= 0x18000 && off < 0x19000) ||
                             (off >= 0x8d000 && off < 0x8e000));
      if (inTrk) {
        std::fprintf(stderr,
                     "\n  === SOTC tracker walk (eboot base=%#llx, fault off=%#llx) ===\n",
                     (unsigned long long)ebase, (unsigned long long)off);
        // Recover (tracker,key) from the saved-register slots on the stack,
        // which are reliable regardless of FEX callee-saved reconstruction.
        // Layout after the prologue pushes (no further stack alloc):
        //   [rsp+0x18]=saved r13=TRACKER   [rsp+0x20]=saved r14=KEY
        uint64_t rsp = g[RSP];
        uint64_t trkStk = 0, keyStk = 0;
        bool haveStk = trkRd64(rsp + 0x18, trkStk) && trkRd64(rsp + 0x20, keyStk);
        std::fprintf(stderr,
                     "  [trkwalk] from-stack: tracker=%#llx key=%#llx (ok=%d) | "
                     "from-reg: r13=%#llx r14=%#llx\n",
                     (unsigned long long)trkStk, (unsigned long long)keyStk,
                     haveStk, (unsigned long long)g[R13], (unsigned long long)g[R14]);
        // Prefer the stack-recovered tracker/key; fall back to the regs if the
        // stack slot doesn't look like a mapped module-space pointer.
        auto plausible = [](uint64_t t) {
          return t >= 0x200000000000ull && t < 0x210000000000ull && trkMincore(t);
        };
        uint64_t tracker = plausible(trkStk) ? trkStk : g[R13];
        uint64_t key = (haveStk && keyStk) ? keyStk : g[R14];
        // Raw tracker header window (fallback context: +0x00..0xa0).
        if (trkMincore(tracker)) {
          auto *q = reinterpret_cast<const uint64_t *>(tracker);
          for (int i = 0; i < 20; i++) {
            if ((i % 4) == 0)
              std::fprintf(stderr, "\n  trk+%03x:", i * 8);
            std::fprintf(stderr, " %016llx", (unsigned long long)q[i]);
          }
          std::fprintf(stderr, "\n");
        }
        // 1) Walk the tracker the fault came from.
        bool inThis = sotcWalkTracker(tracker, key, "fault");
        // 2) Cross-check the other known tracker instances (the two GPU/renderer
        //    tracker globals @ base+0x2ed3350 / +0x2ed33b0) for the same key.
        uint64_t gpuA = ebase + 0x2ed3350, gpuB = ebase + 0x2ed33b0;
        bool inA = false, inB = false;
        if (gpuA != tracker) inA = sotcWalkTracker(gpuA, key, "gpuA");
        if (gpuB != tracker) inB = sotcWalkTracker(gpuB, key, "gpuB");
        // 3) Emulator-side VMA view of the key.
        if (auto *proc = proc::getActive()) {
          auto *pi = proc->getVma().get(reinterpret_cast<uint8_t *>(key));
          if (pi) {
            uint64_t vb = (uint64_t)pi->ptr, ve = vb + pi->size;
            std::fprintf(stderr,
                         "  [trkwalk] emu VMA: key %#llx is IN region [%#llx,%#llx) "
                         "size=%#llx sceProt=%#x reserved=%d name=%s  key-regionbase=%+lld\n",
                         (unsigned long long)key, (unsigned long long)vb,
                         (unsigned long long)ve, (unsigned long long)pi->size,
                         pi->sceProt, pi->reserved, pi->name ? pi->name : "(null)",
                         (long long)(key - vb));
          } else {
            std::fprintf(stderr,
                         "  [trkwalk] emu VMA: key %#llx is in NO tracked region\n",
                         (unsigned long long)key);
          }
        }
        std::fprintf(stderr,
                     "  [trkwalk] SUMMARY: key covered in fault-tracker=%d gpuA=%d gpuB=%d\n",
                     inThis, inA, inB);
        std::fprintf(stderr, "  === end SOTC tracker walk ===\n\n");
        std::fflush(stderr);
      }
    }

    // DELTA_CRASH_PEEK: dump a window of guest memory around each GPR that points
    // into loaded-module space (>= 0x2000_0000_0000). An indirect call/jmp through
    // a garbage/null vtable slot is our most common late-boot fault; seeing the
    // object bytes + where the slot points identifies the uninitialised object.
    if (const char *pk = std::getenv("DELTA_CRASH_PEEK")) {
      const uint64_t regs[] = {g[RAX], g[RBX], g[RDI], g[RSI], g[RCX], g[RDX]};
      const char *rn[] = {"rax", "rbx", "rdi", "rsi", "rcx", "rdx"};
      for (int r = 0; r < 6; r++) {
        uint64_t base = regs[r];
        if (base < 0x200000000000ull || base >= 0x210000000000ull)
          continue;  // only the module VA window is reliably mapped to read
        auto *q = reinterpret_cast<const uint64_t *>(base);
        // rax/rbx are the usual object/this pointers: dump far enough to cover a
        // deep vtable/member-fn slot (the fault operand disp can be several
        // hundred bytes in). Other regs get just a header.
        int n = (r < 2) ? 56 : 8;
        for (int i = 0; i < n; i++) {
          if ((i % 4) == 0)
            std::fprintf(stderr, "\n  peek %s+%03x:", rn[r], i * 8);
          std::fprintf(stderr, " %016llx", (unsigned long long)q[i]);
        }
        std::fprintf(stderr, "\n");
      }
      // Explicit address list: DELTA_CRASH_PEEK=0x1c92d00,0x1c2fc00 also dumps a
      // window of guest memory at each given VA (comma/space separated). Unlike the
      // register scan this is NOT restricted to the loaded-module window, so it can
      // read low ET_SCE_EXEC globals/BSS (e.g. a null singleton pointer). Guarded by
      // mincore so an unmapped address can't fault the crash handler.
      for (const char *p = pk; *p;) {
        while (*p == ',' || *p == ' ') p++;
        char *end = nullptr;
        uint64_t va = std::strtoull(p, &end, 0);
        if (end == p) { if (*p) p++; continue; }
        p = end;
        if (va < 0x10000) continue;  // "1"/tiny -> register scan only
        long pg = sysconf(_SC_PAGESIZE);
        unsigned char vec[2] = {0, 0};
        void *pa = reinterpret_cast<void *>(va & ~((uint64_t)pg - 1));
        if (mincore(pa, 1, vec) != 0) {
          std::fprintf(stderr, "  peek %#llx: <unmapped>\n", (unsigned long long)va);
          continue;
        }
        auto *q = reinterpret_cast<const uint64_t *>(va);
        for (int i = 0; i < 16; i++) {
          if ((i % 4) == 0)
            std::fprintf(stderr, "\n  peek %#llx+%03x:", (unsigned long long)va, i * 8);
          std::fprintf(stderr, " %016llx", (unsigned long long)q[i]);
        }
        std::fprintf(stderr, "\n");
      }
    }
    // DELTA_CRASH_PEEK also dumps the raw stack window around rsp: for a fault
    // inside a leaf helper (e.g. a lookup that returned null) the caller's
    // locals -- the key being freed, the object under operation -- are the
    // fastest route to "what data was this actually working on".
    if (std::getenv("DELTA_CRASH_PEEK") && g[RSP] >= 0x10000) {
      long pg = sysconf(_SC_PAGESIZE);
      unsigned char vec[2] = {0, 0};
      void *pa = reinterpret_cast<void *>(g[RSP] & ~((uint64_t)pg - 1));
      if (mincore(pa, 1, vec) == 0) {
        auto *q = reinterpret_cast<const uint64_t *>(g[RSP] & ~7ull);
        for (int i = -8; i < 64; i++) {
          if (((i + 8) % 4) == 0)
            std::fprintf(stderr, "\n  stack rsp%+05x:", i * 8);
          std::fprintf(stderr, " %016llx", (unsigned long long)q[i]);
        }
        std::fprintf(stderr, "\n");
      }
    }
    // Print the boundary-call trace before the (best-effort, occasionally
    // out-of-bounds) stack scan so it survives even if the scan faults.
    cpu::dumpThreadTrace(stderr);
    std::fflush(stderr);
    backtrace(g[RBP]);
    // Raw stack scan: optimised guest code omits frame pointers, so the rbp
    // chain above misses frames. Scan the guest stack for any value that lands
    // in a loaded module's .text; that's the (super)set of return
    // addresses, i.e. the real call chain.
    std::fprintf(stderr, "  --- stack scan ---\n");
    auto *sp = reinterpret_cast<uintptr_t *>(g[RSP]);
    if (g[RSP] >= 0x10000) {
      for (int i = 0; i < 256; i++) {
        uintptr_t v = sp[i];
        char sym[256];
        symbolize(v, sym, sizeof(sym));
        if (std::strstr(sym, "(.text)"))
          std::fprintf(stderr, "  sp+%-4x %016lx  %s\n", i * 8, v, sym);
      }
    }
  }
#endif
  std::fflush(stderr);
  std::fflush(stdout);  // _Exit won't flush; keep the guest trace up to the fault
  std::_Exit(128 + sig);
}

void setAllocTrace(uintptr_t addr, uint64_t minSize) {
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
void setNullGuard(uintptr_t addr, GuardReg reg, int insnLen) {
#if defined(__x86_64__)
  if (g_nullGuardCount >= 16) return;
  int greg = reg == GuardReg::rax ? REG_RAX : REG_RSI;
  g_nullGuards[g_nullGuardCount++] = {addr, greg, insnLen};
#else
  (void)addr; (void)reg; (void)insnLen;
#endif
}
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
void setRetTrace(uintptr_t addr, const char *label) {
  if (g_retCount < 8) {
    g_retAddrs[g_retCount] = addr;
    g_retLabels[g_retCount] = label;
    g_retCount++;
  }
}

void installSigAltStack() {
  // One alt stack per thread; 256 KiB easily holds our dump path. Leaked on
  // purpose (lives for the thread's lifetime, freed at process exit).
  static thread_local stack_t s_alt{};
  if (s_alt.ss_sp)
    return;  // already installed for this thread
  constexpr size_t kAltSz = 256 * 1024;
  void *mem = std::malloc(kAltSz);
  if (!mem)
    return;
  s_alt.ss_sp = mem;
  s_alt.ss_size = kAltSz;
  s_alt.ss_flags = 0;
  sigaltstack(&s_alt, nullptr);

  // Guarantee the fatal signals are deliverable on this thread. Guest libthr
  // (or a syscall the lifter missed) can leave them blocked in the host mask, in
  // which case a synchronous fault force-kills the process before our handler
  // runs (silent core, no dump). Unblock them unconditionally.
  sigset_t unb;
  sigemptyset(&unb);
  sigaddset(&unb, SIGSEGV);
  sigaddset(&unb, SIGILL);
  sigaddset(&unb, SIGBUS);
  sigaddset(&unb, SIGFPE);
  sigaddset(&unb, SIGTRAP);
  sigaddset(&unb, SIGABRT);
  sigaddset(&unb, SIGUSR1);  // keep the deadlock probe deliverable on guest threads
  pthread_sigmask(SIG_UNBLOCK, &unb, nullptr);
}

void installCrashHandler() {
  struct sigaction sa = {};
  sa.sa_sigaction = crashHandler;
  // SA_NODEFER: don't auto-mask the signal during the handler, so a re-fault
  // inside the dump produces another catchable signal (and our s_dumping guard
  // parks it) instead of the kernel forcing the default action -> silent core.
  // SA_ONSTACK: run the handler on each thread's sigaltstack (installSigAltStack)
  // so a stack-overflow / corrupt-RSP fault is still deliverable. Without it the
  // kernel can't push the signal frame onto the bad guest stack and force-kills
  // the process (silent core, no dump).
  sa.sa_flags = SA_SIGINFO | SA_NODEFER | SA_ONSTACK;
  sigemptyset(&sa.sa_mask);
  installSigAltStack();  // for the installing (ctx) thread
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
  sigaction(SIGTRAP, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGABRT, &sa, nullptr);  // guest/runtime std::abort, assert, libc
#if defined(__x86_64__)
  struct sigaction pa = {};
  pa.sa_sigaction = probeHandler;
  pa.sa_flags = SA_SIGINFO | SA_RESTART;  // don't abort the thread's blocking call
  sigemptyset(&pa.sa_mask);
  sigaction(SIGUSR1, &pa, nullptr);
#endif
}
}  // namespace krnl
