/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#define _GNU_SOURCE
#include <atomic>
#include "base/arch.h"
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <dlfcn.h>
#include <ucontext.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <time.h>
#include <pthread.h>
#include <thread>
#include <chrono>
#include <string>
#include <vector>

#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>

#include "crash.h"
#include "lv2/dispatch.h"

#include <utl/mem.h>
#include "module.h"
#include "proc.h"
#include "vfs.h"
#include "cpu/cpu_backend.h"
#include "gpu/rhi/renderer.h"
#include <logger/logger.h>
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
  BASE_LOGI("crashHandler", "  --- backtrace ---");
  for (int i = 0; i < 32; i++) {
    if (rbp < 0x10000 || (rbp & 7))
      break;
    // "Range-checked" was only a bounds test: a thread parked deep in host code
    // has no frame chain at rbp at all, and the walk faulted reading it. That is
    // survivable in a crash report but not from the SIGUSR1 probe, which is
    // asking a live run what it is stuck on and took the run down instead.
    if (!utl::isMemoryRangeMapped(reinterpret_cast<void *>(rbp),
                                  2 * sizeof(uintptr_t)))
      break;
    auto *frame = reinterpret_cast<uintptr_t *>(rbp);
    uintptr_t next = frame[0];
    uintptr_t ret = frame[1];
    if (!ret)
      break;
    char sym[256];
    symbolize(ret, sym, sizeof(sym));
    BASE_LOGI("crashHandler", "  #{:<2} {:016x}  {}", i, ret, sym);
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
inline bool trkMincore(u64 va) {
  if (va < 0x10000) return false;
  long pg = sysconf(_SC_PAGESIZE);
  unsigned char vec = 0;
  void *pa = reinterpret_cast<void *>(va & ~((u64)pg - 1));
  return mincore(pa, 1, &vec) == 0;
}
inline bool trkRd64(u64 va, u64 &out) {
  if (!trkMincore(va) || !trkMincore(va + 7)) return false;
  out = *reinterpret_cast<const u64 *>(va);
  return true;
}
// Re-walk the title's size-ordered free tree the way the faulting insert does,
// and name the FIELD that holds the bad pointer. The insert loop only ever has
// the bad VALUE in a register (rax) -- the address it was loaded FROM is the one
// thing needed to arm a write census, and it is gone by the time we fault.
//
// The walk (eboot+0x48a70, a dlmalloc-shaped allocator): `state+0x80` is the
// tree head and doubles as the loop's sentinel; a node points at chunk+0x10, so
// its size word is at node-8; children are node[0] and node[1]; the branch taken
// is `newsz < cursz ? 0 : 1`, and an exact size match ends the walk.
void sotcWalkFreeTree(u64 state, u64 newsz) {
  const u64 sentinel = state + 0x80;
  BASE_LOGI("freetree", "  state={:#x} sentinel={:#x} newsz={:#x}",
            (unsigned long long)state, (unsigned long long)sentinel,
            (unsigned long long)newsz);
  u64 cur = 0;
  if (!trkRd64(sentinel, cur)) {
    BASE_LOGI("freetree", "  head not mapped -- nothing to walk");
    return;
  }
  u64 field = sentinel;  // where `cur` was loaded from
  for (int step = 0; step < 64; step++) {
    if (cur == sentinel) {
      BASE_LOGI("freetree",
                "  step {}: back at the sentinel, the tree is intact -- the "
                "bad pointer is NOT here", step);
      return;
    }
    u64 sz = 0;
    if (!trkRd64(cur - 8, sz)) {
      BASE_LOGI("freetree",
                "  step {}: node {:#x} is UNMAPPED (its size word at {:#x} "
                "cannot be read) -- THIS IS THE FAULT",
                step, (unsigned long long)cur, (unsigned long long)(cur - 8));
      BASE_LOGI("freetree",
                "  the bad pointer was loaded FROM {:#x}  <== arm the write "
                "census here (DELTA_GUEST_WHIST={:x}:8:8)",
                (unsigned long long)field, (unsigned long long)field);
      // What surrounds the corrupt field: its neighbours often show the intact
      // originals, which says whether the whole node or just one word was hit.
      const u64 win = field & ~0x3full;
      base::String winwords;
      for (int i = 0; i < 8; i++) {
        u64 v = 0;
        if ((i % 4) == 0)
          base::FormatTo(winwords, "\n  {:#x}:", (unsigned long long)(win + i * 8));
        base::FormatTo(winwords, " {:016x}",
                       (unsigned long long)(trkRd64(win + i * 8, v) ? v : 0));
      }
      BASE_LOGI("freetree", "{}", winwords.c_str());
      // Split the value: SotC's stale links read as a valid 40-bit guest
      // pointer with rubbish above it, because the word is not a pointer at all
      // -- it is whatever the new owner of the reused chunk stored there, and
      // the arrays in question hold packed descriptors.
      BASE_LOGI("freetree", "\n  bad value {:#x}: low40={:#x}, "
                            "bits40+={:#x} (so probably not a pointer)",
                (unsigned long long)cur,
                (unsigned long long)(cur & 0xffffffffffull),
                (unsigned long long)(cur >> 40));
      // Is the corrupt word inside guest memory the GPU module snapshots and
      // copies back? If it is, the compute writeback is reverting the
      // allocator's own stores and this is our corruption, not the title's.
      char csr[256];
      if (gpu::rhi::DescribeCsRangeCovering(field, csr, sizeof(csr)))
        BASE_LOGI("freetree", "  the field IS inside a compute staging range: {}",
                  csr);
      else
        BASE_LOGI("freetree", "  no compute staging range covers the field");
      return;
    }
    sz &= ~7ull;
    const int idx = (newsz < sz) ? 0 : 1;
    // A free chunk's size is small, non-zero and 16-byte aligned. The walk
    // running off the tree shows up HERE, one step before it dereferences
    // something unmapped: the node is memory that has been handed back out and
    // refilled, so its "size" is whatever the new owner stored there. Reporting
    // only the unmapped dereference blames the wrong field -- by then the walk
    // has been reading live application data as nodes for several steps.
    // 8-granular, not 16: the allocator masks the size word with ~7 and a real
    // free chunk of 0x158 turned up in a later crash, which a 16-alignment test
    // flagged as "no longer a free chunk" and blamed the wrong link for.
    const bool plausible = sz && sz < 0x8000000ull && (sz & 7) == 0;
    BASE_LOGI("freetree",
              "  step {}: node={:#x} size={:#x} -> child[{}]{}",
              step, (unsigned long long)cur, (unsigned long long)sz, idx,
              plausible ? "" : "   <== NOT A FREE CHUNK ANY MORE");
    if (!plausible) {
      BASE_LOGI("freetree",
                "  the tree left itself here: the link at {:#x} still points "
                "at {:#x}, which is no longer a free chunk",
                (unsigned long long)field, (unsigned long long)cur);
      char csr1[256];
      if (gpu::rhi::DescribeCsRangeCovering(field, csr1, sizeof(csr1)))
        BASE_LOGI("freetree",
                  "    the STALE LINK is inside a compute staging range: {}",
                  csr1);
      else
        BASE_LOGI("freetree",
                  "    no compute staging range covers the stale link at {:#x}",
                  (unsigned long long)field);
      if (gpu::rhi::DescribeCsRangeCovering(cur, csr1, sizeof(csr1)))
        BASE_LOGI("freetree",
                  "    the reused CHUNK is inside a compute staging range: {}",
                  csr1);
      const u64 win2 = (field & ~0x1full) - 0x20;
      base::String win2words;
      for (int i = 0; i < 12; i++) {
        u64 v = 0;
        if ((i % 4) == 0)
          base::FormatTo(win2words, "\n  {:#x}:",
                         (unsigned long long)(win2 + i * 8));
        base::FormatTo(win2words, " {:016x}",
                       (unsigned long long)(trkRd64(win2 + i * 8, v) ? v : 0));
      }
      base::FormatTo(win2words, "\n");
      BASE_LOGI("freetree", "{}", win2words.c_str());
    }
    if (newsz == sz) {
      BASE_LOGI("freetree", "  exact size match, walk ends here");
      return;
    }
    field = cur + (u64)idx * 8;
    if (!trkRd64(field, cur)) {
      BASE_LOGI("freetree", "  child field {:#x} unmapped -- stop",
                (unsigned long long)field);
      return;
    }
  }
  BASE_LOGI("freetree", "  64 steps without terminating (a cycle?)");
}

// Walk one tracker's circular record list; report count/bytes, whether `key`
// is covered by a record, and the 8 records nearest to `key` by |base-key|.
// Returns true if `key` fell inside some record's [base,base+size).
bool sotcWalkTracker(u64 tracker, u64 key, const char *tag) {
  BASE_LOGI("trkwalk", "{}  tracker={:#x} key={:#x}", tag,
            (unsigned long long)tracker, (unsigned long long)key);
  if (!trkMincore(tracker) || !trkMincore(tracker + 0x98)) {
    BASE_LOGI("trkwalk", "{}    tracker not mapped -- skip", tag);
    return false;
  }
  u64 sentinel = tracker + 0x38;
  u64 first = 0, hdrCount = 0, hdrBytes = 0, listener = 0;
  trkRd64(tracker + 0x48, first);
  trkRd64(tracker + 0x90, hdrCount);
  trkRd64(tracker + 0x80, hdrBytes);
  trkRd64(tracker + 0x28, listener);
  BASE_LOGI("trkwalk",
            "{}    listener={:#x} first={:#x} count(+0x90)={} "
            "bytes(+0x80)={:#x}",
            tag, (unsigned long long)listener, (unsigned long long)first,
            (unsigned long long)hdrCount, (unsigned long long)hdrBytes);
  // Nearest-8 online selection by absolute distance from key.
  u64 nb[8], ns[8], nd[8];
  for (int i = 0; i < 8; i++) { nb[i] = ns[i] = 0; nd[i] = ~0ull; }
  u64 node = first, walked = 0, sumSize = 0;
  bool covered = false, coverPrinted = false;
  for (; walked < 200000; walked++) {
    if (node == sentinel || node == 0) break;
    if (!trkMincore(node) || !trkMincore(node + 0x60 + 7)) {
      BASE_LOGI("trkwalk", "{}    node {:#x} unmapped -- stop", tag,
                (unsigned long long)node);
      break;
    }
    u64 base = 0, size = 0, next = 0;
    trkRd64(node + 0x60, base);
    trkRd64(node + 0x58, size);
    trkRd64(node + 0x10, next);
    sumSize += size;
    if (base <= key && key < base + size) {
      covered = true;
      if (!coverPrinted) {
        BASE_LOGI("trkwalk",
                  "{}    *** COVER: rec {:#x} base={:#x} size={:#x} "
                  "end={:#x} contains key ***",
                  tag, (unsigned long long)node, (unsigned long long)base,
                  (unsigned long long)size, (unsigned long long)(base + size));
        coverPrinted = true;
      }
    }
    u64 d = base > key ? base - key : key - base;
    // insert into nearest-8 if closer than the current worst
    int worst = 0;
    for (int i = 1; i < 8; i++) if (nd[i] > nd[worst]) worst = i;
    if (d < nd[worst]) { nd[worst] = d; nb[worst] = base; ns[worst] = size; }
    node = next;
  }
  BASE_LOGI("trkwalk",
            "{}    walked {} records, sum(size)={:#x}, key {}",
            tag, (unsigned long long)walked, (unsigned long long)sumSize,
            covered ? "IS COVERED" : "is NOT covered by any record");
  // sort nearest-8 by distance (tiny insertion sort)
  for (int i = 0; i < 8; i++)
    for (int j = i + 1; j < 8; j++)
      if (nd[j] < nd[i]) {
        u64 t;
        t = nd[i]; nd[i] = nd[j]; nd[j] = t;
        t = nb[i]; nb[i] = nb[j]; nb[j] = t;
        t = ns[i]; ns[i] = ns[j]; ns[j] = t;
      }
  BASE_LOGI("trkwalk", "{}    8 nearest records to key (by |base-key|):", tag);
  for (int i = 0; i < 8; i++) {
    if (nd[i] == ~0ull) break;
    long long signedDelta = (long long)(nb[i] - key);
    BASE_LOGI("trkwalk",
              "{}     base={:#x} size={:#x} end={:#x}  base-key={:+} ({:#x})",
              tag, (unsigned long long)nb[i], (unsigned long long)ns[i],
              (unsigned long long)(nb[i] + ns[i]), signedDelta,
              (unsigned long long)nd[i]);
  }
  return covered;
}
}  // namespace

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
  BASE_LOGI("probe", "tid={} rip={:016x} {}", (long)gettid(),
            (unsigned long long)gr[REG_RIP], rip);
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

static void crashHandler(int sig, siginfo_t *si, void *ucv) {
#if defined(__x86_64__)
  if (sig == SIGTRAP && ucv && g_wprotStepPage) {
    const size_t pgsz = (size_t)sysconf(_SC_PAGESIZE);
    ::mprotect(reinterpret_cast<void *>(g_wprotStepPage), pgsz, PROT_READ);
    static_cast<ucontext_t *>(ucv)->uc_mcontext.gregs[REG_EFL] &= ~0x100;
    g_wprotStepPage = 0;
    return;
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
      return;
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
        return;
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
      return;
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
        return;
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
      uintptr_t caller = rsp >= 0x10000 ? *reinterpret_cast<u64 *>(rsp) : 0;
      char csym[200];
      symbolize(caller, csym, sizeof(csym));
      char m[320];
      int n = std::snprintf(m, sizeof(m), "[order t=%ldms tid=%ld] %s  <- %s\n",
                            ms, (long)gettid(), g_orderLabels[i], csym);
      if (n > 0) { ssize_t w = write(2, m, (size_t)n); (void)w; }
      gr[REG_RSP] -= 8;  // emulate the displaced `push rbp`
      *reinterpret_cast<u64 *>(gr[REG_RSP]) = (u64)gr[REG_RBP];
      return;
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
      return;
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
      return;
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
        return;
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
      return;
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
      return;
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
      return;
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

  // Guest SDK assert/__debugbreak is a software interrupt (`int 0xNN`, cd NN);
  // in userspace it raises SIGSEGV (#GP). Real hardware with no kernel debugger
  // attached just steps over it and the assert handler's caller continues, so do
  // the same: skip the 2-byte instruction and resume. Otherwise every guest
  // assertion would kill the boot. PS4 uses int 0x41; PS5 (Prospero) also uses
  // int 0x44/0x45 (e.g. libkernel's `int 0x45; xor eax,eax; ret` assert stub).
  // Must stay ahead of the dump latch below: a skipped assert is a resume path,
  // and latching on it would make the next real fault park instead of dump.
  if (sig == SIGSEGV && ucv) {
    auto *uc = static_cast<ucontext_t *>(ucv);
    auto *ip = reinterpret_cast<const u8 *>(uc->uc_mcontext.gregs[REG_RIP]);
    // A jump into unmapped memory faults with rip THERE, so reading the opcode
    // is itself a fault -- and the handler dying re-entrantly buries the real
    // report. Check the page is there first.
    if (ip) {
      const long pgsz = sysconf(_SC_PAGESIZE);
      unsigned char vec = 0;
      if (mincore(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ip) &
                                           ~((uintptr_t)pgsz - 1)),
                  1, &vec) != 0)
        ip = nullptr;
    }
    if (ip && ip[0] == 0xcd &&
        (ip[1] == 0x41 || ip[1] == 0x44 || ip[1] == 0x45)) {
      static std::atomic<int> n{0};
      if (n.fetch_add(1) < 20) {
        char sym[256];
        symbolize(uc->uc_mcontext.gregs[REG_RIP], sym, sizeof(sym));
        auto *g = uc->uc_mcontext.gregs;
        BASE_LOGI("assert",
                  "skipped guest int 0x{:02x} @ {} "
                  "rsi={} rdx={} r15={} rax={} rcx={}",
                  ip[1], sym, (long long)g[REG_RSI], (long long)g[REG_RDX],
                  (long long)g[REG_R15], (long long)g[REG_RAX],
                  (long long)g[REG_RCX]);
      }
      uc->uc_mcontext.gregs[REG_RIP] += 2;
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
  // The dumper re-entering is a different case: a step below re-faulted (e.g. the
  // stack scan on a stack that overflowed into its guard page), and parking it
  // would hang the process instead of ending it, so bail straight out.
  static std::atomic<bool> s_dumping{false};
  static std::atomic<pid_t> s_dumper{0};
  const pid_t self = static_cast<pid_t>(syscall(SYS_gettid));
  if (s_dumping.exchange(true)) {
    if (s_dumper.load() == self) {
      static const char m[] = "[crashHandler] re-faulted while dumping\n";
      ssize_t w = write(2, m, sizeof(m) - 1); (void)w;
      std::_Exit(128 + sig);
    }
    for (;;) pause();  // park until the first thread's _Exit ends the process
  }
  s_dumper.store(self);
  utl::silenceLogging();  // stop the async log thread racing us on stderr

  if (g_heapProfAddr)
    heapProfDump();

  char fault[256];
  symbolize((uintptr_t)si->si_addr, fault, sizeof(fault));
  BASE_LOGI("crashHandler", "\n=== GUEST FAULT: {} (signal {}) ===",
            strsignal(sig), sig);
  if (int sc = cpu::faultingSyscall(); sc >= 0)
    BASE_LOGI("crashHandler", "  in syscall {} ({})", sc,
              syscall_getname((u32)sc));
  BASE_LOGI("crashHandler", "  fault = {:016x}  {}",
            (unsigned long long)si->si_addr, fault);
  // A fault inside the host-thunk pool is the guest calling an HLE import slot
  // we bound but cannot actually service. Name it: the raw address is inside
  // FEX's reserved heap and looks like unrelated garbage otherwise.
  {
    u32 ti = 0;
    if (const char *tn =
            cpu::hostThunkNameForAddr((uintptr_t)si->si_addr, &ti))
      BASE_LOGI("crashHandler",
                "  ^ inside the HLE host-thunk pool: thunk #{} {}", ti,
                *tn ? tn : "(bound without a name)");
  }
  // Show what the host VA space holds around the fault: which mapping it hit,
  // or which two mappings it fell between. Async-signal-safe (read+write only).
  if (si->si_addr) {
    const u64 fa = (u64)si->si_addr;
    int mf = open("/proc/self/maps", O_RDONLY);
    if (mf >= 0) {
      // Stream the file a line at a time. Slurping it into one fixed buffer
      // silently truncated instead: a guest process has tens of thousands of
      // mappings, /proc/self/maps runs past a megabyte, the fill loop's count
      // went to zero at the brim, and the walk then fell off the end and
      // reported the LAST (half-read) line as the neighbour of the fault. Every
      // SotC fault dump so far "landed next to /dev/nvidiactl" for that reason
      // alone -- the one fact the block exists to establish, whether the
      // faulting page was mapped at all, was the fact it destroyed.
      static char buf[65536];
      static char prev[512];
      size_t held = 0;      // bytes of a partial line kept at buf's front
      bool havePrev = false, found = false, eof = false;
      int after = -1;       // counts the trailing lines once the hit is printed
      while (!found || after >= 0) {
        if (!eof) {
          ssize_t r = read(mf, buf + held, sizeof(buf) - held);
          if (r > 0)
            held += (size_t)r;
          else
            eof = true;
        }
        size_t start = 0;
        for (;;) {
          char *nl = static_cast<char *>(
              memchr(buf + start, '\n', held - start));
          if (!nl)
            break;
          *nl = 0;
          char *line = buf + start;
          start = (size_t)(nl - buf) + 1;
          if (after >= 0) {  // trailing context after the hit
            BASE_LOGI("crashHandler", "  maps  +{} : {}", after + 1, line);
            if (++after >= 3) { after = -1; found = true; }
            continue;
          }
          const u64 lo = strtoull(line, nullptr, 16);
          const char *dash = strchr(line, '-');
          const u64 hi = dash ? strtoull(dash + 1, nullptr, 16) : 0;
          if (fa < hi) {
            if (havePrev)
              BASE_LOGI("crashHandler", "  maps prev: {}", prev);
            BASE_LOGI("crashHandler", "  maps {} : {}",
                      (fa >= lo && fa < hi) ? "HIT " : "next (fault is in a "
                                                       "GAP -- unmapped)",
                      line);
            after = 0;
            continue;
          }
          size_t len = (size_t)(nl - line);
          if (len >= sizeof(prev)) len = sizeof(prev) - 1;
          memcpy(prev, line, len);
          prev[len] = 0;
          havePrev = true;
        }
        held -= start;
        memmove(buf, buf + start, held);
        if (held == sizeof(buf))  // a single line longer than the buffer
          held = 0;
        if (eof && held == 0) {
          if (!found && after < 0)
            BASE_LOGI("crashHandler",
                      "  maps: fault is above every mapping (last was {})",
                      havePrev ? prev : "(none)");
          break;
        }
      }
      close(mf);
    }
  }
#if defined(__x86_64__)
  // Native x86 host: the host signal context IS the guest context.
  auto *uc = static_cast<ucontext_t *>(ucv);
  auto *gr = uc->uc_mcontext.gregs;
  char rip[256];
  symbolize(gr[REG_RIP], rip, sizeof(rip));
  BASE_LOGI("crashHandler", "  rip   = {:016x}  {}",
            (unsigned long long)gr[REG_RIP], rip);
  BASE_LOGI("crashHandler", "  rax={:016x} rbx={:016x} rcx={:016x} rdx={:016x}",
            (unsigned long long)gr[REG_RAX], (unsigned long long)gr[REG_RBX],
            (unsigned long long)gr[REG_RCX], (unsigned long long)gr[REG_RDX]);
  BASE_LOGI("crashHandler", "  rsi={:016x} rdi={:016x} rbp={:016x} rsp={:016x}",
            (unsigned long long)gr[REG_RSI], (unsigned long long)gr[REG_RDI],
            (unsigned long long)gr[REG_RBP], (unsigned long long)gr[REG_RSP]);
  BASE_LOGI("crashHandler", "  r8 ={:016x} r9 ={:016x} r10={:016x} r11={:016x}",
            (unsigned long long)gr[REG_R8], (unsigned long long)gr[REG_R9],
            (unsigned long long)gr[REG_R10], (unsigned long long)gr[REG_R11]);
  BASE_LOGI("crashHandler", "  r12={:016x} r13={:016x} r14={:016x} r15={:016x}",
            (unsigned long long)gr[REG_R12], (unsigned long long)gr[REG_R13],
            (unsigned long long)gr[REG_R14], (unsigned long long)gr[REG_R15]);
  // A jump into unmapped memory faults with rip THERE, so the opcode dump would
  // fault again and bury the report -- check the page first.
  if (gr[REG_RIP] && [&] {
        const long pgsz = sysconf(_SC_PAGESIZE);
        unsigned char vec = 0;
        return mincore(reinterpret_cast<void *>((uintptr_t)gr[REG_RIP] &
                                                ~((uintptr_t)pgsz - 1)),
                       1, &vec) == 0;
      }()) {
    auto *b = reinterpret_cast<const u8 *>(gr[REG_RIP]);
    base::String insn;
    base::FormatTo(insn, "  insn bytes:");
    for (int i = 0; i < 16; i++)
      base::FormatTo(insn, " {:02x}", b[i]);
    BASE_LOGI("crashHandler", "{}", insn.c_str());
  }
  // The host TLS base the faulting code was using. A fault in host library code
  // reached from the guest is usually this: something left the host fs base
  // wrong, so every `errno = x` writes near zero instead of into host TLS.
  {
    unsigned long fsb = 0;
    if (::syscall(SYS_arch_prctl, 0x1003 /*ARCH_GET_FS*/, &fsb) == 0)
      BASE_LOGI("crashHandler", "  host fs base = {:#x}", fsb);
  }
  // A rip in HOST code symbolizes to nothing above; name the library and the
  // nearest exported symbol so the handler that was running is identifiable.
  {
    char sym[256];
    symbolize(gr[REG_RIP], sym, sizeof(sym));
    if (std::strstr(sym, "(??)")) {
      Dl_info di{};
      if (dladdr(reinterpret_cast<void *>(gr[REG_RIP]), &di) && di.dli_fname)
        BASE_LOGI("crashHandler", "  host rip = {}+{:#x}  {}+{:#x}",
                  di.dli_fname,
                  (unsigned long)(gr[REG_RIP] - (uintptr_t)di.dli_fbase),
                  di.dli_sname ? di.dli_sname : "?",
                  di.dli_saddr
                      ? (unsigned long)(gr[REG_RIP] - (uintptr_t)di.dli_saddr)
                      : 0ul);
    }
  }
  // DELTA_GUEST_BRK_DUMP=<reg>|all: follow an argument register one level. A
  // planted breakpoint lands where some object is about to be used, and what
  // that object POINTS AT (a byte stream, a descriptor) is the whole question --
  // registers alone say which object, not what is in it.
  if (const char *rn = kBrkDump) {
    static const struct {
      const char *name;
      int idx;
    } kRegs[] = {{"rdi", REG_RDI}, {"rsi", REG_RSI}, {"rdx", REG_RDX},
                 {"rcx", REG_RCX}, {"rbx", REG_RBX}, {"r14", REG_R14},
                 {"r13", REG_R13}, {"r8", REG_R8},   {"r9", REG_R9},
                 {"r11", REG_R11}, {"r12", REG_R12}, {"r15", REG_R15},
                 {"rax", REG_RAX}};
    for (const auto &r : kRegs) {
      if (std::strcmp(rn, r.name) != 0 && std::strcmp(rn, "all") != 0)
        continue;
      const uintptr_t base = gr[r.idx];
      // A register holding a non-pointer must not take the handler down with
      // it: "all" is the mode used when the interesting register is unknown.
      if (base < 0x10000 || !trkMincore(base))
        continue;
      base::String qwords;
      base::FormatTo(qwords, "  --- {} = {:#x} ---", r.name,
                     (unsigned long long)base);
      const auto *q = reinterpret_cast<const u64 *>(base);
      for (int i = 0; i < 16; i++) {
        if (i % 4 == 0)
          base::FormatTo(qwords, "\n  {}+{:03x}:", r.name, i * 8);
        base::FormatTo(qwords, " {:016x}", (unsigned long long)q[i]);
      }
      base::FormatTo(qwords, "\n");
      BASE_LOGI("crashHandler", "{}", qwords.c_str());
      for (int i = 0; i < 16; i++) {
        const uintptr_t p = q[i];
        if (p < 0x8000000000ull || p >= 0x8100000000ull)
          continue;
        const auto *b8 = reinterpret_cast<const u8 *>(p);
        base::String b8bytes;
        base::FormatTo(b8bytes, "  {}+{:03x} -> {:#x}:", r.name, i * 8,
                       (unsigned long long)p);
        for (int k = 0; k < 48; k++)
          base::FormatTo(b8bytes, " {:02x}", b8[k]);
        BASE_LOGI("crashHandler", "{}", b8bytes.c_str());
      }
    }
    std::fflush(stderr);
  }
  // DELTA_GUEST_BRK_PEEK=<hex addr>[:<bytes>]: dump a fixed guest address. The
  // register-following dump can only reach what a register points at, and the
  // object in question is often one the guest reached by arithmetic (a
  // compressed pointer, a heap offset) that no register holds.
  // "scan:<hex base>:<hex size>" instead reports, per 4 KiB block, how many of
  // its bytes are non-zero -- which is how a region that should hold a heap but
  // reads empty shows where the real data went.
  if (const char *pk = kBrkPeek; pk && std::strncmp(pk, "scan:", 5) == 0) {
    const uintptr_t base = std::strtoull(pk + 5, nullptr, 16);
    const char *c2 = std::strchr(pk + 5, ':');
    const u64 size = c2 ? std::strtoull(c2 + 1, nullptr, 16) : 0x100000;
    const long pgsz = sysconf(_SC_PAGESIZE);
    for (u64 off = 0; off < size; off += 0x1000) {
      unsigned char vec = 0;
      const uintptr_t a = base + off;
      if (mincore(reinterpret_cast<void *>(a & ~((uintptr_t)pgsz - 1)), 1,
                  &vec) != 0)
        continue;
      const auto *b = reinterpret_cast<const u8 *>(a);
      unsigned nz = 0;
      for (int i = 0; i < 0x1000; i++)
        nz += b[i] != 0;
      if (nz)
        BASE_LOGI("crashHandler", "  scan {:#x}: {}/4096 non-zero",
                  (unsigned long long)a, nz);
    }
    std::fflush(stderr);
  // "find:<hex base>:<hex size>:<hex value>" reports every 8-byte slot in the
  // range holding that value -- how you name the field a wild pointer came from.
  } else if (const char *pk = kBrkPeek;
             pk && std::strncmp(pk, "find:", 5) == 0) {
    const uintptr_t base = std::strtoull(pk + 5, nullptr, 16);
    const char *c2 = std::strchr(pk + 5, ':');
    const u64 size = c2 ? std::strtoull(c2 + 1, nullptr, 16) : 0x100000;
    const char *c3 = c2 ? std::strchr(c2 + 1, ':') : nullptr;
    const u64 want = c3 ? std::strtoull(c3 + 1, nullptr, 16) : 0;
    const long pgsz = sysconf(_SC_PAGESIZE);
    int hits = 0;
    for (u64 off = 0; off < size && hits < 32; off += pgsz) {
      unsigned char vec = 0;
      const uintptr_t a = base + off;
      if (mincore(reinterpret_cast<void *>(a), 1, &vec) != 0)
        continue;
      const auto *q = reinterpret_cast<const u64 *>(a);
      for (long i = 0; i < pgsz / 8 && hits < 32; i++)
        if (q[i] == want) {
          BASE_LOGI("crashHandler", "  find {:#x} at {:#x}",
                    (unsigned long long)want,
                    (unsigned long long)(a + i * 8));
          hits++;
        }
    }
    BASE_LOGI("crashHandler", "  find: {} hit(s)", hits);
    std::fflush(stderr);
  } else if (const char *pk = kBrkPeek) {
    // "deref:<hex addr>:<bytes>" dumps what the POINTER at addr points at, for
    // a table the guest reaches through a field rather than a register.
    const bool deref = std::strncmp(pk, "deref:", 6) == 0;
    if (deref)
      pk += 6;
    uintptr_t at = std::strtoull(pk, nullptr, 16);
    const char *colon = std::strchr(pk, ':');
    const size_t n = colon ? std::strtoul(colon + 1, nullptr, 0) : 256;
    const long pgsz = sysconf(_SC_PAGESIZE);
    unsigned char vec = 0;
    if (deref && at >= 0x10000 &&
        mincore(reinterpret_cast<void *>(at & ~((uintptr_t)pgsz - 1)), 1,
                &vec) == 0) {
      const uintptr_t via = at;
      at = *reinterpret_cast<const uintptr_t *>(via);
      BASE_LOGI("crashHandler", "  peek deref {:#x} -> {:#x}",
                (unsigned long long)via, (unsigned long long)at);
    }
    if (at >= 0x10000 &&
        mincore(reinterpret_cast<void *>(at & ~((uintptr_t)pgsz - 1)), 1,
                &vec) == 0) {
      if (FILE *m = std::fopen("/proc/self/maps", "r")) {
        char line[512];
        while (std::fgets(line, sizeof(line), m)) {
          unsigned long lo = 0, hi = 0;
          if (std::sscanf(line, "%lx-%lx", &lo, &hi) == 2 && at >= lo &&
              at < hi) {
            BASE_LOGI("crashHandler", "  peek maps: {}", line);
            break;
          }
        }
        std::fclose(m);
      }
      const auto *b = reinterpret_cast<const u8 *>(at);
      base::String peekbytes;
      for (size_t i = 0; i < n; i++) {
        if (i % 32 == 0)
          base::FormatTo(peekbytes, "\n  peek {:#x}:",
                         (unsigned long long)(at + i));
        base::FormatTo(peekbytes, " {:02x}", b[i]);
      }
      base::FormatTo(peekbytes, "\n");
      BASE_LOGI("crashHandler", "{}", peekbytes.c_str());
    } else {
      BASE_LOGI("crashHandler", "  peek {:#x}: not mapped",
                (unsigned long long)at);
    }
    std::fflush(stderr);
  }
  backtrace(gr[REG_RBP]);
  // Raw stack scan: optimised guest code omits frame pointers, so the rbp chain
  // above misses frames. Scan the guest stack for values that land in a loaded
  // module's .text (the real call chain) and for pointers into the guest heap
  // arena that hold a printable ASCII string (an asset filename / tag the
  // faulting code was handling). The arena (0x40_0000_0000..0x41_0000_0000) is
  // always mapped, so reading a string there can't fault.
  // DELTA_CRASH_PEEK: the raw top of the stack. The symbolising scan below only
  // prints values that land in a module's .text, which hides the operand a bad
  // control transfer came through.
  if (kCrashPeek && gr[REG_RSP] >= 0x10000) {
    auto *q = reinterpret_cast<const u64 *>(gr[REG_RSP] & ~7ull);
    base::String rspwords;
    for (int i = 0; i < 16; i++) {
      if (i % 4 == 0)
        base::FormatTo(rspwords, "\n  rsp+{:02x}:", i * 8);
      base::FormatTo(rspwords, " {:016x}", (unsigned long long)q[i]);
    }
    base::FormatTo(rspwords, "\n");
    BASE_LOGI("crashHandler", "{}", rspwords.c_str());
    // ...and the objects rbx/r12 point at. A guest breakpoint's registers name
    // a `this`, and the fields behind it are the whole reason for stopping
    // there: without them a dump says which code ran, never what it was
    // looking at. Guest heap only, and only when the range reads.
    const struct { const char *name; u64 v; } objs[] = {
        {"rbx", (u64)gr[REG_RBX]}, {"r12", (u64)gr[REG_R12]}};
    for (const auto &o : objs) {
      if (o.v < 0x1000000000ull || o.v >= 0x20000000000ull ||
          !utl::isMemoryRangeMapped(reinterpret_cast<const void *>(o.v), 128))
        continue;
      const auto *q = reinterpret_cast<const u64 *>(o.v);
      base::String words;
      for (int i = 0; i < 16; i++) {
        if (i % 4 == 0)
          base::FormatTo(words, "\n  [{}+{:02x}]:", o.name, i * 8);
        base::FormatTo(words, " {:016x}", (unsigned long long)q[i]);
      }
      BASE_LOGI("crashHandler", "{}", words.c_str());
    }
  }
  BASE_LOGI("crashHandler", "  --- stack scan ---");
  if (uintptr_t rsp = gr[REG_RSP]; rsp >= 0x10000) {
    auto *sp = reinterpret_cast<uintptr_t *>(rsp);
    for (int i = 0; i < 512; i++) {
      uintptr_t v = sp[i];
      char sym[256];
      symbolize(v, sym, sizeof(sym));
      if (std::strstr(sym, "(.text)")) {
        BASE_LOGI("crashHandler", "  sp+{:<5x} {:016x}  {}", i * 8, v, sym);
      } else if (v >= 0x4000000000ull && v < 0x4100000000ull) {
        auto *s = reinterpret_cast<const char *>(v);
        int n = 0;
        while (n < 40 && s[n] >= 0x20 && s[n] <= 0x7e) n++;
        if (n >= 5 && s[n] == 0)
          BASE_LOGI("crashHandler", "  sp+{:<5x} {:016x}  str=\"{:.40}\"", i * 8,
                    v, s);
      }
    }
  }
#else
  // aarch64 host: guest x86 state lives in the FEXCore CPUState, not the host
  // ARM signal context. Reconstruct the precise guest RIP from the host JIT PC
  // (CPUState.rip alone is only block-accurate and multiblock hides the site).
  u64 hostpc = 0;
#if defined(__aarch64__)
  if (ucv)
    hostpc = static_cast<ucontext_t *>(ucv)->uc_mcontext.pc;
#endif
  u64 recon = cpu::reconstructGuestRip(hostpc);
  u64 grip = recon ? recon : cpu::currentGuestRip(); // fall back to block rip
  BASE_LOGI("crashHandler", "  host pc in JIT: {}",
            recon ? "yes" : "no (FEX/HLE C++)");
  char ripsym[256];
  symbolize(grip, ripsym, sizeof(ripsym));
  BASE_LOGI("crashHandler", "  host pc   = {:016x}", (unsigned long long)hostpc);
  BASE_LOGI("crashHandler", "  guest rip = {:016x}  {}",
            (unsigned long long)grip, ripsym);
  if (grip) {
    auto *b = reinterpret_cast<const u8 *>(grip);
    base::String insnb;
    base::FormatTo(insnb, "  insn bytes:");
    for (int i = 0; i < 16; i++)
      base::FormatTo(insnb, " {:02x}", b[i]);
    BASE_LOGI("crashHandler", "{}", insnb.c_str());
  }
  // Guest GPR dump + rbp backtrace (parity with the native x86 dump above).
  // gregs order is FEXCore::X86State::REG_* (RAX,RCX,RDX,RBX,RSP,RBP,RSI,RDI,
  // R8..R15); mirror it locally so this TU needs no FEXCore headers.
  //
  // Take the registers from the SIGNAL CONTEXT, not from the in-memory CPUState.
  // FEX pins every guest GPR to a fixed host register, so the host context holds
  // the values AT the faulting instruction; CPUState.gregs is only written back
  // when the JIT leaves a block, and FEX's syscall op spills just the subset the
  // syscall ABI reads. Reading it here reports whichever registers the thread's
  // last syscall happened to publish, dressed up as the fault state -- which is
  // exactly how SotC's New-Game crash got diagnosed as a fiber running on a
  // freed stack: rdi/rsi were verbatim the last sys_umtx_op's arguments, and the
  // "faulting" rax-8 disagreed with si_addr in every log. Label the fallback so
  // a stale dump can never again be mistaken for a precise one.
  u64 sig_gregs[16];
  const bool gexact = cpu::guestGregsFromSignal(ucv, sig_gregs);
  if (!gexact)
    BASE_LOGI("crashHandler",
              "  [regs] NOT from the fault: host pc is outside the JIT, so "
              "these are the last spilled CPUState values (STALE)");
  if (const u64 *g = gexact ? sig_gregs : cpu::currentGuestGregs()) {
    enum { RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15 };
    BASE_LOGI("crashHandler", "  rax={:016x} rbx={:016x} rcx={:016x} rdx={:016x}",
              (unsigned long long)g[RAX], (unsigned long long)g[RBX],
              (unsigned long long)g[RCX], (unsigned long long)g[RDX]);
    BASE_LOGI("crashHandler", "  rsi={:016x} rdi={:016x} rbp={:016x} rsp={:016x}",
              (unsigned long long)g[RSI], (unsigned long long)g[RDI],
              (unsigned long long)g[RBP], (unsigned long long)g[RSP]);
    BASE_LOGI("crashHandler", "  r8 ={:016x} r9 ={:016x} r10={:016x} r11={:016x}",
              (unsigned long long)g[R8], (unsigned long long)g[R9],
              (unsigned long long)g[R10], (unsigned long long)g[R11]);
    BASE_LOGI("crashHandler", "  r12={:016x} r13={:016x} r14={:016x} r15={:016x}",
              (unsigned long long)g[R12], (unsigned long long)g[R13],
              (unsigned long long)g[R14], (unsigned long long)g[R15]);

    // Corroborate the recovery against si_addr: a faulting memory operand is
    // built out of a base register, so SOME GPR should sit within a small
    // displacement of the address that faulted. If none does, the recovery is
    // wrong (vendored FEX's x64::SRA moved under us) and every conclusion drawn
    // from the dump is worthless -- say so instead of printing plausible lies.
    if (gexact && si && si->si_addr) {
      static const char *kN[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp",
                                   "rsi", "rdi", "r8",  "r9",  "r10", "r11",
                                   "r12", "r13", "r14", "r15"};
      const u64 fa = (u64)si->si_addr;
      bool any = false;
      for (int i = 0; i < 16; i++) {
        const i64 d = (i64)fa - (i64)g[i];
        if (d >= -0x2000 && d <= 0x2000) {
          BASE_LOGI("crashHandler",
                    "  [regs] fault = {}{:+}  (exact, from the JIT "
                    "signal context)", kN[i], (long long)d);
          any = true;
        }
      }
      if (!any)
        BASE_LOGI("crashHandler",
                  "  [regs] WARNING no GPR is within 0x2000 of the fault "
                  "address -- the SRA recovery is suspect, do not trust "
                  "these values");
    }

    // ---- SOTC free-tree walk (diagnostic; see helper above) ----
    // Fire when the fault is inside the eboot's size-ordered free-tree insert
    // (+0x48a70..+0x48b64), which is where every heap-corruption fault in this
    // title lands. r15 is arg0 (the allocator state) and rdx the size being
    // inserted; both are live for the whole loop, so the walk can be replayed.
    {
      u64 ebase2 = 0;
      if (auto *proc = proc::getActive()) {
        for (auto &mod : proc->getModuleList()) {
          auto &mi = mod->getInfo();
          auto *t = mi.textSeg.addr;
          if (t && grip >= (uintptr_t)t && grip < (uintptr_t)t + mi.textSeg.size) {
            ebase2 = (u64)t;
            break;
          }
        }
      }
      const u64 off2 = ebase2 ? grip - ebase2 : 0;
      if (gexact && ebase2 && off2 >= 0x48a70 && off2 < 0x48b64)
        sotcWalkFreeTree(g[R15], g[RDX] & ~7ull);
    }

    // ---- SOTC AllocationTracker walk (diagnostic; see helper above) ----
    // Fire only when the fault is inside the eboot's slot-21 untrack-on-free
    // methods: fn @+0x18920 (CPU tracker) / +0x8d930 (GPU tracker). Resolve the
    // eboot base from the module whose .text contains grip (== textSeg.addr, so
    // grip-base is the module-relative ELF vaddr), like symbolize() does.
    {
      u64 ebase = 0;
      if (auto *proc = proc::getActive()) {
        for (auto &mod : proc->getModuleList()) {
          auto &mi = mod->getInfo();
          auto *t = mi.textSeg.addr;
          if (t && grip >= (uintptr_t)t && grip < (uintptr_t)t + mi.textSeg.size) {
            ebase = (u64)t;
            break;
          }
        }
      }
      u64 off = ebase ? grip - ebase : 0;
      bool inTrk = ebase && ((off >= 0x18000 && off < 0x19000) ||
                             (off >= 0x8d000 && off < 0x8e000));
      if (inTrk) {
        BASE_LOGI("trkwalk",
                  "\n  === SOTC tracker walk (eboot base={:#x}, fault off={:#x}) "
                  "===",
                  (unsigned long long)ebase, (unsigned long long)off);
        // Recover (tracker,key) from the saved-register slots on the stack,
        // which are reliable regardless of FEX callee-saved reconstruction.
        // Layout after the prologue pushes (no further stack alloc):
        //   [rsp+0x18]=saved r13=TRACKER   [rsp+0x20]=saved r14=KEY
        u64 rsp = g[RSP];
        u64 trkStk = 0, keyStk = 0;
        bool haveStk = trkRd64(rsp + 0x18, trkStk) && trkRd64(rsp + 0x20, keyStk);
        BASE_LOGI("trkwalk",
                  " from-stack: tracker={:#x} key={:#x} (ok={}) | "
                  "from-reg: r13={:#x} r14={:#x}",
                  (unsigned long long)trkStk, (unsigned long long)keyStk,
                  haveStk, (unsigned long long)g[R13], (unsigned long long)g[R14]);
        // Prefer the stack-recovered tracker/key; fall back to the regs if the
        // stack slot doesn't look like a mapped module-space pointer.
        auto plausible = [](u64 t) {
          return t >= 0x200000000000ull && t < 0x210000000000ull && trkMincore(t);
        };
        u64 tracker = plausible(trkStk) ? trkStk : g[R13];
        u64 key = (haveStk && keyStk) ? keyStk : g[R14];
        // Raw tracker header window (fallback context: +0x00..0xa0).
        if (trkMincore(tracker)) {
          auto *q = reinterpret_cast<const u64 *>(tracker);
          base::String trkwords;
          for (int i = 0; i < 20; i++) {
            if ((i % 4) == 0)
              base::FormatTo(trkwords, "\n  trk+{:03x}:", i * 8);
            base::FormatTo(trkwords, " {:016x}", (unsigned long long)q[i]);
          }
          base::FormatTo(trkwords, "\n");
          BASE_LOGI("trkwalk", "{}", trkwords.c_str());
        }
        // 1) Walk the tracker the fault came from.
        bool inThis = sotcWalkTracker(tracker, key, "fault");
        // 2) Cross-check the other known tracker instances (the two GPU/renderer
        //    tracker globals @ base+0x2ed3350 / +0x2ed33b0) for the same key.
        u64 gpuA = ebase + 0x2ed3350, gpuB = ebase + 0x2ed33b0;
        bool inA = false, inB = false;
        if (gpuA != tracker) inA = sotcWalkTracker(gpuA, key, "gpuA");
        if (gpuB != tracker) inB = sotcWalkTracker(gpuB, key, "gpuB");
        // 3) Emulator-side VMA view of the key.
        if (auto *proc = proc::getActive()) {
          auto *pi = proc->getVma().get(reinterpret_cast<u8 *>(key));
          if (pi) {
            u64 vb = (u64)pi->ptr, ve = vb + pi->size;
            BASE_LOGI("trkwalk",
                      " emu VMA: key {:#x} is IN region [{:#x},{:#x}) "
                      "size={:#x} sceProt={:#x} reserved={} name={}  "
                      "key-regionbase={:+}",
                      (unsigned long long)key, (unsigned long long)vb,
                      (unsigned long long)ve, (unsigned long long)pi->size,
                      pi->sceProt, pi->reserved, pi->name ? pi->name : "(null)",
                      (long long)(key - vb));
          } else {
            BASE_LOGI("trkwalk",
                      " emu VMA: key {:#x} is in NO tracked region",
                      (unsigned long long)key);
          }
        }
        BASE_LOGI("trkwalk",
                  " SUMMARY: key covered in fault-tracker={} gpuA={} gpuB={}",
                  inThis, inA, inB);
        BASE_LOGI("trkwalk", "  === end SOTC tracker walk ===\n");
        std::fflush(stderr);
      }
    }

    // DELTA_CRASH_PEEK: dump a window of guest memory around each GPR that points
    // into loaded-module space (>= 0x2000_0000_0000). An indirect call/jmp through
    // a garbage/null vtable slot is our most common late-boot fault; seeing the
    // object bytes + where the slot points identifies the uninitialised object.
    if (const char *pk = kCrashPeek) {
      const u64 regs[] = {g[RAX], g[RBX], g[RDI], g[RSI], g[RCX], g[RDX]};
      const char *rn[] = {"rax", "rbx", "rdi", "rsi", "rcx", "rdx"};
      for (int r = 0; r < 6; r++) {
        u64 base = regs[r];
        if (base < 0x200000000000ull || base >= 0x210000000000ull)
          continue;  // only the module VA window is reliably mapped to read
        auto *q = reinterpret_cast<const u64 *>(base);
        // rax/rbx are the usual object/this pointers: dump far enough to cover a
        // deep vtable/member-fn slot (the fault operand disp can be several
        // hundred bytes in). Other regs get just a header.
        int n = (r < 2) ? 56 : 8;
        base::String peekr;
        for (int i = 0; i < n; i++) {
          if ((i % 4) == 0)
            base::FormatTo(peekr, "\n  peek {}+{:03x}:", rn[r], i * 8);
          base::FormatTo(peekr, " {:016x}", (unsigned long long)q[i]);
        }
        base::FormatTo(peekr, "\n");
        BASE_LOGI("crashHandler", "{}", peekr.c_str());
      }
      // Explicit address list: DELTA_CRASH_PEEK=0x1c92d00,0x1c2fc00 also dumps a
      // window of guest memory at each given VA (comma/space separated). Unlike the
      // register scan this is NOT restricted to the loaded-module window, so it can
      // read low ET_SCE_EXEC globals/BSS (e.g. a null singleton pointer). Guarded by
      // mincore so an unmapped address can't fault the crash handler.
      for (const char *p = pk; *p;) {
        while (*p == ',' || *p == ' ') p++;
        char *end = nullptr;
        u64 va = std::strtoull(p, &end, 0);
        if (end == p) { if (*p) p++; continue; }
        p = end;
        if (va < 0x10000) continue;  // "1"/tiny -> register scan only
        long pg = sysconf(_SC_PAGESIZE);
        unsigned char vec[2] = {0, 0};
        void *pa = reinterpret_cast<void *>(va & ~((u64)pg - 1));
        if (mincore(pa, 1, vec) != 0) {
          BASE_LOGI("crashHandler", "  peek {:#x}: <unmapped>",
                    (unsigned long long)va);
          continue;
        }
        auto *q = reinterpret_cast<const u64 *>(va);
        base::String peeks;
        for (int i = 0; i < 16; i++) {
          if ((i % 4) == 0)
            base::FormatTo(peeks, "\n  peek {:#x}+{:03x}:",
                           (unsigned long long)va, i * 8);
          base::FormatTo(peeks, " {:016x}", (unsigned long long)q[i]);
        }
        base::FormatTo(peeks, "\n");
        BASE_LOGI("crashHandler", "{}", peeks.c_str());
      }
    }
    // DELTA_CRASH_PEEK also dumps the raw stack window around rsp: for a fault
    // inside a leaf helper (e.g. a lookup that returned null) the caller's
    // locals -- the key being freed, the object under operation -- are the
    // fastest route to "what data was this actually working on".
    if (kCrashPeek && g[RSP] >= 0x10000) {
      long pg = sysconf(_SC_PAGESIZE);
      unsigned char vec[2] = {0, 0};
      void *pa = reinterpret_cast<void *>(g[RSP] & ~((u64)pg - 1));
      if (mincore(pa, 1, vec) == 0) {
        auto *q = reinterpret_cast<const u64 *>(g[RSP] & ~7ull);
        base::String stackwords;
        for (int i = -8; i < 64; i++) {
          if (((i + 8) % 4) == 0)
            base::FormatTo(stackwords, "\n  stack rsp{:+05x}:", i * 8);
          base::FormatTo(stackwords, " {:016x}", (unsigned long long)q[i]);
        }
        base::FormatTo(stackwords, "\n");
        BASE_LOGI("crashHandler", "{}", stackwords.c_str());
      }
    }
    // DELTA_GUEST_BRK_DUMP=<reg>: follow an argument register one level. A
    // planted breakpoint usually lands where some object is about to be used,
    // and what that object POINTS AT (a byte stream, a descriptor) is the whole
    // question -- registers alone only say which object, not what is in it.
    if (const char *rn = kBrkDump) {
      static const struct {
        const char *name;
        int idx;
      } kRegs[] = {{"rdi", RDI}, {"rsi", RSI},   {"rdx", RDX}, {"rcx", RCX},
                   {"rbx", RBX}, {"r14", R14},   {"r13", R13}, {"r8", R8},
                   {"r9", R9},   {"r11", R11},   {"r12", R12}, {"r15", R15},
                   {"rax", RAX}};
      for (const auto &r : kRegs) {
        if (std::strcmp(rn, r.name) != 0 && std::strcmp(rn, "all") != 0)
          continue;
        const uintptr_t base = g[r.idx];
        // A register holding a non-pointer must not take the handler down with
        // it: "all" is the mode used when the interesting register is unknown.
        if (base < 0x10000 || !trkMincore(base))
          continue;
        base::String qw;
        base::FormatTo(qw, "  --- {} = {:#x} ---\n", r.name,
                       (unsigned long long)base);
        const auto *q = reinterpret_cast<const u64 *>(base);
        for (int i = 0; i < 16; i++) {
          if (i % 4 == 0)
            base::FormatTo(qw, "\n  {}+{:03x}:", r.name, i * 8);
          base::FormatTo(qw, " {:016x}", (unsigned long long)q[i]);
        }
        base::FormatTo(qw, "\n");
        BASE_LOGI("crashHandler", "{}", qw.c_str());
        // One level down: any field that looks like a guest pointer gets its
        // first 64 bytes dumped, which is where a byte stream shows itself.
        for (int i = 0; i < 16; i++) {
          const uintptr_t p = q[i];
          if (p < 0x8000000000ull || p >= 0x8100000000ull)
            continue;
          const auto *b = reinterpret_cast<const u8 *>(p);
          base::String b8b;
          base::FormatTo(b8b, "  {}+{:03x} -> {:#x}:", r.name, i * 8,
                         (unsigned long long)p);
          for (int k = 0; k < 48; k++)
            base::FormatTo(b8b, " {:02x}", b[k]);
          BASE_LOGI("crashHandler", "{}", b8b.c_str());
        }
      }
      std::fflush(stderr);
    }
    // DELTA_GUEST_BRK_PEEK=<hex addr>[:<bytes>]: dump a fixed guest address.
    // The register-following dump can only reach what a register points at, and
    // the object in question is often one the guest reached by arithmetic (a
    // compressed pointer, a heap offset) that no register holds.
    if (const char *pk = kBrkPeek) {
      const uintptr_t at = std::strtoull(pk, nullptr, 16);
      const char *colon = std::strchr(pk, ':');
      const size_t n = colon ? std::strtoul(colon + 1, nullptr, 0) : 256;
      long pgsz = sysconf(_SC_PAGESIZE);
      unsigned char vec = 0;
      if (at >= 0x10000 &&
          mincore(reinterpret_cast<void *>(at & ~((uintptr_t)pgsz - 1)), 1,
                  &vec) == 0) {
        const auto *b = reinterpret_cast<const u8 *>(at);
        base::String pbytes;
        for (size_t i = 0; i < n; i++) {
          if (i % 32 == 0)
            base::FormatTo(pbytes, "\n  peek {:#x}:",
                           (unsigned long long)(at + i));
          base::FormatTo(pbytes, " {:02x}", b[i]);
        }
        base::FormatTo(pbytes, "\n");
        BASE_LOGI("crashHandler", "{}", pbytes.c_str());
      } else {
        BASE_LOGI("crashHandler", "  peek {:#x}: not mapped",
                  (unsigned long long)at);
      }
      std::fflush(stderr);
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
    BASE_LOGI("crashHandler", "  --- stack scan ---");
    auto *sp = reinterpret_cast<uintptr_t *>(g[RSP]);
    if (g[RSP] >= 0x10000) {
      for (int i = 0; i < 256; i++) {
        uintptr_t v = sp[i];
        char sym[256];
        symbolize(v, sym, sizeof(sym));
        if (std::strstr(sym, "(.text)"))
          BASE_LOGI("crashHandler", "  sp+{:<4x} {:016x}  {}", i * 8, v, sym);
      }
    }
  }
#endif
  std::fflush(stderr);
  std::fflush(stdout);  // _Exit won't flush; keep the guest trace up to the fault
  std::_Exit(128 + sig);
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
void setRetTrace(uintptr_t addr, const char *label, bool isTest) {
  if (g_retCount < 8) {
    g_retAddrs[g_retCount] = addr;
    g_retLabels[g_retCount] = label;
    g_retIsTest[g_retCount] = isTest;
    g_retCount++;
  }
}

// Scan the CALLING thread's stack for return addresses that land in a loaded
// module and print them. A syscall handler runs on the guest stack (the native
// trampoline does not switch), so this names the guest code that reached the
// handler even when no frame pointer is available -- which is the only way to
// see why a title's worker thread bailed out of its own loop.
static thread_local uintptr_t t_guestSp = 0;
void setGuestStackScanBase(uintptr_t sp) { t_guestSp = sp; }
uintptr_t guestStackScanBase() { return t_guestSp; }

// A stack scan walks towards the top of the stack and runs into the guard page
// at the end of it, so it cannot dereference the window directly: a diagnostic
// must not be the thing that kills the process. Copy it out through
// process_vm_readv, which reports an unmapped page instead of faulting on it.
static size_t copyStackWindow(uintptr_t base, uintptr_t *out, size_t words) {
  while (words) {
    iovec local{out, words * sizeof(uintptr_t)};
    iovec remote{reinterpret_cast<void *>(base), local.iov_len};
    if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) ==
        static_cast<ssize_t>(local.iov_len))
      return words;
    words /= 2;
  }
  return 0;
}

void guestStackTraceFrom(uintptr_t base, const char *tag, int maxFrames,
                         long tid) {
  if (!base)
    return;
  uintptr_t sp[512];
  const size_t n = copyStackWindow(base, sp, 512);
  int printed = 0;
  for (size_t i = 0; i < n && printed < maxFrames; i++) {
    char sym[256];
    symbolize(sp[i], sym, sizeof(sym));
    if (std::strstr(sym, "(.text)")) {
      BASE_LOGI(tag, "    tid={} sp+{:<5x} {}", tid, (unsigned)(i * 8), sym);
      printed++;
    }
  }
}

void guestStackTrace(const char *tag, int maxFrames) {
  uintptr_t here = 0;
  const uintptr_t base =
      t_guestSp ? t_guestSp : reinterpret_cast<uintptr_t>(&here);
  uintptr_t sp[512];
  const size_t n = copyStackWindow(base, sp, 512);
  BASE_LOGI(tag, "tid={} guest stack:", (long)gettid());
  int printed = 0;
  for (size_t i = 0; i < n && printed < maxFrames; i++) {
    char sym[256];
    symbolize(sp[i], sym, sizeof(sym));
    if (std::strstr(sym, "(.text)")) {
      BASE_LOGI(tag, "  sp+{:<5x} {:016x} {}", (unsigned)(i * 8), sp[i], sym);
      printed++;
    }
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
  // Let layers that cannot reach the kernel arm a watch (see utl::armWriteWatch):
  // the GPU only learns the address worth watching -- the descriptor pointer a
  // shader actually read -- while a draw is being processed.
  utl::setWriteWatchArmer([](uintptr_t addr, size_t bytes, unsigned everyMs) {
    startWriteWatch(addr, bytes, everyMs);
  });
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
#if defined(__x86_64__) || defined(__aarch64__)
  struct sigaction pa = {};
  pa.sa_sigaction = probeHandler;
  pa.sa_flags = SA_SIGINFO | SA_RESTART;  // don't abort the thread's blocking call
  sigemptyset(&pa.sa_mask);
  sigaction(SIGUSR1, &pa, nullptr);
#endif
}
}  // namespace krnl
