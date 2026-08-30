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
#include "kern/probe/probe_arm.h"
#include "lv2/dispatch.h"

#include <utl/mem.h>
#include "module.h"
#include "proc.h"
#include "vfs.h"
#include "cpu/cpu_backend.h"
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

namespace {
CsRangeDescriber g_csRangeDescriber = nullptr;
}
void setCsRangeDescriber(CsRangeDescriber fn) { g_csRangeDescriber = fn; }
static bool describeCsRange(u64 addr, char *out, size_t n) {
  return g_csRangeDescriber && g_csRangeDescriber(addr, out, n);
}

const u32 *currentGuestTidPtr();  // sys_thread.cpp: this thread's guest tid
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
void backtrace(uintptr_t rbp) {
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
      if (describeCsRange(field, csr, sizeof(csr)))
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
      if (describeCsRange(field, csr1, sizeof(csr1)))
        BASE_LOGI("freetree",
                  "    the STALE LINK is inside a compute staging range: {}",
                  csr1);
      else
        BASE_LOGI("freetree",
                  "    no compute staging range covers the stale link at {:#x}",
                  (unsigned long long)field);
      if (describeCsRange(cur, csr1, sizeof(csr1)))
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

struct NullGuard {
  uintptr_t addr;
  int greg;  // REG_* index to zero
  int len;   // faulting instruction length
};
NullGuard g_nullGuards[16] = {};
int g_nullGuardCount = 0;
static void crashHandler(int sig, siginfo_t *si, void *ucv) {
  if (probe::onSignal(sig, si, ucv))
    return;
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

  probe::onFatal();

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

void setNullGuard(uintptr_t addr, GuardReg reg, int insnLen) {
#if defined(__x86_64__)
  if (g_nullGuardCount >= 16) return;
  int greg = reg == GuardReg::rax ? REG_RAX : REG_RSI;
  g_nullGuards[g_nullGuardCount++] = {addr, greg, insnLen};
#else
  (void)addr; (void)reg; (void)insnLen;
#endif
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
    probe::startWriteWatch(addr, bytes, everyMs);
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
  probe::installThreadProbeHandler(pa);
  pa.sa_flags = SA_SIGINFO | SA_RESTART;  // don't abort the thread's blocking call
  sigemptyset(&pa.sa_mask);
  sigaction(SIGUSR1, &pa, nullptr);
#endif
}
}  // namespace krnl
