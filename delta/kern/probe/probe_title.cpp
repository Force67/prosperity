/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "kern/probe/probe.h"
#include "kern/probe/probe_arm.h"
#include "kern/probe/probe_internal.h"

#include <sys/mman.h>
#include "base/arch.h"
#include <thread>
#include <chrono>
#include <base.h>
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <utl/file.h>
#include <utl/mem.h>
#include <utl/options.h>
#include <utl/path.h>

#include "cpu/cpu_backend.h"
#include "kern/crash.h"
#include "kern/lv2/sys_dynlib.h"
#include "kern/lv2/sys_mem.h"
#include "kern/module.h"
#include "kern/proc.h"
#include "kern/lv2/ps5/ctor_probe.h"
#include "kern/vfs.h"
#include "runtime/vprx/vprx.h"

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
DELTA_OPTION(bool, kVoForceConnect, "DELTA_VO_FORCE_CONNECT", false);
DELTA_OPTION(const char *, kVoPatch, "DELTA_VO_PATCH", nullptr);
DELTA_OPTION(const char *, kTrapVaddr, "DELTA_TRAP_VADDR", nullptr);
DELTA_OPTION(const char *, kAllocTrace, "DELTA_ALLOC_TRACE", nullptr);
DELTA_OPTION(const char *, kHeapProf, "DELTA_HEAP_PROF", nullptr);
DELTA_OPTION(const char *, kHeapProfScope, "DELTA_HEAP_PROF_SCOPE", nullptr);
DELTA_OPTION(const char *, kCntTrace, "DELTA_CNT_TRACE", nullptr);
DELTA_OPTION(const char *, kFatalTrace, "DELTA_FATAL_TRACE", nullptr);
DELTA_OPTION(const char *, kHdrTrace, "DELTA_HDR_TRACE", nullptr);
DELTA_OPTION(const char *, kRdoffFix, "DELTA_RDOFF_FIX", nullptr);
DELTA_OPTION(const char *, kSkipFn, "DELTA_SKIP_FN", nullptr);
DELTA_OPTION(bool, kGfxctxWatch, "DELTA_GFXCTX_WATCH", false);
DELTA_OPTION(bool, kJobTrace, "DELTA_JOB_TRACE", false);
DELTA_OPTION(bool, kSotcMatTrace, "DELTA_SOTC_MATTRACE", false);
DELTA_OPTION(bool, kJobTraceClaim, "DELTA_JOB_TRACE_CLAIM", false);
DELTA_OPTION(bool, kPs5Dcbwatch, "DELTA_PS5_DCBWATCH", false);
DELTA_OPTION(int, kSotcAllocLock, "DELTA_SOTC_ALLOCLOCK", 0);
DELTA_OPTION(bool, kSotcTreeWatch, "DELTA_SOTC_TREEWATCH", false);
DELTA_OPTION(u64, kSotcTreeWalk, "DELTA_SOTC_TREEWALK", 0);
DELTA_OPTION(u64, kSotcUmtxAddr, "DELTA_SOTC_UMTX_ADDR", 0x200003420ull);
DELTA_OPTION(bool, kSotcHeapRoute, "DELTA_SOTC_HEAPROUTE", false);
DELTA_OPTION(u64, kSotcTreeNode, "DELTA_SOTC_TREEWATCH_NODE", 0x8052e00020ull);
DELTA_OPTION(u64, kSotcTreeState, "DELTA_SOTC_TREEWATCH_STATE", 0x8309e0fd20ull);
DELTA_OPTION(bool, kSotcJobmove, "DELTA_SOTC_JOBMOVE", false);
DELTA_OPTION(bool, kVoOplog, "DELTA_VO_OPLOG", false);
DELTA_OPTION(bool, kVoSkip580, "DELTA_VO_SKIP_580", false);
DELTA_OPTION(bool, kVoWatch, "DELTA_VO_WATCH", false);
}  // namespace

namespace krnl {
const u32 *currentGuestTidPtr();  // sys_thread.cpp: this thread's guest tid
}

namespace krnl::probe {

namespace {
std::atomic<u64> g_jobClaims{0}, g_jobFails{0};

// Host-side JobSystem watcher. Spawned once with the guest jobsys base
// (identity-mapped, safe to read from a host thread). Everything expensive
// happens here, off the guest's hot paths.
void spawnJobWatcher(u64 base);

void PS4ABI jobTraceLogger(u64 hookId, u64 a0, u64 a1,
                           u64 a2, u64 a3, u64 ret) {
  switch (hookId) {
  case 12: { // CLAIM 0x38d40 -> ret = claimed job ptr (0 = failed to claim)
    // Log each worker's ordinal (fs:[-8]) exactly once, cheaply (thread_local).
    thread_local bool logged = false;
    if (!logged) {
      logged = true;
      // Replicate get-core-ordinal fn 0x33350: tcb = *(fsbase); v = [tcb-0x10];
      // ord = (v & 0x8000) ? (v & ~0x8000) : -1.  This is the value the claim
      // path (0x38d70/0x38d7f) uses for 1<<ord AND the direct-assign slot index.
      u64 fsb = cpu::currentGuestFsBase();
      u64 tcb = fsb ? *reinterpret_cast<u64 *>(fsb) : 0;
      u64 v = tcb ? *reinterpret_cast<u64 *>(tcb - 0x10) : 0;
      i64 ord = (v & 0x8000) ? (i64)(v & ~0x8000ULL) : -1;
      static std::mutex m; std::lock_guard lk(m);
      BASE_LOGI("jobclaim",
                "worker fsbase={:#x} tcb={:#x} [tcb-0x10]={:#x} -> ordinal={} "
                "(1<<ord={:#x}) firstClaim->{:#x}",
                (unsigned long long)fsb, (unsigned long long)tcb,
                (unsigned long long)v, (long long)ord,
                (ord >= 0 && ord < 32) ? (1u << (unsigned)ord) : 0u,
                (unsigned long long)ret);
    }
    ++g_jobClaims;
    if (ret == 0) ++g_jobFails;
    if (a0 > 0x10000)
      spawnJobWatcher(a0);
    break;
  }
  case 13: { // "Material Param Update" fills a block ON THE GPU (0x16eb90).
    // SotC's material parameter blocks -- the constants AND the texture
    // descriptors a draw reads through its SRT -- are not written by the CPU.
    // `Shadow_Shipping+0x117930` (its timing print is "Material Param Update")
    // allocates a block, memcpys a template into it, and calls this to build
    // two buffer descriptors over it and dispatch (n+63)/64 threadgroups.
    // Nothing on the CPU ever touches the blocks the failing draws read, so
    // the question is which blocks this path actually targets: a3 = rcx = the
    // block, a2 = rdx = its element count. One line per DISTINCT block, so a
    // block updated every frame costs one line, not thousands.
    static std::mutex m;
    static std::unordered_set<u64> seen;
    static u64 calls = 0;
    std::lock_guard lk(m);
    calls++;
    if (seen.insert(a3).second && seen.size() <= 256)
      BASE_LOGI("mattrace",
                "block={:#x} elems={} (distinct={} of {} updates)",
                (unsigned long long)a3, (unsigned long long)a2, seen.size(),
                (unsigned long long)calls);
    else if ((calls % 2000) == 0)
      BASE_LOGI("mattrace", "{} updates over {} distinct blocks",
                (unsigned long long)calls, seen.size());
    break;
  }
  case 14: { // the DISPATCH_DIRECT emitter (0x883870).
    // The material fills are issued by the guest and never reach our PM4
    // parser (DELTA_GPU_CSDROPS reports zero drops), so the question is which
    // command buffer they are written into. a0 = rdi = the buffer object:
    // [a0+0] .. [a0+8] is its extent and [a0+0x10] its write pointer, which is
    // where this 9-dword packet lands. a1 = the X threadgroup count, so
    // 16384 selects exactly the whole-arena (1048576-element) fills.
    if (a1 != 16384)
      break;
    const auto *cb = reinterpret_cast<const u64 *>(a0);
    const u64 wp = cb[2], end = cb[1];
    static std::mutex m;
    static std::unordered_set<u64> seen;
    static u64 n = 0;
    std::lock_guard lk(m);
    n++;
    if (seen.insert(wp >> 20).second && seen.size() <= 64)
      BASE_LOGI("matcb",
                "fill dispatch -> cmdbuf write={:#x} end={:#x} "
                "(obj={:#x}, {} regions of {} fills)",
                (unsigned long long)wp, (unsigned long long)end,
                (unsigned long long)a0, seen.size(), (unsigned long long)n);
    break;
  }
  case 11: { // CTOR 0x36210 -> a0 = rdi = the JobSystem object being built.
    // The zero-cost watcher bootstrap: this runs ONCE at init, returns
    // normally, and takes its argument in a register. Verified to be the same
    // base the claim path indexes: the ctor zeroes per-ordinal blocks at
    // +0xeb0/+0xfd0/... (stride 0x120), exactly what claim tests at
    // [jobsys + ordinal*0x120 + 0xeb0].
    //
    // Do NOT hook JobKick(0x35480) here: it reads its 7th argument from the
    // CALLER's frame (`mov r14d,[rbp+0x10]`), and an entry detour relocates
    // the `mov rbp,rsp` prologue so rbp lands in the wrapper's frame -- every
    // job then gets a garbage affinity and the title wedges before its main
    // loop. Entry detours are only safe on functions whose arguments never
    // come off the caller's stack.
    if (a0 > 0x10000) {
      BASE_LOGI("jobctor", "JobSystem ctor this={:#x}",
                (unsigned long long)a0);
      spawnJobWatcher(a0);
    }
    break;
  }
  default: break;
  }
}

// Watcher body: every 30s dump the 8 per-ordinal direct-assign blocks
// (jobsys + i*0x120 + 0xeb0) plus the job-table survey. The round-10 livelock
// signature is a persistent marker in a block whose ordinal (4..7) no
// claim-worker (0..3) services while blocks 0..3 sit empty.
// DELTA_SOTC_JOBMOVE (flawed, kept for experiments): when the signature holds
// across two consecutive dumps, move the marker qword into block 2 -- note the
// claim path copies a 0x40-byte descriptor at +0xea8, so a single-qword move
// hands worker 2 a zeroed job; superseded by the job-table affinity survey.
void spawnJobWatcher(u64 base) {
  static std::atomic<u64> once{0};
  u64 expect = 0;
  if (!once.compare_exchange_strong(expect, base))
    return;
  std::thread([base] {
        u64 prev[8] = {0};
        int persist = 0;
        for (;;) {
          std::this_thread::sleep_for(std::chrono::seconds(30));
          u64 slot[8];
          for (int i = 0; i < 8; i++)
            slot[i] = *reinterpret_cast<volatile u64 *>(base + (u64)i * 0x120 + 0xeb0);
          BASE_LOGI("jobwatch",
                    "claims={} fails={} directAssign[0..7]= "
                    "0:{:#x} 1:{:#x} 2:{:#x} 3:{:#x} 4:{:#x} 5:{:#x} 6:{:#x} 7:{:#x}",
                    (unsigned long long)g_jobClaims.load(),
                    (unsigned long long)g_jobFails.load(),
                    (unsigned long long)slot[0], (unsigned long long)slot[1],
                    (unsigned long long)slot[2], (unsigned long long)slot[3],
                    (unsigned long long)slot[4], (unsigned long long)slot[5],
                    (unsigned long long)slot[6], (unsigned long long)slot[7]);
          // Context dump of any occupied block: the direct-assign descriptor
          // is the 0x40 bytes at block+0xea8 (claim copies +0xea8/+0xeb8 ymm
          // pair; +0xeb0 is the occupancy discriminator).
          for (int i = 0; i < 8; i++) {
            if (!slot[i])
              continue;
            const u64 *q = reinterpret_cast<const u64 *>(base + (u64)i * 0x120 + 0xea8);
            BASE_LOGI("jobwatch",
                      "  block{} desc@+0xea8: {:#x} {:#x} {:#x} {:#x} "
                      "{:#x} {:#x} {:#x} {:#x}",
                      i, (unsigned long long)q[0], (unsigned long long)q[1],
                      (unsigned long long)q[2], (unsigned long long)q[3],
                      (unsigned long long)q[4], (unsigned long long)q[5],
                      (unsigned long long)q[6], (unsigned long long)q[7]);
          }
          // The decisive post-drain read: live entries in the 1024-slot job
          // table ([jobsys+0xb60], stride 0x9b8, alloc bitmap at +0xac0).
          // JobKick writes job+0x998 = affinity (arg7, default [jobsys+0x3c])
          // and job+0x9a0 = prio; the claim path tests 1<<worker-ordinal
          // against the affinity. A stuck finalize job shows up here with a
          // mask the 4 workers (0xF) cannot satisfy.
          // Is there pending work at all, and can our workers claim it? The
          // claim (0x38d40) scans 8 priority buckets at
          // `[jobsys+0x1478] + lvl*0x2078` and has TWO consumer paths. An
          // earlier version of this walk guessed the layout (per-ordinal words
          // at bucket+8 / bucket+0x2040) and read zeros out of both, which is
          // why "no queue node has pending work" was never trustworthy; both
          // paths below are taken straight from the disassembly instead:
          //
          //   scan A (0x38fa7, taken when [bucket+0x2038] != 0): a slot array
          //     at [bucket+0x2070], stride 0x120, affinity mask at slot+0xf0 --
          //     `test [slots + i*0x120 + 0xf0], 1<<ord` at 0x38fc5.
          //   scan B (0x39130, taken when [bucket+0x2038] == 0 && [bucket] != 0):
          //     a NODE POINTER array at bucket+0x38 (8 bytes each, count in
          //     [bucket+0]); a node is pending when head [n] != tail [n+8], then
          //     the dependency triple +0x980/+0x988/+0x990 must be satisfied,
          //     and finally `test [n+0x998], 1<<ord` (0x3919e) must pass.
          //
          // Empty everywhere while the coordinator waits = the job was never
          // ENQUEUED (producer-side bug); pending but unclaimable = the workers
          // cannot take what is there (consumer-side). Our workers report
          // ordinals 0..3, so a mask missing 0xf excludes every one of them.
          {
            const u64 qbase =
                *reinterpret_cast<volatile u64 *>(base + 0x1478);
            auto mapped = [](u64 a) {
              return a > 0x10000 && a < (1ULL << 47);
            };
            if (mapped(qbase)) {
              for (int lvl = 0; lvl < 8; lvl++) {
                const u64 bucket = qbase + (u64)lvl * 0x2078;
                const u64 acount =
                    *reinterpret_cast<volatile u64 *>(bucket + 0x2038);
                const u64 nodes =
                    *reinterpret_cast<volatile u64 *>(bucket);
                if (!acount && !nodes)
                  continue;
                BASE_LOGI(
                    "jobwatch",
                    "bucket L{} scanA-slots={} scanB-nodes={}",
                    lvl, (unsigned long long)acount, (unsigned long long)nodes);
                const u64 slots =
                    *reinterpret_cast<volatile u64 *>(bucket + 0x2070);
                if (acount && mapped(slots)) {
                  for (u64 i = 0; i < acount && i < 16; i++) {
                    const u32 mask = *reinterpret_cast<volatile u32 *>(
                        slots + i * 0x120 + 0xf0);
                    BASE_LOGI("jobwatch",
                              "  L{} slot{} mask={:#x} -> {}",
                              lvl, (unsigned long long)i, mask,
                              (mask & 0xf)
                                  ? "claimable"
                                  : "*** MASK EXCLUDES EVERY WORKER ***");
                  }
                }
                for (u64 i = 0; i < nodes && i < 32; i++) {
                  const u64 n = *reinterpret_cast<volatile u64 *>(
                      bucket + 0x38 + i * 8);
                  if (!mapped(n))
                    continue;
                  const u64 head = *reinterpret_cast<volatile u64 *>(n);
                  const u64 tail =
                      *reinterpret_cast<volatile u64 *>(n + 8);
                  if (head == tail)
                    continue;  // empty: the claim skips it at 0x3913e
                  const u64 dep =
                      *reinterpret_cast<volatile u64 *>(n + 0x980);
                  const u64 thresh =
                      *reinterpret_cast<volatile u64 *>(n + 0x988);
                  const u32 mode =
                      *reinterpret_cast<volatile u32 *>(n + 0x990);
                  const u32 aff =
                      *reinterpret_cast<volatile u32 *>(n + 0x998);
                  const i32 prio =
                      *reinterpret_cast<volatile i32 *>(n + 0x9a0);
                  long long depval = 0;
                  bool depread = false;
                  if (mapped(dep)) {
                    depval =
                        (long long)*reinterpret_cast<volatile u64 *>(dep);
                    depread = true;
                  }
                  const bool depok =
                      !dep || (depread && (mode == 0
                                               ? (u64)depval == thresh
                                               : depval > (long long)thresh));
                  BASE_LOGI(
                      "jobwatch",
                      "  L{} node{} @{:#x} head={:#x} tail={:#x} "
                      "dep={:#x} *dep={} thresh={} mode={} aff={:#x} prio={} "
                      "-> {}{}",
                      lvl, (unsigned long long)i, (unsigned long long)n,
                      (unsigned long long)head, (unsigned long long)tail,
                      (unsigned long long)dep, depval, (long long)thresh, mode,
                      aff, prio,
                      depok ? "dep-ok" : "*** BLOCKED ON DEPENDENCY ***",
                      (aff & 0xf) ? "" : " *** AFF EXCLUDES EVERY WORKER ***");
                }
              }
            }
          }
          // Which worker ordinals exist process-wide. The claim path tests
          // `job_affinity & (1 << ordinal)` with ordinal = [*(fsbase)-0x10]
          // (0x8000|core, bit 15 = valid; fn 0x33350). A job whose affinity
          // names only a core no live thread reports -- e.g. SotC's core-6
          // "Resource Loading" pin, mask 0x40 -- is unclaimable by anyone.
          {
            std::vector<u64> fsb;
            cpu::guestThreadFsBases(fsb);
            u32 present = 0;
            int valid = 0;
            std::string list;
            for (u64 f : fsb) {
              if (!f)
                continue;
              u64 tcb = *reinterpret_cast<volatile u64 *>(f);
              if (tcb < 0x10000)
                continue;
              u64 v = *reinterpret_cast<volatile u64 *>(tcb - 0x10);
              if (!(v & 0x8000))
                continue;
              int ord = (int)(v & 0x7fff);
              if (ord >= 0 && ord < 32) {
                present |= 1u << ord;
                valid++;
                char b[16];
                std::snprintf(b, sizeof b, "%d ", ord);
                list += b;
              }
            }
            BASE_LOGI("jobwatch",
                      "threads={} withOrdinal={} ordinals={ {} } claimable-mask={:#x}",
                      fsb.size(), valid, list.c_str(), present);
          }
          // The real claim gate is a DEPENDENCY, not affinity. Claim
          // (0x39130..) walks a queue node, and after finding head != tail it
          // tests: dep = [node+0x980]; if dep, compare *dep against
          // threshold [node+0x988] under mode [node+0x990] (0 = equal,
          // 1 = counter > threshold) and SKIP the node when it does not hold.
          // That is precisely "work is visible but nobody can claim it": a job
          // whose dependency counter never reaches its target stays queued
          // forever while the workers spin. Dump every node with pending work
          // so a stuck dependency shows its counter, target and mode.
          {
            u64 tbl = *reinterpret_cast<volatile u64 *>(base + 0xb60);
            int shown = 0;
            if (tbl > 0x10000) {
              for (int j = 0; j < 1024 && shown < 10; j++) {
                const u64 node = tbl + (u64)j * 0x9b8;
                const u64 head = *reinterpret_cast<volatile u64 *>(node);
                const u64 tail = *reinterpret_cast<volatile u64 *>(node + 8);
                if (!head || head == tail)
                  continue;  // empty queue: nothing pending here
                const u64 dep = *reinterpret_cast<volatile u64 *>(node + 0x980);
                const u64 thresh = *reinterpret_cast<volatile u64 *>(node + 0x988);
                const u32 mode = *reinterpret_cast<volatile u32 *>(node + 0x990);
                const u32 aff = *reinterpret_cast<volatile u32 *>(node + 0x998);
                long long depval = -1;
                bool readable = false;
                if (dep > 0x10000 && dep < (1ULL << 47)) {
                  const u64 inner = *reinterpret_cast<volatile u64 *>(dep);
                  depval = (long long)inner;
                  readable = true;
                }
                const bool satisfied =
                    !dep || (readable && (mode == 0 ? (u64)depval == thresh
                                                    : (long long)depval > (long long)thresh));
                // NOTE: this walk covers the 1024-slot job TABLE, which is not
                // what the claim scans -- so `aff` (+0x998) is the only gate
                // reading meaningfully here. The old `obj = [node+0x38];
                // mask = [obj+0xf0]` chase was wrong twice over: +0x38 is the
                // node-pointer ARRAY inside a priority bucket (not a per-job
                // object pointer), and the +0xf0 mask belongs to the scan-A
                // slot array at [bucket+0x2070]. Both live in the bucket walk
                // above; a table entry's +0x38 reads 0, which is exactly why
                // that chase always reported nothing.
                const u64 obj = 0;
                u32 mask = 0;
                bool mask_read = false;
                BASE_LOGI("jobwatch",
                          "node[{}] head={:#x} tail={:#x} dep={:#x} *dep={} "
                          "thresh={} mode={} aff={:#x} obj={:#x} mask={}{:#x} -> "
                          "{}{}",
                          j, (unsigned long long)head, (unsigned long long)tail,
                          (unsigned long long)dep, depval, (long long)thresh, mode,
                          aff, (unsigned long long)obj, mask_read ? "" : "?", mask,
                          satisfied ? "dep-ok" : "*** BLOCKED ON DEPENDENCY ***",
                          (mask_read && !(mask & 0xf))
                              ? " *** MASK EXCLUDES EVERY WORKER ***"
                              : "");
                shown++;
              }
              if (!shown)
                BASE_LOGI("jobwatch", "no queue node has pending work");
            }
          }
          static int tableDumps = 0;
          u64 defAff = *reinterpret_cast<volatile u32 *>(base + 0x3c);
          u64 tab = *reinterpret_cast<volatile u64 *>(base + 0xb60);
          const u64 *bm = reinterpret_cast<const u64 *>(base + 0xac0);
          u64 used = 0;
          for (int w = 0; w < 16; w++) used += __builtin_popcountll(~bm[w]);
          if (tableDumps < 200) {
            BASE_LOGI("jobwatch",
                      "jobsys={:#x} defaultAffinity[+0x3c]={:#x} table={:#x} "
                      "bitmapFreeInv={} bm0={:#x}",
                      (unsigned long long)base, (unsigned long long)defAff,
                      (unsigned long long)tab, (unsigned long long)used,
                      (unsigned long long)bm[0]);
            if (tab > 0x10000) {
              int shown = 0;
              for (int j = 0; j < 1024 && shown < 12; j++) {
                // Print any table entry with a plausible live affinity word;
                // polarity of the bitmap is unknown, so filter on content.
                const u8 *job = reinterpret_cast<const u8 *>(tab + (u64)j * 0x9b8);
                u32 aff = *reinterpret_cast<const u32 *>(job + 0x998);
                u32 prio = *reinterpret_cast<const u32 *>(job + 0x9a0);
                // The mask the claim path actually tests (0x38fc5:
                // `test [node+0xf0], 1<<ordinal`); kick's 0x34e60 helper
                // copies the submitted spec here.
                u32 mask = *reinterpret_cast<const u32 *>(job + 0xf0);
                u64 q0 = *reinterpret_cast<const u64 *>(job);
                if (!aff && !q0)
                  continue;
                BASE_LOGI("jobwatch",
                          "  job[{}] q0={:#x} claimMask[+0xf0]={:#x} "
                          "aff[+0x998]={:#x} prio={:#x} q+0x9a8={:#x}",
                          j, (unsigned long long)q0, mask, aff, prio,
                          (unsigned long long)*reinterpret_cast<const u64 *>(job + 0x9a8));
                shown++;
              }
              tableDumps++;
            }
          }
          const bool stranded =
              (slot[4] | slot[5] | slot[6] | slot[7]) != 0 &&
              (slot[0] | slot[1] | slot[2] | slot[3]) == 0 &&
              slot[4] == prev[4] && slot[5] == prev[5] && slot[6] == prev[6] &&
              slot[7] == prev[7];
          persist = stranded ? persist + 1 : 0;
          std::memcpy(prev, slot, sizeof(prev));
          if (kSotcJobmove && persist >= 2) {
            for (int i = 4; i < 8; i++) {
              if (!slot[i])
                continue;
              auto *src = reinterpret_cast<volatile u64 *>(base + (u64)i * 0x120 + 0xeb0);
              auto *dst = reinterpret_cast<volatile u64 *>(base + 2ULL * 0x120 + 0xeb0);
              u64 v = *src;
              *dst = v;
              *src = 0;
              BASE_LOGI("jobwatch",
                        "JOBMOVE: moved direct-assign marker {:#x} block{} -> block2",
                        (unsigned long long)v, i);
            }
            persist = 0;
          }
        }
  }).detach();
}

// DELTA_SOTC_ALLOCLOCK: hold ONE host mutex across every call into the title's
// allocator, and count how often a thread had to wait for it.
//
// This is the deterministic form of the question DELTA_RIPRACE could not afford
// to answer. The free tree ends up holding stale child links; only two
// explanations survived the census (which showed the corrupted words are written
// exclusively by the allocator's own nine sites): two guest threads inside at
// once, or a miscompiled store. Every call is observed here, not one in 40000 --
// so `contended` is a decisive count, whatever happens to the crash:
//   contended == 0  -> no two threads were EVER inside together. Concurrency is
//                      eliminated and the remaining suspect is our codegen.
//   contended >  0  -> they were, and the title's own exclusion is not doing what
//                      it does on hardware.
// The mutex is recursive because these entries nest (the allocator frees while
// allocating, and the tracker reenters), and because a fiber switch on the same
// host thread must not deadlock against itself.
// One lock PER SITE, because a shared lock cannot tell the two cases apart.
// Several threads inside malloc at once is normal for a shared heap and proves
// nothing. Several threads inside the FREE-TREE INSERT at once is a violation:
// that function mutates the tree whose child links come out stale. So each hook
// gets its own mutex and its own contention count, and the insert's count is the
// answer.
struct AllocLockSite {
  std::recursive_mutex m;
  std::atomic<u64> calls{0};
  std::atomic<u64> contended{0};
  std::atomic<u64> maxWaitNs{0};
  const char *name = "";
  bool serialise = false;  // only the tree mutators need the shared lock
};
constexpr int kAllocLockSites = 7;
AllocLockSite g_allocSites[kAllocLockSites];

// DELTA_SOTC_ALLOCLOCK=2 gives every site its own mutex, which is what MEASURES
// the violation: a failed try_lock at the insert names a second thread inside the
// insert itself. DELTA_SOTC_ALLOCLOCK=1 makes them share one mutex, which is what
// MITIGATES it -- per-site locks cannot, because a thread inside 0x12820 and a
// thread inside 0x12af0 hold different locks and still meet over the same tree.
// (Learned the hard way: the first mitigation run still crashed, with the per-site
// counters proving the exclusion it needed was never in place.)
std::recursive_mutex g_allocSharedM;

// ONE LOCK PER ALLOCATOR STATE, not one lock overall. The contention forensics
// showed 22 of 24 collisions were between DIFFERENT allocator instances -- this
// title runs three (two static in module .data, one in direct memory) and they
// share no tree, so serialising them against each other is pure false sharing. It
// also made the experiment unrunnable: locking the hot coalescer across all three
// heaps ran ~20x slower than locking four sites across one, and the repro never
// reached the crash window.
//
// Keyed on arg0, which is the state pointer at the three sites that take one
// (insert, remove, coalesce all use arg0+0x70). Rebalance and fixup are called
// from inside those and take node pointers rather than a state, so they are
// observed and not locked -- their callers already hold the right lock, which is
// why their contention counts were 0 all along.
//
// The payoff is that contention now MEANS something: with per-state locking, a
// failed try_lock is two threads inside the tree mutators of the SAME tree.
constexpr int kAllocStates = 16;
struct StateLock {
  std::atomic<u64> state{0};
  std::recursive_mutex m;
};
StateLock g_stateLocks[kAllocStates];
std::atomic<u64> g_stateLockOverflow{0};

static std::recursive_mutex &allocMutexFor(int i, u64 a0) {
  if (kSotcAllocLock == 2)
    return g_allocSites[i].m;
  if (!a0)
    return g_allocSharedM;
  for (int k = 0; k < kAllocStates; k++) {
    u64 cur = g_stateLocks[k].state.load(std::memory_order_acquire);
    if (cur == a0)
      return g_stateLocks[k].m;
    if (cur == 0) {
      u64 expect = 0;
      if (g_stateLocks[k].state.compare_exchange_strong(expect, a0))
        return g_stateLocks[k].m;
      if (g_stateLocks[k].state.load(std::memory_order_acquire) == a0)
        return g_stateLocks[k].m;
    }
  }
  g_stateLockOverflow.fetch_add(1, std::memory_order_relaxed);
  return g_allocSharedM;  // more states than slots: fall back to one lock
}

// Ring of the most recent allocator calls, so that when the periodic walk trips
// there is a story for the window it brackets instead of only a verdict: which
// sites ran, with what argument, on which thread. Read back on the first trip.
// Who is inside the shared lock right now, so a thread that contends can say what
// it collided with. Set on acquire, cleared on release.
// DELTA_SOTC_HEAPROUTE: do the title's three heaps share address space?
//
// The corruption on the crashing heap needs no concurrency if a chunk is freed into
// one state's tree while its backing memory belongs to another state's arena that
// later gets recycled wholesale -- the tree would keep a link to memory handed out
// again elsewhere, which is exactly the observed shape (a live record array holding
// memory the tree still thinks is free), and it would explain why no same-tree
// collision has ever been seen on that heap.
//
// The insert takes the state in rdi and inserts the chunk cached at state+0xc0 (its
// designated victim), so both sides are readable at the hook with no pointer map:
// bucket the chunk address by 256 MiB per state and print the table. Disjoint
// buckets per state means chunks never cross heaps and the idea is dead; a bucket
// shared by two states is where to look next.
struct RouteBucket { u64 prefix; u64 count; };
struct RouteTab {
  std::atomic<u64> state{0};
  u64 inserts = 0;
  u64 lo = ~0ull, hi = 0;
  RouteBucket b[12] {};
};
constexpr int kRouteTabs = 8;
RouteTab g_routeTabs[kRouteTabs];
std::mutex g_routeM;

static void heapRouteNote(u64 state, u64 chunk) {
  if (!state || !chunk || chunk < 0x8000000000ull || chunk >= 0x8700000000ull)
    return;
  std::lock_guard<std::mutex> lk(g_routeM);
  RouteTab *t = nullptr;
  for (int i = 0; i < kRouteTabs; i++) {
    const u64 cur = g_routeTabs[i].state.load(std::memory_order_relaxed);
    if (cur == state) { t = &g_routeTabs[i]; break; }
    if (cur == 0) { g_routeTabs[i].state.store(state); t = &g_routeTabs[i]; break; }
  }
  if (!t)
    return;
  t->inserts++;
  if (chunk < t->lo) t->lo = chunk;
  if (chunk > t->hi) t->hi = chunk;
  const u64 pref = chunk >> 28;
  for (auto &e : t->b) {
    if (e.count && e.prefix == pref) { e.count++; return; }
    if (!e.count) { e.prefix = pref; e.count = 1; return; }
  }
}

static void heapRouteReport() {
  std::lock_guard<std::mutex> lk(g_routeM);
  for (auto &t : g_routeTabs) {
    const u64 st = t.state.load(std::memory_order_relaxed);
    if (!st) continue;
    base::String bytes;
    base::FormatTo(bytes, "state {:#x}: {} inserts, chunks {:#x}..{:#x}, 256MB buckets:",
                  (unsigned long long)st, (unsigned long long)t.inserts,
                  (unsigned long long)t.lo, (unsigned long long)t.hi);
    for (auto &e : t.b)
      if (e.count)
        base::FormatTo(bytes, " {:#x}0000000(x{})",
                       (unsigned long long)e.prefix, (unsigned long long)e.count);
    BASE_LOGI("heaproute", "{}", bytes.c_str());
  }
}

struct LockHolder {
  std::atomic<u32> gtid{0};
  std::atomic<u32> site{0};
  std::atomic<u64> a0{0};
};
LockHolder g_lockHolder;
std::atomic<int> g_contendReported{0};

// The guest pthread mutex the allocator's shared heap is locked with -- the one the
// crashing thread spins on with MUTEX_WAIT/MUTEX_WAKE. FreeBSD's umutex keeps the
// owner tid in the low bits of its first word with UMUTEX_CONTESTED (0x80000000)
// on top, which is how sys_umtx_op reads it.
static u32 umtxOwnerWord() {
  const u64 a = kSotcUmtxAddr;
  if (a < 0x200000000ull || a >= 0x201000000ull)
    return 0xFFFFFFFFu;
  return *reinterpret_cast<const volatile u32 *>(a);
}

// The discriminating report. Two threads inside the tree mutators at once is only a
// race on the SAME tree if they are working on the same allocator state -- this
// title hands out per-scope allocators, so a0 (the state pointer for insert and
// remove) has to match before the collision means anything. And if it does match,
// the owner word says whether the guest's own mutex thought it was excluding them.
static void reportContention(int site, u64 a0) {
  if (g_contendReported.fetch_add(1) >= 24)
    return;
  const u32 myGtid = *currentGuestTidPtr();
  const u32 hg = g_lockHolder.gtid.load();
  const u32 hs = g_lockHolder.site.load();
  const u64 ha = g_lockHolder.a0.load();
  const u32 ow = umtxOwnerWord();
  const u32 owner = ow & ~0x80000000u;
  const char *verdict =
      (ha && a0 && ha != a0)
          ? "DIFFERENT allocator states -- not one tree, not a race"
          : (owner == myGtid || owner == hg)
                ? "SAME state, and the guest mutex names one of them as owner -- "
                  "the other entered without it"
                : (owner == 0)
                      ? "SAME state, guest mutex UNOWNED -- neither holds it"
                      : "SAME state, guest mutex owned by a THIRD thread";
  BASE_LOGI("contend",
            "me: gtid={} site={} a0={:#x} | holder: gtid={} site={} "
            "a0={:#x} | umtx@{:#x} word={:#x} owner={} contested={} -> {}",
            myGtid, g_allocSites[site].name, (unsigned long long)a0, hg,
            hs < kAllocLockSites ? g_allocSites[hs].name : "?",
            (unsigned long long)ha, (unsigned long long)kSotcUmtxAddr, ow, owner,
            (ow & 0x80000000u) ? 1 : 0, verdict);
}

struct AllocEvt { u32 site; u32 tid; u64 a0, a1; };
// Big enough to cover the walk cadence: the walk only looks every N calls, so a
// 64-entry ring showed nothing but the allocs immediately before the check and
// none of the mutators in the window that actually did the damage.
constexpr int kAllocRing = 8192;
AllocEvt g_allocRing[kAllocRing];
std::atomic<u64> g_allocRingPos{0};
std::atomic<u64> g_allocCallSeq{0};

// Which allocator state this thread locked, per nesting level: the leave hook has
// no argument to re-derive it from.
static thread_local u64 t_lockedState[8];
static thread_local int t_lockDepth = 0;

static void treeWatchAt(int site, bool onExit);
static void treeWalkPeriodic();

static void allocLockEnterAt(int i, u64 a0, u64 a1) {
  AllocLockSite &s = g_allocSites[i];
  s.calls.fetch_add(1, std::memory_order_relaxed);
  g_allocCallSeq.fetch_add(1, std::memory_order_relaxed);
  if (kSotcTreeWalk) {
    const u64 k = g_allocRingPos.fetch_add(1, std::memory_order_relaxed);
    AllocEvt &e = g_allocRing[k % kAllocRing];
    e.site = (u32)i;
    e.tid = (u32)syscall(SYS_gettid);
    e.a0 = a0;
    e.a1 = a1;
  }
  if (kSotcHeapRoute && i == 2 && a0 >= 0x200000000ull) {
    // site 2 is treeInsert: rdi is the state, and the chunk it will insert is the
    // designated victim cached at state+0xc0.
    const u64 chunk = *reinterpret_cast<const volatile u64 *>(a0 + 0xc0);
    heapRouteNote(a0, chunk);
  }
  if (kSotcTreeWatch)
    treeWatchAt(i, false);
  if (!kSotcAllocLock || !s.serialise)
    return;  // observation-only site, or watch-only run: do not serialise
  std::recursive_mutex &mx = allocMutexFor(i, a0);
  if (mx.try_lock()) {
    if (t_lockDepth < 8) t_lockedState[t_lockDepth++] = a0;
    g_lockHolder.gtid.store(*currentGuestTidPtr());
    g_lockHolder.site.store((u32)i);
    g_lockHolder.a0.store(a0);
    return;
  }
  // The try_lock failed, so another thread is inside this very function right
  // now. That is the measurement; the blocking acquire below is the mitigation.
  s.contended.fetch_add(1, std::memory_order_relaxed);
  reportContention(i, a0);
  const auto t0 = std::chrono::steady_clock::now();
  mx.lock();
  if (t_lockDepth < 8) t_lockedState[t_lockDepth++] = a0;
  g_lockHolder.gtid.store(*currentGuestTidPtr());
  g_lockHolder.site.store((u32)i);
  g_lockHolder.a0.store(a0);
  const u64 ns = (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - t0).count();
  u64 prev = s.maxWaitNs.load(std::memory_order_relaxed);
  while (ns > prev &&
         !s.maxWaitNs.compare_exchange_weak(prev, ns, std::memory_order_relaxed)) {}
}

// DELTA_SOTC_TREEWATCH: catch the free tree going bad AT ITS BIRTH.
//
// Five crashes all corrupt the same field -- node 0x8052e00020's child[0] -- so
// there is no need to walk the tree: check that one word, on both sides of every
// allocator call. Entry clean and exit dirty names the call that CONTAINED the
// corrupting store, which is what a "miscompiled store" claim needs and what the
// crash-time walk can never give (by then the store is millions of calls old).
// Two loads and a few compares per call, against ~6M calls a run.
//
// A healthy child[0] is either the tree's sentinel or a pointer to a free chunk,
// whose size word sits at child-8 and is small, non-zero and 16-byte aligned --
// the same plausibility test the crash-time walker uses. Every bad value observed
// so far is a mapped pointer into reused live data, so reading it is safe; a value
// outside guest direct memory is reported without being dereferenced.
static thread_local bool t_treeOkAtEntry = true;
std::atomic<u64> g_treeChecks{0};
std::atomic<u64> g_treeBadAtEntry{0};
std::atomic<u64> g_treeWentBad{0};
std::atomic<int> g_treeReported{0};

// The watched node does not exist at boot -- the heap has not grown into it yet --
// so the check has to stay disarmed until its page is mapped, or the very first
// allocator call dereferences nothing and takes the process down. (It did.) Poll
// for the page the way the write census does, then confirm the address really
// looks like a free-tree node before trusting anything it says.
std::atomic<bool> g_treeArmed{false};

static void treeWatchArm() {
  std::thread([] {
    const long pg = sysconf(_SC_PAGESIZE);
    void *page = reinterpret_cast<void *>(kSotcTreeNode & ~(u64)(pg - 1));
    // The walk reads the allocator STATE, the field watch reads the node: both
    // pages have to exist before either is allowed to dereference anything.
    void *spage = reinterpret_cast<void *>(kSotcTreeState & ~(u64)(pg - 1));
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      unsigned char vec = 0;
      if (mincore(page, 1, &vec) != 0 || mincore(spage, 1, &vec) != 0)
        continue;
      const u64 own = *reinterpret_cast<const u64 *>(kSotcTreeNode - 8) & ~7ull;
      BASE_LOGI("treewatch",
                "armed on node {:#x} (its own size word reads "
                "{:#x}) state {:#x} sentinel {:#x}",
                (unsigned long long)kSotcTreeNode, (unsigned long long)own,
                (unsigned long long)kSotcTreeState,
                (unsigned long long)(kSotcTreeState + 0x80));
      g_treeArmed.store(true, std::memory_order_release);
      return;
    }
  }).detach();
}

static bool treeFieldOk(u64 &valOut, u64 &szOut) {
  const u64 node = kSotcTreeNode;
  const u64 sentinel = kSotcTreeState + 0x80;
  valOut = szOut = 0;
  if (node < 0x8000000000ull || node >= 0x8700000000ull)
    return true;  // not the layout this watch was aimed at; stay quiet
  // Only judge the field while the node is ITSELF a live free chunk. Before the
  // tree grows into this address its words are whatever the previous owner left,
  // and calling that "bad" buries the real signal: the first run of this watch
  // reported 124 bad-on-entry hits with 0 transitions, which is what pre-
  // membership noise looks like.
  const u64 own = *reinterpret_cast<const u64 *>(node - 8) & ~7ull;
  if (!own || own >= 0x8000000ull)
    return true;  // 8-granular sizes: masking with ~7 is the only test available
  const u64 c = *reinterpret_cast<const u64 *>(node);
  valOut = c;
  if (c == sentinel)
    return true;
  if (c < 0x8000000000ull || c >= 0x8700000000ull)
    return false;  // not a guest pointer at all -- do not dereference it
  const u64 sz = *reinterpret_cast<const u64 *>(c - 8) & ~7ull;
  szOut = sz;
  return sz && sz < 0x8000000ull;
}

static void treeWatchAt(int site, bool onExit) {
  if (!g_treeArmed.load(std::memory_order_acquire))
    return;
  u64 val = 0, sz = 0;
  const bool ok = treeFieldOk(val, sz);
  g_treeChecks.fetch_add(1, std::memory_order_relaxed);
  if (!onExit) {
    t_treeOkAtEntry = ok;
    if (!ok)
      g_treeBadAtEntry.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  if (ok || !t_treeOkAtEntry)
    return;  // already bad on the way in: some earlier call did it
  g_treeWentBad.fetch_add(1, std::memory_order_relaxed);
  if (g_treeReported.fetch_add(1) < 8) {
    BASE_LOGI("treewatch",
              "node {:#x} child[0] WENT BAD inside {} (tid {}): "
              "now {:#x}, its size word reads {:#x} -- this call contained the "
              "corrupting store",
              (unsigned long long)kSotcTreeNode, g_allocSites[site].name,
              (long)syscall(SYS_gettid), (unsigned long long)val,
              (unsigned long long)sz);
  }
}

// DELTA_SOTC_TREEWALK=<N>: every N allocator calls, walk the WHOLE free tree and
// report the first link pointing at something that is no longer a free chunk.
// This replaces the single-field watch, which was aimed at node 0x8052e00020 and
// went silent the run the victim moved elsewhere. Bracketing to N calls turns a
// 25-minute repro into "the corruption appeared within these N calls", and the
// ring buffer above says what ran in that window.
//
// Depth-first over both children with an explicit stack; a node is judged by its
// size word at node-8 (8-granular, non-zero, not absurd) exactly as the crash-time
// walker does. Pointers are range-checked against guest direct memory before any
// dereference, so a corrupt link cannot fault the walker.
std::atomic<bool> g_treeWalkTripped{false};

static inline bool inDmem(u64 p) {
  return p >= 0x8000000000ull && p < 0x8700000000ull;
}

static void treeWalkPeriodic() {
  const u64 n = kSotcTreeWalk;
  if (!n || g_treeWalkTripped.load(std::memory_order_relaxed))
    return;
  if ((g_allocCallSeq.load(std::memory_order_relaxed) % n) != 0)
    return;
  if (!g_treeArmed.load(std::memory_order_acquire))
    return;
  const u64 state = kSotcTreeState;
  const u64 sentinel = state + 0x80;
  if (!inDmem(sentinel))
    return;
  u64 stack[256], fields[256];
  int sp = 0, visited = 0;
  stack[sp] = *reinterpret_cast<const u64 *>(sentinel);
  fields[sp] = sentinel;
  sp++;
  while (sp > 0) {
    --sp;
    const u64 cur = stack[sp], field = fields[sp];
    if (cur == sentinel || cur == 0)
      continue;
    if (++visited > 4096)
      return;  // pathological; say nothing rather than guess
    bool bad = !inDmem(cur);
    u64 sz = 0;
    if (!bad) {
      sz = *reinterpret_cast<const u64 *>(cur - 8) & ~7ull;
      bad = !sz || sz >= 0x8000000ull;
    }
    if (bad) {
      if (g_treeWalkTripped.exchange(true))
        return;
      BASE_LOGI("treewalk",
                "\nTRIPPED after {} allocator calls, {} nodes "
                "visited: the link at {:#x} points at {:#x}, whose size word "
                "reads {:#x} -- no longer a free chunk",
                (unsigned long long)g_allocCallSeq.load(), visited,
                (unsigned long long)field, (unsigned long long)cur,
                (unsigned long long)sz);
      const u64 pos = g_allocRingPos.load(std::memory_order_relaxed);
      const int have = (int)(pos < kAllocRing ? pos : kAllocRing);
      BASE_LOGI("treewalk",
                "the last {} allocator calls, oldest first (the "
                "corruption happened inside this window):", have);
      for (int k = have; k > 0; k--) {
        const AllocEvt &e = g_allocRing[(pos - k) % kAllocRing];
        BASE_LOGI("treewalk", "  {:<24} tid={} a0={:#x} a1={:#x}",
                  e.site < kAllocLockSites ? g_allocSites[e.site].name : "?",
                  e.tid, (unsigned long long)e.a0, (unsigned long long)e.a1);
      }
      return;
    }
    for (int c = 0; c < 2 && sp < 254; c++) {
      const u64 f = cur + (u64)c * 8;
      stack[sp] = *reinterpret_cast<const u64 *>(f);
      fields[sp] = f;
      sp++;
    }
  }
}

static void allocLockLeaveAt(int i) {
  if (kSotcAllocLock && g_allocSites[i].serialise) {
    // Unlock the same mutex the entry took: arg0 is not available here, so the
    // entry records which state it locked on a small per-thread stack.
    const u64 a0 = t_lockDepth > 0 ? t_lockedState[--t_lockDepth] : 0;
    allocMutexFor(i, a0).unlock();
  }
  if (kSotcTreeWatch)
    treeWatchAt(i, true);
  if (kSotcTreeWalk)
    treeWalkPeriodic();
}

template <int N>
static void PS4ABI allocLockEnterT(u64 a0, u64 a1) {
  allocLockEnterAt(N, a0, a1);
}
template <int N> static void PS4ABI allocLockLeaveT() { allocLockLeaveAt(N); }

// Install an entry detour on an eboot-internal function. prologueLen must be the
// smallest instruction boundary >= 14 in the prologue (position-independent).
void installInternalHook(u8 *base, u32 off, u32 prologueLen,
                         u32 hookId, const char *name) {
  u8 *target = base + off;
  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(target) & ~0xFFFull),
                  0x2000, utl::pageProtection::rwx);
  uintptr_t tramp = cpu::makeGuestTrampoline(target, prologueLen, target + prologueLen);
  if (!tramp) { LOG_WARNING("jobtrace: trampoline failed for {}", name); return; }
  uintptr_t wrap = cpu::makeGuestReturnHook(reinterpret_cast<void *>(tramp), hookId,
                                            reinterpret_cast<void *>(&jobTraceLogger), name);
  if (!wrap) { LOG_WARNING("jobtrace: wrapper failed for {}", name); return; }
  u8 patch[32];
  patch[0] = 0xFF; patch[1] = 0x25;
  patch[2] = patch[3] = patch[4] = patch[5] = 0x00;   // jmp qword [rip+0]
  u64 w = wrap; std::memcpy(patch + 6, &w, 8);
  for (u32 i = 14; i < prologueLen; i++) patch[i] = 0x90;
  std::memcpy(target, patch, prologueLen);
  LOG_INFO("jobtrace: hooked {} eboot+{:#x} tramp={:#x} wrapper={:#x}", name, off,
           (unsigned long)tramp, (unsigned long)wrap);
}

// Same detour, but the wrapper takes the host allocator mutex across the call.
static void installAllocLockHook(u8 *base, u32 off, u32 prologueLen,
                                 const char *name, void *enterFn, void *leaveFn) {
  u8 *target = base + off;
  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(target) & ~0xFFFull),
                  0x2000, utl::pageProtection::rwx);
  uintptr_t tramp = cpu::makeGuestTrampoline(target, prologueLen, target + prologueLen);
  if (!tramp) { LOG_WARNING("alloclock: trampoline failed for {}", name); return; }
  uintptr_t wrap = cpu::makeGuestLockWrapper(reinterpret_cast<void *>(tramp),
                                             enterFn, leaveFn, name);
  if (!wrap) { LOG_WARNING("alloclock: wrapper failed for {}", name); return; }
  u8 patch[32];
  patch[0] = 0xFF; patch[1] = 0x25;
  patch[2] = patch[3] = patch[4] = patch[5] = 0x00;   // jmp qword [rip+0]
  u64 w = wrap; std::memcpy(patch + 6, &w, 8);
  for (u32 i = 14; i < prologueLen; i++) patch[i] = 0x90;
  std::memcpy(target, patch, prologueLen);
  LOG_INFO("alloclock: serialising {} eboot+{:#x} tramp={:#x} wrapper={:#x}", name,
           off, (unsigned long)tramp, (unsigned long)wrap);
}

} // namespace

// Prologue cut points are the smallest instruction boundary >= 14 that stays
// position-independent; each stops before the function's first rip-relative load:
//   0x12820 alloc: pushes + `sub rsp,0x28` end at 17, then `mov r14,[rip+..]`
//   0x12af0 free : pushes + `sub rsp,0x30` end at 15, then `mov r15,[rip+..]`
//   0x48a70 free-tree insert: pushes + `mov r15,rdi` + `mov r14,rsi` end at 16
void installAllocLock(smodule &m) {
  if (!kSotcAllocLock && !kSotcTreeWatch && !kSotcTreeWalk && !kSotcHeapRoute)
    return;
  u8 *base = m.getInfo().base;
  struct Site { u32 off; u32 cut; const char *name; bool lock; };
  // Prologue cuts are the smallest position-independent instruction boundary >= 14,
  // each stopping before the function's first rip-relative load; every one was
  // checked for positive-rbp stack-argument reads, which an entry detour breaks.
  // 0x4a040 uses rbp as a NODE pointer (it loads it from rdx at 0x4a075) rather
  // than as a frame pointer, so its [rbp+0x20] accesses are node fields and safe.
  //
  // The four TREE MUTATORS carry the shared lock; the two API entries are observed
  // only. Locking the API was the earlier mistake: the allocator core has 22
  // external direct entry points scattered across the eboot, so serialising
  // 0x12820 and 0x12af0 left most doors open. All 11 call sites of the mutators
  // are inside the core, so locking THEM covers every door at once.
  static const Site kSites[kAllocLockSites] = {
      {0x12820, 17, "alloc(0x12820)",          false},
      {0x12af0, 15, "free(0x12af0)",           false},
      {0x48a70, 16, "treeInsert(0x48a70)",     true},
      {0x497f0, 16, "treeRemove(0x497f0)",     true},
      {0x4a040, 14, "treeRebalance(0x4a040)",  false},
      {0x4a210, 15, "treeFixup(0x4a210)",      false},
      // The census site 0x48dfb lives here. An earlier pass mis-attributed it to
      // 0x48c70 because that address is a 9-byte leaf whose `ret` is followed by
      // seven nops, one short of the eight the function-boundary scan wanted --
      // so this mutator went unserialised through the whole first experiment.
      // Observed, NOT locked. Serialising this one stalls the title outright:
      // hooked-but-unlocked runs 174093 inserts and 3.3 fps in five minutes, and
      // hooked-and-locked manages 129 inserts and never renders a frame -- with
      // per-state locks too, so it is not false sharing across heaps. The
      // coalescer evidently must not be held across by a foreign lock (it is
      // reached with guest locks already held, and the title's sched_yield spin
      // convoys behind it). Left in the table because its contention count is the
      // measurement; flipping this to true is how the stall reproduces.
      {0x48c80, 17, "treeCoalesce(0x48c80)",   false},
  };
  static void *const kEnter[kAllocLockSites] = {
      reinterpret_cast<void *>(&allocLockEnterT<0>),
      reinterpret_cast<void *>(&allocLockEnterT<1>),
      reinterpret_cast<void *>(&allocLockEnterT<2>),
      reinterpret_cast<void *>(&allocLockEnterT<3>),
      reinterpret_cast<void *>(&allocLockEnterT<4>),
      reinterpret_cast<void *>(&allocLockEnterT<5>),
      reinterpret_cast<void *>(&allocLockEnterT<6>)};
  static void *const kLeave[kAllocLockSites] = {
      reinterpret_cast<void *>(&allocLockLeaveT<0>),
      reinterpret_cast<void *>(&allocLockLeaveT<1>),
      reinterpret_cast<void *>(&allocLockLeaveT<2>),
      reinterpret_cast<void *>(&allocLockLeaveT<3>),
      reinterpret_cast<void *>(&allocLockLeaveT<4>),
      reinterpret_cast<void *>(&allocLockLeaveT<5>),
      reinterpret_cast<void *>(&allocLockLeaveT<6>)};
  for (int i = 0; i < kAllocLockSites; i++) {
    g_allocSites[i].name = kSites[i].name;
    g_allocSites[i].serialise = kSites[i].lock;
    installAllocLockHook(base, kSites[i].off, kSites[i].cut, kSites[i].name,
                         kEnter[i], kLeave[i]);
  }
  if (kSotcTreeWatch || kSotcTreeWalk)
    treeWatchArm();
  std::thread([] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(10));
      if (kSotcHeapRoute)
        heapRouteReport();
      if (kSotcTreeWatch)
        BASE_LOGI("treewatch",
                  "{} checks, {} found it already bad on "
                  "entry, {} caught it GOING bad",
                  (unsigned long long)g_treeChecks.load(),
                  (unsigned long long)g_treeBadAtEntry.load(),
                  (unsigned long long)g_treeWentBad.load());
      for (int i = 0; i < kAllocLockSites; i++)
        BASE_LOGI("alloclock",
                  "{:<24} {} calls, {} CONTENDED (another "
                  "thread was inside), max wait {} ns",
                  g_allocSites[i].name,
                  (unsigned long long)g_allocSites[i].calls.load(),
                  (unsigned long long)g_allocSites[i].contended.load(),
                  (unsigned long long)g_allocSites[i].maxWaitNs.load());
    }
  }).detach();
}

// DELTA_SOTC_MATTRACE: name the blocks SotC's "Material Param Update" fills.
// Hooks the dispatch builder rather than the update routine itself, because the
// block is that call's 4th argument (rcx) and only a local inside the caller.
// Prologue cut point 17 (push rbp / mov rbp,rsp / 5 pushes / sub rsp,0x58); the
// function takes all five arguments in registers and reads nothing at a
// positive rbp offset, which is the thing an entry detour would break.
void installMatTrace(smodule &m) {
  if (!kSotcMatTrace)
    return;
  installInternalHook(m.getInfo().base, 0x16eb90, 17, 13,
                      "MaterialParamDispatch(0x16eb90)");
  // ...and the packet emitter it ends in, to name the command buffer the
  // dispatch is written into. Prologue is exactly 14 (7 pushes after
  // `mov rbp,rsp`) and position-independent; all five arguments are in
  // registers.
  installInternalHook(m.getInfo().base, 0x883870, 14, 14,
                      "DispatchDirectEmit(0x883870)");
}

void installJobTrace(smodule &m) {
  if (!kJobTrace)
    return;
  u8 *base = m.getInfo().base;
  // prologue cut points (smallest instr boundary >= 14, verified by disasm):
  //   claim 0x38d40 -> 20 (through `sub rsp,0xa8`)
  //   kick  0x35480 -> 17 (through `sub rsp,0x38`)
  //   ctor  0x36210 -> 15 (through `sub rsp,0x40`)
  // The ctor hook alone is enough to arm the watcher and costs one call per
  // run. The claim hook additionally counts claim/fail rates, but the workers
  // hit it ~2M times/sec, and the magic-syscall round trip roughly halves the
  // world-load drain rate -- opt in with DELTA_JOB_TRACE_CLAIM when the rates
  // are what you're after.
  installInternalHook(base, 0x36210, 15, 11, "JobSystemCtor(0x36210)");
  if (kJobTraceClaim)
    installInternalHook(base, 0x38d40, 20, 12, "DoClaimJob(0x38d40)");
}

void investigateDcbGate(smodule &m) {
  if (!kPs5Dcbwatch)
    return;
  u8 *base = m.getInfo().base;
  struct { u32 off; const char *label; } pts[] = {
      {0x425ef0, "app_main(0x425ef0)"},   {0x4cc830, "app_render(0x4cc830)"},
      {0x5535d0, "RenderInit(0x5535d0)"}, {0x58fb10, "VOInit(0x58fb10)"},
      {0x58fd50, "DCBframeInit(0x58fd50)"}, {0x5901a0, "rendererFrame(0x5901a0)"},
  };
  for (auto &pt : pts) {
    auto *c = base + pt.off;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x1000, utl::pageProtection::rwx);
    if (c[0] == 0x55) {
      c[0] = 0xCC;
      setOrderTrace(reinterpret_cast<uintptr_t>(c), pt.label);
    } else {
      LOG_WARNING("dcbwatch: {} first byte {:#x} != push rbp", pt.label, c[0]);
    }
  }
  // 0x69e720 is the graphics-subsystem run-once init that VOInit calls before
  // sceVideoOutOpen; when it returns non-zero VOInit bails and the DCB is never
  // created. It accumulates its error in ebx from three sub-calls; trace each
  // `mov ebx,eax` return so we see which import fails.
  struct { u32 off; const char *label; } rets[] = {
      {0x69e761, "0x69e720:vSMAm3cxYTY#1"},
      {0x69e78f, "0x69e720:vSMAm3cxYTY#2"},
      {0x69e7aa, "0x69e720:23LRUSvYu1M"},
  };
  for (auto &r : rets) {
    auto *c = base + r.off;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x1000, utl::pageProtection::rwx);
    if (c[0] == 0x89 && c[1] == 0xc3) {
      c[0] = 0xCC;
      setRetTrace(reinterpret_cast<uintptr_t>(c), r.label);
    } else {
      LOG_WARNING("dcbwatch: {} bytes {:#x} {:#x} != mov ebx,eax", r.label, c[0], c[1]);
    }
  }
  // Identify the imports 0x69e720 calls by symbolizing their resolved GOT slots
  // (imports are already bound at this point).
  auto *p = proc::getActive();
  struct { u32 got; const char *nid; } gots[] = {
      {0x8e8e38, "vSMAm3cxYTY"}, {0x8e8e40, "23LRUSvYu1M(FAILING)"},
      {0x8e8e48, "2JtWUUiYBXs"}, {0x8e8d80, "1jfXLRVzisc"},
  };
  for (auto &g : gots) {
    u64 tgt = *reinterpret_cast<u64 *>(base + g.got);
    const char *mod = "??";
    u64 off = tgt;
    if (p)
      for (auto &mm : p->getModuleList()) {
        auto &mi = mm->getInfo();
        auto tb = reinterpret_cast<u64>(mi.textSeg.addr);
        if (tb && tgt >= tb && tgt < tb + mi.textSeg.size) {
          mod = mi.name.c_str();
          off = tgt - tb;
          break;
        }
      }
    // Does any loaded module export this NID's hash? (is it resolvable?)
    u64 hid = 0;
    const char *expMod = "NONE";
    u64 expAddr = 0;
    if (runtime::decode_nid(g.nid, 11, hid) && p)
      for (auto &mm : p->getModuleList())
        if (uintptr_t a = mm->getExport(hid)) {
          expMod = mm->getInfo().name.c_str();
          expAddr = a;
          break;
        }
    BASE_LOGI("dcbimp", "{} got={}+{:#x} exportedBy={}({:#x})", g.nid, mod,
              (unsigned long long)off, expMod, (unsigned long long)expAddr);
  }
  auto *slot = reinterpret_cast<volatile u64 *>(base + 0x985a00);
  std::thread([slot] {
    u64 last = ~1ull;
    for (int i = 0; i < 400000; i++) {
      u64 v = *slot;
      if (v != last) {
        BASE_LOGI("dcbwatch", "t={}ms manager[0] (eboot+0x985a00) = {:#x}",
                  i / 2, (unsigned long long)v);
        last = v;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
  }).detach();
}

/*does not expect an extension*/
// Isaac-specific surface setup. The game's global surface-name registry, a
// fixed-bucket hash map<string,surface> at rebirth+0x687a90, is constructed
// with a NULL bucket-array pointer (its ctor at +0x1e9bcd just zeroes it) and is
// meant to get its storage lazily when the renderer registers the base surfaces
// ("Floor Surface"/"Wall Surface"). That renderer path is gated on the Gnm->
// Vulkan graphics device, which isn't brought up yet, so the bucket array is
// never allocated and every registry find/insert dereferences null + idx*0x20
// (rebirth+0x1e8c8b on a worker insert, +0x1e7e09 on a main-thread find). Until
// the real gfx/renderer init exists, hand the registry valid empty storage up
// front: allocate the N=0x20 * 0x20-byte zeroed bucket array, point the registry
// at it, and NOP the ctor's null-write so it can't clobber the pointer when it
// later runs. The map then works as a valid empty registry independent of order
// or the missing gfx init. (Verified offsets via the decrypted rebirth.elf.)
void bringUpRebirthSurfaceRegistry(smodule &m) {
  u8 *base = m.getInfo().base;
  constexpr u32 kRegistryOff = 0x687a90; // bucket-array base pointer
  constexpr u32 kCtorZeroOff = 0x1e9bcd; // `mov qword [registry], 0` (11 bytes)
  constexpr size_t kBucketBytes = 0x20 * 0x20; // N buckets * stride

  u8 *buckets = allocLowGuest(kBucketBytes);
  if (!buckets) {
    LOG_ERROR("rebirth surface-registry: bucket alloc failed");
    return;
  }
  *reinterpret_cast<u64 *>(base + kRegistryOff) =
      reinterpret_cast<u64>(buckets);

  u8 *ctor = base + kCtorZeroOff;
  utl::protectMem(
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ctor) & ~0xFFFull),
      0x1000, utl::pageProtection::rwx);
  std::memset(ctor, 0x90, 11); // NOP the ctor's null-write

  LOG_INFO("rebirth surface-registry: installed empty buckets@{} -> [+{:#x}]",
           (void *)buckets, kRegistryOff);

  // DIAGNOSTIC (DELTA_GFXCTX_WATCH): poll the rebirth GfxContext singleton's
  // command-buffer pointer (rebirth+0x687b30, field +0x38). The both-LLE render
  // thread faults at rebirth+0x23f027 dereferencing this when it is null; this
  // shows whether/when it gets allocated. Logs every transition.
  if (kGfxctxWatch) {
    auto *slot = reinterpret_cast<volatile u64 *>(base + 0x687b30 + 0x38);
    std::thread([slot] {
      u64 last = ~0ull;
      for (int i = 0; i < 200000; i++) {
        u64 v = *slot;
        if (v != last) {
          BASE_LOGI("gfxctx", "+0x38 = {:#x}  (t={}ms)",
                    (unsigned long long)v, i / 2);
          last = v;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500));
      }
    }).detach();
  }
}

// PS5 bring-up (mirror of bringUpRebirthSurfaceRegistry above). The PS5 eboot
// statically links the same "rebirth" engine, so it has the same global surface-
// name registry, here at eboot+0x985458 (bucket-array base pointer, a fixed 0x20
// buckets * 0x20-byte stride). Its ctor at eboot+0x56a5d0 zeroes the pointer
// (the null-write at eboot+0x56a5e7) and never allocates the array -- that
// storage is meant to come from the renderer registering the base surfaces,
// which needs the AGC/GPU device that isn't up yet. Main-init's first registry
// find (eboot+0x568840) then iterates null+idx*0x20 and faults (eboot+0x56889e).
// Hand it a valid empty bucket array up front and NOP the ctor's null-write, as
// on PS4. Guarded by the exact ctor-instruction bytes so a different PS5 title's
// eboot is left untouched. (Offsets verified against the decrypted eboot.)
void bringUpRebirthEbootRegistry(smodule &m) {
  u8 *base = m.getInfo().base;
  constexpr u32 kRegistryOff = 0x985458; // bucket-array base pointer
  constexpr u32 kCtorZeroOff = 0x56a5e7; // `mov qword [registry], 0` (11 bytes)
  constexpr size_t kBucketBytes = 0x20 * 0x20; // N buckets * stride

  // `mov qword [rip+0x41ae66], 0` -> eboot+0x985458. Only this build has it.
  static const u8 kCtorBytes[] = {0x48, 0xc7, 0x05, 0x66, 0xae, 0x41,
                                       0x00, 0x00, 0x00, 0x00, 0x00};
  if (std::memcmp(base + kCtorZeroOff, kCtorBytes, sizeof(kCtorBytes)) != 0)
    return; // not this title's eboot; nothing to bring up

  u8 *buckets = allocLowGuest(kBucketBytes);
  if (!buckets) {
    LOG_ERROR("rebirth eboot-registry: bucket alloc failed");
    return;
  }
  *reinterpret_cast<u64 *>(base + kRegistryOff) =
      reinterpret_cast<u64>(buckets);

  u8 *ctor = base + kCtorZeroOff;
  utl::protectMem(
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(ctor) & ~0xFFFull),
      0x1000, utl::pageProtection::rwx);
  std::memset(ctor, 0x90, 11); // NOP the ctor's null-write

  LOG_INFO("rebirth eboot-registry: installed empty buckets@{} -> [+{:#x}]",
           (void *)buckets, kRegistryOff);
}

// DIAGNOSTIC (DELTA_VO_PATCH): patch real libSceVideoOut export entries to
// `mov eax,<v>; ret`, to isolate which real setup function's error return makes
// rebirth skip command-buffer creation. List: open,regbuf,fliprate,addflip,getlabel.
// open returns 1 (a valid handle); the rest return 0 (SCE_OK).
// DIAGNOSTIC (DELTA_VO_WATCH): poll libSceVideoOut's internal display-config state
// during the NATURAL flow (its init runs via pthread_once on first Open). Shows
// how count[0x1cb30]/idx[0x1cb40]/cfg[*].f0[0x1cb50 stride 0x140] evolve, to pin
// exactly what the driver fails to set (the display-connected state f0==4 Open needs).
static void watchVideoOutState(smodule &m) {
  if (!kVoWatch)
    return;
  u8 *base = m.getInfo().base;
  std::thread([base] {
    i32 lc = 0x7fffffff, li = 0x7fffffff;
    u32 lf[3] = {0xdead, 0xdead, 0xdead};
    for (int i = 0; i < 120000; i++) {
      i32 c = *reinterpret_cast<volatile i32 *>(base + 0x1cb30);
      i32 idx = *reinterpret_cast<volatile i32 *>(base + 0x1cb40);
      u32 f[3];
      for (int s = 0; s < 3; s++)
        f[s] = *reinterpret_cast<volatile u32 *>(base + 0x1cb50 + s * 0x140);
      // port[0] @ vaddr 0x1d550 (stride 0xb0): field@0x14=open flag; field@0x48 set
      // to 0xfffffff3 once Open reaches the deep success path (just before op@0x580).
      u32 portOpen = *reinterpret_cast<volatile u32 *>(base + 0x1d564);
      u32 port48 = *reinterpret_cast<volatile u32 *>(base + 0x1d550 + 0x48);
      static u32 lpo = 0xdead, lp48 = 0xdead;
      if (c != lc || idx != li || f[0] != lf[0] || f[1] != lf[1] || f[2] != lf[2] ||
          portOpen != lpo || port48 != lp48) {
        BASE_LOGI("vowatch",
                  "t={}ms count={} idx={} cfg.f0=[{:#x} {:#x} {:#x}] "
                  "port0.open={} port0.f48={:#x}",
                  i / 2, c, idx, f[0], f[1], f[2], portOpen, port48);
        lc = c; li = idx; lf[0] = f[0]; lf[1] = f[1]; lf[2] = f[2]; lpo = portOpen;
        lp48 = port48;
      }
      // DELTA_VO_FORCE_CONNECT: once the driver registered the placeholder into
      // cfg[0] (f0 went -1 -> 0) but Open reads cfg[idx] (still free -1), make a
      // connected display: copy the populated cfg[0] slot into cfg[idx] and mark
      // it connected (f0=4). Proves the display-config mechanism end to end.
      static bool patched = false;
      if (kVoForceConnect && !patched && idx >= 1 &&
          idx < 8 && f[0] == 0 && f[1] == 0xffffffff) {
        u8 *cfg0 = base + 0x1cb50;
        u8 *cfgi = base + 0x1cb50 + (size_t)idx * 0x140;
        std::memcpy(cfgi, cfg0, 0x140);                 // copy ops/vtable
        *reinterpret_cast<u32 *>(cfgi) = 4;        // f0 = connected
        patched = true;
        BASE_LOGI("vowatch", "FORCE_CONNECT: cfg[{}] <- cfg[0], f0=4", idx);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
  }).detach();
}

// Host hook for the videoout busType/index map-op (vaddr 0x1020). Open calls it
// with esi=userId, edx=busType, ecx=index, r8=param. Via makeHostThunk those land
// in args (rsi,rdx,r10,r8) -> a2,a3,a4,a5. Logs the title's real Open() args and
// returns 0 (the op's value for the main display). If this never logs, Open failed
// before the op (count/f0/param gate).
static u64 PS4ABI voOpMapLog(u64 a1, u64 userId, u64 busType,
                                  u64 index, u64 param, u64 a6) {
  u32 pv = 0;
  if (param > 0x10000 && param < 0x800000000000ull)
    pv = *reinterpret_cast<u32 *>(param);
  BASE_LOGI("voop",
            "sceVideoOutOpen(userId={:#x} busType={} index={} param={:#x} "
            "[param]={:#x} [param]&0xf={:#x}) -> map-op returns 0",
            (unsigned long)userId, (long)busType, (long)index,
            (unsigned long)param, pv, pv & 0xf);
  return 0;
}

void patchVideoOutDiag(smodule &m) {
  watchVideoOutState(m);
  if (kVoOplog) {
    uintptr_t thunk = cpu::makeHostThunk(reinterpret_cast<void *>(&voOpMapLog));
    u8 *o = m.getInfo().base + 0x1020;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(o) & ~0xFFFull),
                    0x2000, utl::pageProtection::rwx);
    o[0] = 0x48; o[1] = 0xb8;                       // mov rax, imm64
    *reinterpret_cast<u64 *>(o + 2) = thunk;
    o[10] = 0xff; o[11] = 0xe0;                     // jmp rax
    BASE_LOGI("voop", "hooked map-op @ +0x1020 -> thunk {:#x}",
              (unsigned long)thunk);
  }
  // TEST (DELTA_VO_SKIP_580): nop the `js error` after Open's `call op@0x580`
  // (config-validate op). If Open then progresses, op@0x580's return was a gate.
  if (kVoSkip580) {
    u8 *c = m.getInfo().base + 0xaeb8;  // js 0xef09 after the op call
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x2000, utl::pageProtection::rwx);
    if (c[0] == 0x78) {  // js rel8
      c[0] = 0x90; c[1] = 0x90;
      BASE_LOGI("votest", "nop'd op@0x580 error-js @ +0xaeb8");
    } else {
      BASE_LOGI("votest", "op@0x580 js bytes mismatch: {:#x} {:#x}", c[0], c[1]);
    }
  }
  const char *list = kVoPatch;
  if (!list)
    return;
  u8 *base = m.getInfo().base;
  struct { const char *name; u32 off; u8 ret; } fns[] = {
      {"open", 0xaad0, 1}, {"regbuf", 0xb620, 0}, {"fliprate", 0xbde0, 0},
      {"addflip", 0xc6c0, 0}, {"getlabel", 0xbb80, 0}};
  for (auto &fn : fns) {
    if (!std::strstr(list, fn.name))
      continue;
    u8 *c = base + fn.off;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x2000, utl::pageProtection::rwx);
    c[0] = 0xb8; c[1] = fn.ret; c[2] = 0; c[3] = 0; c[4] = 0;  // mov eax, imm32
    c[5] = 0xc3;                                               // ret
    BASE_LOGI("vopatch", "libSceVideoOut!{} -> return {}", fn.name, fn.ret);
  }
}


// Patch a guest function to `xor eax,eax; ret`. Steps over libkernel-internal
// validation that rejects our externally-loaded module set (11.00 offsets).
static void forceReturn0(proc &p, const char *mod, u32 off) {
  auto m = p.getModule(base::StringRef(mod));
  if (!m)
    return;
  u8 *c = m->getInfo().base + off;
  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                  0x1000, utl::pageProtection::rwx);
  c[0] = 0x31;  // xor eax, eax
  c[1] = 0xC0;
  c[2] = 0xC3;  // ret
}

// Patch a (rdi=paramId, rsi=int* out) getter to `*out = val; return 0`.
static void forceGetterOk(proc &p, const char *mod, u32 off, u32 val) {
  auto m = p.getModule(base::StringRef(mod));
  if (!m)
    return;
  u8 *c = m->getInfo().base + off;
  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                  0x1000, utl::pageProtection::rwx);
  c[0] = 0xC7; c[1] = 0x06;  // mov dword [rsi], imm32
  std::memcpy(c + 2, &val, 4);
  c[6] = 0x31; c[7] = 0xC0;  // xor eax, eax
  c[8] = 0xC3;               // ret
}

// Boot patches applied before entering the guest, needed by every boot path
// (modexec and the real pkg boot), not just the modexec harness.
void applyBootPatches(proc &p) {
  // Redirect libkernel's __tls_get_addr (NID vNe1w4diLCs) to our per-thread HLE;
  // libkernel's own dynamic-TLS allocator leaves DTV entries null. NID lookup is
  // firmware-independent.
  //
  // NATIVE ONLY: this patches the guest to `jmp` a *host* function pointer,
  // which only works when the host runs x86 directly. Under the FEXCore JIT
  // (aarch64) that address is ARM code and jumping to it as x86 faults wildly.
  // TODO(boot/fex): redirect __tls_get_addr via a FEXCore thunk trampoline, or
  // satisfy guest dynamic TLS another way.
#if defined(DELTA_BACKEND_NATIVE)
  if (auto k = p.getModule(base::StringRef("libkernel"))) {
    if (uintptr_t a = k->getSymbolByNid("vNe1w4diLCs")) {
      auto *c = reinterpret_cast<u8 *>(a);
      utl::protectMem(reinterpret_cast<void *>(a & ~0xFFFull), 0x2000,
                      utl::pageProtection::rwx);
      c[0] = 0x48;  // movabs rax, imm64
      c[1] = 0xB8;
      *reinterpret_cast<u64 *>(c + 2) =
          reinterpret_cast<u64>(&guest_tls_get_addr);
      c[10] = 0xFF;  // jmp rax
      c[11] = 0xE0;
      LOG_INFO("patched libkernel __tls_get_addr -> host HLE");
    }
  }
#else  // DELTA_BACKEND_FEX
  // FEX path: a host jump is invalid inside the x86 JIT, so patch the export to
  // a tiny `mov eax, <magic>; syscall; ret` stub that the FEX syscall handler
  // bridges to krnl::guest_tls_get_addr (tls_index ptr arrives in rdi).
  if (auto k = p.getModule(base::StringRef("libkernel"))) {
    if (uintptr_t a = k->getSymbolByNid("vNe1w4diLCs")) {
      auto *c = reinterpret_cast<u8 *>(a);
      utl::protectMem(reinterpret_cast<void *>(a & ~0xFFFull), 0x2000,
                      utl::pageProtection::rwx);
      c[0] = 0xB8; // mov eax, imm32
      *reinterpret_cast<u32 *>(c + 1) = cpu::kTlsGetAddrSyscall;
      c[5] = 0x0F; // syscall
      c[6] = 0x05;
      c[7] = 0xC3; // ret
      LOG_INFO("patched libkernel __tls_get_addr -> magic syscall (FEX)");
    }
  }
#endif
  // DEBUG (DELTA_TRAP_VADDR=0xADDR[,0xADDR...]): plant an int3 at guest code
  // address(es) so reaching them traps into the crash handler, which dumps the
  // guest RIP/registers/backtrace + stack scan. Lets us capture the context of a
  // deterministic-but-hard-to-breakpoint site (e.g. a fatal-error spin) without
  // gdb, which is far too slow under the boot's threading.
  if (const char *t = kTrapVaddr) {
    base::String spec(t);
    char *cur = const_cast<char *>(spec.c_str());
    while (cur && *cur) {
      u64 addr = std::strtoull(cur, &cur, 0);
      if (addr) {
        auto *c = reinterpret_cast<u8 *>(addr);
        utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                        utl::pageProtection::rwx);
        c[0] = 0xCC; // int3 -> SIGTRAP -> crash handler dump
        LOG_INFO("DELTA_TRAP_VADDR: planted int3 @ {:#x}", addr);
      }
      while (*cur == ',' || *cur == ' ') cur++;
    }
  }
  // DEBUG (DELTA_ALLOC_TRACE=0xADDR[,minMB]): trace large allocations through a
  // guest allocator whose entry (ADDR) begins with `push rbp`. Replace it with
  // int3; the fatal handler logs the size (rsi) and emulates the push. Lets us
  // see what fills a fixed heap (e.g. SOTTR's 1 GiB pool) without gdb.
  if (const char *at = kAllocTrace) {
    char *end = nullptr;
    u64 addr = std::strtoull(at, &end, 0);
    u64 minB = 0x1000000;
    if (end && *end == ',') minB = std::strtoull(end + 1, nullptr, 0) * 1024 * 1024;
    // DELTA_ALLOC_TRACE doubles as a boolean toggle for the [lowalloc] tracer in
    // sys_mem.cpp, so a bare "=1" is legitimate and must NOT be treated as a code
    // address here: addr=1 would protect/deref page 0 (host null-deref SIGSEGV).
    // A real allocator entry is a guest .text vaddr (>= 64 KiB); ignore anything
    // smaller so tracing can be enabled without planting an int3 at a bogus addr.
    if (addr >= 0x10000) {
      auto *c = reinterpret_cast<u8 *>(addr);
      utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                      utl::pageProtection::rwx);
      if (c[0] == 0x55) {  // push rbp
        c[0] = 0xCC;       // int3
        setAllocTrace(addr, minB);
        LOG_INFO("DELTA_ALLOC_TRACE: hooked alloc entry {:#x} (>= {} MB)", addr,
                 minB / (1024 * 1024));
      } else {
        LOG_WARNING("DELTA_ALLOC_TRACE: {:#x} first byte {:#x} != push rbp", addr,
                    c[0]);
      }
    }
  }
  // DELTA_HEAP_PROF=0xADDR: plant int3 at an operator-new/malloc entry (push rbp,
  // size in rdi) and aggregate bytes+count by guest caller; SIGUSR1 dumps top sites.
  // DELTA_HEAP_PROF_SCOPE=<tls-slot-global>:<depth-offset>: additionally split
  // each site by whether the thread had a scoped allocator live (see crash.h).
  if (const char *hs = kHeapProfScope) {
    char *cur = const_cast<char *>(hs);
    const u64 slot = std::strtoull(cur, &cur, 0);
    const u64 depth = (*cur == ':') ? std::strtoull(cur + 1, nullptr, 0) : 0;
    if (slot >= 0x10000) {
      setHeapProfScope(slot, depth);
      LOG_INFO("DELTA_HEAP_PROF_SCOPE: tls slot {:#x} depth +{:#x}", slot, depth);
    }
  }
  if (const char *hp = kHeapProf) {
    char *cur = const_cast<char *>(hp);
    while (cur && *cur) {
      u64 addr = std::strtoull(cur, &cur, 0);
      // "<addr>:c" marks a deallocator: its first argument is a pointer, so
      // the site is reported by call count instead of by bytes.
      bool countOnly = false;
      if (*cur == ':' && (cur[1] == 'c' || cur[1] == 'C')) {
        countOnly = true;
        cur += 2;
      }
      if (addr >= 0x10000) {
        auto *c = reinterpret_cast<u8 *>(addr);
        utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                        utl::pageProtection::rwx);
        if (c[0] == 0x55) { c[0] = 0xCC; setHeapProf(addr, countOnly);
          LOG_INFO("DELTA_HEAP_PROF: hooked alloc entry {:#x}", addr);
        } else {
          LOG_WARNING("DELTA_HEAP_PROF: {:#x} first byte {:#x} != push rbp", addr, c[0]);
        }
      }
      while (*cur == ',' || *cur == ' ') cur++;
    }
  }
  if (const char *ct = kCntTrace) {
    u64 addr = std::strtoull(ct, nullptr, 0);
    if (addr) {
      auto *c = reinterpret_cast<u8 *>(addr);
      utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                      utl::pageProtection::rwx);
      if (c[0] == 0x55) { c[0] = 0xCC; setCntTrace(addr); }
    }
  }
  if (const char *ft = kFatalTrace) {
    u64 addr = std::strtoull(ft, nullptr, 0);
    if (addr) {
      auto *c = reinterpret_cast<u8 *>(addr);
      utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                      utl::pageProtection::rwx);
      if (c[0] == 0x55) { c[0] = 0xCC; setFatalTrace(addr); }
    }
  }
  if (const char *ht = kHdrTrace) {
    // Comma-separated list of consumer entry vaddrs to hook (e.g. 0x606150,0x6063a0).
    const char *s = ht;
    while (*s) {
      char *end = nullptr;
      u64 addr = std::strtoull(s, &end, 0);
      if (addr) {
        auto *c = reinterpret_cast<u8 *>(addr);
        utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                        utl::pageProtection::rwx);
        if (c[0] == 0x55) { c[0] = 0xCC; setHdrTrace(addr); }
      }
      s = (end && *end == ',') ? end + 1 : (end ? end : s + 1);
      if (!*s || (end && *end != ',')) break;
    }
  }
  if (const char *ro = kRdoffFix) {
    u64 addr = std::strtoull(ro, nullptr, 0);
    if (addr) {
      auto *c = reinterpret_cast<u8 *>(addr);
      utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                      utl::pageProtection::rwx);
      if (c[0] == 0x55) { c[0] = 0xCC; setRdoffFix(addr); }
    }
  }
  if (const char *sf = kSkipFn) {
    const char *s2 = sf;
    while (*s2) {
      char *end = nullptr;
      u64 addr = std::strtoull(s2, &end, 0);
      if (addr) {
        auto *c = reinterpret_cast<u8 *>(addr);
        utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                        utl::pageProtection::rwx);
        if (c[0] == 0x55) { c[0] = 0xCC; setSkipFn(addr); }
      }
      s2 = (end && *end == ',') ? end + 1 : (end ? end : s2 + 1);
      if (!*s2 || (end && *end != ',')) break;
    }
  }
  ps5::maybePrependCtor(p);
  forceReturn0(p, "libkernel", 0x287e0);            // module-gen lib-id validator
  forceReturn0(p, "libSceAppContentUtil", 0x1a00);  // AppContent IPMI init
  // AppContent's IPMI client is stubbed, so the real Initialize/AppParamGetInt
  // error out and the engine's GameSystemInit (big.cpp:1298/1302) asserts. Force
  // both to SCE_OK: Initialize (bootParam out is pre-zeroed = normal boot) and
  // AppParamGetInt returning SKU_FLAG = FULL (3).
  forceReturn0(p, "libSceAppContentUtil", 0x1610);     // sceAppContentInitialize
  forceGetterOk(p, "libSceAppContentUtil", 0x1630, 3); // sceAppContentAppParamGetInt
}
}  // namespace krnl::probe
