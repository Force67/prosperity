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
#include "kern/vfs.h"

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
DELTA_OPTION(const char *, kFnWatch, "DELTA_FNWATCH", nullptr);
DELTA_OPTION(const char *, kRetTrace, "DELTA_RETTRACE", nullptr);
DELTA_OPTION(const char *, kGuestPatch, "DELTA_GUEST_PATCH", nullptr);
DELTA_OPTION(const char *, kFnArgs, "DELTA_FNARGS", nullptr);
DELTA_OPTION(const char *, kFiosProbe, "DELTA_FIOS_PROBE", nullptr);
DELTA_OPTION(bool, kFiosAllopen, "DELTA_FIOS_ALLOPEN", false);
DELTA_OPTION(bool, kFiosTrace, "DELTA_FIOS_TRACE", false);
DELTA_OPTION(const char *, kNullGuard, "DELTA_GUEST_NULLGUARD", nullptr);
DELTA_OPTION(bool, kPs5Glyphguard, "DELTA_PS5_GLYPHGUARD", false);
DELTA_OPTION(bool, kPs5Noforce, "DELTA_PS5_NOFORCE", false);
DELTA_OPTION(bool, kSotcForcePayload, "DELTA_SOTC_FORCE_PAYLOAD", false);
DELTA_OPTION(bool, kSotcForceWorlddone, "DELTA_SOTC_FORCE_WORLDDONE", false);
DELTA_OPTION(bool, kSotcJobfix, "DELTA_SOTC_JOBFIX", false);
DELTA_OPTION(const char *, kGuestPopcnt, "DELTA_GUEST_POPCNT", nullptr);
DELTA_OPTION(const char *, kGuestWprot, "DELTA_GUEST_WPROT", nullptr);
DELTA_OPTION(const char *, kGuestRprot, "DELTA_GUEST_RPROT", nullptr);
DELTA_OPTION(const char *, kGuestWhist, "DELTA_GUEST_WHIST", nullptr);
DELTA_OPTION(const char *, kGuestSumwatch, "DELTA_GUEST_SUMWATCH", nullptr);
DELTA_OPTION(const char *, kPoolMap, "DELTA_POOLMAP", nullptr);
DELTA_OPTION(const char *, kMemDump, "DELTA_MEMDUMP", nullptr);
DELTA_OPTION(bool, kSotcSkipWorldwait, "DELTA_SOTC_SKIP_WORLDWAIT", false);
}  // namespace

namespace krnl {
const u32 *currentGuestTidPtr();  // sys_thread.cpp: this thread's guest tid
}

namespace krnl::probe {

static void probeFiosPaths();

// DELTA_PS5_DCBWATCH: diagnose the null frame-0 DrawCommandBuffer gate.
// (1) poll the DCB pointer slot manager[0] @ eboot+0x985a00 (renderer eboot+
//     0x985508 + idx0*0x600 + 0x138 + 0x3c0) + adjacent manager fields, logging
//     every change -> answers "is the DCB ever created before the render uses it?"
// (2) int3 call-order trace over the renderer/DCB-creation entry points so the
//     actual execution order (and which are reached) is visible before the crash.
// DELTA_FNWATCH="hexoff:label,hexoff:label,...": arm an int3 hit-counter at each
// guest function entry (first byte must be push rbp). Offsets are relative to the
// eboot base. The crash-handler counts hits and a printer thread logs totals every
// 2s (see crash.h). Generic; used to probe which functions in a stuck pipeline run.
static void investigatePopcnt();
static void investigateSumWatch();
static void investigateWriteWatch();
static void investigateWriteHist();
static void investigatePoolMap();
static void investigateMemDump();
static void investigateRetTrace(proc &);

// DELTA_RETTRACE="[module+]hexoff:label,...": arm a return-value trace at each
// `mov ebx,eax` / `test eax,eax` right after a call. Offsets are relative to the
// eboot unless a module name is given, which is what lets a failure be followed
// out of the title and into the system module that actually reports it.
static void investigateRetTrace(proc &pr) {
  const char *e = kRetTrace;
  if (!e)
    return;
  for (const char *p = e; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    base::String modName;
    if (const char *plus = std::strchr(p, '+')) {
      const char *comma = std::strchr(p, ',');
      if (!comma || plus < comma) {
        modName.append(p, (size_t)(plus - p));
        p = plus + 1;
      }
    }
    u8 *base = pr.getModuleList()[0]->getInfo().base;
    if (!modName.empty()) {
      auto mod = pr.getModule(base::StringRef(modName));
      if (!mod) {
        LOG_WARNING("rettrace: {} is not loaded", modName.c_str());
        while (*p && *p != ',')
          p++;
        continue;
      }
      base = mod->getInfo().base;
    }
    const char *where = modName.empty() ? "eboot" : modName.c_str();
    char *endp = nullptr;
    u64 off = std::strtoull(p, &endp, 16);
    const char *label = "ret";
    if (endp && *endp == ':') {
      const char *lb = endp + 1, *le = lb;
      while (*le && *le != ',')
        le++;
      label = strndup(lb, (size_t)(le - lb));  // persists for setRetTrace
      p = le;
    } else {
      p = endp;
    }
    auto *c = base + off;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x1000, utl::pageProtection::rwx);
    const bool isTest = c[0] == 0x85 && c[1] == 0xc0;
    if ((c[0] == 0x89 && c[1] == 0xc3) || isTest) {
      c[0] = 0xCC;
      setRetTrace(reinterpret_cast<uintptr_t>(c), label, isTest);
      LOG_INFO("rettrace: armed {} @ {}+{:#x}", label, where, off);
    } else {
      LOG_WARNING("rettrace: {}+{:#x} bytes {:#x} {:#x} is neither "
                  "mov ebx,eax nor test eax,eax", where, off, c[0], c[1]);
    }
  }
}

static void investigateFnWatch(smodule &m) {
  investigatePopcnt();
  investigateSumWatch();
  investigateWriteWatch();
  investigateWriteHist();
  investigatePoolMap();
  investigateMemDump();
  investigateRetTrace(*proc::getActive());
  const char *e = kFnWatch;
  if (!e)
    return;
  u8 *base = m.getInfo().base;
  for (const char *p = e; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    char *endp = nullptr;
    u64 off = std::strtoull(p, &endp, 0);
    const char *label = "fn";
    if (endp && *endp == ':') {
      const char *lb = endp + 1;
      const char *le = lb;
      while (*le && *le != ',')
        le++;
      label = strndup(lb, (size_t)(le - lb));  // persists for setFnWatch
      p = le;
    } else {
      p = endp;
    }
    auto *c = base + off;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x1000, utl::pageProtection::rwx);
    if (c[0] == 0x55) {
      c[0] = 0xCC;
      setFnWatch(reinterpret_cast<uintptr_t>(c), label);
      LOG_INFO("fnwatch: armed {} @ eboot+{:#x}", label, off);
    } else {
      LOG_WARNING("fnwatch: eboot+{:#x} first byte {:#x} != push rbp", off, c[0]);
    }
  }
  startFnWatchPrinter();
}

static void investigatePopcnt() {
  const char *pc = kGuestPopcnt;
  if (!pc)
    return;
  const uintptr_t at = std::strtoull(pc, nullptr, 16);
  const char *colon = std::strchr(pc, ':');
  size_t bytes = 0x1000;
  unsigned ms = 2000;
  if (colon) {
    bytes = std::strtoull(colon + 1, nullptr, 16);
    if (const char *c2 = std::strchr(colon + 1, ':'))
      ms = (unsigned)std::strtoul(c2 + 1, nullptr, 0);
  }
  startPopcntPrinter(at, bytes, ms);
}

static void investigateWriteWatch() {
  const bool reads = kGuestRprot != nullptr;
  const char *wp = reads ? kGuestRprot : kGuestWprot;
  if (!wp)
    return;
  const uintptr_t at = std::strtoull(wp, nullptr, 16);
  const char *colon = std::strchr(wp, ':');
  size_t bytes = 0x4000;
  unsigned ms = 200;
  if (colon) {
    bytes = std::strtoull(colon + 1, nullptr, 16);
    if (const char *c2 = std::strchr(colon + 1, ':'))
      ms = (unsigned)std::strtoul(c2 + 1, nullptr, 0);
  }
  startWriteWatch(at, bytes, ms, reads, false);
}

static void investigateWriteHist() {
  const char *wh = kGuestWhist;
  if (!wh)
    return;
  const uintptr_t at = std::strtoull(wh, nullptr, 16);
  const char *colon = std::strchr(wh, ':');
  size_t bytes = 0x1000000;
  unsigned ms = 500;
  if (colon) {
    bytes = std::strtoull(colon + 1, nullptr, 16);
    if (const char *c2 = std::strchr(colon + 1, ':'))
      ms = (unsigned)std::strtoul(c2 + 1, nullptr, 0);
  }
  startWriteHist(at, bytes, ms);
}

static void investigatePoolMap() {
  const char *pm = kPoolMap;
  if (!pm)
    return;
  const char *colon = std::strchr(pm, ':');
  if (std::strncmp(pm, "all", 3) == 0) {
    startPoolCensus(colon ? (unsigned)std::strtoul(colon + 1, nullptr, 0)
                          : 10000);
    return;
  }
  const uintptr_t at = std::strtoull(pm, nullptr, 16);
  size_t bytes = 0x1000000;
  unsigned ms = 10000;
  if (colon) {
    bytes = std::strtoull(colon + 1, nullptr, 16);
    if (const char *c2 = std::strchr(colon + 1, ':'))
      ms = (unsigned)std::strtoul(c2 + 1, nullptr, 0);
  }
  startPoolMap(at, bytes, ms);
}

static void investigateMemDump() {
  const char *e = kMemDump;
  if (!e)
    return;
  std::string list(e);
  size_t start = 0;
  while (start < list.size()) {
    size_t comma = list.find(',', start);
    std::string spec = list.substr(start, comma == std::string::npos
                                              ? std::string::npos
                                              : comma - start);
    uintptr_t at = 0;
    size_t bytes = 0;
    unsigned ms = 0;
    size_t p1 = spec.find(':'), p2 = spec.find(':', p1 + 1),
           p3 = spec.find(':', p2 + 1);
    if (p3 != std::string::npos) {
      at = std::strtoull(spec.c_str(), nullptr, 16);
      bytes = std::strtoull(spec.c_str() + p1 + 1, nullptr, 16);
      ms = (unsigned)std::strtoul(spec.c_str() + p2 + 1, nullptr, 0);
      startMemDump(at, bytes, ms, spec.c_str() + p3 + 1);
    }
    if (comma == std::string::npos)
      break;
    start = comma + 1;
  }
}

static void investigateSumWatch() {
  const char *sw = kGuestSumwatch;
  if (!sw)
    return;
  u64 f[5] = {0, 0, 0, 0, 1000};
  const char *p = sw;
  for (int i = 0; i < 5 && p && *p; i++) {
    f[i] = std::strtoull(p, nullptr, i == 3 ? 10 : 16);
    p = std::strchr(p, ':');
    if (p)
      p++;
  }
  startSumWatchPrinter((uintptr_t)f[0], (size_t)f[1], (size_t)f[2], (int)f[3],
                       (unsigned)f[4]);
}

// DELTA_GUEST_PATCH="hexoff=hexbytes,...": overwrite guest code/data at an eboot
// offset. For bisecting a wedge: NOP out a poll and see whether the thread behind
// it is the only thing blocked, without waiting for the real fix.
static void applyGuestPatches(smodule &m) {
  const char *e = kGuestPatch;
  if (!e)
    return;
  u8 *base = m.getInfo().base;
  for (const char *p = e; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    char *endp = nullptr;
    u64 off = std::strtoull(p, &endp, 16);
    if (!endp || *endp != '=')
      break;
    const char *h = endp + 1;
    u8 bytes[32];
    int n = 0;
    while (n < 32 && std::isxdigit((unsigned char)h[0]) &&
           std::isxdigit((unsigned char)h[1])) {
      bytes[n++] = (u8)std::strtoul(base::String(h, 2).c_str(), nullptr, 16);
      h += 2;
    }
    p = h;
    auto *c = base + off;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x2000, utl::pageProtection::rwx);
    std::memcpy(c, bytes, (size_t)n);
    LOG_INFO("guestpatch: {} byte(s) at eboot+{:#x}", n, off);
  }
}

// DELTA_FNARGS="hexoff[+o1+o2...]:label,...": arm an int3 at each guest function
// entry (first byte must be push rbp) that logs rdi and walks the offset chain
// from it. See setFnArgs in crash.h.
static void investigateFnArgs(smodule &m) {
  const char *e = kFnArgs;
  if (!e)
    return;
  u8 *base = m.getInfo().base;
  for (const char *p = e; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    char *endp = nullptr;
    u64 off = std::strtoull(p, &endp, 16);
    u64 offs[8];
    int noffs = 0;
    while (endp && *endp == '+' && noffs < 8)
      offs[noffs++] = std::strtoull(endp + 1, &endp, 16);
    const char *label = "fn";
    if (endp && *endp == ':') {
      const char *lb = endp + 1, *le = lb;
      while (*le && *le != ',')
        le++;
      label = strndup(lb, (size_t)(le - lb));  // persists for setFnArgs
      p = le;
    } else {
      p = endp;
    }
    auto *c = base + off;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x1000, utl::pageProtection::rwx);
    if (c[0] == 0x55) {
      c[0] = 0xCC;
      setFnArgs(reinterpret_cast<uintptr_t>(c), label, offs, noffs);
      LOG_INFO("fnargs: armed {} @ eboot+{:#x} ({} offsets)", label, off, noffs);
    } else {
      LOG_WARNING("fnargs: eboot+{:#x} first byte {:#x} != push rbp", off, c[0]);
    }
  }
}

// DELTA_SOTC_FORCE_PAYLOAD: experiment to get past LoadInitialWorld. The SotC
// world-container's whole-file libSceFios2 read returns actualCount 0 (root cause
// still open), so the loader's payload accessor at eboot+0x14c000 --
//   xor eax,eax; cmp [rdi+0x80],0; je +0x9 (return null); mov rax,[rdi+0x90]; ret
// -- returns NULL, FinalizeResource commits result=0, CommitResult re-enqueues
// forever and the game-logic thread wedges polling [op+0x98]==0xb. The buffer at
// [ldr+0x90] IS allocated (before the read), so NOP the `je 74 07` -> `90 90` and
// the accessor always returns that buffer; an empty precache container should
// parse as 0 entries and let boot proceed to the title/main menu. This is a
// runtime patching EXPERIMENT (env-gated, off by default; not a shipped fix).
static void forceSotcPayload(smodule &m) {
  u8 *base = m.getInfo().base;
  auto rwx = [](u8 *p) {
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(p) & ~0xFFFull),
                    0x1000, utl::pageProtection::rwx);
  };
  if (kSotcForcePayload) {
    u8 *je = base + 0x14c00a;  // `74 07` je +9 (return null) in accessor 0x14c000
    rwx(je);
    if (je[0] == 0x74 && je[1] == 0x07) {
      je[0] = 0x90; je[1] = 0x90;
      LOG_INFO("sotc force-payload: NOPed payload-null je @ eboot+0x14c00a");
    } else {
      LOG_WARNING("sotc force-payload: unexpected bytes {:#x} {:#x}", je[0], je[1]);
    }
  }
  // DELTA_SOTC_SKIP_WORLDWAIT: force CGame::LoadInitialWorld's busy-poll loop
  // (0x3c54e0..0x3c55fd) to EXIT immediately instead of spinning until the world-
  // container op reaches state 0xb (which never happens -- FHGetSize=0 on the
  // container, infinite retry). The loop tail `0f84 ddfeffff` (je 0x3c54e0 = "not
  // done -> loop") is NOPed (6x 0x90) so it falls through to the epilogue 0x3c5603
  // and returns to the boot driver at 0x25c1ff, which already tolerates a NULL
  // result (`test rbx,rbx; je 0x25c253` skips world-registration). Lets boot
  // proceed past the wedge to the title/menu, skipping the (broken) precache.
  // Runtime patching EXPERIMENT (env-gated, off by default).
  if (kSotcSkipWorldwait) {
    u8 *je = base + 0x3c55fd;
    rwx(je);
    if (je[0] == 0x0f && je[1] == 0x84) {
      for (int i = 0; i < 6; i++) je[i] = 0x90;
      LOG_INFO("sotc skip-worldwait: NOPed LoadInitialWorld poll-loop je @ eboot+0x3c55fd");
    } else {
      LOG_WARNING("sotc skip-worldwait: unexpected bytes {:#x} {:#x}", je[0], je[1]);
    }
  }
  // DELTA_SOTC_FORCE_WORLDDONE: force the world-container async op to read as
  // COMPLETE so the main-loop gate passes and boot proceeds to the title/menu,
  // skipping the (broken, FHGetSize=0) precache. Two byte-patches:
  //  (1) CommitResult 0x14d768 `je 0x14d777` (result==0 -> retry) NOPed so a
  //      null-payload commit always writes op-state 0xb (0x14d76a) instead of
  //      re-enqueuing forever;
  //  (2) IsDone 0x14cf0e `setne al` (needs op+0x20 result != 0) -> `mov al,1` so
  //      IsDone returns done on state==0xb alone (the op has no result object).
  // The boot continuation at 0x25c1ff already tolerates a NULL result
  // (`test rbx,rbx; je 0x25c253`). Runtime patch EXPERIMENT, env-gated.
  // DELTA_SOTC_JOBFIX: the world-load hang is a BPE-JobSystem livelock. The
  // world-op finalize job is DIRECT-ASSIGNED to worker ordinal 6 (the "Resource
  // Loading" coordinator, hardcoded-pinned to core 6, mask 0x40 -- outside our
  // 6-core cpuset), but SotC spawns only 4 job-claim workers with ordinals 0..3;
  // none services direct-assign slot 6 and the coordinator thread itself parks on
  // its evf "job done" flag, so the job never runs and the main loop spins the
  // loading screen forever (proven live: [jobclaim] directAssign[6]=0x1, workers
  // ordinal 0..3, ~99.4% claim failures). The ordinal comes from fn 0x33350
  // (reads [tcb-0x10]=0x8000|core, returns core or -1). Clamp its result into the
  // worker range [0,3]: the coordinator (ord 6 -> 6&3=2) then direct-assigns to a
  // serviced slot, worker 2 claims+runs the finalize, the op reaches state 0xb and
  // the game advances. Workers (0..3) and unbound threads (-1) are unchanged.
  // Patch the fn tail 0x3337f (`pop rbp; ret` + pad) in place:
  //   test eax,eax; js .r; cmp eax,4; jb .r; and eax,3; .r: pop rbp; ret
  if (kSotcJobfix) {
    u8 *t = base + 0x3337f;
    rwx(t);
    if (t[0] == 0x5d && t[1] == 0xc3) {
      static const u8 code[] = {0x85, 0xc0,             // test eax,eax
                                     0x78, 0x08,             // js .r (+8 -> pop)
                                     0x83, 0xf8, 0x04,       // cmp eax,4
                                     0x72, 0x03,             // jb .r (+3 -> pop)
                                     0x83, 0xe0, 0x03,       // and eax,3
                                     0x5d, 0xc3};            // .r: pop rbp; ret
      std::memcpy(t, code, sizeof(code));
      t[14] = 0x90; t[15] = 0x90;
      LOG_INFO("sotc jobfix: clamped worker-ordinal fn 0x33350 to [0,3] "
               "(core-6 coordinator's direct-assign now serviced)");
    } else {
      LOG_WARNING("sotc jobfix: unexpected bytes at 0x3337f {:#x} {:#x}", t[0], t[1]);
    }
  }
  if (kSotcForceWorlddone) {
    u8 *je = base + 0x14d768;   // 74 0d  je 0x14d777 (retry)
    u8 *sn = base + 0x14cf0e;   // 0f 95 c0  setne al
    rwx(je); rwx(sn);
    bool ok = true;
    if (je[0] == 0x74 && je[1] == 0x0d) { je[0] = 0x90; je[1] = 0x90; }
    else { LOG_WARNING("sotc force-worlddone: retry je bytes {:#x} {:#x}", je[0], je[1]); ok = false; }
    if (sn[0] == 0x0f && sn[1] == 0x95 && sn[2] == 0xc0) { sn[0] = 0xb0; sn[1] = 0x01; sn[2] = 0x90; }
    else { LOG_WARNING("sotc force-worlddone: setne bytes {:#x} {:#x} {:#x}", sn[0], sn[1], sn[2]); ok = false; }
    if (ok)
      LOG_INFO("sotc force-worlddone: patched CommitResult retry->0xb + IsDone always-done");
  }
}

// ===========================================================================
// DELTA_FIOS_TRACE: ARM-compatible guest-function trace of libSceFios2's
// FHOpen/FHGetSize/FHRead/FHPread (the whole-file API the SotC world-container
// loads through). int3 hooks are x86-host-only and inert under FEX on aarch64;
// this uses cpu::makeGuestReturnHook to WRAP each import via a guest x86
// trampoline that runs the real PRX export and reports its args AND return.
//
// Wrapping happens at IMPORT-RESOLUTION time (maybeWrapFiosImport, called from
// smodule::resolveImports): the eboot's PLT jump-slots are lazy and unresolved
// until the guest runs sys_dynlib_process_needed_and_relocate, so a GOT patch at
// proc::create is too early (the slot still holds the PLT stub, and eager
// resolution would overwrite it). At resolveImports the REAL export address is in
// hand; we substitute the wrapper for it before it is written to the GOT, so the
// hook is installed exactly when the slot is bound and stays for the whole run.
// The smoking gun is FHGetSize returning 0 for "$/misc/****/mainMenuPrecacheList
// .calt" while the 18k sibling members size non-zero.
// ===========================================================================
namespace {
struct FiosOpen {          // one FHOpen (all opens tracked so any fh maps to a path)
  u64 pOutFH;         // guest ptr the async open writes the SceFiosFH into
  std::string path;
};
std::mutex g_fiosMx;
std::vector<FiosOpen> g_fiosOpens;          // all opens, for fh->path reverse lookup
std::set<u64> g_zeroSizeFh;            // FHGetSize==0 fh's already reported
std::set<u64> g_zeroActOp;             // OpGetActualCount==0 ops already reported
std::set<std::string> g_seenPaths;          // DELTA_FIOS_ALLOPEN: dedup full open list
std::atomic<u64> g_fiosOpenN{0}, g_zeroSizeChurn{0}, g_zeroActChurn{0};

// Safe-ish read of a guest C string (guest memory is identity-mapped to host).
const char *guestStr(u64 va, char *buf, size_t cap) {
  if (!va || va < 0x10000) { buf[0] = 0; return buf; }
  const char *s = reinterpret_cast<const char *>(va);
  size_t i = 0;
  for (; i < cap - 1; ++i) { char c = s[i]; if (!c) break; buf[i] = c; }
  buf[i] = 0;
  return buf;
}

// Reverse-map an SceFiosFH to the path that opened it (newest first). Only ever
// called on the RARE zero-size/zero-count anomaly, so the O(n) scan is fine.
// Caller holds g_fiosMx.
std::string pathForFh(u64 fh) {
  if (!fh) return "<null-fh>";
  for (auto it = g_fiosOpens.rbegin(); it != g_fiosOpens.rend(); ++it) {
    u64 v = it->pOutFH ? *reinterpret_cast<u64 *>(it->pOutFH) : 0;
    if (v && v == fh) return it->path;
  }
  return "<unknown-fh>";
}

// The native logger the guest wrapper calls AFTER the real FIOS2 fn returns.
// hookId: 1=FHOpen 2=FHGetSize 3=FHRead 4=FHPread 5=OpGetActualCount.
//  FHOpen           : a1=pOutFH  a2=path   ret=op
//  FHGetSize        : a0=fh                ret=size          <-- smoking gun
//  FHRead/FHPread   : a1=fh a2=buf a3=len  ret=op
//  OpGetActualCount : a0=op                ret=actualCount   <-- smoking gun
// Detection is path-agnostic: a size or count of 0 is the anomaly; we then map
// the fh back to the file that opened it. Zero events are deduped per-fh/op so
// the post-drain retry churn (millions of calls) can't flood the log.
void PS4ABI fiosTraceLogger(u64 hookId, u64 a0, u64 a1,
                            u64 a2, u64 a3, u64 ret) {
  char pb[512];
  switch (hookId) {
  case 1: { // FHOpen
    u64 n = ++g_fiosOpenN;
    // Run the DELTA_FIOS_PROBE once, after enough opens that PlayGo chunks for
    // /app0/misc and /app0/scripts are mounted (the guest opens /app0/misc files
    // by open ~#905). Firing at proc::create or the first open is too early.
    if (n == 2000) {
      static std::once_flag probeOnce;
      std::call_once(probeOnce, [] { probeFiosPaths(); });
    }
    const char *path = guestStr(a2, pb, sizeof(pb));
    bool firstSeen = false;
    { std::lock_guard lk(g_fiosMx);
      if (g_fiosOpens.size() < 80000) g_fiosOpens.push_back({a1, std::string(path)});
      if (kFiosAllopen) firstSeen = g_seenPaths.insert(std::string(path)).second; }
    // DELTA_FIOS_ALLOPEN: log every DISTINCT path once (full file inventory, to
    // find whether the world-op file is ever even opened). Else sample first 40 +
    // container types.
    if (firstSeen || n <= 40 || std::strstr(path, ".calt") ||
        std::strstr(path, "recacheList") || std::strstr(path, "PrecacheList")) {
      BASE_LOGI("fios", "FHOpen path='{}' pOutFH={:#x} op={:#x} (#{})", path,
                (unsigned long long)a1, (unsigned long long)ret,
                (unsigned long long)n);
    }
    break;
  }
  case 2: { // FHGetSize(fh) -> size ; ANY zero/neg is the anomaly
    if ((i64)ret <= 0) {
      std::lock_guard lk(g_fiosMx);
      if (g_zeroSizeFh.insert(a0).second) {
        BASE_LOGI("fios",
                  "*** FHGetSize ZERO fh={:#x} -> size={}  path='{}'",
                  (unsigned long long)a0, (long long)ret, pathForFh(a0).c_str());
      } else if ((++g_zeroSizeChurn % 500000) == 0) {
        BASE_LOGI("fios", "(FHGetSize-zero churn count={})",
                  (unsigned long long)g_zeroSizeChurn.load());
      }
    } else if (kFiosAllopen && (i64)ret > (4 << 20)) {
      // Large files (>4MB) are candidates for the world container; log once/fh.
      std::lock_guard lk(g_fiosMx);
      if (g_zeroSizeFh.insert(a0 ^ 0x5A5A5A5Aull).second)
        BASE_LOGI("fios", "FHGetSize BIG fh={:#x} -> size={} path='{}'",
                  (unsigned long long)a0, (long long)ret,
                  pathForFh(a0).c_str());
    }
    break;
  }
  case 3:
  case 4: { // FHRead / FHPread ; log zero-length reads (the container's read)
    if (a3 == 0) {
      std::lock_guard lk(g_fiosMx);
      BASE_LOGI("fios",
                "*** {} ZERO-LEN fh={:#x} buf={:#x} len=0 op={:#x} path='{}'",
                hookId == 3 ? "FHRead" : "FHPread", (unsigned long long)a1,
                (unsigned long long)a2, (unsigned long long)ret,
                pathForFh(a1).c_str());
    }
    break;
  }
  case 5: { // OpGetActualCount(op) -> count ; ANY zero/neg is the anomaly
    if ((i64)ret <= 0) {
      std::lock_guard lk(g_fiosMx);
      if (g_zeroActOp.insert(a0).second) {
        BASE_LOGI("fios", "*** OpGetActualCount ZERO op={:#x} -> count={}",
                  (unsigned long long)a0, (long long)ret);
      } else if ((++g_zeroActChurn % 500000) == 0) {
        BASE_LOGI("fios", "(OpGetActualCount-zero churn count={})",
                  (unsigned long long)g_zeroActChurn.load());
      }
    }
    break;
  }
  default: break;
  }
}
} // namespace

// Called from smodule::resolveImports for every PLT import. If DELTA_FIOS_TRACE
// is set and `nidName` (an encoded "NID#lib#mod") is one of the libSceFios2
// whole-file APIs, return a guest wrapper around the resolved `realAddr` that
// logs args+return; otherwise return realAddr unchanged. NID prefixes from the
// SCE dynamic tables (report §10.2), cryptographically verified there.
uintptr_t maybeWrapFiosImport(const char *nidName, uintptr_t realAddr) {
  if (!kFiosTrace || !nidName || !realAddr)
    return realAddr;
  struct { const char *nid; u32 hookId; const char *nm; } tbl[] = {
      {"er6TkQFUvp0", 1, "sceFiosFHOpen"},
      {"FdjoqFQOlt0", 2, "sceFiosFHGetSize"},
      {"cg-VoPqZYss", 3, "sceFiosFHRead"},
      {"rR8wq7YFRZs", 4, "sceFiosFHPread"},
      {"+FRvKknUj1I", 5, "sceFiosOpGetActualCount"},
  };
  for (auto &e : tbl) {
    if (std::strncmp(nidName, e.nid, 11) == 0) {
      uintptr_t wrap = cpu::makeGuestReturnHook(reinterpret_cast<void *>(realAddr),
                                                e.hookId,
                                                reinterpret_cast<void *>(&fiosTraceLogger),
                                                e.nm);
      if (wrap) {
        LOG_INFO("fiostrace: wrapped {} real={:#x} -> wrapper={:#x}", e.nm,
                 (unsigned long)realAddr, (unsigned long)wrap);
        return wrap;
      }
      LOG_WARNING("fiostrace: makeGuestReturnHook failed for {}", e.nm);
      return realAddr;
    }
  }
  return realAddr;
}

// DELTA_FIOS_PROBE=path1,path2,...  Boot-time probe: resolve each guest path
// through the VFS exactly as the guest would and log Exists/size. Lets us read
// the world-op container's backing WITHOUT waiting ~50min for the JobSystem to
// schedule its (low-priority, retry-churning) load job. Paths are guest paths
// like "/app0/misc/_cmn/mainmenuprecachelist.calt" (FIOS2 lowercases; $ -> /app0).
static void probeFiosPaths() {
  const char *e = kFiosProbe;
  if (!e)
    return;
  std::string list(e);
  size_t start = 0;
  while (start < list.size()) {
    size_t comma = list.find(',', start);
    std::string path = list.substr(start, comma == std::string::npos
                                              ? std::string::npos
                                              : comma - start);
    if (!path.empty()) {
      utl::File f = vfs::openRead(path.c_str());
       if (f.Exists())
         BASE_LOGI("fiosprobe", "'{}' EXISTS size={}", path.c_str(),
                   (unsigned long long)f.GetSize());
       else
         BASE_LOGI("fiosprobe", "'{}' MISSING (openRead empty)", path.c_str());
    }
    if (comma == std::string::npos)
      break;
    start = comma + 1;
  }
}

// ===========================================================================
// DELTA_JOB_TRACE: instrument SotC's BPE JobSystem to find why the 4 workers
// livelock post-drain, unable to claim the final world-op finalize job. Uses an
// INTERNAL-function entry detour (not GOT-based, since these are eboot-internal
// calls): overwrite the target's prologue with an abs jmp to a return-capturing
// wrapper whose realTarget is a trampoline (relocated prologue + jmp back). The
// wrapper reports args+return via the magic-syscall to jobTraceLogger.
//   claim 0x38d40: worker ordinal = fs:[-8]; ret = claimed job (0 = fail)
//   kick  0x35480: job affinity/prio submitted
// ===========================================================================

// ---- the interface kern calls ---------------------------------------------

void onProcessCreated(proc &p, smodule &mainModule, bool ps5) {
  (void)p;
  smodule *first = &mainModule;
  // Engine bring-up: give Isaac's surface-name registry valid empty storage so
  // main-init doesn't deref a null bucket array (self-gated by ctor signature).
    // DELTA_GUEST_NULLGUARD="<hexoff>:<rax|rsi>:<len>[,...]": recover a guest
  // deref of a bad pointer by zeroing the destination register and stepping
  // over the instruction. For an allocator that faults instead of reporting
  // "no space", this is how you find out whether the engine has an
  // out-of-memory path at all.
  if (const char *ng = kNullGuard) {
    for (const char *p = ng; *p;) {
      char *endp = nullptr;
      const u64 off = std::strtoull(p, &endp, 16);
      if (!endp || *endp != ':')
        break;
      const char *r = endp + 1;
      const krnl::GuardReg reg =
          (*r == 's') ? krnl::GuardReg::rsi : krnl::GuardReg::rax;
      const char *c = std::strchr(r, ':');
      if (!c)
        break;
      const int len = (int)std::strtol(c + 1, const_cast<char **>(&p), 10);
      krnl::setNullGuard(reinterpret_cast<uintptr_t>(first->getInfo().base) + off, reg, len);
      LOG_INFO("nullguard: armed eboot+{:#x} len={}", off, len);
      while (*p == ',' || *p == ' ')
        p++;
    }
  }
  if (ps5) {
    bringUpRebirthEbootRegistry(*first);
    investigateDcbGate(*first);
    // DELTA_PS5_GLYPHGUARD: recover the first-frame unbound-font null derefs in
    // the UI/text renderer so the render reaches real draws (diagnostic; the real
    // fix binds the font before rendererFrame).
    if (kPs5Glyphguard) {
      auto *base8 = first->getInfo().base;
      auto eb = reinterpret_cast<uintptr_t>(base8);
      // movzx esi,[rdi+rcx*2+0x2e] (glyph cmap count), rdi==0
      krnl::setNullGuard(eb + 0x5cab56, krnl::GuardReg::rsi, 5);
      // mov rax,[rax+0x28]; mov rax,[rax+0x18] (chained font-object load), rax==0
      krnl::setNullGuard(eb + 0x5c7c53, krnl::GuardReg::rax, 8);
      // ROOT FIX: the renderer-init chain 0x5535d0 bails at its gate checks
      // (`test al,al; je 0x55365d`) when VOInit (gate C, 0x58fb10) returns false
      // -- a GPU render-context vtable step that fails in our env -- SKIPPING the
      // Shape-Renderer install at 0x55361b (0x58ec90). That leaves the global
      // active renderer *(0x9854f0) null, which is the source of the whole
      // first-frame null-object cascade. Force the chain past its three bail
      // branches so the game installs the renderer + builds its RTs/fonts itself.
      // DELTA_PS5_NOFORCE: skip the RenderInit gate force-through. Now that the PS5
      // videoout NIDs are HLE'd (RegisterBuffers returns 0), VOInit (gate C) should
      // return TRUE on its own -- forcing past it leaves an INVALID render context
      // (null pipelines / zero shader PGM). Test whether it succeeds naturally.
      struct { u32 off; u8 b1; } gates[] = {
          {0x553602, 0x59}, {0x553612, 0x49}, {0x553622, 0x39}};
      bool noForce = kPs5Noforce;
      for (auto &g : gates) {
        if (noForce) break;
        u8 *c = base8 + g.off;
        if (c[0] == 0x74 && c[1] == g.b1) {  // je 0x55365d
          utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                          0x1000, utl::pageProtection::rwx);
          c[0] = 0x90;  // NOP the bail so the chain runs the Shape-Renderer install
          c[1] = 0x90;
        }
      }
      LOG_INFO("ps5 glyphguard: forced renderer-init chain through the Shape-Renderer install");
    }
  }
  // Generic guest-function hit counter (works for PS4 and PS5 eboots).
  investigateFnWatch(*first);
  investigateFnArgs(*first);
  applyGuestPatches(*first);
  forceSotcPayload(*first);
  installJobTrace(*first);
  installMatTrace(*first);
  installAllocLock(*first);
  // Note: DELTA_FIOS_PROBE runs lazily on the first FIOS2 call (see
  // fiosTraceLogger); at proc::create the /app0 PFS provider isn't mounted yet.
}

void onModuleLoaded(smodule &m, base::StringRef name) {
  if (name == base::StringRef("rebirth"))
    bringUpRebirthSurfaceRegistry(m);
  else if (name == base::StringRef("libSceVideoOut"))
    patchVideoOutDiag(m);
}

void onBeforeStart(proc &p) {
  applyBootPatches(p);
}

uintptr_t wrapImport(const char *nidName, uintptr_t realAddr) {
  return maybeWrapFiosImport(nidName, realAddr);
}

}  // namespace krnl::probe
