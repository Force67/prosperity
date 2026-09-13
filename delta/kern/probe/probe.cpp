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

// DELTA_PS5_DCBWATCH: diagnose the null frame-0 DrawCommandBuffer gate: poll the DCB
// pointer slot (manager[0] @ eboot+0x985a00) + adjacent fields, logging every change
// (is the DCB ever created before the render uses it?), plus an int3 call-order trace
// over the renderer/DCB-creation entry points. DELTA_FNWATCH="hexoff:label,...":
// int3 hit-counters at guest function entries (first byte push rbp), eboot-relative;
// the handler counts and a thread prints totals every 2s (see crash.h).
static void investigatePopcnt();
static void investigateSumWatch();
static void investigateWriteWatch();
static void investigateWriteHist();
static void investigatePoolMap();
static void investigateMemDump();
static void investigateRetTrace(proc &);

// DELTA_RETTRACE="[module+]hexoff:label,...": return-value trace at each
// `mov ebx,eax` / `test eax,eax` after a call, eboot-relative unless a module is
// named, so a failure can be followed into the system module reporting it.
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

// DELTA_SOTC_FORCE_PAYLOAD: experiment past LoadInitialWorld. The world-container's
// whole-file FIOS2 read returns actualCount 0 (root cause open), so the payload
// accessor at eboot+0x14c000 (cmp [rdi+0x80],0; je -> null) returns NULL and
// CommitResult re-enqueues forever, wedging the game thread on [op+0x98]==0xb.
// [ldr+0x90] IS allocated pre-read, so NOP the `je 74 07` -> `90 90`: the accessor
// always returns that buffer; an empty container should parse as 0 entries. Env-
// gated EXPERIMENT, not a shipped fix.
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
  // DELTA_SOTC_SKIP_WORLDWAIT: make CGame::LoadInitialWorld's busy-poll (0x3c54e0..)
  // exit at once instead of spinning on a world-op that never reaches 0xb (FHGetSize=0).
  // NOP the loop-tail `je 0x3c54e0` (6x 0x90) so it falls into the epilogue and returns
  // to 0x25c1ff, which tolerates a NULL result. Env-gated EXPERIMENT.
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
  // DELTA_SOTC_FORCE_WORLDDONE: force the world-container async op COMPLETE so boot
  // reaches the title, skipping the broken precache. (1) CommitResult 0x14d768
  // `je 0x14d777` NOPed so a null-payload commit writes op-state 0xb instead of
  // re-enqueuing; (2) IsDone 0x14cf0e `setne al` -> `mov al,1` (done on 0xb alone).
  // The continuation at 0x25c1ff tolerates NULL. Env-gated EXPERIMENT.
  // DELTA_SOTC_JOBFIX: the hang is a BPE-JobSystem livelock; the world-op finalize
  // job is DIRECT-ASSIGNED to ordinal 6 (the coordinator, pinned to core 6, mask 0x40,
  // outside our 6-core cpuset) while only workers 0..3 exist, so slot 6 is never
  // serviced and the coordinator parks forever (proven live: directAssign[6]=1,
  // ~99.4% claim failures). Ordinal from fn 0x33350 ([tcb-0x10]=0x8000|core). Clamp
  // its result to [0,3] (patch tail 0x3337f: test/js/cmp 4/jb/and 3/ret) so the
  // coordinator assigns to a serviced slot; workers and unbound (-1) unchanged.
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
// DELTA_FIOS_TRACE: ARM-compatible guest trace of libSceFios2's
// FHOpen/FHGetSize/FHRead/FHPread (the whole-file API the SotC world-container
// loads through); int3 hooks are inert under FEX, so each import is WRAPPED via
// cpu::makeGuestReturnHook. Wrapping happens at IMPORT-RESOLUTION time
// (maybeWrapFiosImport from resolveImports): the PLT slots are lazy, so a GOT
// patch at proc::create is too early; at resolveImports the real export address is
// in hand and substituting the wrapper installs the hook exactly when bound.
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

// Native logger called after the real FIOS2 fn returns. hookId: 1=FHOpen 2=FHGetSize
// 3=FHRead 4=FHPread 5=OpGetActualCount; FHGetSize ret = size, OpGetActualCount
// ret = actualCount (the smoking guns); FHOpen a1=pOutFH a2=path ret=op;
// FHRead/Pread a1=fh a2=buf a3=len ret=op. Path-agnostic detection: a 0 size/count
// is the anomaly, mapped back to the opener. Zero events deduped per fh/op so the
// post-drain retry churn can't flood the log.
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

// Called from resolveImports per PLT import: if DELTA_FIOS_TRACE is set and nidName
// ("NID#lib#mod") is one of the libSceFios2 whole-file APIs, return a logging
// wrapper around realAddr, else unchanged.
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

// DELTA_FIOS_PROBE=path1,...: resolve each guest path through the VFS as the guest
// would and log Exists/size, reading the world-op container's backing without
// waiting ~50min for the retry-churning load job (FIOS2 lowercases; $ -> /app0).
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
// DELTA_JOB_TRACE: instrument SotC's BPE JobSystem (why do the 4 workers livelock
// post-drain?). INTERNAL-function entry detours (not GOT-based): overwrite the
// prologue with an abs jmp to a return-capturing wrapper whose realTarget is a
// trampoline (relocated prologue + jmp back); args+return go to jobTraceLogger
// via the magic syscall. claim 0x38d40: ordinal = fs:[-8], ret = claimed job;
// kick 0x35480: affinity/prio submitted.
// ===========================================================================

// ---- the interface kern calls ---------------------------------------------

void onProcessCreated(proc &p, smodule &mainModule, bool ps5) {
  (void)p;
  smodule *first = &mainModule;
  // Engine bring-up: give Isaac's surface-name registry valid empty storage so
  // main-init doesn't deref a null bucket array (self-gated by ctor signature).
  // DELTA_GUEST_NULLGUARD="<hexoff>:<rax|rsi>:<len>[,...]": recover a guest deref of
  // a bad pointer by zeroing the destination register and stepping over the
  // instruction; for an allocator that faults instead of reporting "no space", this
  // finds whether an out-of-memory path exists at all.
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
      // ROOT FIX: the renderer-init chain 0x5535d0 bails at its gates when VOInit
      // (gate C, 0x58fb10) returns false in our env, skipping the Shape-Renderer install
      // at 0x55361b and leaving *(0x9854f0) null, the source of the first-frame null-
      // object cascade. Force the chain past its three bail branches. DELTA_PS5_NOFORCE:
      // skip the force now that videoout NIDs are HLE'd (VOInit should return TRUE on
      // its own; forcing leaves an INVALID context with null pipelines).
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
