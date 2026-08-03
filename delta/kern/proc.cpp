/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <thread>
#include <chrono>
#include <base.h>
#include <utl/file.h>
#include <utl/mem.h>
#include <utl/path.h>

#include "crash.h"
#include "module.h"
#include "proc.h"

#include "ps5/lv2/ctor_probe.h"
#include "ps5/lv2/initial_tcb.h"
#include "vfs.h"
#include "cpu/cpu_backend.h"
#include "ps4/hardware_mode.h"
#include "ps4/lv2/sys_dynlib.h"
#include "ps4/lv2/sys_mem.h"
#include "runtime/vprx/vprx.h"

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>
#include <set>
#include <atomic>
#include <utl/options.h>

namespace {
DELTA_OPTION(const char *, kPs5Modules, "DELTA_PS5_MODULES", nullptr);
DELTA_OPTION(const char *, kFnWatch, "DELTA_FNWATCH", nullptr);
DELTA_OPTION(const char *, kRetTrace, "DELTA_RETTRACE", nullptr);
DELTA_OPTION(const char *, kGuestPatch, "DELTA_GUEST_PATCH", nullptr);
DELTA_OPTION(const char *, kFnArgs, "DELTA_FNARGS", nullptr);
DELTA_OPTION(const char *, kFiosProbe, "DELTA_FIOS_PROBE", nullptr);
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
DELTA_OPTION(bool, kFiosAllopen, "DELTA_FIOS_ALLOPEN", false);
DELTA_OPTION(bool, kFiosTrace, "DELTA_FIOS_TRACE", false);
DELTA_OPTION(bool, kGfxctxWatch, "DELTA_GFXCTX_WATCH", false);
DELTA_OPTION(bool, kJobTrace, "DELTA_JOB_TRACE", false);
DELTA_OPTION(bool, kJobTraceClaim, "DELTA_JOB_TRACE_CLAIM", false);
DELTA_OPTION(bool, kPs5Dcbwatch, "DELTA_PS5_DCBWATCH", false);
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
DELTA_OPTION(bool, kSotcJobmove, "DELTA_SOTC_JOBMOVE", false);
DELTA_OPTION(bool, kSotcSkipWorldwait, "DELTA_SOTC_SKIP_WORLDWAIT", false);
DELTA_OPTION(bool, kVoOplog, "DELTA_VO_OPLOG", false);
DELTA_OPTION(bool, kVoSkip580, "DELTA_VO_SKIP_580", false);
DELTA_OPTION(bool, kVoWatch, "DELTA_VO_WATCH", false);
}  // namespace

namespace krnl {
static proc *g_activeProc{nullptr};

// The guest fs base (TLS) and how the guest entry is run are backend-specific
// (see delta/cpu): native uses a host thread_local + direct call, FEX uses the
// FEXCore CPUState + JIT. setThreadFsBase() is defined by the active backend.

proc::proc() : vmem(env) { g_activeProc = this; }

proc *proc::getActive() { return g_activeProc; }

static void bringUpRebirthEbootRegistry(smodule &m);
static void investigateDcbGate(smodule &m);
static void investigateFnWatch(smodule &m);
static void investigateFnArgs(smodule &m);
static void applyGuestPatches(smodule &m);
static void forceSotcPayload(smodule &m);
static void probeFiosPaths();
static void installJobTrace(smodule &m);

bool proc::create(const base::String &path, bool fromVfs) {
  /*register HLE prx overrides*/
  runtime::vprx_init();

  /*init memory manager*/
  LOG_ASSERT(vmem.init());

  /*reserve slot for main module*/
  auto first = utl::make_ref<smodule>(this);
  first->getInfo().handle = 0;

  modules.emplace_back(first);

  const bool ps5 = plat == platform::ps5;
  if (ps5 && !kPs5Modules)
    LOG_WARNING("PS5 title but DELTA_PS5_MODULES is unset; system modules "
                "(libkernel etc.) won't be found");

  /*pre-load required modules
   (the kernel does it, so do we)*/
  if (!loadModule(base::StringRef("libkernel")) ||
      !loadModule(base::StringRef("libSceLibcInternal"))) {
    LOG_ERROR("unable to preload sys modules");
    return false;
  }

  // PS4 libkernel is a thin forwarder: a chunk of its exports (memory-pool
  // helpers) live in libkernel_sys. PS5 has no such split, so only preload it
  // for PS4. Tolerate absence on minimal module sets.
  if (!ps5)
    loadModule(base::StringRef("libkernel_sys"));

  bool loaded = fromVfs ? first->fromVfs(path) : first->fromFile(path);
  if (!loaded) {
    LOG_ERROR("unable to load main process module");
    return false;
  }
  // PS5 modules carry no DT_SCE_MODULEINFO, so name the main module ourselves.
  if (ps5 && first->getInfo().name.empty())
    first->getInfo().name = base::String("eboot");

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
      const uint64_t off = std::strtoull(p, &endp, 16);
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
      struct { uint32_t off; uint8_t b1; } gates[] = {
          {0x553602, 0x59}, {0x553612, 0x49}, {0x553622, 0x39}};
      bool noForce = kPs5Noforce;
      for (auto &g : gates) {
        if (noForce) break;
        uint8_t *c = base8 + g.off;
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
  // Note: DELTA_FIOS_PROBE runs lazily on the first FIOS2 call (see
  // fiosTraceLogger); at proc::create the /app0 PFS provider isn't mounted yet.

  return true;
}

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
    uint8_t *base = pr.getModuleList()[0]->getInfo().base;
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
    uint64_t off = std::strtoull(p, &endp, 16);
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
  uint8_t *base = m.getInfo().base;
  for (const char *p = e; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    char *endp = nullptr;
    uint64_t off = std::strtoull(p, &endp, 0);
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
  startWriteWatch(at, bytes, ms, reads);
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
  uint64_t f[5] = {0, 0, 0, 0, 1000};
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
  uint8_t *base = m.getInfo().base;
  for (const char *p = e; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    char *endp = nullptr;
    uint64_t off = std::strtoull(p, &endp, 16);
    if (!endp || *endp != '=')
      break;
    const char *h = endp + 1;
    uint8_t bytes[32];
    int n = 0;
    while (n < 32 && std::isxdigit((unsigned char)h[0]) &&
           std::isxdigit((unsigned char)h[1])) {
      bytes[n++] = (uint8_t)std::strtoul(base::String(h, 2).c_str(), nullptr, 16);
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
  uint8_t *base = m.getInfo().base;
  for (const char *p = e; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    char *endp = nullptr;
    uint64_t off = std::strtoull(p, &endp, 16);
    uint64_t offs[8];
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
  uint8_t *base = m.getInfo().base;
  auto rwx = [](uint8_t *p) {
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(p) & ~0xFFFull),
                    0x1000, utl::pageProtection::rwx);
  };
  if (kSotcForcePayload) {
    uint8_t *je = base + 0x14c00a;  // `74 07` je +9 (return null) in accessor 0x14c000
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
    uint8_t *je = base + 0x3c55fd;
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
    uint8_t *t = base + 0x3337f;
    rwx(t);
    if (t[0] == 0x5d && t[1] == 0xc3) {
      static const uint8_t code[] = {0x85, 0xc0,             // test eax,eax
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
    uint8_t *je = base + 0x14d768;   // 74 0d  je 0x14d777 (retry)
    uint8_t *sn = base + 0x14cf0e;   // 0f 95 c0  setne al
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
  uint64_t pOutFH;         // guest ptr the async open writes the SceFiosFH into
  std::string path;
};
std::mutex g_fiosMx;
std::vector<FiosOpen> g_fiosOpens;          // all opens, for fh->path reverse lookup
std::set<uint64_t> g_zeroSizeFh;            // FHGetSize==0 fh's already reported
std::set<uint64_t> g_zeroActOp;             // OpGetActualCount==0 ops already reported
std::set<std::string> g_seenPaths;          // DELTA_FIOS_ALLOPEN: dedup full open list
std::atomic<uint64_t> g_fiosOpenN{0}, g_zeroSizeChurn{0}, g_zeroActChurn{0};

// Safe-ish read of a guest C string (guest memory is identity-mapped to host).
const char *guestStr(uint64_t va, char *buf, size_t cap) {
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
std::string pathForFh(uint64_t fh) {
  if (!fh) return "<null-fh>";
  for (auto it = g_fiosOpens.rbegin(); it != g_fiosOpens.rend(); ++it) {
    uint64_t v = it->pOutFH ? *reinterpret_cast<uint64_t *>(it->pOutFH) : 0;
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
void PS4ABI fiosTraceLogger(uint64_t hookId, uint64_t a0, uint64_t a1,
                            uint64_t a2, uint64_t a3, uint64_t ret) {
  char pb[512];
  switch (hookId) {
  case 1: { // FHOpen
    uint64_t n = ++g_fiosOpenN;
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
      std::fprintf(stderr, "[fios] FHOpen path='%s' pOutFH=%#llx op=%#llx (#%llu)\n",
                   path, (unsigned long long)a1, (unsigned long long)ret,
                   (unsigned long long)n);
      std::fflush(stderr);
    }
    break;
  }
  case 2: { // FHGetSize(fh) -> size ; ANY zero/neg is the anomaly
    if ((int64_t)ret <= 0) {
      std::lock_guard lk(g_fiosMx);
      if (g_zeroSizeFh.insert(a0).second) {
        std::fprintf(stderr,
          "[fios] *** FHGetSize ZERO fh=%#llx -> size=%lld  path='%s'\n",
          (unsigned long long)a0, (long long)ret, pathForFh(a0).c_str());
        std::fflush(stderr);
      } else if ((++g_zeroSizeChurn % 500000) == 0) {
        std::fprintf(stderr, "[fios] (FHGetSize-zero churn count=%llu)\n",
                     (unsigned long long)g_zeroSizeChurn.load());
        std::fflush(stderr);
      }
    } else if (kFiosAllopen && (int64_t)ret > (4 << 20)) {
      // Large files (>4MB) are candidates for the world container; log once/fh.
      std::lock_guard lk(g_fiosMx);
      if (g_zeroSizeFh.insert(a0 ^ 0x5A5A5A5Aull).second)
        std::fprintf(stderr, "[fios] FHGetSize BIG fh=%#llx -> size=%lld path='%s'\n",
                     (unsigned long long)a0, (long long)ret, pathForFh(a0).c_str()),
        std::fflush(stderr);
    }
    break;
  }
  case 3:
  case 4: { // FHRead / FHPread ; log zero-length reads (the container's read)
    if (a3 == 0) {
      std::lock_guard lk(g_fiosMx);
      std::fprintf(stderr,
        "[fios] *** %s ZERO-LEN fh=%#llx buf=%#llx len=0 op=%#llx path='%s'\n",
        hookId == 3 ? "FHRead" : "FHPread", (unsigned long long)a1,
        (unsigned long long)a2, (unsigned long long)ret, pathForFh(a1).c_str());
      std::fflush(stderr);
    }
    break;
  }
  case 5: { // OpGetActualCount(op) -> count ; ANY zero/neg is the anomaly
    if ((int64_t)ret <= 0) {
      std::lock_guard lk(g_fiosMx);
      if (g_zeroActOp.insert(a0).second) {
        std::fprintf(stderr,
          "[fios] *** OpGetActualCount ZERO op=%#llx -> count=%lld\n",
          (unsigned long long)a0, (long long)ret);
        std::fflush(stderr);
      } else if ((++g_zeroActChurn % 500000) == 0) {
        std::fprintf(stderr, "[fios] (OpGetActualCount-zero churn count=%llu)\n",
                     (unsigned long long)g_zeroActChurn.load());
        std::fflush(stderr);
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
  struct { const char *nid; uint32_t hookId; const char *nm; } tbl[] = {
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
        std::fprintf(stderr, "[fiosprobe] '%s' EXISTS size=%llu\n", path.c_str(),
                     (unsigned long long)f.GetSize());
      else
        std::fprintf(stderr, "[fiosprobe] '%s' MISSING (openRead empty)\n",
                     path.c_str());
      std::fflush(stderr);
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
namespace {
std::atomic<uint64_t> g_jobClaims{0}, g_jobFails{0};

// Host-side JobSystem watcher. Spawned once with the guest jobsys base
// (identity-mapped, safe to read from a host thread). Everything expensive
// happens here, off the guest's hot paths.
void spawnJobWatcher(uint64_t base);

void PS4ABI jobTraceLogger(uint64_t hookId, uint64_t a0, uint64_t a1,
                           uint64_t a2, uint64_t a3, uint64_t ret) {
  switch (hookId) {
  case 12: { // CLAIM 0x38d40 -> ret = claimed job ptr (0 = failed to claim)
    // Log each worker's ordinal (fs:[-8]) exactly once, cheaply (thread_local).
    thread_local bool logged = false;
    if (!logged) {
      logged = true;
      // Replicate get-core-ordinal fn 0x33350: tcb = *(fsbase); v = [tcb-0x10];
      // ord = (v & 0x8000) ? (v & ~0x8000) : -1.  This is the value the claim
      // path (0x38d70/0x38d7f) uses for 1<<ord AND the direct-assign slot index.
      uint64_t fsb = cpu::currentGuestFsBase();
      uint64_t tcb = fsb ? *reinterpret_cast<uint64_t *>(fsb) : 0;
      uint64_t v = tcb ? *reinterpret_cast<uint64_t *>(tcb - 0x10) : 0;
      int64_t ord = (v & 0x8000) ? (int64_t)(v & ~0x8000ULL) : -1;
      static std::mutex m; std::lock_guard lk(m);
      std::fprintf(stderr,
        "[jobclaim] worker fsbase=%#llx tcb=%#llx [tcb-0x10]=%#llx -> ordinal=%lld "
        "(1<<ord=%#x) firstClaim->%#llx\n",
        (unsigned long long)fsb, (unsigned long long)tcb, (unsigned long long)v,
        (long long)ord, (ord >= 0 && ord < 32) ? (1u << (unsigned)ord) : 0u,
        (unsigned long long)ret);
      std::fflush(stderr);
    }
    ++g_jobClaims;
    if (ret == 0) ++g_jobFails;
    if (a0 > 0x10000)
      spawnJobWatcher(a0);
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
      std::fprintf(stderr, "[jobctor] JobSystem ctor this=%#llx\n",
                   (unsigned long long)a0);
      std::fflush(stderr);
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
void spawnJobWatcher(uint64_t base) {
  static std::atomic<uint64_t> once{0};
  uint64_t expect = 0;
  if (!once.compare_exchange_strong(expect, base))
    return;
  std::thread([base] {
        uint64_t prev[8] = {0};
        int persist = 0;
        for (;;) {
          std::this_thread::sleep_for(std::chrono::seconds(30));
          uint64_t slot[8];
          for (int i = 0; i < 8; i++)
            slot[i] = *reinterpret_cast<volatile uint64_t *>(base + (uint64_t)i * 0x120 + 0xeb0);
          std::fprintf(stderr,
                       "[jobwatch] claims=%llu fails=%llu directAssign[0..7]= "
                       "0:%#llx 1:%#llx 2:%#llx 3:%#llx 4:%#llx 5:%#llx 6:%#llx 7:%#llx\n",
                       (unsigned long long)g_jobClaims.load(), (unsigned long long)g_jobFails.load(),
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
            const uint64_t *q = reinterpret_cast<const uint64_t *>(base + (uint64_t)i * 0x120 + 0xea8);
            std::fprintf(stderr,
                         "[jobwatch]   block%d desc@+0xea8: %#llx %#llx %#llx %#llx %#llx %#llx %#llx %#llx\n",
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
            const uint64_t qbase =
                *reinterpret_cast<volatile uint64_t *>(base + 0x1478);
            auto mapped = [](uint64_t a) {
              return a > 0x10000 && a < (1ULL << 47);
            };
            if (mapped(qbase)) {
              for (int lvl = 0; lvl < 8; lvl++) {
                const uint64_t bucket = qbase + (uint64_t)lvl * 0x2078;
                const uint64_t acount =
                    *reinterpret_cast<volatile uint64_t *>(bucket + 0x2038);
                const uint64_t nodes =
                    *reinterpret_cast<volatile uint64_t *>(bucket);
                if (!acount && !nodes)
                  continue;
                std::fprintf(
                    stderr,
                    "[jobwatch] bucket L%d scanA-slots=%llu scanB-nodes=%llu\n",
                    lvl, (unsigned long long)acount, (unsigned long long)nodes);
                const uint64_t slots =
                    *reinterpret_cast<volatile uint64_t *>(bucket + 0x2070);
                if (acount && mapped(slots)) {
                  for (uint64_t i = 0; i < acount && i < 16; i++) {
                    const uint32_t mask = *reinterpret_cast<volatile uint32_t *>(
                        slots + i * 0x120 + 0xf0);
                    std::fprintf(stderr,
                                 "[jobwatch]   L%d slot%llu mask=%#x -> %s\n",
                                 lvl, (unsigned long long)i, mask,
                                 (mask & 0xf)
                                     ? "claimable"
                                     : "*** MASK EXCLUDES EVERY WORKER ***");
                  }
                }
                for (uint64_t i = 0; i < nodes && i < 32; i++) {
                  const uint64_t n = *reinterpret_cast<volatile uint64_t *>(
                      bucket + 0x38 + i * 8);
                  if (!mapped(n))
                    continue;
                  const uint64_t head = *reinterpret_cast<volatile uint64_t *>(n);
                  const uint64_t tail =
                      *reinterpret_cast<volatile uint64_t *>(n + 8);
                  if (head == tail)
                    continue;  // empty: the claim skips it at 0x3913e
                  const uint64_t dep =
                      *reinterpret_cast<volatile uint64_t *>(n + 0x980);
                  const uint64_t thresh =
                      *reinterpret_cast<volatile uint64_t *>(n + 0x988);
                  const uint32_t mode =
                      *reinterpret_cast<volatile uint32_t *>(n + 0x990);
                  const uint32_t aff =
                      *reinterpret_cast<volatile uint32_t *>(n + 0x998);
                  const int32_t prio =
                      *reinterpret_cast<volatile int32_t *>(n + 0x9a0);
                  long long depval = 0;
                  bool depread = false;
                  if (mapped(dep)) {
                    depval =
                        (long long)*reinterpret_cast<volatile uint64_t *>(dep);
                    depread = true;
                  }
                  const bool depok =
                      !dep || (depread && (mode == 0
                                               ? (uint64_t)depval == thresh
                                               : depval > (long long)thresh));
                  std::fprintf(
                      stderr,
                      "[jobwatch]   L%d node%llu @%#llx head=%#llx tail=%#llx "
                      "dep=%#llx *dep=%lld thresh=%lld mode=%u aff=%#x prio=%d "
                      "-> %s%s\n",
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
            std::vector<uint64_t> fsb;
            cpu::guestThreadFsBases(fsb);
            uint32_t present = 0;
            int valid = 0;
            std::string list;
            for (uint64_t f : fsb) {
              if (!f)
                continue;
              uint64_t tcb = *reinterpret_cast<volatile uint64_t *>(f);
              if (tcb < 0x10000)
                continue;
              uint64_t v = *reinterpret_cast<volatile uint64_t *>(tcb - 0x10);
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
            std::fprintf(stderr,
                         "[jobwatch] threads=%zu withOrdinal=%d ordinals={ %s} claimable-mask=%#x\n",
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
            uint64_t tbl = *reinterpret_cast<volatile uint64_t *>(base + 0xb60);
            int shown = 0;
            if (tbl > 0x10000) {
              for (int j = 0; j < 1024 && shown < 10; j++) {
                const uint64_t node = tbl + (uint64_t)j * 0x9b8;
                const uint64_t head = *reinterpret_cast<volatile uint64_t *>(node);
                const uint64_t tail = *reinterpret_cast<volatile uint64_t *>(node + 8);
                if (!head || head == tail)
                  continue;  // empty queue: nothing pending here
                const uint64_t dep = *reinterpret_cast<volatile uint64_t *>(node + 0x980);
                const uint64_t thresh = *reinterpret_cast<volatile uint64_t *>(node + 0x988);
                const uint32_t mode = *reinterpret_cast<volatile uint32_t *>(node + 0x990);
                const uint32_t aff = *reinterpret_cast<volatile uint32_t *>(node + 0x998);
                long long depval = -1;
                bool readable = false;
                if (dep > 0x10000 && dep < (1ULL << 47)) {
                  const uint64_t inner = *reinterpret_cast<volatile uint64_t *>(dep);
                  depval = (long long)inner;
                  readable = true;
                }
                const bool satisfied =
                    !dep || (readable && (mode == 0 ? (uint64_t)depval == thresh
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
                const uint64_t obj = 0;
                uint32_t mask = 0;
                bool mask_read = false;
                std::fprintf(stderr,
                             "[jobwatch] node[%d] head=%#llx tail=%#llx dep=%#llx *dep=%lld "
                             "thresh=%lld mode=%u aff=%#x obj=%#llx mask=%s%#x -> %s%s\n",
                             j, (unsigned long long)head, (unsigned long long)tail,
                             (unsigned long long)dep, depval, (long long)thresh, mode, aff,
                             (unsigned long long)obj, mask_read ? "" : "?", mask,
                             satisfied ? "dep-ok" : "*** BLOCKED ON DEPENDENCY ***",
                             (mask_read && !(mask & 0xf))
                                 ? " *** MASK EXCLUDES EVERY WORKER ***"
                                 : "");
                shown++;
              }
              if (!shown)
                std::fprintf(stderr, "[jobwatch] no queue node has pending work\n");
            }
          }
          static int tableDumps = 0;
          uint64_t defAff = *reinterpret_cast<volatile uint32_t *>(base + 0x3c);
          uint64_t tab = *reinterpret_cast<volatile uint64_t *>(base + 0xb60);
          const uint64_t *bm = reinterpret_cast<const uint64_t *>(base + 0xac0);
          uint64_t used = 0;
          for (int w = 0; w < 16; w++) used += __builtin_popcountll(~bm[w]);
          if (tableDumps < 200) {
            std::fprintf(stderr, "[jobwatch] jobsys=%#llx defaultAffinity[+0x3c]=%#llx table=%#llx bitmapFreeInv=%llu bm0=%#llx\n",
                         (unsigned long long)base, (unsigned long long)defAff,
                         (unsigned long long)tab, (unsigned long long)used,
                         (unsigned long long)bm[0]);
            if (tab > 0x10000) {
              int shown = 0;
              for (int j = 0; j < 1024 && shown < 12; j++) {
                // Print any table entry with a plausible live affinity word;
                // polarity of the bitmap is unknown, so filter on content.
                const uint8_t *job = reinterpret_cast<const uint8_t *>(tab + (uint64_t)j * 0x9b8);
                uint32_t aff = *reinterpret_cast<const uint32_t *>(job + 0x998);
                uint32_t prio = *reinterpret_cast<const uint32_t *>(job + 0x9a0);
                // The mask the claim path actually tests (0x38fc5:
                // `test [node+0xf0], 1<<ordinal`); kick's 0x34e60 helper
                // copies the submitted spec here.
                uint32_t mask = *reinterpret_cast<const uint32_t *>(job + 0xf0);
                uint64_t q0 = *reinterpret_cast<const uint64_t *>(job);
                if (!aff && !q0)
                  continue;
                std::fprintf(stderr,
                             "[jobwatch]   job[%d] q0=%#llx claimMask[+0xf0]=%#x aff[+0x998]=%#x prio=%#x q+0x9a8=%#llx\n",
                             j, (unsigned long long)q0, mask, aff, prio,
                             (unsigned long long)*reinterpret_cast<const uint64_t *>(job + 0x9a8));
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
              auto *src = reinterpret_cast<volatile uint64_t *>(base + (uint64_t)i * 0x120 + 0xeb0);
              auto *dst = reinterpret_cast<volatile uint64_t *>(base + 2ULL * 0x120 + 0xeb0);
              uint64_t v = *src;
              *dst = v;
              *src = 0;
              std::fprintf(stderr,
                           "[jobwatch] JOBMOVE: moved direct-assign marker %#llx block%d -> block2\n",
                           (unsigned long long)v, i);
            }
            persist = 0;
          }
          std::fflush(stderr);
        }
  }).detach();
}

// Install an entry detour on an eboot-internal function. prologueLen must be the
// smallest instruction boundary >= 14 in the prologue (position-independent).
void installInternalHook(uint8_t *base, uint32_t off, uint32_t prologueLen,
                         uint32_t hookId, const char *name) {
  uint8_t *target = base + off;
  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(target) & ~0xFFFull),
                  0x2000, utl::pageProtection::rwx);
  uintptr_t tramp = cpu::makeGuestTrampoline(target, prologueLen, target + prologueLen);
  if (!tramp) { LOG_WARNING("jobtrace: trampoline failed for {}", name); return; }
  uintptr_t wrap = cpu::makeGuestReturnHook(reinterpret_cast<void *>(tramp), hookId,
                                            reinterpret_cast<void *>(&jobTraceLogger), name);
  if (!wrap) { LOG_WARNING("jobtrace: wrapper failed for {}", name); return; }
  uint8_t patch[32];
  patch[0] = 0xFF; patch[1] = 0x25;
  patch[2] = patch[3] = patch[4] = patch[5] = 0x00;   // jmp qword [rip+0]
  uint64_t w = wrap; std::memcpy(patch + 6, &w, 8);
  for (uint32_t i = 14; i < prologueLen; i++) patch[i] = 0x90;
  std::memcpy(target, patch, prologueLen);
  LOG_INFO("jobtrace: hooked {} eboot+{:#x} tramp={:#x} wrapper={:#x}", name, off,
           (unsigned long)tramp, (unsigned long)wrap);
}
} // namespace

static void installJobTrace(smodule &m) {
  if (!kJobTrace)
    return;
  uint8_t *base = m.getInfo().base;
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

static void investigateDcbGate(smodule &m) {
  if (!kPs5Dcbwatch)
    return;
  uint8_t *base = m.getInfo().base;
  struct { uint32_t off; const char *label; } pts[] = {
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
  struct { uint32_t off; const char *label; } rets[] = {
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
  struct { uint32_t got; const char *nid; } gots[] = {
      {0x8e8e38, "vSMAm3cxYTY"}, {0x8e8e40, "23LRUSvYu1M(FAILING)"},
      {0x8e8e48, "2JtWUUiYBXs"}, {0x8e8d80, "1jfXLRVzisc"},
  };
  for (auto &g : gots) {
    uint64_t tgt = *reinterpret_cast<uint64_t *>(base + g.got);
    const char *mod = "??";
    uint64_t off = tgt;
    if (p)
      for (auto &mm : p->getModuleList()) {
        auto &mi = mm->getInfo();
        auto tb = reinterpret_cast<uint64_t>(mi.textSeg.addr);
        if (tb && tgt >= tb && tgt < tb + mi.textSeg.size) {
          mod = mi.name.c_str();
          off = tgt - tb;
          break;
        }
      }
    // Does any loaded module export this NID's hash? (is it resolvable?)
    uint64_t hid = 0;
    const char *expMod = "NONE";
    uint64_t expAddr = 0;
    if (runtime::decode_nid(g.nid, 11, hid) && p)
      for (auto &mm : p->getModuleList())
        if (uintptr_t a = mm->getExport(hid)) {
          expMod = mm->getInfo().name.c_str();
          expAddr = a;
          break;
        }
    std::printf("[dcbimp] %s got=%s+%#llx exportedBy=%s(%#llx)\n", g.nid, mod,
                (unsigned long long)off, expMod, (unsigned long long)expAddr);
  }
  auto *slot = reinterpret_cast<volatile uint64_t *>(base + 0x985a00);
  std::thread([slot] {
    uint64_t last = ~1ull;
    for (int i = 0; i < 400000; i++) {
      uint64_t v = *slot;
      if (v != last) {
        std::printf("[dcbwatch t=%dms] manager[0] (eboot+0x985a00) = %#llx\n",
                    i / 2, (unsigned long long)v);
        last = v;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
  }).detach();
}

modulePtr proc::getModule(base::StringRef name) {
  for (auto &mod : modules) {
    // module name is base::String, compare via c_str.
    if (name == base::StringRef(mod->getInfo().name))
      return mod;
  }
  return {nullptr};
}

modulePtr proc::getModule(uint32_t handle) {
  for (auto &mod : modules) {
    if (mod->getInfo().handle == handle)
      return mod;
  }
  return {nullptr};
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
static void bringUpRebirthSurfaceRegistry(smodule &m) {
  uint8_t *base = m.getInfo().base;
  constexpr uint32_t kRegistryOff = 0x687a90; // bucket-array base pointer
  constexpr uint32_t kCtorZeroOff = 0x1e9bcd; // `mov qword [registry], 0` (11 bytes)
  constexpr size_t kBucketBytes = 0x20 * 0x20; // N buckets * stride

  uint8_t *buckets = allocLowGuest(kBucketBytes);
  if (!buckets) {
    LOG_ERROR("rebirth surface-registry: bucket alloc failed");
    return;
  }
  *reinterpret_cast<uint64_t *>(base + kRegistryOff) =
      reinterpret_cast<uint64_t>(buckets);

  uint8_t *ctor = base + kCtorZeroOff;
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
    auto *slot = reinterpret_cast<volatile uint64_t *>(base + 0x687b30 + 0x38);
    std::thread([slot] {
      uint64_t last = ~0ull;
      for (int i = 0; i < 200000; i++) {
        uint64_t v = *slot;
        if (v != last) {
          std::printf("[gfxctx] +0x38 = %#llx  (t=%dms)\n",
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
static void bringUpRebirthEbootRegistry(smodule &m) {
  uint8_t *base = m.getInfo().base;
  constexpr uint32_t kRegistryOff = 0x985458; // bucket-array base pointer
  constexpr uint32_t kCtorZeroOff = 0x56a5e7; // `mov qword [registry], 0` (11 bytes)
  constexpr size_t kBucketBytes = 0x20 * 0x20; // N buckets * stride

  // `mov qword [rip+0x41ae66], 0` -> eboot+0x985458. Only this build has it.
  static const uint8_t kCtorBytes[] = {0x48, 0xc7, 0x05, 0x66, 0xae, 0x41,
                                       0x00, 0x00, 0x00, 0x00, 0x00};
  if (std::memcmp(base + kCtorZeroOff, kCtorBytes, sizeof(kCtorBytes)) != 0)
    return; // not this title's eboot; nothing to bring up

  uint8_t *buckets = allocLowGuest(kBucketBytes);
  if (!buckets) {
    LOG_ERROR("rebirth eboot-registry: bucket alloc failed");
    return;
  }
  *reinterpret_cast<uint64_t *>(base + kRegistryOff) =
      reinterpret_cast<uint64_t>(buckets);

  uint8_t *ctor = base + kCtorZeroOff;
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
  uint8_t *base = m.getInfo().base;
  std::thread([base] {
    int32_t lc = 0x7fffffff, li = 0x7fffffff;
    uint32_t lf[3] = {0xdead, 0xdead, 0xdead};
    for (int i = 0; i < 120000; i++) {
      int32_t c = *reinterpret_cast<volatile int32_t *>(base + 0x1cb30);
      int32_t idx = *reinterpret_cast<volatile int32_t *>(base + 0x1cb40);
      uint32_t f[3];
      for (int s = 0; s < 3; s++)
        f[s] = *reinterpret_cast<volatile uint32_t *>(base + 0x1cb50 + s * 0x140);
      // port[0] @ vaddr 0x1d550 (stride 0xb0): field@0x14=open flag; field@0x48 set
      // to 0xfffffff3 once Open reaches the deep success path (just before op@0x580).
      uint32_t portOpen = *reinterpret_cast<volatile uint32_t *>(base + 0x1d564);
      uint32_t port48 = *reinterpret_cast<volatile uint32_t *>(base + 0x1d550 + 0x48);
      static uint32_t lpo = 0xdead, lp48 = 0xdead;
      if (c != lc || idx != li || f[0] != lf[0] || f[1] != lf[1] || f[2] != lf[2] ||
          portOpen != lpo || port48 != lp48) {
        std::printf("[vowatch t=%dms] count=%d idx=%d cfg.f0=[%#x %#x %#x] "
                    "port0.open=%u port0.f48=%#x\n",
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
        uint8_t *cfg0 = base + 0x1cb50;
        uint8_t *cfgi = base + 0x1cb50 + (size_t)idx * 0x140;
        std::memcpy(cfgi, cfg0, 0x140);                 // copy ops/vtable
        *reinterpret_cast<uint32_t *>(cfgi) = 4;        // f0 = connected
        patched = true;
        std::printf("[vowatch] FORCE_CONNECT: cfg[%d] <- cfg[0], f0=4\n", idx);
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
static uint64_t PS4ABI voOpMapLog(uint64_t a1, uint64_t userId, uint64_t busType,
                                  uint64_t index, uint64_t param, uint64_t a6) {
  uint32_t pv = 0;
  if (param > 0x10000 && param < 0x800000000000ull)
    pv = *reinterpret_cast<uint32_t *>(param);
  std::printf("[voop] sceVideoOutOpen(userId=%#lx busType=%ld index=%ld param=%#lx "
              "[param]=%#x [param]&0xf=%#x) -> map-op returns 0\n",
              (unsigned long)userId, (long)busType, (long)index,
              (unsigned long)param, pv, pv & 0xf);
  return 0;
}

static void patchVideoOutDiag(smodule &m) {
  watchVideoOutState(m);
  if (kVoOplog) {
    uintptr_t thunk = cpu::makeHostThunk(reinterpret_cast<void *>(&voOpMapLog));
    uint8_t *o = m.getInfo().base + 0x1020;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(o) & ~0xFFFull),
                    0x2000, utl::pageProtection::rwx);
    o[0] = 0x48; o[1] = 0xb8;                       // mov rax, imm64
    *reinterpret_cast<uint64_t *>(o + 2) = thunk;
    o[10] = 0xff; o[11] = 0xe0;                     // jmp rax
    std::printf("[voop] hooked map-op @ +0x1020 -> thunk %#lx\n",
                (unsigned long)thunk);
  }
  // TEST (DELTA_VO_SKIP_580): nop the `js error` after Open's `call op@0x580`
  // (config-validate op). If Open then progresses, op@0x580's return was a gate.
  if (kVoSkip580) {
    uint8_t *c = m.getInfo().base + 0xaeb8;  // js 0xef09 after the op call
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x2000, utl::pageProtection::rwx);
    if (c[0] == 0x78) {  // js rel8
      c[0] = 0x90; c[1] = 0x90;
      std::printf("[votest] nop'd op@0x580 error-js @ +0xaeb8\n");
    } else {
      std::printf("[votest] op@0x580 js bytes mismatch: %#x %#x\n", c[0], c[1]);
    }
  }
  const char *list = kVoPatch;
  if (!list)
    return;
  uint8_t *base = m.getInfo().base;
  struct { const char *name; uint32_t off; uint8_t ret; } fns[] = {
      {"open", 0xaad0, 1}, {"regbuf", 0xb620, 0}, {"fliprate", 0xbde0, 0},
      {"addflip", 0xc6c0, 0}, {"getlabel", 0xbb80, 0}};
  for (auto &fn : fns) {
    if (!std::strstr(list, fn.name))
      continue;
    uint8_t *c = base + fn.off;
    utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                    0x2000, utl::pageProtection::rwx);
    c[0] = 0xb8; c[1] = fn.ret; c[2] = 0; c[3] = 0; c[4] = 0;  // mov eax, imm32
    c[5] = 0xc3;                                               // ret
    std::printf("[vopatch] libSceVideoOut!%s -> return %d\n", fn.name, fn.ret);
  }
}

modulePtr proc::loadModule(base::StringRef name) {
  const bool isPs4GnmDriver =
      plat == platform::ps4 &&
      (name == base::StringRef("libSceGnmDriver") ||
       name == base::StringRef("libSceGnmDriverForNeoMode"));
  if (isPs4GnmDriver)
    name = base::StringRef("libSceGnmDriver");

  auto mod = getModule(name);
  if (mod)
    return mod;

  auto lib = utl::make_ref<smodule>(this);
  lib->getInfo().handle = handleCounter;
  handleCounter++;

  modules.emplace_back(lib);

  base::String sname;
  sname.append(name.data(), name.length());

  // PS5 titles use a coherent Prospero module set: system modules from a
  // firmware dump (DELTA_PS5_MODULES, a ':'-separated list of dirs holding
  // <name>.sprx), the title's own SDK modules from its decrypted/ tree. A PS5
  // process never touches the PS4 modules/ dir - the ABIs are incompatible
  // (FreeBSD 11 vs 9 syscall numbers, different struct layouts).
  if (plat == platform::ps5) {
    lib->getInfo().name = sname; // PS5 modules have no DT_SCE_MODULEINFO
    bool ok = false;
    if (const char *env = kPs5Modules) {
      for (const char *p = env; *p && !ok;) {
        const char *sep = std::strchr(p, ':');
        size_t len = sep ? static_cast<size_t>(sep - p) : std::strlen(p);
        if (len) {
          base::String hp;
          hp.append(p, len);
          if (hp.back() != '/')
            hp += "/";
          hp += sname.c_str();
          hp += ".sprx";
          if (utl::File(hp, utl::fileMode::read).IsOpen() && lib->fromFile(hp))
            ok = true;
        }
        p = sep ? sep + 1 : p + len;
      }
    }
    if (!ok) {
      const char *roots[] = {"/app0/decrypted/sce_module/", "/app0/decrypted/",
                             "/app0/sce_module/", "/app0/"};
      for (const char *root : roots) {
        base::String vp(root);
        vp += sname.c_str();
        vp += ".prx";
        if (vfs::openRead(vp.c_str()).Exists() && lib->fromVfs(vp)) {
          ok = true;
          break;
        }
      }
    }
    if (!ok) {
      // Drop the placeholder we emplaced above: a failed module left in the list
      // gets enumerated and libkernel calls its null module_start (crash). Its
      // imports fall through to the badcall stub instead, which is non-fatal.
      LOG_ERROR("unable to load ps5 module {}", sname.c_str());
      modules.pop_back();
      return nullptr;
    }
    return lib;
  }

  const bool isPackagedSdkModule =
      name == base::StringRef("libc") || name == base::StringRef("libSceFios2");
  auto loadPackagedModule = [&] {
    // Retail applications provide these SDK compatibility modules in
    // /app0/sce_module rather than using the firmware copies. The firmware-
    // derived classification used here is documented by shadPS4:
    // https://github.com/shadps4-emu/shadPS4/blob/2c9caf6bfbe7e1dc7a1b4565af8d84c56469dd56/src/core/libraries/sysmodule/sysmodule_table.h#L363-L372
    const char *roots[] = {"/app0/sce_module/", "/app0/"};
    for (const char *root : roots) {
      base::String vfsPath(root);
      vfsPath.append(name.data(), name.length());
      vfsPath += ".prx";
      if (!vfs::openRead(vfsPath.c_str()).Exists())
        continue;
      if (!lib->fromVfs(vfsPath))
        return false;
      if (name == base::StringRef("rebirth"))
        bringUpRebirthSurfaceRegistry(*lib);
      return true;
    }
    return false;
  };

  if (isPackagedSdkModule && loadPackagedModule())
    return lib;

  // HLE/system modules ship with the emulator; prefer those for modules that
  // are not supplied by the application.
  base::String hostRel("modules/");
  if (isPs4GnmDriver)
    hostRel += ps4::gnmDriverModule();
  else
    hostRel.append(name.data(), name.length());
  hostRel += ".sprx";
  base::String hostPath = utl::make_abs_path(hostRel);
  const bool isRebirth = name == base::StringRef("rebirth");
  const bool isVideoOut = name == base::StringRef("libSceVideoOut");
  if (utl::File(hostPath, utl::fileMode::read).IsOpen()) {
    if (lib->fromFile(hostPath)) {
      if (isPs4GnmDriver)
        lib->getInfo().name = sname;
      if (isRebirth)
        bringUpRebirthSurfaceRegistry(*lib);
      if (isVideoOut)
        patchVideoOutDiag(*lib);
      return lib;
    }
  } else {
    // The game's own modules live inside the pkg: SDK prx under
    // /app0/sce_module, the title's own prx at the app root.
    if (loadPackagedModule())
      return lib;
  }

  LOG_ERROR("unable to load module {}", sname.c_str());
  return nullptr;
}

// Patch a guest function to `xor eax,eax; ret`. Steps over libkernel-internal
// validation that rejects our externally-loaded module set (11.00 offsets).
static void forceReturn0(proc &p, const char *mod, uint32_t off) {
  auto m = p.getModule(base::StringRef(mod));
  if (!m)
    return;
  uint8_t *c = m->getInfo().base + off;
  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                  0x1000, utl::pageProtection::rwx);
  c[0] = 0x31;  // xor eax, eax
  c[1] = 0xC0;
  c[2] = 0xC3;  // ret
}

// Patch a (rdi=paramId, rsi=int* out) getter to `*out = val; return 0`.
static void forceGetterOk(proc &p, const char *mod, uint32_t off, uint32_t val) {
  auto m = p.getModule(base::StringRef(mod));
  if (!m)
    return;
  uint8_t *c = m->getInfo().base + off;
  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
                  0x1000, utl::pageProtection::rwx);
  c[0] = 0xC7; c[1] = 0x06;  // mov dword [rsi], imm32
  std::memcpy(c + 2, &val, 4);
  c[6] = 0x31; c[7] = 0xC0;  // xor eax, eax
  c[8] = 0xC3;               // ret
}

#if defined(DELTA_BACKEND_NATIVE)
// PS5 libc-heap / pthread-mutex bootstrap fix. Once we hand libc a sceLibcParam
// (so its C++ operator-new arena can grow past the tiny 16 MiB default), libc's
// malloc turns thread-safe and locks a static-initialised mutex whose kernel
// state libkernel lazily allocates through the libc-malloc callback held at a
// libkernel data slot (fw 01.14.00: +0x5cfd8, loaded into rdx before every
// call to the lazy-init helper 0x34a10; was +0x68ec0 on fw 12.60). That malloc
// re-locks the still-uninitialised mutex -> unbounded recursion (stack overflow)
// or deadlock. Interpose the allocator with
// a per-thread re-entrancy guard: the outer call delegates to the real allocator
// (so ordinary pthread objects are heap-backed and freed normally), while a
// re-entrant call -- the bootstrap -- is served from a small malloc-free bump
// pool so the mutex can finish initialising. Native x86 only (the thunk is a host
// function the guest calls directly); PS5 only.
using PthreadAllocFn = uint64_t(PS4ABI *)(uint64_t op, uint64_t arg);
static PthreadAllocFn g_origPthreadAlloc = nullptr;
static thread_local int g_pthreadAllocDepth = 0;

static uint64_t PS4ABI ps5PthreadAlloc(uint64_t op, uint64_t arg) {
  if (op == 1 && g_pthreadAllocDepth > 0) {
    static std::atomic<size_t> off{0};
    static uint8_t pool[64 * 1024];
    size_t sz = (arg + 0xF) & ~size_t(0xF);
    size_t o = off.fetch_add(sz, std::memory_order_relaxed);
    return o + sz <= sizeof(pool) ? reinterpret_cast<uint64_t>(pool + o) : 0;
  }
  if (!g_origPthreadAlloc)
    return 0;
  ++g_pthreadAllocDepth;
  uint64_t r = g_origPthreadAlloc(op, arg);
  --g_pthreadAllocDepth;
  return r;
}
#endif

// libkernel populates its pthread-object allocator pointer at runtime, after
// boot patches run, so install our interpose lazily the first time it is
// non-null (called from thread creation, which happens after libc init but before
// the multithreaded malloc-mutex bootstrap). Idempotent; PS5 + native only.
constexpr uintptr_t kPthreadAllocSlot = 0x5cfd8;  // fw 01.14.00 libkernel data
void ps5MaybeInterposePthreadAlloc() {
#if defined(DELTA_BACKEND_NATIVE)
  static std::atomic<bool> done{false};
  auto *p = proc::getActive();
  if (!p || p->getPlatform() != proc::platform::ps5 || done.load())
    return;
  auto k = p->getModule(base::StringRef("libkernel"));
  if (!k)
    return;
  auto *slot = reinterpret_cast<uint64_t *>(k->getInfo().base + kPthreadAllocSlot);
  uint64_t cur = *slot;
  if (!cur || cur == reinterpret_cast<uint64_t>(&ps5PthreadAlloc))
    return;  // not populated yet, or already ours
  if (done.exchange(true))
    return;
  utl::protectMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(slot) & ~0xFFFull),
                  0x1000, utl::pageProtection::rwx);
  g_origPthreadAlloc = reinterpret_cast<PthreadAllocFn>(cur);
  *slot = reinterpret_cast<uint64_t>(&ps5PthreadAlloc);
  LOG_INFO("interposed libkernel pthread-state alloc (+{:#x}) orig={:#x}",
           kPthreadAllocSlot, cur);
#endif
}

// Boot patches applied before entering the guest, needed by every boot path
// (modexec and the real pkg boot), not just the modexec harness.
static void applyBootPatches(proc &p) {
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
      auto *c = reinterpret_cast<uint8_t *>(a);
      utl::protectMem(reinterpret_cast<void *>(a & ~0xFFFull), 0x2000,
                      utl::pageProtection::rwx);
      c[0] = 0x48;  // movabs rax, imm64
      c[1] = 0xB8;
      *reinterpret_cast<uint64_t *>(c + 2) =
          reinterpret_cast<uint64_t>(&guest_tls_get_addr);
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
      auto *c = reinterpret_cast<uint8_t *>(a);
      utl::protectMem(reinterpret_cast<void *>(a & ~0xFFFull), 0x2000,
                      utl::pageProtection::rwx);
      c[0] = 0xB8; // mov eax, imm32
      *reinterpret_cast<uint32_t *>(c + 1) = cpu::kTlsGetAddrSyscall;
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
      uint64_t addr = std::strtoull(cur, &cur, 0);
      if (addr) {
        auto *c = reinterpret_cast<uint8_t *>(addr);
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
    uint64_t addr = std::strtoull(at, &end, 0);
    uint64_t minB = 0x1000000;
    if (end && *end == ',') minB = std::strtoull(end + 1, nullptr, 0) * 1024 * 1024;
    // DELTA_ALLOC_TRACE doubles as a boolean toggle for the [lowalloc] tracer in
    // sys_mem.cpp, so a bare "=1" is legitimate and must NOT be treated as a code
    // address here: addr=1 would protect/deref page 0 (host null-deref SIGSEGV).
    // A real allocator entry is a guest .text vaddr (>= 64 KiB); ignore anything
    // smaller so tracing can be enabled without planting an int3 at a bogus addr.
    if (addr >= 0x10000) {
      auto *c = reinterpret_cast<uint8_t *>(addr);
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
    const uint64_t slot = std::strtoull(cur, &cur, 0);
    const uint64_t depth = (*cur == ':') ? std::strtoull(cur + 1, nullptr, 0) : 0;
    if (slot >= 0x10000) {
      setHeapProfScope(slot, depth);
      LOG_INFO("DELTA_HEAP_PROF_SCOPE: tls slot {:#x} depth +{:#x}", slot, depth);
    }
  }
  if (const char *hp = kHeapProf) {
    char *cur = const_cast<char *>(hp);
    while (cur && *cur) {
      uint64_t addr = std::strtoull(cur, &cur, 0);
      if (addr >= 0x10000) {
        auto *c = reinterpret_cast<uint8_t *>(addr);
        utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                        utl::pageProtection::rwx);
        if (c[0] == 0x55) { c[0] = 0xCC; setHeapProf(addr);
          LOG_INFO("DELTA_HEAP_PROF: hooked alloc entry {:#x}", addr);
        } else {
          LOG_WARNING("DELTA_HEAP_PROF: {:#x} first byte {:#x} != push rbp", addr, c[0]);
        }
      }
      while (*cur == ',' || *cur == ' ') cur++;
    }
  }
  if (const char *ct = kCntTrace) {
    uint64_t addr = std::strtoull(ct, nullptr, 0);
    if (addr) {
      auto *c = reinterpret_cast<uint8_t *>(addr);
      utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                      utl::pageProtection::rwx);
      if (c[0] == 0x55) { c[0] = 0xCC; setCntTrace(addr); }
    }
  }
  if (const char *ft = kFatalTrace) {
    uint64_t addr = std::strtoull(ft, nullptr, 0);
    if (addr) {
      auto *c = reinterpret_cast<uint8_t *>(addr);
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
      uint64_t addr = std::strtoull(s, &end, 0);
      if (addr) {
        auto *c = reinterpret_cast<uint8_t *>(addr);
        utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                        utl::pageProtection::rwx);
        if (c[0] == 0x55) { c[0] = 0xCC; setHdrTrace(addr); }
      }
      s = (end && *end == ',') ? end + 1 : (end ? end : s + 1);
      if (!*s || (end && *end != ',')) break;
    }
  }
  if (const char *ro = kRdoffFix) {
    uint64_t addr = std::strtoull(ro, nullptr, 0);
    if (addr) {
      auto *c = reinterpret_cast<uint8_t *>(addr);
      utl::protectMem(reinterpret_cast<void *>(addr & ~0xFFFull), 0x1000,
                      utl::pageProtection::rwx);
      if (c[0] == 0x55) { c[0] = 0xCC; setRdoffFix(addr); }
    }
  }
  if (const char *sf = kSkipFn) {
    const char *s2 = sf;
    while (*s2) {
      char *end = nullptr;
      uint64_t addr = std::strtoull(s2, &end, 0);
      if (addr) {
        auto *c = reinterpret_cast<uint8_t *>(addr);
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

void proc::start() {
  LOG_ASSERT(modules[1]->getInfo().name == "libkernel");

  installCrashHandler();
  applyBootPatches(*this);

  auto &info = modules[0]->getInfo();
  auto &kinfo = modules[1]->getInfo();

  if (!info.entry) {
    LOG_WARNING("entry missing for {}", info.name.c_str());
    return;
  }

  union stack_entry {
    const void *ptr;
    uint64_t val;
  } stack[128];

  stack[0].val = 1 + 0; // argc
  auto s = reinterpret_cast<stack_entry *>(&stack[1]);
  (*s++).ptr = info.name.c_str();
  (*s++).ptr = nullptr;
  (*s++).ptr = nullptr;
  (*s++).val = 9ull;
  (*s++).ptr = (const void *)(info.entry);
  (*s++).ptr = nullptr;
  (*s++).ptr = nullptr;

  // Enter libkernel's entry with the PS4 convention (arg block in rdi). The
  // backend runs it natively (x86 host) or via the FEXCore JIT (aarch64 host).
  // PS5 starts with the TCB its kernel would have installed; libkernel reads
  // fs:0x10 before it gets around to setting up its own (see makeInitialTcb).
  const uint64_t fsbase = plat == platform::ps5 ? ps5::makeInitialTcb() : 0;
  cpu::backend().enterGuest(reinterpret_cast<uintptr_t>(kinfo.entry), stack,
                            fsbase);
}
}  // namespace krnl
