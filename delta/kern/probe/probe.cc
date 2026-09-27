/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "kern/probe/probe.h"
#include "kern/probe/probe_arm.h"
#include "kern/probe/probe_internal.h"

#include <sys/mman.h>
#include "base/arch.h"
#include "base/logging.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "guest_abi.h"
#include "host_memory/host_memory.h"
#include "io/file.h"
#include "io/path.h"
#include "options/options.h"

#include "cpu/backend.h"
#include "kern/crash.h"
#include "kern/lv2/sys_dynlib.h"
#include "kern/lv2/sys_mem.h"
#include "kern/module.h"
#include "kern/process.h"
#include "kern/vfs.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "base/atomic.h"
#include "base/containers/set.h"
#include "base/containers/vector.h"
#include "base/strings/string_ref.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "kern/lv2/sys_thread.h"

namespace {
DELTA_OPTION(const char*, kFnWatch, "DELTA_FNWATCH", nullptr);
DELTA_OPTION(const char*, kRetTrace, "DELTA_RETTRACE", nullptr);
DELTA_OPTION(const char*, kGuestPatch, "DELTA_GUEST_PATCH", nullptr);
DELTA_OPTION(const char*, kFnArgs, "DELTA_FNARGS", nullptr);
DELTA_OPTION(const char*, kFiosProbe, "DELTA_FIOS_PROBE", nullptr);
DELTA_OPTION(bool, kFiosAllopen, "DELTA_FIOS_ALLOPEN", false);
DELTA_OPTION(bool, kFiosTrace, "DELTA_FIOS_TRACE", false);
DELTA_OPTION(const char*, kNullGuard, "DELTA_GUEST_NULLGUARD", nullptr);
DELTA_OPTION(bool, kPs5Glyphguard, "DELTA_PS5_GLYPHGUARD", false);
DELTA_OPTION(bool, kPs5Noforce, "DELTA_PS5_NOFORCE", false);
DELTA_OPTION(bool, kSotcForcePayload, "DELTA_SOTC_FORCE_PAYLOAD", false);
DELTA_OPTION(bool, kSotcForceWorlddone, "DELTA_SOTC_FORCE_WORLDDONE", false);
DELTA_OPTION(bool, kSotcJobfix, "DELTA_SOTC_JOBFIX", false);
DELTA_OPTION(const char*, kGuestPopcnt, "DELTA_GUEST_POPCNT", nullptr);
DELTA_OPTION(const char*, kGuestWprot, "DELTA_GUEST_WPROT", nullptr);
DELTA_OPTION(const char*, kGuestRprot, "DELTA_GUEST_RPROT", nullptr);
DELTA_OPTION(const char*, kGuestWhist, "DELTA_GUEST_WHIST", nullptr);
DELTA_OPTION(const char*, kGuestSumwatch, "DELTA_GUEST_SUMWATCH", nullptr);
DELTA_OPTION(const char*, kPoolMap, "DELTA_POOLMAP", nullptr);
DELTA_OPTION(const char*, kMemDump, "DELTA_MEMDUMP", nullptr);
DELTA_OPTION(bool, kSotcSkipWorldwait, "DELTA_SOTC_SKIP_WORLDWAIT", false);
}  // namespace

namespace kern {}

namespace kern::probe {

static void ProbeFiosPaths();

// DELTA_PS5_DCBWATCH: diagnose the null frame-0 DrawCommandBuffer gate: poll
// the DCB pointer slot (manager[0] @ eboot+0x985a00) + adjacent fields, logging
// every change (is the DCB ever created before the render uses it?), plus an
// int3 call-order trace over the renderer/DCB-creation entry points.
// DELTA_FNWATCH="hexoff:label,...": int3 hit-counters at guest function entries
// (first byte push rbp), eboot-relative; the handler counts and a thread prints
// totals every 2s (see crash.h).
static void InvestigatePopcnt();
static void InvestigateSumWatch();
static void InvestigateWriteWatch();
static void InvestigateWriteHist();
static void InvestigatePoolMap();
static void InvestigateMemDump();
static void InvestigateRetTrace(Process&);

// DELTA_RETTRACE="[module+]hexoff:label,...": return-value trace at each
// `mov ebx,eax` / `test eax,eax` after a call, eboot-relative unless a module
// is named, so a failure can be followed into the system module reporting it.
static void InvestigateRetTrace(Process& pr) {
  const char* e = kRetTrace;
  if (!e)
    return;
  for (const char* p = e; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    base::String mod_name;
    if (const char* plus = std::strchr(p, '+')) {
      const char* comma = std::strchr(p, ',');
      if (!comma || plus < comma) {
        mod_name.append(p, (size_t)(plus - p));
        p = plus + 1;
      }
    }
    u8* base = pr.GetModuleList()[0]->GetInfo().base;
    if (!mod_name.empty()) {
      auto mod = pr.GetModule(base::StringRef(mod_name));
      if (!mod) {
        LOG_WARNING("rettrace: {} is not loaded", mod_name.c_str());
        while (*p && *p != ',')
          p++;
        continue;
      }
      base = mod->GetInfo().base;
    }
    const char* where = mod_name.empty() ? "eboot" : mod_name.c_str();
    char* endp = nullptr;
    u64 off = std::strtoull(p, &endp, 16);
    const char* label = "ret";
    if (endp && *endp == ':') {
      const char *lb = endp + 1, *le = lb;
      while (*le && *le != ',')
        le++;
      label = strndup(lb, (size_t)(le - lb));  // persists for SetRetTrace
      p = le;
    } else {
      p = endp;
    }
    auto* c = base + off;
    host_memory::ProtectMem(
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
        0x1000, host_memory::PageProtection::kRwx);
    const bool is_test = c[0] == 0x85 && c[1] == 0xc0;
    if ((c[0] == 0x89 && c[1] == 0xc3) || is_test) {
      c[0] = 0xCC;
      SetRetTrace(reinterpret_cast<uintptr_t>(c), label, is_test);
      LOG_INFO("rettrace: armed {} @ {}+{:#x}", label, where, off);
    } else {
      LOG_WARNING(
          "rettrace: {}+{:#x} bytes {:#x} {:#x} is neither "
          "mov ebx,eax nor test eax,eax",
          where, off, c[0], c[1]);
    }
  }
}

static void InvestigateFnWatch(Module& m) {
  InvestigatePopcnt();
  InvestigateSumWatch();
  InvestigateWriteWatch();
  InvestigateWriteHist();
  InvestigatePoolMap();
  InvestigateMemDump();
  InvestigateRetTrace(*Process::GetActive());
  const char* e = kFnWatch;
  if (!e)
    return;
  u8* base = m.GetInfo().base;
  for (const char* p = e; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    char* endp = nullptr;
    u64 off = std::strtoull(p, &endp, 0);
    const char* label = "fn";
    if (endp && *endp == ':') {
      const char* lb = endp + 1;
      const char* le = lb;
      while (*le && *le != ',')
        le++;
      label = strndup(lb, (size_t)(le - lb));  // persists for SetFnWatch
      p = le;
    } else {
      p = endp;
    }
    auto* c = base + off;
    host_memory::ProtectMem(
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
        0x1000, host_memory::PageProtection::kRwx);
    if (c[0] == 0x55) {
      c[0] = 0xCC;
      SetFnWatch(reinterpret_cast<uintptr_t>(c), label);
      LOG_INFO("fnwatch: armed {} @ eboot+{:#x}", label, off);
    } else {
      LOG_WARNING("fnwatch: eboot+{:#x} first byte {:#x} != push rbp", off,
                  c[0]);
    }
  }
  StartFnWatchPrinter();
}

static void InvestigatePopcnt() {
  const char* pc = kGuestPopcnt;
  if (!pc)
    return;
  const uintptr_t at = std::strtoull(pc, nullptr, 16);
  const char* colon = std::strchr(pc, ':');
  size_t bytes = 0x1000;
  unsigned ms = 2000;
  if (colon) {
    bytes = std::strtoull(colon + 1, nullptr, 16);
    if (const char* c2 = std::strchr(colon + 1, ':'))
      ms = (unsigned)std::strtoul(c2 + 1, nullptr, 0);
  }
  StartPopcntPrinter(at, bytes, ms);
}

static void InvestigateWriteWatch() {
  const bool reads = kGuestRprot != nullptr;
  const char* wp = reads ? kGuestRprot : kGuestWprot;
  if (!wp)
    return;
  const uintptr_t at = std::strtoull(wp, nullptr, 16);
  const char* colon = std::strchr(wp, ':');
  size_t bytes = 0x4000;
  unsigned ms = 200;
  if (colon) {
    bytes = std::strtoull(colon + 1, nullptr, 16);
    if (const char* c2 = std::strchr(colon + 1, ':'))
      ms = (unsigned)std::strtoul(c2 + 1, nullptr, 0);
  }
  StartWriteWatch(at, bytes, ms, reads, false);
}

static void InvestigateWriteHist() {
  const char* wh = kGuestWhist;
  if (!wh)
    return;
  const uintptr_t at = std::strtoull(wh, nullptr, 16);
  const char* colon = std::strchr(wh, ':');
  size_t bytes = 0x1000000;
  unsigned ms = 500;
  if (colon) {
    bytes = std::strtoull(colon + 1, nullptr, 16);
    if (const char* c2 = std::strchr(colon + 1, ':'))
      ms = (unsigned)std::strtoul(c2 + 1, nullptr, 0);
  }
  StartWriteHist(at, bytes, ms);
}

static void InvestigatePoolMap() {
  const char* pm = kPoolMap;
  if (!pm)
    return;
  const char* colon = std::strchr(pm, ':');
  if (std::strncmp(pm, "all", 3) == 0) {
    StartPoolCensus(colon ? (unsigned)std::strtoul(colon + 1, nullptr, 0)
                          : 10000);
    return;
  }
  const uintptr_t at = std::strtoull(pm, nullptr, 16);
  size_t bytes = 0x1000000;
  unsigned ms = 10000;
  if (colon) {
    bytes = std::strtoull(colon + 1, nullptr, 16);
    if (const char* c2 = std::strchr(colon + 1, ':'))
      ms = (unsigned)std::strtoul(c2 + 1, nullptr, 0);
  }
  StartPoolMap(at, bytes, ms);
}

static void InvestigateMemDump() {
  const char* e = kMemDump;
  if (!e)
    return;
  base::String list(e);
  size_t start = 0;
  while (start < list.size()) {
    size_t comma = list.find(',', start);
    base::String spec =
        list.substr(start, comma == base::String::npos ? base::String::npos
                                                       : comma - start);
    uintptr_t at = 0;
    size_t bytes = 0;
    unsigned ms = 0;
    size_t p1 = spec.find(':'), p2 = spec.find(':', p1 + 1),
           p3 = spec.find(':', p2 + 1);
    if (p3 != base::String::npos) {
      at = std::strtoull(spec.c_str(), nullptr, 16);
      bytes = std::strtoull(spec.c_str() + p1 + 1, nullptr, 16);
      ms = (unsigned)std::strtoul(spec.c_str() + p2 + 1, nullptr, 0);
      StartMemDump(at, bytes, ms, spec.c_str() + p3 + 1);
    }
    if (comma == base::String::npos)
      break;
    start = comma + 1;
  }
}

static void InvestigateSumWatch() {
  const char* sw = kGuestSumwatch;
  if (!sw)
    return;
  u64 f[5] = {0, 0, 0, 0, 1000};
  const char* p = sw;
  for (int i = 0; i < 5 && p && *p; i++) {
    f[i] = std::strtoull(p, nullptr, i == 3 ? 10 : 16);
    p = std::strchr(p, ':');
    if (p)
      p++;
  }
  StartSumWatchPrinter((uintptr_t)f[0], (size_t)f[1], (size_t)f[2], (int)f[3],
                       (unsigned)f[4]);
}

// DELTA_GUEST_PATCH="hexoff=hexbytes,...": overwrite guest code/data at an
// eboot offset. For bisecting a wedge: NOP out a poll and see whether the
// thread behind it is the only thing blocked, without waiting for the real fix.
static void ApplyGuestPatches(Module& m) {
  const char* e = kGuestPatch;
  if (!e)
    return;
  u8* base = m.GetInfo().base;
  for (const char* p = e; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    char* endp = nullptr;
    u64 off = std::strtoull(p, &endp, 16);
    if (!endp || *endp != '=')
      break;
    const char* h = endp + 1;
    u8 bytes[32];
    int n = 0;
    while (n < 32 && std::isxdigit((unsigned char)h[0]) &&
           std::isxdigit((unsigned char)h[1])) {
      bytes[n++] = (u8)std::strtoul(base::String(h, 2).c_str(), nullptr, 16);
      h += 2;
    }
    p = h;
    auto* c = base + off;
    host_memory::ProtectMem(
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
        0x2000, host_memory::PageProtection::kRwx);
    std::memcpy(c, bytes, (size_t)n);
    LOG_INFO("guestpatch: {} byte(s) at eboot+{:#x}", n, off);
  }
}

// DELTA_FNARGS="hexoff[+o1+o2...]:label,...": arm an int3 at each guest
// function entry (first byte must be push rbp) that logs rdi and walks the
// offset chain from it. See SetFnArgs in crash.h.
static void InvestigateFnArgs(Module& m) {
  const char* e = kFnArgs;
  if (!e)
    return;
  u8* base = m.GetInfo().base;
  for (const char* p = e; *p;) {
    while (*p == ',' || *p == ' ')
      p++;
    if (!*p)
      break;
    char* endp = nullptr;
    u64 off = std::strtoull(p, &endp, 16);
    u64 offs[8];
    int noffs = 0;
    while (endp && *endp == '+' && noffs < 8)
      offs[noffs++] = std::strtoull(endp + 1, &endp, 16);
    const char* label = "fn";
    if (endp && *endp == ':') {
      const char *lb = endp + 1, *le = lb;
      while (*le && *le != ',')
        le++;
      label = strndup(lb, (size_t)(le - lb));  // persists for SetFnArgs
      p = le;
    } else {
      p = endp;
    }
    auto* c = base + off;
    host_memory::ProtectMem(
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(c) & ~0xFFFull),
        0x1000, host_memory::PageProtection::kRwx);
    if (c[0] == 0x55) {
      c[0] = 0xCC;
      SetFnArgs(reinterpret_cast<uintptr_t>(c), label, offs, noffs);
      LOG_INFO("fnargs: armed {} @ eboot+{:#x} ({} offsets)", label, off,
               noffs);
    } else {
      LOG_WARNING("fnargs: eboot+{:#x} first byte {:#x} != push rbp", off,
                  c[0]);
    }
  }
}

// DELTA_SOTC_FORCE_PAYLOAD: experiment past LoadInitialWorld. The
// world-container's whole-file FIOS2 read returns actualCount 0 (root cause
// open), so the payload accessor at eboot+0x14c000 (cmp [rdi+0x80],0; je ->
// null) returns NULL and CommitResult re-enqueues forever, wedging the game
// thread on [op+0x98]==0xb. [ldr+0x90] IS allocated pre-read, so NOP the `je 74
// 07` -> `90 90`: the accessor always returns that buffer; an empty container
// should parse as 0 entries. Env- gated EXPERIMENT, not a shipped fix.
static void ForceSotcPayload(Module& m) {
  u8* base = m.GetInfo().base;
  auto rwx = [](u8* p) {
    host_memory::ProtectMem(
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(p) & ~0xFFFull),
        0x1000, host_memory::PageProtection::kRwx);
  };
  if (kSotcForcePayload) {
    u8* je =
        base + 0x14c00a;  // `74 07` je +9 (return null) in accessor 0x14c000
    rwx(je);
    if (je[0] == 0x74 && je[1] == 0x07) {
      je[0] = 0x90;
      je[1] = 0x90;
      LOG_INFO("sotc force-payload: NOPed payload-null je @ eboot+0x14c00a");
    } else {
      LOG_WARNING("sotc force-payload: unexpected bytes {:#x} {:#x}", je[0],
                  je[1]);
    }
  }
  // DELTA_SOTC_SKIP_WORLDWAIT: make CGame::LoadInitialWorld's busy-poll
  // (0x3c54e0..) exit at once instead of spinning on a world-op that never
  // reaches 0xb (FHGetSize=0). NOP the loop-tail `je 0x3c54e0` (6x 0x90) so it
  // falls into the epilogue and returns to 0x25c1ff, which tolerates a NULL
  // result. Env-gated EXPERIMENT.
  if (kSotcSkipWorldwait) {
    u8* je = base + 0x3c55fd;
    rwx(je);
    if (je[0] == 0x0f && je[1] == 0x84) {
      for (int i = 0; i < 6; i++)
        je[i] = 0x90;
      LOG_INFO(
          "sotc skip-worldwait: NOPed LoadInitialWorld poll-loop je @ "
          "eboot+0x3c55fd");
    } else {
      LOG_WARNING("sotc skip-worldwait: unexpected bytes {:#x} {:#x}", je[0],
                  je[1]);
    }
  }
  // DELTA_SOTC_FORCE_WORLDDONE: force the world-container async op COMPLETE so
  // boot reaches the title, skipping the broken precache. (1) CommitResult
  // 0x14d768 `je 0x14d777` NOPed so a null-payload commit writes op-state 0xb
  // instead of re-enqueuing; (2) IsDone 0x14cf0e `setne al` -> `mov al,1` (done
  // on 0xb alone). The continuation at 0x25c1ff tolerates NULL. Env-gated
  // EXPERIMENT. DELTA_SOTC_JOBFIX: the hang is a BPE-JobSystem livelock; the
  // world-op finalize job is DIRECT-ASSIGNED to ordinal 6 (the coordinator,
  // pinned to core 6, mask 0x40, outside our 6-core cpuset) while only workers
  // 0..3 exist, so slot 6 is never serviced and the coordinator parks forever
  // (proven live: directAssign[6]=1, ~99.4% claim failures). Ordinal from fn
  // 0x33350 ([tcb-0x10]=0x8000|core). Clamp its result to [0,3] (patch tail
  // 0x3337f: test/js/cmp 4/jb/and 3/ret) so the coordinator assigns to a
  // serviced slot; workers and unbound (-1) unchanged.
  if (kSotcJobfix) {
    u8* t = base + 0x3337f;
    rwx(t);
    if (t[0] == 0x5d && t[1] == 0xc3) {
      static const u8 kCode[] = {0x85, 0xc0,        // test eax,eax
                                 0x78, 0x08,        // js .r (+8 -> pop)
                                 0x83, 0xf8, 0x04,  // cmp eax,4
                                 0x72, 0x03,        // jb .r (+3 -> pop)
                                 0x83, 0xe0, 0x03,  // and eax,3
                                 0x5d, 0xc3};       // .r: pop rbp; ret
      std::memcpy(t, kCode, sizeof(kCode));
      t[14] = 0x90;
      t[15] = 0x90;
      LOG_INFO(
          "sotc jobfix: clamped worker-ordinal fn 0x33350 to [0,3] "
          "(core-6 coordinator's direct-assign now serviced)");
    } else {
      LOG_WARNING("sotc jobfix: unexpected bytes at 0x3337f {:#x} {:#x}", t[0],
                  t[1]);
    }
  }
  if (kSotcForceWorlddone) {
    u8* je = base + 0x14d768;  // 74 0d  je 0x14d777 (retry)
    u8* sn = base + 0x14cf0e;  // 0f 95 c0  setne al
    rwx(je);
    rwx(sn);
    bool ok = true;
    if (je[0] == 0x74 && je[1] == 0x0d) {
      je[0] = 0x90;
      je[1] = 0x90;
    } else {
      LOG_WARNING("sotc force-worlddone: retry je bytes {:#x} {:#x}", je[0],
                  je[1]);
      ok = false;
    }
    if (sn[0] == 0x0f && sn[1] == 0x95 && sn[2] == 0xc0) {
      sn[0] = 0xb0;
      sn[1] = 0x01;
      sn[2] = 0x90;
    } else {
      LOG_WARNING("sotc force-worlddone: setne bytes {:#x} {:#x} {:#x}", sn[0],
                  sn[1], sn[2]);
      ok = false;
    }
    if (ok)
      LOG_INFO(
          "sotc force-worlddone: patched CommitResult retry->0xb + IsDone "
          "always-done");
  }
}

// ===========================================================================
// DELTA_FIOS_TRACE: ARM-compatible guest trace of libSceFios2's
// FHOpen/FHGetSize/FHRead/FHPread (the whole-file API the SotC world-container
// loads through); int3 hooks are inert under FEX, so each import is WRAPPED via
// cpu::MakeGuestReturnHook. Wrapping happens at IMPORT-RESOLUTION time
// (maybeWrapFiosImport from ResolveImports): the PLT slots are lazy, so a GOT
// patch at proc::create is too early; at ResolveImports the real export address
// is in hand and substituting the wrapper installs the hook exactly when bound.
// ===========================================================================
namespace {
struct FiosOpen {  // one FHOpen (all opens tracked so any fh maps to a path)
  u64 p_out_fh;    // guest ptr the async open writes the SceFiosFH into
  base::String path;
};
base::Mutex g_fios_mx;
base::Vector<FiosOpen> g_fios_opens;  // all opens, for fh->path reverse lookup
base::Set<u64> g_zero_size_fh;        // FHGetSize==0 fh's already reported
base::Set<u64> g_zero_act_op;  // OpGetActualCount==0 ops already reported
base::Set<base::String>
    g_seen_paths;  // DELTA_FIOS_ALLOPEN: dedup full open list
base::Atomic<u64> g_fios_open_n{0}, g_zero_size_churn{0}, g_zero_act_churn{0};

// Safe-ish read of a guest C string (guest memory is identity-mapped to host).
const char* GuestStr(u64 va, char* buf, size_t cap) {
  if (!va || va < 0x10000) {
    buf[0] = 0;
    return buf;
  }
  const char* s = reinterpret_cast<const char*>(va);
  size_t i = 0;
  for (; i < cap - 1; ++i) {
    char c = s[i];
    if (!c)
      break;
    buf[i] = c;
  }
  buf[i] = 0;
  return buf;
}

// Reverse-map an SceFiosFH to the path that opened it (newest first). Only ever
// called on the RARE zero-size/zero-count anomaly, so the O(n) scan is fine.
// Caller holds g_fiosMx.
base::String PathForFh(u64 fh) {
  if (!fh)
    return "<null-fh>";
  for (auto it = g_fios_opens.rbegin(); it != g_fios_opens.rend(); ++it) {
    u64 v = it->p_out_fh ? *reinterpret_cast<u64*>(it->p_out_fh) : 0;
    if (v && v == fh)
      return it->path;
  }
  return "<unknown-fh>";
}

// Native logger called after the real FIOS2 fn returns. hookId: 1=FHOpen
// 2=FHGetSize 3=FHRead 4=FHPread 5=OpGetActualCount; FHGetSize ret = size,
// OpGetActualCount ret = actualCount (the smoking guns); FHOpen a1=pOutFH
// a2=path ret=op; FHRead/Pread a1=fh a2=buf a3=len ret=op. Path-agnostic
// detection: a 0 size/count is the anomaly, mapped back to the opener. Zero
// events deduped per fh/op so the post-drain retry churn can't flood the log.
void PS4ABI
FiosTraceLogger(u64 hook_id, u64 a0, u64 a1, u64 a2, u64 a3, u64 ret) {
  char pb[512];
  switch (hook_id) {
    case 1: {  // FHOpen
      u64 n = ++g_fios_open_n;
      // Run the DELTA_FIOS_PROBE once, after enough opens that PlayGo chunks
      // for /app0/misc and /app0/scripts are mounted (the guest opens
      // /app0/misc files by open ~#905). Firing at proc::create or the first
      // open is too early.
      if (n == 2000) {
        static const bool kProbeOnce = ([] { ProbeFiosPaths(); }(), true);
        (void)kProbeOnce;
      }
      const char* path = GuestStr(a2, pb, sizeof(pb));
      bool first_seen = false;
      {
        base::LockGuard lk(g_fios_mx);
        if (g_fios_opens.size() < 80000)
          g_fios_opens.push_back({a1, base::String(path)});
        if (kFiosAllopen)
          first_seen = g_seen_paths.insert(base::String(path)).second;
      }
      // DELTA_FIOS_ALLOPEN: log every DISTINCT path once (full file inventory,
      // to find whether the world-op file is ever even opened). Else sample
      // first 40 + container types.
      if (first_seen || n <= 40 || std::strstr(path, ".calt") ||
          std::strstr(path, "recacheList") ||
          std::strstr(path, "PrecacheList")) {
        BASE_LOGI("fios", "FHOpen path='{}' pOutFH={:#x} op={:#x} (#{})", path,
                  (unsigned long long)a1, (unsigned long long)ret,
                  (unsigned long long)n);
      }
      break;
    }
    case 2: {  // FHGetSize(fh) -> size ; ANY zero/neg is the anomaly
      if ((i64)ret <= 0) {
        base::LockGuard lk(g_fios_mx);
        if (g_zero_size_fh.insert(a0).second) {
          BASE_LOGI("fios", "*** FHGetSize ZERO fh={:#x} -> size={}  path='{}'",
                    (unsigned long long)a0, (long long)ret,
                    PathForFh(a0).c_str());
        } else if ((++g_zero_size_churn % 500000) == 0) {
          BASE_LOGI("fios", "(FHGetSize-zero churn count={})",
                    (unsigned long long)g_zero_size_churn.load());
        }
      } else if (kFiosAllopen && (i64)ret > (4 << 20)) {
        // Large files (>4MB) are candidates for the world container; log
        // once/fh.
        base::LockGuard lk(g_fios_mx);
        if (g_zero_size_fh.insert(a0 ^ 0x5A5A5A5Aull).second)
          BASE_LOGI("fios", "FHGetSize BIG fh={:#x} -> size={} path='{}'",
                    (unsigned long long)a0, (long long)ret,
                    PathForFh(a0).c_str());
      }
      break;
    }
    case 3:
    case 4: {  // FHRead / FHPread ; log zero-length reads (the container's
               // read)
      if (a3 == 0) {
        base::LockGuard lk(g_fios_mx);
        BASE_LOGI("fios",
                  "*** {} ZERO-LEN fh={:#x} buf={:#x} len=0 op={:#x} path='{}'",
                  hook_id == 3 ? "FHRead" : "FHPread", (unsigned long long)a1,
                  (unsigned long long)a2, (unsigned long long)ret,
                  PathForFh(a1).c_str());
      }
      break;
    }
    case 5: {  // OpGetActualCount(op) -> count ; ANY zero/neg is the anomaly
      if ((i64)ret <= 0) {
        base::LockGuard lk(g_fios_mx);
        if (g_zero_act_op.insert(a0).second) {
          BASE_LOGI("fios", "*** OpGetActualCount ZERO op={:#x} -> count={}",
                    (unsigned long long)a0, (long long)ret);
        } else if ((++g_zero_act_churn % 500000) == 0) {
          BASE_LOGI("fios", "(OpGetActualCount-zero churn count={})",
                    (unsigned long long)g_zero_act_churn.load());
        }
      }
      break;
    }
    default:
      break;
  }
}
}  // namespace

// Called from ResolveImports per PLT import: if DELTA_FIOS_TRACE is set and
// nidName
// ("NID#lib#mod") is one of the libSceFios2 whole-file APIs, return a logging
// wrapper around realAddr, else unchanged.
uintptr_t MaybeWrapFiosImport(const char* nid_name, uintptr_t real_addr) {
  if (!kFiosTrace || !nid_name || !real_addr)
    return real_addr;
  struct {
    const char* nid;
    u32 hook_id;
    const char* nm;
  } tbl[] = {
      {"er6TkQFUvp0", 1, "sceFiosFHOpen"},
      {"FdjoqFQOlt0", 2, "sceFiosFHGetSize"},
      {"cg-VoPqZYss", 3, "sceFiosFHRead"},
      {"rR8wq7YFRZs", 4, "sceFiosFHPread"},
      {"+FRvKknUj1I", 5, "sceFiosOpGetActualCount"},
  };
  for (auto& e : tbl) {
    if (std::strncmp(nid_name, e.nid, 11) == 0) {
      uintptr_t wrap = cpu::MakeGuestReturnHook(
          reinterpret_cast<void*>(real_addr), e.hook_id,
          reinterpret_cast<void*>(&FiosTraceLogger), e.nm);
      if (wrap) {
        LOG_INFO("fiostrace: wrapped {} real={:#x} -> wrapper={:#x}", e.nm,
                 (unsigned long)real_addr, (unsigned long)wrap);
        return wrap;
      }
      LOG_WARNING("fiostrace: MakeGuestReturnHook failed for {}", e.nm);
      return real_addr;
    }
  }
  return real_addr;
}

// DELTA_FIOS_PROBE=path1,...: resolve each guest path through the VFS as the
// guest would and log Exists/size, reading the world-op container's backing
// without waiting ~50min for the retry-churning load job (FIOS2 lowercases; $
// -> /app0).
static void ProbeFiosPaths() {
  const char* e = kFiosProbe;
  if (!e)
    return;
  base::String list(e);
  size_t start = 0;
  while (start < list.size()) {
    size_t comma = list.find(',', start);
    base::String path =
        list.substr(start, comma == base::String::npos ? base::String::npos
                                                       : comma - start);
    if (!path.empty()) {
      io::File f = vfs::OpenRead(path.c_str());
      if (f.Exists())
        BASE_LOGI("fiosprobe", "'{}' EXISTS size={}", path.c_str(),
                  (unsigned long long)f.GetSize());
      else
        BASE_LOGI("fiosprobe", "'{}' MISSING (openRead empty)", path.c_str());
    }
    if (comma == base::String::npos)
      break;
    start = comma + 1;
  }
}

// ===========================================================================
// DELTA_JOB_TRACE: instrument SotC's BPE JobSystem (why do the 4 workers
// livelock post-drain?). INTERNAL-function entry detours (not GOT-based):
// overwrite the prologue with an abs jmp to a return-capturing wrapper whose
// realTarget is a trampoline (relocated prologue + jmp back); args+return go to
// jobTraceLogger via the magic syscall. claim 0x38d40: ordinal = fs:[-8], ret =
// claimed job; kick 0x35480: affinity/prio submitted.
// ===========================================================================

// ---- the interface kern calls ---------------------------------------------

void OnProcessCreated(Process& p, Module& main_module, bool ps5) {
  (void)p;
  Module* first = &main_module;
  // Engine bring-up: give Isaac's surface-name registry valid empty storage so
  // main-init doesn't deref a null bucket array (self-gated by ctor signature).
  // DELTA_GUEST_NULLGUARD="<hexoff>:<rax|rsi>:<len>[,...]": recover a guest
  // deref of a bad pointer by zeroing the destination register and stepping
  // over the instruction; for an allocator that faults instead of reporting "no
  // space", this finds whether an out-of-memory path exists at all.
  if (const char* ng = kNullGuard) {
    for (const char* p = ng; *p;) {
      char* endp = nullptr;
      const u64 off = std::strtoull(p, &endp, 16);
      if (!endp || *endp != ':')
        break;
      const char* r = endp + 1;
      const kern::GuardReg reg =
          (*r == 's') ? kern::GuardReg::rsi : kern::GuardReg::rax;
      const char* c = std::strchr(r, ':');
      if (!c)
        break;
      const int len = (int)std::strtol(c + 1, const_cast<char**>(&p), 10);
      kern::SetNullGuard(
          reinterpret_cast<uintptr_t>(first->GetInfo().base) + off, reg, len);
      LOG_INFO("nullguard: armed eboot+{:#x} len={}", off, len);
      while (*p == ',' || *p == ' ')
        p++;
    }
  }
  if (ps5) {
    BringUpRebirthEbootRegistry(*first);
    InvestigateDcbGate(*first);
    // DELTA_PS5_GLYPHGUARD: recover the first-frame unbound-font null derefs in
    // the UI/text renderer so the render reaches real draws (diagnostic; the
    // real fix binds the font before rendererFrame).
    if (kPs5Glyphguard) {
      auto* base8 = first->GetInfo().base;
      auto eb = reinterpret_cast<uintptr_t>(base8);
      // movzx esi,[rdi+rcx*2+0x2e] (glyph cmap count), rdi==0
      kern::SetNullGuard(eb + 0x5cab56, kern::GuardReg::rsi, 5);
      // mov rax,[rax+0x28]; mov rax,[rax+0x18] (chained font-object load),
      // rax==0
      kern::SetNullGuard(eb + 0x5c7c53, kern::GuardReg::rax, 8);
      // ROOT FIX: the renderer-init chain 0x5535d0 bails at its gates when
      // VOInit (gate C, 0x58fb10) returns false in our env, skipping the
      // Shape-Renderer install at 0x55361b and leaving *(0x9854f0) null, the
      // source of the first-frame null- object cascade. Force the chain past
      // its three bail branches. DELTA_PS5_NOFORCE: skip the force now that
      // videoout NIDs are HLE'd (VOInit should return TRUE on its own; forcing
      // leaves an INVALID context with null pipelines).
      struct {
        u32 off;
        u8 b1;
      } gates[] = {{0x553602, 0x59}, {0x553612, 0x49}, {0x553622, 0x39}};
      bool no_force = kPs5Noforce;
      for (auto& g : gates) {
        if (no_force)
          break;
        u8* c = base8 + g.off;
        if (c[0] == 0x74 && c[1] == g.b1) {  // je 0x55365d
          host_memory::ProtectMem(
              reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(c) &
                                      ~0xFFFull),
              0x1000, host_memory::PageProtection::kRwx);
          c[0] = 0x90;  // NOP the bail so the chain runs the Shape-Renderer
                        // install
          c[1] = 0x90;
        }
      }
      LOG_INFO(
          "ps5 glyphguard: forced renderer-init chain through the "
          "Shape-Renderer install");
    }
  }
  // Generic guest-function hit counter (works for PS4 and PS5 eboots).
  InvestigateFnWatch(*first);
  InvestigateFnArgs(*first);
  ApplyGuestPatches(*first);
  ForceSotcPayload(*first);
  InstallJobTrace(*first);
  InstallMatTrace(*first);
  InstallAllocLock(*first);
  // Note: DELTA_FIOS_PROBE runs lazily on the first FIOS2 call (see
  // fiosTraceLogger); at proc::create the /app0 PFS provider isn't mounted yet.
}

void OnModuleLoaded(Module& m, base::StringRef name) {
  if (name == base::StringRef("rebirth"))
    BringUpRebirthSurfaceRegistry(m);
  else if (name == base::StringRef("libSceVideoOut"))
    PatchVideoOutDiag(m);
}

void OnBeforeStart(Process& p) {
  ApplyBootPatches(p);
}

uintptr_t WrapImport(const char* nid_name, uintptr_t real_addr) {
  return MaybeWrapFiosImport(nid_name, real_addr);
}

}  // namespace kern::probe
