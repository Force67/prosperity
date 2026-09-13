/*
 * PS4Delta: FexBackend (aarch64 host). Guest x86-64 code runs inside an
 * embedded FEXCore JIT; nothing in the guest image is rewritten. The JIT
 * decodes `syscall` into HandleSyscall (lv2) and emulates fs/gs from the guest
 * CPUState. Guest threads pin 1:1 to host threads, so guest TLS (fs base) is
 * the host thread_local's CPUState.fs_cached. Mirrors tools fex-embed/harness.
 */
#if defined(DELTA_BACKEND_FEX)

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>
#include <logger/logger.h>

#include <atomic>
#include <thread>
#include <condition_variable>
#include <csetjmp>
#include <csignal>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <ucontext.h>
#include <vector>
#include <sys/mman.h>

#include <FEXCore/Utils/ArchHelpers/Arm64.h>
#include <FEXCore/Utils/Allocator.h>

#include <FEXCore/Core/Context.h>
#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/X86Enums.h>
#include <FEXCore/Core/SignalDelegator.h>
#include <FEXCore/HLE/SyscallHandler.h>
#include <FEXCore/Config/Config.h>
#include <FEXCore/Core/HostFeatures.h>
#include <FEXCore/Debug/InternalThreadState.h>

#include "Common/HostFeatures.h" // FEX::FetchHostFeatures

#include "cpu_backend.h"
#include "kern/crash.h"
#include "kern/lv2/dispatch.h"
#include "kern/module.h"
#include "kern/proc.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(int, kSampleMs, "DELTA_SAMPLE_MS", 0);
DELTA_OPTION(const char *, kRipRace, "DELTA_RIPRACE", nullptr);
DELTA_OPTION(int, kLoadWatch, "DELTA_LOADWATCH", 0);
DELTA_OPTION(u64, kLoadWatchBase, "DELTA_LOADWATCH_BASE", 0x201400000000ull);
DELTA_OPTION(u64, kLoadWatchGoff, "DELTA_LOADWATCH_GOFF", 0x2ee2d00ull);
DELTA_OPTION(bool, kVectorTso, "DELTA_FEX_VECTOR_TSO", false);
DELTA_OPTION(bool, kMemcpyTso, "DELTA_FEX_MEMCPY_TSO", false);
DELTA_OPTION(bool, kFexSctrace, "FEX_SCTRACE", false);
DELTA_OPTION(bool, kHleTrace, "DELTA_HLE_TRACE", false);
DELTA_OPTION(bool, kWatchdog, "DELTA_WATCHDOG", false);
}  // namespace

namespace krnl {
struct tls_index;
void *PS4ABI guest_tls_get_addr(tls_index *ti); // HLE dynamic-TLS resolver
const u32 *currentGuestTidPtr();           // this thread's guest tid TLS addr
}

namespace cpu {

// FEXCore thread on this host thread (1:1); the syscall handler and
// krnl::setThreadFsBase (same TU) reach the live guest CPUState through it.
static thread_local FEXCore::Core::InternalThreadState *t_curThread = nullptr;

// Raw context for the signal-path helpers; owned by FexBackend's unique_ptr.
static FEXCore::Context::Context *g_ctxPtr = nullptr;

// Last syscall this host thread entered, for the crash handler.
static thread_local u32 t_lastSyscall = 0xFFFFFFFFu;
static thread_local bool t_inSyscall = false;

// Per-thread ring of recent guest->host crossings, dumped by the crash
// handler; thread_local so a signal handler can touch it.
struct TraceEvt {
  char kind;        // 's' syscall, 'h' HLE thunk, 0 = empty
  u32 id;      // syscall number / thunk index
  u64 a0, a1, a2, a3;
  u64 ret;     // result (or sentinel before return)
  u64 caller;  // guest caller PC (HLE only)
  const char *name; // static HLE name pointer (HLE only), else nullptr
};
static constexpr u32 kTraceRing = 96;
static thread_local TraceEvt t_trace[kTraceRing];
static thread_local u32 t_tracePos = 0;
static inline TraceEvt &traceNext() {
  TraceEvt &e = t_trace[t_tracePos % kTraceRing];
  t_tracePos++;
  return e;
}

// Set around CTX->ExecuteThread so thr_exit can longjmp out of the JIT.
static thread_local std::jmp_buf t_exitJmp;
static thread_local bool t_exitJmpValid = false;

namespace {

// Executable guest ranges registered by the loader, queried by the JIT.
struct ExecRange {
  u64 base, size;
};
std::mutex g_rangeMutex;
std::vector<ExecRange> g_ranges;

// Per-thunk HLE name (libname!NID), parallel to g_hostThunks, for DELTA_HLE_TRACE.
std::vector<std::string> g_thunkNames;

// Named module ranges, for the deadlock watchdog's symbolization.
struct NamedRange { u64 base, size; std::string name; };
std::mutex g_namedMutex;
std::vector<NamedRange> g_named;
static void symRange(u64 a, char *out, size_t n) {
  std::lock_guard lk(g_namedMutex);
  for (auto &r : g_named)
    if (a >= r.base && a < r.base + r.size) {
      std::snprintf(out, n, "%s[%#llx]+%#llx",
                    r.name.empty() ? "?" : r.name.c_str(),
                    (unsigned long long)r.base,
                    (unsigned long long)(a - r.base));
      return;
    }
  std::snprintf(out, n, "%#llx", (unsigned long long)a);
}

// Live guest threads, for the DELTA_WATCHDOG deadlock dump and DELTA_RIPRACE
// sampling; a signalled thread stamps the round it is answering.
static std::atomic<u64> g_sampleGen{0};
static thread_local std::atomic<u64> t_sampleGen{0};
static thread_local std::atomic<u64> t_sampleRip{0};
// When the sample was taken; delivery is not simultaneous, so "co-occurrence"
// needs a timestamp.
static thread_local std::atomic<u64> t_sampleNs{0};

struct LiveThread {
  FEXCore::Core::InternalThreadState *thread;
  u32 id;
  // This thread's TLS trace ring, so the stall dump shows parked threads'
  // last syscalls with arguments, not just the rip.
  const TraceEvt *trace = nullptr;
  const u32 *tracePos = nullptr;
  const u32 *gtid = nullptr;  // this thread's guest tid (umutex owner space)
  // Parked in a syscall? State.rip is only written at block boundaries, so a
  // waiting thread keeps its last published rip.
  const bool *inSyscall = nullptr;
  // DELTA_RIPRACE sample slots and the host tid to signal for one.
  std::atomic<u64> *sampleGen = nullptr;
  std::atomic<u64> *sampleRip = nullptr;
  std::atomic<u64> *sampleNs = nullptr;
  pid_t hostTid = 0;
};
std::mutex g_liveMutex;
std::vector<LiveThread> g_live;
std::atomic<u32> g_liveSeq{0};
static void startWatchdog() {
  static std::once_flag once;
  std::call_once(once, [] {
    // DELTA_SAMPLE_MS: one-line RIP per live thread every <ms>; pins pre-crash state.
    if (kSampleMs) {
      int ms = kSampleMs;
      if (ms <= 0) ms = 50;
      std::thread([ms] {
        for (u64 tick = 0;; tick++) {
          std::this_thread::sleep_for(std::chrono::milliseconds(ms));
          std::lock_guard lk(g_liveMutex);
          for (auto &t : g_live) {
            auto &S = t.thread->CurrentFrame->State;
            char sym[200];
            symRange(S.rip, sym, sizeof(sym));
            BASE_LOGI("smp", "{} tid={} rip={:#x} {}", (unsigned long long)tick,
                      t.id, (unsigned long long)S.rip, sym);
          }
          std::fflush(stderr);
        }
      }).detach();
    }
    // DELTA_RIPRACE=<ms>:<lo>-<hi>[,...] (hex guest VAs): do two threads ever
    // execute these ranges at the same time? State.rip is stale inside the JIT,
    // so each thread is signalled to reconstruct its own rip. A zero result is
    // strong; a nonzero one wants a second look (delivery is not simultaneous).
    if (kRipRace) {
      std::string spec(kRipRace);
      int ms = 1;
      std::vector<std::pair<u64, u64>> ranges;
      const size_t colon = spec.find(':');
      if (colon != std::string::npos) {
        ms = std::atoi(spec.substr(0, colon).c_str());
        if (ms <= 0) ms = 1;
        const std::string rest = spec.substr(colon + 1);
        size_t i = 0;
        while (i < rest.size()) {
          const size_t comma = rest.find(',', i);
          const std::string one =
              rest.substr(i, comma == std::string::npos ? comma : comma - i);
          i = comma == std::string::npos ? rest.size() : comma + 1;
          const size_t dash = one.find('-');
          if (dash == std::string::npos)
            continue;
          const u64 lo = std::strtoull(one.c_str(), nullptr, 16);
          const u64 hi = std::strtoull(one.c_str() + dash + 1, nullptr, 16);
          if (lo && hi > lo)
            ranges.push_back({lo, hi});
        }
      }
      if (!ranges.empty()) {
        struct sigaction sa {};
        sa.sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK;
        sigemptyset(&sa.sa_mask);
        sa.sa_sigaction = [](int, siginfo_t *, void *ucv) {
          auto *uc = static_cast<ucontext_t *>(ucv);
          const u64 rip = reconstructGuestRip(uc->uc_mcontext.pc);
          struct timespec ts {};
          clock_gettime(CLOCK_MONOTONIC, &ts);
          t_sampleNs.store((u64)ts.tv_sec * 1000000000ull + ts.tv_nsec,
                           std::memory_order_relaxed);
          t_sampleRip.store(rip, std::memory_order_relaxed);
          t_sampleGen.store(g_sampleGen.load(std::memory_order_relaxed),
                            std::memory_order_release);
        };
        sigaction(SIGPROF, &sa, nullptr);
        std::thread([ms, ranges] {
          u64 rounds = 0, withOne = 0, withTwo = 0, maxSeen = 0, reported = 0,
                   answered = 0, withTwoTight = 0;
          for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            const u64 gen =
                g_sampleGen.fetch_add(1, std::memory_order_relaxed) + 1;
            struct Slot { u32 id; pid_t tid;
                          std::atomic<u64> *gp, *rp, *np; };
            Slot slots[64];
            unsigned ns = 0;
            {
              std::lock_guard lk(g_liveMutex);
              for (auto &t : g_live) {
                if (ns >= 64 || !t.sampleGen || !t.sampleRip || !t.hostTid)
                  continue;
                slots[ns++] = {t.id, t.hostTid, t.sampleGen, t.sampleRip,
                               t.sampleNs};
              }
            }
            for (unsigned k = 0; k < ns; k++)
              ::syscall(SYS_tgkill, ::getpid(), slots[k].tid, SIGPROF);
            std::this_thread::sleep_for(std::chrono::microseconds(300));
            struct Hit { u32 id; u64 rip; u64 ns; };
            Hit hits[64];
            unsigned n = 0;
            for (unsigned k = 0; k < ns; k++) {
              if (slots[k].gp->load(std::memory_order_acquire) != gen)
                continue;  // did not answer this round
              answered++;
              const u64 rip = slots[k].rp->load(std::memory_order_relaxed);
              for (auto &r : ranges)
                if (rip >= r.first && rip < r.second) {
                  hits[n++] = {slots[k].id, rip,
                               slots[k].np ? slots[k].np->load(
                                                 std::memory_order_relaxed)
                                           : 0};
                  break;
                }
            }
            rounds++;
            if (n > maxSeen) maxSeen = n;
            if (n == 1) withOne++;
            if (n >= 2) {
              withTwo++;
              // Only a spread well under a microsecond means "both inside at once".
              u64 lo = hits[0].ns, hi = hits[0].ns;
              for (unsigned k = 1; k < n; k++) {
                lo = std::min(lo, hits[k].ns);
                hi = std::max(hi, hits[k].ns);
              }
              const u64 spreadNs = hi - lo;
              if (spreadNs <= 1000)
                withTwoTight++;
              if (reported++ < 40) {
                base::String line;
                base::FormatTo(line, "{} threads inside, samples spread {} ns{}:",
                               n, (unsigned long long)spreadNs,
                               spreadNs <= 1000 ? "  <== SIMULTANEOUS" : "");
                for (unsigned k = 0; k < n; k++) {
                  char sym[160];
                  symRange(hits[k].rip, sym, sizeof(sym));
                  base::FormatTo(line, "  tid={} rip={:#x} {}", hits[k].id,
                                 (unsigned long long)hits[k].rip, sym);
                }
                BASE_LOGI("riprace", "{}", line.c_str());
              }
            }
            if ((rounds % 5000) == 0)
              BASE_LOGI("riprace",
                        "{} rounds ({} thread-answers): {} with one inside, {} with "
                        "TWO OR MORE ({} of them within 1us), max {}",
                        (unsigned long long)rounds,
                        (unsigned long long)answered,
                        (unsigned long long)withOne,
                        (unsigned long long)withTwo,
                        (unsigned long long)withTwoTight,
                        (unsigned long long)maxSeen);
            std::fflush(stderr);
          }
        }).detach();
      }
    }
    // DELTA_LOADWATCH: poll the sum of SotC's 10 per-category load queues
    // (obj+0x110+i*0x28, obj = *(base+0x2ee2d00)) to see the count move or stall.
    if (kLoadWatch) {
      int ms = kLoadWatch;
      if (ms <= 0) ms = 2000;
      const u64 base = kLoadWatchBase;
      const u64 goff = kLoadWatchGoff;
      std::thread([ms, base, goff] {
        auto rd = [](u64 a, void *dst, size_t n) -> bool {
          long pg = sysconf(_SC_PAGESIZE);
          for (u64 p = a & ~((u64)pg - 1); p < a + n; p += pg) {
            unsigned char mv = 0;
            if (mincore(reinterpret_cast<void *>(p), 1, &mv) != 0) return false;
          }
          std::memcpy(dst, reinterpret_cast<void *>(a), n);
          return true;
        };
        u64 pobj = base + goff;
        int lastTotal = -1, plateau = 0;
        for (u64 tick = 0;; tick++) {
          std::this_thread::sleep_for(std::chrono::milliseconds(ms));
          u64 obj = 0;
          if (!rd(pobj, &obj, 8) || obj < 0x1000) {
            BASE_LOGI("loadwatch", "{} obj ptr @{:#x} not ready (obj={:#x})",
                      (unsigned long long)tick, (unsigned long long)pobj,
                      (unsigned long long)obj);
            std::fflush(stderr);
            continue;
          }
          i32 c[10] = {0};
          int total = 0;
          bool ok = true;
          for (int i = 0; i < 10; i++) {
            i32 v = 0;
            if (!rd(obj + 0x110 + (u64)i * 0x28, &v, 4)) { ok = false; break; }
            c[i] = v;
            total += v;
          }
          if (!ok) { BASE_LOGI("loadwatch", "{} obj={:#x} read fault",
                               (unsigned long long)tick, (unsigned long long)obj);
                     std::fflush(stderr); continue; }
          if (total == lastTotal) plateau++; else plateau = 0;
          lastTotal = total;
          BASE_LOGI("loadwatch",
              "{} obj={:#x} REMAINING={} plateau={}x q=[{} {} {} {} {} {} {} {} {} {}]",
              (unsigned long long)tick, (unsigned long long)obj, total, plateau,
              c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);
          std::fflush(stderr);
        }
      }).detach();
    }
    if (!kWatchdog) return;
    int secs = kWatchdog;
    if (secs <= 0) secs = 20;
    std::thread([secs] {
      for (int round = 0;; round++) {
        std::this_thread::sleep_for(std::chrono::seconds(secs));
        std::lock_guard lk(g_liveMutex);
        BASE_LOGI("watchdog", "=== WATCHDOG round {}: {} live guest threads ===",
                  round, g_live.size());
        for (auto &t : g_live) {
          auto &S = t.thread->CurrentFrame->State;
          char sym[256];
          symRange(S.rip, sym, sizeof(sym));
          // scN frozen = stuck in one wait; advancing = looping through waits.
          BASE_LOGI("watchdog", "  tid={} gtid={} rip={:#x} scN={} ({})", t.id,
                    t.gtid ? *t.gtid : 0, (unsigned long long)S.rip,
                    t.tracePos ? *t.tracePos : 0, sym);
          // Raw stack scan for module return addresses (the wait stub omits
          // frame pointers), plus the last syscalls with arguments.
          if (t.trace && t.tracePos) {
            u32 pos = *t.tracePos;
            u32 cnt = pos < kTraceRing ? pos : kTraceRing;
            u32 from = cnt > 6 ? cnt - 6 : 0;
            for (u32 k = from; k < cnt; k++) {
              const TraceEvt &e = t.trace[(pos - cnt + k) % kTraceRing];
              if (e.kind != 's')
                continue;
              BASE_LOGI("watchdog",
                        "      sc {:3} {:<18} ({:#x},{:#x},{:#x},{:#x}) -> {:#x}",
                        e.id, krnl::syscall_getname(e.id),
                        (unsigned long long)e.a0, (unsigned long long)e.a1,
                        (unsigned long long)e.a2, (unsigned long long)e.a3,
                        (unsigned long long)e.ret);
              // Parked in UMTX_OP_MUTEX_WAIT: decode the umutex owner word.
              if (k == cnt - 1 && e.id == 454 && e.a1 == 17 && e.a0 >= 0x10000) {
                unsigned char mv = 0;
                long pg = sysconf(_SC_PAGESIZE);
                if (mincore(reinterpret_cast<void *>(e.a0 & ~((u64)pg - 1)),
                            1, &mv) == 0) {
                  u32 ow = *reinterpret_cast<volatile u32 *>(e.a0);
                  u32 ownerTid = ow & 0x7fffffff;
                  BASE_LOGI("watchdog",
                            "      ^ umutex {:#x} word={:#x} owner-tid={}{}",
                            (unsigned long long)e.a0, ow, ownerTid,
                            (ow & 0x80000000u) ? " CONTESTED" : "");
                  // Print what the owner thread is doing; its wait is the deadlock root.
                  for (auto &o : g_live) {
                    if (!o.gtid || *o.gtid != ownerTid || &o == &t) continue;
                    const TraceEvt *ot = o.trace;
                    u32 opos = o.tracePos ? *o.tracePos : 0;
                    const char *osc = "?";
                    u64 oa0 = 0, oa1 = 0;
                    u32 oid = 0;
                    if (ot && opos) {
                      const TraceEvt &le = ot[(opos - 1) % kTraceRing];
                      if (le.kind == 's') { oid = le.id; osc = krnl::syscall_getname(le.id); oa0 = le.a0; oa1 = le.a1; }
                    }
                    BASE_LOGI("watchdog",
                              "      ^^ OWNER is watchdog tid={} rip={:#x} scN={} last: sc {} {} ({:#x},{:#x})",
                              o.id, (unsigned long long)o.thread->CurrentFrame->State.rip,
                              opos, oid, osc, (unsigned long long)oa0, (unsigned long long)oa1);
                    break;
                  }
                }
              }
            }
          }
          u64 rsp = S.gregs[FEXCore::X86State::REG_RSP];
          int shown = 0;
          for (int i = 0; i < 1024 && shown < 12; i++) {
            u64 a = rsp + (u64)i * 8;
            if (a < 0x1000) break;
            // Guard every read; the scan walks past stack tops.
            unsigned char mv = 0;
            long pg = sysconf(_SC_PAGESIZE);
            if (mincore(reinterpret_cast<void *>(a & ~((u64)pg - 1)), 1,
                        &mv) != 0)
              break;
            u64 v = 0;
            std::memcpy(&v, reinterpret_cast<void *>(a), 8);
            // a plausible code return address that lands in a named module range
            if (v < 0x200000000000ull || v >= 0x210000000000ull) continue;
            char s2[256];
            symRange(v, s2, sizeof(s2));
            if (s2[0] == '0') continue;  // unnamed range -> skip noise
            BASE_LOGI("watchdog", "      stk+{:#x} {:#x} ({})", i * 8,
                      (unsigned long long)v, s2);
            shown++;
          }
        }
        std::fflush(stderr);
      }
    }).detach();
  });
}

// Host-thunk table: index -> native HLE function, dispatched from a guest
// trampoline via the kHostThunkSyscallBase magic syscall. See makeHostThunk.
std::mutex g_thunkMutex;
std::vector<void *> g_hostThunks;
// Bump-allocated pool of guest-executable trampolines (one per bound HLE export).
u8 *g_thunkPool = nullptr;
size_t g_thunkPoolUsed = 0;
constexpr size_t g_thunkPoolSize = 0x100000; // 1 MiB -> ~95k trampolines
constexpr size_t kThunkStride = 16;

class FexSyscallHandler final : public FEXCore::HLE::SyscallHandler {
public:
  FexSyscallHandler() { OSABI = FEXCore::HLE::SyscallOSABI::OS_LINUX64; }

  u64 HandleSyscall(FEXCore::Core::CpuStateFrame *Frame,
                         FEXCore::HLE::SyscallArguments *Args) override {
    // Args->Argument[0] = syscall number (RAX); [1..6] = RDI,RSI,RDX,R10,R8,R9.
    const u32 num = static_cast<u32>(Args->Argument[0]);

    // Dynamic-TLS bridge: patched __tls_get_addr issues this magic syscall.
    if (num == kTlsGetAddrSyscall) {
      u64 r = reinterpret_cast<u64>(krnl::guest_tls_get_addr(
          reinterpret_cast<krnl::tls_index *>(Args->Argument[1])));
      if (g_ctxPtr) {
        u32 ef = g_ctxPtr->ReconstructCompactedEFLAGS(Frame->Thread, false, nullptr, 0);
        g_ctxPtr->SetFlagsFromCompactedEFLAGS(Frame->Thread, ef & ~1u);
      }
      return r;
    }

    // Host-thunk bridge: a planted trampoline issued this magic syscall. It did
    // `mov r10,rcx`, so the original 4th arg (rcx, clobbered by `syscall`) is in
    // Argument[4]; args 7+ sit on the guest stack above the return address.
    if ((num & 0xFF000000u) == kHostThunkSyscallBase) {
      const u32 idx = num & 0x00FFFFFFu;
      void *fn = nullptr;
      {
        std::lock_guard lk(g_thunkMutex);
        if (idx < g_hostThunks.size())
          fn = g_hostThunks[idx];
      }
      u64 ret = 0;
      if (fn) {
        // Args 7..14 sit on the guest stack above the return address; extras a
        // callee ignores are harmless.
        const u64 rsp = Frame->State.gregs[FEXCore::X86State::REG_RSP];
        u64 s[8] = {};
        if (rsp)
          for (int i = 0; i < 8; i++)
            s[i] = reinterpret_cast<u64 *>(rsp)[i + 1];
        using Fn = u64(PS4ABI *)(u64, u64, u64, u64,
                                      u64, u64, u64, u64,
                                      u64, u64, u64, u64,
                                      u64, u64);
        u64 caller = (rsp) ? reinterpret_cast<u64 *>(rsp)[0] : 0;
        const char *evName = nullptr;
        { std::lock_guard lk(g_thunkMutex);
          if (idx < g_thunkNames.size()) evName = g_thunkNames[idx].c_str(); }
        TraceEvt &ev = traceNext();
        ev = {'h', idx, Args->Argument[1], Args->Argument[2], Args->Argument[3],
              Args->Argument[4], ~0ull, caller, evName};
        ret = reinterpret_cast<Fn>(fn)(
            Args->Argument[1], Args->Argument[2], Args->Argument[3],
            Args->Argument[4], Args->Argument[5], Args->Argument[6], s[0], s[1],
            s[2], s[3], s[4], s[5], s[6], s[7]);
        ev.ret = ret;
        if (kHleTrace) {
          char cs[256]; symRange(caller, cs, sizeof(cs));
          const char *nm = "";
          { std::lock_guard lk(g_thunkMutex);
            if (idx < g_thunkNames.size()) nm = g_thunkNames[idx].c_str(); }
          BASE_LOGI("hle", "{} thunk#{}({:#x},{:#x},{:#x},{:#x}) -> {:#x}  from {}",
                    nm, idx, Args->Argument[1], Args->Argument[2],
                    Args->Argument[3], Args->Argument[4],
                    (unsigned long)ret, cs);
        }
      }
      if (g_ctxPtr) {
        u32 ef = g_ctxPtr->ReconstructCompactedEFLAGS(Frame->Thread, false, nullptr, 0);
        g_ctxPtr->SetFlagsFromCompactedEFLAGS(Frame->Thread, ef & ~1u);
      }
      return ret;
    }

    const uintptr_t handler = krnl::lv2_lookup(num);
    if (!handler)
      return 0;

    // Optional syscall trace: FEX_SCTRACE=1.
    if (kFexSctrace)
      BASE_LOGI("sc", "{:3} {:<22} ({:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x})",
                num, krnl::syscall_getname(num), Args->Argument[1],
                Args->Argument[2], Args->Argument[3], Args->Argument[4],
                Args->Argument[5], Args->Argument[6]);

    // lv2 handlers are plain AArch64 functions; translate Linux-style negative
    // errno to the BSD carry + positive errno convention.
    using Fn = u64(PS4ABI *)(u64, u64, u64, u64, u64, u64);
    auto fn = reinterpret_cast<Fn>(handler);
    // The native x86 bsd trampoline normally counts this; count here too.
    if (krnl::g_scHist)
      g_sysHist[num & 1023]++;
    t_lastSyscall = num;
    t_inSyscall = true;
    TraceEvt &ev = traceNext();
    ev = {'s', num, Args->Argument[1], Args->Argument[2], Args->Argument[3],
          Args->Argument[4], ~0ull, 0, nullptr};
    u64 ret = fn(Args->Argument[1], Args->Argument[2], Args->Argument[3],
                       Args->Argument[4], Args->Argument[5], Args->Argument[6]);
    const u32 error = krnl::krnl_syscall_errno(ret);
    if (error)
      ret = error;
    ev.ret = ret;
    t_inSyscall = false;
    if (kFexSctrace)
      BASE_LOGI("sc", "    -> {:#x}", ret);

    // CF isn't in flags[]; set it through the compacted-EFLAGS API so the
    // guest's `jb cerror` sees the syscall result.
    if (g_ctxPtr) {
      u32 ef = g_ctxPtr->ReconstructCompactedEFLAGS(Frame->Thread, false, nullptr, 0);
      if (error)
        ef |= 1u;  // EFLAGS.CF
      else
        ef &= ~1u;
      g_ctxPtr->SetFlagsFromCompactedEFLAGS(Frame->Thread, ef);
    }
    return ret;
  }

  FEXCore::HLE::ExecutableRangeInfo
  QueryGuestExecutableRange(FEXCore::Core::InternalThreadState *, u64 Address) override {
    std::lock_guard lk(g_rangeMutex);
    for (auto &r : g_ranges)
      if (Address >= r.base && Address < r.base + r.size)
        return {r.base, r.size, false};
    return {0, 0, false};
  }

  std::optional<FEXCore::ExecutableFileSectionInfo>
  LookupExecutableFileSection(FEXCore::Core::InternalThreadState *, u64) override {
    return std::nullopt;
  }
};

// Sufficient for fault-free guest code; self-modifying code needs the real
// SIGSEGV/SIGILL plumbing.
class FexSignalDelegator final : public FEXCore::SignalDelegator {};

// Return target for runGuestFunction: longjmp out of the JIT like thr_exit;
// dispatched as a host thunk, so extra args are ignored.
static u64 PS4ABI guestFnReturnExit() {
  exitGuestThread();
  return 0;  // unreachable (exitGuestThread longjmps)
}

class FexBackend final : public ICpuBackend {
public:
  void onImageMapped(krnl::moduleInfo &info) override {
    ensureInit();
    std::lock_guard lk(g_rangeMutex);
    g_ranges.push_back({reinterpret_cast<u64>(info.base), info.codeSize});
    {
      std::lock_guard nk(g_namedMutex);
      g_named.push_back({reinterpret_cast<u64>(info.base), info.codeSize,
                         std::string(info.name.c_str())});
    }
    LOG_INFO("fex: registered exec range {} +{:#x}", (void *)info.base, info.codeSize);
  }

  // Per-guest-thread state; the GDT is per thread (FEX requirement) and the
  // stacks are pooled for reuse.
  struct FexThread {
    FEXCore::Core::InternalThreadState *thread;
    void *stack;
    size_t stackSize;
    void *callret;
    size_t callretSize;
    FEXCore::Core::CPUState::gdt_segment gdt[32];
  };

  // Retired stacks are pooled, never unmapped: guest code captures rsp into
  // long-lived structures, and unmapping made those dangle or corrupted the
  // next reuse (SotC AllocationTracker faults ~10s into LoadInitialWorld).
  std::mutex stackPoolM;
  std::vector<std::pair<void *, size_t>> stackPool;    // guest rsp stacks
  std::vector<std::pair<void *, size_t>> callretPool;  // FEX call-ret stacks

  void *poolTake(std::vector<std::pair<void *, size_t>> &pool, size_t size) {
    std::lock_guard<std::mutex> lk(stackPoolM);
    for (size_t i = 0; i < pool.size(); i++) {
      if (pool[i].second == size) {
        void *p = pool[i].first;
        pool.erase(pool.begin() + i);
        return p;
      }
    }
    return nullptr;
  }
  void poolPut(std::vector<std::pair<void *, size_t>> &pool, void *p,
               size_t size) {
    std::lock_guard<std::mutex> lk(stackPoolM);
    pool.push_back({p, size});
  }

  void *createGuestThread(uintptr_t entry, void *arg, u64 fsbase) override {
    ensureInit();
    auto *h = new FexThread{};

    // Guest stack; HLE handlers run on the host stack, so this only serves
    // guest code. Reuse a pooled retired stack when one exists.
    h->stackSize = 8ull * 1024 * 1024;
    h->stack = poolTake(stackPool, h->stackSize);
    if (!h->stack)
      h->stack = mmap(nullptr, h->stackSize, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    u64 rsp = (reinterpret_cast<u64>(h->stack) + h->stackSize - 0x200) & ~0xFULL;

    // Create on the calling thread, as FEX's ThreadManager does.
    auto *thread = CTX->CreateThread(entry, rsp, nullptr);
    h->thread = thread;
    auto &S = thread->CurrentFrame->State;

    // FEX call/return prediction stack (guard-paged), seeded to Base + SIZE/4.
    {
      constexpr size_t kSize = FEXCore::Core::InternalThreadState::CALLRET_STACK_SIZE;
      constexpr size_t kPage = 0x1000;
      h->callretSize = kSize + 2 * kPage;
      void *alloc = poolTake(callretPool, h->callretSize);
      if (!alloc)
        alloc = mmap(nullptr, h->callretSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (alloc != MAP_FAILED) {
        h->callret = alloc;
        void *crBase = reinterpret_cast<u8 *>(alloc) + kPage;
        mprotect(crBase, kSize, PROT_READ | PROT_WRITE);
        thread->CallRetStackBase = crBase;
        S.callret_sp = reinterpret_cast<u64>(crBase) + kSize / 4;
      }
    }

    // Per-thread 64-bit segments (the decoder reads CS.L).
    S.segment_arrays[FEXCore::Core::CPUState::SEGMENT_ARRAY_INDEX_GDT] = &h->gdt[0];
    S.segment_arrays[FEXCore::Core::CPUState::SEGMENT_ARRAY_INDEX_LDT] = &h->gdt[0];
    S.cs_idx = FEXCore::Core::CPUState::DEFAULT_USER_CS << 3;
    auto *gdt = FEXCore::Core::CPUState::GetSegmentFromIndex(S, S.cs_idx);
    FEXCore::Core::CPUState::SetGDTBase(gdt, 0);
    FEXCore::Core::CPUState::SetGDTLimit(gdt, 0xFFFFFU);
    S.cs_cached = FEXCore::Core::CPUState::CalculateGDTBase(*gdt);
    gdt->L = 1;
    gdt->D = 0;

    // PS4 entry convention: argument block pointer in RDI.
    S.gregs[FEXCore::X86State::REG_RDI] = reinterpret_cast<u64>(arg);

    // Scratch TLS with a TCB self-pointer at [fs:0] until the guest installs
    // its own; an unset base faults early TLS reads.
    u64 fs = fsbase;
    if (fs == 0) {
      constexpr size_t kTls = 0x10000;
      void *t = mmap(nullptr, kTls, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (t != MAP_FAILED) {
        fs = reinterpret_cast<u64>(t) + kTls / 2;
        *reinterpret_cast<u64 *>(fs) = fs;
      }
    }
    S.fs_cached = fs;
    S.gs_cached = fs;
    return h;
  }

  void runGuestThread(void *handle) override {
    auto *h = static_cast<FexThread *>(handle);
    t_curThread = h->thread;
    krnl::installSigAltStack();  // fatal handler must survive a blown guest stack
    // Re-assert the fatal handler: FEXCore init may have registered its own
    // SIGSEGV/SIGILL handlers; sigaction is idempotent.
    krnl::installCrashHandler();
    FEXCore::Allocator::RegisterTLSData(h->thread); // FEX per-thread registration
    startWatchdog();
    u32 myId = g_liveSeq.fetch_add(1);
    u64 entryRip = h->thread->CurrentFrame->State.rip;
    {
      // Log guest + host pthread stack ranges, to attribute fault addresses later.
      pthread_attr_t at;
      void *hsp = nullptr;
      size_t hsz = 0;
      if (pthread_getattr_np(pthread_self(), &at) == 0) {
        pthread_attr_getstack(&at, &hsp, &hsz);
        pthread_attr_destroy(&at);
      }
      BASE_LOGI("fex",
                "gthread rip={:#x} gstack=[{:p}+{:#x}] hoststack=[{:p}+{:#x}]",
                (unsigned long long)entryRip, h->stack, h->stackSize, hsp, hsz);
    }
    { std::lock_guard lk(g_liveMutex);
      g_live.push_back({h->thread, myId, t_trace, &t_tracePos,
                        krnl::currentGuestTidPtr(), &t_inSyscall,
                        &t_sampleGen, &t_sampleRip, &t_sampleNs,
                        static_cast<pid_t>(::syscall(SYS_gettid))}); }
    LOG_INFO("fex: running guest thread rip={:#x} (watchdog tid={})",
             h->thread->CurrentFrame->State.rip, myId);
    // thr_exit longjmps here to leave the JIT; the thread is being torn down anyway.
    if (setjmp(t_exitJmp) == 0) {
      t_exitJmpValid = true;
      CTX->ExecuteThread(h->thread);
    } else {
      LOG_INFO("fex: guest thread exited via thr_exit");
    }
    t_exitJmpValid = false;
    auto &endS = h->thread->CurrentFrame->State;
    LOG_INFO("fex: guest thread returned rip={:#x}", (unsigned long)endS.rip);
    if (kWatchdog) {
      char es[256]; symRange(entryRip, es, sizeof(es));
      char rs[256]; symRange(endS.rip, rs, sizeof(rs));
      BASE_LOGI("watchdog", "=== THREAD tid={} RETURNED entry={:#x} ({}) ret={:#x} ({}) ===",
                myId, (unsigned long long)entryRip, es,
                (unsigned long long)endS.rip, rs);
      if (endS.rip >= 0x200000000000ull && endS.rip < 0x210000000000ull) {
        const u8 *b = reinterpret_cast<const u8 *>(endS.rip);
        BASE_LOGI("watchdog", "      bytes@rip: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                  b[0], b[1], b[2], b[3], b[4], b[5]);
      }
      u64 rsp = endS.gregs[FEXCore::X86State::REG_RSP];
      int shown = 0;
      for (int i = 0; i < 2048 && shown < 20; i++) {
        u64 a = rsp + (u64)i * 8, v = 0;
        if (a < 0x1000) break;
        std::memcpy(&v, reinterpret_cast<void *>(a), 8);
        if (v < 0x200000000000ull || v >= 0x210000000000ull) continue;
        char s2[256]; symRange(v, s2, sizeof(s2));
        if (s2[0] == '0') continue;
        BASE_LOGI("watchdog", "      stk+{:#x} {:#x} ({})", i * 8,
                  (unsigned long long)v, s2);
        shown++;
      }
    }
    // A "return" to a tiny rip is a bad/unset function pointer; dump registers.
    if (endS.rip < 0x100000ull) {
      BASE_LOGI("fex", "=== BOGUS THREAD RETURN rip={:#x} ===",
                (unsigned long)endS.rip);
      const char *rn[] = {"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
                          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"};
      for (int i = 0; i < 16; i++)
        BASE_LOGI("fex", "  {}={:#x}", rn[i], (unsigned long)endS.gregs[i]);
      u64 rsp = endS.gregs[FEXCore::X86State::REG_RSP];
      if (rsp) {
        for (int i = 0; i < 64; i++) {
          u64 v = reinterpret_cast<u64 *>(rsp)[i];
          if (v >= 0x200000000000ull && v < 0x206000000000ull)
            BASE_LOGI("fex", "  stk+{:#x} = {:#x}", i * 8, (unsigned long)v);
        }
      }
    }
    { std::lock_guard lk(g_liveMutex);
      for (size_t i = 0; i < g_live.size(); i++)
        if (g_live[i].thread == h->thread) { g_live.erase(g_live.begin() + i); break; } }
    FEXCore::Allocator::UninstallTLSData(h->thread);
    CTX->DestroyThread(h->thread);
    t_curThread = nullptr;
    // Pool, never unmap: guest code may hold pointers into a retired stack.
    if (h->stack) poolPut(stackPool, h->stack, h->stackSize);
    if (h->callret) poolPut(callretPool, h->callret, h->callretSize);
    delete h;
  }

  u64 runGuestFunction(uintptr_t fn, u64 a0, u64 a1,
                            u64 a2, u64 a3) override {
    // Point a guest function's return address at a thunk that exits the thread.
    static uintptr_t exitThunk =
        makeHostThunk(reinterpret_cast<void *>(&guestFnReturnExit));

    // Inherit the caller's fs base: module init calls libkernel, which reads
    // TLS, and the caller is blocked on join meanwhile.
    u64 fsbase = t_curThread ? t_curThread->CurrentFrame->State.fs_cached : 0;

    // createGuestThread sets RDI=arg; add RSI/RDX for the 2nd/3rd SysV args.
    auto *h = static_cast<FexThread *>(
        createGuestThread(fn, reinterpret_cast<void *>(a0), fsbase));
    auto &S = h->thread->CurrentFrame->State;
    S.gregs[FEXCore::X86State::REG_RSI] = a1;
    S.gregs[FEXCore::X86State::REG_RDX] = a2;
    S.gregs[FEXCore::X86State::REG_RCX] = a3;
    // After the implicit `call`, x86 wants rsp%16==8 at the callee's first instruction.
    u64 rsp = S.gregs[FEXCore::X86State::REG_RSP] & ~0xFULL;
    rsp -= 8;
    *reinterpret_cast<u64 *>(rsp) = exitThunk;
    S.gregs[FEXCore::X86State::REG_RSP] = rsp;
    // Run on a persistent host worker, never the caller's thread, and block:
    // module inits record host-thread-derived pointers (glibc TCB above the
    // pthread stack) that must outlive the call (SotC FIOS2). The console runs
    // inits on a permanent thread; mirror that.
    {
      std::unique_lock<std::mutex> lk(initWorkerM);
      if (!initWorkerStarted) {
        initWorkerStarted = true;
        std::thread([this] {
          for (;;) {
            std::function<void()> job;
            {
              std::unique_lock<std::mutex> wl(initWorkerM);
              initWorkerCv.wait(wl, [this] { return (bool)initWorkerJob; });
              job = std::move(initWorkerJob);
              initWorkerJob = nullptr;
            }
            job();
            {
              std::lock_guard<std::mutex> wl(initWorkerM);
              initWorkerDone = true;
            }
            initWorkerCv.notify_all();
          }
        }).detach();  // process-lifetime worker; its TCB/TLS stay mapped
      }
      initWorkerDone = false;
      initWorkerJob = [this, h] { runGuestThread(h); };
      initWorkerCv.notify_all();
      initWorkerCv.wait(lk, [this] { return initWorkerDone; });
    }
    return 0;
  }

  std::mutex initWorkerM;
  std::condition_variable initWorkerCv;
  bool initWorkerStarted = false;
  std::function<void()> initWorkerJob;
  bool initWorkerDone = false;

private:
  void ensureInit() {
    std::call_once(initFlag, [this] {
      FEXCore::Config::Initialize();
      FEXCore::Config::ReloadMetaLayer();
      FEXCore::Config::Set(FEXCore::Config::CONFIG_IS64BIT_MODE, "1");
      // Serialize unaligned LOCK RMWs: the dual-CAS emulation tears on ARM
      // (SotC's BPE allocator has >1000 unaligned-atomic sites).
      FEXCore::Config::Set(FEXCore::Config::CONFIG_STRICTINPROCESSSPLITLOCKS, "1");
      // FEX orders scalar loadstores only; SIMD copies and REP MOVS reorder on
      // ARM. SotC publishes job descriptors with vmovups pairs, so workers can
      // observe a half-published descriptor. Both knobs default off.
      if (kVectorTso)
        FEXCore::Config::Set(FEXCore::Config::CONFIG_VECTORTSOENABLED, "1");
      if (kMemcpyTso)
        FEXCore::Config::Set(FEXCore::Config::CONFIG_MEMCPYSETTSOENABLED, "1");

      auto HostFeatures = FEX::FetchHostFeatures();
      CTX = FEXCore::Context::Context::CreateNewContext(HostFeatures);
      g_ctxPtr = CTX.get();
      CTX->SetSignalDelegator(&sigDelegator);
      CTX->SetSyscallHandler(&syscallHandler);
      CTX->EnableExitOnHLT();
      if (!CTX->InitCore())
        LOG_ERROR("fex: FEXCore InitCore failed");
      else
        LOG_INFO("fex: FEXCore context initialised");
    });
  }

  std::once_flag initFlag;
  fextl::unique_ptr<FEXCore::Context::Context> CTX;
  FexSyscallHandler syscallHandler;
  FexSignalDelegator sigDelegator;
};

FexBackend g_backend;

} // namespace

namespace {
// Dedicated VA window for FEXCore internals, disjoint from guest memory: a
// guest MAP_FIXED range high in VA would clobber kernel-placed FEX mappings
// (zero-fills the JIT's block-link map, the null-node crash).
#ifdef __ANDROID__
// 39-bit user VA: above the guest arena, below bionic's mmap_base. Reserved in earlyInit.
constexpr uintptr_t kFexHeapBase = 0x0000'0060'0000'0000ull; // 384 GiB
constexpr size_t kFexHeapSize = 32ull * 1024 * 1024 * 1024;  // 32 GiB
#else
constexpr uintptr_t kFexHeapBase = 0x0000'5000'0000'0000ull; // 80 TiB
constexpr size_t kFexHeapSize = 96ull * 1024 * 1024 * 1024;  // 96 GiB
#endif
std::atomic<uintptr_t> g_fexHeapNext{0};
uintptr_t g_fexHeapEnd = 0;

void *fexInternalMmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
  // MAP_FIXED means FEX requires that exact address; honour it.
  if ((flags & MAP_FIXED) || !g_fexHeapEnd)
    return ::mmap(addr, len, prot, flags, fd, off);
  // A bare hint is advisory and the kernel may place the mapping in a range the
  // guest MAP_FIXEDs later; the window matters more, so drop the hint.
  (void)addr;
  // Bump-allocate from the reserved window with MAP_FIXED, never overlapping guest memory.
  const size_t alen = (len + 0xFFFull) & ~0xFFFull;
  uintptr_t base = g_fexHeapNext.fetch_add(alen, std::memory_order_relaxed);
  if (base + alen > g_fexHeapEnd)
    return ::mmap(nullptr, len, prot, flags, fd, off); // window exhausted: fall back
  return ::mmap(reinterpret_cast<void *>(base), len, prot, flags | MAP_FIXED, fd, off);
}
int fexInternalMunmap(void *addr, size_t len) { return ::munmap(addr, len); }
} // namespace

void earlyInit() {
  // Reserve PROT_NONE before any guest mapping, then hook FEXCore's allocator.
  int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
#ifdef MAP_FIXED_NOREPLACE
  void *r = ::mmap(reinterpret_cast<void *>(kFexHeapBase), kFexHeapSize, PROT_NONE,
                   flags | MAP_FIXED_NOREPLACE, -1, 0);
  if (r == MAP_FAILED || r != reinterpret_cast<void *>(kFexHeapBase))
    r = ::mmap(nullptr, kFexHeapSize, PROT_NONE, flags, -1, 0);
#else
  void *r = ::mmap(nullptr, kFexHeapSize, PROT_NONE, flags, -1, 0);
#endif
  if (r == MAP_FAILED) {
    LOG_WARNING("fex: could not reserve internal heap window; JIT memory shares "
                "guest VA (may corrupt under heavy guest mmap use)");
    return;
  }
  g_fexHeapNext.store(reinterpret_cast<uintptr_t>(r), std::memory_order_relaxed);
  g_fexHeapEnd = reinterpret_cast<uintptr_t>(r) + kFexHeapSize;
  FEXCore::Allocator::mmap = fexInternalMmap;
  FEXCore::Allocator::munmap = fexInternalMunmap;
  LOG_INFO("fex: reserved internal heap {} +{:#x}", r, kFexHeapSize);
}

ICpuBackend &backend() { return g_backend; }

void exitGuestThread() {
  if (t_exitJmpValid) {
    t_exitJmpValid = false;
    std::longjmp(t_exitJmp, 1);
  }
  // Not in a guest thread context: nothing to unwind.
}

// Map a host-thunk-pool address back to the HLE export planted there; a guest
// fault in this pool is a call through a bound-but-unserviced import slot.
const char *hostThunkNameForAddr(uintptr_t addr, u32 *idxOut) {
  std::lock_guard lk(g_thunkMutex);
  if (!g_thunkPool)
    return nullptr;
  const auto base = reinterpret_cast<uintptr_t>(g_thunkPool);
  if (addr < base || addr >= base + g_thunkPoolUsed)
    return nullptr;
  const u32 idx = static_cast<u32>((addr - base) / kThunkStride);
  if (idxOut)
    *idxOut = idx;
  if (idx < g_thunkNames.size() && !g_thunkNames[idx].empty())
    return g_thunkNames[idx].c_str();
  return "";
}

// Plant a guest x86 trampoline bouncing into native hostFn via the magic
// syscall; preserves rcx into r10 before `syscall` clobbers rcx.
uintptr_t makeHostThunk(void *hostFn, const char *name) {
  std::lock_guard lk(g_thunkMutex);
  g_thunkNames.resize(g_hostThunks.size() + 1);
  g_thunkNames[g_hostThunks.size()] = name ? name : "";
  if (!g_thunkPool) {
    g_thunkPool = static_cast<u8 *>(
        mmap(nullptr, g_thunkPoolSize, PROT_READ | PROT_WRITE | PROT_EXEC,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (g_thunkPool == MAP_FAILED) {
      g_thunkPool = nullptr;
      LOG_ERROR("fex: host-thunk pool mmap failed");
      return 0;
    }
    // FEX hooks mmap, so the pool lands in the reserved heap; log the range
    // (faults here are bad HLE import slots).
    LOG_INFO("fex: host-thunk pool {:#x}+{:#x}",
             reinterpret_cast<u64>(g_thunkPool),
             (u64)g_thunkPoolSize);
    // FEX won't JIT code outside a registered executable range.
    std::lock_guard rk(g_rangeMutex);
    g_ranges.push_back({reinterpret_cast<u64>(g_thunkPool), g_thunkPoolSize});
  }
  if (g_thunkPoolUsed + kThunkStride > g_thunkPoolSize) {
    LOG_ERROR("fex: host-thunk pool exhausted");
    return 0;
  }
  const u32 idx = static_cast<u32>(g_hostThunks.size());
  g_hostThunks.push_back(hostFn);

  u8 *t = g_thunkPool + g_thunkPoolUsed;
  g_thunkPoolUsed += kThunkStride;
  const u32 sc = kHostThunkSyscallBase | idx;
  u8 *p = t;
  *p++ = 0x49; *p++ = 0x89; *p++ = 0xCA;           // mov r10, rcx
  *p++ = 0xB8;                                       // mov eax, imm32
  std::memcpy(p, &sc, 4); p += 4;
  *p++ = 0x0F; *p++ = 0x05;                          // syscall
  *p++ = 0xC3;                                       // ret
  return reinterpret_cast<uintptr_t>(t);
}

// Plant a trampoline that calls realTarget with the original args, then
// loggerFn(hookId, a0..a3, ret) via the magic syscall, and returns realTarget's
// result. Install by writing the returned address into the import GOT slot;
// the ARM-compatible replacement for int3 return hooks. 0 on failure.
uintptr_t makeGuestReturnHook(void *realTarget, u32 hookId, void *loggerFn,
                              const char *name) {
  std::lock_guard lk(g_thunkMutex);
  if (!g_thunkPool) {
    g_thunkPool = static_cast<u8 *>(
        mmap(nullptr, g_thunkPoolSize, PROT_READ | PROT_WRITE | PROT_EXEC,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (g_thunkPool == MAP_FAILED) {
      g_thunkPool = nullptr;
      LOG_ERROR("fex: guest-hook pool mmap failed");
      return 0;
    }
    std::lock_guard rk(g_rangeMutex);
    g_ranges.push_back({reinterpret_cast<u64>(g_thunkPool), g_thunkPoolSize});
  }
  // Register the native logger as a host-thunk index for the magic syscall.
  const u32 loggerIdx = static_cast<u32>(g_hostThunks.size());
  g_hostThunks.push_back(loggerFn);
  g_thunkNames.resize(g_hostThunks.size());
  g_thunkNames[loggerIdx] = name ? name : "guesthook";
  const u32 magic = kHostThunkSyscallBase | loggerIdx;

  constexpr size_t kWrapStride = 64; // emitted body is ~51 bytes
  if (g_thunkPoolUsed + kWrapStride > g_thunkPoolSize) {
    LOG_ERROR("fex: guest-hook pool exhausted");
    return 0;
  }
  u8 *t = g_thunkPool + g_thunkPoolUsed;
  g_thunkPoolUsed += kWrapStride;
  u8 *p = t;
  auto emit = [&](std::initializer_list<u8> b) { for (u8 x : b) *p++ = x; };
  auto emit32 = [&](u32 v) { std::memcpy(p, &v, 4); p += 4; };
  auto emit64 = [&](u64 v) { std::memcpy(p, &v, 8); p += 8; };
  // On entry: rsp%16==8; args rdi,rsi,rdx,rcx; [rsp]=caller return address.
  emit({0x57});                    // push rdi                 ; save a0
  emit({0x56});                    // push rsi                 ; save a1
  emit({0x52});                    // push rdx                 ; save a2
  emit({0x51});                    // push rcx                 ; save a3
  emit({0x48, 0x83, 0xEC, 0x08});  // sub rsp, 8               ; realign to 16
  emit({0x49, 0xBB});              // movabs r11, realTarget
  emit64(reinterpret_cast<u64>(realTarget));
  emit({0x41, 0xFF, 0xD3});        // call r11                 ; run real fn -> rax=ret
  emit({0x48, 0x83, 0xC4, 0x08});  // add rsp, 8
  emit({0x49, 0x89, 0xC1});        // mov r9, rax              ; logger arg6 = ret
  emit({0x41, 0x58});              // pop r8                   ; a3 -> r8
  emit({0x59});                    // pop rcx                  ; a2 -> rcx
  emit({0x5A});                    // pop rdx                  ; a1 -> rdx
  emit({0x5E});                    // pop rsi                  ; a0 -> rsi
  emit({0xBF});                    // mov edi, imm32
  emit32(hookId);                  //   = hookId
  emit({0x50});                    // push rax                 ; preserve ret across syscall
  emit({0x49, 0x89, 0xCA});        // mov r10, rcx             ; handler reads a2 from r10
  emit({0xB8});                    // mov eax, imm32
  emit32(magic);                   //   = kHostThunkSyscallBase|loggerIdx
  emit({0x0F, 0x05});              // syscall                  ; -> loggerFn(rdi,rsi,rdx,rcx,r8,r9)
  emit({0x58});                    // pop rax                  ; restore realTarget's ret
  emit({0xC3});                    // ret                      ; return to caller with ret
  return reinterpret_cast<uintptr_t>(t);
}

// Wrap a guest function so a NATIVE lock is held across it: syscall(lockFn),
// call realTarget, syscall(unlockFn). Fires before and after, unlike
// makeGuestReturnHook; observes every call DELTA_RIPRACE could only sample.
uintptr_t makeGuestLockWrapper(void *realTarget, void *lockFn, void *unlockFn,
                               const char *name) {
  std::lock_guard lk(g_thunkMutex);
  if (!g_thunkPool) {
    g_thunkPool = static_cast<u8 *>(
        mmap(nullptr, g_thunkPoolSize, PROT_READ | PROT_WRITE | PROT_EXEC,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (g_thunkPool == MAP_FAILED) { g_thunkPool = nullptr; return 0; }
    std::lock_guard rk(g_rangeMutex);
    g_ranges.push_back({reinterpret_cast<u64>(g_thunkPool), g_thunkPoolSize});
  }
  const u32 lockIdx = static_cast<u32>(g_hostThunks.size());
  g_hostThunks.push_back(lockFn);
  g_thunkNames.resize(g_hostThunks.size());
  g_thunkNames[lockIdx] = name ? name : "guestlock";
  const u32 unlockIdx = static_cast<u32>(g_hostThunks.size());
  g_hostThunks.push_back(unlockFn);
  g_thunkNames.resize(g_hostThunks.size());
  g_thunkNames[unlockIdx] = name ? name : "guestunlock";

  constexpr size_t kStride = 64;  // emitted body is 46 bytes
  if (g_thunkPoolUsed + kStride > g_thunkPoolSize) return 0;
  u8 *t = g_thunkPool + g_thunkPoolUsed;
  g_thunkPoolUsed += kStride;
  u8 *p = t;
  auto emit = [&](std::initializer_list<u8> b) { for (u8 x : b) *p++ = x; };
  auto emit32 = [&](u32 v) { std::memcpy(p, &v, 4); p += 4; };
  auto emit64 = [&](u64 v) { std::memcpy(p, &v, 8); p += 8; };
  // Reached by `jmp` from the patched entry: [rsp] is still the original
  // caller's return address. Args are saved around the syscall (a C handler may
  // clobber every caller-saved register), pushes paired for call alignment.
  emit({0x57});                    // push rdi        ; save a0
  emit({0x56});                    // push rsi        ; save a1
  emit({0x52});                    // push rdx        ; save a2
  emit({0x51});                    // push rcx        ; save a3
  emit({0xB8});                    // mov eax, imm32
  emit32(kHostThunkSyscallBase | lockIdx);
  emit({0x0F, 0x05});              // syscall         ; -> lockFn()
  emit({0x59});                    // pop rcx
  emit({0x5A});                    // pop rdx
  emit({0x5E});                    // pop rsi
  emit({0x5F});                    // pop rdi
  emit({0x48, 0x83, 0xEC, 0x08});  // sub rsp, 8      ; realign for the call
  emit({0x49, 0xBB});              // movabs r11, realTarget
  emit64(reinterpret_cast<u64>(realTarget));
  emit({0x41, 0xFF, 0xD3});        // call r11        ; the real function
  emit({0x48, 0x83, 0xC4, 0x08});  // add rsp, 8
  emit({0x50});                    // push rax        ; preserve the return value
  emit({0xB8});                    // mov eax, imm32
  emit32(kHostThunkSyscallBase | unlockIdx);
  emit({0x0F, 0x05});              // syscall         ; -> unlockFn()
  emit({0x58});                    // pop rax
  emit({0xC3});                    // ret
  return reinterpret_cast<uintptr_t>(t);
}

// Callable copy of a guest function whose prologue an entry detour overwrites:
// [relocated prologue] + [abs jmp to continueAt]. prologueLen must be
// position-independent and cover >= 14 bytes on an instruction boundary.
uintptr_t makeGuestTrampoline(const void *fnBytes, u32 prologueLen,
                              const void *continueAt) {
  std::lock_guard lk(g_thunkMutex);
  if (!g_thunkPool) {
    g_thunkPool = static_cast<u8 *>(
        mmap(nullptr, g_thunkPoolSize, PROT_READ | PROT_WRITE | PROT_EXEC,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (g_thunkPool == MAP_FAILED) { g_thunkPool = nullptr; return 0; }
    std::lock_guard rk(g_rangeMutex);
    g_ranges.push_back({reinterpret_cast<u64>(g_thunkPool), g_thunkPoolSize});
  }
  const size_t need = prologueLen + 14;
  const size_t stride = (need + 15) & ~size_t(15);
  if (g_thunkPoolUsed + stride > g_thunkPoolSize) return 0;
  u8 *t = g_thunkPool + g_thunkPoolUsed;
  g_thunkPoolUsed += stride;
  std::memcpy(t, fnBytes, prologueLen);           // relocated prologue
  u8 *p = t + prologueLen;
  *p++ = 0xFF; *p++ = 0x25;                        // jmp qword [rip+0]
  u32 zero = 0; std::memcpy(p, &zero, 4); p += 4;
  u64 cont = reinterpret_cast<u64>(continueAt);
  std::memcpy(p, &cont, 8);
  return reinterpret_cast<uintptr_t>(t);
}

u64 currentGuestRip() {
  return t_curThread ? t_curThread->CurrentFrame->State.rip : 0;
}

// Guest fs base of the thread on this host thread, so a native logger can
// read guest TLS without emitting fs-relative code.
u64 currentGuestFsBase() {
  return t_curThread ? t_curThread->CurrentFrame->State.fs_cached : 0;
}

// Every live thread's fs base, to survey guest TLS (the BPE worker ordinal at
// [*(fsbase)-0x10] decides which job affinities are claimable).
void guestThreadFsBases(std::vector<u64> &out) {
  out.clear();
  std::lock_guard lk(g_liveMutex);
  out.reserve(g_live.size());
  for (auto &t : g_live)
    if (t.thread && t.thread->CurrentFrame)
      out.push_back(t.thread->CurrentFrame->State.fs_cached);
}

const u64 *currentGuestGregs() {
  return t_curThread ? t_curThread->CurrentFrame->State.gregs : nullptr;
}

bool guestGregsFromSignal(const void *ucontext, u64 out[16]) {
#if defined(__aarch64__)
  if (!ucontext || !t_curThread)
    return false;
  const auto *uc = static_cast<const ucontext_t *>(ucontext);
  // Only meaningful inside JIT'd code; same test the RIP reconstruction uses.
  if (!reconstructGuestRip(uc->uc_mcontext.pc))
    return false;
  // FEX's arm64 backend keeps every guest GPR in a fixed host register
  // (x64::SRA), exact at the faulting pc unlike the block-boundary gregs; kSra
  // mirrors SRA element for element.
  static constexpr int kSra[16] = {4,  7,  5,  6,  8,  9,  10, 11,
                                   12, 13, 14, 15, 16, 17, 19, 29};
  for (int i = 0; i < 16; i++)
    out[i] = uc->uc_mcontext.regs[kSra[i]];
  return true;
#else
  (void)ucontext;
  (void)out;
  return false;
#endif
}

int faultingSyscall() { return t_inSyscall ? static_cast<int>(t_lastSyscall) : -1; }

void dumpThreadTrace(void *fileStar) {
  auto *f = static_cast<std::FILE *>(fileStar);
  if (!f)
    return;
  std::fprintf(f, "  --- last guest->host calls (this thread, oldest first) ---\n");
  u32 count = t_tracePos < kTraceRing ? t_tracePos : kTraceRing;
  u32 start = t_tracePos - count;
  for (u32 i = 0; i < count; i++) {
    const TraceEvt &e = t_trace[(start + i) % kTraceRing];
    if (e.kind == 's') {
      std::fprintf(f, "  sc  %3u %-22s (%#llx,%#llx,%#llx,%#llx) -> %#llx\n",
                   e.id, krnl::syscall_getname(e.id),
                   (unsigned long long)e.a0, (unsigned long long)e.a1,
                   (unsigned long long)e.a2, (unsigned long long)e.a3,
                   (unsigned long long)e.ret);
    } else if (e.kind == 'h') {
      char cs[256];
      symRange(e.caller, cs, sizeof(cs));
      std::fprintf(f, "  hle %s(%#llx,%#llx,%#llx,%#llx) -> %#llx  from %s\n",
                   e.name ? e.name : "?", (unsigned long long)e.a0,
                   (unsigned long long)e.a1, (unsigned long long)e.a2,
                   (unsigned long long)e.a3, (unsigned long long)e.ret, cs);
    }
  }
}

u64 reconstructGuestRip(u64 hostPC) {
  if (!g_ctxPtr || !t_curThread)
    return 0;
  if (!g_ctxPtr->IsAddressInCodeBuffer(t_curThread, hostPC))
    return 0;
  return g_ctxPtr->RestoreRIPFromHostPC(t_curThread, hostPC);
}

bool tryHandleJitSignal(int sig, void *infop, void *ucv) {
#if defined(__aarch64__)
  if (!g_ctxPtr || !t_curThread || !ucv || !infop)
    return false;
  auto *uc = static_cast<ucontext_t *>(ucv);

  // Guest call/ret that do not pair up (fiber switches) drift FEX's call-ret
  // predictor into its guard pages; upstream FEX treats that as expected and
  // resets x25. Mirror it, or SotC's job system dies mid-LoadInitialWorld.
  if (sig == SIGSEGV && t_curThread->CallRetStackBase) {
    const u64 fa =
        reinterpret_cast<u64>(static_cast<siginfo_t *>(infop)->si_addr);
    const u64 crBase =
        reinterpret_cast<u64>(t_curThread->CallRetStackBase);
    constexpr size_t kCrSize =
        FEXCore::Core::InternalThreadState::CALLRET_STACK_SIZE;
    constexpr size_t kPage = 0x1000;
    if (fa >= crBase - kPage && fa < crBase + kCrSize + kPage) {
      uc->uc_mcontext.regs[25] = crBase + kCrSize / 4;
      static std::atomic<u32> n{0};
      u32 c = n.fetch_add(1);
      if (c < 8 || (c & (c - 1)) == 0)  // first few, then powers of two
        BASE_LOGI("fex",
                  "callret predictor over/underflow #{} reset (fault {:#x})",
                  c, (unsigned long long)fa);
      return true;
    }
  }

  // FEX raises SIGBUS(BUS_ADRALN) from the JIT for unaligned atomics and
  // backpatches them; mirror its frontend handler.
  if (sig != SIGBUS)
    return false;
  const u64 pc = uc->uc_mcontext.pc;
  if (!g_ctxPtr->IsAddressInCodeBuffer(t_curThread, pc))
    return false;
  if (static_cast<siginfo_t *>(infop)->si_code != BUS_ADRALN)
    return false;
  auto result = FEXCore::ArchHelpers::Arm64::HandleUnalignedAccess(
      t_curThread, FEXCore::ArchHelpers::Arm64::UnalignedHandlerType::HalfBarrier,
      pc, reinterpret_cast<u64 *>(&uc->uc_mcontext.regs[0]));
  // A backpatched atomic stops being atomic (HalfBarrier); log each site once.
  static std::mutex logM;
  static std::set<u64> seenRips;
  const u64 grip = reconstructGuestRip(pc);
  {
    std::lock_guard<std::mutex> lk(logM);
    if (seenRips.insert(grip).second)
      BASE_LOGI("fex",
                "unaligned-atomic backpatch site guest rip={:#x} ({} sites)",
                (unsigned long long)grip, (unsigned)seenRips.size());
  }
  uc->uc_mcontext.pc = pc + result.value_or(0);
  return result.has_value();
#else
  (void)sig; (void)infop; (void)ucv;
  return false;
#endif
}

} // namespace cpu

// Guest fs base lives in the current FEXCore thread's CPUState; called via
// sys_sysarch(AMD64_SET_FSBASE) and on thread spawn.
namespace krnl {
void setThreadFsBase(u64 v) {
  if (cpu::t_curThread)
    cpu::t_curThread->CurrentFrame->State.fs_cached = v;
}
u64 threadFsBase() {
  return cpu::t_curThread ? cpu::t_curThread->CurrentFrame->State.fs_cached : 0;
}
} // namespace krnl

#endif // DELTA_BACKEND_FEX
