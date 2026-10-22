#include "cpu/aarch64/monitors.h"

#include <sys/mman.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/X86Enums.h>
#include <FEXCore/Debug/InternalThreadState.h>

#include "base/containers/pair.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/strings/format.h"
#include "base/strings/xstring.h"
#include "cpu/aarch64/code_map.h"
#include "cpu/aarch64/crossing_trace.h"
#include "cpu/aarch64/live_threads.h"
#include "cpu/backend.h"
#include "guest/session.h"
#include "kern/lv2/dispatch.h"
#include "kern/module.h"
#include "options/options.h"

namespace {
DELTA_OPTION(int, kSampleMs, "DELTA_SAMPLE_MS", 0);
DELTA_OPTION(const char*, kRipRace, "DELTA_RIPRACE", nullptr);
DELTA_OPTION(int, kLoadWatch, "DELTA_LOADWATCH", 0);
DELTA_OPTION(u64, kLoadWatchBase, "DELTA_LOADWATCH_BASE", 0x201400000000ull);
DELTA_OPTION(u64, kLoadWatchGoff, "DELTA_LOADWATCH_GOFF", 0x2ee2d00ull);
DELTA_OPTION(bool, kWatchdog, "DELTA_WATCHDOG", false);
}  // namespace

namespace cpu::fex {
namespace {

bool IsPageResident(u64 address) {
  unsigned char mv = 0;
  long pg = sysconf(_SC_PAGESIZE);
  return mincore(reinterpret_cast<void*>(address & ~((u64)pg - 1)), 1, &mv) ==
         0;
}

void StartRipSampler(int ms) {
  if (ms <= 0)
    ms = 50;
  guest::SpawnThread("fex_backend", [ms] {
    for (u64 tick = 0;; tick++) {
      guest::SleepForMilliseconds(ms);
      VisitLiveThreads([&](const base::Vector<LiveThread>& live) {
        for (auto& t : live) {
          auto& s = t.thread->CurrentFrame->State;
          char sym[200];
          SymbolizeModuleAddress(s.rip, sym, sizeof(sym));
          BASE_LOGI("smp", "{} tid={} rip={:#x} {}", (unsigned long long)tick,
                    t.id, (unsigned long long)s.rip, sym);
        }
      });
      std::fflush(stderr);
    }
  });
}

// DELTA_RIPRACE=<ms>:<lo>-<hi>[,...] (hex guest VAs). State.rip is stale
// inside the JIT, so each thread is signalled to reconstruct its own.
void StartRipRace(const char* spec_text) {
  base::String spec(spec_text);
  int ms = 1;
  base::Vector<base::Pair<u64, u64>> ranges;
  const size_t colon = spec.find(':');
  if (colon != base::String::npos) {
    ms = std::atoi(spec.substr(0, colon).c_str());
    if (ms <= 0)
      ms = 1;
    const base::String rest = spec.substr(colon + 1);
    size_t i = 0;
    while (i < rest.size()) {
      const size_t comma = rest.find(',', i);
      const base::String one =
          rest.substr(i, comma == base::String::npos ? comma : comma - i);
      i = comma == base::String::npos ? rest.size() : comma + 1;
      const size_t dash = one.find('-');
      if (dash == base::String::npos)
        continue;
      const u64 lo = std::strtoull(one.c_str(), nullptr, 16);
      const u64 hi = std::strtoull(one.c_str() + dash + 1, nullptr, 16);
      if (lo && hi > lo)
        ranges.push_back({lo, hi});
    }
  }
  if (ranges.empty())
    return;
  struct sigaction sa {};
  sa.sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = [](int, siginfo_t*, void* ucv) {
    auto* uc = static_cast<ucontext_t*>(ucv);
    const u64 rip = ReconstructGuestRip(uc->uc_mcontext.pc);
    struct timespec ts {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    RecordRipSample(rip, (u64)ts.tv_sec * 1000000000ull + ts.tv_nsec);
  };
  sigaction(SIGPROF, &sa, nullptr);
  guest::SpawnThread("fex_backend", [ms, ranges] {
    u64 rounds = 0, with_one = 0, with_two = 0, max_seen = 0, reported = 0,
        answered = 0, with_two_tight = 0;
    for (;;) {
      guest::SleepForMilliseconds(ms);
      const u64 gen = BeginRipSampleRound();
      struct Slot {
        u32 id;
        pid_t tid;
        base::Atomic<u64>*gp, *rp, *np;
      };
      Slot slots[64];
      unsigned ns = 0;
      VisitLiveThreads([&](const base::Vector<LiveThread>& live) {
        for (auto& t : live) {
          if (ns >= 64 || !t.sample_gen || !t.sample_rip || !t.host_tid)
            continue;
          slots[ns++] = {t.id, t.host_tid, t.sample_gen, t.sample_rip,
                         t.sample_ns};
        }
      });
      for (unsigned k = 0; k < ns; k++)
        ::syscall(SYS_tgkill, ::getpid(), slots[k].tid, SIGPROF);
      guest::SleepForMicroseconds(300);
      struct Hit {
        u32 id;
        u64 rip;
        u64 ns;
      };
      Hit hits[64];
      unsigned n = 0;
      for (unsigned k = 0; k < ns; k++) {
        if (slots[k].gp->load(base::memory_order_acquire) != gen)
          continue;
        answered++;
        const u64 rip = slots[k].rp->load(base::memory_order_relaxed);
        for (auto& r : ranges)
          if (rip >= r.first && rip < r.second) {
            hits[n++] = {
                slots[k].id, rip,
                slots[k].np ? slots[k].np->load(base::memory_order_relaxed)
                            : 0};
            break;
          }
      }
      rounds++;
      if (n > max_seen)
        max_seen = n;
      if (n == 1)
        with_one++;
      if (n >= 2) {
        with_two++;
        u64 lo = hits[0].ns, hi = hits[0].ns;
        for (unsigned k = 1; k < n; k++) {
          lo = base::Min(lo, hits[k].ns);
          hi = base::Max(hi, hits[k].ns);
        }
        const u64 spread_ns = hi - lo;
        if (spread_ns <= 1000)
          with_two_tight++;
        if (reported++ < 40) {
          base::String line;
          base::FormatTo(line, "{} threads inside, samples spread {} ns{}:", n,
                         (unsigned long long)spread_ns,
                         spread_ns <= 1000 ? "  <== SIMULTANEOUS" : "");
          for (unsigned k = 0; k < n; k++) {
            char sym[160];
            SymbolizeModuleAddress(hits[k].rip, sym, sizeof(sym));
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
                  (unsigned long long)rounds, (unsigned long long)answered,
                  (unsigned long long)with_one, (unsigned long long)with_two,
                  (unsigned long long)with_two_tight,
                  (unsigned long long)max_seen);
      std::fflush(stderr);
    }
  });
}

// SotC's 10 load queues: count at obj+0x110+i*0x28, obj = *(base+0x2ee2d00).
void StartLoadWatch(int ms, u64 image_base, u64 goff) {
  if (ms <= 0)
    ms = 2000;
  guest::SpawnThread("fex_backend", [ms, image_base, goff] {
    auto rd = [](u64 a, void* dst, size_t n) -> bool {
      long pg = sysconf(_SC_PAGESIZE);
      for (u64 p = a & ~((u64)pg - 1); p < a + n; p += pg)
        if (!IsPageResident(p))
          return false;
      std::memcpy(dst, reinterpret_cast<void*>(a), n);
      return true;
    };
    u64 pobj = image_base + goff;
    int last_total = -1, plateau = 0;
    for (u64 tick = 0;; tick++) {
      guest::SleepForMilliseconds(ms);
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
        if (!rd(obj + 0x110 + (u64)i * 0x28, &v, 4)) {
          ok = false;
          break;
        }
        c[i] = v;
        total += v;
      }
      if (!ok) {
        BASE_LOGI("loadwatch", "{} obj={:#x} read fault",
                  (unsigned long long)tick, (unsigned long long)obj);
        std::fflush(stderr);
        continue;
      }
      if (total == last_total)
        plateau++;
      else
        plateau = 0;
      last_total = total;
      BASE_LOGI("loadwatch",
                "{} obj={:#x} REMAINING={} plateau={}x q=[{} {} {} {} {} {} {} "
                "{} {} {}]",
                (unsigned long long)tick, (unsigned long long)obj, total,
                plateau, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8],
                c[9]);
      std::fflush(stderr);
    }
  });
}

void LogUmutexOwner(const base::Vector<LiveThread>& live,
                    const LiveThread& waiter,
                    u64 umutex) {
  if (!IsPageResident(umutex))
    return;
  u32 ow = *reinterpret_cast<volatile u32*>(umutex);
  u32 owner_tid = ow & 0x7fffffff;
  BASE_LOGI("watchdog", "      ^ umutex {:#x} word={:#x} owner-tid={}{}",
            (unsigned long long)umutex, ow, owner_tid,
            (ow & 0x80000000u) ? " CONTESTED" : "");
  for (auto& o : live) {
    if (!o.gtid || *o.gtid != owner_tid || &o == &waiter)
      continue;
    const TraceEvt* ot = o.trace;
    u32 opos = o.trace_pos ? *o.trace_pos : 0;
    const char* osc = "?";
    u64 oa0 = 0, oa1 = 0;
    u32 oid = 0;
    if (ot && opos) {
      const TraceEvt& le = ot[(opos - 1) % kTraceRing];
      if (le.kind == 's') {
        oid = le.id;
        osc = kern::SyscallGetname(le.id);
        oa0 = le.a0;
        oa1 = le.a1;
      }
    }
    BASE_LOGI("watchdog",
              "      ^^ OWNER is watchdog tid={} rip={:#x} scN={} last: sc {} "
              "{} ({:#x},{:#x})",
              o.id, (unsigned long long)o.thread->CurrentFrame->State.rip, opos,
              oid, osc, (unsigned long long)oa0, (unsigned long long)oa1);
    break;
  }
}

void LogStalledThread(const base::Vector<LiveThread>& live,
                      const LiveThread& t) {
  auto& s = t.thread->CurrentFrame->State;
  char sym[256];
  SymbolizeModuleAddress(s.rip, sym, sizeof(sym));
  BASE_LOGI("watchdog", "  tid={} gtid={} rip={:#x} scN={} ({})", t.id,
            t.gtid ? *t.gtid : 0, (unsigned long long)s.rip,
            t.trace_pos ? *t.trace_pos : 0, sym);
  if (t.trace && t.trace_pos) {
    u32 pos = *t.trace_pos;
    u32 cnt = pos < kTraceRing ? pos : kTraceRing;
    u32 from = cnt > 6 ? cnt - 6 : 0;
    for (u32 k = from; k < cnt; k++) {
      const TraceEvt& e = t.trace[(pos - cnt + k) % kTraceRing];
      if (e.kind != 's')
        continue;
      BASE_LOGI("watchdog", "      sc {:3} {:<18} ({:#x},{:#x},{:#x},{:#x}) -> {:#x}",
                e.id, kern::SyscallGetname(e.id), (unsigned long long)e.a0,
                (unsigned long long)e.a1, (unsigned long long)e.a2,
                (unsigned long long)e.a3, (unsigned long long)e.ret);
      // UMTX_OP_MUTEX_WAIT
      if (k == cnt - 1 && e.id == 454 && e.a1 == 17 && e.a0 >= 0x10000)
        LogUmutexOwner(live, t, e.a0);
    }
  }
  u64 rsp = s.gregs[FEXCore::X86State::REG_RSP];
  int shown = 0;
  for (int i = 0; i < 1024 && shown < 12; i++) {
    u64 a = rsp + (u64)i * 8;
    if (a < 0x1000 || !IsPageResident(a))
      break;
    u64 v = 0;
    std::memcpy(&v, reinterpret_cast<void*>(a), 8);
    if (!kern::InModuleRegion(v))
      continue;
    char s2[256];
    SymbolizeModuleAddress(v, s2, sizeof(s2));
    if (s2[0] == '0')
      continue;
    BASE_LOGI("watchdog", "      stk+{:#x} {:#x} ({})", i * 8,
              (unsigned long long)v, s2);
    shown++;
  }
}

void StartWatchdog(int secs) {
  if (secs <= 0)
    secs = 20;
  guest::SpawnThread("fex_backend", [secs] {
    for (int round = 0;; round++) {
      guest::SleepForMilliseconds((secs) * 1000);
      VisitLiveThreads([&](const base::Vector<LiveThread>& live) {
        BASE_LOGI("watchdog", "=== WATCHDOG round {}: {} live guest threads ===",
                  round, live.size());
        for (auto& t : live)
          LogStalledThread(live, t);
      });
      std::fflush(stderr);
    }
  });
}

}  // namespace

void StartMonitors() {
  static const bool kOnce = [] {
    if (kSampleMs)
      StartRipSampler(kSampleMs);
    if (kRipRace)
      StartRipRace(kRipRace);
    if (kLoadWatch)
      StartLoadWatch(kLoadWatch, kLoadWatchBase, kLoadWatchGoff);
    if (kWatchdog)
      StartWatchdog(kWatchdog);
    return true;
  }();
  (void)kOnce;
}

void ReportThreadReturn(u32 id,
                        u64 entry_rip,
                        const FEXCore::Core::CPUState& end) {
  if (kWatchdog) {
    char es[256];
    SymbolizeModuleAddress(entry_rip, es, sizeof(es));
    char rs[256];
    SymbolizeModuleAddress(end.rip, rs, sizeof(rs));
    BASE_LOGI("watchdog",
              "=== THREAD tid={} RETURNED entry={:#x} ({}) ret={:#x} ({}) ===",
              id, (unsigned long long)entry_rip, es, (unsigned long long)end.rip,
              rs);
    if (kern::InModuleRegion(end.rip)) {
      const u8* b = reinterpret_cast<const u8*>(end.rip);
      BASE_LOGI("watchdog",
                "      bytes@rip: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                b[0], b[1], b[2], b[3], b[4], b[5]);
    }
    u64 rsp = end.gregs[FEXCore::X86State::REG_RSP];
    int shown = 0;
    for (int i = 0; i < 2048 && shown < 20; i++) {
      u64 a = rsp + (u64)i * 8, v = 0;
      if (a < 0x1000)
        break;
      std::memcpy(&v, reinterpret_cast<void*>(a), 8);
      if (!kern::InModuleRegion(v))
        continue;
      char s2[256];
      SymbolizeModuleAddress(v, s2, sizeof(s2));
      if (s2[0] == '0')
        continue;
      BASE_LOGI("watchdog", "      stk+{:#x} {:#x} ({})", i * 8,
                (unsigned long long)v, s2);
      shown++;
    }
  }
  if (end.rip < 0x100000ull) {
    BASE_LOGI("fex", "=== BOGUS THREAD RETURN rip={:#x} ===",
              (unsigned long)end.rip);
    const char* rn[] = {"rax", "rbx", "rcx", "rdx", "rsi", "rdi",
                        "rbp", "rsp", "r8",  "r9",  "r10", "r11",
                        "r12", "r13", "r14", "r15"};
    for (int i = 0; i < 16; i++)
      BASE_LOGI("fex", "  {}={:#x}", rn[i], (unsigned long)end.gregs[i]);
    u64 rsp = end.gregs[FEXCore::X86State::REG_RSP];
    if (rsp) {
      for (int i = 0; i < 64; i++) {
        u64 v = reinterpret_cast<const u64*>(rsp)[i];
        if (kern::InModuleRegion(v))
          BASE_LOGI("fex", "  stk+{:#x} = {:#x}", i * 8, (unsigned long)v);
      }
    }
  }
}

}  // namespace cpu::fex
