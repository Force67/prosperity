
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <base/strings/format.h>
#include <base/strings/xstring.h>

#include "wait_probe.h"
#include "kern/thread_names.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <pthread.h>
#include <unordered_map>
#include <vector>

#include <utl/mem.h>

#include "kern/crash.h"
#include "kern/module.h"
#include "kern/proc.h"
#include "cpu/cpu_backend.h"
#include "sys_thread.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(const char *, kHostStackMb, "DELTA_HOST_STACK_MB", nullptr);
DELTA_OPTION(long, kUmtxTimeoutMs, "DELTA_UMTX_TIMEOUT_MS", 2);
DELTA_OPTION(const char *, kAddrWatch, "DELTA_UMTX_ADDRWATCH", nullptr);
DELTA_OPTION(u32, kAddrWatchLimit, "DELTA_UMTX_ADDRWATCH_MAX", 200000);
DELTA_OPTION(u64, kCvTrace, "DELTA_UMTX_CVTRACE", 0);
DELTA_OPTION(bool, kNoThrBarrier, "DELTA_NO_THR_BARRIER", false);
DELTA_OPTION(bool, kUmtxHist, "DELTA_UMTX_HIST", false);
DELTA_OPTION(bool, kUmtxTrace, "DELTA_UMTX_TRACE", false);
// DELTA_UMTX_INJECT_NS: burn N ns inside MUTEX_WAIT to test if the op is on the critical path.
DELTA_OPTION(long, kUmtxInjectNs, "DELTA_UMTX_INJECT_NS", 0);
// DELTA_UMTX_STALL=<s>: report a wait that outlives it (object word + waiting guest code).
DELTA_OPTION(long, kUmtxStallSecs, "DELTA_UMTX_STALL", 0);
}  // namespace

namespace krnl {
moduleInfo *called_in(void *addr);

// Initial-exec TLS for fast, allocation-free access; the Android dlopen'd .so
// rejects it (STATIC_TLS), so fall back to global-dynamic there.
#if defined(__ANDROID__) && defined(DELTA_ANDROID_APP)
#define DELTA_TLS_IE
#else
#define DELTA_TLS_IE __attribute__((tls_model("initial-exec")))
#endif

// Per-thread guest thread id (sys_thr_self). Main thread is 1.
static DELTA_TLS_IE thread_local u32 t_tid = 1;
static std::atomic<u32> g_nextTid{2};

// Calling thread's guest tid TLS; the FEX watchdog maps umutex owner words to
// the thread that holds the lock with it.
const u32 *currentGuestTidPtr() { return &t_tid; }

// guest tid -> host tid. A umutex owner word names a guest thread; finding out
// what that thread is doing means finding its OS thread first.
static std::atomic<long> g_hostByGuest[4096];

void noteGuestThreadHost(u32 gtid) {
  if (gtid < 4096)
    g_hostByGuest[gtid].store(static_cast<long>(::syscall(SYS_gettid)),
                              std::memory_order_relaxed);
}

long hostTidForGuest(u32 gtid) {
  return gtid < 4096 ? g_hostByGuest[gtid].load(std::memory_order_relaxed) : 0;
}

// Thread-startup handshake: thr_new blocks until the new thread reaches its
// first sync point, so the main thread doesn't read not-yet-produced shared state.
static std::mutex g_startM;
static std::condition_variable g_startCv;
static DELTA_TLS_IE thread_local std::atomic<bool> *t_started = nullptr;

static void markThreadStarted() {
  if (t_started && !t_started->exchange(true)) {
    std::lock_guard<std::mutex> lk(g_startM);
    g_startCv.notify_all();
  }
}

// FreeBSD thr_param (rdi layout, 0x68 bytes); +40 tls_size / +64 flags exist
// for the user-mode libthr wrapper, untouched by the kernel path.
struct thr_param {
  void(PS4ABI *start_func)(void *);  // +0x00
  void *arg;                          // +0x08
  u8 *stack_base;                // +0x10
  size_t stack_size;                  // +0x18
  u8 *tls_base;                  // +0x20
  size_t tls_size;                    // +0x28  (not read by kern_thr_new)
  i64 *child_tid;                 // +0x30
  i64 *parent_tid;                // +0x38
  i32 flags;                      // +0x40  (not read by kern_thr_new)
  i32 pad;
  void *rtp;                          // +0x48  rtprio (4-byte struct copied in)
  const char *name;                   // +0x50  thread name (max 32 chars)
  void *spare[2];                     // +0x58
};

int PS4ABI sys_thr_new(thr_param *p, int size) {
  // The kernel rejects an oversized param block; param_size == 0x68 for every libthr.
  if (!p || size < 0 || static_cast<size_t>(size) > sizeof(thr_param))
    return -22 /*EINVAL*/;
  u32 tid = g_nextTid.fetch_add(1);
  // start_func is libkernel's pthread trampoline; the real routine is a field
  // of the pthread object passed as arg. Report its first .text pointer.
  char entry[256] = "?";
  if (p->arg) {
    auto *w = static_cast<const uintptr_t *>(p->arg);
    for (int i = 0; i < 64; i++) {
      char sym[256];
      symbolize(w[i], sym, sizeof(sym));
      if (std::strstr(sym, "(.text)")) {
        std::snprintf(entry, sizeof(entry), "%s", sym);
        break;
      }
    }
  }
  BASE_LOGI("thr_new",
            "tid={} start={:p} arg={:p} stack={:p}+{:#x} tls={:p} entry={}",
            tid, (void *)p->start_func, p->arg, (void *)p->stack_base,
            p->stack_size, (void *)p->tls_base, entry);

  if (p->child_tid)
    *p->child_tid = tid;
  if (p->parent_tid)
    *p->parent_tid = tid;

  // libthr switches RSP onto this caller-provided stack immediately, so every
  // page of it must be backed first (P.T. worker faulted at stack_top-0x18).
  // Only currently-FREE pages are filled; guards and live stacks are untouched.
  if (auto *proc = proc::getActive(); proc && p->stack_base && p->stack_size) {
    constexpr size_t kPage = 0x4000;  // PS4 16 KiB page
    const auto sb = reinterpret_cast<uintptr_t>(p->stack_base);
    const auto lo = sb & ~(kPage - 1);
    const auto hi = (sb + p->stack_size + kPage - 1) & ~(kPage - 1);
    int filled = 0;
    for (uintptr_t a = lo; a < hi; a += kPage) {
      void *pg = reinterpret_cast<void *>(a);
      if (!utl::allocMem(pg, kPage, utl::pageProtection::w,
                         utl::allocationType::reserve))
        continue;  // page already mapped -> leave it (guard / live stack)
      void *c = utl::allocMem(pg, kPage, utl::pageProtection::w,
                              utl::allocationType::commit);
      if (c) {
        utl::protectMem(c, kPage, utl::pageProtection::rwx);
        // Only register pages the VMA doesn't know; re-adding would fragment
        // the stack's existing entry.
        if (!proc->getVma().get(static_cast<u8 *>(c)))
          proc->getVma().add(static_cast<u8 *>(c), kPage,
                             utl::pageProtection::w);
        filled++;
      }
    }
    if (filled)
      BASE_LOGI("thr_new", "tid={} backed {} unmapped stack page(s) for {:p}+{:#x}",
                tid, filled, (void *)p->stack_base, p->stack_size);
  }

  auto fn = p->start_func;
  auto arg = p->arg;
  auto fsbase = reinterpret_cast<u64>(p->tls_base);
  auto started = std::make_shared<std::atomic<bool>>(false);

  // Run on the host thread's large stack, not the guest's (64 KiB overflows
  // our syscall handlers); the guest thread is created on THIS thread as FEX
  // requires, the worker host thread only runs it.
  void *gthread =
      cpu::backend().createGuestThread(reinterpret_cast<uintptr_t>(fn), arg, fsbase);

  // DELTA_HOST_STACK_MB=<N>: guest workers get an N-MiB native stack; deep BPE
  // streaming jobs blow the 8 MiB glibc default ~9s into LoadInitialWorld.
  auto *gsb = p->stack_base;
  const size_t gss = p->stack_size;
  const char *hsEnv = kHostStackMb;
  if (hsEnv) {
    auto *ctx = new std::tuple<void *, u32, std::shared_ptr<std::atomic<bool>>,
                               void *, size_t>(gthread, tid, started, gsb, gss);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    size_t hostStack = (size_t)std::strtoull(hsEnv, nullptr, 0) * 1024 * 1024;
    if (hostStack < 8ull * 1024 * 1024) hostStack = 256ull * 1024 * 1024;
    pthread_attr_setstacksize(&attr, hostStack);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    auto trampoline = +[](void *pv) -> void * {
      auto *c = static_cast<std::tuple<void *, u32,
                                       std::shared_ptr<std::atomic<bool>>,
                                       void *, size_t> *>(pv);
      t_tid = std::get<1>(*c);
      noteGuestThreadHost(t_tid);
      t_started = std::get<2>(*c).get();
      void *gt = std::get<0>(*c);
      registerGuestThreadStack(std::get<3>(*c), std::get<4>(*c));
      delete c;
      cpu::backend().runGuestThread(gt);
      unregisterGuestThreadStack();
      return nullptr;
    };
    pthread_t th;
    if (pthread_create(&th, &attr, trampoline, ctx) != 0) {
      delete ctx;
      std::thread([gthread, tid, started, gsb, gss] {
        t_tid = tid;
        noteGuestThreadHost(t_tid);
        t_started = started.get();
        registerGuestThreadStack(gsb, gss);
        cpu::backend().runGuestThread(gthread);
        unregisterGuestThreadStack();
      }).detach();
    }
    pthread_attr_destroy(&attr);
  } else {
    std::thread([gthread, tid, started, gsb, gss] {
      t_tid = tid;
      noteGuestThreadHost(t_tid);
      t_started = started.get();
      registerGuestThreadStack(gsb, gss);
      cpu::backend().runGuestThread(gthread);
      unregisterGuestThreadStack();
    }).detach();
  }

  // Wait for the new thread's first sync point so it wins the races the game
  // expects; bounded, DELTA_NO_THR_BARRIER disables (for a spawner that is
  // itself the producer).
  if (!kNoThrBarrier) {
    std::unique_lock<std::mutex> lk(g_startM);
    g_startCv.wait_for(lk, std::chrono::milliseconds(200),
                       [&] { return started->load(); });
  }
  return 0;
}

int PS4ABI sys_thr_self(i64 *tid) {
  if (!tid)
    return -14 /*EFAULT*/;
  // The kernel stores td_tid through suword64 (64-bit); writing 32 bits left
  // stack garbage in the caller's high word.
  *tid = t_tid;
  return 0;
}

int PS4ABI sys_rtprio_thread(int function, u64 lwpid, thread_prio *rtp) {
  if (!rtp)
    return -14 /*EFAULT*/;
  // RTP_SET applies the caller's priority; we don't model scheduling, so accept
  // it unchanged and report a normal class on LOOKUP.
  constexpr int RTP_SET = 1;
  if (function == RTP_SET)
    return 0;
  rtp->type = 3; /*RTP_PRIO_NORMAL: time-sharing*/
  rtp->prio = 1; /*almost highest prio*/
  return 0;
}

// Address-keyed wait/wake (a small futex). Fixed buckets; hash collisions only
// cause harmless spurious host-condvar wakeups, never spurious guest returns
// (every wait re-checks its own predicate before returning).
namespace {
// Per-address wake state. Simple WAIT uses an explicit queue (WAKE's val limits
// releases); unordered_map keeps references stable across inserts.
struct SimpleWaiter {
  SimpleWaiter *prev = nullptr;
  SimpleWaiter *next = nullptr;
  bool queued = false;
  bool selected = false;
};
struct WaitChan {
  u64 gen = 0;      // broadcast generation for mutex/CV/semaphore waits
  u64 signals = 0;  // pending single-waiter releases (CV_SIGNAL)
  u32 waiters = 0;  // live CV_WAIT sleepers (drives ucond c_has_waiters)
  // A signal releases a sleeper QUEUED WHEN IT WAS SENT: each waiter takes a
  // ticket, only tickets below the cutoff may consume (Astro Bot's shared
  // condvar deadlocked when a late waiter stole the wake).
  u64 nextTicket = 0;
  u64 signalCutoff = 0;
  // Live umutex sleepers; FreeBSD's umtxq_count decides UNOWNED vs CONTESTED
  // on release, so the count must be tracked, not guessed.
  u32 mutexWaiters = 0;
  SimpleWaiter *simpleHead = nullptr;
  SimpleWaiter *simpleTail = nullptr;

  void queueSimple(SimpleWaiter &waiter) {
    waiter.prev = simpleTail;
    if (simpleTail)
      simpleTail->next = &waiter;
    else
      simpleHead = &waiter;
    simpleTail = &waiter;
    waiter.queued = true;
  }

  void removeSimple(SimpleWaiter &waiter) {
    if (!waiter.queued)
      return;
    if (waiter.prev)
      waiter.prev->next = waiter.next;
    else
      simpleHead = waiter.next;
    if (waiter.next)
      waiter.next->prev = waiter.prev;
    else
      simpleTail = waiter.prev;
    waiter.prev = nullptr;
    waiter.next = nullptr;
    waiter.queued = false;
  }

  u32 wakeSimple(u64 requested) {
    // FreeBSD narrows val to int and its queue loop wakes one even for <= 0.
    const i32 count = static_cast<i32>(requested);
    u32 remaining = count > 1 ? static_cast<u32>(count) : 1;
    u32 woken = 0;
    while (simpleHead && remaining-- > 0) {
      auto *waiter = simpleHead;
      removeSimple(*waiter);
      waiter->selected = true;
      ++woken;
    }
    return woken;
  }
};
struct Bucket {
  std::mutex m;
  std::condition_variable cv;
  std::unordered_map<const void *, WaitChan> chan;
};
std::array<Bucket, 256> g_umtxBuckets;
Bucket &umtxBucket(const void *a) {
  return g_umtxBuckets[(reinterpret_cast<uintptr_t>(a) >> 4) & 0xff];
}
constexpr u32 UMUTEX_CONTESTED = 0x80000000u;
// Re-poll tick: some guest code publishes its predicate with a plain store
// and no wake syscall (Doom64's job scheduler stalled ~1s/frame). The tick is
// re-poll only; no wait returns to the guest because of it (engines deref
// half-built state on a spurious return).
std::chrono::milliseconds umtxTimeout() {
  return std::chrono::milliseconds(kUmtxTimeoutMs);
}

// One report per stalled wait; the object word names who it waits for
// (umutex low 31 bits = owner tid).
struct StallReport {
  const char *op;
  const void *obj;
  const void *obj2 = nullptr;  // a CV wait's umutex, whose owner is the other half
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  bool done = kUmtxStallSecs <= 0;

  void tick() {
    if (done)
      return;
    const auto waited = std::chrono::steady_clock::now() - start;
    if (waited < std::chrono::seconds(kUmtxStallSecs))
      return;
    done = true;
    const u32 word = obj ? *static_cast<const volatile u32 *>(obj) : 0;
    const u32 word2 = obj2 ? *static_cast<const volatile u32 *>(obj2) : 0;
    BASE_LOGI("umtxstall",
              "{} obj={:p} word={:#x} (owner gtid={}) mutex={:p} word={:#x} "
              "(owner gtid={}) waited {}s",
              op, obj, word, word & 0x7fffffffu, obj2, word2,
              word2 & 0x7fffffffu,
              (long long)std::chrono::duration_cast<std::chrono::seconds>(
                  waited).count());
    // Name what the OWNER is stuck in; that wait is the other half of the cycle.
    if (std::strcmp(op, "MUTEX_WAIT") == 0) {
      const u32 owner_gtid = word & 0x7fffffffu;
      char owner[128];
      if (waitProbeDescribeGuest(owner_gtid, owner, sizeof(owner)))
        BASE_LOGI("umtxstall", "  owner gtid={} is parked in {}", owner_gtid,
                  owner);
      else
        BASE_LOGI("umtxstall",
                  "  owner gtid={} (host tid {}) is in no probed wait",
                  owner_gtid, hostTidForGuest(owner_gtid));
    }
    guestStackTrace("umtxstall", 8);
  }
};

// WAIT-class ops take an optional relative timeout at uaddr2 (FreeBSD 9
// timespec; NULL = wait forever).
struct GuestTimespec {
  i64 sec;
  i64 nsec;
};
using SteadyTp = std::chrono::steady_clock::time_point;
std::optional<SteadyTp> umtxRelDeadline(const void *b) {
  if (!b)
    return std::nullopt;
  auto *ts = static_cast<const GuestTimespec *>(b);
  auto d = std::chrono::seconds(ts->sec) + std::chrono::nanoseconds(ts->nsec);
  if (d < d.zero())
    d = d.zero();
  return std::chrono::steady_clock::now() +
         std::chrono::duration_cast<std::chrono::steady_clock::duration>(d);
}

// CV_WAIT's val argument carries flags: with CVWAIT_ABSTIME the timespec is
// absolute on the ucond's c_clockid clock; convert to a steady deadline.
constexpr u64 kCvWaitAbsTime = 0x02;
std::optional<SteadyTp> cvDeadline(const void *ucond, u64 flags,
                                   const void *b) {
  if (!b)
    return std::nullopt;
  if (!(flags & kCvWaitAbsTime))
    return umtxRelDeadline(b);
  auto *ts = static_cast<const GuestTimespec *>(b);
  const u32 clockid = *reinterpret_cast<const u32 *>(
      static_cast<const u8 *>(ucond) + 8);  // ucond.c_clockid
  struct timespec now {};
  clock_gettime(clockid == 0 ? CLOCK_REALTIME : CLOCK_MONOTONIC, &now);
  auto rel = std::chrono::seconds(ts->sec - now.tv_sec) +
             std::chrono::nanoseconds(ts->nsec - now.tv_nsec);
  if (rel < rel.zero())
    rel = rel.zero();
  return std::chrono::steady_clock::now() +
         std::chrono::duration_cast<std::chrono::steady_clock::duration>(rel);
}

void threadComm(char *out, size_t n) {
  out[0] = '\0';
  if (std::FILE *f = std::fopen("/proc/thread-self/comm", "r")) {
    if (std::fgets(out, static_cast<int>(n), f))
      if (char *nl = std::strchr(out, '\n'))
        *nl = '\0';
    std::fclose(f);
  }
}

// DELTA_UMTX_ADDRWATCH=<hex addr>[:<window>][,...] (or =1 for the built-in
// probe address): watch address windows, since CVTRACE only sees ops 8/9/10 on
// one exact pointer and misses wakes through other ops or neighbouring fields.
constexpr u64 kAddrWatchWindow = 0x200;   // default +/- around each addr
constexpr u64 kAddrWatchDefault = 0x2000040a4ull;
constexpr size_t kAddrWatchMax = 8;

struct AddrWatchSpec {
  u64 lo, hi;
};
struct AddrWatchList {
  AddrWatchSpec v[kAddrWatchMax];
  size_t n = 0;
};

const AddrWatchList &addrWatchList() {
  static const AddrWatchList list = [] {
    AddrWatchList l;
    const char *e = kAddrWatch;
    if (!e || !*e)
      return l;
    for (const char *p = e; *p && l.n < kAddrWatchMax;) {
      char *end = nullptr;
      u64 base = std::strtoull(p, &end, 16);
      if (end == p)
        break;
      u64 win = kAddrWatchWindow;
      if (*end == ':')
        win = std::strtoull(end + 1, &end, 16);
      if (base <= 1)                     // "=1" -> built-in probe address
        base = kAddrWatchDefault;
      l.v[l.n++] = {base - std::min(base, win), base + win};
      while (*end && *end != ',')
        ++end;
      p = *end == ',' ? end + 1 : end;
    }
    return l;
  }();
  return list;
}

bool addrWatchEnabled() { return addrWatchList().n != 0; }

// Line budget per log kind, so a wide window can't fill the disk; raise with
// DELTA_UMTX_ADDRWATCH_MAX.
u32 addrWatchMax() {
  return kAddrWatchLimit;
}

bool addrWatched(const void *p) {
  if (!p)
    return false;
  const auto &l = addrWatchList();
  const u64 v = reinterpret_cast<u64>(p);
  for (size_t i = 0; i < l.n; ++i)
    if (v >= l.v[i].lo && v <= l.v[i].hi)
      return true;
  return false;
}

// "aa bb cc dd  ee ff .." of n bytes, read one at a time (racy is fine; we only
// care what the words look like).
void hexBytes(char *out, size_t outN, const void *p, size_t n) {
  const auto *b = static_cast<const volatile u8 *>(p);
  size_t w = 0;
  for (size_t i = 0; i < n && w + 4 < outN; ++i)
    w += static_cast<size_t>(
        std::snprintf(out + w, outN - w, "%02x%s", b[i],
                      (i % 8 == 7 && i + 1 < n) ? "  " : " "));
  out[w < outN ? w : outN - 1] = '\0';
}
}  // namespace

// DELTA_UMTX_CVTRACE=<hex ucond addr>: log every CV_WAIT/SIGNAL/BROADCAST on
// one condvar with tid + host thread name; naming both sides of the handshake
// identifies the producer that stopped running.
static void cvTrace(const char *what, const void *ucond, u32 self) {
  if (!kCvTrace || reinterpret_cast<u64>(ucond) != kCvTrace)
    return;
  char comm[32] = "";
  threadComm(comm, sizeof(comm));
  BASE_LOGI("cv", "{:<12} ucond={:p} tid={} ({})", what, ucond, self, comm);
  std::fflush(stderr);
}

// DELTA_UMTX_ADDRWATCH: one line per umtx op touching the watched window.
static void addrWatchLog(int op, const void *ptr, const void *a, u64 val,
                         u32 self) {
  if (!addrWatchEnabled())
    return;
  const bool hitPtr = addrWatched(ptr);
  const bool hitA = addrWatched(a);
  bool hitBatch = false;
  if (op == 21 && ptr) {  // NWAKE_PRIVATE: ptr is an array of val addresses
    auto *const *addrs = static_cast<void *const *>(ptr);
    for (u64 i = 0; i < val && i < 4096; ++i)
      if (addrWatched(addrs[i])) {
        hitBatch = true;
        break;
      }
  }
  if (!hitPtr && !hitA && !hitBatch)
    return;
  static std::atomic<u32> n{0};
  if (n.fetch_add(1) >= addrWatchMax())
    return;
  char comm[32] = "";
  threadComm(comm, sizeof(comm));
  BASE_LOGI("umtxw", "op={:<2} ptr={:p} a={:p} val={:#x} tid={}{}{}{} ({})",
            op, ptr, a, static_cast<unsigned long long>(val), self,
            hitPtr ? " HIT:ptr" : "", hitA ? " HIT:a" : "",
            hitBatch ? " HIT:nwake-batch" : "", comm);
  std::fflush(stderr);
}

// Raw guest bytes around a watched word (16 before + 32 at) to check struct
// ucond's layout against where we store our has-waiters flag.
static void addrWatchDump(const char *what, const void *p, u32 self) {
  if (!addrWatched(p))
    return;
  static std::atomic<u32> n{0};
  if (n.fetch_add(1) >= addrWatchMax())
    return;
  char pre[80], at[160];
  hexBytes(pre, sizeof(pre), static_cast<const u8 *>(p) - 16, 16);
  hexBytes(at, sizeof(at), p, 32);
  BASE_LOGI("umtxw", "{:<16} {:p} tid={}\n  [-16] {}\n  [ +0] {}", what, p,
            self, pre, at);
  std::fflush(stderr);
}

// DELTA_UMTX_HIST: sys_umtx_op is a dozen primitives behind one syscall number,
// so bucket every call by op and by (op, addr, tid) in a lock-free open-addressed
// table (hottest path in the process: no locks, no allocation) and dump the top
// offenders on a timer.
namespace umtxhist {
constexpr size_t kSlots = 1024;
struct Slot {
  std::atomic<u64> key{0};  // hash of (op, addr, tid), 0 = empty
  std::atomic<u64> n{0};
  std::atomic<u64> addr{0};
  std::atomic<u32> op{0};
  std::atomic<u32> tid{0};
};
Slot g_slots[kSlots];
std::atomic<u64> g_op[64];
std::atomic<u64> g_total{0};
std::atomic<u64> g_dropped{0};

bool enabled() {
  return kUmtxHist;
}

void startTimer();

inline void count(int op, const void *ptr, u32 tid) {
  startTimer();
  const u64 a = reinterpret_cast<u64>(ptr);
  g_total.fetch_add(1, std::memory_order_relaxed);
  g_op[op & 63].fetch_add(1, std::memory_order_relaxed);
  u64 key = (static_cast<u64>(op & 63) << 56) ^
                 (static_cast<u64>(tid) << 44) ^ a;
  if (!key)
    key = 1;
  size_t h = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 54) % kSlots;
  for (size_t i = 0; i < 32; ++i) {
    Slot &s = g_slots[(h + i) % kSlots];
    u64 k = s.key.load(std::memory_order_relaxed);
    if (k == key) {
      s.n.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    if (k == 0) {
      u64 expect = 0;
      if (s.key.compare_exchange_strong(expect, key,
                                        std::memory_order_relaxed)) {
        s.addr.store(a, std::memory_order_relaxed);
        s.op.store(static_cast<u32>(op), std::memory_order_relaxed);
        s.tid.store(tid, std::memory_order_relaxed);
        s.n.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      if (s.key.load(std::memory_order_relaxed) == key) {
        s.n.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
  }
  g_dropped.fetch_add(1, std::memory_order_relaxed);
}

const char *opName(u32 op) {
  switch (op) {
  case 0: return "LOCK";
  case 1: return "UNLOCK";
  case 2: return "WAIT";
  case 3: return "WAKE";
  case 4: return "MUTEX_TRYLOCK";
  case 5: return "MUTEX_LOCK";
  case 6: return "MUTEX_UNLOCK";
  case 7: return "SET_CEILING";
  case 8: return "CV_WAIT";
  case 9: return "CV_SIGNAL";
  case 10: return "CV_BROADCAST";
  case 11: return "WAIT_UINT";
  case 12: return "RW_RDLOCK";
  case 13: return "RW_WRLOCK";
  case 14: return "RW_UNLOCK";
  case 15: return "WAIT_UINT_PRIVATE";
  case 16: return "WAKE_PRIVATE";
  case 17: return "MUTEX_WAIT";
  case 18: return "MUTEX_WAKE";
  case 19: return "SEM_WAIT";
  case 20: return "SEM_WAKE";
  case 21: return "NWAKE_PRIVATE";
  case 22: return "CV_SIGNALTO";
  default: return "?";
  }
}

void dump() {
  BASE_LOGI("umtxhist", "total={} dropped={}",
            (unsigned long long)g_total.load(),
            (unsigned long long)g_dropped.load());
  for (u32 i = 0; i < 64; ++i)
    if (u64 n = g_op[i].load())
      BASE_LOGI("umtxhist", "  op {:<2} {:<18} {}", i, opName(i),
                (unsigned long long)n);
  struct Row { u64 n, addr; u32 op, tid; };
  std::vector<Row> rows;
  for (auto &s : g_slots)
    if (u64 n = s.n.load())
      rows.push_back({n, s.addr.load(), s.op.load(), s.tid.load()});
  std::sort(rows.begin(), rows.end(),
            [](const Row &a, const Row &b) { return a.n > b.n; });
  for (size_t i = 0; i < rows.size() && i < 28; ++i)
    BASE_LOGI("umtxhist", "  {:<18} {:#012x} gtid={:<3} {}",
              opName(rows[i].op), (unsigned long long)rows[i].addr,
              rows[i].tid, (unsigned long long)rows[i].n);
  std::fflush(stderr);
}

// Started from the first counted call, not static init: options are read long
// after a namespace-scope initializer would have asked.
void startTimer() {
  static const bool once = [] {
    std::thread([] {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(20));
        dump();
      }
    }).detach();
    return true;
  }();
  (void)once;
}
}  // namespace umtxhist

static void umtxTrace(int op, void *ptr, u32 self, u32 owner) {
  if (!kUmtxTrace)
    return;
  static std::atomic<int> n{0};
  if (n.fetch_add(1) < 4000)
    BASE_LOGI("umtx", "op={} ptr={:p} self={} owner={:#x}", op, ptr, self,
              owner);
}

// DELTA_UMTX_PROF=1: WALL time each thread spends inside sys_umtx_op. A CPU
// profile cannot see a thread blocked in futex_wait (it burns no cycles and
// vanishes from the samples), so this is the only way to tell "the guest's
// locks are the critical path" from "a worker is parked on a condvar".
namespace umtxwall {
struct Acc {
  u32 tid = 0;
  std::atomic<u64> ns{0};
  std::atomic<u64> calls{0};
};
std::mutex g_mtx;
std::vector<Acc *> g_accs;
DELTA_TLS_IE thread_local Acc *t_acc = nullptr;
// Which op is being hammered, and how long each spends. Same window.
std::atomic<u64> g_op_n[64];
std::atomic<u64> g_op_ns[64];
// How often the unlocked owner-word check answers the whole call.
std::atomic<u64> g_fast[64];

bool enabled() {
  static const bool on = [] {
    const char *e = std::getenv("DELTA_UMTX_PROF");
    return e && *e && *e != '0';
  }();
  return on;
}

Acc &acc() {
  if (!t_acc) {
    t_acc = new Acc{t_tid};
    std::lock_guard<std::mutex> lk(g_mtx);
    g_accs.push_back(t_acc);
  }
  return *t_acc;
}

// Report every ~2s from whichever thread notices; no hook in the frame loop.
void maybeReport() {
  static std::atomic<u64> last{0};
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  const u64 now_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  u64 prev = last.load(std::memory_order_relaxed);
  if (!prev) {
    last.compare_exchange_strong(prev, now_ns);
    return;
  }
  if (now_ns - prev < 2000000000ull)
    return;
  if (!last.compare_exchange_strong(prev, now_ns))
    return;
  const double window = double(now_ns - prev);
  std::lock_guard<std::mutex> lk(g_mtx);
  base::String line1;
  base::FormatTo(line1, "over {:.1f}s:", window / 1e9);
  for (Acc *a : g_accs) {
    const u64 ns = a->ns.exchange(0);
    const u64 n = a->calls.exchange(0);
    if (ns * 100.0 / window < 5.0)
      continue;  // only threads it actually holds up
    base::FormatTo(line1, " tid{}={:.0f}%(x{})", a->tid, ns * 100.0 / window,
                   (unsigned long long)n);
  }
  BASE_LOGI("umtxwall", "{}", line1.c_str());
  base::String line2;
  base::FormatTo(line2, "  ops:");
  for (u32 i = 0; i < 64; ++i) {
    const u64 n = g_op_n[i].exchange(0);
    const u64 ns = g_op_ns[i].exchange(0);
    if (n)
      base::FormatTo(line2, " {}={}({:.0f}ns,fast={:.0f}%)", umtxhist::opName(i),
                     (unsigned long long)n, n ? double(ns) / n : 0.0,
                     n ? g_fast[i].exchange(0) * 100.0 / n : 0.0);
  }
  BASE_LOGI("umtxwall", "{}", line2.c_str());
}
}  // namespace umtxwall

struct UmtxWallScope {
  std::chrono::steady_clock::time_point t0;
  int op = -1;
  bool on = umtxwall::enabled();
  explicit UmtxWallScope(int o) : op(o) {
    if (on)
      t0 = std::chrono::steady_clock::now();
  }
  ~UmtxWallScope() {
    if (!on)
      return;
    const u64 d = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
    auto &a = umtxwall::acc();
    a.ns.fetch_add(d, std::memory_order_relaxed);
    a.calls.fetch_add(1, std::memory_order_relaxed);
    umtxwall::g_op_n[op & 63].fetch_add(1, std::memory_order_relaxed);
    umtxwall::g_op_ns[op & 63].fetch_add(d, std::memory_order_relaxed);
    umtxwall::maybeReport();
  }
};

int PS4ABI sys_umtx_op(void *ptr, int op, u64 val, void *a, void *b) {
  UmtxWallScope _uw(op);
  WaitProbe _wp("umtx_op", (long)(long)ptr, (long)op);
  using namespace std::chrono_literals;
  markThreadStarted();  // first sync point => our init is done
  if (umtxhist::enabled())
    umtxhist::count(op, ptr, t_tid);
  addrWatchLog(op, ptr, a, val, t_tid);
  switch (op) {
  // WAIT registers under the same bucket lock WAKE takes, so a wake can't slip
  // between the value check and the sleep; a waiter leaves only on value
  // change, WAKE selecting it, or its own timeout.
  case 2:    // UMTX_OP_WAIT (compares a full u_long, unlike the UINT ops)
  case 11:   // UMTX_OP_WAIT_UINT
  case 15: { // UMTX_OP_WAIT_UINT_PRIVATE
    auto &bk = umtxBucket(ptr);
    auto changed = [&] {
      return op == 2 ? *static_cast<volatile u64 *>(ptr) != val
                     : *static_cast<volatile u32 *>(ptr) !=
                           static_cast<u32>(val);
    };
    const auto dl = umtxRelDeadline(b);
    std::unique_lock<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    SimpleWaiter waiter;
    ch.queueSimple(waiter);
    while (!changed() && !waiter.selected) {
      if (dl && std::chrono::steady_clock::now() >= *dl) {
        ch.removeSimple(waiter);
        return -SysError::eTIMEDOUT;
      }
      bk.cv.wait_for(lk, umtxTimeout());
    }
    ch.removeSimple(waiter);
    return 0;
  }
  case 17: { // UMTX_OP_MUTEX_WAIT: block while the umutex is owned
    auto *p = static_cast<std::atomic<u32> *>(ptr);
    if (kUmtxInjectNs) {
      const auto until = std::chrono::steady_clock::now() +
                         std::chrono::nanoseconds(kUmtxInjectNs);
      while (std::chrono::steady_clock::now() < until)
        __builtin_ia32_pause();
    }
    // The word is usually already free by the time we get here (libthr calls
    // this only after its userland CAS lost, and the winner often releases
    // first). The unlocked check needs no bucket lock and is no weaker; taking
    // them anyway cost ~1us across 2.9M calls/s in SotC, half the wall time of
    // every command-buffer-submitting thread.
    if ((p->load() & ~UMUTEX_CONTESTED) == 0) {
      umtxwall::g_fast[17].fetch_add(1, std::memory_order_relaxed);
      return 0;
    }
    auto &bk = umtxBucket(ptr);
    std::unique_lock<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    u32 owner = p->load();
    // FreeBSD _do_lock_normal (kern_umtx.c, _UMUTEX_WAIT mode): a word that is
    // free (UNOWNED, or CONTESTED with no owner tid) returns immediately.
    if ((owner & ~UMUTEX_CONTESTED) == 0)
      return 0;
    const u64 g0 = ch.gen;
    // THE contested handshake: FreeBSD publishes CONTESTED before sleeping
    // (casuword32 owner|CONTESTED) so a userland release knows to enter the
    // kernel; without it no MUTEX_WAKE ever fires and SotC's JobSystem
    // livelocked at 2ms per acquire. CAS, not a store: re-evaluate if the word
    // changed under us instead of stamping a stale owner back.
    if (!p->compare_exchange_strong(owner, owner | UMUTEX_CONTESTED))
      return 0;  // raced; libthr re-runs its own CAS loop
    ch.mutexWaiters++;
    // Leave only on "free" or an explicit MUTEX_WAKE, so a heavily contended
    // mutex doesn't degrade into a 2ms spin per waiter (a spurious exit is
    // harmless, libthr re-runs its CAS loop).
    StallReport stall{"MUTEX_WAIT", ptr};
    while ((p->load() & ~UMUTEX_CONTESTED) != 0 && ch.gen == g0) {
      bk.cv.wait_for(lk, umtxTimeout());
      stall.tick();
    }
    ch.mutexWaiters--;
    return 0;
  }
  case 3:    // UMTX_OP_WAKE
  case 16: { // UMTX_OP_WAKE_PRIVATE
    auto &bk = umtxBucket(ptr);
    bool woke = false;
    {
      std::lock_guard<std::mutex> lk(bk.m);
      woke = bk.chan[ptr].wakeSimple(val) != 0;
    }
    if (woke)
      bk.cv.notify_all();
    return 0;
  }
  // FreeBSD do_wake_umutex/do_wake2_umutex: refuse to release anyone while the
  // word still names an owner (waking would stampede every sleeper into a
  // failed re-CAS and back into MUTEX_WAIT); when the queue drains to <=1 the
  // kernel clears CONTESTED, since libthr's contested release relies on it.
  // PS5 numbers this wake 23 (its unlock is _thr_umutex_unlock2 verbatim);
  // answering EINVAL parked Astro Bot's main thread in scePthreadMutexLock.
  case 23:   // PS5 UMTX_OP_MUTEX_WAKE2
  case 18: { // UMTX_OP_MUTEX_WAKE
    auto *p = static_cast<std::atomic<u32> *>(ptr);
    // "Still held, so wake nobody" is decided by the owner word alone; the
    // release that just happened enters the kernel to do the waking (the word
    // stays CONTESTED while waiters are queued).
    if ((p->load() & ~UMUTEX_CONTESTED) != 0) {
      umtxwall::g_fast[18].fetch_add(1, std::memory_order_relaxed);
      return 0;
    }
    auto &bk = umtxBucket(ptr);
    std::unique_lock<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    u32 owner = p->load();
    if ((owner & ~UMUTEX_CONTESTED) != 0) {
      // Still held: leave the contested bit for the holder's unlock to clear.
      return 0;
    }
    if (ch.mutexWaiters <= 1) {
      u32 contested = UMUTEX_CONTESTED;
      if (p->compare_exchange_strong(contested, 0u))
        owner = 0u;
    }
    if (ch.mutexWaiters != 0)
      ch.gen++;  // release the queued MUTEX_WAIT sleepers to re-CAS
    lk.unlock();
    bk.cv.notify_all();
    return 0;
  }
  // Kernel-arbitrated mutex (PI/PROTECT): libthr hands the whole lock/unlock to
  // the kernel here, so enforce real mutual exclusion on the owner word under
  // the bucket lock. Returning 0 without arbitrating let two threads both "own"
  // one mutex (Fios2 worker pool: cond_wait owner check -> EPERM).
  case 4:    // UMTX_OP_MUTEX_TRYLOCK
  case 5: {  // UMTX_OP_MUTEX_LOCK
    auto &bk = umtxBucket(ptr);
    auto *p = static_cast<std::atomic<u32> *>(ptr);
    const u32 self = t_tid;
    std::unique_lock<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    bool queued = false;
    struct Dequeue {  // keep mutexWaiters exact on every exit path
      WaitChan *c;
      bool *q;
      ~Dequeue() {
        if (*q)
          c->mutexWaiters--;
      }
    } dequeue{&ch, &queued};
    StallReport stall{"MUTEX_LOCK", ptr};
    for (;;) {
      stall.tick();
      u32 owner = p->load();
      u32 held = owner & ~UMUTEX_CONTESTED;
      if (held == 0) {                 // free: claim it with a CAS. libthr's
        // userland fast path runs without our bucket lock, and two winners put
        // the same mutex on one thread's owned-PI list twice ("Fatal error
        // 'mutex is on list'"). Keep a waiter's CONTESTED bit, don't set it
        // spuriously (an uncontended unlock must stay in userland).
        if (!p->compare_exchange_strong(owner, self | (owner & UMUTEX_CONTESTED)))
          continue;                    // lost the race; re-evaluate
        umtxTrace(op, ptr, self, owner);
        return 0;
      }
      if (held == self)                // already ours (libthr counts recursion)
        return 0;
      if (op == 4)                     // trylock: don't block
        return -16 /*EBUSY*/;
      // CAS here too: a concurrent userland unlock may have freed the word.
      if (!p->compare_exchange_strong(owner, owner | UMUTEX_CONTESTED))
        continue;
      if (!queued) {  // now visible to op 6 / op 18's count-based decisions
        ch.mutexWaiters++;
        queued = true;
      }
      bk.cv.wait_for(lk, umtxTimeout());  // re-check on wake / safety timeout
    }
  }
  case 6: { // UMTX_OP_MUTEX_UNLOCK
    // libthr only reaches kernel unlock for a CONTESTED mutex and its userland
    // path has already verified ownership (and cleared the tid for the
    // contested PI/PROTECT release). Re-checking would EPERM and make libthr
    // skip dequeueing ("Fatal error 'mutex is on list'"). Just release and wake.
    auto &bk = umtxBucket(ptr);
    auto *p = static_cast<std::atomic<u32> *>(ptr);
    std::unique_lock<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    u32 owner = p->load();
    umtxTrace(6, ptr, t_tid, owner);
    // FreeBSD do_unlock_normal: UNOWNED at <=1 waiter, CONTESTED otherwise. A
    // blind 0 with 2+ waiters queued hands the next acquirer an UNCONTESTED
    // word, so its release stays in userland and never wakes the rest: the
    // contested chain breaks at every handoff.
    p->compare_exchange_strong(owner, ch.mutexWaiters <= 1
                                          ? 0u
                                          : UMUTEX_CONTESTED);
    ch.gen++;  // releases MUTEX_WAIT sleepers
    lk.unlock();
    bk.cv.notify_all();
    return 0;
  }
  // CV_WAIT atomically releases the umutex (uaddr1=a) and sleeps on the ucond
  // (ptr), holding the cond bucket across the release so a signal can't be
  // lost. c_has_waiters must rise BEFORE the mutex drops (pthread_cond_signal
  // only issues the syscall when it sees the flag), and a waiter returns only
  // on a real signal/broadcast or its own timeout, never the safety re-poll
  // (libthr reports 0/EINTR as "signaled"; engines deref unpublished state).
  case 8: { // UMTX_OP_CV_WAIT: ptr=ucond, a=umutex, b=timespec, val=flags
    cvTrace("CV_WAIT", ptr, t_tid);
    auto &cbk = umtxBucket(ptr);
    auto &mbk = umtxBucket(a);
    const auto dl = cvDeadline(ptr, val, b);
    std::unique_lock<std::mutex> clk(cbk.m);
    auto &ch = cbk.chan[ptr];  // stable ref: unordered_map never moves nodes
    ch.waiters++;
    const u64 myTicket = ch.nextTicket++;
    addrWatchDump("cv-wait pre", ptr, t_tid);
    static_cast<std::atomic<u32> *>(ptr)->store(1);  // c_has_waiters
    addrWatchDump("cv-wait post", ptr, t_tid);
    if (a)
      addrWatchDump("cv-wait mutex", a, t_tid);
    // Snapshot ucond + umutex so the poll loop can report ANY guest write to
    // them: a plain-store wake and libthr's userland CAS lock/unlock leave no
    // syscall trace, so an unchanging word is the evidence the producer side
    // never even runs.
    const bool watch = addrWatched(ptr);
    const bool watchM = a && addrWatched(a);
    u8 snap[32], snapM[32];
    if (watch)
      __builtin_memcpy(snap, ptr, sizeof(snap));
    if (watchM)
      __builtin_memcpy(snapM, a, sizeof(snapM));
    // Snapshot the wake generation BEFORE releasing the mutex, so the cond
    // bucket can be dropped during the release (never hold two bucket locks;
    // the old order deadlocked on opposite-bucket CV_WAITs, wedging the whole
    // process ~9s in).
    const u64 g0 = ch.gen;
    if (a) {                           // release the mutex (waking its waiters)
      umtxTrace(8, a, t_tid,
                static_cast<std::atomic<u32> *>(a)->load());
      auto *m = static_cast<std::atomic<u32> *>(a);
      // Same release rule as op 6 (do_cv_wait calls do_unlock_umutex, not a raw
      // store): leave CONTESTED while 2+ waiters are queued, or the next
      // acquirer's release stays in userland and strands them.
      auto releaseMutex = [&](WaitChan &mch) {
        u32 owner = m->load();
        m->compare_exchange_strong(owner,
                                   mch.mutexWaiters <= 1 ? 0u
                                                         : UMUTEX_CONTESTED);
        mch.gen++;
      };
      if (&mbk == &cbk) {              // same bucket: already locked
        releaseMutex(mbk.chan[a]);
        mbk.cv.notify_all();
      } else {
        clk.unlock();
        {
          std::lock_guard<std::mutex> mlk(mbk.m);
          releaseMutex(mbk.chan[a]);
        }
        mbk.cv.notify_all();
        clk.lock();
      }
    }
    int r = 0;
    StallReport stall{"CV_WAIT", ptr, a};
    for (;;) {
      stall.tick();
      if (ch.gen != g0)                // broadcast: releases every sleeper
        break;
      if (ch.signals > 0 && myTicket < ch.signalCutoff) {
        ch.signals--;                  // exactly one queued sleeper consumes it
        break;
      }
      if (dl && std::chrono::steady_clock::now() >= *dl) {
        r = -SysError::eTIMEDOUT;
        break;
      }
      cbk.cv.wait_for(clk, umtxTimeout());
      if (watch && __builtin_memcmp(snap, ptr, sizeof(snap)) != 0) {
        __builtin_memcpy(snap, ptr, sizeof(snap));
        addrWatchDump("ucond changed", ptr, t_tid);
      }
      if (watchM && __builtin_memcmp(snapM, a, sizeof(snapM)) != 0) {
        __builtin_memcpy(snapM, a, sizeof(snapM));
        addrWatchDump("mutex changed", a, t_tid);
      }
    }
    addrWatchDump("cv-wait exit", ptr, t_tid);
    if (--ch.waiters == 0) {           // last one out lowers c_has_waiters
      ch.signals = 0;                  // unconsumed signals don't outlive waiters
      static_cast<std::atomic<u32> *>(ptr)->store(0);
    }
    return r;
  }
  case 9: {  // UMTX_OP_CV_SIGNAL: release one waiter
    cvTrace("CV_SIGNAL", ptr, t_tid);
    auto &bk = umtxBucket(ptr);
    std::lock_guard<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    if (ch.waiters > ch.signals) {     // signal with nobody waiting is lost
      ch.signals++;
      ch.signalCutoff = ch.nextTicket;  // only sleepers queued by now may take it
    }
    bk.cv.notify_all();
    return 0;
  }
  case 10: { // UMTX_OP_CV_BROADCAST: release all waiters
    cvTrace("CV_BROADCAST", ptr, t_tid);
    auto &bk = umtxBucket(ptr);
    std::lock_guard<std::mutex> lk(bk.m);
    bk.chan[ptr].gen++;
    bk.cv.notify_all();
    return 0;
  }
  // Userland semaphore (struct _usem at ptr): SEM_WAIT publishes _has_waiters
  // and blocks on _count; returning 0 immediately turned sem_wait into a hot
  // spin (a SotTR savedata worker pegged a core re-issuing the syscall).
  // Reader/writer lock (struct urwlock at ptr): libthr enters the kernel only
  // to block, so every state change here is a CAS too; a blind store stomps a
  // concurrent userland acquire, and the old default arm granted the lock to
  // every caller at once.
  case 12:   // UMTX_OP_RW_RDLOCK
  case 13: { // UMTX_OP_RW_WRLOCK
    constexpr u32 kWriteOwner = 0x80000000u;
    constexpr u32 kWriteWaiters = 0x40000000u;
    constexpr u32 kReadWaiters = 0x20000000u;
    constexpr u32 kMaxReaders = 0x1fffffffu;
    constexpr u32 kPreferReader = 0x0002u;
    const bool wr = op == 13;
    auto &bk = umtxBucket(ptr);
    auto *p = static_cast<std::atomic<u32> *>(ptr);
    auto *blocked = reinterpret_cast<volatile u32 *>(
        static_cast<u8 *>(ptr) + (wr ? 12 : 8));
    const u32 flags = reinterpret_cast<volatile u32 *>(
        static_cast<u8 *>(ptr))[1];
    std::unique_lock<std::mutex> lk(bk.m);
    for (;;) {
      u32 st = p->load();
      if (wr) {
        // A writer needs the lock completely idle.
        if (!(st & kWriteOwner) && (st & kMaxReaders) == 0) {
          if (p->compare_exchange_strong(st, st | kWriteOwner))
            return 0;
          continue;
        }
      } else {
        // A reader yields to a queued writer unless the lock prefers readers or
        // a reader is already in (queueing behind it would deadlock).
        const bool yield_to_writer = (st & kWriteWaiters) &&
                                     !(flags & kPreferReader) &&
                                     (st & kMaxReaders) == 0;
        if (!(st & kWriteOwner) && !yield_to_writer) {
          if ((st & kMaxReaders) == kMaxReaders)
            return -SysError::eAGAIN;
          if (p->compare_exchange_strong(st, st + 1))
            return 0;
          continue;
        }
      }
      // Publish the waiter bit so a userland unlock knows to enter the kernel.
      const u32 want = st | (wr ? kWriteWaiters : kReadWaiters);
      if (st != want && !p->compare_exchange_strong(st, want))
        continue;
      (*blocked)++;
      bk.cv.wait_for(lk, umtxTimeout());  // re-check on wake / safety timeout
      (*blocked)--;
    }
  }
  case 14: { // UMTX_OP_RW_UNLOCK
    constexpr u32 kWriteOwner = 0x80000000u;
    constexpr u32 kWriteWaiters = 0x40000000u;
    constexpr u32 kReadWaiters = 0x20000000u;
    constexpr u32 kMaxReaders = 0x1fffffffu;
    auto &bk = umtxBucket(ptr);
    auto *p = static_cast<std::atomic<u32> *>(ptr);
    auto *blocked = reinterpret_cast<volatile u32 *>(
        static_cast<u8 *>(ptr));
    std::unique_lock<std::mutex> lk(bk.m);
    u32 st = p->load();
    for (;;) {
      u32 next;
      if (st & kWriteOwner)
        next = st & ~kWriteOwner;
      else if ((st & kMaxReaders) != 0)
        next = st - 1;
      else
        return -SysError::ePERM;
      // Retire a waiter bit nobody is behind any more; a stale WRITE_WAITERS
      // starves readers forever (they yield to a writer that no longer exists).
      if (blocked[3] == 0)
        next &= ~kWriteWaiters;
      if (blocked[2] == 0)
        next &= ~kReadWaiters;
      if (p->compare_exchange_strong(st, next))
        break;
    }
    lk.unlock();
    bk.cv.notify_all();
    return 0;
  }
  case 19: { // UMTX_OP_SEM_WAIT
    auto &bk = umtxBucket(ptr);
    auto *hasWaiters = static_cast<std::atomic<u32> *>(ptr);
    auto *count = reinterpret_cast<volatile u32 *>(
        static_cast<u8 *>(ptr) + 4);
    const auto dl = umtxRelDeadline(b);
    std::unique_lock<std::mutex> lk(bk.m);
    u32 z = 0;
    hasWaiters->compare_exchange_strong(z, 1);  // publish "has waiters"
    auto &ch = bk.chan[ptr];
    const u64 g0 = ch.gen;
    while (*count == 0 && ch.gen == g0) {
      if (dl && std::chrono::steady_clock::now() >= *dl)
        return -SysError::eTIMEDOUT;
      bk.cv.wait_for(lk, umtxTimeout());
    }
    return 0;
  }
  case 20: { // UMTX_OP_SEM_WAKE
    auto &bk = umtxBucket(ptr);
    std::lock_guard<std::mutex> lk(bk.m);
    bk.chan[ptr].gen++;
    bk.cv.notify_all();
    return 0;
  }
  case 21: { // UMTX_OP_NWAKE_PRIVATE: wake all waiters on each listed address
    // FreeBSD's libthr batches deferred CV wakes into an array and releases
    // every address when the mutex unlocks (kern_umtx.c ~L2995).
    auto **addrs = static_cast<void **>(ptr);
    for (u64 i = 0; i < val; ++i) {
      auto &bk = umtxBucket(addrs[i]);
      bool woke = false;
      {
        std::lock_guard<std::mutex> lk(bk.m);
        woke = bk.chan[addrs[i]].wakeSimple(0x7fffffff) != 0;
      }
      if (woke)
        bk.cv.notify_all();
    }
    return 0;
  }
  case 22: { // UMTX_OP_CV_SIGNALTO: signal one specific waiter (val = its tid).
    // We don't track per-waiter tids, so release one waiter; same observable
    // effect, and a stray wake is harmless (libthr re-checks the predicate).
    cvTrace("CV_SIGNALTO", ptr, t_tid);
    auto &bk = umtxBucket(ptr);
    std::lock_guard<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    if (ch.waiters > ch.signals) {
      ch.signals++;
      ch.signalCutoff = ch.nextTicket;
    }
    bk.cv.notify_all();
    return 0;
  }
  default: {
    // FreeBSD 9 (PS4) stops at 22; PS5's table is one longer (op 23), with
    // different numbering past 17.
    if (op < 0 || op > 22)
      return -SysError::eINVAL;
    static std::atomic<u32> seen[32]{};
    if (op >= 0 && op < 32 && seen[op].fetch_add(1) == 0)
      BASE_LOGI("umtx", "unhandled op={}", op);
    return 0;
  }
  }
}
} // namespace krnl
