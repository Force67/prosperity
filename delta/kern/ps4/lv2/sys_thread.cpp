
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>

#include "wait_probe.h"
#include "../../thread_names.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <pthread.h>
#include <unordered_map>

#include <utl/mem.h>

#include "../../module.h"
#include "../../proc.h"
#include "cpu/cpu_backend.h"
#include "sys_thread.h"

namespace krnl {
moduleInfo *called_in(void *addr);

// These per-thread vars use initial-exec TLS for fast, allocation-free access.
// The Android app is a dlopen'd .so, where IE TLS is rejected (STATIC_TLS), so
// there we fall back to the toolchain default (-ftls-model=global-dynamic).
#if defined(__ANDROID__) && defined(DELTA_ANDROID_APP)
#define DELTA_TLS_IE
#else
#define DELTA_TLS_IE __attribute__((tls_model("initial-exec")))
#endif

// Per-thread guest thread id (sys_thr_self). Main thread is 1.
static DELTA_TLS_IE thread_local uint32_t t_tid = 1;
static std::atomic<uint32_t> g_nextTid{2};

// Address of the calling thread's guest tid TLS. The FEX watchdog captures this
// per live thread so it can map a umutex owner word (a guest tid) to the actual
// thread that holds the lock and dump WHAT that owner is blocked on.
const uint32_t *currentGuestTidPtr() { return &t_tid; }

// Thread-startup handshake: sys_thr_new blocks until the new thread has run its
// init and reached its first sync point (umtx). The game spawns workers that
// produce shared state the main thread then reads with no explicit ordering
// (it relies on the worker, on another core, having finished). Without the
// head start the main thread races ahead and reads not-yet-produced data.
static std::mutex g_startM;
static std::condition_variable g_startCv;
static DELTA_TLS_IE thread_local std::atomic<bool> *t_started = nullptr;

static void markThreadStarted() {
  if (t_started && !t_started->exchange(true)) {
    std::lock_guard<std::mutex> lk(g_startM);
    g_startCv.notify_all();
  }
}

// FreeBSD thr_param (the layout sys_thr_new receives in rdi).
struct thr_param {
  void(PS4ABI *start_func)(void *);
  void *arg;
  uint8_t *stack_base;
  size_t stack_size;
  uint8_t *tls_base;
  size_t tls_size;
  int64_t *child_tid;
  int64_t *parent_tid;
  int32_t flags;
  void *rtp;
  void *spare[3];
};

void ps5MaybeInterposePthreadAlloc();

int PS4ABI sys_thr_new(thr_param *p, int size) {
  // The kernel rejects an oversized param block (copyin guard); param_size is
  // exactly sizeof(thr_param) == 0x68 for every libthr we see.
  if (!p || size < 0 || static_cast<size_t>(size) > sizeof(thr_param))
    return -22 /*EINVAL*/;
  // A new thread means malloc is about to go multithreaded; make sure libkernel's
  // pthread-state allocator is interposed so the libc-mutex bootstrap can't recurse.
  ps5MaybeInterposePthreadAlloc();
  uint32_t tid = g_nextTid.fetch_add(1);
  std::printf("[thr_new] tid=%u start=%p arg=%p stack=%p+%#zx tls=%p\n", tid,
              (void *)p->start_func, p->arg, (void *)p->stack_base,
              p->stack_size, (void *)p->tls_base);

  if (p->child_tid)
    *p->child_tid = tid;
  if (p->parent_tid)
    *p->parent_tid = tid;

  // The guest's libthr switches RSP onto this caller-provided stack the instant
  // the thread starts (its trampoline does a `mov rsp, <stack_top>`). If any page
  // of [stack_base, stack_base+stack_size) is not actually backed -- e.g. the
  // guest's own stack mmap landed part of the region at a different address than
  // the base it hands us -- that very first push faults. This surfaced as P.T.'s
  // "GameSave" worker crashing on entry at stack_top-0x18 (the top page of its
  // 64 KiB stack was unmapped). Guarantee the whole range is committed + writable
  // before the thread runs. We only fill pages that are currently FREE (reserve
  // == MAP_FIXED_NOREPLACE returns non-null only then), so an intentional
  // PROT_NONE guard page or already-valid stack memory is never clobbered.
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
        // Only register pages the VMA doesn't know: the stack region usually
        // already has a guest mmap entry, and re-adding pages would punch it
        // into fragments (add() keeps intervals disjoint by splitting).
        if (!proc->getVma().get(static_cast<uint8_t *>(c)))
          proc->getVma().add(static_cast<uint8_t *>(c), kPage,
                             utl::pageProtection::w);
        filled++;
      }
    }
    if (filled)
      std::printf(
          "[thr_new] tid=%u backed %d unmapped stack page(s) for %p+%#zx\n", tid,
          filled, (void *)p->stack_base, p->stack_size);
  }

  auto fn = p->start_func;
  auto arg = p->arg;
  auto fsbase = reinterpret_cast<uint64_t>(p->tls_base);
  auto started = std::make_shared<std::atomic<bool>>(false);

  // Run on the host thread's (large) stack, NOT the guest's thr_param stack: our
  // host syscall handlers execute on whatever stack the guest code is using, and
  // the guest stack (e.g. 64 KiB) is far too small for them (std::thread/printf/
  // C++ exceptions overflow it). The host thread stack is bigger and works.
  // Create the guest thread on THIS (parent) thread, as FEX requires. Creating
  // it on the freshly spawned worker while other guest threads run in the JIT
  // races on shared context state. The worker host thread only runs it.
  void *gthread =
      cpu::backend().createGuestThread(reinterpret_cast<uintptr_t>(fn), arg, fsbase);

  // DIAGNOSTIC (env-gated, default off): the FEX guest-worker
  // HOST pthread stack is the glibc default (8 MiB). Deep BPE streaming jobs
  // (FIOS2 decompress -> libkernel HLE -> FEX dispatcher/thunk round-trips) blow
  // it ~9s into LoadInitialWorld, faulting at the host-stack guard 0x...feff0
  // (same root cause fex_backend.cpp ties to the SotC AllocationTracker crash).
  // DELTA_HOST_STACK_MB=<N> spawns guest workers with an N-MiB native stack to
  // test/mitigate the overflow; UNSET keeps the original std::thread behaviour.
  // The worker registers the guest's thr_param stack range so the sys_mname
  // tag titles put on it becomes the host thread's name (thread_names.cpp).
  auto *gsb = p->stack_base;
  const size_t gss = p->stack_size;
  const char *hsEnv = std::getenv("DELTA_HOST_STACK_MB");
  if (hsEnv) {
    auto *ctx = new std::tuple<void *, uint32_t, std::shared_ptr<std::atomic<bool>>,
                               void *, size_t>(gthread, tid, started, gsb, gss);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    size_t hostStack = (size_t)std::strtoull(hsEnv, nullptr, 0) * 1024 * 1024;
    if (hostStack < 8ull * 1024 * 1024) hostStack = 256ull * 1024 * 1024;
    pthread_attr_setstacksize(&attr, hostStack);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    auto trampoline = +[](void *pv) -> void * {
      auto *c = static_cast<std::tuple<void *, uint32_t,
                                       std::shared_ptr<std::atomic<bool>>,
                                       void *, size_t> *>(pv);
      t_tid = std::get<1>(*c);
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
      t_started = started.get();
      registerGuestThreadStack(gsb, gss);
      cpu::backend().runGuestThread(gthread);
      unregisterGuestThreadStack();
    }).detach();
  }

  // Wait for the new thread to finish its init and hit its first sync point so
  // it wins the races the game expects it to. Bounded so a thread that never
  // syncs can't hang us. DELTA_NO_THR_BARRIER disables the wait so the spawner
  // runs ahead, for cases where the spawner is the producer the spawned thread
  // depends on.
  static const bool noBarrier = std::getenv("DELTA_NO_THR_BARRIER") != nullptr;
  if (!noBarrier) {
    std::unique_lock<std::mutex> lk(g_startM);
    g_startCv.wait_for(lk, std::chrono::milliseconds(200),
                       [&] { return started->load(); });
  }
  return 0;
}

int PS4ABI sys_thr_self(int64_t *tid) {
  if (!tid)
    return -14 /*EFAULT*/;
  // The kernel stores td_tid through suword64 (a full 64-bit, zero-extended
  // store). Writing only 32 bits left the caller's high word as stack garbage.
  *tid = t_tid;
  return 0;
}

int PS4ABI sys_rtprio_thread(int function, uint64_t lwpid, thread_prio *rtp) {
  if (!rtp)
    return -14 /*EFAULT*/;
  // function: RTP_LOOKUP(0) reports the priority, RTP_SET(1) applies the
  // caller's. We don't model real-time scheduling, so accept a SET unchanged and
  // report a normal class on LOOKUP.
  constexpr int RTP_SET = 1;
  if (function == RTP_SET)
    return 0;
  rtp->type = 3; /*RTP_PRIO_NORMAL: time-sharing*/
  rtp->prio = 1; /*almost highest prio*/
  return 0;
}

// Address-keyed wait/wake (a small futex). A fixed bucket array avoids per-
// address allocation; hash collisions only cause harmless spurious wakeups of
// the HOST condvar -- never spurious returns to the guest, because every wait
// below re-checks its own exit condition in a loop before returning.
namespace {
// Per-address wake state. Simple WAIT uses an explicit queue because WAKE's val
// limits how many already-registered waiters it releases. The other object
// classes retain generation/tokens appropriate to their own operations.
// unordered_map guarantees reference stability across inserts, so a sleeping
// waiter may hold its WaitChan& while others register.
struct SimpleWaiter {
  SimpleWaiter *prev = nullptr;
  SimpleWaiter *next = nullptr;
  bool queued = false;
  bool selected = false;
};
struct WaitChan {
  uint64_t gen = 0;      // broadcast generation for mutex/CV/semaphore waits
  uint64_t signals = 0;  // pending single-waiter releases (CV_SIGNAL)
  uint32_t waiters = 0;  // live CV_WAIT sleepers (drives ucond c_has_waiters)
  // Live umutex sleepers (op 5 MUTEX_LOCK + op 17 MUTEX_WAIT). FreeBSD's
  // umtxq_count() on the mutex's own queue decides whether a release leaves the
  // word UNOWNED or CONTESTED, so the count has to be tracked, not guessed.
  uint32_t mutexWaiters = 0;
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

  uint32_t wakeSimple(uint64_t requested) {
    // FreeBSD narrows val to int and its queue loop wakes one even for <= 0.
    const int32_t count = static_cast<int32_t>(requested);
    uint32_t remaining = count > 1 ? static_cast<uint32_t>(count) : 1;
    uint32_t woken = 0;
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
constexpr uint32_t UMUTEX_CONTESTED = 0x80000000u;
// Re-poll interval for a blocked waiter. A waiter is woken promptly by the
// matching WAKE/SIGNAL, but some guest code publishes its predicate with a plain
// lock-free store and NO wake syscall (it expects the waiter to re-check), so a
// long timeout left such a waiter asleep for the whole interval -> Doom64's KEX
// job scheduler stalled ~1s per frame (~1fps). Re-checking every few ms turns
// that into full speed (Doom64 1fps -> 60fps). The tick is ONLY a re-poll: no
// wait below returns to the guest because of it (a poll tick with the exit
// condition still false goes back to sleep). Returning "woken" with the
// predicate unpublished let engines deref half-built state -- SotC's fiber
// job/stream managers crashed intermittently on null / -1 / partially-written
// pointers exactly that way. DELTA_UMTX_TIMEOUT_MS overrides the tick.
std::chrono::milliseconds umtxTimeout() {
  static const long ms = [] {
    const char *e = std::getenv("DELTA_UMTX_TIMEOUT_MS");
    return e ? std::atol(e) : 2;
  }();
  return std::chrono::milliseconds(ms);
}

// The WAIT-class ops take an optional timeout at uaddr2 (FreeBSD 9 passes a
// struct timespec there; NULL = wait forever). Relative time.
struct GuestTimespec {
  int64_t sec;
  int64_t nsec;
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
constexpr uint64_t kCvWaitAbsTime = 0x02;
std::optional<SteadyTp> cvDeadline(const void *ucond, uint64_t flags,
                                   const void *b) {
  if (!b)
    return std::nullopt;
  if (!(flags & kCvWaitAbsTime))
    return umtxRelDeadline(b);
  auto *ts = static_cast<const GuestTimespec *>(b);
  const uint32_t clockid = *reinterpret_cast<const uint32_t *>(
      static_cast<const uint8_t *>(ucond) + 8);  // ucond.c_clockid
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

// DELTA_UMTX_ADDRWATCH=<hex addr>[:<hex window>][,<hex addr>[:<hex window>]]...
// (or =1 for the built-in probe address): diagnostic-only address watch.
// DELTA_UMTX_CVTRACE only sees ops 8/9/10 on one exact pointer, so a wake aimed
// at the same object through a DIFFERENT op (WAKE, MUTEX_WAKE, NWAKE_PRIVATE)
// or at a neighbouring field of the same struct is invisible to it. This
// watches address windows instead, so "does the guest try to wake this thing at
// ALL?" can be answered. Off unless the env var is set.
constexpr uint64_t kAddrWatchWindow = 0x200;   // default +/- around each addr
constexpr uint64_t kAddrWatchDefault = 0x2000040a4ull;
constexpr size_t kAddrWatchMax = 8;

struct AddrWatchSpec {
  uint64_t lo, hi;
};
struct AddrWatchList {
  AddrWatchSpec v[kAddrWatchMax];
  size_t n = 0;
};

const AddrWatchList &addrWatchList() {
  static const AddrWatchList list = [] {
    AddrWatchList l;
    const char *e = std::getenv("DELTA_UMTX_ADDRWATCH");
    if (!e || !*e)
      return l;
    for (const char *p = e; *p && l.n < kAddrWatchMax;) {
      char *end = nullptr;
      uint64_t base = std::strtoull(p, &end, 16);
      if (end == p)
        break;
      uint64_t win = kAddrWatchWindow;
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

// Line budget per log kind, so a wide window can't fill the disk. Raise with
// DELTA_UMTX_ADDRWATCH_MAX when a hot address is inside the window -- a budget
// that runs out mid-run makes "no wake ever arrived" unprovable.
uint32_t addrWatchMax() {
  static const uint32_t n = [] {
    const char *e = std::getenv("DELTA_UMTX_ADDRWATCH_MAX");
    return e ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 200000u;
  }();
  return n;
}

bool addrWatched(const void *p) {
  if (!p)
    return false;
  const auto &l = addrWatchList();
  const uint64_t v = reinterpret_cast<uint64_t>(p);
  for (size_t i = 0; i < l.n; ++i)
    if (v >= l.v[i].lo && v <= l.v[i].hi)
      return true;
  return false;
}

// "aa bb cc dd  ee ff .." of n bytes read one at a time (the guest may be
// racing us; we only care what the words look like, not about atomicity).
void hexBytes(char *out, size_t outN, const void *p, size_t n) {
  const auto *b = static_cast<const volatile uint8_t *>(p);
  size_t w = 0;
  for (size_t i = 0; i < n && w + 4 < outN; ++i)
    w += static_cast<size_t>(
        std::snprintf(out + w, outN - w, "%02x%s", b[i],
                      (i % 8 == 7 && i + 1 < n) ? "  " : " "));
  out[w < outN ? w : outN - 1] = '\0';
}
}  // namespace

// DELTA_UMTX_CVTRACE=<hex ucond addr>: log every CV_WAIT / CV_SIGNAL /
// CV_BROADCAST on one condvar, with the calling tid and the host thread name
// (thread_names.cpp). A title that stalls waiting on a condvar nobody signals
// looks identical to an idle one; naming both sides of the handshake -- and
// showing that only the wait side ever appears -- is what identifies the
// producer that stopped running.
static void cvTrace(const char *what, const void *ucond, uint32_t self) {
  static const uint64_t want = [] {
    const char *e = std::getenv("DELTA_UMTX_CVTRACE");
    return e ? std::strtoull(e, nullptr, 16) : 0ull;
  }();
  if (!want || reinterpret_cast<uint64_t>(ucond) != want)
    return;
  char comm[32] = "";
  threadComm(comm, sizeof(comm));
  std::fprintf(stderr, "[cv] %-12s ucond=%p tid=%u (%s)\n", what, ucond, self,
               comm);
  std::fflush(stderr);
}

// DELTA_UMTX_ADDRWATCH: one line per umtx op touching the watched window.
static void addrWatchLog(int op, const void *ptr, const void *a, uint64_t val,
                         uint32_t self) {
  if (!addrWatchEnabled())
    return;
  const bool hitPtr = addrWatched(ptr);
  const bool hitA = addrWatched(a);
  bool hitBatch = false;
  if (op == 21 && ptr) {  // NWAKE_PRIVATE: ptr is an array of val addresses
    auto *const *addrs = static_cast<void *const *>(ptr);
    for (uint64_t i = 0; i < val && i < 4096; ++i)
      if (addrWatched(addrs[i])) {
        hitBatch = true;
        break;
      }
  }
  if (!hitPtr && !hitA && !hitBatch)
    return;
  static std::atomic<uint32_t> n{0};
  if (n.fetch_add(1) >= addrWatchMax())
    return;
  char comm[32] = "";
  threadComm(comm, sizeof(comm));
  std::fprintf(stderr,
               "[umtxw] op=%-2d ptr=%p a=%p val=%#llx tid=%u%s%s%s (%s)\n", op,
               ptr, a, static_cast<unsigned long long>(val), self,
               hitPtr ? " HIT:ptr" : "", hitA ? " HIT:a" : "",
               hitBatch ? " HIT:nwake-batch" : "", comm);
  std::fflush(stderr);
}

// Raw guest bytes around a watched word: 16 before + 32 at, so struct ucond's
// real layout (c_has_waiters / c_flags / c_clockid) can be checked against
// where we store our has-waiters flag.
static void addrWatchDump(const char *what, const void *p, uint32_t self) {
  if (!addrWatched(p))
    return;
  static std::atomic<uint32_t> n{0};
  if (n.fetch_add(1) >= addrWatchMax())
    return;
  char pre[80], at[160];
  hexBytes(pre, sizeof(pre), static_cast<const uint8_t *>(p) - 16, 16);
  hexBytes(at, sizeof(at), p, 32);
  std::fprintf(stderr, "[umtxw] %-16s %p tid=%u\n  [-16] %s\n  [ +0] %s\n",
               what, p, self, pre, at);
  std::fflush(stderr);
}

static void umtxTrace(int op, void *ptr, uint32_t self, uint32_t owner) {
  static const bool tr = std::getenv("DELTA_UMTX_TRACE") != nullptr;
  if (!tr)
    return;
  static std::atomic<int> n{0};
  if (n.fetch_add(1) < 4000)
    std::fprintf(stderr, "[umtx] op=%d ptr=%p self=%u owner=%#x\n", op, ptr,
                 self, owner);
}

int PS4ABI sys_umtx_op(void *ptr, int op, uint64_t val, void *a, void *b) {
  WaitProbe _wp("umtx_op", (long)(long)ptr, (long)op);
  using namespace std::chrono_literals;
  markThreadStarted();  // first sync point => our init is done
  addrWatchLog(op, ptr, a, val, t_tid);
  switch (op) {
  // WAIT registers before checking the value while holding the same bucket lock
  // as WAKE, so a wake can't slip between the check and sleep. A waiter leaves
  // only when the value changed, WAKE selected it, or its own timeout expired.
  case 2:    // UMTX_OP_WAIT (compares a full u_long, unlike the UINT ops)
  case 11:   // UMTX_OP_WAIT_UINT
  case 15: { // UMTX_OP_WAIT_UINT_PRIVATE
    auto &bk = umtxBucket(ptr);
    auto changed = [&] {
      return op == 2 ? *static_cast<volatile uint64_t *>(ptr) != val
                     : *static_cast<volatile uint32_t *>(ptr) !=
                           static_cast<uint32_t>(val);
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
    auto &bk = umtxBucket(ptr);
    auto *p = static_cast<std::atomic<uint32_t> *>(ptr);
    std::unique_lock<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    uint32_t owner = p->load();
    // FreeBSD _do_lock_normal (kern_umtx.c, _UMUTEX_WAIT mode): a word that is
    // free -- UNOWNED, or CONTESTED with no owner tid -- returns immediately.
    if ((owner & ~UMUTEX_CONTESTED) == 0)
      return 0;
    const uint64_t g0 = ch.gen;
    // THE contested handshake. FreeBSD publishes the bit before sleeping:
    //   old = casuword32(&m->m_owner, owner, owner | UMUTEX_CONTESTED);
    // "Set the contested bit so that a release in user space knows to use the
    // system call for unlock." Without it libthr's inline release
    // (atomic_cmpset_rel_32(m_owner, id, UMUTEX_UNOWNED)) always succeeds, the
    // owner never enters the kernel, no MUTEX_WAKE is ever issued, and every
    // waiter here is released only by the safety re-poll below -- which turned
    // SotC's JobSystem handoffs into a 2ms-per-acquire livelock (15708 failed
    // re-acquires on one mutex) that stalled the whole FIOS streaming pipeline.
    // CAS, not a store: if the word changed under us (a concurrent userland
    // unlock freed it) re-evaluate instead of stamping a stale owner back.
    if (!p->compare_exchange_strong(owner, owner | UMUTEX_CONTESTED))
      return 0;  // raced; libthr re-runs its own CAS loop
    ch.mutexWaiters++;
    // A spurious exit here is harmless (libthr re-runs its CAS loop), but
    // still leave only on "free" or an explicit MUTEX_WAKE so a heavily
    // contended mutex doesn't degrade into a 2ms spin per waiter.
    while ((p->load() & ~UMUTEX_CONTESTED) != 0 && ch.gen == g0)
      bk.cv.wait_for(lk, umtxTimeout());
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
  // FreeBSD do_wake_umutex / do_wake2_umutex (kern_umtx.c). Both refuse to
  // release anyone while the word still names an owner: waking then would send
  // every sleeper into a failed re-CAS and straight back into MUTEX_WAIT, an
  // N-way stampede on each handoff. When the queue drains to <=1 the kernel --
  // not userland -- clears CONTESTED, because libthr's contested release
  // deliberately leaves the word at CONTESTED and relies on this.
  case 18:   // UMTX_OP_MUTEX_WAKE
  case 22: { // UMTX_OP_MUTEX_WAKE2
    auto &bk = umtxBucket(ptr);
    auto *p = static_cast<std::atomic<uint32_t> *>(ptr);
    std::unique_lock<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    uint32_t owner = p->load();
    if ((owner & ~UMUTEX_CONTESTED) != 0) {
      // Still held. WAKE2 repairs the bit so the eventual release still traps;
      // plain WAKE just leaves it alone.
      if (op == 22 && ch.mutexWaiters > 0)
        p->compare_exchange_strong(owner, owner | UMUTEX_CONTESTED);
      return 0;
    }
    if (ch.mutexWaiters <= 1) {
      uint32_t contested = UMUTEX_CONTESTED;
      if (p->compare_exchange_strong(contested, 0u))
        owner = 0u;
    }
    if (ch.mutexWaiters != 0)
      ch.gen++;  // release the queued MUTEX_WAIT sleepers to re-CAS
    lk.unlock();
    bk.cv.notify_all();
    return 0;
  }
  // Kernel-arbitrated mutex (UMUTEX_PRIO_INHERIT/PROTECT). libthr hands the whole
  // lock/unlock to the kernel here instead of the userland CAS + MUTEX_WAIT path;
  // *ptr is the umutex m_owner word (low 31 bits = owner tid, bit31 = contested).
  // Returning 0 without actually arbitrating lets two threads both "own" one mutex
  // (the Fios2 worker pool: tid A locks, tid B also "locks", then B's cond_wait
  // owner check [*mutex+0x28]==curthread fails -> EPERM 0x80020001). So enforce
  // real mutual exclusion on the owner word under the bucket lock.
  case 4:    // UMTX_OP_MUTEX_TRYLOCK
  case 5: {  // UMTX_OP_MUTEX_LOCK
    auto &bk = umtxBucket(ptr);
    auto *p = static_cast<std::atomic<uint32_t> *>(ptr);
    const uint32_t self = t_tid;
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
    for (;;) {
      uint32_t owner = p->load();
      uint32_t held = owner & ~UMUTEX_CONTESTED;
      if (held == 0) {                 // free: claim it atomically. A blind store
        // would race libthr's userland CAS fast path (which runs WITHOUT our bucket
        // lock): both could "win" the same free mutex -> two owners -> libthr's
        // per-thread owned-PI-mutex list gets the same mutex twice -> "Fatal error
        // 'mutex is on list'". CAS makes the claim atomic against that fast path.
        // Preserve a waiter's CONTESTED bit but don't set it spuriously (an
        // uncontended unlock must stay in userland, not hit op 6).
        if (!p->compare_exchange_strong(owner, self | (owner & UMUTEX_CONTESTED)))
          continue;                    // lost the race; re-evaluate
        umtxTrace(op, ptr, self, owner);
        return 0;
      }
      if (held == self)                // already ours (libthr counts recursion)
        return 0;
      if (op == 4)                     // trylock: don't block
        return -16 /*EBUSY*/;
      // Mark contested with a CAS too: if the owner word changed under us (a
      // concurrent userland unlock cleared it), re-evaluate instead of stomping a
      // stale owner|CONTESTED back over a now-free mutex.
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
    // libthr only reaches the kernel unlock for a CONTESTED mutex, and its
    // userland path has ALREADY verified the caller owns it (and, for the
    // contested PI/PROTECT release, already cleared the owner tid to 0/CONTESTED
    // before the syscall). So don't re-check ownership here -- the owner word is
    // often already 0 by now, and an EPERM makes libthr skip dequeueing the mutex
    // -> "Fatal error 'mutex is on list'". Just release and wake a waiter.
    auto &bk = umtxBucket(ptr);
    auto *p = static_cast<std::atomic<uint32_t> *>(ptr);
    std::unique_lock<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    uint32_t owner = p->load();
    umtxTrace(6, ptr, t_tid, owner);
    // FreeBSD do_unlock_normal (kern_umtx.c):
    //   old = casuword32(&m->m_owner, owner,
    //                    count <= 1 ? UMUTEX_UNOWNED : UMUTEX_CONTESTED);
    // "When unlocking the umtx, it must be marked as unowned if there is zero
    // or one thread only waiting for it. Otherwise, it must be marked as
    // contested." Storing a blind 0 with 2+ waiters queued hands the next
    // acquirer an UNCONTESTED word, so ITS release stays in userland and never
    // wakes the rest -- the contested chain breaks at every handoff, not just
    // the first.
    p->compare_exchange_strong(owner, ch.mutexWaiters <= 1
                                          ? 0u
                                          : UMUTEX_CONTESTED);
    ch.gen++;  // releases MUTEX_WAIT sleepers
    lk.unlock();
    bk.cv.notify_all();
    return 0;
  }
  // Kernel condvar. CV_WAIT atomically releases the umutex (uaddr1=a) and sleeps
  // on the ucond (ptr) until signaled; libthr re-locks the mutex (op 5) on return.
  // Hold the cond bucket across the mutex release so a concurrent signal can't be
  // lost between unlock and sleep. Two details real engines depend on:
  //  - ucond.c_has_waiters must be raised BEFORE the mutex is released:
  //    pthread_cond_signal only issues the CV_SIGNAL syscall when it sees the
  //    flag (it reads it under the mutex the waiter just held); without it no
  //    signal ever reaches the kernel and every wake here would be spurious.
  //  - a waiter returns only on a real signal/broadcast or its own timeout,
  //    never on the safety re-poll: libthr reports any 0/EINTR return of this
  //    op as "signaled" to the app, and engines deref state the signaling
  //    thread has not published yet (SotC fiber/stream managers).
  case 8: { // UMTX_OP_CV_WAIT: ptr=ucond, a=umutex, b=timespec, val=flags
    cvTrace("CV_WAIT", ptr, t_tid);
    auto &cbk = umtxBucket(ptr);
    auto &mbk = umtxBucket(a);
    const auto dl = cvDeadline(ptr, val, b);
    std::unique_lock<std::mutex> clk(cbk.m);
    auto &ch = cbk.chan[ptr];  // stable ref: unordered_map never moves nodes
    ch.waiters++;
    addrWatchDump("cv-wait pre", ptr, t_tid);
    static_cast<std::atomic<uint32_t> *>(ptr)->store(1);  // c_has_waiters
    addrWatchDump("cv-wait post", ptr, t_tid);
    if (a)
      addrWatchDump("cv-wait mutex", a, t_tid);
    // Snapshot the ucond and its umutex so the poll loop below can report ANY
    // guest write to them. A producer that publishes a wake with a plain store
    // instead of a syscall leaves no other trace, and libthr's uncontended
    // lock/unlock of the mutex is a userland CAS with no syscall either -- so
    // an unchanging mutex word is the evidence that the producer side of the
    // handshake never even runs, as opposed to running but never signalling.
    const bool watch = addrWatched(ptr);
    const bool watchM = a && addrWatched(a);
    uint8_t snap[32], snapM[32];
    if (watch)
      __builtin_memcpy(snap, ptr, sizeof(snap));
    if (watchM)
      __builtin_memcpy(snapM, a, sizeof(snapM));
    // Snapshot the wake generation BEFORE releasing the guest mutex: any
    // signal that lands from here on bumps gen/signals under cbk.m and the
    // wait loop below will see it. That makes it safe to DROP the cond bucket
    // while releasing the mutex -- never hold two bucket locks at once. The
    // old hold-cbk-lock-mbk order deadlocked against a second CV_WAIT whose
    // cond/mutex hashed to the opposite buckets (SotC: whole process wedged
    // ~9s in, every thread queued on two host bucket mutexes while the guest
    // umutex word itself read 0/free).
    const uint64_t g0 = ch.gen;
    if (a) {                           // release the mutex (waking its waiters)
      umtxTrace(8, a, t_tid,
                static_cast<std::atomic<uint32_t> *>(a)->load());
      auto *m = static_cast<std::atomic<uint32_t> *>(a);
      // Same release rule as op 6 (FreeBSD do_cv_wait calls do_unlock_umutex,
      // not a raw store): leave the word CONTESTED while 2+ waiters are queued,
      // or the next acquirer's release stays in userland and strands them.
      auto releaseMutex = [&](WaitChan &mch) {
        uint32_t owner = m->load();
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
    for (;;) {
      if (ch.gen != g0)                // broadcast: releases every sleeper
        break;
      if (ch.signals > 0) {            // signal: exactly one sleeper consumes it
        ch.signals--;
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
      static_cast<std::atomic<uint32_t> *>(ptr)->store(0);
    }
    return r;
  }
  case 9: {  // UMTX_OP_CV_SIGNAL: release one waiter
    cvTrace("CV_SIGNAL", ptr, t_tid);
    auto &bk = umtxBucket(ptr);
    std::lock_guard<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    if (ch.waiters > ch.signals)       // signal with nobody waiting is lost
      ch.signals++;
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
  // Userland semaphore (struct _usem { u32 _has_waiters; u32 _count; u32 _flags; }
  // at ptr). SEM_WAIT publishes _has_waiters and blocks while _count == 0; the
  // poster bumps _count (userland) and calls SEM_WAKE. Returning 0 immediately
  // (the old default) turned sem_wait into a hot spin -- a savedata worker
  // (Shadow of the Tomb Raider) pegged a core re-issuing the syscall, starving
  // the threads it was waiting on. Block on _count like the other WAIT ops.
  case 19: { // UMTX_OP_SEM_WAIT
    auto &bk = umtxBucket(ptr);
    auto *hasWaiters = static_cast<std::atomic<uint32_t> *>(ptr);
    auto *count = reinterpret_cast<volatile uint32_t *>(
        static_cast<uint8_t *>(ptr) + 4);
    const auto dl = umtxRelDeadline(b);
    std::unique_lock<std::mutex> lk(bk.m);
    uint32_t z = 0;
    hasWaiters->compare_exchange_strong(z, 1);  // publish "has waiters"
    auto &ch = bk.chan[ptr];
    const uint64_t g0 = ch.gen;
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
    // FreeBSD's libthr batches deferred condition-variable wakes into an array
    // and releases all waiters for every address when the mutex is unlocked:
    // https://github.com/freebsd/freebsd-src/blob/release/9.0.0/sys/kern/kern_umtx.c#L2995-L3019
    auto **addrs = static_cast<void **>(ptr);
    for (uint64_t i = 0; i < val; ++i) {
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
  default: {
    static std::atomic<uint32_t> seen[32]{};
    if (op >= 0 && op < 32 && seen[op].fetch_add(1) == 0)
      std::printf("[umtx] unhandled op=%d\n", op);
    return 0;
  }
  }
}
} // namespace krnl
