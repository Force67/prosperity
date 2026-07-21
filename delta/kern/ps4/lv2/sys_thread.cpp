
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
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

  std::thread([gthread, tid, started] {
    t_tid = tid;
    t_started = started.get();
    cpu::backend().runGuestThread(gthread);
  }).detach();

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
// Per-address wake state. CV_WAIT has no value to re-check, so a waiter needs
// proof "a signal aimed at MY address happened since I went to sleep" that
// survives bucket collisions and re-poll ticks; the value-WAIT ops use it to
// keep real futex semantics (an explicit WAKE releases the waiter even if the
// value never changed). unordered_map guarantees reference stability across
// inserts, so a sleeping waiter may hold its WaitChan& while others register.
struct WaitChan {
  uint64_t gen = 0;      // bumped by WAKE/BROADCAST: releases every waiter
  uint64_t signals = 0;  // pending single-waiter releases (CV_SIGNAL)
  uint32_t waiters = 0;  // live CV_WAIT sleepers (drives ucond c_has_waiters)
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
}  // namespace

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
  using namespace std::chrono_literals;
  markThreadStarted();  // first sync point => our init is done
  switch (op) {
  // WAIT/WAKE both take the bucket lock, so a wake can't slip in between a
  // waiter's value check and its sleep => no lost wake. A waiter leaves ONLY
  // when the value changed, an explicit WAKE hit this address (gen bump), or
  // the caller's own timeout expired -- never on the internal re-poll tick.
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
    const uint64_t g0 = ch.gen;
    while (!changed() && ch.gen == g0) {
      if (dl && std::chrono::steady_clock::now() >= *dl)
        return -SysError::eTIMEDOUT;
      bk.cv.wait_for(lk, umtxTimeout());
    }
    return 0;
  }
  case 17: { // UMTX_OP_MUTEX_WAIT: block while the umutex is owned
    auto &bk = umtxBucket(ptr);
    auto *p = static_cast<volatile uint32_t *>(ptr);
    std::unique_lock<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    const uint64_t g0 = ch.gen;
    // A spurious exit here is harmless (libthr re-runs its CAS loop), but
    // still leave only on "free" or an explicit MUTEX_WAKE so a heavily
    // contended mutex doesn't degrade into a 2ms spin per waiter.
    while ((*p & ~UMUTEX_CONTESTED) != 0 && ch.gen == g0)
      bk.cv.wait_for(lk, umtxTimeout());
    return 0;
  }
  case 3:    // UMTX_OP_WAKE
  case 16:   // UMTX_OP_WAKE_PRIVATE
  case 18:   // UMTX_OP_MUTEX_WAKE
  case 22: { // UMTX_OP_MUTEX_WAKE2
    auto &bk = umtxBucket(ptr);
    std::lock_guard<std::mutex> lk(bk.m);
    bk.chan[ptr].gen++;
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
    umtxTrace(6, ptr, t_tid, p->load());
    p->store(0);                       // release
    bk.chan[ptr].gen++;                // releases MUTEX_WAIT sleepers
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
    auto &cbk = umtxBucket(ptr);
    auto &mbk = umtxBucket(a);
    const auto dl = cvDeadline(ptr, val, b);
    std::unique_lock<std::mutex> clk(cbk.m);
    auto &ch = cbk.chan[ptr];  // stable ref: unordered_map never moves nodes
    ch.waiters++;
    static_cast<std::atomic<uint32_t> *>(ptr)->store(1);  // c_has_waiters
    if (a) {                           // release the mutex (waking its waiters)
      umtxTrace(8, a, t_tid,
                static_cast<std::atomic<uint32_t> *>(a)->load());
      auto *m = static_cast<std::atomic<uint32_t> *>(a);
      if (&mbk == &cbk) {              // same bucket: already locked, don't re-lock
        m->store(0);
        mbk.chan[a].gen++;
      } else {
        std::lock_guard<std::mutex> mlk(mbk.m);
        m->store(0);
        mbk.chan[a].gen++;
      }
      mbk.cv.notify_all();
    }
    const uint64_t g0 = ch.gen;
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
    }
    if (--ch.waiters == 0) {           // last one out lowers c_has_waiters
      ch.signals = 0;                  // unconsumed signals don't outlive waiters
      static_cast<std::atomic<uint32_t> *>(ptr)->store(0);
    }
    return r;
  }
  case 9: {  // UMTX_OP_CV_SIGNAL: release one waiter
    auto &bk = umtxBucket(ptr);
    std::lock_guard<std::mutex> lk(bk.m);
    auto &ch = bk.chan[ptr];
    if (ch.waiters > ch.signals)       // signal with nobody waiting is lost
      ch.signals++;
    bk.cv.notify_all();
    return 0;
  }
  case 10: { // UMTX_OP_CV_BROADCAST: release all waiters
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
      std::lock_guard<std::mutex> lk(bk.m);
      bk.chan[addrs[i]].gen++;
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
