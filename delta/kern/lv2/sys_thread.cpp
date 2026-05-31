
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
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

#include "../module.h"
#include "../proc.h"
#include "sys_thread.h"

namespace krnl {
moduleInfo *called_in(void *addr);

// Per-thread guest thread id (sys_thr_self). Main thread is 1.
static __attribute__((tls_model("initial-exec"))) thread_local uint32_t t_tid = 1;
static std::atomic<uint32_t> g_nextTid{2};

// Thread-startup handshake: sys_thr_new blocks until the new thread has run its
// init and reached its first sync point (umtx). The game spawns workers that
// produce shared state the main thread then reads with no explicit ordering
// (it relies on the worker, on another core, having finished). Without the
// head start the main thread races ahead and reads not-yet-produced data.
static std::mutex g_startM;
static std::condition_variable g_startCv;
static __attribute__((tls_model("initial-exec"))) thread_local std::atomic<bool>
    *t_started = nullptr;

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

int PS4ABI sys_thr_new(thr_param *p, int size) {
  uint32_t tid = g_nextTid.fetch_add(1);
  std::printf("[thr_new] tid=%u start=%p arg=%p stack=%p+%#zx tls=%p\n", tid,
              (void *)p->start_func, p->arg, (void *)p->stack_base,
              p->stack_size, (void *)p->tls_base);

  if (p->child_tid)
    *p->child_tid = tid;
  if (p->parent_tid)
    *p->parent_tid = tid;

  auto fn = p->start_func;
  auto arg = p->arg;
  auto fsbase = reinterpret_cast<uint64_t>(p->tls_base);
  auto started = std::make_shared<std::atomic<bool>>(false);

  // Run on the host thread's (large) stack, NOT the guest's thr_param stack: our
  // host syscall handlers execute on whatever stack the guest code is using, and
  // the guest stack (e.g. 64 KiB) is far too small for them (std::thread/printf/
  // C++ exceptions overflow it). The host thread stack is bigger and works.
  std::thread([fn, arg, fsbase, tid, started] {
    setThreadFsBase(fsbase);
    t_tid = tid;
    t_started = started.get();
    fn(arg);
  }).detach();

  // Wait for the new thread to finish its init and hit its first sync point so
  // it wins the races the game expects it to. Bounded so a thread that never
  // syncs can't hang us.
  std::unique_lock<std::mutex> lk(g_startM);
  g_startCv.wait_for(lk, std::chrono::milliseconds(200),
                     [&] { return started->load(); });
  return 0;
}

int PS4ABI sys_thr_self(uint32_t *tid) {
  *tid = t_tid;
  return 0;
}

int PS4ABI sys_rtprio_thread(int a1, uint64_t a2, thread_prio *rtp) {
  rtp->type = 3; /*normal time sharing process*/
  rtp->prio = 1; /*almost highest prio*/
  return 0;
}

// Address-keyed wait/wake (a small futex). A fixed bucket array avoids per-
// address allocation; hash collisions only cause harmless spurious wakeups
// since every waiter re-checks its condition.
namespace {
struct Bucket {
  std::mutex m;
  std::condition_variable cv;
};
std::array<Bucket, 256> g_umtxBuckets;
Bucket &umtxBucket(const void *a) {
  return g_umtxBuckets[(reinterpret_cast<uintptr_t>(a) >> 4) & 0xff];
}
constexpr uint32_t UMUTEX_CONTESTED = 0x80000000u;
}  // namespace

int PS4ABI sys_umtx_op(void *ptr, int op, uint32_t val, void *a, void *b) {
  using namespace std::chrono_literals;
  markThreadStarted();  // first sync point => our init is done
  switch (op) {
  case 2:    // UMTX_OP_WAIT
  case 11:   // UMTX_OP_WAIT_UINT
  case 15: { // UMTX_OP_WAIT_UINT_PRIVATE
    auto &bk = umtxBucket(ptr);
    std::unique_lock<std::mutex> lk(bk.m);
    // sleep only while the value still matches what the caller waited on
    if (*static_cast<volatile uint32_t *>(ptr) == val)
      bk.cv.wait_for(lk, 5ms);
    return 0;
  }
  case 17: { // UMTX_OP_MUTEX_WAIT: block while the umutex is owned
    auto &bk = umtxBucket(ptr);
    std::unique_lock<std::mutex> lk(bk.m);
    if ((*static_cast<volatile uint32_t *>(ptr) & ~UMUTEX_CONTESTED) != 0)
      bk.cv.wait_for(lk, 5ms);  // timeout guards against a missed wake
    return 0;
  }
  case 3:    // UMTX_OP_WAKE
  case 16:   // UMTX_OP_WAKE_PRIVATE
  case 18:   // UMTX_OP_MUTEX_WAKE
  case 22: { // UMTX_OP_MUTEX_WAKE2
    auto &bk = umtxBucket(ptr);
    std::lock_guard<std::mutex> lk(bk.m);
    bk.cv.notify_all();
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