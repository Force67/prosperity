
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <atomic>
#include <cstdio>
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

  // Run the guest thread entry on a real host thread with its own guest fs base
  // and tid. (Runs on the host thread's stack for now; the guest stack/guard is
  // ignored.)
  std::thread([fn, arg, fsbase, tid] {
    setThreadFsBase(fsbase);
    t_tid = tid;
    fn(arg);
  }).detach();

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

int PS4ABI sys_umtx_op(void *ptr, int op, uint32_t, void *, void *) {
  // userspace mutex op. Single-threaded boot for now: pretend every lock/wait
  // succeeds immediately. TODO: real futex-backed implementation once threads
  // run concurrently.
  (void)ptr;
  (void)op;
  return 0;
}
} // namespace krnl