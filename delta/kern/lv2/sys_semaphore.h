#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <condition_variable>
#include <mutex>

#include "kern/object.h"

namespace krnl {
class proc;

// SCE kernel semaphore ("osem"): a counting semaphore threads wait on (taking N
// units) and others post to. Like the event flag, this is a core sync primitive;
// stubbed (the syscalls were null_handler, returning 0 without blocking) waiters
// never actually wait, so a consumer races ahead of the producer it depends on
// and reads half-built / null state.
class semaphore : public kObject {
public:
  semaphore(proc *p, const char *name, int init, int max);

  // Block until at least `need` units are available, then take them. Waits up to
  // *timeoutUs micros (null = forever). Returns 0, -eTIMEDOUT, or -errno.
  int wait(int need, uint32_t *timeoutUs);
  int trywait(int need);
  int post(int count);
  // Wake every waiter; if setCount >= 0 the count is reset to it. Reports how
  // many were waiting via *numWaiters.
  int cancel(int setCount, int *numWaiters);

  const base::String &fname() const { return name; }

private:
  std::mutex m;
  std::condition_variable cv;
  int count;
  int maxCount;
  int waiters = 0;
};

int PS4ABI sys_osem_create(const char *name, uint32_t attr, int init, int max);
int PS4ABI sys_osem_delete(int id);
int PS4ABI sys_osem_open(const char *name);
int PS4ABI sys_osem_close(int id);
int PS4ABI sys_osem_wait(int id, int need, uint32_t *timeoutUs);
int PS4ABI sys_osem_trywait(int id, int need);
int PS4ABI sys_osem_post(int id, int count);
int PS4ABI sys_osem_cancel(int id, int setCount, int *numWaiters);
}  // namespace krnl
