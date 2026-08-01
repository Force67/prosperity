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
//
// Kernel osem object (0x54 bytes):
//   +0x00  mtx     lock        -- "osem mtx"
//   +0x20  cv      cond        -- "osem cv"
//   +0x30  void*   waiters.fst -- sleepq head
//   +0x38  void*   waiters.lst -- sleepq tail
//   +0x40  int32   count       -- current available units
//   +0x44  int32   attr        -- create flags (byte +0x45 bit 0x10 = deleted)
//   +0x48  int32   initCount   -- value reset to on cancel(setCount<0)
//   +0x4c  int32   maxCount    -- ceiling; post past this is EINVAL
//   +0x50  int32   nwaiters
class semaphore : public kObject {
public:
  semaphore(proc *p, const char *name, int init, int max);

  // Block until at least `need` units are available, then take them. Waits up to
  // *timeoutUs micros (null = forever). Returns 0, -eTIMEDOUT, or -errno.
  // If `need` exceeds maxCount the request can never be satisfied: the kernel
  // returns -eINVAL without blocking.
  int wait(int need, uint32_t *timeoutUs);
  // Non-blocking: takes `need` units if available. Returns -eBUSY if the count
  // is too low but the request is otherwise valid, or -eINVAL if need > max
  // (impossible request).
  int trywait(int need);
  // Add `count` units. Returns -eEINVAL if count+avail would exceed maxCount
  // (the count is left unchanged in that case, matching the kernel).
  int post(int count);
  // Wake every waiter with EINTR; if setCount >= 0 the count is reset to it, if
  // setCount < 0 it is reset to the initial count. Reports how many were waiting
  // via *numWaiters. Returns -eINVAL if setCount > maxCount.
  int cancel(int setCount, int *numWaiters);

  const base::String &fname() const { return name; }

private:
  std::mutex m;
  std::condition_variable cv;
  int count;
  int maxCount;
  int initCount;
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
