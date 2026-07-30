#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "kern/object.h"

namespace krnl {
class proc;

// SCE event flag: a 64-bit bitmask threads wait on (AND/OR a pattern) and others
// set/clear. This is a core thread-sync primitive; with it stubbed, waiters
// never block and read producer state before it is built (garbage / crashes).
class eventFlag : public kObject {
public:
  // `sticky` bits are re-asserted after every clear: used for system focus/
  // ready flags the (absent) ShellCore would keep set, so a game that polls
  // them with clear-on-wait stays "focused" instead of latching off.
  eventFlag(proc *p, const char *name, uint64_t init, uint64_t sticky = 0);

  // Wait until the bits satisfy pattern per mode (AND=all, OR=any). Blocks up to
  // *timeoutUs micros (null = forever). Writes the matched bits to *result, then
  // applies the clear mode. Returns 0 or -errno.
  int wait(uint64_t pattern, uint32_t mode, uint64_t *result,
           uint32_t *timeoutUs);
  int trywait(uint64_t pattern, uint32_t mode, uint64_t *result);
  void set(uint64_t bits);
  void clear(uint64_t bits);

  const base::String &fname() const { return name; }

  // Tid of the last set() caller: lets trywait detect the request/response
  // handshake pattern (this thread just posted a request bit and now polls for
  // the responder's done bit). See sys_evf_trywait.
  std::atomic<long> lastSetTid{0};

private:
  struct Waiter {
    uint64_t pattern;
    uint32_t mode;
    uint64_t result = 0;
    bool done = false;
  };

  bool satisfied(uint64_t pattern, uint32_t mode) const;
  int take(uint64_t pattern, uint32_t mode, uint64_t *result);
  void removeWaiter(Waiter *waiter);

  std::mutex m;
  std::condition_variable cv;
  std::vector<Waiter *> waiters;
  uint64_t bits;
  uint64_t sticky;
};

// Set `bits` on the first named event flag whose name contains `substr`.
// Returns false if no such flag exists (yet). Unlike the syscalls this takes no
// handle and touches no object table, so a HOST thread with no guest proc (the
// audio daemon stand-in, kern/ps4/audio_daemon.cpp) can signal a guest flag.
bool evfSetByNameSubstr(const char *substr, uint64_t bits);

int PS4ABI sys_evf_create(const char *name, uint32_t attr, uint64_t initPattern);
int PS4ABI sys_evf_delete(int id);
int PS4ABI sys_evf_open(const char *name);
int PS4ABI sys_evf_close(int id);
int PS4ABI sys_evf_wait(int id, uint64_t pattern, uint32_t mode,
                        uint64_t *result, uint32_t *timeoutUs);
int PS4ABI sys_evf_trywait(int id, uint64_t pattern, uint32_t mode,
                           uint64_t *result);
int PS4ABI sys_evf_set(int id, uint64_t bits);
int PS4ABI sys_evf_clear(int id, uint64_t bits);
int PS4ABI sys_evf_cancel(int id, uint64_t pattern, int *numWaiters);
}  // namespace krnl
