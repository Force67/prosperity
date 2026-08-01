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

// Kernel event-flag attr / wait-mode bits. The attr is a persistent default
// stored on the object; the mode is supplied per wait/trywait call. Both share
// the same bit layout.
//   0x0001  AND  -- all bits of pattern must be set
//   0x0002  OR   -- any bit of pattern
//   0x0010  CLEAR_ALL -- clear every bit on a successful wait
//   0x0020  CLEAR_PAT -- clear only the waited-for bits on success
//   0x0100  SHARED    -- publish in the global name table (cross-process open)
//   0x1000  (internal) destroyed/cancelling marker, checked by evf_wait/osem_wait
// The kernel rejects: attr/mode with both AND and OR (bits & 3 == 3), both clear
// modes (bits & 0x30 == 0x30), or any bit outside the 0x133 mask.
enum evfAttr : uint32_t {
  kEvfAnd = 0x01,
  kEvfOr = 0x02,
  kEvfClearAll = 0x10,
  kEvfClearPat = 0x20,
  kEvfShared = 0x100,
};

// Kernel event_flag object (0x58 bytes):
//   +0x00  uint64  bits       -- current pattern
//   +0x08  mtx     lock       -- "evf mtx"
//   +0x28  cv      cond       -- "evf cv"
//   +0x38  void*   waiters.fst -- sleepq head
//   +0x40  void*   waiters.lst -- sleepq tail
//   +0x48  uint32  attr       -- persisted create flags
//   +0x4c  uint32  nwaiters
//   +0x50  uint32  is_shared
//   +0x54  uint32  proc_type
// We model it with host primitives (std::mutex/cv) rather than the exact layout.

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
  // Wake every waiter; returns how many were released. The woken threads see
  // an error status (not a match), matching kernel evf_cancel semantics.
  int cancel(uint64_t pattern);

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
    bool cancelled = false;
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
