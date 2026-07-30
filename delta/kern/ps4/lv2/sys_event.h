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

#include <base/containers/vector.h>

#include "kern/object.h"

namespace krnl {
class proc;

// FreeBSD/SCE kevent (SceKernelEvent), 0x20 bytes. The game reads ident/data
// to learn which flip completed and udata to find its own context.
struct kevent_t {
  uint64_t ident;
  int16_t filter;
  uint16_t flags;
  uint32_t fflags;
  int64_t data;
  void *udata;
};
static_assert(sizeof(kevent_t) == 0x20, "kevent layout");

// FreeBSD kevent action flags (the ones we honour).
enum : uint16_t {
  kEV_ADD = 0x0001,
  kEV_DELETE = 0x0002,
  kEV_ENABLE = 0x0004,
  kEV_DISABLE = 0x0008,
  kEV_ONESHOT = 0x0010,
  kEV_CLEAR = 0x0020,
};

// guest timespec (kevent timeout).
struct ktimespec {
  int64_t tv_sec;
  int64_t tv_nsec;
};

// An equeue is a FreeBSD kqueue: a set of registered knotes that become ready
// when their source fires. Games wait on it for flip/vblank notifications.
class equeue : public kObject {
public:
  equeue(proc *p, const char *name);
  ~equeue() override;

  // apply changelist (EV_ADD/DELETE/...), then collect up to nout ready events,
  // blocking up to the timeout (null = forever) until at least one is ready.
  int kevent(const kevent_t *changes, int nchanges, kevent_t *out, int nout,
             const ktimespec *to);

  // Mark every knote matching (ident,filter) ready and wake waiters. ident<0
  // matches any ident. Called by the vblank pump / dce on flip.
  void trigger(int64_t ident, int16_t filter, int64_t data);

  // Register a knote directly (no kevent syscall). Used by the HLE VideoOut
  // flip/vblank-event APIs, which add the equeue entry on the game's behalf.
  void addEvent(uint64_t ident, int16_t filter, void *udata);

  // Remove a knote by (ident,filter). Returns true if one was removed.
  bool removeEvent(uint64_t ident, int16_t filter);

private:
  struct knote {
    kevent_t ev;
    bool active = false;
  };
  knote *find(uint64_t ident, int16_t filter);

  std::mutex m;
  std::condition_variable cv;
  base::Vector<knote> notes;
  bool warnedEmptyWait = false;
};

// Fan a (filter,data) trigger out to every live equeue. The vblank pump uses
// this so it needn't know which equeue a flip event landed on.
void triggerAllEqueues(int64_t ident, int16_t filter, int64_t data);

// Count one submitted flip (LLE gc submit-and-flip / HLE SubmitFlip) and post a
// display event carrying the new flip count. The engine's render-frame pacing
// reads the EVFILT_DISPLAY event's data>>16 as "how many frames have flipped",
// so it must track real flips, not the free-running 60 Hz vblank tick (which
// races ahead while the title is still loading -> GetRenderFrameParams asks for
// a frame far beyond the last produced -> "frame number out of range" halt).
void noteFlip();
uint64_t flipCount();

int PS4ABI sys_kqueue();
int PS4ABI sys_kqueueex(const char *name, int flags);
int PS4ABI sys_kevent(int kq, const kevent_t *changelist, int nchanges,
                      kevent_t *eventlist, int nevents, const ktimespec *to);
}  // namespace krnl
