#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include "base/arch.h"
#include <condition_variable>
#include <mutex>

#include <base/containers/vector.h>

#include "kern/object.h"

namespace krnl {
class proc;

// FreeBSD/SCE kevent (SceKernelEvent), 0x20 bytes. The game reads ident/data
// to learn which flip completed and udata to find its own context.
struct kevent_t {
  u64 ident;
  i16 filter;
  u16 flags;
  u32 fflags;
  i64 data;
  void *udata;
};
static_assert(sizeof(kevent_t) == 0x20, "kevent layout");

// FreeBSD kevent action flags (the ones we honour).
enum : u16 {
  kEV_ADD = 0x0001,
  kEV_DELETE = 0x0002,
  kEV_ENABLE = 0x0004,
  kEV_DISABLE = 0x0008,
  kEV_ONESHOT = 0x0010,
  kEV_CLEAR = 0x0020,
};

// guest timespec (kevent timeout).
struct ktimespec {
  i64 tv_sec;
  i64 tv_nsec;
};

// An equeue is a FreeBSD kqueue: a set of registered knotes that become ready
// when their source fires. Games wait on it for flip/vblank notifications.
class equeue : public kObject {
public:
  equeue(objectTable &objects, const char *name);
  ~equeue() override;

  // apply changelist (EV_ADD/DELETE/...), then collect up to nout ready events,
  // blocking up to the timeout (null = forever) until at least one is ready.
  int kevent(const kevent_t *changes, int nchanges, kevent_t *out, int nout,
             const ktimespec *to);

  // Mark every knote matching (ident,filter) ready and wake waiters. ident<0
  // matches any ident. Called by the vblank pump / dce on flip.
  void trigger(i64 ident, i16 filter, i64 data);
  void triggerGnm(i64 data, bool raw_data = false);

  // Register a knote directly (no kevent syscall). Used by the HLE VideoOut
  // flip/vblank-event APIs, which add the equeue entry on the game's behalf.
  void addEvent(u64 ident, i16 filter, void *udata);

  // Remove a knote by (ident,filter). Returns true if one was removed.
  bool removeEvent(u64 ident, i16 filter);

  // What this queue is registered for, for the report a long wait triggers:
  // the event that never arrived is the one nothing here is active on.
  void reportRegistrations(int fd);

private:
  void reportRegistrationsLocked(int fd);
  struct knote {
    kevent_t ev;
    bool active = false;
    // The end-of-pipe count this knote last accounted for. Our submits run to
    // completion inside the submit call, so a GPU event can fire before the
    // title has armed its wait; comparing counts turns that lost edge back
    // into the level it really is ("the GPU has finished work since you last
    // looked"), which is what the title is asking.
    u64 eop_seen = 0;
  };
  knote *find(u64 ident, i16 filter);

  std::mutex m;
  std::condition_variable cv;
  base::Vector<knote> notes;
  bool warnedEmptyWait = false;
  // A queue nothing has posted to since this point. A title that polls with a
  // timeout looks busy from the outside, so the idle span is what says its
  // event never came.
  std::chrono::steady_clock::time_point idleSince{};
  bool reportedIdle = false;
};

// Fan a (filter,data) trigger out to every live equeue. The vblank pump uses
// this so it needn't know which equeue a flip event landed on.
void triggerAllEqueues(i64 ident, i16 filter, i64 data);

// Count one submitted flip (LLE gc submit-and-flip / HLE SubmitFlip) and post a
// display event carrying the new flip count. The engine's render-frame pacing
// reads the EVFILT_DISPLAY event's data>>16 as "how many frames have flipped",
// so it must track real flips, not the free-running 60 Hz vblank tick (which
// races ahead while the title is still loading -> GetRenderFrameParams asks for
// a frame far beyond the last produced -> "frame number out of range" halt).
void noteFlip();
// A GPU end-of-pipe interrupt reached the CP: wake the graphics-core events.
void noteGpuEndOfPipe();

// The AGC form: libSceAgcDriver's GetEqEventType reads the event's IDENT and
// GetEqContextId reads its DATA, so the data has to be the context id the
// RELEASE_MEM packet carried (or its fence value when it carried none). A
// counter of our own there means the title never recognises its own submit.
void noteGpuEndOfPipeCtx(u64 context_id);

// End-of-pipe interrupts raised so far (see equeue::knote::eop_seen).
u64 gpuEndOfPipeCount();
u64 flipCount();

int PS4ABI sys_kqueue();
int PS4ABI sys_kqueueex(const char *name, int flags);
int PS4ABI sys_kevent(int kq, const kevent_t *changelist, int nchanges,
                      kevent_t *eventlist, int nevents, const ktimespec *to);
}  // namespace krnl
