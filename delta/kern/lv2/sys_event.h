#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "guest_abi.h"

#include "base/containers/vector.h"

#include "base/threading/condition_variable.h"
#include "base/threading/mutex.h"
#include "base/time/time.h"
#include "kern/object.h"

namespace kern {
class Process;

// FreeBSD/SCE kevent (SceKernelEvent), 0x20 bytes. The game reads ident/data
// to learn which flip completed and udata to find its own context.
struct kevent_t {  // NOLINT(readability-identifier-naming): guest ABI name
  u64 ident;
  i16 filter;
  u16 flags;
  u32 fflags;
  i64 data;
  void* udata;
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
struct ktimespec {  // NOLINT(readability-identifier-naming): guest ABI name
  i64 tv_sec;
  i64 tv_nsec;
};

// An equeue is a FreeBSD kqueue: a set of registered knotes that become ready
// when their source fires. Games wait on it for flip/vblank notifications.
class Equeue : public Object {
 public:
  Equeue(ObjectTable& objects, const char* name);
  ~Equeue() override;

  // apply changelist (EV_ADD/DELETE/...), then collect up to nout ready events,
  // blocking up to the timeout (null = forever) until at least one is ready.
  int Kevent(const kevent_t* changes,
             int nchanges,
             kevent_t* out,
             int nout,
             const ktimespec* to);

  // Mark every knote matching (ident,filter) ready and wake waiters. ident<0
  // matches any ident. Called by the vblank pump / dce on flip.
  void Trigger(i64 ident, i16 filter, i64 data);
  void TriggerGnm(i64 data, bool raw_data = false);

  // Register a knote directly (no kevent syscall). Used by the HLE VideoOut
  // flip/vblank-event APIs, which add the equeue entry on the game's behalf.
  void AddEvent(u64 ident, i16 filter, void* udata);

  // Remove a knote by (ident,filter). Returns true if one was removed.
  bool RemoveEvent(u64 ident, i16 filter);

  // What this queue is registered for, for the report a long wait triggers:
  // the event that never arrived is the one nothing here is active on.
  void ReportRegistrations(int fd);

 private:
  void ReportRegistrationsLocked(int fd);
  struct Knote {
    kevent_t ev;
    bool active = false;
    // EOP count this knote last accounted for: our submits finish inside the
    // submit call, so an event can fire before the wait is armed; comparing
    // counts turns the lost edge back into the level the title is asking about.
    u64 eop_seen = 0;
  };
  Knote* Find(u64 ident, i16 filter);

  base::Mutex m_;
  base::ConditionVariable cv_;
  base::Vector<Knote> notes_;
  bool warned_empty_wait_ = false;
  // A queue nothing has posted to since this point. A title that polls with a
  // timeout looks busy from the outside, so the idle span is what says its
  // event never came.
  base::TimeTicks idle_since_{};
  bool reported_idle_ = false;
};

// Fan a (filter,data) trigger out to every live equeue. The vblank pump uses
// this so it needn't know which equeue a flip event landed on.
void TriggerAllEqueues(i64 ident, i16 filter, i64 data);

// Count one submitted flip and post a display event with the new count. The
// engine's pacing reads data>>16 as "frames flipped", so it must track real
// flips, not the 60 Hz vblank tick (which races ahead during loading -> "frame
// number out of range").
void NoteFlip();
// A GPU end-of-pipe interrupt reached the CP: wake the graphics-core events.
void NoteGpuEndOfPipe();

// The AGC form: libSceAgcDriver's GetEqEventType reads the event's IDENT and
// GetEqContextId reads its DATA, so the data has to be the context id the
// RELEASE_MEM packet carried (or its fence value when it carried none). A
// counter of our own there means the title never recognises its own submit.
void NoteGpuEndOfPipeCtx(u64 context_id);

// End-of-pipe interrupts raised so far (see equeue::knote::eop_seen).
u64 GpuEndOfPipeCount();
u64 FlipCount();

int PS4ABI sys_kqueue();
int PS4ABI sys_kqueueex(const char* name, int flags);
int PS4ABI sys_kevent(int kq,
                      const kevent_t* changelist,
                      int nchanges,
                      kevent_t* eventlist,
                      int nevents,
                      const ktimespec* to);

int PS4ABI sys_eport_create();
int PS4ABI sys_eport_delete();
int PS4ABI sys_eport_trigger();
int PS4ABI sys_eport_open();
int PS4ABI sys_eport_close();

}  // namespace kern
