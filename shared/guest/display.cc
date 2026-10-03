#include "guest/display.h"

#include <algorithm>
#include <deque>
#include <memory>

#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/time/time.h"
#include "guest/session.h"

namespace guest::display {
namespace {
struct Flip {
  u64 tick;
  base::Function<void()> complete;
};
struct State {
  base::Mutex mutex;
  base::ConditionVariable cv;
  u64 epoch = static_cast<u64>(base::TickClock::NowNs());
  u64 last_flip = 0;
  int rate = 0;
  bool started = false;
  size_t pending = 0;
  std::deque<Flip> flips;
};
State& Display() {
  static State state;
  return state;
}

u64 TickTime(u64 tick) {
  return Display().epoch + (tick * 1'000'000'000 + 59) / 60;
}

void WaitUntil(u64 deadline) {
  while (!Stopping()) {
    const u64 now = static_cast<u64>(base::TickClock::NowNs());
    if (now >= deadline)
      return;
    SleepForMicroseconds((deadline - now + 999) / 1000);
  }
}

void Run() {
  auto& state = Display();
  while (!Stopping()) {
    base::UniqueLock lock(state.mutex);
    if (!Wait(state.cv, lock, [&] { return !state.flips.empty(); }))
      return;
    Flip flip = base::move(state.flips.front());
    state.flips.pop_front();
    lock.unlock();
    WaitUntil(TickTime(flip.tick));
    if (Stopping())
      return;
    flip.complete();
    lock.lock();
    --state.pending;
    state.cv.NotifyAll();
  }
}

const SessionReset g_reset([] { ResetResource(Display()); });
}  // namespace

u64 VblankCount() {
  return (static_cast<u64>(base::TickClock::NowNs()) - Display().epoch) * 60 /
         1'000'000'000;
}

u64 VblankTimeNs() {
  return TickTime(VblankCount());
}

void WaitVblank() {
  if (kEmulateTiming)
    WaitUntil(TickTime(VblankCount() + 1));
}

void SetFlipRate(int rate) {
  auto& state = Display();
  base::LockGuard lock(state.mutex);
  state.rate = std::clamp(rate, 0, 2);
}

bool QueueFlip(int mode, base::Function<void()> complete) {
  if (Stopping())
    return false;
  if (!kEmulateTiming) {
    complete();
    return true;
  }
  auto& state = Display();
  base::UniqueLock lock(state.mutex);
  // Bound outstanding frames even when the guest does not wait on flip events.
  if (!Wait(state.cv, lock, [&] { return state.pending < 2; }) || Stopping())
    return false;
  if (!state.started) {
    state.started = SpawnThread("display", Run);
    if (!state.started)
      return false;
  }
  const u64 interval = static_cast<u64>(state.rate + 1);
  const u64 next = std::max(VblankCount() + 1, state.last_flip + interval);
  const u64 tick =
      mode == 2 ? VblankCount() : ((next + interval - 1) / interval) * interval;
  state.last_flip = tick;
  ++state.pending;
  state.flips.push_back({tick, base::move(complete)});
  state.cv.NotifyAll();
  return true;
}

bool WaitFlip(int mode, base::Function<void()> complete) {
  struct Completion {
    base::Mutex mutex;
    base::ConditionVariable cv;
    bool ready = false;
    bool cancelled = false;
  };
  auto state = std::make_shared<Completion>();
  if (!QueueFlip(mode, [state, complete = base::move(complete)] {
        base::LockGuard lock(state->mutex);
        if (!state->cancelled)
          complete();
        state->ready = true;
        state->cv.NotifyAll();
      }))
    return false;
  base::UniqueLock lock(state->mutex);
  if (!Wait(state->cv, lock, [&] { return state->ready; })) {
    state->cancelled = true;
    return false;
  }
  return true;
}

}  // namespace guest::display
