#pragma once

#include <new>

#include "base/functional/function.h"
#include "base/strings/string_ref.h"
#include "base/threading/condition_variable.h"
#include "base/time/time.h"

namespace guest {

// Reset callbacks run after session workers have joined. Workers must use
// SpawnThread and cancellable waits before accessing guest resources.
void BeginSession();
bool Stopping();
void RequestStop();
bool SpawnThread(base::StringRef name,
                 base::Function<void()> run,
                 size_t stack_size = 0);
void JoinThreads();
size_t LiveThreads();
void SleepForMilliseconds(u64 milliseconds);
void SleepForMicroseconds(u64 microseconds);
void RegisterSessionReset(void (*reset)());
void ResetSessionResources();

struct SessionReset {
  explicit SessionReset(void (*reset)()) { RegisterSessionReset(reset); }
};

template <class T>
void ResetResource(T& value) {
  value.~T();
  new (&value) T{};
}

template <class T, size_t N>
void ResetResource(T (&values)[N]) {
  for (auto& value : values)
    ResetResource(value);
}

template <class Lock, class Predicate>
bool WaitFor(base::ConditionVariable& cv,
             Lock& lock,
             base::TimeDelta timeout,
             Predicate ready) {
  const auto deadline = base::TimeTicks::Now() + timeout;
  while (!ready()) {
    if (Stopping())
      return false;
    const auto left = deadline - base::TimeTicks::Now();
    if (left <= base::TimeDelta())
      return false;
    cv.WaitFor(lock,
               left < base::Milliseconds(20) ? left : base::Milliseconds(20));
  }
  return true;
}

template <class Lock, class Predicate>
bool Wait(base::ConditionVariable& cv, Lock& lock, Predicate ready) {
  while (!ready()) {
    if (Stopping())
      return false;
    cv.WaitFor(lock, base::Milliseconds(20));
  }
  return true;
}

}  // namespace guest
