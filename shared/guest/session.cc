#include "guest/session.h"

#include <pthread.h>
#include <algorithm>
#include <memory>
#include <vector>

#include "base/atomic.h"
#include "base/strings/xstring.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "guest/pause.h"

namespace guest {
namespace {
base::Atomic<bool> g_stopping{false};
base::Atomic<size_t> g_live{0};
thread_local bool t_worker = false;
struct StopWorker {};
struct State {
  base::Mutex mutex;
  std::vector<pthread_t> threads;
  std::vector<void (*)()> resets;
};
State& Session() {
  static State state;
  return state;
}
struct Thread {
  base::String name;
  base::Function<void()> run;
};
void* Run(void* pointer) {
  std::unique_ptr<Thread> thread(static_cast<Thread*>(pointer));
  base::SetCurrentThreadName(thread->name.c_str());
  t_worker = true;
  try {
    thread->run();
  } catch (const StopWorker&) {
  }
  t_worker = false;
  g_live.fetch_sub(1);
  return nullptr;
}
}  // namespace

void BeginSession() {
  g_stopping.store(false);
  SetPaused(false);
}
bool Stopping() {
  return g_stopping.load();
}
void RequestStop() {
  g_stopping.store(true);
  SetPaused(true);
}
bool SpawnThread(base::StringRef name,
                 base::Function<void()> run,
                 size_t stack_size) {
  auto& state = Session();
  base::LockGuard lock(state.mutex);
  // A guest worker may already own a CPU handle. Let its entry gate observe
  // stop and dispose of that handle instead of dropping its closure.
  auto thread = std::make_unique<Thread>(
      Thread{base::String(name.data(), name.size()), base::move(run)});
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  if (stack_size)
    pthread_attr_setstacksize(&attributes, stack_size);
  pthread_t handle;
  g_live.fetch_add(1);
  const int error = pthread_create(&handle, &attributes, Run, thread.get());
  pthread_attr_destroy(&attributes);
  if (error) {
    g_live.fetch_sub(1);
    return false;
  }
  thread.release();
  state.threads.push_back(handle);
  return true;
}
void JoinThreads() {
  auto& state = Session();
  for (;;) {
    std::vector<pthread_t> threads;
    {
      base::LockGuard lock(state.mutex);
      threads.swap(state.threads);
    }
    if (threads.empty())
      break;
    for (auto thread : threads)
      pthread_join(thread, nullptr);
  }
}
size_t LiveThreads() {
  return g_live.load();
}
void SleepForMicroseconds(u64 microseconds) {
  while (microseconds) {
    if (Stopping()) {
      if (t_worker && !OnGuestThread())
        throw StopWorker{};
      return;
    }
    const auto slice = std::min(microseconds, u64(20'000));
    base::SleepForMicroseconds(slice);
    microseconds -= slice;
  }
}
void SleepForMilliseconds(u64 milliseconds) {
  SleepForMicroseconds(milliseconds * 1000);
}
void RegisterSessionReset(void (*reset)()) {
  auto& state = Session();
  base::LockGuard lock(state.mutex);
  state.resets.push_back(reset);
}
void ResetSessionResources() {
  std::vector<void (*)()> resets;
  {
    auto& state = Session();
    base::LockGuard lock(state.mutex);
    resets = state.resets;
  }
  for (auto reset : resets)
    reset();
}
}  // namespace guest
