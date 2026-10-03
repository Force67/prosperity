#include "guest/pause.h"
#include "guest/session.h"

#include <cerrno>

#include "base/atomic.h"
#include "base/containers/array.h"
#include "base/containers/vector.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"

#if defined(__linux__)
#include <linux/futex.h>
#include <pthread.h>
#include <signal.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>
#endif

namespace guest {
namespace {
base::Atomic<i32> g_paused{0};
thread_local bool t_guest = false;
thread_local void (*t_exit)() = nullptr;
thread_local bool (*t_is_guest_pc)(uintptr_t) = nullptr;
thread_local base::Atomic<bool> t_parked{false};
base::Mutex g_mutex;
struct Thread {
  int tid;
  base::Atomic<bool>* parked;
};
base::Vector<Thread> g_threads;
struct CodeRange {
  uintptr_t begin;
  uintptr_t end;
};
base::Array<CodeRange, 4096> g_code;
base::Atomic<u32> g_code_count{0};
bool g_signal_installed = false;

#if defined(__linux__)
void PauseSignal(int, siginfo_t*, void* context) {
  if (!t_guest || !Paused())
    return;
  auto* uc = static_cast<ucontext_t*>(context);
#if defined(__x86_64__)
  const auto pc = static_cast<uintptr_t>(uc->uc_mcontext.gregs[REG_RIP]);
#elif defined(__aarch64__)
  const auto pc = static_cast<uintptr_t>(uc->uc_mcontext.pc);
#else
  return;
#endif
  if (t_is_guest_pc && t_is_guest_pc(pc)) {
    PausePoint();
    return;
  }
  // Host calls finish and park at their return gate, outside emulator locks.
  const u32 count = g_code_count.load(base::memory_order_acquire);
  for (u32 i = 0; i < count; ++i) {
    if (pc >= g_code[i].begin && pc < g_code[i].end) {
      PausePoint();
      return;
    }
  }
}
#endif
}  // namespace

bool Paused() {
  return g_paused.load(base::memory_order_acquire) != 0;
}

const i32* PauseFlagAddress() {
  static_assert(sizeof(g_paused) == sizeof(i32));
  return reinterpret_cast<const i32*>(&g_paused);
}

void PausePoint() {
  if (!t_guest || !Paused())
    return;
  const int saved_errno = errno;
  t_parked.store(true, base::memory_order_release);
  while (Paused() && !Stopping()) {
#if defined(__linux__)
    ::syscall(SYS_futex, &g_paused, FUTEX_WAIT_PRIVATE, 1, nullptr, nullptr, 0);
#else
    base::SleepForMilliseconds(1);
#endif
  }
  t_parked.store(false, base::memory_order_release);
  errno = saved_errno;
  if (Stopping() && t_exit)
    t_exit();
}

void PollPause() {
#if defined(__linux__)
  if (!Paused())
    return;
  base::LockGuard<base::Mutex> lock(g_mutex);
  if (!g_signal_installed)
    return;
  for (const auto& thread : g_threads)
    if (!thread.parked->load(base::memory_order_acquire))
      ::syscall(SYS_tgkill, ::getpid(), thread.tid, SIGUSR2);
#endif
}

void SetPaused(bool paused) {
  g_paused.store(paused ? 1 : 0, base::memory_order_release);
#if defined(__linux__)
  ::syscall(SYS_futex, &g_paused, FUTEX_WAKE_PRIVATE, 0x7fffffff, nullptr,
            nullptr, 0);
#endif
  if (paused)
    PollPause();
}

bool OnGuestThread() {
  return t_guest;
}

void RegisterCode(uintptr_t address, mem_size size) {
  if (!address || !size)
    return;
  base::LockGuard<base::Mutex> lock(g_mutex);
  const u32 count = g_code_count.load(base::memory_order_relaxed);
  if (count < g_code.size()) {
    g_code[count] = {address, address + size};
    g_code_count.store(count + 1, base::memory_order_release);
  }
}

void SetThreadExitHandler(void (*exit)()) {
  t_exit = exit;
}

void ResetCodeRanges() {
  base::LockGuard lock(g_mutex);
  g_threads = {};
  g_code_count.store(0);
}

ThreadRegistration::ThreadRegistration(bool (*is_guest_pc)(uintptr_t)) {
  if (t_guest)
    return;
  owner_ = true;
  t_is_guest_pc = is_guest_pc;
  t_guest = true;
#if defined(__linux__)
  {
    base::LockGuard<base::Mutex> lock(g_mutex);
    if (!g_signal_installed) {
      struct sigaction action {};
      action.sa_sigaction = PauseSignal;
      action.sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK;
      sigemptyset(&action.sa_mask);
      g_signal_installed = ::sigaction(SIGUSR2, &action, nullptr) == 0;
    }
    g_threads.push_back({static_cast<int>(::syscall(SYS_gettid)), &t_parked});
  }
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGUSR2);
  ::pthread_sigmask(SIG_UNBLOCK, &signals, nullptr);
#endif
}

ThreadRegistration::~ThreadRegistration() {
  if (!owner_)
    return;
#if defined(__linux__)
  base::LockGuard<base::Mutex> lock(g_mutex);
  for (mem_size i = 0; i < g_threads.size(); ++i) {
    if (g_threads[i].parked == &t_parked) {
      g_threads.erase(g_threads.begin() + i);
      break;
    }
  }
#endif
  t_guest = false;
  t_is_guest_pc = nullptr;
  t_exit = nullptr;
}

}  // namespace guest
