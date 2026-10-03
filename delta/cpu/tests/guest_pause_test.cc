#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xbyak.h>

#include "base/atomic.h"
#include "base/memory/unique_pointer.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "cpu/backend.h"
#include "guest/pause.h"
#include "guest/session.h"
#include "guest_abi.h"
#include "host_memory/host_memory.h"
#include "kern/lv2/sys_mem.h"

namespace {
template <class Predicate>
bool WaitFor(Predicate predicate) {
  for (int i = 0; i < 1000; ++i) {
    if (predicate())
      return true;
    base::SleepForMilliseconds(1);
  }
  return false;
}

class CounterGuest {
 public:
  CounterGuest() : code_(128) {
    code_.mov(code_.rax, reinterpret_cast<uintptr_t>(&counter));
    code_.mov(code_.rdx, reinterpret_cast<uintptr_t>(&stop_));
    Xbyak::Label loop;
    code_.L(loop);
    code_.lock();
    code_.inc(code_.qword[code_.rax]);
    code_.cmp(code_.byte[code_.rdx], 0);
    code_.je(loop);
    code_.ret();
    const auto entry = reinterpret_cast<uintptr_t>(code_.getCode());
    guest::RegisterCode(entry, code_.getSize());
    auto* handle = cpu::GetBackend().CreateGuestThread(entry, nullptr, 0);
    thread_ = base::MakeUnique<base::Thread>(
        "pause-test", [handle] { cpu::GetBackend().RunGuestThread(handle); },
        true);
  }
  ~CounterGuest() {
    guest::SetPaused(false);
    stop_.store(true);
    thread_->Join();
  }
  base::Atomic<u64> counter{0};

 private:
  base::Atomic<bool> stop_{false};
  Xbyak::CodeGenerator code_;
  base::UniquePointer<base::Thread> thread_;
};

u64 PS4ABI Sum14(u64 a,
                 u64 b,
                 u64 c,
                 u64 d,
                 u64 e,
                 u64 f,
                 u64 g,
                 u64 h,
                 u64 i,
                 u64 j,
                 u64 k,
                 u64 l,
                 u64 m,
                 u64 n) {
  return a + b + c + d + e + f + g + h + i + j + k + l + m + n;
}
double PS4ABI SumFloat(double a, double b) {
  return a + b;
}

base::Mutex g_host_lock;
base::Atomic<bool> g_entered{false}, g_release{false};
struct Result {
  u64 first;
  u64 second;
};
Result PS4ABI BlockingHostCall(u64 value) {
  base::LockGuard<base::Mutex> lock(g_host_lock);
  g_entered.store(true);
  while (!g_release.load())
    base::SleepForMilliseconds(1);
  return {value, value + 1};
}
}  // namespace

TEST(GuestPause, StopsPureGuestCodeAcrossThreadsAndResumesIt) {
  CounterGuest first, second;
  ASSERT_TRUE(
      WaitFor([&] { return first.counter.load() && second.counter.load(); }));
  guest::SetPaused(true);
  base::SleepForMilliseconds(30);
  const auto a = first.counter.load(), b = second.counter.load();
  base::SleepForMilliseconds(30);
  EXPECT_EQ(first.counter.load(), a);
  EXPECT_EQ(second.counter.load(), b);
  CounterGuest new_thread;
  base::SleepForMilliseconds(20);
  EXPECT_EQ(new_thread.counter.load(), 0u);
  guest::SetPaused(false);
  EXPECT_TRUE(WaitFor([&] {
    return first.counter.load() > a && second.counter.load() > b &&
           new_thread.counter.load() > 0;
  }));
}

TEST(GuestPause, NativeThunksPreserveRegisterStackAndFloatingPointArguments) {
  using Integers = decltype(&Sum14);
  auto integers = reinterpret_cast<Integers>(
      cpu::MakeHostThunk(reinterpret_cast<void*>(&Sum14)));
  EXPECT_EQ(integers(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 105u);
  auto floats = reinterpret_cast<decltype(&SumFloat)>(
      cpu::MakeHostThunk(reinterpret_cast<void*>(&SumFloat)));
  EXPECT_DOUBLE_EQ(floats(2.25, 4.5), 6.75);
}

TEST(GuestPause, SyscallThunkDoesNotReadAboveTheKernelStack) {
  const auto page_size = static_cast<mem_size>(::sysconf(_SC_PAGESIZE));
  void* stack = ::mmap(nullptr, page_size * 2, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(stack, MAP_FAILED);
  auto* top = static_cast<char*>(stack) + page_size;
  ASSERT_EQ(::mprotect(top, page_size, PROT_NONE), 0);
  Xbyak::CodeGenerator call(128);
  call.push(call.r12);
  call.mov(call.r12, call.rsp);
  call.mov(call.rsp, reinterpret_cast<uintptr_t>(top));
  call.mov(call.rax, cpu::MakeSyscallPauseThunk(
                         reinterpret_cast<const void*>(&SumFloat)));
  call.call(call.rax);
  call.mov(call.rsp, call.r12);
  call.pop(call.r12);
  call.ret();
  auto invoke = call.getCode<decltype(&SumFloat)>();
  EXPECT_DOUBLE_EQ(invoke(2.25, 4.5), 6.75);
  EXPECT_EQ(::munmap(stack, page_size * 2), 0);
}

TEST(GuestPause, EntryGatePreservesArgumentsWhileParked) {
  guest::ThreadRegistration registration;
  auto integers = reinterpret_cast<decltype(&Sum14)>(
      cpu::MakeHostThunk(reinterpret_cast<void*>(&Sum14)));
  auto floats = reinterpret_cast<decltype(&SumFloat)>(
      cpu::MakeHostThunk(reinterpret_cast<void*>(&SumFloat)));
  auto paused_call = [](auto invoke) {
    guest::SetPaused(true);
    base::Thread resume(
        "resume-entry-gate",
        [] {
          base::SleepForMilliseconds(30);
          guest::SetPaused(false);
        },
        true);
    const auto result = invoke();
    resume.Join();
    return result;
  };
  EXPECT_EQ(paused_call([&] {
              return integers(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
            }),
            105u);
  EXPECT_DOUBLE_EQ(paused_call([&] { return floats(2.25, 4.5); }), 6.75);
}

TEST(GuestPause, HostCallReleasesItsLockBeforeTheReturnGateParks) {
  g_entered.store(false);
  g_release.store(false);
  base::Atomic<bool> done{false};
  Result result{};
  auto call = reinterpret_cast<decltype(&BlockingHostCall)>(
      cpu::MakeHostThunk(reinterpret_cast<void*>(&BlockingHostCall)));
  base::Thread worker(
      "host-gate-test",
      [&] {
        guest::ThreadRegistration registration;
        result = call(41);
        done.store(true);
      },
      true);
  EXPECT_TRUE(WaitFor([] { return g_entered.load(); }));
  guest::SetPaused(true);
  g_release.store(true);
  base::SleepForMilliseconds(30);
  const bool released = g_host_lock.try_lock();
  EXPECT_TRUE(released);
  if (released)
    g_host_lock.unlock();
  EXPECT_FALSE(done.load());
  guest::SetPaused(false);
  worker.Join();
  EXPECT_TRUE(done.load());
  EXPECT_EQ(result.first, 41u);
  EXPECT_EQ(result.second, 42u);
}

TEST(GuestSession, StopsPausedGuestAndStartsAnotherSession) {
  for (int session = 0; session < 3; ++session) {
    guest::BeginSession();
    base::Atomic<u64> counter{0};
    Xbyak::CodeGenerator code(128);
    code.mov(code.rax, reinterpret_cast<uintptr_t>(&counter));
    Xbyak::Label loop;
    code.L(loop);
    code.lock();
    code.inc(code.qword[code.rax]);
    code.jmp(loop);
    const auto entry = reinterpret_cast<uintptr_t>(code.getCode());
    guest::RegisterCode(entry, code.getSize());
    auto* handle = cpu::GetBackend().CreateGuestThread(entry, nullptr, 0);
    ASSERT_TRUE(guest::SpawnThread("session-test", [handle] {
      cpu::GetBackend().RunGuestThread(handle);
    }));
    EXPECT_TRUE(WaitFor([&] { return counter.load() != 0; }));
    guest::SetPaused(true);
    base::SleepForMilliseconds(20);
    guest::RequestStop();
    guest::JoinThreads();
    EXPECT_EQ(guest::LiveThreads(), 0u);
    guest::ResetCodeRanges();
  }
  guest::BeginSession();
}

TEST(GuestSession, StopsGuestAfterSwitchingStacks) {
  guest::BeginSession();
  constexpr size_t kStackSize = 0x20000;
  void* stack = mmap(nullptr, kStackSize, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(stack, MAP_FAILED);
  base::Atomic<u64> counter{0};
  Xbyak::CodeGenerator code(128);
  code.mov(code.rsp, reinterpret_cast<uintptr_t>(stack) + kStackSize - 0x100);
  code.mov(code.rax, reinterpret_cast<uintptr_t>(&counter));
  Xbyak::Label loop;
  code.L(loop);
  code.lock();
  code.inc(code.qword[code.rax]);
  code.jmp(loop);
  const auto entry = reinterpret_cast<uintptr_t>(code.getCode());
  guest::RegisterCode(entry, code.getSize());
  auto* handle = cpu::GetBackend().CreateGuestThread(entry, nullptr, 0);
  ASSERT_TRUE(guest::SpawnThread("stop-switched-stack", [handle] {
    cpu::GetBackend().RunGuestThread(handle);
  }));
  EXPECT_TRUE(WaitFor([&] { return counter.load() != 0; }));
  guest::RequestStop();
  guest::JoinThreads();
  EXPECT_EQ(guest::LiveThreads(), 0u);
  munmap(stack, kStackSize);
  guest::ResetCodeRanges();
  guest::BeginSession();
}

TEST(GuestSession, StopLeavesHostLocksUnlocked) {
  guest::BeginSession();
  g_entered.store(false);
  g_release.store(false);
  const auto thunk =
      cpu::MakeHostThunk(reinterpret_cast<void*>(&BlockingHostCall));
  auto* handle = cpu::GetBackend().CreateGuestThread(thunk, nullptr, 0);
  ASSERT_TRUE(guest::SpawnThread("stop-host-call", [handle] {
    cpu::GetBackend().RunGuestThread(handle);
  }));
  EXPECT_TRUE(WaitFor([] { return g_entered.load(); }));
  guest::RequestStop();
  g_release.store(true);
  guest::JoinThreads();
  const bool unlocked = g_host_lock.try_lock();
  EXPECT_TRUE(unlocked);
  if (unlocked)
    g_host_lock.unlock();
  EXPECT_EQ(guest::LiveThreads(), 0u);
  guest::BeginSession();
}

TEST(GuestSession, CancelsBackgroundSleepAndWait) {
  guest::BeginSession();
  base::Mutex mutex;
  base::ConditionVariable cv;
  base::Atomic<bool> waiting{false};
  ASSERT_TRUE(guest::SpawnThread("stop-wait", [&] {
    base::UniqueLock lock(mutex);
    waiting.store(true);
    EXPECT_FALSE(guest::Wait(cv, lock, [] { return false; }));
  }));
  ASSERT_TRUE(guest::SpawnThread("stop-sleep",
                                 [] { guest::SleepForMilliseconds(60'000); }));
  EXPECT_TRUE(WaitFor([&] { return waiting.load(); }));
  const auto start = base::TimeTicks::Now();
  guest::RequestStop();
  guest::JoinThreads();
  EXPECT_LT(base::TimeTicks::Now() - start, base::Seconds(1));
  EXPECT_EQ(guest::LiveThreads(), 0u);
  guest::BeginSession();
}

TEST(GuestSession, ReleasesReservationsAndReusesGuestArena) {
  void* previous = nullptr;
  for (int session = 0; session < 3; ++session) {
    guest::BeginSession();
    host_memory::BeginMemorySession();
    auto* arena = kern::AllocLowGuest(0x10000);
    ASSERT_NE(arena, nullptr);
    if (previous)
      EXPECT_EQ(arena, previous);
    previous = arena;
    void* reservation = host_memory::AllocMem(
        nullptr, 0x10000, host_memory::PageProtection::kPriv,
        host_memory::AllocationType::kReserve);
    ASSERT_NE(reservation, nullptr);
    ASSERT_NE(host_memory::AllocMem(reservation, 0x4000,
                                    host_memory::PageProtection::kW,
                                    host_memory::AllocationType::kCommit),
              nullptr);
    EXPECT_EQ(host_memory::SessionMappedBytes(), 0x20000u);
    host_memory::FreeMem(static_cast<u8*>(reservation) + 0x4000, 0x4000);
    EXPECT_EQ(host_memory::SessionMappedBytes(), 0x1c000u);
    guest::RequestStop();
    guest::JoinThreads();
    guest::ResetSessionResources();
    EXPECT_EQ(host_memory::EndMemorySession(), 0x1c000u);
    EXPECT_EQ(host_memory::SessionMappedBytes(), 0u);
    EXPECT_FALSE(host_memory::IsMemoryRangeMapped(arena, 0x10000));
    EXPECT_FALSE(host_memory::IsMemoryRangeMapped(reservation, 0x4000));
  }
  guest::BeginSession();
}
