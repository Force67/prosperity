// Guest signal delivery: at a syscall exit, into running guest code, and
// after an unblock; the interrupted code resumes with its registers intact.

#include <gtest/gtest.h>
#include <xbyak.h>
#include <cstring>

#include "base/atomic.h"
#include "base/threading/thread.h"
#include "guest/pause.h"
#include "guest_abi.h"
#include "kern/lv2/dispatch.h"
#include "kern/lv2/sys_procid.h"
#include "kern/lv2/sys_signal.h"
#include "kern/lv2/sys_time.h"

namespace {
constexpr int kSigusr1 = 30;
constexpr int kSigusr2 = 31;

u64 g_hits = 0;
u64 g_seen_sig = 0;
volatile u64 g_flag = 0;

// handler(sig, info, uc): counts, records the signal, sets the flag.
struct Handler : Xbyak::CodeGenerator {
  Handler() {
    mov(rax, reinterpret_cast<uintptr_t>(&g_hits));
    inc(qword[rax]);
    mov(rax, reinterpret_cast<uintptr_t>(&g_seen_sig));
    mov(qword[rax], rdi);
    mov(rax, reinterpret_cast<uintptr_t>(&g_flag));
    mov(qword[rax], 1);
    ret();
  }
};

// Calls one syscall trampoline the way a lifted libkernel stub does, with
// r12 holding a value that must survive the handler.
struct Call : Xbyak::CodeGenerator {
  explicit Call(uintptr_t trampoline) {
    push(r12);
    mov(r12, 0x5a5a);
    mov(rax, trampoline);
    call(rax);
    cmp(r12, 0x5a5a);
    je("ok");
    mov(rax, -1);
    L("ok");
    pop(r12);
    ret();
  }
};

// Spins in guest code until the handler sets the flag.
struct Spin : Xbyak::CodeGenerator {
  Spin() {
    mov(rax, reinterpret_cast<uintptr_t>(&g_flag));
    L("loop");
    pause();
    cmp(qword[rax], 0);
    je("loop");
    ret();
  }
};

void Install(int sig, const Handler& h) {
  struct {
    u64 handler;
    u32 flags;
    u32 mask[4];
    u32 pad;
  } act{reinterpret_cast<u64>(h.getCode()), 0x40, {}, 0};
  ASSERT_EQ(kern::sys_sigaction(sig, &act, nullptr), 0);
}

class Signals : public ::testing::Test {
 protected:
  void SetUp() override {
    g_hits = 0;
    g_seen_sig = 0;
    g_flag = 0;
    kern::SignalThreadStart(1);
  }
  void TearDown() override { kern::SignalThreadExit(); }
  Handler handler;
};

TEST_F(Signals, DeliveredAtSyscallExit) {
  Install(kSigusr1, handler);
  ASSERT_EQ(kern::sys_kill(kern::sys_getpid(), kSigusr1), 0);
  Call call(kern::Lv2Trampoline(
      reinterpret_cast<const void*>(&kern::sys_getpid), 20));
  const u64 r = call.getCode<u64 (*)()>()();
  EXPECT_EQ(g_hits, 1u);
  EXPECT_EQ(g_seen_sig, static_cast<u64>(kSigusr1));
  EXPECT_EQ(r, static_cast<u64>(kern::sys_getpid()));  // the syscall's result
}

TEST_F(Signals, BlockedUntilUnmasked) {
  Install(kSigusr1, handler);
  u32 set[4] = {1u << (kSigusr1 - 1), 0, 0, 0};
  ASSERT_EQ(kern::sys_sigprocmask(1, set, nullptr), 0);  // SIG_BLOCK
  ASSERT_EQ(kern::sys_kill(kern::sys_getpid(), kSigusr1), 0);
  Call getpid(kern::Lv2Trampoline(
      reinterpret_cast<const void*>(&kern::sys_getpid), 20));
  getpid.getCode<u64 (*)()>()();
  EXPECT_EQ(g_hits, 0u);

  // The unblocking sigprocmask takes it on its own way out.
  struct Unblock : Xbyak::CodeGenerator {
    Unblock(uintptr_t trampoline, const u32* set) {
      mov(edi, 2);  // SIG_UNBLOCK
      mov(rsi, reinterpret_cast<uintptr_t>(set));
      xor_(edx, edx);
      mov(rax, trampoline);
      call(rax);
      ret();
    }
  } unblock(kern::Lv2Trampoline(
                reinterpret_cast<const void*>(&kern::sys_sigprocmask), 340),
            set);
  unblock.getCode<u64 (*)()>()();
  EXPECT_EQ(g_hits, 1u);
}

TEST_F(Signals, InterruptsRunningGuestCode) {
  Install(kSigusr2, handler);
  Spin spin;
  guest::RegisterCode(reinterpret_cast<uintptr_t>(spin.getCode()),
                      spin.getSize());
  base::Atomic<bool> started{false};
  base::Thread worker(
      "spin",
      [&] {
        kern::SignalThreadStart(2);
        started = true;
        spin.getCode<void (*)()>()();
        kern::SignalThreadExit();
      },
      true);
  while (!started)
    base::SleepForMilliseconds(1);
  base::SleepForMilliseconds(20);  // let it reach the loop
  ASSERT_EQ(kern::sys_thr_kill(2, kSigusr2), 0);
  worker.Join();
  EXPECT_EQ(g_hits, 1u);
  EXPECT_EQ(g_seen_sig, static_cast<u64>(kSigusr2));
}

TEST_F(Signals, SigwaitTakesABlockedSignal) {
  u32 set[4] = {1u << (kSigusr1 - 1), 0, 0, 0};
  ASSERT_EQ(kern::sys_sigprocmask(1, set, nullptr), 0);
  ASSERT_EQ(kern::sys_kill(kern::sys_getpid(), kSigusr1), 0);
  int sig = 0;
  EXPECT_EQ(kern::sys_sigwait(set, &sig), 0);
  EXPECT_EQ(sig, kSigusr1);
  u32 pending[4] = {};
  ASSERT_EQ(kern::sys_sigpending(pending), 0);
  EXPECT_EQ(pending[0], 0u);
}
}  // namespace

namespace {
TEST_F(Signals, KtimerFiresIntoSigwait) {
  u32 set[4] = {1u << (kSigusr1 - 1), 0, 0, 0};
  ASSERT_EQ(kern::sys_sigprocmask(1, set, nullptr), 0);
  struct {
    int notify, signo;
    u64 value;
    int tid;
  } ev{1, kSigusr1, 77, 0};
  int id = -1;
  ASSERT_EQ(kern::sys_ktimer_create(4, &ev, &id), 0);
  EXPECT_GE(id, 3);
  kern::sce_itimerspec spec{{0, 0}, {0, 5'000'000}};
  ASSERT_EQ(kern::sys_ktimer_settime(id, 0, &spec, nullptr), 0);
  u8 info[0x50] = {};
  EXPECT_EQ(kern::sys_sigwaitinfo(set, info), kSigusr1);
  u64 value = 0;
  std::memcpy(&value, info + 0x20, 8);
  EXPECT_EQ(value, 77u);
  kern::sce_itimerspec left{};
  ASSERT_EQ(kern::sys_ktimer_gettime(id, &left), 0);
  EXPECT_EQ(left.it_value.tv_nsec, 0);  // one-shot: disarmed
  EXPECT_EQ(kern::sys_ktimer_delete(id), 0);
  EXPECT_EQ(kern::sys_ktimer_delete(id), -22);
}

TEST_F(Signals, ItimerRealRaisesSigalrm) {
  constexpr int kSigalrm = 14;
  u32 set[4] = {1u << (kSigalrm - 1), 0, 0, 0};
  ASSERT_EQ(kern::sys_sigprocmask(1, set, nullptr), 0);
  kern::sce_itimerval it{{0, 0}, {0, 10'000}};
  ASSERT_EQ(kern::sys_setitimer(0, &it, nullptr), 0);
  int sig = 0;
  EXPECT_EQ(kern::sys_sigwait(set, &sig), 0);
  EXPECT_EQ(sig, kSigalrm);
}
}  // namespace

namespace {
volatile u64 g_spin_stop = 0;

// Spins with r13 = 0xc0ffee until g_spin_stop is set.
struct MarkedSpin : Xbyak::CodeGenerator {
  MarkedSpin() {
    push(r13);
    mov(r13, 0xc0ffee);
    mov(rax, reinterpret_cast<uintptr_t>(&g_spin_stop));
    L("loop");
    pause();
    cmp(qword[rax], 0);
    je("loop");
    pop(r13);
    ret();
  }
};

TEST_F(Signals, SuspendReadsRegistersAndResumes) {
  MarkedSpin spin;
  guest::RegisterCode(reinterpret_cast<uintptr_t>(spin.getCode()),
                      spin.getSize());
  g_spin_stop = 0;
  base::Atomic<bool> started{false};
  base::Thread worker(
      "spin",
      [&] {
        kern::SignalThreadStart(3);
        started = true;
        spin.getCode<void (*)()>()();
        kern::SignalThreadExit();
      },
      true);
  while (!started)
    base::SleepForMilliseconds(1);
  base::SleepForMilliseconds(20);
  EXPECT_EQ(kern::sys_thr_suspend_ucontext(1), -11);  // itself: EDEADLK
  ASSERT_EQ(kern::sys_thr_suspend_ucontext(3), 0);
  alignas(64) u8 uc[0x500] = {};
  ASSERT_EQ(kern::sys_thr_get_ucontext(3, uc), 0);
  u64 r13 = 0;
  std::memcpy(&r13, uc + 0x40 + 0x68, 8);
  EXPECT_EQ(r13, 0xc0ffeeu);
  g_spin_stop = 1;  // seen only once it runs again
  ASSERT_EQ(kern::sys_thr_resume_ucontext(3), 0);
  worker.Join();
  EXPECT_EQ(kern::sys_thr_get_ucontext(3, uc), -3);  // gone: ESRCH
}

TEST_F(Signals, ThrWakeBeforeSuspendIsKept) {
  ASSERT_EQ(kern::sys_thr_wake(1), 0);
  EXPECT_EQ(kern::sys_thr_suspend(nullptr), 0);
  kern::sce_timespec ts{0, 2'000'000};
  EXPECT_EQ(kern::sys_thr_suspend(&ts), -60);  // ETIMEDOUT
}
}  // namespace
