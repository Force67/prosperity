#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Guest signals, as FreeBSD delivers them: per-process actions, per-thread
 * masks and pending sets, delivery into a FreeBSD sigframe on the guest stack,
 * sigreturn through the context trampoline. A signal reaches a thread at its
 * next syscall exit, or at once (through a host signal) while it runs guest
 * code.
 */

#include "base/arch.h"
#include "guest_abi.h"
#include "kern/lv2/dispatch.h"

namespace kern {

inline constexpr int kGuestNsig = 128;
// si_code values.
inline constexpr int kSiUser = 0x10001;
inline constexpr int kSiQueue = 0x10002;
inline constexpr int kSiTimer = 0x10003;
inline constexpr int kSiKernel = 0x10006;
inline constexpr int kSiLwp = 0x10007;

struct GuestSigset {
  u32 bits[4];
};

// What a posted signal carries into the handler's siginfo.
struct SignalInfo {
  int signo = 0;
  int code = 0;
  int pid = 0;
  u64 addr = 0;
  u64 value = 0;
  int timerid = 0;
  int overrun = 0;
};

// The calling host thread now runs (or stops running) guest thread `tid`.
void SignalThreadStart(u32 tid);
void SignalThreadExit();

// Posts to one thread (`tid`) or to the process (`tid` 0). EINVAL for a bad
// number, ESRCH for a thread we don't run.
int PostSignal(u32 tid, const SignalInfo& info);

// Whether `sig` still waits for delivery to `tid` (0: the process).
bool IsSignalPending(u32 tid, int sig);

// The calling thread's mask, as getcontext saves and setcontext restores it.
GuestSigset CurrentSigmask();
void SetCurrentSigmask(const GuestSigset& mask);

// True when the calling thread has a signal it would take now: a blocking
// syscall returns EINTR then.
bool SignalInterrupts();

// pselect and ppoll: `mask` (a guest sigset, or null for none) replaces the
// caller's for the wait, as sigsuspend's does. A signal it admits runs its
// handler under it and the old mask comes back at sigreturn; otherwise
// EndWaitMask restores it.
int BeginWaitMask(const void* mask);
void EndWaitMask(const void* mask);

// Rewrites a frame the context trampoline captured so the thread resumes in
// its signal handler, when one is due. Returns whether it did.
bool DeliverSignalToFrame(GuestFrame* frame, u8* xsave);

// The context-trampoline handler run at a syscall exit while g_signal_due is
// set.
void PS4ABI SignalAtSyscallExit(u32 sid, GuestFrame* frame, u8* xsave);

// From the host fault handler: hands a fault in guest code to the guest's
// handler for it. Returns false when the guest has none (the fault is ours).
bool DeliverGuestFault(int host_signal,
                       void* host_siginfo,
                       void* host_ucontext);

// Nonzero while some thread has a signal it would take; the syscall
// trampolines test it on the way out.
extern "C" u32 g_signal_due;

// Ends the process as the kernel's sigexit does: no handler can catch it.
void TerminateBySignal(int sig);

// The kernel's nosys: posts SIGSYS to the caller and fails with ENOSYS.
int PS4ABI sys_nosys();

int PS4ABI sys_sigaction(int sig, const void* act, void* oact);
int PS4ABI sys_sigprocmask(int how, const void* set, void* oset);
int PS4ABI sys_sigpending(void* set);
int PS4ABI sys_sigsuspend(const void* mask);
int PS4ABI sys_sigaltstack(const void* ss, void* oss);
int PS4ABI sys_sigwait(const void* set, int* sig);
int PS4ABI sys_sigwaitinfo(const void* set, void* info);
int PS4ABI sys_sigtimedwait(const void* set, void* info, const void* timeout);
int PS4ABI sys_kill(int pid, int sig);
int PS4ABI sys_sigqueue(int pid, int sig, u64 value);
int PS4ABI sys_thr_kill(i64 tid, int sig);
int PS4ABI sys_thr_kill2(int pid, i64 tid, int sig);
int PS4ABI sys_thr_suspend(const void* timeout);
int PS4ABI sys_thr_wake(i64 tid);
int PS4ABI sys_thr_suspend_ucontext(i64 tid);
int PS4ABI sys_thr_resume_ucontext(i64 tid);
int PS4ABI sys_thr_get_ucontext(i64 tid, void* ucontext);
int PS4ABI sys_thr_set_ucontext(i64 tid, const void* ucontext);

}  // namespace kern
