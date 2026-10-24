/*
 * PS4Delta : PS4 emulation and research project
 *
 * Guest signals. See sys_signal.h.
 */

#include "kern/lv2/sys_signal.h"

#include <signal.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>
#include <cstring>
#if defined(DELTA_BACKEND_NATIVE)
#include <xbyak.h>
#endif

#include "base/arch.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/time/time.h"
#include "cpu/backend.h"
#include "guest/pause.h"
#include "guest/session.h"
#include "guest_abi.h"
#include "host_memory/host_memory.h"

#include "kern/crash.h"
#include "kern/lv2/error_table.h"
#include "kern/lv2/sys_context.h"
#include "kern/lv2/sys_procid.h"
#include "kern/lv2/sys_thread.h"
#include "kern/lv2/sys_time.h"

extern "C" u32 g_signal_due = 0;

namespace kern {
namespace {
constexpr int kSigIll = 4, kSigTrap = 5, kSigFpe = 8, kSigKill = 9,
              kSigBus = 10, kSigSegv = 11, kSigStop = 17, kSigCont = 19;
constexpr u32 kSaOnstack = 0x1, kSaResethand = 0x4, kSaNodefer = 0x10,
              kSaSiginfo = 0x40;
constexpr u64 kSigDfl = 0, kSigIgn = 1;
constexpr int kSsOnstack = 1, kSsDisable = 4;
constexpr size_t kMinSigStackSize = 2048;
constexpr int kSigBlock = 1, kSigUnblock = 2, kSigSetmask = 3;

constexpr u64 kRflagsTrap = 0x100, kRflagsDirection = 0x400;
constexpr u64 kRedZone = 128;

// The sigframe: the handler (sigcode calls through it), the ucontext, then
// the siginfo.
constexpr size_t kFrameUc = 0x40;
constexpr size_t kFrameSi = kFrameUc + 0x500;
constexpr size_t kFrameSize = kFrameSi + 0x50;

enum Default { kTerminate, kIgnore, kStop, kContinue };

Default DefaultAction(int sig) {
  switch (sig) {
    case 16:  // SIGURG
    case 20:  // SIGCHLD
    case 23:  // SIGIO
    case 28:  // SIGWINCH
    case 29:  // SIGINFO
      return kIgnore;
    case kSigCont:
      return kContinue;
    case kSigStop:
    case 18:  // SIGTSTP
    case 21:  // SIGTTIN
    case 22:  // SIGTTOU
      return kStop;
    default:
      return kTerminate;
  }
}

struct Action {
  u64 handler = kSigDfl;
  u32 flags = 0;
  GuestSigset mask{};
};
static_assert(sizeof(Action) == 32);  // struct sigaction

struct Thread {
  u32 tid = 0;
  long host_tid = 0;
  GuestSigset mask{};
  GuestSigset pending{};
  base::Vector<SignalInfo> queue;  // siginfo of the pending thread signals
  u64 ss_sp = 0;
  u64 ss_size = 0;
  int ss_flags = kSsDisable;
  // sigsuspend's mask is the one the handler runs under; the frame records
  // the caller's.
  bool suspended = false;
  GuestSigset saved_mask{};
  const GuestSigset* waiting = nullptr;  // sigwait's set while it waits
  // thr_suspend_ucontext: asked to stop, stopped with its registers saved,
  // and whether thr_set_ucontext replaced them.
  bool stop_req = false;
  bool stopped = false;
  bool snap_dirty = false;
  GuestFrame snap{};
  alignas(64) u8 snap_xsave[0x340]{};
  bool wake_pending = false;  // thr_wake before thr_suspend
  base::ConditionVariable cv;
};

base::Mutex g_mutex;
Action g_actions[kGuestNsig];
GuestSigset g_process_pending{};
base::Vector<SignalInfo> g_process_queue;
base::Vector<Thread*> g_threads;
thread_local Thread* t_thread = nullptr;
int g_host_signal = 0;

bool Valid(int sig) {
  return sig >= 1 && sig <= kGuestNsig;
}
bool Has(const GuestSigset& s, int sig) {
  return (s.bits[(sig - 1) / 32] >> ((sig - 1) % 32)) & 1;
}
void Add(GuestSigset& s, int sig) {
  s.bits[(sig - 1) / 32] |= 1u << ((sig - 1) % 32);
}
void Del(GuestSigset& s, int sig) {
  s.bits[(sig - 1) / 32] &= ~(1u << ((sig - 1) % 32));
}
GuestSigset Maskable(GuestSigset s) {
  Del(s, kSigKill);
  Del(s, kSigStop);
  return s;
}
bool IsRealtime(int sig) {
  return sig >= 65 && sig <= 126;
}

bool Ignored(int sig) {
  const Action& a = g_actions[sig - 1];
  if (a.handler == kSigIgn)
    return true;
  return a.handler == kSigDfl &&
         (DefaultAction(sig) == kIgnore || DefaultAction(sig) == kContinue);
}

// A pending signal `t` would act on now: unmasked and not ignored.
int NextDeliverable(const Thread& t) {
  for (int sig = 1; sig <= kGuestNsig; sig++) {
    if (Has(t.mask, sig) || Ignored(sig))
      continue;
    if (Has(t.pending, sig) || Has(g_process_pending, sig))
      return sig;
  }
  return 0;
}

void RecomputeDue() {
  u32 due = 0;
  for (const Thread* t : g_threads)
    if (t->stop_req || NextDeliverable(*t)) {
      due = 1;
      break;
    }
  __atomic_store_n(&g_signal_due, due, __ATOMIC_RELEASE);
}

Thread* FindThread(u32 tid) {
  for (Thread* t : g_threads)
    if (t->tid == tid)
      return t;
  return nullptr;
}

void HostSignalHandler(int, siginfo_t*, void* context);

// Only the native backend interrupts running guest code; elsewhere a signal
// waits for the next syscall exit.
void InstallHostSignal() {
#if defined(DELTA_BACKEND_NATIVE)
  static const bool installed = [] {
    g_host_signal = SIGRTMIN + 6;
    struct sigaction sa{};
    sa.sa_sigaction = HostSignalHandler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    return ::sigaction(g_host_signal, &sa, nullptr) == 0;
  }();
  (void)installed;
#endif
}

// Under g_mutex.
Thread* Register(u32 tid) {
  auto* t = new Thread;
  t->tid = tid;
  t->host_tid = static_cast<long>(::syscall(SYS_gettid));
  g_threads.push_back(t);
  t_thread = t;
  return t;
}

Thread* CurrentLocked() {
  return t_thread ? t_thread : Register(*CurrentGuestTidPtr());
}

void Kick(Thread* t) {
  t->cv.NotifyAll();
  if (t != t_thread && t->host_tid && g_host_signal)
    ::syscall(SYS_tgkill, ::getpid(), t->host_tid, g_host_signal);
}

// The default action that ends the process. Does not return on a guest
// thread.
void Terminate(int sig) {
  BASE_LOGI("signal", "guest terminated by signal {}", sig);
  guest::RequestStop();
  cpu::ExitGuestThread();
}

SignalInfo TakeQueued(base::Vector<SignalInfo>& q,
                      GuestSigset& pending,
                      int sig) {
  SignalInfo info{};
  info.signo = sig;
  info.code = kSiUser;
  for (size_t i = 0; i < q.size(); i++)
    if (q[i].signo == sig) {
      info = q[i];
      q.erase(q.begin() + i);
      break;
    }
  // Another queued instance of a realtime signal keeps it pending.
  bool more = false;
  for (const auto& queued : q)
    more |= queued.signo == sig;
  if (!more)
    Del(pending, sig);
  return info;
}

// Removes `sig` from what is pending for `t` and returns its siginfo.
SignalInfo Dequeue(Thread& t, int sig) {
  if (Has(t.pending, sig))
    return TakeQueued(t.queue, t.pending, sig);
  return TakeQueued(g_process_queue, g_process_pending, sig);
}

void DiscardPending(int sig) {
  Del(g_process_pending, sig);
  auto drop = [sig](base::Vector<SignalInfo>& q) {
    for (size_t i = 0; i < q.size();)
      if (q[i].signo == sig)
        q.erase(q.begin() + i);
      else
        i++;
  };
  drop(g_process_queue);
  for (Thread* t : g_threads) {
    Del(t->pending, sig);
    drop(t->queue);
  }
}

// Picks the signal `t` takes now and applies its disposition; the default
// actions that end or stop the process happen here. Under g_mutex; returns
// false (and leaves the lock held) when nothing calls for a handler.
bool TakeForHandler(Thread& t,
                    base::UniqueLock<base::Mutex>& lk,
                    bool may_terminate,
                    SignalInfo* info,
                    Action* action,
                    GuestSigset* old_mask) {
  for (;;) {
    const int sig = NextDeliverable(t);
    if (!sig)
      return false;
    const Action a = g_actions[sig - 1];
    if (a.handler == kSigDfl) {
      const Default d = DefaultAction(sig);
      if (d == kTerminate) {
        if (!may_terminate)
          return false;
        Dequeue(t, sig);
        lk.unlock();
        Terminate(sig);
        lk.lock();
        return false;
      }
      Dequeue(t, sig);
      if (d == kStop)
        guest::SetPaused(true);
      continue;
    }
    *info = Dequeue(t, sig);
    *action = a;
    *old_mask = t.suspended ? t.saved_mask : t.mask;
    t.suspended = false;
    for (int i = 0; i < 4; i++)
      t.mask.bits[i] |= a.mask.bits[i];
    if (!(a.flags & kSaNodefer))
      Add(t.mask, sig);
    t.mask = Maskable(t.mask);
    if (a.flags & kSaResethand)
      g_actions[sig - 1] = Action{};
    RecomputeDue();
    return true;
  }
}

bool OnAltStack(const Thread& t, u64 sp) {
  return !(t.ss_flags & kSsDisable) && sp >= t.ss_sp &&
         sp < t.ss_sp + t.ss_size;
}

uintptr_t SigcodeAddress();

// Writes the sigframe for `info` below the state in `f`/`xsave` and points
// `f` at the handler. Returns false when the frame does not fit in guest
// memory.
bool BuildFrame(Thread& t,
                const SignalInfo& info,
                const Action& a,
                const GuestSigset& old_mask,
                GuestFrame* f,
                const u8* xsave) {
  const bool was_on_alt = OnAltStack(t, f->rsp);
  u64 sp = (a.flags & kSaOnstack) && !(t.ss_flags & kSsDisable) && !was_on_alt
               ? t.ss_sp + t.ss_size
               : f->rsp - kRedZone;
  sp = (sp - kFrameSize) & ~u64{63};
  auto* frame = reinterpret_cast<u8*>(sp);
  if (!host_memory::IsMemoryRangeMapped(frame, kFrameSize))
    return false;

  std::memcpy(frame, &a.handler, sizeof(u64));
  u8* uc = frame + kFrameUc;
  FillUcontext(*f, xsave, old_mask, false, uc);
  // uc_stack: the alternate stack and whether the thread was on it.
  const u64 stack[3] = {
      t.ss_sp, t.ss_size,
      static_cast<u64>(t.ss_flags | (was_on_alt ? kSsOnstack : 0))};
  std::memcpy(uc + 0x4c8, stack, sizeof(stack));

  u8* si = frame + kFrameSi;
  std::memset(si, 0, 0x50);
  const i32 head[6] = {info.signo, 0, info.code, info.pid, 0, 0};
  std::memcpy(si, head, sizeof(head));
  std::memcpy(si + 0x18, &info.addr, sizeof(u64));
  std::memcpy(si + 0x20, &info.value, sizeof(u64));
  const i32 timer[2] = {info.timerid, info.overrun};  // _reason._timer
  std::memcpy(si + 0x28, timer, sizeof(timer));

  f->rdi = static_cast<u64>(info.signo);
  f->rsi = a.flags & kSaSiginfo ? reinterpret_cast<u64>(si)
                                : static_cast<u64>(info.code);
  f->rdx = reinterpret_cast<u64>(uc);
  f->rcx = info.addr;
  f->rip = SigcodeAddress();
  f->rsp = sp;
  f->rflags &= ~(kRflagsTrap | kRflagsDirection);
  return true;
}

// The handler starts with a clean FPU, as the kernel's fpstate_drop leaves it.
void ResetXsave(u8* xsave) {
  std::memset(xsave, 0, 0x200 + 0x40);
  const u16 fcw = 0x37f;
  const u32 mxcsr = 0x1f80;
  std::memcpy(xsave, &fcw, sizeof(fcw));
  std::memcpy(xsave + 0x18, &mxcsr, sizeof(mxcsr));
}

// Delivers whatever `t` has due into `f`. `may_terminate` is false inside the
// host signal handler, which must not unwind the thread.
bool Deliver(Thread& t,
             GuestFrame* f,
             u8* xsave,
             bool may_terminate,
             bool* frame_failed) {
  base::UniqueLock<base::Mutex> lk(g_mutex);
  SignalInfo info;
  Action action;
  GuestSigset old_mask;
  if (!TakeForHandler(t, lk, may_terminate, &info, &action, &old_mask))
    return false;
  if (!BuildFrame(t, info, action, old_mask, f, xsave)) {
    *frame_failed = true;
    return false;
  }
  return true;
}

constexpr size_t kXsaveSize = 0x340;

#if defined(DELTA_BACKEND_NATIVE)
constexpr u32 kFpXstateMagic1 = 0x46505853;

GuestFrame FrameFromHost(const ucontext_t& uc) {
  const greg_t* g = uc.uc_mcontext.gregs;
  GuestFrame f{};
  f.r15 = g[REG_R15], f.r14 = g[REG_R14], f.r13 = g[REG_R13];
  f.r12 = g[REG_R12], f.r11 = g[REG_R11], f.r10 = g[REG_R10];
  f.r9 = g[REG_R9], f.r8 = g[REG_R8], f.rdi = g[REG_RDI], f.rsi = g[REG_RSI];
  f.rbp = g[REG_RBP], f.rdx = g[REG_RDX], f.rcx = g[REG_RCX];
  f.rbx = g[REG_RBX], f.rax = g[REG_RAX], f.rflags = g[REG_EFL];
  f.rsp = g[REG_RSP], f.rip = g[REG_RIP];
  return f;
}

// The host frame's FPU state as an XSAVE image (legacy area alone when the
// kernel saved no extended state).
void XsaveFromHost(const ucontext_t& uc, u8* out) {
  std::memset(out, 0, kXsaveSize);
  const auto* fp = reinterpret_cast<const u8*>(uc.uc_mcontext.fpregs);
  if (!fp)
    return;
  u32 magic, size;
  std::memcpy(&magic, fp + 464, sizeof(magic));
  std::memcpy(&size, fp + 468, sizeof(size));
  if (magic == kFpXstateMagic1 && size >= kXsaveSize) {
    std::memcpy(out, fp, kXsaveSize);
  } else {
    std::memcpy(out, fp, 512);
    const u64 bv = 3;  // x87 | SSE
    std::memcpy(out + 0x200, &bv, sizeof(bv));
  }
  std::memset(out + 464, 0, 48);  // the host's software-reserved bytes
}

void ApplyToHost(const GuestFrame& f, ucontext_t* uc) {
  greg_t* g = uc->uc_mcontext.gregs;
  g[REG_RDI] = f.rdi, g[REG_RSI] = f.rsi, g[REG_RDX] = f.rdx;
  g[REG_RCX] = f.rcx, g[REG_RIP] = f.rip, g[REG_RSP] = f.rsp;
  g[REG_EFL] = f.rflags;
  // The handler's clean FPU.
  if (auto* fp = uc->uc_mcontext.fpregs) {
    fp->cwd = 0x37f;
    fp->swd = 0;
    fp->ftw = 0;
    fp->mxcsr = 0x1f80;
    std::memset(fp->_st, 0, sizeof(fp->_st));
    std::memset(fp->_xmm, 0, sizeof(fp->_xmm));
    auto* raw = reinterpret_cast<u8*>(fp);
    u32 magic;
    std::memcpy(&magic, raw + 464, sizeof(magic));
    if (magic == kFpXstateMagic1) {
      u64 bv;
      std::memcpy(&bv, raw + 512, sizeof(bv));
      bv &= ~u64{4};  // AVX upper halves to their init state
      std::memcpy(raw + 512, &bv, sizeof(bv));
    }
  }
}

#endif  // DELTA_BACKEND_NATIVE

// Stops `t` with the state in `f`/`xsave` until thr_resume_ucontext, then
// hands back what thr_set_ucontext may have put there. Returns whether it did.
bool Park(Thread& t, GuestFrame* f, u8* xsave) {
  base::UniqueLock<base::Mutex> lk(g_mutex);
  if (!t.stop_req)
    return false;
  t.snap = *f;
  std::memcpy(t.snap_xsave, xsave, kXsaveSize);
  t.snap_dirty = false;
  t.stopped = true;
  t.cv.NotifyAll();
  while (t.stop_req && !guest::Stopping())
    t.cv.WaitFor(lk, base::Milliseconds(50));
  t.stopped = false;
  if (!t.snap_dirty)
    return false;
  *f = t.snap;
  std::memcpy(xsave, t.snap_xsave, kXsaveSize);
  return true;
}

#if defined(DELTA_BACKEND_NATIVE)
// Every register and the FPU state from `f`/`xsave` into the host frame.
void ApplyAllToHost(const GuestFrame& f, const u8* xsave, ucontext_t* uc) {
  greg_t* g = uc->uc_mcontext.gregs;
  g[REG_R15] = f.r15, g[REG_R14] = f.r14, g[REG_R13] = f.r13;
  g[REG_R12] = f.r12, g[REG_R11] = f.r11, g[REG_R10] = f.r10;
  g[REG_R9] = f.r9, g[REG_R8] = f.r8, g[REG_RDI] = f.rdi, g[REG_RSI] = f.rsi;
  g[REG_RBP] = f.rbp, g[REG_RDX] = f.rdx, g[REG_RCX] = f.rcx;
  g[REG_RBX] = f.rbx, g[REG_RAX] = f.rax, g[REG_EFL] = f.rflags;
  g[REG_RSP] = f.rsp, g[REG_RIP] = f.rip;
  if (auto* fp = reinterpret_cast<u8*>(uc->uc_mcontext.fpregs)) {
    u32 magic;
    std::memcpy(&magic, fp + 464, sizeof(magic));
    u8 sw[48];
    std::memcpy(sw, fp + 464, sizeof(sw));
    std::memcpy(fp, xsave, magic == kFpXstateMagic1 ? kXsaveSize : 512);
    std::memcpy(fp + 464, sw, sizeof(sw));  // the host's frame description
  }
}

// A guest thread interrupted in guest code stops or takes its signal at
// once; one in host code does so at its next syscall exit.
void HostSignalHandler(int, siginfo_t*, void* context) {
  Thread* t = t_thread;
  auto* uc = static_cast<ucontext_t*>(context);
  if (!t || !guest::IsGuestCode(uc->uc_mcontext.gregs[REG_RIP]))
    return;
  const int saved_errno = errno;
  GuestFrame f = FrameFromHost(*uc);
  alignas(64) u8 xsave[kXsaveSize];
  XsaveFromHost(*uc, xsave);
  if (Park(*t, &f, xsave))
    ApplyAllToHost(f, xsave, uc);
  bool frame_failed = false;
  if (Deliver(*t, &f, xsave, false, &frame_failed))
    ApplyToHost(f, uc);
  errno = saved_errno;
}
#else
void HostSignalHandler(int, siginfo_t*, void*) {}
#endif  // DELTA_BACKEND_NATIVE

#if defined(DELTA_BACKEND_NATIVE)
// sigcode: calls the handler the frame names, then sigreturns to the context
// the frame saved.
struct Sigcode : Xbyak::CodeGenerator {
  Sigcode() {
    call(ptr[rsp]);
    lea(rdi, ptr[rsp + kFrameUc]);
    mov(rax, Lv2ContextTrampoline(&sys_context, 417));
    call(rax);
    ud2();
  }
};

uintptr_t SigcodeAddress() {
  static Sigcode* code = [] {
    auto* c = new Sigcode();
    guest::RegisterCode(reinterpret_cast<uintptr_t>(c->getCode()),
                        c->getSize());
    return c;
  }();
  return reinterpret_cast<uintptr_t>(code->getCode());
}
#else
uintptr_t SigcodeAddress() {
  return 0;
}
#endif

int ToGuestSignal(int host) {
  switch (host) {
    case 4:
      return kSigIll;
    case 5:
      return kSigTrap;
    case 7:
      return kSigBus;
    case 8:
      return kSigFpe;
    case 11:
      return kSigSegv;
    default:
      return 0;
  }
}

// Waits on the calling thread's condition for `ready`, giving up after
// `timeout` (negative: never) or when the session stops. Under g_mutex.
template <typename Ready>
bool WaitSignal(Thread& t,
                base::UniqueLock<base::Mutex>& lk,
                i64 timeout_us,
                Ready ready) {
  const auto deadline = base::TimeTicks::Now() +
                        base::Microseconds(timeout_us < 0 ? 0 : timeout_us);
  while (!ready()) {
    if (guest::Stopping())
      return false;
    auto slice = base::Milliseconds(50);
    if (timeout_us >= 0) {
      const auto left = deadline - base::TimeTicks::Now();
      if (left <= base::TimeDelta())
        return false;
      if (left < slice)
        slice = left;
    }
    t.cv.WaitFor(lk, slice);
  }
  return true;
}

// sigwaitinfo/sigtimedwait: the lowest signal of `set` pending for `t`.
int WaitForSet(const GuestSigset& set, SignalInfo* out, i64 timeout_us) {
  base::UniqueLock<base::Mutex> lk(g_mutex);
  Thread& t = *CurrentLocked();
  const GuestSigset wanted = Maskable(set);
  int got = 0;
  auto ready = [&] {
    for (int sig = 1; sig <= kGuestNsig; sig++)
      if (Has(wanted, sig) &&
          (Has(t.pending, sig) || Has(g_process_pending, sig))) {
        got = sig;
        return true;
      }
    // A signal it would handle instead interrupts the wait.
    return NextDeliverable(t) != 0;
  };
  t.waiting = &wanted;
  const bool woke = WaitSignal(t, lk, timeout_us, ready);
  t.waiting = nullptr;
  if (!woke)
    return guest::Stopping() ? -SysError::eINTR : -SysError::eAGAIN;
  if (!got)
    return -SysError::eINTR;
  *out = Dequeue(t, got);
  RecomputeDue();
  return got;
}

int ReadSet(const void* p, GuestSigset* out) {
  if (!host_memory::IsMemoryRangeMapped(p, sizeof(GuestSigset)))
    return -SysError::eFAULT;
  std::memcpy(out, p, sizeof(GuestSigset));
  return 0;
}

void WriteInfo(void* p, const SignalInfo& info) {
  auto* si = static_cast<u8*>(p);
  std::memset(si, 0, 0x50);
  const i32 head[4] = {info.signo, 0, info.code, info.pid};
  std::memcpy(si, head, sizeof(head));
  std::memcpy(si + 0x18, &info.addr, sizeof(u64));
  std::memcpy(si + 0x20, &info.value, sizeof(u64));
}

bool IsOurPid(int pid) {
  const int self = sys_getpid();
  return pid == 0 || pid == -1 || pid == self || pid == -self;
}

const guest::SessionReset g_reset([] {
  base::LockGuard<base::Mutex> lk(g_mutex);
  for (auto& a : g_actions)
    a = Action{};
  g_process_pending = {};
  guest::ResetResource(g_process_queue);
  for (Thread* t : g_threads)
    delete t;
  guest::ResetResource(g_threads);
  t_thread = nullptr;
  __atomic_store_n(&g_signal_due, 0, __ATOMIC_RELEASE);
});
}  // namespace

void SignalThreadStart(u32 tid) {
  InstallHostSignal();
  sigset_t host;
  sigemptyset(&host);
  sigaddset(&host, g_host_signal);
  ::pthread_sigmask(SIG_UNBLOCK, &host, nullptr);
  base::LockGuard<base::Mutex> lk(g_mutex);
  if (!t_thread)
    Register(tid);
}

void SignalThreadExit() {
  base::LockGuard<base::Mutex> lk(g_mutex);
  Thread* t = t_thread;
  if (!t)
    return;
  for (size_t i = 0; i < g_threads.size(); i++)
    if (g_threads[i] == t) {
      g_threads.erase(g_threads.begin() + i);
      break;
    }
  t_thread = nullptr;
  delete t;
  RecomputeDue();
}

int PostSignal(u32 tid, const SignalInfo& in) {
  const int sig = in.signo;
  if (sig && !Valid(sig))
    return -SysError::eINVAL;
  base::UniqueLock<base::Mutex> lk(g_mutex);
  Thread* target = tid ? FindThread(tid) : nullptr;
  if (tid && !target && tid != *CurrentGuestTidPtr())
    return -SysError::eSRCH;
  if (tid && !target)
    target = CurrentLocked();
  if (!sig)
    return 0;
  if (sig == kSigKill) {
    lk.unlock();
    Terminate(sig);
    return 0;
  }
  if (sig == kSigCont)
    guest::SetPaused(false);

  SignalInfo info = in;
  if (!info.pid)
    info.pid = sys_getpid();
  if (!target) {
    // Process-directed: a thread waiting for it, else one not blocking it.
    for (Thread* t : g_threads)
      if (t->waiting && Has(*t->waiting, sig)) {
        target = t;
        break;
      }
    if (!target)
      for (Thread* t : g_threads)
        if (!Has(t->mask, sig)) {
          target = t;
          break;
        }
  }
  const bool waited = target && target->waiting && Has(*target->waiting, sig);
  if (Ignored(sig) && !waited)
    return 0;  // discarded, as the kernel does

  GuestSigset& pending = tid ? target->pending : g_process_pending;
  base::Vector<SignalInfo>& queue = tid ? target->queue : g_process_queue;
  if (!Has(pending, sig) || IsRealtime(sig))
    queue.push_back(info);
  Add(pending, sig);
  RecomputeDue();
  if (target)
    Kick(target);
  else
    for (Thread* t : g_threads)
      t->cv.NotifyAll();
  return 0;
}

bool IsSignalPending(u32 tid, int sig) {
  if (!Valid(sig))
    return false;
  base::LockGuard<base::Mutex> lk(g_mutex);
  if (!tid)
    return Has(g_process_pending, sig);
  const Thread* t = FindThread(tid);
  return t && Has(t->pending, sig);
}

GuestSigset CurrentSigmask() {
  base::LockGuard<base::Mutex> lk(g_mutex);
  return CurrentLocked()->mask;
}

void SetCurrentSigmask(const GuestSigset& mask) {
  base::LockGuard<base::Mutex> lk(g_mutex);
  CurrentLocked()->mask = Maskable(mask);
  RecomputeDue();
}

bool SignalInterrupts() {
  if (!__atomic_load_n(&g_signal_due, __ATOMIC_ACQUIRE))
    return false;
  base::LockGuard<base::Mutex> lk(g_mutex);
  return NextDeliverable(*CurrentLocked()) != 0;
}

bool DeliverSignalToFrame(GuestFrame* f, u8* xsave) {
  if (!__atomic_load_n(&g_signal_due, __ATOMIC_ACQUIRE))
    return false;
  Thread* t;
  {
    base::LockGuard<base::Mutex> lk(g_mutex);
    t = CurrentLocked();
  }
  bool frame_failed = false;
  if (!Deliver(*t, f, xsave, true, &frame_failed)) {
    if (frame_failed)
      Terminate(kSigIll);  // the kernel's sigexit when the frame won't fit
    return false;
  }
  ResetXsave(xsave);
  return true;
}

void PS4ABI SignalAtSyscallExit(u32, GuestFrame* f, u8* xsave) {
  f->rsp += 8;  // past the lifted call's return address
  Thread* t;
  {
    base::LockGuard<base::Mutex> lk(g_mutex);
    t = CurrentLocked();
  }
  Park(*t, f, xsave);
  DeliverSignalToFrame(f, xsave);
}

bool DeliverGuestFault(int host_signal, void* host_info, void* host_uc) {
#if !defined(DELTA_BACKEND_NATIVE)
  (void)host_signal, (void)host_info, (void)host_uc;
  return false;
#else
  Thread* t = t_thread;
  auto* uc = static_cast<ucontext_t*>(host_uc);
  const int sig = ToGuestSignal(host_signal);
  if (!t || !sig || !guest::IsGuestCode(uc->uc_mcontext.gregs[REG_RIP]))
    return false;
  base::UniqueLock<base::Mutex> lk(g_mutex);
  const Action a = g_actions[sig - 1];
  // A fault the guest blocks or leaves to the default action ends it; that
  // is ours to report.
  if (a.handler == kSigDfl || a.handler == kSigIgn || Has(t->mask, sig))
    return false;
  const auto* si = static_cast<const siginfo_t*>(host_info);
  SignalInfo info{};
  info.signo = sig;
  info.code = si->si_code;
  info.addr = reinterpret_cast<u64>(si->si_addr);
  GuestFrame f = FrameFromHost(*uc);
  alignas(64) u8 xsave[kXsaveSize];
  XsaveFromHost(*uc, xsave);
  GuestSigset old_mask = t->mask;
  for (int i = 0; i < 4; i++)
    t->mask.bits[i] |= a.mask.bits[i];
  if (!(a.flags & kSaNodefer))
    Add(t->mask, sig);
  t->mask = Maskable(t->mask);
  if (a.flags & kSaResethand)
    g_actions[sig - 1] = Action{};
  if (!BuildFrame(*t, info, a, old_mask, &f, xsave)) {
    t->mask = old_mask;
    return false;
  }
  RecomputeDue();
  BASE_LOGI("signal", "guest handles signal {} at {:#x} (addr {:#x})", sig,
            static_cast<u64>(uc->uc_mcontext.gregs[REG_RIP]), info.addr);
  ApplyToHost(f, uc);
  return true;
#endif
}

int PS4ABI sys_sigaction(int sig, const void* act, void* oact) {
  if (!Valid(sig))
    return -SysError::eINVAL;
  Action next;
  if (act) {
    if (!host_memory::IsMemoryRangeMapped(act, 28))
      return -SysError::eFAULT;
    std::memcpy(&next, act, 28);
    if ((sig == kSigKill || sig == kSigStop) && next.handler != kSigDfl)
      return -SysError::eINVAL;
  }
  base::LockGuard<base::Mutex> lk(g_mutex);
  if (oact) {
    if (!host_memory::IsMemoryRangeMapped(oact, sizeof(Action)))
      return -SysError::eFAULT;
    std::memcpy(oact, &g_actions[sig - 1], sizeof(Action));
  }
  if (act) {
    next.mask = Maskable(next.mask);
    g_actions[sig - 1] = next;
    if (Ignored(sig))
      DiscardPending(sig);
    RecomputeDue();
  }
  return 0;
}

int PS4ABI sys_sigprocmask(int how, const void* set, void* oset) {
  GuestSigset in{};
  if (set)
    if (const int err = ReadSet(set, &in))
      return err;
  base::LockGuard<base::Mutex> lk(g_mutex);
  Thread& t = *CurrentLocked();
  if (oset) {
    if (!host_memory::IsMemoryRangeMapped(oset, sizeof(GuestSigset)))
      return -SysError::eFAULT;
    std::memcpy(oset, &t.mask, sizeof(GuestSigset));
  }
  if (!set)
    return 0;
  switch (how) {
    case kSigBlock:
      for (int i = 0; i < 4; i++)
        t.mask.bits[i] |= in.bits[i];
      break;
    case kSigUnblock:
      for (int i = 0; i < 4; i++)
        t.mask.bits[i] &= ~in.bits[i];
      break;
    case kSigSetmask:
      t.mask = in;
      break;
    default:
      return -SysError::eINVAL;
  }
  t.mask = Maskable(t.mask);
  RecomputeDue();
  return 0;
}

int PS4ABI sys_sigpending(void* set) {
  if (!host_memory::IsMemoryRangeMapped(set, sizeof(GuestSigset)))
    return -SysError::eFAULT;
  base::LockGuard<base::Mutex> lk(g_mutex);
  const Thread& t = *CurrentLocked();
  GuestSigset out;
  for (int i = 0; i < 4; i++)
    out.bits[i] = t.pending.bits[i] | g_process_pending.bits[i];
  std::memcpy(set, &out, sizeof(out));
  return 0;
}

// Runs under `mask` until a signal is taken; the handler runs at this
// syscall's exit and its frame keeps the caller's mask.
int BeginWaitMask(const void* mask) {
  if (!mask)
    return 0;
  GuestSigset in;
  if (const int err = ReadSet(mask, &in))
    return err;
  base::LockGuard<base::Mutex> lk(g_mutex);
  Thread& t = *CurrentLocked();
  t.saved_mask = t.mask;
  t.suspended = true;
  t.mask = Maskable(in);
  RecomputeDue();
  return 0;
}

void EndWaitMask(const void* mask) {
  if (!mask)
    return;
  base::LockGuard<base::Mutex> lk(g_mutex);
  Thread& t = *CurrentLocked();
  if (!t.suspended || NextDeliverable(t))
    return;  // the syscall exit delivers under the wait's mask
  t.mask = t.saved_mask;
  t.suspended = false;
  RecomputeDue();
}

int PS4ABI sys_sigsuspend(const void* mask) {
  GuestSigset in;
  if (const int err = ReadSet(mask, &in))
    return err;
  base::UniqueLock<base::Mutex> lk(g_mutex);
  Thread& t = *CurrentLocked();
  t.saved_mask = t.mask;
  t.suspended = true;
  t.mask = Maskable(in);
  RecomputeDue();
  WaitSignal(t, lk, -1, [&] { return NextDeliverable(t) != 0; });
  if (!NextDeliverable(t)) {  // the session stopped instead
    t.mask = t.saved_mask;
    t.suspended = false;
    RecomputeDue();
  }
  return -SysError::eINTR;
}

int PS4ABI sys_sigaltstack(const void* ss, void* oss) {
  base::LockGuard<base::Mutex> lk(g_mutex);
  Thread& t = *CurrentLocked();
  const bool on = OnAltStack(t, GuestStackScanBase());
  if (oss) {
    if (!host_memory::IsMemoryRangeMapped(oss, 24))
      return -SysError::eFAULT;
    const u64 out[3] = {t.ss_sp, t.ss_size,
                        static_cast<u64>(t.ss_flags | (on ? kSsOnstack : 0))};
    std::memcpy(oss, out, sizeof(out));
  }
  if (!ss)
    return 0;
  if (!host_memory::IsMemoryRangeMapped(ss, 24))
    return -SysError::eFAULT;
  u64 in[3];
  std::memcpy(in, ss, sizeof(in));
  const int flags = static_cast<int>(in[2]);
  if (on)
    return -SysError::ePERM;
  if (flags & ~kSsDisable)
    return -SysError::eINVAL;
  if (!(flags & kSsDisable) && in[1] < kMinSigStackSize)
    return -SysError::eNOMEM;
  t.ss_sp = in[0];
  t.ss_size = in[1];
  t.ss_flags = flags;
  return 0;
}

// sigwait answers with an error number, not errno. A handled signal that
// arrives instead runs at this syscall's exit.
int PS4ABI sys_sigwait(const void* set, int* sig) {
  GuestSigset in;
  if (ReadSet(set, &in))
    return SysError::eFAULT;
  SignalInfo info;
  const int got = WaitForSet(in, &info, -1);
  if (got <= 0)
    return SysError::eINTR;
  if (sig)
    *sig = got;
  return 0;
}

int PS4ABI sys_sigwaitinfo(const void* set, void* info) {
  return sys_sigtimedwait(set, info, nullptr);
}

int PS4ABI sys_sigtimedwait(const void* set, void* info, const void* timeout) {
  GuestSigset in;
  if (const int err = ReadSet(set, &in))
    return err;
  i64 us = -1;
  if (timeout) {
    if (!host_memory::IsMemoryRangeMapped(timeout, sizeof(sce_timespec)))
      return -SysError::eFAULT;
    const auto* ts = static_cast<const sce_timespec*>(timeout);
    if (ts->tv_sec < 0 || ts->tv_nsec < 0 || ts->tv_nsec >= 1'000'000'000)
      return -SysError::eINVAL;
    us = ts->tv_sec * 1'000'000 + (ts->tv_nsec + 999) / 1000;
  }
  SignalInfo got;
  const int sig = WaitForSet(in, &got, us);
  if (sig > 0 && info) {
    if (!host_memory::IsMemoryRangeMapped(info, 0x50))
      return -SysError::eFAULT;
    WriteInfo(info, got);
  }
  return sig;
}

int PS4ABI sys_kill(int pid, int sig) {
  if (!IsOurPid(pid))
    return -SysError::eSRCH;
  SignalInfo info{};
  info.signo = sig;
  info.code = kSiUser;
  return PostSignal(0, info);
}

int PS4ABI sys_sigqueue(int pid, int sig, u64 value) {
  if (!IsOurPid(pid) || pid <= 0)
    return -SysError::eSRCH;
  SignalInfo info{};
  info.signo = sig;
  info.code = kSiQueue;
  info.value = value;
  return PostSignal(0, info);
}

int PS4ABI sys_thr_kill(i64 tid, int sig) {
  SignalInfo info{};
  info.signo = sig;
  info.code = kSiLwp;
  if (tid != -1)
    return tid > 0 ? PostSignal(static_cast<u32>(tid), info) : -SysError::eSRCH;
  // Every thread of the process.
  base::Vector<u32> tids;
  {
    base::LockGuard<base::Mutex> lk(g_mutex);
    for (const Thread* t : g_threads)
      tids.push_back(t->tid);
  }
  for (u32 t : tids)
    if (const int err = PostSignal(t, info))
      return err;
  return 0;
}

int PS4ABI sys_thr_kill2(int pid, i64 tid, int sig) {
  if (pid != sys_getpid())
    return -SysError::eSRCH;
  return sys_thr_kill(tid, sig);
}

// Sleeps until thr_wake (one posted earlier counts), the timeout, or a signal.
int PS4ABI sys_thr_suspend(const void* timeout) {
  i64 us = -1;
  if (timeout) {
    if (!host_memory::IsMemoryRangeMapped(timeout, sizeof(sce_timespec)))
      return -SysError::eFAULT;
    const auto* ts = static_cast<const sce_timespec*>(timeout);
    if (ts->tv_sec < 0 || ts->tv_nsec < 0 || ts->tv_nsec >= 1'000'000'000)
      return -SysError::eINVAL;
    us = ts->tv_sec * 1'000'000 + (ts->tv_nsec + 999) / 1000;
  }
  base::UniqueLock<base::Mutex> lk(g_mutex);
  Thread& t = *CurrentLocked();
  WaitSignal(t, lk, us,
             [&] { return t.wake_pending || NextDeliverable(t) != 0; });
  if (t.wake_pending) {
    t.wake_pending = false;
    return 0;
  }
  if (NextDeliverable(t) || guest::Stopping())
    return -SysError::eINTR;
  return -SysError::eTIMEDOUT;
}

int PS4ABI sys_thr_wake(i64 tid) {
  base::LockGuard<base::Mutex> lk(g_mutex);
  Thread* t = tid == *CurrentGuestTidPtr() ? CurrentLocked()
                                           : FindThread(static_cast<u32>(tid));
  if (!t)
    return -SysError::eSRCH;
  t->wake_pending = true;
  t->cv.NotifyAll();
  return 0;
}

// Stops another thread where it runs; it parks in guest code at once, or at
// its next syscall exit while it is inside one.
int PS4ABI sys_thr_suspend_ucontext(i64 tid) {
  if (tid == *CurrentGuestTidPtr())
    return -SysError::eDEADLK;
  base::UniqueLock<base::Mutex> lk(g_mutex);
  Thread* t = FindThread(static_cast<u32>(tid));
  if (!t)
    return -SysError::eSRCH;
  if (t->stop_req)
    return 0;
  t->stop_req = true;
  RecomputeDue();
  Kick(t);
  t->cv.WaitFor(lk, base::Milliseconds(200), [&] { return t->stopped; });
  return 0;
}

int PS4ABI sys_thr_resume_ucontext(i64 tid) {
  if (tid == *CurrentGuestTidPtr())
    return -SysError::eINVAL;
  base::LockGuard<base::Mutex> lk(g_mutex);
  Thread* t = FindThread(static_cast<u32>(tid));
  if (!t)
    return -SysError::eSRCH;
  t->stop_req = false;
  RecomputeDue();
  t->cv.NotifyAll();
  return 0;
}

// The registers of a stopped thread. One still inside a syscall has none to
// show yet: its user context is the syscall's, which only its exit captures.
int PS4ABI sys_thr_get_ucontext(i64 tid, void* ucontext) {
  if (!host_memory::IsMemoryRangeMapped(ucontext, 0x4c0))
    return -SysError::eFAULT;
  base::LockGuard<base::Mutex> lk(g_mutex);
  Thread* t = FindThread(static_cast<u32>(tid));
  if (!t)
    return -SysError::eSRCH;
  if (!t->stop_req)
    return -SysError::eINVAL;
  GuestFrame none{};
  none.rsp = GuestStackScanBase();
  alignas(64) u8 clean[kXsaveSize];
  ResetXsave(clean);
  if (t->stopped)
    FillUcontext(t->snap, t->snap_xsave, t->mask, false,
                 static_cast<u8*>(ucontext));
  else
    FillUcontext(none, clean, t->mask, false, static_cast<u8*>(ucontext));
  return 0;
}

int PS4ABI sys_thr_set_ucontext(i64 tid, const void* ucontext) {
  if (!host_memory::IsMemoryRangeMapped(ucontext, 0x4c0))
    return -SysError::eFAULT;
  base::LockGuard<base::Mutex> lk(g_mutex);
  Thread* t = FindThread(static_cast<u32>(tid));
  if (!t)
    return -SysError::eSRCH;
  if (!t->stop_req)
    return -SysError::eINVAL;
  if (!t->stopped)
    return -SysError::eBUSY;
  bool has_fsbase;
  u64 fsbase;
  GuestFrame f = t->snap;
  alignas(64) u8 xsave[kXsaveSize];
  std::memcpy(xsave, t->snap_xsave, kXsaveSize);
  if (const int err =
          LoadMcontext(&f, xsave, static_cast<const u8*>(ucontext) + 0x40,
                       &has_fsbase, &fsbase))
    return err;
  t->snap = f;
  std::memcpy(t->snap_xsave, xsave, kXsaveSize);
  t->snap_dirty = true;
  return 0;
}

void TerminateBySignal(int sig) {
  Terminate(sig);
}

int PS4ABI sys_nosys() {
  constexpr int kSigSys = 12;
  SignalInfo info{};
  info.signo = kSigSys;
  info.code = kSiKernel;
  PostSignal(*CurrentGuestTidPtr(), info);
  return -SysError::eNOSYS;
}

}  // namespace kern
