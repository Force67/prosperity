#include "cpu/aarch64/syscall_handler.h"

#include <FEXCore/Core/Context.h>
#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/X86Enums.h>
#include <FEXCore/Debug/InternalThreadState.h>

#include <sys/mman.h>
#include <ucontext.h>
#include <csignal>
#include <cstring>
#include "base/logging.h"
#include "cpu/aarch64/code_map.h"
#include "cpu/aarch64/crossing_trace.h"
#include "cpu/aarch64/fex_context.h"
#include "cpu/aarch64/host_thunks.h"
#include "cpu/backend.h"
#include "guest/pause.h"
#include "guest_abi.h"

#include "kern/lv2/dispatch.h"
#include "kern/lv2/sys_dynlib.h"
#include "kern/lv2/sys_signal.h"
#include "memory_debug/memory_debug.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kFexSctrace, "FEX_SCTRACE", false);
DELTA_OPTION(bool, kHleTrace, "DELTA_HLE_TRACE", false);
}  // namespace

namespace cpu::fex {
namespace {

// CF isn't in flags[]; the guest's `jb cerror` reads it.
void SetGuestCarry(FEXCore::Core::CpuStateFrame* frame, bool carry) {
  if (!g_context)
    return;
  u32 ef = g_context->ReconstructCompactedEFLAGS(frame->Thread, false, nullptr,
                                                 0);
  if (carry)
    ef |= 1u;
  else
    ef &= ~1u;
  g_context->SetFlagsFromCompactedEFLAGS(frame->Thread, ef);
}

namespace x86 = FEXCore::X86State;

u16 FswOf(const FEXCore::Core::CPUState& s) {
  return static_cast<u16>(
      s.flags[x86::X87FLAG_TOP_LOC] << 11 | s.flags[x86::X87FLAG_C0_LOC] << 8 |
      s.flags[x86::X87FLAG_C1_LOC] << 9 | s.flags[x86::X87FLAG_C2_LOC] << 10 |
      s.flags[x86::X87FLAG_C3_LOC] << 14 | s.flags[x86::X87FLAG_IE_LOC]);
}

// The thread's FPU and vector state as the XSAVE image the context handlers
// read (x87, SSE and AVX components), and back.
void SaveXsave(FEXCore::Core::CpuStateFrame* frame, u8* x) {
  const auto& s = frame->State;
  std::memset(x, 0, kern::kGuestXsaveSize);
  const u16 fsw = FswOf(s);
  const u32 mxcsr_mask = 0xffbf;
  const u64 bv = 7;
  std::memcpy(x, &s.FCW, 2);
  std::memcpy(x + 2, &fsw, 2);
  x[4] = s.AbridgedFTW;
  std::memcpy(x + 0x18, &s.mxcsr, 4);
  std::memcpy(x + 0x1c, &mxcsr_mask, 4);
  const u32 top = s.flags[x86::X87FLAG_TOP_LOC];
  for (u32 i = 0; i < 8; i++)
    std::memcpy(x + 0x20 + i * 16, &s.mm[(i + top) % 8], 16);
  __uint128_t lo[16], hi[16];
  g_context->ReconstructXMMRegisters(frame->Thread, lo, hi);
  std::memcpy(x + 0xa0, lo, sizeof(lo));
  std::memcpy(x + 0x200, &bv, 8);
  std::memcpy(x + 0x240, hi, sizeof(hi));
}

void LoadXsave(FEXCore::Core::CpuStateFrame* frame, const u8* x) {
  auto& s = frame->State;
  u16 fsw;
  u64 bv;
  std::memcpy(&s.FCW, x, 2);
  std::memcpy(&fsw, x + 2, 2);
  s.AbridgedFTW = x[4];
  std::memcpy(&s.mxcsr, x + 0x18, 4);
  std::memcpy(&bv, x + 0x200, 8);
  s.flags[x86::X87FLAG_IE_LOC] = fsw & 1;
  s.flags[x86::X87FLAG_C0_LOC] = (fsw >> 8) & 1;
  s.flags[x86::X87FLAG_C1_LOC] = (fsw >> 9) & 1;
  s.flags[x86::X87FLAG_C2_LOC] = (fsw >> 10) & 1;
  s.flags[x86::X87FLAG_C3_LOC] = (fsw >> 14) & 1;
  s.flags[x86::X87FLAG_TOP_LOC] = (fsw >> 11) & 7;
  const u32 top = (fsw >> 11) & 7;
  for (u32 i = 0; i < 8; i++)
    std::memcpy(&s.mm[(i + top) % 8], x + 0x20 + i * 16, 16);
  __uint128_t lo[16] = {}, hi[16] = {};
  if (bv & 2)
    std::memcpy(lo, x + 0xa0, sizeof(lo));
  if (bv & 4)
    std::memcpy(hi, x + 0x240, sizeof(hi));
  g_context->SetXMMRegistersFromState(frame->Thread, lo, hi);
}

// Runs a handler that takes the whole register state. It sees the layout the
// native context trampoline gives it: rsp one slot below the guest's, as if a
// call had pushed the return address, and rip past the syscall.
void RunContextHandler(FEXCore::Core::CpuStateFrame* frame,
                       kern::ContextHandler handler,
                       u32 sid,
                       u64 resume_rip) {
  auto& g = frame->State.gregs;
  kern::GuestFrame f{};
  f.r15 = g[x86::REG_R15], f.r14 = g[x86::REG_R14], f.r13 = g[x86::REG_R13];
  f.r12 = g[x86::REG_R12], f.r11 = g[x86::REG_R11], f.r10 = g[x86::REG_R10];
  f.r9 = g[x86::REG_R9], f.r8 = g[x86::REG_R8], f.rdi = g[x86::REG_RDI];
  f.rsi = g[x86::REG_RSI], f.rbp = g[x86::REG_RBP], f.rdx = g[x86::REG_RDX];
  f.rcx = g[x86::REG_RCX], f.rbx = g[x86::REG_RBX], f.rax = g[x86::REG_RAX];
  f.rflags =
      g_context->ReconstructCompactedEFLAGS(frame->Thread, false, nullptr, 0);
  f.rsp = g[x86::REG_RSP] - 8;
  f.rip = resume_rip;
  alignas(64) u8 xsave[kern::kGuestXsaveSize];
  SaveXsave(frame, xsave);
  handler(sid, &f, xsave);
  g[x86::REG_R15] = f.r15, g[x86::REG_R14] = f.r14, g[x86::REG_R13] = f.r13;
  g[x86::REG_R12] = f.r12, g[x86::REG_R11] = f.r11, g[x86::REG_R10] = f.r10;
  g[x86::REG_R9] = f.r9, g[x86::REG_R8] = f.r8, g[x86::REG_RDI] = f.rdi;
  g[x86::REG_RSI] = f.rsi, g[x86::REG_RBP] = f.rbp, g[x86::REG_RDX] = f.rdx;
  g[x86::REG_RCX] = f.rcx, g[x86::REG_RBX] = f.rbx, g[x86::REG_RAX] = f.rax;
  g[x86::REG_RSP] = f.rsp;
  frame->State.rip = f.rip;
  g_context->SetFlagsFromCompactedEFLAGS(frame->Thread,
                                         static_cast<u32>(f.rflags));
  LoadXsave(frame, xsave);
}

thread_local cpu::GuestInterruptHandler t_deferred = nullptr;

// The host FP/SIMD registers saved in a signal frame.
const __uint128_t* HostVregs(const ucontext_t* uc) {
  const auto* ctx =
      reinterpret_cast<const fpsimd_context*>(uc->uc_mcontext.__reserved);
  return ctx->head.magic == FPSIMD_MAGIC
             ? reinterpret_cast<const __uint128_t*>(ctx->vregs)
             : nullptr;
}

// Moves the guest registers the JIT keeps in host registers, and the guest
// rip it reached, into the thread's CPU state.
void SpillGuestRegisters(FEXCore::Core::InternalThreadState* thread,
                         ucontext_t* uc) {
  const auto& cfg = g_signal_delegator->GetConfig();
  auto& s = thread->CurrentFrame->State;
  s.rip = g_context->RestoreRIPFromHostPC(thread, uc->uc_mcontext.pc);
  for (u32 i = 0; i < cfg.SRAGPRCount; i++)
    s.gregs[i] = uc->uc_mcontext.regs[cfg.SRAGPRMapping[i]];
  if (const __uint128_t* v = HostVregs(uc)) {
    __uint128_t lo[16], hi[16];
    g_context->ReconstructXMMRegisters(thread, lo, hi);
    for (u32 i = 0; i < cfg.SRAFPRCount && i < 16; i++)
      lo[i] = v[cfg.SRAFPRMapping[i]];
    g_context->SetXMMRegistersFromState(thread, lo, hi);
  }
  const u32 ef = g_context->ReconstructCompactedEFLAGS(
      thread, true, reinterpret_cast<const uint64_t*>(uc->uc_mcontext.regs),
      uc->uc_mcontext.pstate);
  g_context->SetFlagsFromCompactedEFLAGS(thread, ef);
}

// Runs `handler` on the interrupted thread's state, then sends it to the
// dispatcher, which reloads that state and continues at its rip.
void InterruptAt(FEXCore::Core::InternalThreadState* thread,
                 ucontext_t* uc,
                 cpu::GuestInterruptHandler handler) {
  auto* frame = thread->CurrentFrame;
  SpillGuestRegisters(thread, uc);
  RunContextHandler(frame, handler, 0, frame->State.rip);
  const auto& cfg = g_signal_delegator->GetConfig();
  uc->uc_mcontext.pc = cfg.AbsoluteLoopTopAddressFillSRA;
  uc->uc_mcontext.regs[1] = 0;  // fill every static register, not one
  uc->uc_mcontext.regs[28] = reinterpret_cast<u64>(frame);
}

void ArmInterruptFaultPage(FEXCore::Core::InternalThreadState* thread,
                           int prot) {
  ::mprotect(thread->InterruptFaultPage, sizeof(thread->InterruptFaultPage),
             prot);
}

}  // namespace

bool RetryDeferredInterrupt(void* info, void* ucv) {
  auto* thread = t_thread;
  const auto* si = static_cast<const siginfo_t*>(info);
  if (!thread || !si || si->si_addr != thread->InterruptFaultPage)
    return false;
  auto* uc = static_cast<ucontext_t*>(ucv);
  if (thread->CurrentFrame->State.DeferredSignalRefCount.Load() != 0) {
    uc->uc_mcontext.pc += 4;  // a nested section's end: go on to the outer one
    return true;
  }
  ArmInterruptFaultPage(thread, PROT_READ | PROT_WRITE);
  const cpu::GuestInterruptHandler handler = t_deferred;
  t_deferred = nullptr;
  if (handler)
    InterruptAt(thread, uc, handler);
  return true;
}

void FexSyscallHandler::HandleSyscall(FEXCore::Core::CpuStateFrame* frame) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kRuntime);
  guest::PausePoint();
  auto& state = frame->State;
  const u32 num = static_cast<u32>(state.gregs[FEXCore::X86State::REG_RAX]);
  if (const kern::ContextHandler handler = kern::Lv2ContextHandler(num)) {
    RunContextHandler(frame, handler, num, state.rip + 2);
    guest::PausePoint();
    return;
  }
  const SyscallArguments args = {{
      state.gregs[FEXCore::X86State::REG_RAX],
      state.gregs[FEXCore::X86State::REG_RDI],
      state.gregs[FEXCore::X86State::REG_RSI],
      state.gregs[FEXCore::X86State::REG_RDX],
      state.gregs[FEXCore::X86State::REG_R10],
      state.gregs[FEXCore::X86State::REG_R8],
      state.gregs[FEXCore::X86State::REG_R9],
  }};
  state.gregs[FEXCore::X86State::REG_RAX] = DispatchSyscall(frame, args);
  state.rip += 2;
  if (__atomic_load_n(&kern::g_signal_due, __ATOMIC_ACQUIRE))
    RunContextHandler(frame, &kern::SignalAtSyscallExit, 0, state.rip);
  guest::PausePoint();
}

FEXCore::HLE::ExecutableRangeInfo FexSyscallHandler::QueryGuestExecutableRange(
    FEXCore::Core::InternalThreadState*,
    uint64_t address) {
  const ExecRange r = FindExecRange(address);
  return {r.base, r.size, false};
}

std::optional<FEXCore::ExecutableFileSectionInfo>
FexSyscallHandler::LookupExecutableFileSection(
    FEXCore::Core::InternalThreadState*,
    uint64_t) {
  return std::nullopt;
}

u64 FexSyscallHandler::DispatchSyscall(FEXCore::Core::CpuStateFrame* frame,
                                       const SyscallArguments& args) {
  const u32 num = static_cast<u32>(args.Argument[0]);

  if (num == kTlsGetAddrSyscall) {
    u64 r = reinterpret_cast<u64>(kern::GuestTlsGetAddr(
        reinterpret_cast<kern::tls_index*>(args.Argument[1])));
    SetGuestCarry(frame, false);
    return r;
  }

  if ((num & 0xFF000000u) == kHostThunkSyscallBase) {
    const u64 ret = CallHostThunk(frame, num & 0x00FFFFFFu, args);
    SetGuestCarry(frame, false);
    return ret;
  }

  return CallLv2(frame, num, args);
}

// The trampoline moved rcx (clobbered by `syscall`) to r10; args 7+ are on
// the guest stack above the return address.
u64 FexSyscallHandler::CallHostThunk(FEXCore::Core::CpuStateFrame* frame,
                                     u32 idx,
                                     const SyscallArguments& args) {
  const HostThunk thunk = LookupHostThunk(idx);
  if (!thunk.fn)
    return 0;
  const u64 rsp = frame->State.gregs[FEXCore::X86State::REG_RSP];
  u64 s[8] = {};
  if (rsp)
    for (int i = 0; i < 8; i++)
      s[i] = reinterpret_cast<u64*>(rsp)[i + 1];
  using Fn = u64(PS4ABI*)(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64,
                          u64, u64, u64);
  u64 caller = (rsp) ? reinterpret_cast<u64*>(rsp)[0] : 0;
  TraceEvt& ev = TraceNext();
  ev = {'h',
        idx,
        args.Argument[1],
        args.Argument[2],
        args.Argument[3],
        args.Argument[4],
        ~0ull,
        caller,
        thunk.name};
  const u64 ret = reinterpret_cast<Fn>(thunk.fn)(
      args.Argument[1], args.Argument[2], args.Argument[3], args.Argument[4],
      args.Argument[5], args.Argument[6], s[0], s[1], s[2], s[3], s[4], s[5],
      s[6], s[7]);
  ev.ret = ret;
  if (kHleTrace) {
    char cs[256];
    SymbolizeModuleAddress(caller, cs, sizeof(cs));
    BASE_LOGI("hle", "{} thunk#{}({:#x},{:#x},{:#x},{:#x}) -> {:#x}  from {}",
              thunk.name, idx, args.Argument[1], args.Argument[2],
              args.Argument[3], args.Argument[4], (unsigned long)ret, cs);
  }
  return ret;
}

u64 FexSyscallHandler::CallLv2(FEXCore::Core::CpuStateFrame* frame,
                               u32 num,
                               const SyscallArguments& args) {
  const uintptr_t handler = kern::Lv2Lookup(num);
  if (!handler)
    return 0;

  if (kFexSctrace)
    BASE_LOGI("sc", "{:3} {:<22} ({:#x}, {:#x}, {:#x}, {:#x}, {:#x}, {:#x})",
              num, kern::SyscallGetname(num), args.Argument[1],
              args.Argument[2], args.Argument[3], args.Argument[4],
              args.Argument[5], args.Argument[6]);

  using Fn = u64(PS4ABI*)(u64, u64, u64, u64, u64, u64);
  auto fn = reinterpret_cast<Fn>(handler);
  if (kern::g_sc_hist)
    g_sys_hist[num & 1023]++;
  BeginSyscall(num);
  TraceEvt& ev = TraceNext();
  ev = {'s',
        num,
        args.Argument[1],
        args.Argument[2],
        args.Argument[3],
        args.Argument[4],
        ~0ull,
        0,
        nullptr};
  u64 ret = fn(args.Argument[1], args.Argument[2], args.Argument[3],
               args.Argument[4], args.Argument[5], args.Argument[6]);
  const u32 error = kern::SyscallErrno(ret);
  if (num == 42 && !error) {
    frame->State.gregs[FEXCore::X86State::REG_RDX] = ret >> 32;
    ret = static_cast<u32>(ret);
  }
  if (error)
    ret = error;
  ev.ret = ret;
  EndSyscall();
  if (kFexSctrace)
    BASE_LOGI("sc", "    -> {:#x}", ret);
  SetGuestCarry(frame, error != 0);
  return ret;
}

}  // namespace cpu::fex

namespace cpu {

bool InterruptGuest(void* ucv, GuestInterruptHandler handler) {
  auto* thread = fex::t_thread;
  if (!thread || !fex::g_context || !fex::g_signal_delegator || !ucv)
    return false;
  auto* uc = static_cast<ucontext_t*>(ucv);
  if (!fex::g_context->IsAddressInCodeBuffer(thread, uc->uc_mcontext.pc))
    return false;
  // A syscall leaving the JIT has half its registers spilled already; its exit
  // delivers instead.
  if (thread->CurrentFrame->InSyscallInfo)
    return false;
  if (thread->CurrentFrame->State.DeferredSignalRefCount.Load() != 0) {
    fex::t_deferred = handler;
    fex::ArmInterruptFaultPage(thread, PROT_NONE);
    return true;
  }
  fex::InterruptAt(thread, uc, handler);
  return true;
}

}  // namespace cpu
