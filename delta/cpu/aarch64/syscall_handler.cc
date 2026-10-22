#include "cpu/aarch64/syscall_handler.h"

#include <FEXCore/Core/Context.h>
#include <FEXCore/Core/CoreState.h>
#include <FEXCore/Core/X86Enums.h>
#include <FEXCore/Debug/InternalThreadState.h>

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

}  // namespace

void FexSyscallHandler::HandleSyscall(FEXCore::Core::CpuStateFrame* frame) {
  const memory_debug::Scope memory_scope(memory_debug::Bucket::kRuntime);
  guest::PausePoint();
  auto& state = frame->State;
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
