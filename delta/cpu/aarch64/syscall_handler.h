#pragma once

#include <FEXCore/HLE/SyscallHandler.h>

#include "base/arch.h"

namespace cpu::fex {

// RAX, RDI, RSI, RDX, R10, R8, R9.
struct SyscallArguments {
  u64 Argument[7];
};

// The interrupt-fault page of the calling thread was hit at the end of a
// section that deferred an interruption: interrupts it now. True when the
// fault was that.
bool RetryDeferredInterrupt(void* info, void* ucontext);

class FexSyscallHandler final : public FEXCore::HLE::SyscallHandler {
 public:
  // FEX leaves rip at `syscall`; the handler stores RAX and steps past it.
  void HandleSyscall(FEXCore::Core::CpuStateFrame* frame) override;

  FEXCore::HLE::ExecutableRangeInfo QueryGuestExecutableRange(
      FEXCore::Core::InternalThreadState*,
      uint64_t address) override;

  std::optional<FEXCore::ExecutableFileSectionInfo> LookupExecutableFileSection(
      FEXCore::Core::InternalThreadState*,
      uint64_t) override;

 private:
  u64 DispatchSyscall(FEXCore::Core::CpuStateFrame* frame,
                      const SyscallArguments& args);
  u64 CallHostThunk(FEXCore::Core::CpuStateFrame* frame,
                    u32 idx,
                    const SyscallArguments& args);
  u64 CallLv2(FEXCore::Core::CpuStateFrame* frame,
              u32 num,
              const SyscallArguments& args);
};

}  // namespace cpu::fex
