#include <ucontext.h>
#include <csignal>

#include <FEXCore/Core/Context.h>
#include <FEXCore/Debug/InternalThreadState.h>
#include <FEXCore/Utils/ArchHelpers/Arm64.h>

#include "base/atomic.h"
#include "base/containers/set.h"
#include "base/logging.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "cpu/aarch64/fex_context.h"
#include "cpu/aarch64/syscall_handler.h"
#include "cpu/backend.h"

namespace cpu {

using fex::g_context;
using fex::t_thread;

u64 ReconstructGuestRip(u64 host_pc) {
  if (!g_context || !t_thread)
    return 0;
  if (!g_context->IsAddressInCodeBuffer(t_thread, host_pc))
    return 0;
  return g_context->RestoreRIPFromHostPC(t_thread, host_pc);
}

bool GuestGregsFromSignal(const void* ucontext, u64 out[16]) {
  if (!ucontext || !t_thread)
    return false;
  const auto* uc = static_cast<const ucontext_t*>(ucontext);
  if (!ReconstructGuestRip(uc->uc_mcontext.pc))
    return false;
  // FEX's x64::SRA: the host register pinned to each guest GPR.
  static constexpr int kSra[16] = {4,  7,  5,  6,  8,  9,  10, 11,
                                   12, 13, 14, 15, 16, 17, 19, 29};
  for (int i = 0; i < 16; i++)
    out[i] = uc->uc_mcontext.regs[kSra[i]];
  return true;
}

bool TryHandleJitSignal(int sig, void* infop, void* ucv) {
  if (!g_context || !t_thread || !ucv || !infop)
    return false;
  auto* uc = static_cast<ucontext_t*>(ucv);

  if (sig == SIGSEGV && fex::RetryDeferredInterrupt(infop, ucv))
    return true;

  // Unpaired call/ret (fiber switches) run FEX's call-ret predictor into its
  // guard pages; upstream FEX resets x25.
  if (sig == SIGSEGV && t_thread->CallRetStackBase) {
    const u64 fa =
        reinterpret_cast<u64>(static_cast<siginfo_t*>(infop)->si_addr);
    const u64 cr_base = reinterpret_cast<u64>(t_thread->CallRetStackBase);
    constexpr size_t kCrSize =
        FEXCore::Core::InternalThreadState::CALLRET_STACK_SIZE;
    constexpr size_t kPage = 0x1000;
    if (fa >= cr_base - kPage && fa < cr_base + kCrSize + kPage) {
      uc->uc_mcontext.regs[25] = cr_base + kCrSize / 4;
      static base::Atomic<u32> n{0};
      u32 c = n.fetch_add(1);
      if (c < 8 || (c & (c - 1)) == 0)
        BASE_LOGI("fex",
                  "callret predictor over/underflow #{} reset (fault {:#x})", c,
                  (unsigned long long)fa);
      return true;
    }
  }

  // Unaligned atomics: FEX expects its frontend to backpatch on BUS_ADRALN.
  if (sig != SIGBUS)
    return false;
  const u64 pc = uc->uc_mcontext.pc;
  if (!g_context->IsAddressInCodeBuffer(t_thread, pc))
    return false;
  if (static_cast<siginfo_t*>(infop)->si_code != BUS_ADRALN)
    return false;
  auto result = FEXCore::ArchHelpers::Arm64::HandleUnalignedAccess(
      t_thread, FEXCore::ArchHelpers::Arm64::UnalignedHandlerType::HalfBarrier,
      pc, reinterpret_cast<uint64_t*>(&uc->uc_mcontext.regs[0]));
  // HalfBarrier backpatches are no longer atomic.
  static base::Mutex log_m;
  static base::Set<u64> seen_rips;
  const u64 grip = ReconstructGuestRip(pc);
  {
    base::LockGuard<base::Mutex> lk(log_m);
    if (seen_rips.insert(grip).second)
      BASE_LOGI("fex",
                "unaligned-atomic backpatch site guest rip={:#x} ({} sites)",
                (unsigned long long)grip, (unsigned)seen_rips.size());
  }
  uc->uc_mcontext.pc = pc + result.value_or(0);
  return result.has_value();
}

}  // namespace cpu
