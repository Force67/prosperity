#include "kern/memory_budget.h"

#include "base/logging.h"
#include "base/atomic.h"
#include "kern/process.h"
#include "kern/ps4/dev/dma_dev.h"
#include "kern/vm_map.h"
#include "options/options.h"

namespace {
// The value vm.budgets.mlock_* reported before it was accounted.
constexpr u64 kDefaultFlexibleBudget = 0x180000000ull;  // 6 GiB

DELTA_OPTION(u64, kFlexibleBudget, "DELTA_FLEXIBLE_BUDGET", 0);
DELTA_OPTION(bool, kEnforceGuestMemory, "DELTA_ENFORCE_GUEST_MEMORY", false);
}  // namespace

namespace kern {

u64 FlexibleMemoryBudget() {
  return kFlexibleBudget ? kFlexibleBudget.get() : kDefaultFlexibleBudget;
}

GuestMemoryUsage ReadGuestMemoryUsage() {
  GuestMemoryUsage usage;
  usage.direct_used = DmemAllocatedBytes();
  usage.direct_total = DmemBackingSize();
  usage.flexible_budget = FlexibleMemoryBudget();
  if (auto* proc = Process::GetActive())
    usage.flexible_used = proc->GetVma().FlexibleBytes();
  return usage;
}

bool AdmitFlexible(const VmMap& vm, u8* at, size_t bytes) {
  const u64 used = vm.FlexibleBytes(at, bytes);
  const u64 budget = FlexibleMemoryBudget();
  if (used + bytes <= budget)
    return true;
  static base::Atomic<bool> logged{false};
  if (!logged.exchange(true))
    BASE_LOGW("memory",
              "flexible memory over budget: {} MiB mapped + {} MiB requested > "
              "{} MiB{}",
              used >> 20, bytes >> 20, budget >> 20,
              kEnforceGuestMemory ? ", refused" : " (not enforced)");
  return !kEnforceGuestMemory;
}

}  // namespace kern
