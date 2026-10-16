#pragma once

#include <cstddef>

#include "base/arch.h"

namespace kern {
class VmMap;

// What a title's own allocations use against what the console grants it.
// Direct memory is a fixed pool (DmemBackingSize); flexible memory is every
// committed anonymous mapping, paid from the vm.budgets.mlock budget.
struct GuestMemoryUsage {
  u64 direct_used = 0;
  u64 direct_total = 0;
  u64 flexible_used = 0;
  u64 flexible_budget = 0;
};

u64 FlexibleMemoryBudget();
GuestMemoryUsage ReadGuestMemoryUsage();

// Whether an anonymous mapping of `bytes` at `at` (replacing what a MAP_FIXED
// covers) fits the flexible budget. Over budget it logs once, and refuses only
// with DELTA_ENFORCE_GUEST_MEMORY until the defaults are measured per title.
bool AdmitFlexible(const VmMap& vm, u8* at, size_t bytes);
}  // namespace kern
