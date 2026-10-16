#include "main/memory_report.h"

#include <cstdio>

#include "base/arch.h"
#include "guest/session.h"
#include "kern/memory_budget.h"
#include "base/logging.h"
#include "memory_debug/memory_debug.h"
#include "options/options.h"

namespace {
DELTA_OPTION(u32, kReportSeconds, "DELTA_MEMORY_REPORT", 30);
DELTA_OPTION(u64, kBudgetMiB, "DELTA_MEMORY_BUDGET_MB", 0);

constexpr u64 kGiB = 1ull << 30;

u64 ResidentBytes() {
  u64 pages = 0, resident = 0;
  std::FILE* f = std::fopen("/proc/self/statm", "r");
  if (!f)
    return 0;
  if (std::fscanf(f, "%llu %llu", reinterpret_cast<unsigned long long*>(&pages),
                  reinterpret_cast<unsigned long long*>(&resident)) != 2)
    resident = 0;
  std::fclose(f);
  return resident * 4096;
}

double Gib(u64 bytes) {
  return static_cast<double>(bytes) / kGiB;
}
}  // namespace

namespace cli {

void StartMemoryReport(bool ps5) {
  if (!kReportSeconds)
    return;
  // The console's whole RAM: what a title plus its OS share ever occupy.
  const u64 budget = kBudgetMiB ? kBudgetMiB.get() << 20 : (ps5 ? 16 : 8) * kGiB;
  guest::SpawnThread("memory-report", [budget] {
    u64 warned_gib = 0;
    while (!guest::Stopping()) {
      guest::SleepForMilliseconds(u64(kReportSeconds) * 1000);
      if (guest::Stopping())
        break;
      const kern::GuestMemoryUsage guest = kern::ReadGuestMemoryUsage();
      const memory_debug::Snapshot tags = memory_debug::ReadSnapshot();
      const u64 resident = ResidentBytes();
      const u64 total = resident + tags.gpu_backing;
      BASE_LOGI(
          "memory",
          "guest direct {:.2f}/{:.2f} GiB flexible {:.2f}/{:.2f} GiB | "
          "host resident {:.2f} GiB (emulator heap {:.2f}) | gpu {:.2f} GiB | "
          "total {:.2f} of {:.2f} GiB",
          Gib(guest.direct_used), Gib(guest.direct_total),
          Gib(guest.flexible_used), Gib(guest.flexible_budget), Gib(resident),
          Gib(tags.host_live), Gib(tags.gpu_backing), Gib(total), Gib(budget));
      if (total > budget && (total - budget) / kGiB >= warned_gib) {
        warned_gib = (total - budget) / kGiB + 1;
        BASE_LOGW("memory", "{:.2f} GiB over the {:.2f} GiB console budget",
                  Gib(total - budget), Gib(budget));
      }
    }
  });
}

}  // namespace cli
