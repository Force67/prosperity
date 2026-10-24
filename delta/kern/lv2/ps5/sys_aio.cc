/*
 * PS4Delta : PS4 emulation and research project
 *
 * aio_init (0x29e) on a PS5: configures, once, the AIO scheduler the caller's
 * application category is served by. A game (category up to 0xffff) gets the
 * big-application scheduler and may tune it with a SceKernelAioParam; system
 * categories share another. Requests here complete inside submit, so the
 * scheduling parameters are checked and kept but order nothing.
 */

#include <cstring>
#include "base/arch.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "guest/session.h"
#include "guest_abi.h"
#include "host_memory/host_memory.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/ps5/handlers.h"
#include "kern/process.h"

namespace kern {
namespace {
// SceKernelAioSchedulingParam.
struct SchedulingParam {
  u32 window;
  u32 delayed_count;
  u32 enable_split;
  u32 split_size;
  u32 split_chunk_size;
};

// SceKernelAioParam: low, mid, high.
constexpr size_t kParamSize = 3 * sizeof(SchedulingParam);
static_assert(kParamSize == 0x3c);

constexpr u32 kCountMax = 128;
constexpr u32 kSplitMax = 0x1000000;

struct Scheduler {
  bool started = false;
  SchedulingParam params[3];
};

enum Which { kBigApp, kSystemApp, kShared };

base::Mutex g_mutex;
Scheduler g_schedulers[3];

const guest::SessionReset g_reset([] {
  base::LockGuard<base::Mutex> lk(g_mutex);
  for (auto& s : g_schedulers)
    s = Scheduler{};
});

// Which scheduler serves the caller, and whether it may set parameters.
Which SchedulerOf(bool* tunable) {
  const Process* proc = Process::GetActive();
  *tunable = false;
  if (!proc || proc->GetAppCategory() > 0xffffff)
    return kShared;
  *tunable = true;
  const u32 category = proc->GetAppCategory();
  if (category <= 0xffff)
    return kBigApp;
  return (category & 0xff0000) == 0x10000 ? kSystemApp : kShared;
}

bool Valid(const SchedulingParam& p) {
  if (p.window > kCountMax || p.delayed_count > kCountMax)
    return false;
  if (p.enable_split > 1)
    return false;
  return !p.enable_split ||
         (p.split_size <= kSplitMax && p.split_chunk_size <= kSplitMax);
}

// Starts `which` once; a second start of anything but the shared scheduler is
// EBUSY. Under g_mutex.
int Start(Which which, bool defaults) {
  Scheduler& s = g_schedulers[which];
  if (s.started)
    return which == kShared ? 0 : -SysError::eBUSY;
  if (defaults)
    for (u32 i = 0; i < 3; i++)
      s.params[i] = {32, 32, i < 2 ? 1u : 0u, 0x100000, 0x100000};
  s.started = true;
  return 0;
}
}  // namespace

int PS4ABI Ps5AioInit(const void* param,
                      int size,
                      u64 reserved0,
                      u32 reserved1) {
  bool tunable;
  const Which which = SchedulerOf(&tunable);
  if (reserved0 || reserved1)
    return -SysError::eINVAL;
  base::LockGuard<base::Mutex> lk(g_mutex);
  if (!param || !tunable || g_schedulers[which].started)
    return Start(which, true);
  if (size != static_cast<int>(kParamSize))
    return -SysError::eINVAL;
  if (!host_memory::IsMemoryRangeMapped(param, kParamSize))
    return -SysError::eFAULT;
  SchedulingParam p[3];
  std::memcpy(p, param, kParamSize);
  for (const auto& group : p) {
    if (!Valid(group))
      return -SysError::eINVAL;
  }
  Scheduler& s = g_schedulers[which];
  for (u32 i = 0; i < 3; i++) {
    s.params[i] = p[i];
    if (!p[i].enable_split)
      s.params[i].split_size = s.params[i].split_chunk_size = 0;
  }
  return Start(which, false);
}

}  // namespace kern
