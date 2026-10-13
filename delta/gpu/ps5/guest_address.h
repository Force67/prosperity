#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "base/arch.h"
#include "base/atomic.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"

namespace gpu::ps5 {

inline constexpr u64 kGuestBase = 0x10000ull;
inline constexpr u64 kGuestEnd = 0x1000000000000ull;

inline bool IsGuestAddress(u64 address) {
  return address >= kGuestBase && address < kGuestEnd;
}

inline constexpr u64 kGpuBase = 0x1000000000ull;
inline constexpr u64 kGpuEnd = 0x10000000000ull;

inline constexpr u32 kMaxNotedPools = 128;

struct NotedPools {
  struct Range {
    u64 base, end;
  } ranges[kMaxNotedPools];
  base::Atomic<u32> count{0};
  base::Mutex lock;
};

inline NotedPools& GpuPools() {
  static NotedPools pools;
  return pools;
}

inline void NoteGpuPool(u64 base, u64 size) {
  if (!base || !size)
    return;
  auto& pools = GpuPools();
  base::LockGuard<base::Mutex> lk(pools.lock);
  const u32 n = pools.count.load(base::memory_order_relaxed);
  for (u32 i = 0; i < n; i++)
    if (pools.ranges[i].base <= base && pools.ranges[i].end >= base + size)
      return;
  if (n >= kMaxNotedPools)
    return;
  pools.ranges[n] = {base, base + size};
  pools.count.store(n + 1, base::memory_order_release);
}

inline bool GpuPoolRange(u64 address, u64& base, u64& end) {
  auto& pools = GpuPools();
  const u32 n = pools.count.load(base::memory_order_acquire);
  for (u32 i = 0; i < n; i++)
    if (address >= pools.ranges[i].base && address < pools.ranges[i].end) {
      base = pools.ranges[i].base;
      end = pools.ranges[i].end;
      return true;
    }
  return false;
}

inline bool IsGpuAddress(u64 address) {
  if (address >= kGpuBase && address < kGpuEnd)
    return true;
  auto& pools = GpuPools();
  const u32 n = pools.count.load(base::memory_order_acquire);
  for (u32 i = 0; i < n; i++)
    if (address >= pools.ranges[i].base && address < pools.ranges[i].end)
      return true;
  return false;
}

}  // namespace gpu::ps5
