#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The window of the guest address space an AGC packet may point at.
 *
 * Two windows, because a PS5 title's allocations are not all in one place. The
 * GPU aperture is where the allocator hands out the things the command
 * processor follows -- command buffers, register images, state blocks -- and is
 * narrow enough to reject a mis-parsed packet field. Everything else a packet
 * may name (shader code, a fence label, a surface) can sit anywhere in the user
 * map, so it gets only the wide test.
 *
 * A range test, not a mapping test: gpu/guest_memory.h answers whether the
 * bytes are readable right now and costs a syscall. Ask this one first.
 */

#include <atomic>
#include <mutex>

#include "base/arch.h"

namespace gpu::ps5 {

// Anything the guest can hand out: an eboot lands around 0x2014_0000_0000, the
// GPU dmem tagged regions at 0x8000_0000_0000, the AGC doorbell page at
// 0xfe0_0000_0000. Only null and the low pages are excluded.
inline constexpr u64 kGuestBase = 0x10000ull;
inline constexpr u64 kGuestEnd = 0x1000000000000ull;

inline bool IsGuestAddress(u64 address) {
  return address >= kGuestBase && address < kGuestEnd;
}

// The GPU aperture: from the lowest slot a title fixed-maps a pool at (64 GiB)
// up to the 2^40 user ceiling allocLowGuest() bumps towards. Isaac's AGC pool
// sits in the 0x80_xx_xx_xx_xx band and Skyrim's command buffers land around
// 0x10_00_00_00_00, but a title that allocates more than a few GiB runs well
// past either: Minecraft's bgfx command buffers are at 0x89_xx_xx_xx_xx, and an
// earlier 0x81_00_00_00_00 ceiling silently dropped every submit naming one.
inline constexpr u64 kGpuBase = 0x1000000000ull;
inline constexpr u64 kGpuEnd = 0x10000000000ull;

// ...and the pools a title actually maps, because that band is only ever a
// guess about where a title puts them. Astro Bot fixed-maps its GPU pools at 12
// to 25 GiB, all of it under the 64 GiB floor, so every draw submit naming one
// was dropped and the title waited forever on a fence the dropped work would
// have written. Direct memory is where GPU-visible memory comes from, so the
// kernel notes each mapping it makes (kern/ps5/dev/dma_dev.cpp).
inline constexpr u32 kMaxNotedPools = 64;

struct NotedPools {
  struct Range {
    u64 base, end;
  } ranges[kMaxNotedPools];
  std::atomic<u32> count{0};
  std::mutex lock;
};

inline NotedPools& GpuPools() {
  static NotedPools pools;
  return pools;
}

inline void NoteGpuPool(u64 base, u64 size) {
  if (!base || !size)
    return;
  auto& pools = GpuPools();
  std::lock_guard<std::mutex> lk(pools.lock);
  const u32 n = pools.count.load(std::memory_order_relaxed);
  for (u32 i = 0; i < n; i++)
    if (pools.ranges[i].base <= base && pools.ranges[i].end >= base + size)
      return;
  if (n >= kMaxNotedPools)
    return;
  pools.ranges[n] = {base, base + size};
  pools.count.store(n + 1, std::memory_order_release);
}

inline bool IsGpuAddress(u64 address) {
  if (address >= kGpuBase && address < kGpuEnd)
    return true;
  auto& pools = GpuPools();
  const u32 n = pools.count.load(std::memory_order_acquire);
  for (u32 i = 0; i < n; i++)
    if (address >= pools.ranges[i].base && address < pools.ranges[i].end)
      return true;
  return false;
}

}  // namespace gpu::ps5
