#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * The window of the guest address space an AGC packet may point at.
 *
 * Two windows, because a PS5 title's allocations are not all in one place. The
 * GPU aperture is where the allocator hands out the things the command
 * processor follows (command buffers, register images, state blocks) and is
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

// GPU aperture: from the lowest slot a title fixed-maps a pool at (64 GiB) to the 2^40
// user ceiling. Isaac's AGC pool sits ~0x80_xx.., Skyrim's command buffers ~0x10_xx..,
// but bigger titles run past either (Minecraft's bgfx buffers at 0x89_xx..; an earlier
// 0x81 ceiling silently dropped every submit naming one).
inline constexpr u64 kGpuBase = 0x1000000000ull;
inline constexpr u64 kGpuEnd = 0x10000000000ull;

// ...and the pools a title actually maps, since the band is only ever a guess. Astro Bot
// fixed-maps its GPU pools at 12-25 GiB, under the 64 GiB floor, so every submit naming
// one was dropped and the title waited forever on the fence they'd have written. The
// kernel notes each dmem mapping (kern/ps5/dev/dma_dev.cpp).
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

// The pool an address belongs to, when the kernel noted one. A raw pointer is
// valid to the end of its own allocation and no further, which is the only
// honest extent available for a global_* access.
inline bool GpuPoolRange(u64 address, u64& base, u64& end) {
  auto& pools = GpuPools();
  const u32 n = pools.count.load(std::memory_order_acquire);
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
  const u32 n = pools.count.load(std::memory_order_acquire);
  for (u32 i = 0; i < n; i++)
    if (address >= pools.ranges[i].base && address < pools.ranges[i].end)
      return true;
  return false;
}

}  // namespace gpu::ps5
