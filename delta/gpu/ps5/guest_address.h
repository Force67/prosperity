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

inline bool IsGpuAddress(u64 address) {
  return address >= kGpuBase && address < kGpuEnd;
}

}  // namespace gpu::ps5
