#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"

namespace gpu::ps4 {

// Titles map GPU memory as low as 16 GiB (DRIVECLUB 1.00 scans out from
// 0x400108000); the floor only keeps 32-bit garbage out.
inline constexpr u64 kGuestBase = 0x100000000ull;
inline constexpr u64 kGuestEnd = 0x20000000000ull;

inline bool IsGuestAddress(u64 address) {
  return address >= kGuestBase && address < kGuestEnd;
}

inline bool IsGuestRange(u64 address, u64 bytes) {
  return bytes && address >= kGuestBase && address < kGuestEnd &&
         bytes <= kGuestEnd - address;
}

}  // namespace gpu::ps4
