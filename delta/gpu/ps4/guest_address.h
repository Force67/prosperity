#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 */

#include "base/arch.h"

namespace gpu::ps4 {

inline constexpr u64 kGuestBase = 0x1000000000ull;
inline constexpr u64 kGuestEnd = 0x20000000000ull;

inline bool IsGuestAddress(u64 address) {
  return address >= kGuestBase && address < kGuestEnd;
}

// Test address < kGuestEnd first: kGuestEnd - address wraps above the window.
inline bool IsGuestRange(u64 address, u64 bytes) {
  return bytes && address >= kGuestBase && address < kGuestEnd &&
         bytes <= kGuestEnd - address;
}

}  // namespace gpu::ps4
