/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_hash.h"

#include <cstring>

namespace gpu::vk {

// frame per texture unless a compute write explicitly invalidates the
// resource, and big atlases make it the dominant per-frame CPU cost -- so it
// runs four independent FNV lanes over 64-bit words (instead of one dependent
// multiply per dword) to break the serial multiply chain and go memory-bound.
uint64_t TexHash(uint64_t base, uint64_t bytes) {
  constexpr uint64_t kPrime = 1099511628211ull;
  const uint64_t* w = reinterpret_cast<const uint64_t*>(base);
  const uint64_t nw = bytes / 8;
  uint64_t h0 = 1469598103934665603ull, h1 = 0x9e3779b97f4a7c15ull,
           h2 = 0xc2b2ae3d27d4eb4full, h3 = 0x165667b19e3779f9ull;
  uint64_t i = 0;
  for (; i + 4 <= nw; i += 4) {
    h0 = (h0 ^ w[i + 0]) * kPrime;
    h1 = (h1 ^ w[i + 1]) * kPrime;
    h2 = (h2 ^ w[i + 2]) * kPrime;
    h3 = (h3 ^ w[i + 3]) * kPrime;
  }
  for (; i < nw; i++)
    h0 = (h0 ^ w[i]) * kPrime;
  if (const uint64_t tail = bytes & 7) {
    uint64_t last = 0;
    std::memcpy(&last, reinterpret_cast<const uint8_t*>(base) + bytes - tail,
                tail);
    h1 = (h1 ^ last) * kPrime;
  }
  uint64_t h = ((h0 * kPrime + h1) * kPrime + h2) * kPrime + h3;
  return h ^ (bytes << 1);
}

}  // namespace gpu::vk
