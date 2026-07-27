/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Hashing shared by the renderer caches: FNV-1a mixing for descriptor keys, and
// a content fingerprint for a range of guest memory.

#include <cstdint>

namespace gpu::vk {

inline uint64_t HashWord(uint64_t h, uint64_t v) {
  return (h ^ v) * 1099511628211ull;
}

uint64_t TexHash(uint64_t base, uint64_t bytes);

}  // namespace gpu::vk
