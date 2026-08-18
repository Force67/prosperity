/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Hashing shared by the renderer caches: FNV-1a mixing for descriptor keys, and
// a content fingerprint for a range of guest memory.

#include "base/arch.h"

namespace gpu::vk {

inline u64 HashWord(u64 h, u64 v) {
  return (h ^ v) * 1099511628211ull;
}

u64 TexHash(u64 base, u64 bytes);
// Length plus 256 evenly spaced 64-byte windows: a fixed ~16 KB read whatever
// the surface's size, so it can be checked every frame where TexHash cannot.
// It can miss a write that falls between every window, which is why it guards
// the frequent check and TexHash still runs on a slower sweep.
u64 TexSampleHash(u64 base, u64 bytes);

}  // namespace gpu::vk
