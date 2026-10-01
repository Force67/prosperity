/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// FNV-1a mixing for descriptor keys and xxHash content fingerprints for the
// renderer's in-memory caches.

#include "base/arch.h"

namespace gpu::render {

inline u64 HashWord(u64 h, u64 v) {
  return (h ^ v) * 1099511628211ull;
}

// A content fingerprint: TexHashRange over each kTexHashChunk-byte chunk,
// combined by TexHashCombine. A caller that already walks the bytes chunk by
// chunk can build the same value itself.
constexpr u64 kTexHashChunk = 64 * 1024;
u64 TexHashRange(u64 base, u64 bytes);
u64 TexHashCombine(const u64* parts, u64 count, u64 bytes);
u64 TexHash(u64 base, u64 bytes);
// The same value computed on the calling thread only, for callers that are
// themselves pool workers.
u64 TexHashSerial(u64 base, u64 bytes);
// Length plus 256 evenly spaced 64-byte windows: a fixed ~16 KB read whatever
// the surface's size, so it can be checked every frame where TexHash cannot.
// It can miss a write that falls between every window, which is why it guards
// the frequent check and TexHash still runs on a slower sweep.
u64 TexSampleHash(u64 base, u64 bytes);

}  // namespace gpu::render
