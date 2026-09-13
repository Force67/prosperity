/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_hash.h"
#include "base/arch.h"
#include "gpu/gcn/gcn_detile.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace gpu::vk {

namespace {
constexpr u64 kHashPrime = 1099511628211ull;
// Chunk size for the parallel sweep. Fixed, so a surface hashes to the same
// value however many lanes the pool gives it.
constexpr u64 kHashChunk = 1ull << 20;

// frame per texture unless a compute write explicitly invalidates the
// resource, and big atlases make it the dominant per-frame CPU cost -- so it
// runs four independent FNV lanes over 64-bit words (instead of one dependent
// multiply per dword) to break the serial multiply chain and go memory-bound.
u64 TexHashRange(u64 base, u64 bytes) {
  constexpr u64 kPrime = kHashPrime;
  const u64* w = reinterpret_cast<const u64*>(base);
  const u64 nw = bytes / 8;
  u64 h0 = 1469598103934665603ull, h1 = 0x9e3779b97f4a7c15ull,
           h2 = 0xc2b2ae3d27d4eb4full, h3 = 0x165667b19e3779f9ull;
  u64 i = 0;
  for (; i + 4 <= nw; i += 4) {
    h0 = (h0 ^ w[i + 0]) * kPrime;
    h1 = (h1 ^ w[i + 1]) * kPrime;
    h2 = (h2 ^ w[i + 2]) * kPrime;
    h3 = (h3 ^ w[i + 3]) * kPrime;
  }
  for (; i < nw; i++)
    h0 = (h0 ^ w[i]) * kPrime;
  if (const u64 tail = bytes & 7) {
    u64 last = 0;
    std::memcpy(&last, reinterpret_cast<const u8*>(base) + bytes - tail,
                tail);
    h1 = (h1 ^ last) * kPrime;
  }
  u64 h = ((h0 * kPrime + h1) * kPrime + h2) * kPrime + h3;
  return h ^ (bytes << 1);
}
}  // namespace

// One core sweeps ~2.5 GB/s of cold texture memory, and a compute-heavy title
// re-hashes over a hundred megabytes a frame. Chunking is by byte offset only,
// so the result is independent of how the work is spread.
u64 TexHash(u64 base, u64 bytes) {
  if (bytes < 2 * kHashChunk)
    return TexHashRange(base, bytes);
  const u32 chunks = static_cast<u32>((bytes + kHashChunk - 1) / kHashChunk);
  std::vector<u64> parts(chunks);
  gcn::DetileParallelWork(chunks, bytes, [&](u32 c0, u32 c1) {
    for (u32 c = c0; c < c1; c++) {
      const u64 off = u64(c) * kHashChunk;
      parts[c] = TexHashRange(base + off, std::min(kHashChunk, bytes - off));
    }
  });
  u64 h = 1469598103934665603ull;
  for (const u64 p : parts)
    h = (h ^ p) * kHashPrime;
  return h ^ (bytes << 1);
}

u64 TexSampleHash(u64 base, u64 bytes) {
  constexpr u64 kPrime = 1099511628211ull;
  if (bytes <= 16384)
    return TexHash(base, bytes);
  u64 h = 1469598103934665603ull ^ (bytes * kPrime);
  const u64 step = (bytes - 64) / 255;
  for (u32 i = 0; i < 256; i++) {
    u64 w[8];
    std::memcpy(w, reinterpret_cast<const void*>(base + i * step), 64);
    for (int j = 0; j < 8; j++)
      h = (h ^ w[j]) * kPrime;
  }
  return h;
}

}  // namespace gpu::vk
