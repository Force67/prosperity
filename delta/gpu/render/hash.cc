/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/hash.h"
#include "base/arch.h"
#include "gpu/gcn/gcn_detile.h"

#include <cstring>
#include "base/containers/vector.h"
#include "base/math/value_bounds.h"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define XXH_X86DISPATCH
#define XXH_DISPATCH_AVX2 1
#define XXH_TARGET_AVX2 __attribute__((target("avx2")))
#endif
#define XXH_INLINE_ALL
#include <tracy_xxhash.h>

namespace gpu::render {

namespace {
constexpr u64 kHashPrime = 1099511628211ull;
// Below this a sweep stays on the calling thread.
constexpr u64 kParallelBytes = 2ull << 20;

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) u64 HashLongAvx2(const void* data,
                                                 u64 bytes,
                                                 u64 seed) {
  return XXH3_hashLong_64b_withSeed_internal(
      data, bytes, seed, XXH3_accumulate_avx2, XXH3_scrambleAcc_avx2,
      XXH3_initCustomSecret_avx2);
}
#endif

u64 HashBytes(const void* data, u64 bytes, u64 seed = 0) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (bytes > XXH3_MIDSIZE_MAX && __builtin_cpu_supports("avx2"))
    return HashLongAvx2(data, bytes, seed);
#endif
  return XXH3_64bits_withSeed(data, bytes, seed);
}
}  // namespace

u64 TexHashRange(u64 base, u64 bytes) {
  return HashBytes(reinterpret_cast<const void*>(base), bytes);
}

u64 TexHashCombine(const u64* parts, u64 count, u64 bytes) {
  if (count == 1)
    return parts[0];
  u64 h = 1469598103934665603ull;
  for (u64 i = 0; i < count; i++)
    h = (h ^ parts[i]) * kHashPrime;
  return h ^ (bytes << 1);
}

// Chunking is by byte offset only, so the result is independent of how the
// work is spread, including hashes assembled during compute writeback.
u64 TexHash(u64 base, u64 bytes) {
  if (bytes < kParallelBytes)
    return TexHashSerial(base, bytes);
  const u32 chunks =
      static_cast<u32>((bytes + kTexHashChunk - 1) / kTexHashChunk);
  base::Vector<u64> parts(chunks);
  gcn::DetileParallelWork(chunks, bytes, [&](u32 c0, u32 c1) {
    for (u32 c = c0; c < c1; c++) {
      const u64 off = u64(c) * kTexHashChunk;
      parts[c] =
          TexHashRange(base + off, base::Min(kTexHashChunk, bytes - off));
    }
  });
  return TexHashCombine(parts.data(), chunks, bytes);
}

u64 TexHashSerial(u64 base, u64 bytes) {
  if (bytes <= kTexHashChunk)
    return TexHashRange(base, bytes);
  u64 h = 1469598103934665603ull;
  for (u64 off = 0; off < bytes; off += kTexHashChunk)
    h = (h ^ TexHashRange(base + off, base::Min(kTexHashChunk, bytes - off))) *
        kHashPrime;
  return h ^ (bytes << 1);
}

u64 TexSampleHash(u64 base, u64 bytes) {
  if (bytes <= 16384)
    return TexHash(base, bytes);
  u8 samples[256 * 64];
  const u64 step = (bytes - 64) / 255;
  for (u32 i = 0; i < 256; i++)
    std::memcpy(samples + i * 64,
                reinterpret_cast<const void*>(base + i * step), 64);
  return HashBytes(samples, sizeof(samples), bytes);
}

}  // namespace gpu::render
