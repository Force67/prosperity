#include "gpu/render/hash.h"

#include <gtest/gtest.h>

#define XXH_INLINE_ALL
#include <tracy_xxhash.h>

#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "options/options.h"

namespace gpu::render {
namespace {

TEST(RenderHash, SerialParallelAndWritebackChunksAgree) {
  options::Init();
  base::Vector<u8> data((3 << 20) + 8);
  for (size_t i = 0; i < data.size(); i++)
    data[i] = static_cast<u8>((i * 73) ^ (i >> 7));
  for (u64 skew : {0ull, 1ull, 7ull}) {
    const u64 address = reinterpret_cast<u64>(data.data() + skew);
    for (u64 bytes : {1ull, 7ull, 8ull, 31ull, 256ull, kTexHashChunk,
                      kTexHashChunk + 1, 2ull << 20, (3ull << 20) + 1}) {
      base::Vector<u64> chunks;
      for (u64 off = 0; off < bytes; off += kTexHashChunk)
        chunks.push_back(
            TexHashRange(address + off, base::Min(kTexHashChunk, bytes - off)));
      EXPECT_EQ(TexHashRange(address, bytes),
                XXH3_64bits(reinterpret_cast<const void*>(address), bytes));
      const u64 expected = TexHashCombine(chunks.data(), chunks.size(), bytes);
      EXPECT_EQ(TexHashSerial(address, bytes), expected);
      EXPECT_EQ(TexHash(address, bytes), expected);
    }
  }
}

TEST(RenderHash, SampleMatchesSeededReference) {
  base::Vector<u8> data((2 << 20) + 33, 0x7b);
  const u64 address = reinterpret_cast<u64>(data.data());
  for (u64 bytes : {16385ull, 65536ull, (2ull << 20) + 33})
    EXPECT_EQ(TexSampleHash(address, bytes),
              XXH3_64bits_withSeed(data.data(), 16384, bytes));
}

TEST(RenderHash, DetectsTailAndSampleChanges) {
  base::Vector<u8> data(kTexHashChunk + 1, 0);
  const u64 address = reinterpret_cast<u64>(data.data());
  const u64 full = TexHashSerial(address, data.size());
  data.back() = 1;
  EXPECT_NE(TexHashSerial(address, data.size()), full);
  const u64 sample = TexSampleHash(address, data.size());
  data[0] = 1;
  EXPECT_NE(TexSampleHash(address, data.size()), sample);
  EXPECT_NE(TexSampleHash(address, data.size() - 1),
            TexSampleHash(address, data.size()));
}

}  // namespace
}  // namespace gpu::render
