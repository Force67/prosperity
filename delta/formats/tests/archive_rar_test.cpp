// Exercises the RAR backend against a real 54 GB RAR5 game dump. The archive
// is too big to ship, so every test skips when it is absent. The cross-check
// test shells out to the system unrar; that is test-only scaffolding, the
// backend itself never does.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <zlib.h>

#include "formats/archive_backend.h"

namespace {

const char kArchive[] = "/home/vince/Documents/dumps/PS5/PPSA01342 (1.05).rar";

struct Ctx {
  std::unique_ptr<vfs::ArchiveBackend> backend;
  std::vector<vfs::ArchiveEntry> entries;
  double indexSeconds = 0;
};

Ctx *ctx() {
  static Ctx c = [] {
    Ctx c;
    c.backend = vfs::openRarBackend(kArchive);
    if (c.backend) {
      auto t0 = std::chrono::steady_clock::now();
      if (!c.backend->index(c.entries))
        c.backend.reset();
      c.indexSeconds = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
    }
    return c;
  }();
  return &c;
}

const vfs::ArchiveEntry *find(const char *path) {
  for (auto &e : ctx()->entries)
    if (e.path == path)
      return &e;
  return nullptr;
}

#define REQUIRE_ARCHIVE()                                                      \
  if (!ctx()->backend)                                                         \
    GTEST_SKIP() << "test archive not present: " << kArchive;

} // namespace

TEST(ArchiveRar, Index) {
  REQUIRE_ARCHIVE();
  printf("indexed %zu entries in %.2fs\n", ctx()->entries.size(),
         ctx()->indexSeconds);
  // 242953 headers total: 223259 regular files + 19694 directories, which the
  // index skips (verified against `unrar vt`).
  EXPECT_EQ(ctx()->entries.size(), 223259u);
}

TEST(ArchiveRar, ParamJson) {
  REQUIRE_ARCHIVE();
  auto *e = find("PPSA01342-app/sce_sys/param.json");
  ASSERT_NE(e, nullptr);
  std::string data(e->size, 0);
  ASSERT_EQ(ctx()->backend->extractRange(*e, data.data(), 0, i64(e->size)),
            i64(e->size));
  printf("param.json (%llu bytes):\n%s\n", e->size, data.c_str());
  EXPECT_NE(data.find("\"titleId\""), std::string::npos);
}

// Full sequential extract of the boot executable: chunked forward reads must
// resume one decoder session, and a whole-file decode must match the CRC the
// archive stores. B00EA44D is the known-good value from `unrar l`.
TEST(ArchiveRar, EbootCrc) {
  REQUIRE_ARCHIVE();
  auto *e = find("PPSA01342-app/eboot.bin");
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->size, 46574897u);
  EXPECT_EQ(e->crc, 0xB00EA44Du);

  constexpr i64 kChunk = 1 << 20;
  std::vector<unsigned char> buf(kChunk);
  uLong crc = crc32(0, nullptr, 0);
  auto t0 = std::chrono::steady_clock::now();
  for (i64 off = 0; off < i64(e->size); off += kChunk) {
    i64 got = ctx()->backend->extractRange(*e, buf.data(), off, kChunk);
    ASSERT_GT(got, 0) << "at offset " << off;
    crc = crc32(crc, buf.data(), uInt(got));
  }
  double sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  printf("eboot.bin: %llu bytes in %.2fs (%.1f MB/s), crc %08lX\n", e->size,
         sec, e->size / sec / 1e6, crc);
  EXPECT_EQ(crc, 0xB00EA44Du);
}

// A ranged read in the middle of a large compressed entry must return the
// same bytes the reference unrar produces for that offset.
TEST(ArchiveRar, RangedReadMatchesUnrar) {
  REQUIRE_ARCHIVE();
  const vfs::ArchiveEntry *e = nullptr;
  for (auto &c : ctx()->entries)
    if (c.method != 0 && c.size > 100u << 20 && c.size < 1u << 30 &&
        (!e || c.size < e->size))
      e = &c;
  ASSERT_NE(e, nullptr) << "no mid-size compressed entry found";

  const i64 off = i64(e->size / 2);
  const i64 len = 64 << 10;
  std::vector<unsigned char> ours(len);
  ASSERT_EQ(ctx()->backend->extractRange(*e, ours.data(), off, len), len);

  std::string cmd = "unrar p -inul \"";
  cmd += kArchive;
  cmd += "\" \"";
  cmd += e->path;
  cmd += "\"";
  FILE *p = popen(cmd.c_str(), "r");
  ASSERT_NE(p, nullptr);
  std::vector<unsigned char> theirs(len);
  std::vector<unsigned char> skip(1 << 20);
  i64 pos = 0;
  while (pos < off) {
    size_t n = fread(skip.data(),
                     1, size_t(std::min(i64(skip.size()), off - pos)), p);
    ASSERT_GT(n, 0u) << "unrar pipe ended early at " << pos;
    pos += i64(n);
  }
  size_t got = 0;
  while (got < size_t(len)) {
    size_t n = fread(theirs.data() + got, 1, size_t(len) - got, p);
    ASSERT_GT(n, 0u);
    got += n;
  }
  pclose(p);
  printf("cross-checked %lld bytes of %s at offset %lld\n", len,
         e->path.c_str(), off);
  EXPECT_EQ(memcmp(ours.data(), theirs.data(), size_t(len)), 0);
}

// Sequential chunked reads over a multi-GB entry: if each read restarted the
// stream this would be quadratic and visibly slow down chunk over chunk.
TEST(ArchiveRar, SequentialResume) {
  REQUIRE_ARCHIVE();
  const vfs::ArchiveEntry *e = nullptr;
  for (auto &c : ctx()->entries)
    if (c.method != 0 && (!e || c.size > e->size))
      e = &c;
  ASSERT_NE(e, nullptr);
  printf("streaming %s (%llu bytes)\n", e->path.c_str(), e->size);

  constexpr i64 kChunk = 8 << 20;
  const i64 total = std::min<i64>(i64(e->size), 512 << 20);
  std::vector<unsigned char> buf(kChunk);

  auto readSpan = [&](i64 begin, i64 end) {
    auto t0 = std::chrono::steady_clock::now();
    for (i64 off = begin; off < end; off += kChunk) {
      i64 want = std::min(kChunk, end - off);
      EXPECT_EQ(ctx()->backend->extractRange(*e, buf.data(), off, want), want);
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
  };

  double first = readSpan(0, total / 2);
  double second = readSpan(total / 2, total);
  printf("first half:  %.2fs (%.1f MB/s)\n", first, total / 2.0 / first / 1e6);
  printf("second half: %.2fs (%.1f MB/s)\n", second,
         total / 2.0 / second / 1e6);
  // A restarting decoder would make the second half re-decode the first, i.e.
  // take several times longer. Resuming keeps the halves comparable.
  EXPECT_LT(second, first * 2.5);
}
