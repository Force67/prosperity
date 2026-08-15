#include "base/arch.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <zlib.h>

#include "formats/archive_backend.h"

namespace {

void p16(std::vector<u8> &o, u16 v) {
  o.push_back(v & 0xff);
  o.push_back(v >> 8);
}
void p32(std::vector<u8> &o, u32 v) {
  for (int i = 0; i < 4; ++i)
    o.push_back((v >> (8 * i)) & 0xff);
}
void p64(std::vector<u8> &o, u64 v) {
  for (int i = 0; i < 8; ++i)
    o.push_back((v >> (8 * i)) & 0xff);
}

std::vector<u8> rawDeflate(const std::vector<u8> &in) {
  z_stream zs{};
  deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8,
               Z_DEFAULT_STRATEGY);
  std::vector<u8> out(deflateBound(&zs, in.size()));
  zs.next_in = const_cast<u8 *>(in.data());
  zs.avail_in = static_cast<uInt>(in.size());
  zs.next_out = out.data();
  zs.avail_out = static_cast<uInt>(out.size());
  EXPECT_EQ(deflate(&zs, Z_FINISH), Z_STREAM_END);
  out.resize(zs.total_out);
  deflateEnd(&zs);
  return out;
}

// Minimal zip writer for test fixtures; forceZip64 saturates the 32-bit size
// and offset fields and emits the ZIP64 extras + EOCD64, like a >4 GB archive.
struct ZipWriter {
  std::vector<u8> out;
  bool forceZip64 = false;
  u64 cdOff = 0;

  struct Member {
    std::string name;
    u32 crc;
    u64 comp, uncomp, lho;
    u16 method;
  };
  std::vector<Member> members;

  void add(const std::string &name, const std::vector<u8> &data, u16 method) {
    Member m;
    m.name = name;
    m.lho = out.size();
    m.method = method;
    m.uncomp = data.size();
    m.crc = crc32(crc32(0, Z_NULL, 0), data.data(),
                  static_cast<uInt>(data.size()));
    std::vector<u8> payload =
        method == 8 ? rawDeflate(data) : data;
    m.comp = payload.size();

    p32(out, 0x04034b50);
    p16(out, forceZip64 ? 45 : 20);
    p16(out, 0); // flags
    p16(out, method);
    p32(out, 0); // dos time/date
    p32(out, m.crc);
    p32(out, forceZip64 ? 0xFFFFFFFF : static_cast<u32>(m.comp));
    p32(out, forceZip64 ? 0xFFFFFFFF : static_cast<u32>(m.uncomp));
    p16(out, static_cast<u16>(name.size()));
    p16(out, forceZip64 ? 20 : 0);
    out.insert(out.end(), name.begin(), name.end());
    if (forceZip64) {
      p16(out, 0x0001);
      p16(out, 16);
      p64(out, m.uncomp);
      p64(out, m.comp);
    }
    out.insert(out.end(), payload.begin(), payload.end());
    members.push_back(std::move(m));
  }

  // A raw central-directory row with no local header behind it, to exercise
  // the skip paths (directories, unsupported methods).
  void addDirEntry(const std::string &name) {
    Member m{name, 0, 0, 0, 0, 0};
    members.push_back(std::move(m));
  }

  void finish(const std::string &comment = "") {
    cdOff = out.size();
    for (const auto &m : members) {
      bool dir = !m.name.empty() && m.name.back() == '/';
      bool z64 = forceZip64 && !dir;
      p32(out, 0x02014b50);
      p16(out, z64 ? 45 : 20);
      p16(out, z64 ? 45 : 20);
      p16(out, 0);
      p16(out, m.method);
      p32(out, 0);
      p32(out, m.crc);
      p32(out, z64 ? 0xFFFFFFFF : static_cast<u32>(m.comp));
      p32(out, z64 ? 0xFFFFFFFF : static_cast<u32>(m.uncomp));
      p16(out, static_cast<u16>(m.name.size()));
      p16(out, z64 ? 28 : 0);
      p16(out, 0); // comment len
      p16(out, 0); // disk
      p16(out, 0); // internal attrs
      p32(out, dir ? 0x10 : 0);
      p32(out, z64 ? 0xFFFFFFFF : static_cast<u32>(m.lho));
      out.insert(out.end(), m.name.begin(), m.name.end());
      if (z64) {
        p16(out, 0x0001);
        p16(out, 24);
        p64(out, m.uncomp);
        p64(out, m.comp);
        p64(out, m.lho);
      }
    }
    u64 cdSize = out.size() - cdOff;

    if (forceZip64) {
      u64 recOff = out.size();
      p32(out, 0x06064b50);
      p64(out, 44);
      p16(out, 45);
      p16(out, 45);
      p32(out, 0);
      p32(out, 0);
      p64(out, members.size());
      p64(out, members.size());
      p64(out, cdSize);
      p64(out, cdOff);
      p32(out, 0x07064b50);
      p32(out, 0);
      p64(out, recOff);
      p32(out, 1);
    }
    p32(out, 0x06054b50);
    p16(out, 0);
    p16(out, 0);
    p16(out, forceZip64 ? 0xFFFF : static_cast<u16>(members.size()));
    p16(out, forceZip64 ? 0xFFFF : static_cast<u16>(members.size()));
    p32(out, forceZip64 ? 0xFFFFFFFF : static_cast<u32>(cdSize));
    p32(out, forceZip64 ? 0xFFFFFFFF : static_cast<u32>(cdOff));
    p16(out, static_cast<u16>(comment.size()));
    out.insert(out.end(), comment.begin(), comment.end());
  }
};

std::string writeTemp(const std::vector<u8> &bytes, const char *name) {
  std::string path = testing::TempDir() + name;
  FILE *f = fopen(path.c_str(), "wb");
  EXPECT_TRUE(f);
  EXPECT_EQ(fwrite(bytes.data(), 1, bytes.size(), f), bytes.size());
  fclose(f);
  return path;
}

std::vector<u8> pattern(size_t n, u32 seed) {
  std::vector<u8> v(n);
  u32 s = seed;
  for (size_t i = 0; i < n; ++i) {
    // compressible but non-trivial
    s = s * 1664525 + 1013904223;
    v[i] = static_cast<u8>((s >> 24) & 0x3f);
  }
  return v;
}

const vfs::ArchiveEntry *find(const std::vector<vfs::ArchiveEntry> &es,
                              const char *path) {
  for (const auto &e : es)
    if (e.path == path)
      return &e;
  return nullptr;
}

std::vector<u8> readAll(vfs::ArchiveBackend &b, const vfs::ArchiveEntry &e) {
  std::vector<u8> v(e.size);
  EXPECT_EQ(b.extractRange(e, v.data(), 0, e.size), static_cast<i64>(e.size));
  return v;
}

TEST(ArchiveZip, DeflateRoundTrip) {
  auto a = pattern(300000, 1), b = pattern(17, 2);
  std::vector<u8> empty;
  ZipWriter w;
  w.add("dir/a.bin", a, 8);
  w.add("b.bin", b, 8);
  w.add("empty.bin", empty, 8);
  w.addDirEntry("dir/");
  w.finish("trailing archive comment to force the EOCD back-scan");
  auto path = writeTemp(w.out, "zt_deflate.zip");

  auto z = vfs::openZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  EXPECT_STREQ(z->name(), "zip");

  std::vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->index(es));
  ASSERT_EQ(es.size(), 3u); // the directory row is skipped

  auto *ea = find(es, "dir/a.bin");
  ASSERT_TRUE(ea);
  EXPECT_EQ(ea->size, a.size());
  EXPECT_EQ(readAll(*z, *ea), a);
  auto *eb = find(es, "b.bin");
  ASSERT_TRUE(eb);
  EXPECT_EQ(readAll(*z, *eb), b);

  auto *ee = find(es, "empty.bin");
  ASSERT_TRUE(ee);
  u8 tmp[8];
  EXPECT_EQ(z->extractRange(*ee, tmp, 0, 8), 0);

  // reads past the end clamp / return 0
  std::vector<u8> tail(100);
  EXPECT_EQ(z->extractRange(*eb, tail.data(), 10, 100), 7);
  EXPECT_EQ(memcmp(tail.data(), b.data() + 10, 7), 0);
  EXPECT_EQ(z->extractRange(*eb, tail.data(), 17, 1), 0);
  std::remove(path.c_str());
}

TEST(ArchiveZip, StoredRangedRead) {
  auto a = pattern(100000, 3);
  ZipWriter w;
  w.add("stored.bin", a, 0);
  w.finish();
  auto path = writeTemp(w.out, "zt_stored.zip");

  auto z = vfs::openZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  std::vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->index(es));
  ASSERT_EQ(es.size(), 1u);
  EXPECT_EQ(es[0].method, 0u);
  EXPECT_EQ(readAll(*z, es[0]), a);

  std::vector<u8> mid(5000);
  EXPECT_EQ(z->extractRange(es[0], mid.data(), 40000, 5000), 5000);
  EXPECT_EQ(memcmp(mid.data(), a.data() + 40000, 5000), 0);
  std::remove(path.c_str());
}

TEST(ArchiveZip, Zip64) {
  auto a = pattern(200000, 4), b = pattern(999, 5);
  ZipWriter w;
  w.forceZip64 = true;
  w.add("big/a.bin", a, 8);
  w.add("b.bin", b, 0);
  w.finish();
  auto path = writeTemp(w.out, "zt_zip64.zip");

  auto z = vfs::openZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  std::vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->index(es));
  ASSERT_EQ(es.size(), 2u);
  auto *ea = find(es, "big/a.bin");
  ASSERT_TRUE(ea);
  EXPECT_EQ(ea->size, a.size());
  EXPECT_EQ(readAll(*z, *ea), a);
  auto *eb = find(es, "b.bin");
  ASSERT_TRUE(eb);
  EXPECT_EQ(readAll(*z, *eb), b);
  std::remove(path.c_str());
}

TEST(ArchiveZip, SkipsUnsupportedMethod) {
  auto a = pattern(1000, 6);
  ZipWriter w;
  w.add("good.bin", a, 8);
  w.add("bad.bz2", a, 12); // bzip2: indexed as unsupported, skipped
  w.finish();
  auto path = writeTemp(w.out, "zt_method.zip");

  auto z = vfs::openZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  std::vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->index(es));
  ASSERT_EQ(es.size(), 1u);
  EXPECT_EQ(es[0].path, "good.bin");
  std::remove(path.c_str());
}

TEST(ArchiveZip, NotAZip) {
  auto junk = pattern(4096, 7);
  auto path = writeTemp(junk, "zt_junk.bin");
  EXPECT_FALSE(vfs::openZipBackend(base::String(path.c_str())));
  std::remove(path.c_str());
}

// Sequential streaming must resume the decoder, not restart it. 64 MB in 64 KB
// chunks is ~1000 reads; a quadratic restart implementation would decode ~32 TB
// and time out, resume decodes 64 MB once.
TEST(ArchiveZip, StreamingResumesCursor) {
  const size_t kSize = 64u << 20;
  auto a = pattern(kSize, 8);
  ZipWriter w;
  w.add("stream.bin", a, 8);
  w.finish();
  auto path = writeTemp(w.out, "zt_stream.zip");

  auto z = vfs::openZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  std::vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->index(es));
  ASSERT_EQ(es.size(), 1u);

  std::vector<u8> buf(kSize);
  auto t0 = std::chrono::steady_clock::now();
  for (size_t off = 0; off < kSize; off += 64u << 10) {
    ASSERT_EQ(z->extractRange(es[0], buf.data() + off, off, 64u << 10),
              64 << 10);
  }
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0)
                .count();
  EXPECT_EQ(buf, a);
  EXPECT_LT(ms, 30000);
  printf("[ perf ] sequential 64 MB in 64 KB chunks: %lld ms (%.0f MB/s)\n",
         static_cast<long long>(ms), ms ? 64000.0 / ms : 0.0);

  // ranged read in the middle (restart + discard forward)
  std::vector<u8> mid(1 << 20);
  auto t1 = std::chrono::steady_clock::now();
  ASSERT_EQ(z->extractRange(es[0], mid.data(), 32u << 20, 1 << 20), 1 << 20);
  auto midMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - t1)
                   .count();
  EXPECT_EQ(memcmp(mid.data(), a.data() + (32u << 20), 1 << 20), 0);
  printf("[ perf ] cold 1 MB range at +32 MB: %lld ms\n",
         static_cast<long long>(midMs));

  // backward seek restarts cleanly
  ASSERT_EQ(z->extractRange(es[0], mid.data(), 1 << 20, 1 << 20), 1 << 20);
  EXPECT_EQ(memcmp(mid.data(), a.data() + (1 << 20), 1 << 20), 0);
  std::remove(path.c_str());
}

TEST(ArchiveZip, CorruptCrcDetected) {
  auto a = pattern(50000, 9);
  ZipWriter w;
  w.add("a.bin", a, 8);
  w.finish();
  w.out[w.cdOff + 16] ^= 0xFF; // crc field of the first central row
  auto path = writeTemp(w.out, "zt_badcrc.zip");

  auto z = vfs::openZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  std::vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->index(es));
  ASSERT_EQ(es.size(), 1u);
  std::vector<u8> buf(es[0].size);
  // a full decode from 0 must fail the CRC check
  EXPECT_EQ(z->extractRange(es[0], buf.data(), 0, es[0].size), -1);
  std::remove(path.c_str());
}

TEST(ArchiveZip, CorruptStreamDetected) {
  auto a = pattern(200000, 10);
  ZipWriter w;
  w.add("a.bin", a, 8);
  w.finish();
  // clobber compressed bytes mid-stream (local header is 30 + name)
  for (u64 off = 20000; off < 20016; ++off)
    w.out[30 + 5 + off] ^= 0x5A;
  auto path = writeTemp(w.out, "zt_badstream.zip");

  auto z = vfs::openZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  std::vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->index(es));
  ASSERT_EQ(es.size(), 1u);
  std::vector<u8> buf(es[0].size);
  EXPECT_EQ(z->extractRange(es[0], buf.data(), 0, es[0].size), -1);
  std::remove(path.c_str());
}

// Interop check against an archive made by a real zip tool: index it and
// fully decode every member, relying on the CRC verification for correctness.
// Gated on DELTA_ZIPTEST_FILE so CI does not need external tooling.
TEST(ArchiveZip, ExternalArchive) {
  const char *p = getenv("DELTA_ZIPTEST_FILE");
  if (!p)
    GTEST_SKIP() << "set DELTA_ZIPTEST_FILE to a zip to run";
  auto z = vfs::openZipBackend(base::String(p));
  ASSERT_TRUE(z);
  std::vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->index(es));
  ASSERT_FALSE(es.empty());
  std::vector<u8> buf(1 << 20);
  for (const auto &e : es) {
    for (u64 off = 0; off < e.size; off += buf.size()) {
      i64 want = std::min<u64>(buf.size(), e.size - off);
      ASSERT_EQ(z->extractRange(e, buf.data(), off, want), want)
          << e.path << " @" << off;
    }
  }
  printf("[ perf ] external archive: %zu members decoded + CRC ok\n",
         es.size());
}

} // namespace
