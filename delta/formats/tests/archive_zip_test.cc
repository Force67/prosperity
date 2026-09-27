#include "base/arch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>
#include <zlib.h>

#include "base/containers/vector.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/strings/xstring.h"
#include "base/time/time.h"
#include "formats/archive_backend.h"

namespace {

void P16(base::Vector<u8>& o, u16 v) {
  o.push_back(v & 0xff);
  o.push_back(v >> 8);
}
void P32(base::Vector<u8>& o, u32 v) {
  for (int i = 0; i < 4; ++i)
    o.push_back((v >> (8 * i)) & 0xff);
}
void P64(base::Vector<u8>& o, u64 v) {
  for (int i = 0; i < 8; ++i)
    o.push_back((v >> (8 * i)) & 0xff);
}

base::Vector<u8> RawDeflate(const base::Vector<u8>& in) {
  z_stream zs{};
  deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8,
               Z_DEFAULT_STRATEGY);
  base::Vector<u8> out(deflateBound(&zs, in.size()));
  zs.next_in = const_cast<u8*>(in.data());
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
  base::Vector<u8> out;
  bool force_zip64 = false;
  u64 cd_off = 0;

  struct Member {
    base::String name;
    u32 crc;
    u64 comp, uncomp, lho;
    u16 method;
  };
  base::Vector<Member> members;

  void Add(const base::String& name, const base::Vector<u8>& data, u16 method) {
    Member m;
    m.name = name;
    m.lho = out.size();
    m.method = method;
    m.uncomp = data.size();
    m.crc =
        crc32(crc32(0, Z_NULL, 0), data.data(), static_cast<uInt>(data.size()));
    base::Vector<u8> payload = method == 8 ? RawDeflate(data) : data;
    m.comp = payload.size();

    P32(out, 0x04034b50);
    P16(out, force_zip64 ? 45 : 20);
    P16(out, 0);  // flags
    P16(out, method);
    P32(out, 0);  // dos time/date
    P32(out, m.crc);
    P32(out, force_zip64 ? 0xFFFFFFFF : static_cast<u32>(m.comp));
    P32(out, force_zip64 ? 0xFFFFFFFF : static_cast<u32>(m.uncomp));
    P16(out, static_cast<u16>(name.size()));
    P16(out, force_zip64 ? 20 : 0);
    out.insert(out.end(), name.begin(), name.end());
    if (force_zip64) {
      P16(out, 0x0001);
      P16(out, 16);
      P64(out, m.uncomp);
      P64(out, m.comp);
    }
    out.insert(out.end(), payload.begin(), payload.end());
    members.push_back(base::move(m));
  }

  // A raw central-directory row with no local header behind it, to exercise
  // the skip paths (directories, unsupported methods).
  void AddDirEntry(const base::String& name) {
    Member m{name, 0, 0, 0, 0, 0};
    members.push_back(base::move(m));
  }

  void Finish(const base::String& comment = "") {
    cd_off = out.size();
    for (const auto& m : members) {
      bool dir = !m.name.empty() && m.name.back() == '/';
      bool z64 = force_zip64 && !dir;
      P32(out, 0x02014b50);
      P16(out, z64 ? 45 : 20);
      P16(out, z64 ? 45 : 20);
      P16(out, 0);
      P16(out, m.method);
      P32(out, 0);
      P32(out, m.crc);
      P32(out, z64 ? 0xFFFFFFFF : static_cast<u32>(m.comp));
      P32(out, z64 ? 0xFFFFFFFF : static_cast<u32>(m.uncomp));
      P16(out, static_cast<u16>(m.name.size()));
      P16(out, z64 ? 28 : 0);
      P16(out, 0);  // comment len
      P16(out, 0);  // disk
      P16(out, 0);  // internal attrs
      P32(out, dir ? 0x10 : 0);
      P32(out, z64 ? 0xFFFFFFFF : static_cast<u32>(m.lho));
      out.insert(out.end(), m.name.begin(), m.name.end());
      if (z64) {
        P16(out, 0x0001);
        P16(out, 24);
        P64(out, m.uncomp);
        P64(out, m.comp);
        P64(out, m.lho);
      }
    }
    u64 cd_size = out.size() - cd_off;

    if (force_zip64) {
      u64 rec_off = out.size();
      P32(out, 0x06064b50);
      P64(out, 44);
      P16(out, 45);
      P16(out, 45);
      P32(out, 0);
      P32(out, 0);
      P64(out, members.size());
      P64(out, members.size());
      P64(out, cd_size);
      P64(out, cd_off);
      P32(out, 0x07064b50);
      P32(out, 0);
      P64(out, rec_off);
      P32(out, 1);
    }
    P32(out, 0x06054b50);
    P16(out, 0);
    P16(out, 0);
    P16(out, force_zip64 ? 0xFFFF : static_cast<u16>(members.size()));
    P16(out, force_zip64 ? 0xFFFF : static_cast<u16>(members.size()));
    P32(out, force_zip64 ? 0xFFFFFFFF : static_cast<u32>(cd_size));
    P32(out, force_zip64 ? 0xFFFFFFFF : static_cast<u32>(cd_off));
    P16(out, static_cast<u16>(comment.size()));
    out.insert(out.end(), comment.begin(), comment.end());
  }
};

base::String WriteTemp(const base::Vector<u8>& bytes, const char* name) {
  base::String path = testing::TempDir() + name;
  FILE* f = fopen(path.c_str(), "wb");
  EXPECT_TRUE(f);
  EXPECT_EQ(fwrite(bytes.data(), 1, bytes.size(), f), bytes.size());
  fclose(f);
  return path;
}

base::Vector<u8> Pattern(size_t n, u32 seed) {
  base::Vector<u8> v(n);
  u32 s = seed;
  for (size_t i = 0; i < n; ++i) {
    // compressible but non-trivial
    s = s * 1664525 + 1013904223;
    v[i] = static_cast<u8>((s >> 24) & 0x3f);
  }
  return v;
}

const vfs::ArchiveEntry* Find(const base::Vector<vfs::ArchiveEntry>& es,
                              const char* path) {
  for (const auto& e : es)
    if (e.path == path)
      return &e;
  return nullptr;
}

base::Vector<u8> ReadAll(vfs::ArchiveBackend& b, const vfs::ArchiveEntry& e) {
  base::Vector<u8> v(e.size);
  EXPECT_EQ(b.ExtractRange(e, v.data(), 0, e.size), static_cast<i64>(e.size));
  return v;
}

TEST(ArchiveZip, DeflateRoundTrip) {
  auto a = Pattern(300000, 1), b = Pattern(17, 2);
  base::Vector<u8> empty;
  ZipWriter w;
  w.Add("dir/a.bin", a, 8);
  w.Add("b.bin", b, 8);
  w.Add("empty.bin", empty, 8);
  w.AddDirEntry("dir/");
  w.Finish("trailing archive comment to force the EOCD back-scan");
  auto path = WriteTemp(w.out, "zt_deflate.zip");

  auto z = vfs::OpenZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  EXPECT_STREQ(z->Name(), "zip");

  base::Vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->Index(es));
  ASSERT_EQ(es.size(), 3u);  // the directory row is skipped

  auto* ea = Find(es, "dir/a.bin");
  ASSERT_TRUE(ea);
  EXPECT_EQ(ea->size, a.size());
  EXPECT_EQ(ReadAll(*z, *ea), a);
  auto* eb = Find(es, "b.bin");
  ASSERT_TRUE(eb);
  EXPECT_EQ(ReadAll(*z, *eb), b);

  auto* ee = Find(es, "empty.bin");
  ASSERT_TRUE(ee);
  u8 tmp[8];
  EXPECT_EQ(z->ExtractRange(*ee, tmp, 0, 8), 0);

  // reads past the end clamp / return 0
  base::Vector<u8> tail(100);
  EXPECT_EQ(z->ExtractRange(*eb, tail.data(), 10, 100), 7);
  EXPECT_EQ(memcmp(tail.data(), b.data() + 10, 7), 0);
  EXPECT_EQ(z->ExtractRange(*eb, tail.data(), 17, 1), 0);
  std::remove(path.c_str());
}

TEST(ArchiveZip, StoredRangedRead) {
  auto a = Pattern(100000, 3);
  ZipWriter w;
  w.Add("stored.bin", a, 0);
  w.Finish();
  auto path = WriteTemp(w.out, "zt_stored.zip");

  auto z = vfs::OpenZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  base::Vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->Index(es));
  ASSERT_EQ(es.size(), 1u);
  EXPECT_EQ(es[0].method, 0u);
  EXPECT_EQ(ReadAll(*z, es[0]), a);

  base::Vector<u8> mid(5000);
  EXPECT_EQ(z->ExtractRange(es[0], mid.data(), 40000, 5000), 5000);
  EXPECT_EQ(memcmp(mid.data(), a.data() + 40000, 5000), 0);
  std::remove(path.c_str());
}

TEST(ArchiveZip, Zip64) {
  auto a = Pattern(200000, 4), b = Pattern(999, 5);
  ZipWriter w;
  w.force_zip64 = true;
  w.Add("big/a.bin", a, 8);
  w.Add("b.bin", b, 0);
  w.Finish();
  auto path = WriteTemp(w.out, "zt_zip64.zip");

  auto z = vfs::OpenZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  base::Vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->Index(es));
  ASSERT_EQ(es.size(), 2u);
  auto* ea = Find(es, "big/a.bin");
  ASSERT_TRUE(ea);
  EXPECT_EQ(ea->size, a.size());
  EXPECT_EQ(ReadAll(*z, *ea), a);
  auto* eb = Find(es, "b.bin");
  ASSERT_TRUE(eb);
  EXPECT_EQ(ReadAll(*z, *eb), b);
  std::remove(path.c_str());
}

TEST(ArchiveZip, SkipsUnsupportedMethod) {
  auto a = Pattern(1000, 6);
  ZipWriter w;
  w.Add("good.bin", a, 8);
  w.Add("bad.bz2", a, 12);  // bzip2: indexed as unsupported, skipped
  w.Finish();
  auto path = WriteTemp(w.out, "zt_method.zip");

  auto z = vfs::OpenZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  base::Vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->Index(es));
  ASSERT_EQ(es.size(), 1u);
  EXPECT_EQ(es[0].path, "good.bin");
  std::remove(path.c_str());
}

TEST(ArchiveZip, NotAZip) {
  auto junk = Pattern(4096, 7);
  auto path = WriteTemp(junk, "zt_junk.bin");
  EXPECT_FALSE(vfs::OpenZipBackend(base::String(path.c_str())));
  std::remove(path.c_str());
}

// Sequential streaming must resume the decoder, not restart it. 64 MB in 64 KB
// chunks is ~1000 reads; a quadratic restart implementation would decode ~32 TB
// and time out, resume decodes 64 MB once.
TEST(ArchiveZip, StreamingResumesCursor) {
  const size_t kSize = 64u << 20;
  auto a = Pattern(kSize, 8);
  ZipWriter w;
  w.Add("stream.bin", a, 8);
  w.Finish();
  auto path = WriteTemp(w.out, "zt_stream.zip");

  auto z = vfs::OpenZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  base::Vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->Index(es));
  ASSERT_EQ(es.size(), 1u);

  base::Vector<u8> buf(kSize);
  auto t0 = base::TimeTicks::Now();
  for (size_t off = 0; off < kSize; off += 64u << 10) {
    ASSERT_EQ(z->ExtractRange(es[0], buf.data() + off, off, 64u << 10),
              64 << 10);
  }
  auto ms = (base::TimeTicks::Now() - t0).InMilliseconds();
  EXPECT_EQ(buf, a);
  EXPECT_LT(ms, 30000);
  printf("[ perf ] sequential 64 MB in 64 KB chunks: %lld ms (%.0f MB/s)\n",
         static_cast<long long>(ms), ms ? 64000.0 / ms : 0.0);

  // ranged read in the middle (restart + discard forward)
  base::Vector<u8> mid(1 << 20);
  auto t1 = base::TimeTicks::Now();
  ASSERT_EQ(z->ExtractRange(es[0], mid.data(), 32u << 20, 1 << 20), 1 << 20);
  auto mid_ms = (base::TimeTicks::Now() - t1).InMilliseconds();
  EXPECT_EQ(memcmp(mid.data(), a.data() + (32u << 20), 1 << 20), 0);
  printf("[ perf ] cold 1 MB range at +32 MB: %lld ms\n",
         static_cast<long long>(mid_ms));

  // backward seek restarts cleanly
  ASSERT_EQ(z->ExtractRange(es[0], mid.data(), 1 << 20, 1 << 20), 1 << 20);
  EXPECT_EQ(memcmp(mid.data(), a.data() + (1 << 20), 1 << 20), 0);
  std::remove(path.c_str());
}

TEST(ArchiveZip, CorruptCrcDetected) {
  auto a = Pattern(50000, 9);
  ZipWriter w;
  w.Add("a.bin", a, 8);
  w.Finish();
  w.out[w.cd_off + 16] ^= 0xFF;  // crc field of the first central row
  auto path = WriteTemp(w.out, "zt_badcrc.zip");

  auto z = vfs::OpenZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  base::Vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->Index(es));
  ASSERT_EQ(es.size(), 1u);
  base::Vector<u8> buf(es[0].size);
  // a full decode from 0 must fail the CRC check
  EXPECT_EQ(z->ExtractRange(es[0], buf.data(), 0, es[0].size), -1);
  std::remove(path.c_str());
}

TEST(ArchiveZip, CorruptStreamDetected) {
  auto a = Pattern(200000, 10);
  ZipWriter w;
  w.Add("a.bin", a, 8);
  w.Finish();
  // clobber compressed bytes mid-stream (local header is 30 + name)
  for (u64 off = 20000; off < 20016; ++off)
    w.out[30 + 5 + off] ^= 0x5A;
  auto path = WriteTemp(w.out, "zt_badstream.zip");

  auto z = vfs::OpenZipBackend(base::String(path.c_str()));
  ASSERT_TRUE(z);
  base::Vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->Index(es));
  ASSERT_EQ(es.size(), 1u);
  base::Vector<u8> buf(es[0].size);
  EXPECT_EQ(z->ExtractRange(es[0], buf.data(), 0, es[0].size), -1);
  std::remove(path.c_str());
}

// Interop check against an archive made by a real zip tool: index it and
// fully decode every member, relying on the CRC verification for correctness.
// Gated on DELTA_ZIPTEST_FILE so CI does not need external tooling.
TEST(ArchiveZip, ExternalArchive) {
  const char* p = getenv("DELTA_ZIPTEST_FILE");
  if (!p)
    GTEST_SKIP() << "set DELTA_ZIPTEST_FILE to a zip to run";
  auto z = vfs::OpenZipBackend(base::String(p));
  ASSERT_TRUE(z);
  base::Vector<vfs::ArchiveEntry> es;
  ASSERT_TRUE(z->Index(es));
  ASSERT_FALSE(es.empty());
  base::Vector<u8> buf(1 << 20);
  for (const auto& e : es) {
    for (u64 off = 0; off < e.size; off += buf.size()) {
      i64 want = base::Min<u64>(buf.size(), e.size - off);
      ASSERT_EQ(z->ExtractRange(e, buf.data(), off, want), want)
          << e.path.c_str() << " @" << off;
    }
  }
  printf("[ perf ] external archive: %zu members decoded + CRC ok\n",
         es.size());
}

}  // namespace
