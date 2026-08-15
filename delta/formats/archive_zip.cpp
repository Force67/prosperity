
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// ZIP container backend: parses the central directory (ZIP64-aware) and decodes
// stored/deflate members on demand. Archives can be tens of GB with multi-GB
// members, so nothing is ever extracted; deflate reads go through persistent
// per-entry decoder cursors that resume where the previous read left off.

#include "archive_backend.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include <zlib.h>

#include <logger/logger.h>
#include <utl/file.h>

namespace vfs {
namespace {

inline u16 rd16(const u8 *p) {
  return static_cast<u16>(p[0] | (p[1] << 8));
}
inline u32 rd32(const u8 *p) {
  return static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8) |
         (static_cast<u32>(p[2]) << 16) | (static_cast<u32>(p[3]) << 24);
}
inline u64 rd64(const u8 *p) {
  u64 v = 0;
  for (int i = 0; i < 8; ++i)
    v |= static_cast<u64>(p[i]) << (8 * i);
  return v;
}

constexpr u32 kEocdSig = 0x06054b50;
constexpr u32 kEocd64LocSig = 0x07064b50;
constexpr u32 kEocd64Sig = 0x06064b50;
constexpr u32 kCenSig = 0x02014b50;
constexpr u32 kLocSig = 0x04034b50;

constexpr u16 kMethodStore = 0;
constexpr u16 kMethodDeflate = 8;

class ZipBackend final : public ArchiveBackend {
public:
  explicit ZipBackend(const base::String &path)
      : file_(path, utl::fileMode::read) {}

  ~ZipBackend() override {
    for (auto &s : sess_)
      drop(s);
  }

  // Cheap sniff for openZipBackend: locate the EOCD (and the ZIP64 record when
  // present) so index() can walk the central directory directly.
  bool sniff() {
    if (!file_.Exists() || !file_.IsOpen())
      return false;
    fileSize_ = file_.GetSize();
    if (fileSize_ < 22)
      return false;

    // The EOCD floats up to 64 KB before EOF (trailing archive comment), so
    // scan the tail backwards for its signature instead of assuming a fixed
    // position, and require the comment length to run exactly to EOF so a
    // stray signature inside the comment cannot win.
    u64 win = std::min<u64>(fileSize_, 22 + 0xFFFF);
    std::vector<u8> tail(win);
    if (!pread(fileSize_ - win, tail.data(), win))
      return false;
    for (i64 i = static_cast<i64>(win) - 22; i >= 0; --i) {
      const u8 *p = tail.data() + i;
      if (rd32(p) != kEocdSig || i + 22 + rd16(p + 20) != static_cast<i64>(win))
        continue;
      cdCount_ = rd16(p + 10);
      cdSize_ = rd32(p + 12);
      cdOff_ = rd32(p + 16);

      // ZIP64: the EOCD64 locator sits directly before the EOCD and points at
      // the real record. Its 64-bit fields supersede the (possibly saturated)
      // 32-bit ones.
      u64 eocdPos = fileSize_ - win + i;
      if (eocdPos >= 20) {
        u8 loc[20];
        if (pread(eocdPos - 20, loc, 20) && rd32(loc) == kEocd64LocSig) {
          u64 rec = rd64(loc + 8);
          u8 h[56];
          if (rec + 56 <= fileSize_ && pread(rec, h, 56) &&
              rd32(h) == kEocd64Sig) {
            cdCount_ = rd64(h + 32);
            cdSize_ = rd64(h + 40);
            cdOff_ = rd64(h + 48);
          }
        }
      }
      return cdOff_ + cdSize_ <= fileSize_;
    }
    return false;
  }

  bool index(std::vector<ArchiveEntry> &out) override {
    std::lock_guard<std::mutex> lk(mtx_);
    if (cdSize_ > (1ull << 30)) // a sane central directory is a few MB
      return false;
    std::vector<u8> cd(cdSize_);
    if (!pread(cdOff_, cd.data(), cdSize_))
      return false;

    u64 p = 0, skipped = 0;
    for (u64 i = 0; i < cdCount_; ++i) {
      if (p + 46 > cdSize_ || rd32(&cd[p]) != kCenSig)
        return false;
      const u8 *h = cd.data() + p;
      u16 flags = rd16(h + 8);
      u16 method = rd16(h + 10);
      u32 crc = rd32(h + 16);
      u64 comp = rd32(h + 20);
      u64 uncomp = rd32(h + 24);
      u16 nameLen = rd16(h + 28);
      u16 extraLen = rd16(h + 30);
      u16 cmtLen = rd16(h + 32);
      u32 extAttr = rd32(h + 38);
      u64 lho = rd32(h + 42);
      if (p + 46 + nameLen + extraLen + cmtLen > cdSize_)
        return false;
      p += 46u + nameLen + extraLen + cmtLen;

      // ZIP64 extra 0x0001 carries a u64 for each 32-bit field that saturated,
      // in the fixed order size / packed size / local header offset.
      const u8 *xp = h + 46 + nameLen;
      const u8 *xe = xp + extraLen;
      while (xp + 4 <= xe) {
        u16 id = rd16(xp), sz = rd16(xp + 2);
        const u8 *d = xp + 4;
        if (d + sz > xe)
          break;
        if (id == 0x0001) {
          u16 left = sz;
          if (uncomp == 0xFFFFFFFF && left >= 8) {
            uncomp = rd64(d);
            d += 8;
            left -= 8;
          }
          if (comp == 0xFFFFFFFF && left >= 8) {
            comp = rd64(d);
            d += 8;
            left -= 8;
          }
          if (lho == 0xFFFFFFFF && left >= 8)
            lho = rd64(d);
        }
        xp += 4u + sz;
      }

      // Bit 11 flags a UTF-8 name; either way the bytes pass through as-is
      // (legacy CP437 names are not transcoded).
      std::string nm(reinterpret_cast<const char *>(h + 46), nameLen);
      for (auto &c : nm)
        if (c == '\\')
          c = '/';
      while (!nm.empty() && nm.front() == '/')
        nm.erase(nm.begin());

      bool isDir = nm.empty() || nm.back() == '/' || (extAttr & 0x10) ||
                   ((extAttr >> 16) & 0xF000) == 0x4000;
      if (isDir)
        continue;
      if (flags & 1) {
        LOG_WARNING("zip: skipping encrypted entry {}", nm.c_str());
        ++skipped;
        continue;
      }
      if (method != kMethodStore && method != kMethodDeflate) {
        LOG_WARNING("zip: skipping {} (unsupported method {})", nm.c_str(),
                    method);
        ++skipped;
        continue;
      }

      ArchiveEntry e;
      e.path = std::move(nm);
      e.size = uncomp;
      e.packedSize = comp;
      e.dataOffset = 0; // resolved lazily from the local header
      e.extra = lho;
      e.method = method;
      e.crc = crc;
      out.push_back(std::move(e));
    }
    LOG_INFO("zip: indexed {} files ({} skipped)", out.size(), skipped);
    return true;
  }

  i64 extractRange(const ArchiveEntry &entry, void *buf, i64 off,
                   i64 len) override {
    if (!buf || off < 0 || len < 0)
      return -1;
    if (static_cast<u64>(off) >= entry.size)
      return 0;
    u64 want = std::min<u64>(len, entry.size - static_cast<u64>(off));

    std::lock_guard<std::mutex> lk(mtx_);
    i64 ds = dataStart(entry);
    if (ds < 0)
      return -1;

    if (entry.method == kMethodStore)
      return pread(static_cast<u64>(ds) + off, buf, want)
                 ? static_cast<i64>(want)
                 : -1;

    Session *s = acquire(entry, static_cast<u64>(ds), static_cast<u64>(off));
    if (!s)
      return -1;

    // Discard-decode forward to `off`. Sessions always start at member byte 0,
    // so the running CRC stays valid for the end-of-stream check.
    if (skip_.empty())
      skip_.resize(256 << 10);
    while (s->outPos < static_cast<u64>(off)) {
      u64 chunk = std::min<u64>(skip_.size(), off - s->outPos);
      if (inflateTo(*s, skip_.data(), chunk) != static_cast<i64>(chunk)) {
        drop(*s);
        return -1;
      }
    }

    if (inflateTo(*s, static_cast<u8 *>(buf), want) != static_cast<i64>(want)) {
      drop(*s);
      return -1;
    }

    if (s->outPos == entry.size) {
      // The member is fully decoded; the stream must terminate here and the
      // CRC must match. (A ranged read that never reaches the end cannot be
      // CRC'd, that is inherent.)
      if (!s->ended) {
        u8 dummy;
        s->zs.next_out = &dummy;
        s->zs.avail_out = 1;
        if (s->zs.avail_in == 0 && s->compLeft > 0)
          (void)fill(*s);
        s->ended = ::inflate(&s->zs, Z_NO_FLUSH) == Z_STREAM_END;
      }
      bool ok = s->ended && s->crc == entry.crc;
      if (!ok)
        LOG_ERROR("zip: {} failed end-of-stream check (crc {:08x} want {:08x})",
                  entry.path.c_str(), s->crc, entry.crc);
      drop(*s);
      if (!ok)
        return -1;
    }
    return static_cast<i64>(want);
  }

  const char *name() const override { return "zip"; }

private:
  struct Session {
    z_stream zs{};
    std::vector<u8> in;
    u64 key = 0;      // entry identity: local header offset
    u64 filePos = 0;  // next compressed byte, absolute in the archive
    u64 compLeft = 0;
    u64 outPos = 0;   // uncompressed bytes produced so far
    u64 stamp = 0;
    u32 crc = 0;
    bool ended = false;
    bool open = false;
  };

  bool pread(u64 off, void *dst, u64 len) {
    file_.Seek(off, utl::seekMode::seek_set);
    return file_.Read(dst, len) == len;
  }

  // The central directory stores the LOCAL header offset; the data begins after
  // that header, whose name/extra lengths differ from the central copy. Resolve
  // once per entry and cache, keyed on the header offset so it works for entry
  // copies rehydrated from an index cache.
  i64 dataStart(const ArchiveEntry &e) {
    if (e.dataOffset)
      return static_cast<i64>(e.dataOffset);
    auto it = dataOff_.find(e.extra);
    if (it != dataOff_.end())
      return static_cast<i64>(it->second);
    u8 lh[30];
    if (!pread(e.extra, lh, sizeof(lh)) || rd32(lh) != kLocSig)
      return -1;
    u64 off = e.extra + 30 + rd16(lh + 26) + rd16(lh + 28);
    dataOff_.emplace(e.extra, off);
    return static_cast<i64>(off);
  }

  bool fill(Session &s) {
    u64 n = std::min<u64>(s.in.size(), s.compLeft);
    if (n == 0 || !pread(s.filePos, s.in.data(), n))
      return false;
    s.zs.next_in = s.in.data();
    s.zs.avail_in = static_cast<uInt>(n);
    s.filePos += n;
    s.compLeft -= n;
    return true;
  }

  i64 inflateTo(Session &s, u8 *dst, u64 want) {
    u64 done = 0;
    while (done < want && !s.ended) {
      if (s.zs.avail_in == 0 && s.compLeft > 0 && !fill(s))
        return -1;
      s.zs.next_out = dst + done;
      uInt avail = static_cast<uInt>(std::min<u64>(want - done, 1u << 30));
      s.zs.avail_out = avail;
      int rc = ::inflate(&s.zs, Z_NO_FLUSH);
      u64 got = avail - s.zs.avail_out;
      s.crc = crc32(s.crc, dst + done, static_cast<uInt>(got));
      s.outPos += got;
      done += got;
      if (rc == Z_STREAM_END) {
        s.ended = true;
        break;
      }
      if (rc != Z_OK)
        return -1;
      if (got == 0 && s.zs.avail_in == 0 && s.compLeft == 0)
        return -1; // truncated stream
    }
    return static_cast<i64>(done);
  }

  // Find (or create) the decoder cursor for this entry, positioned at or before
  // wantOff. A forward read resumes the existing stream; a backward seek
  // restarts it from byte 0. At most kSessions live at once, evicted LRU.
  Session *acquire(const ArchiveEntry &e, u64 dataOff, u64 wantOff) {
    Session *s = nullptr;
    for (auto &c : sess_)
      if (c.open && c.key == e.extra) {
        s = &c;
        break;
      }
    if (s && (s->outPos > wantOff || s->ended)) {
      drop(*s);
      s = nullptr;
    }
    if (!s) {
      Session *victim = &sess_[0];
      for (auto &c : sess_) {
        if (!c.open) {
          victim = &c;
          break;
        }
        if (c.stamp < victim->stamp)
          victim = &c;
      }
      drop(*victim);
      s = victim;
      std::memset(&s->zs, 0, sizeof(s->zs));
      if (inflateInit2(&s->zs, -15) != Z_OK) // raw deflate, no zlib header
        return nullptr;
      s->in.resize(256 << 10);
      s->key = e.extra;
      s->filePos = dataOff;
      s->compLeft = e.packedSize;
      s->outPos = 0;
      s->crc = crc32(0, Z_NULL, 0);
      s->ended = false;
      s->open = true;
    }
    s->stamp = ++stamp_;
    return s;
  }

  void drop(Session &s) {
    if (s.open)
      inflateEnd(&s.zs);
    s.open = false;
  }

  utl::File file_;
  std::mutex mtx_;
  u64 fileSize_ = 0;
  u64 cdOff_ = 0, cdSize_ = 0, cdCount_ = 0;
  std::unordered_map<u64, u64> dataOff_;
  std::vector<u8> skip_;
  Session sess_[4];
  u64 stamp_ = 0;
};

} // namespace

std::unique_ptr<ArchiveBackend> openZipBackend(const base::String &path) {
  auto b = std::make_unique<ZipBackend>(path);
  if (!b->sniff())
    return nullptr;
  return b;
}

} // namespace vfs
