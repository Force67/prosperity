/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// Read-only UFS2 (FreeBSD FFSv2) walker for PS5 *.ffpkg game backups. Only the
// superblock geometry, inodes and directory entries are parsed; file bytes are
// read on demand. Field offsets follow sys/ufs/ffs/fs.h and sys/ufs/ufs/dinode.h.

#include "ufs2_object.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#include <utl/file.h>

namespace vfs {
namespace {
constexpr uint64_t kSblockUfs2 = 65536; // SBLOCK_UFS2
constexpr uint32_t kUfs2Magic = 0x19540119u;
constexpr uint32_t kRootIno = 2;   // ROOTINO
constexpr uint32_t kDinodeSize = 256; // sizeof(struct ufs2_dinode)
constexpr int kNumDirect = 12;     // UFS_NDADDR
constexpr int kNumIndirect = 3;    // UFS_NIADDR
constexpr uint8_t kDtDir = 4;      // DT_DIR

// Superblock field offsets (struct fs).
constexpr size_t kSbIblkno = 0x10;
constexpr size_t kSbBsize = 0x30;
constexpr size_t kSbFsize = 0x34;
constexpr size_t kSbIpg = 0xb8;
constexpr size_t kSbFpg = 0xbc;
constexpr size_t kSbMagic = 0x55c;

template <typename T> T rdle(const uint8_t *p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

bool dbg() {
  static bool on = std::getenv("DELTA_UFS_DBG") != nullptr;
  return on;
}
} // namespace

struct Ufs2Impl {
  utl::File file;
  uint64_t imageSize = 0;
  bool ok = false;

  uint32_t bsize = 0, fsize = 0, iblkno = 0, ipg = 0, fpg = 0;
  uint32_t nindir = 0; // pointers per indirect block = bsize/8

  std::unordered_map<std::string, Ufs2Filesystem::Node> files;

  struct Dinode {
    uint16_t mode = 0;
    uint64_t size = 0;
    int64_t db[kNumDirect]{};
    int64_t ib[kNumIndirect]{};
  };

  explicit Ufs2Impl(const base::String &path) : file(path) {
    if (!file.IsOpen())
      return;
    imageSize = file.GetSize();
    if (!parseSuperblock())
      return;
    walk(kRootIno, std::string(), 0);
    ok = true;
  }

  bool readAt(uint64_t off, void *buf, size_t n) {
    if (off + n > imageSize)
      return false;
    file.Seek(off, utl::seekMode::seek_set);
    return file.Read(buf, n) == n;
  }

  bool parseSuperblock() {
    uint8_t sb[0x600];
    if (!readAt(kSblockUfs2, sb, sizeof(sb)))
      return false;
    if (rdle<uint32_t>(sb + kSbMagic) != kUfs2Magic)
      return false;
    bsize = rdle<uint32_t>(sb + kSbBsize);
    fsize = rdle<uint32_t>(sb + kSbFsize);
    iblkno = rdle<uint32_t>(sb + kSbIblkno);
    ipg = rdle<uint32_t>(sb + kSbIpg);
    fpg = rdle<uint32_t>(sb + kSbFpg);
    // Sanity: block/frag sizes are powers of two and the group geometry is set.
    if (bsize == 0 || fsize == 0 || (bsize & (bsize - 1)) ||
        (fsize & (fsize - 1)) || bsize < fsize || ipg == 0 || fpg == 0)
      return false;
    nindir = bsize / 8;
    if (dbg())
      std::fprintf(stderr,
                   "[ufs2] bsize=%u fsize=%u iblkno=%u ipg=%u fpg=%u\n", bsize,
                   fsize, iblkno, ipg, fpg);
    return true;
  }

  uint64_t inodeOffset(uint32_t ino) const {
    uint64_t cg = ino / ipg;
    uint64_t idx = ino % ipg;
    uint64_t frag = static_cast<uint64_t>(fpg) * cg + iblkno;
    return frag * fsize + idx * kDinodeSize;
  }

  bool readDinode(uint32_t ino, Dinode &out) {
    uint8_t d[kDinodeSize];
    if (!readAt(inodeOffset(ino), d, sizeof(d)))
      return false;
    out.mode = rdle<uint16_t>(d + 0x00);
    out.size = rdle<uint64_t>(d + 0x10);
    for (int i = 0; i < kNumDirect; i++)
      out.db[i] = rdle<int64_t>(d + 0x70 + i * 8);
    for (int i = 0; i < kNumIndirect; i++)
      out.ib[i] = rdle<int64_t>(d + 0xd0 + i * 8);
    return true;
  }

  // Read one indirect pointer: entry `idx` of the pointer block at frag `ptr`.
  int64_t indirect(int64_t ptr, uint64_t idx) {
    if (ptr <= 0)
      return 0;
    int64_t v = 0;
    if (!readAt(static_cast<uint64_t>(ptr) * fsize + idx * 8, &v, 8))
      return 0;
    return v;
  }

  // Logical block number -> frag address of that block (0 == sparse hole).
  int64_t blockAddr(const Dinode &din, uint64_t lbn) {
    if (lbn < kNumDirect)
      return din.db[lbn];
    lbn -= kNumDirect;
    if (lbn < nindir)
      return indirect(din.ib[0], lbn);
    lbn -= nindir;
    if (lbn < static_cast<uint64_t>(nindir) * nindir) {
      int64_t mid = indirect(din.ib[1], lbn / nindir);
      return indirect(mid, lbn % nindir);
    }
    lbn -= static_cast<uint64_t>(nindir) * nindir;
    int64_t l1 = indirect(din.ib[2], lbn / (static_cast<uint64_t>(nindir) * nindir));
    int64_t l2 = indirect(l1, (lbn / nindir) % nindir);
    return indirect(l2, lbn % nindir);
  }

  int64_t readInode(const Dinode &din, void *buf, int64_t off, int64_t len) {
    if (off < 0 || len < 0)
      return -1;
    uint64_t size = din.size;
    if (static_cast<uint64_t>(off) >= size)
      return 0;
    uint64_t want = std::min<uint64_t>(len, size - off);
    auto *dst = static_cast<uint8_t *>(buf);
    uint64_t done = 0;
    while (done < want) {
      uint64_t pos = off + done;
      uint64_t lbn = pos / bsize;
      uint64_t boff = pos % bsize;
      uint64_t chunk = std::min<uint64_t>(bsize - boff, want - done);
      int64_t frag = blockAddr(din, lbn);
      if (frag <= 0) {
        std::memset(dst + done, 0, chunk); // hole
      } else if (!readAt(static_cast<uint64_t>(frag) * fsize + boff, dst + done,
                         chunk)) {
        return done > 0 ? static_cast<int64_t>(done) : -1;
      }
      done += chunk;
    }
    return static_cast<int64_t>(done);
  }

  void walk(uint32_t dirIno, const std::string &prefix, int depth) {
    if (depth > 64 || files.size() > 200000)
      return;
    Dinode din;
    if (!readDinode(dirIno, din) || din.size == 0 || din.size > (64u << 20))
      return;
    std::vector<uint8_t> data(static_cast<size_t>(din.size));
    if (readInode(din, data.data(), 0, data.size()) !=
        static_cast<int64_t>(data.size()))
      return;

    size_t p = 0;
    while (p + 8 <= data.size()) {
      uint32_t ino = rdle<uint32_t>(&data[p]);
      uint16_t reclen = rdle<uint16_t>(&data[p + 4]);
      uint8_t type = data[p + 6];
      uint8_t namlen = data[p + 7];
      if (reclen == 0 || p + reclen > data.size())
        break;
      if (ino != 0 && namlen != 0 && p + 8 + namlen <= data.size()) {
        std::string name(reinterpret_cast<char *>(&data[p + 8]), namlen);
        if (name != "." && name != "..") {
          std::string full = prefix + "/" + name;
          if (type == kDtDir) {
            walk(ino, full, depth + 1);
          } else {
            Dinode fd;
            if (readDinode(ino, fd))
              files[full] = {fd.size, ino};
          }
        }
      }
      p += reclen;
    }
  }
};

Ufs2Filesystem::Ufs2Filesystem(const base::String &path)
    : impl_(std::make_unique<Ufs2Impl>(path)) {}
Ufs2Filesystem::~Ufs2Filesystem() = default;

bool Ufs2Filesystem::valid() const { return impl_ && impl_->ok; }

const Ufs2Filesystem::Node *Ufs2Filesystem::find(const char *relPath) const {
  if (!impl_ || !relPath)
    return nullptr;
  std::string key = relPath[0] == '/' ? relPath : std::string("/") + relPath;
  auto it = impl_->files.find(key);
  return it == impl_->files.end() ? nullptr : &it->second;
}

int64_t Ufs2Filesystem::read(const Node &node, void *buf, int64_t off,
                             int64_t len) {
  if (!impl_)
    return -1;
  Ufs2Impl::Dinode din;
  if (!impl_->readDinode(node.inode, din))
    return -1;
  return impl_->readInode(din, buf, off, len);
}

void Ufs2Filesystem::paths(std::vector<std::string> &out) const {
  if (!impl_)
    return;
  out.reserve(out.size() + impl_->files.size());
  for (const auto &kv : impl_->files)
    out.push_back(kv.first);
}
} // namespace vfs
