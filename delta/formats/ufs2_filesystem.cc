/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */
// Read-only UFS2 (FreeBSD FFSv2) walker for PS5 *.ffpkg game backups. Only the
// superblock geometry, inodes and directory entries are parsed; file bytes are
// read on demand. Field offsets follow sys/ufs/ffs/fs.h and
// sys/ufs/ufs/dinode.h.

#include "formats/ufs2_filesystem.h"
#include "base/arch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/logging.h"
#include "base/math/value_bounds.h"
#include "base/memory/unique_pointer.h"
#include "base/strings/xstring.h"
#include "io/file.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kUfsDbg, "DELTA_UFS_DBG", false);
}  // namespace

namespace formats {
namespace {
constexpr u64 kSblockUfs2 = 65536;  // SBLOCK_UFS2
constexpr u32 kUfs2Magic = 0x19540119u;
constexpr u32 kRootIno = 2;       // ROOTINO
constexpr u32 kDinodeSize = 256;  // sizeof(struct ufs2_dinode)
constexpr int kNumDirect = 12;    // UFS_NDADDR
constexpr int kNumIndirect = 3;   // UFS_NIADDR
constexpr u8 kDtDir = 4;          // DT_DIR

// Superblock field offsets (struct fs).
constexpr size_t kSbIblkno = 0x10;
constexpr size_t kSbBsize = 0x30;
constexpr size_t kSbFsize = 0x34;
constexpr size_t kSbIpg = 0xb8;
constexpr size_t kSbFpg = 0xbc;
constexpr size_t kSbMagic = 0x55c;

template <typename T>
T Rdle(const u8* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

bool Dbg() {
  return kUfsDbg;
}
}  // namespace

struct Ufs2Impl {
  io::File file;
  u64 image_size = 0;
  bool ok = false;

  u32 bsize = 0, fsize = 0, iblkno = 0, ipg = 0, fpg = 0;
  u32 nindir = 0;  // pointers per indirect block = bsize/8

  base::HashMap<base::String, Ufs2Filesystem::Node> files;

  struct Dinode {
    u16 mode = 0;
    u64 size = 0;
    i64 db[kNumDirect]{};
    i64 ib[kNumIndirect]{};
  };

  explicit Ufs2Impl(const base::String& path) : file(path) {
    if (!file.IsOpen())
      return;
    image_size = file.GetSize();
    if (!ParseSuperblock())
      return;
    Walk(kRootIno, base::String(), 0);
    ok = true;
  }

  bool ReadAt(u64 off, void* buf, size_t n) {
    if (off + n > image_size)
      return false;
    file.Seek(off, io::SeekMode::kSeekSet);
    return file.Read(buf, n) == n;
  }

  bool ParseSuperblock() {
    u8 sb[0x600];
    if (!ReadAt(kSblockUfs2, sb, sizeof(sb)))
      return false;
    if (Rdle<u32>(sb + kSbMagic) != kUfs2Magic)
      return false;
    bsize = Rdle<u32>(sb + kSbBsize);
    fsize = Rdle<u32>(sb + kSbFsize);
    iblkno = Rdle<u32>(sb + kSbIblkno);
    ipg = Rdle<u32>(sb + kSbIpg);
    fpg = Rdle<u32>(sb + kSbFpg);
    // Sanity: block/frag sizes are powers of two and the group geometry is set.
    if (bsize == 0 || fsize == 0 || (bsize & (bsize - 1)) ||
        (fsize & (fsize - 1)) || bsize < fsize || ipg == 0 || fpg == 0)
      return false;
    nindir = bsize / 8;
    if (Dbg())
      BASE_LOGI("ufs2", "bsize={} fsize={} iblkno={} ipg={} fpg={}", bsize,
                fsize, iblkno, ipg, fpg);
    return true;
  }

  u64 InodeOffset(u32 ino) const {
    u64 cg = ino / ipg;
    u64 idx = ino % ipg;
    u64 frag = static_cast<u64>(fpg) * cg + iblkno;
    return frag * fsize + idx * kDinodeSize;
  }

  bool ReadDinode(u32 ino, Dinode& out) {
    u8 d[kDinodeSize];
    if (!ReadAt(InodeOffset(ino), d, sizeof(d)))
      return false;
    out.mode = Rdle<u16>(d + 0x00);
    out.size = Rdle<u64>(d + 0x10);
    for (int i = 0; i < kNumDirect; i++)
      out.db[i] = Rdle<i64>(d + 0x70 + i * 8);
    for (int i = 0; i < kNumIndirect; i++)
      out.ib[i] = Rdle<i64>(d + 0xd0 + i * 8);
    return true;
  }

  // Read one indirect pointer: entry `idx` of the pointer block at frag `ptr`.
  i64 Indirect(i64 ptr, u64 idx) {
    if (ptr <= 0)
      return 0;
    i64 v = 0;
    if (!ReadAt(static_cast<u64>(ptr) * fsize + idx * 8, &v, 8))
      return 0;
    return v;
  }

  // Logical block number -> frag address of that block (0 == sparse hole).
  i64 BlockAddr(const Dinode& din, u64 lbn) {
    if (lbn < kNumDirect)
      return din.db[lbn];
    lbn -= kNumDirect;
    if (lbn < nindir)
      return Indirect(din.ib[0], lbn);
    lbn -= nindir;
    if (lbn < static_cast<u64>(nindir) * nindir) {
      i64 mid = Indirect(din.ib[1], lbn / nindir);
      return Indirect(mid, lbn % nindir);
    }
    lbn -= static_cast<u64>(nindir) * nindir;
    i64 l1 = Indirect(din.ib[2], lbn / (static_cast<u64>(nindir) * nindir));
    i64 l2 = Indirect(l1, (lbn / nindir) % nindir);
    return Indirect(l2, lbn % nindir);
  }

  i64 ReadInode(const Dinode& din, void* buf, i64 off, i64 len) {
    if (off < 0 || len < 0)
      return -1;
    u64 size = din.size;
    if (static_cast<u64>(off) >= size)
      return 0;
    u64 want = base::Min<u64>(len, size - off);
    auto* dst = static_cast<u8*>(buf);
    u64 done = 0;
    while (done < want) {
      u64 pos = off + done;
      u64 lbn = pos / bsize;
      u64 boff = pos % bsize;
      u64 chunk = base::Min<u64>(bsize - boff, want - done);
      i64 frag = BlockAddr(din, lbn);
      if (frag <= 0) {
        std::memset(dst + done, 0, chunk);  // hole
      } else if (!ReadAt(static_cast<u64>(frag) * fsize + boff, dst + done,
                         chunk)) {
        return done > 0 ? static_cast<i64>(done) : -1;
      }
      done += chunk;
    }
    return static_cast<i64>(done);
  }

  void Walk(u32 dir_ino, const base::String& prefix, int depth) {
    if (depth > 64 || files.size() > 200000)
      return;
    Dinode din;
    if (!ReadDinode(dir_ino, din) || din.size == 0 || din.size > (64u << 20))
      return;
    base::Vector<u8> data(static_cast<size_t>(din.size));
    if (ReadInode(din, data.data(), 0, data.size()) !=
        static_cast<i64>(data.size()))
      return;

    size_t p = 0;
    while (p + 8 <= data.size()) {
      u32 ino = Rdle<u32>(&data[p]);
      u16 reclen = Rdle<u16>(&data[p + 4]);
      u8 type = data[p + 6];
      u8 namlen = data[p + 7];
      if (reclen == 0 || p + reclen > data.size())
        break;
      if (ino != 0 && namlen != 0 && p + 8 + namlen <= data.size()) {
        base::String name(reinterpret_cast<char*>(&data[p + 8]), namlen);
        if (name != "." && name != "..") {
          base::String full = prefix + "/" + name;
          if (type == kDtDir) {
            Walk(ino, full, depth + 1);
          } else {
            Dinode fd;
            if (ReadDinode(ino, fd))
              files[full] = {fd.size, ino};
          }
        }
      }
      p += reclen;
    }
  }
};

Ufs2Filesystem::Ufs2Filesystem(const base::String& path)
    : impl_(base::MakeUnique<Ufs2Impl>(path)) {}
Ufs2Filesystem::~Ufs2Filesystem() = default;

bool Ufs2Filesystem::Valid() const {
  return impl_ && impl_->ok;
}

const Ufs2Filesystem::Node* Ufs2Filesystem::Find(const char* rel_path) const {
  if (!impl_ || !rel_path)
    return nullptr;
  base::String key =
      rel_path[0] == '/' ? rel_path : base::String("/") + rel_path;
  auto it = impl_->files.find(key);
  return it == impl_->files.end() ? nullptr : &it->second;
}

i64 Ufs2Filesystem::Read(const Node& node, void* buf, i64 off, i64 len) {
  if (!impl_)
    return -1;
  Ufs2Impl::Dinode din;
  if (!impl_->ReadDinode(node.inode, din))
    return -1;
  return impl_->ReadInode(din, buf, off, len);
}

void Ufs2Filesystem::Paths(base::Vector<base::String>& out) const {
  if (!impl_)
    return;
  out.reserve(out.size() + impl_->files.size());
  for (const auto& kv : impl_->files)
    out.push_back(kv.first);
}
}  // namespace formats
