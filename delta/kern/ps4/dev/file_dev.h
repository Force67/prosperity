#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdint>

#include <utl/file.h>

#include "device.h"

namespace krnl {
// FreeBSD-style stat the PS4 returns (SceKernelStat, 0x78 / 120 bytes).
// The kernel fills this from vn_stat and copies it out. Without privilege 0x2AC
// the kernel zeroes st_dev, st_ino, st_nlink, st_uid, st_gid, st_rdev, st_flags,
// st_gen, st_lspare and st_birthtim. We zero the whole struct first, so every
// non-filled field is always 0.
struct SceKernelStat {
  uint32_t st_dev;          // +0x00
  uint32_t st_ino;          // +0x04
  uint16_t st_mode;         // +0x08
  uint16_t st_nlink;        // +0x0A
  uint32_t st_uid;          // +0x0C
  uint32_t st_gid;          // +0x10
  uint32_t st_rdev;         // +0x14
  int64_t st_atim[2];       // +0x18 atime sec/nsec
  int64_t st_mtim[2];       // +0x28 mtime sec/nsec
  int64_t st_ctim[2];       // +0x38 ctime sec/nsec
  int64_t st_size;          // +0x48
  int64_t st_blocks;        // +0x50
  uint32_t st_blksize;      // +0x58
  uint32_t st_flags;        // +0x5C
  uint32_t st_gen;          // +0x60
  int32_t st_lspare;        // +0x64
  int64_t st_birthtim[2];   // +0x68 btime sec/nsec
};
static_assert(sizeof(SceKernelStat) == 0x78, "SceKernelStat layout");

constexpr uint16_t kSceFileModeReg = 0x8000;  // S_IFREG
constexpr uint16_t kSceFileModeDir = 0x4000;  // S_IFDIR

void fillStat(SceKernelStat &out, uint16_t mode, int64_t size);

// A regular host-backed file exposed to the guest through the object table.
class fileDevice : public device {
public:
  explicit fileDevice(proc *p);

  // Open the resolved host path. Returns false if it doesn't exist.
  bool open(const base::String &hostPath, uint32_t flags);

  // Open a resolved host path for writing (savedata). `create` creates the file
  // if absent; `truncate` discards existing contents. The file is opened
  // read+write so the guest can read back what it wrote. Returns false on
  // failure (e.g. open-existing with no create and the file is absent).
  bool openWritable(const base::String &hostPath, bool create, bool truncate);

  // Back this device with an already-opened file (e.g. a virtual VFS stream).
  bool adopt(utl::File &&file);

  bool isRegularFile() const override { return true; }

  // SOTTR's TAFS loader issues manifest reads with an uninitialised (garbage)
  // file offset, so the header at off 0 never loads. In sequential mode the
  // device ignores the bogus absolute offsets and serves reads in order from an
  // internal cursor, which reads the whole manifest correctly. Scoped to
  // .manifest.bin opens (set in sys_open) so it can't affect random-access asset
  // reads. Gated by DELTA_MANIFEST_SEQ.
  void setSeqMode() { seq_ = true; }

  // A file mmap is satisfied by sys_mmap's anonymous-alloc + file-content fill
  // (via readAt), not a device-owned region; return -1 silently to take that path.
  uint8_t *map(void *, size_t, uint32_t, uint32_t, size_t) override {
    return reinterpret_cast<uint8_t *>(-1);
  }
  int64_t read(void *buf, size_t n) override;
  int64_t write(const void *buf, size_t n) override;
  int64_t lseek(int64_t off, int whence) override;
  int64_t readAt(void *buf, size_t n, int64_t off) override;
  int fstat(void *stat) override;

private:
  utl::File file_;
  bool open_ = false;
  bool writable_ = false;  // opened for writing (savedata)
  bool seq_ = false;       // manifest sequential-read mode
  uint64_t seqPos_ = 0;    // internal read cursor for seq_ mode
};
} // namespace krnl
