#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"

#include "io/file.h"

#include "base/strings/xstring.h"
#include "kern/ps4/dev/device.h"

namespace kern {
// FreeBSD-style stat the PS4 returns (SceKernelStat, 0x78 bytes), filled from
// vn_stat. Without privilege 0x2AC the kernel zeroes
// st_dev/ino/nlink/uid/gid/rdev/flags/gen/ lspare/birthtim; we zero the whole
// struct first, so unfilled fields are always 0.
struct SceKernelStat {
  u32 st_dev;          // +0x00
  u32 st_ino;          // +0x04
  u16 st_mode;         // +0x08
  u16 st_nlink;        // +0x0A
  u32 st_uid;          // +0x0C
  u32 st_gid;          // +0x10
  u32 st_rdev;         // +0x14
  i64 st_atim[2];      // +0x18 atime sec/nsec
  i64 st_mtim[2];      // +0x28 mtime sec/nsec
  i64 st_ctim[2];      // +0x38 ctime sec/nsec
  i64 st_size;         // +0x48
  i64 st_blocks;       // +0x50
  u32 st_blksize;      // +0x58
  u32 st_flags;        // +0x5C
  u32 st_gen;          // +0x60
  i32 st_lspare;       // +0x64
  i64 st_birthtim[2];  // +0x68 btime sec/nsec
};
static_assert(sizeof(SceKernelStat) == 0x78, "SceKernelStat layout");

constexpr u16 kSceFileModeReg = 0x8000;  // S_IFREG
constexpr u16 kSceFileModeDir = 0x4000;  // S_IFDIR

void FillStat(SceKernelStat& out, u16 mode, i64 size);

// A regular host-backed file exposed to the guest through the object table.
class FileDevice : public Device {
 public:
  explicit FileDevice(ObjectTable& objects);

  // Open the resolved host path. Returns false if it doesn't exist.
  bool Open(const base::String& host_path, u32 flags);

  // Open a resolved host path for writing (savedata). `create` creates the file
  // if absent; `truncate` discards existing contents. The file is opened
  // read+write so the guest can read back what it wrote. Returns false on
  // failure (e.g. open-existing with no create and the file is absent).
  bool OpenWritable(const base::String& host_path, bool create, bool truncate);

  // Back this device with an already-opened file (e.g. a virtual VFS stream).
  bool Adopt(io::File&& file);

  bool IsRegularFile() const override { return true; }

  // SOTTR's TAFS loader issues manifest reads with a garbage offset, so the
  // header at 0 never loads. Sequential mode ignores the bogus absolute offsets
  // and serves reads in order from an internal cursor. Scoped to .manifest.bin
  // opens (set in sys_open); gated by DELTA_MANIFEST_SEQ.
  void SetSeqMode() { seq_ = true; }

  // A file mmap is satisfied by sys_mmap's anonymous-alloc + file-content fill
  // (via ReadAt), not a device-owned region; return -1 silently to take that
  // path.
  u8* Map(void*, size_t, u32, u32, size_t) override {
    return reinterpret_cast<u8*>(-1);
  }
  i64 Read(void* buf, size_t n) override;
  i64 Write(const void* buf, size_t n) override;
  i64 Lseek(i64 off, int whence) override;
  i64 ReadAt(void* buf, size_t n, i64 off) override;
  int Fstat(void* stat) override;

 private:
  io::File file_;
  bool open_ = false;
  bool writable_ = false;  // opened for writing (savedata)
  bool seq_ = false;       // manifest sequential-read mode
  u64 seqPos_ = 0;         // internal read cursor for seq_ mode
};
}  // namespace kern
