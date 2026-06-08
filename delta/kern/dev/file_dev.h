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
// FreeBSD-style stat the PS4 returns (SceKernelStat, 0x78 bytes). We only fill
// the fields games actually look at (mode + size); the rest stays zeroed.
struct SceKernelStat {
  uint32_t st_dev;
  uint32_t st_ino;
  uint16_t st_mode;
  uint16_t st_nlink;
  uint32_t st_uid;
  uint32_t st_gid;
  uint32_t st_rdev;
  int64_t st_atim[2];
  int64_t st_mtim[2];
  int64_t st_ctim[2];
  int64_t st_size;
  int64_t st_blocks;
  uint32_t st_blksize;
  uint32_t st_flags;
  uint32_t st_gen;
  int32_t st_lspare;
  int64_t st_birthtim[2];
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

  // Back this device with an already-opened file (e.g. a virtual VFS stream).
  bool adopt(utl::File &&file);

  // A file mmap is satisfied by sys_mmap's anonymous-alloc + file-content fill
  // (via readAt), not a device-owned region; return -1 silently to take that path.
  uint8_t *map(void *, size_t, uint32_t, uint32_t, size_t) override {
    return reinterpret_cast<uint8_t *>(-1);
  }
  int64_t read(void *buf, size_t n) override;
  int64_t lseek(int64_t off, int whence) override;
  int64_t readAt(void *buf, size_t n, int64_t off) override;
  int fstat(void *stat) override;

private:
  utl::File file_;
  bool open_ = false;
};
} // namespace krnl
