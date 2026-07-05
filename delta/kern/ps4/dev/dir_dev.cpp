/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstring>

#include "dir_dev.h"
#include "file_dev.h"

namespace krnl {
// FreeBSD dirent (PS4 is FreeBSD 9, pre-ino64): 8-byte header + name, each
// record padded to an 8-byte boundary.
struct fbsd_dirent {
  uint32_t d_fileno;
  uint16_t d_reclen;
  uint8_t d_type;
  uint8_t d_namlen;
  char d_name[256];
};
enum { kDtDir = 4, kDtReg = 8 };

dirDevice::dirDevice(proc *p, std::vector<vfs::DirEntry> &&entries)
    : device(p), entries_(std::move(entries)) {}

int64_t dirDevice::getdents(void *buf, size_t len) {
  auto *p = static_cast<uint8_t *>(buf);
  size_t used = 0;
  while (cursor_ < entries_.size()) {
    const auto &e = entries_[cursor_];
    uint8_t namlen =
        static_cast<uint8_t>(e.name.size() > 255 ? 255 : e.name.size());
    uint16_t reclen = static_cast<uint16_t>((8 + namlen + 1 + 7) & ~7);
    if (used + reclen > len)
      break;
    auto *d = reinterpret_cast<fbsd_dirent *>(p + used);
    std::memset(d, 0, reclen);
    d->d_fileno = static_cast<uint32_t>(cursor_ + 1);  // must be nonzero
    d->d_reclen = reclen;
    d->d_type = e.isDir ? kDtDir : kDtReg;
    d->d_namlen = namlen;
    std::memcpy(d->d_name, e.name.data(), namlen);
    used += reclen;
    cursor_++;
  }
  return static_cast<int64_t>(used);  // 0 once exhausted
}

int dirDevice::fstat(void *stat) {
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), kSceFileModeDir, 0);
  return 0;
}

int64_t dirDevice::read(void *, size_t) { return -SysError::eISDIR; }
}  // namespace krnl
