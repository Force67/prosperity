/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "file_dev.h"

namespace krnl {
void fillStat(SceKernelStat &out, uint16_t mode, int64_t size) {
  std::memset(&out, 0, sizeof(out));
  out.st_mode = mode;
  out.st_size = size;
  out.st_nlink = 1;
  out.st_blksize = 0x4000;
  out.st_blocks = (size + 0x3FFF) / 0x4000;
}

fileDevice::fileDevice(proc *p) : device(p) {}

bool fileDevice::open(const base::String &hostPath, uint32_t /*flags*/) {
  // Read-only for now: the disc image is immutable.
  utl::File tmp(hostPath, utl::fileMode::read);
  // Exists() only means a PhysFile object was constructed; IsOpen() means the
  // underlying fopen actually succeeded. Without the IsOpen() check a missing
  // file would register an fd whose later read fread()s a null FILE* and faults.
  if (!tmp.Exists() || !tmp.IsOpen())
    return false;
  file_.Reset(tmp.GetBase());
  open_ = true;
  return true;
}

bool fileDevice::adopt(utl::File &&file) {
  if (!file.Exists())
    return false;
  file_.Reset(file.GetBase());
  open_ = true;
  return true;
}

int64_t fileDevice::read(void *buf, size_t n) {
  if (!open_)
    return -SysError::eBADF;
  if (seq_)
    file_.Seek(static_cast<int64_t>(seqPos_), utl::seekMode::seek_set);
  int64_t r = static_cast<int64_t>(file_.Read(buf, n));
  if (seq_ && r > 0)
    seqPos_ += static_cast<uint64_t>(r);
  if (r >= 16 && std::getenv("DELTA_FILEREAD_TRACE")) {
    auto *b = static_cast<const uint8_t *>(buf);
    // TAFS manifest? dump the entry_count at +0x0c the game will read back.
    if (b[0] == 'T' && b[1] == 'A' && b[2] == 'F' && b[3] == 'S') {
      uint32_t cc = b[0x0c] | (b[0x0d] << 8) | (b[0x0e] << 16) | (b[0x0f] << 24);
      std::fprintf(stderr, "[fread TAFS] n=%zu -> %lld  entry_count@0xc=%u\n",
                   n, (long long)r, cc);
    }
  }
  return r;
}

int64_t fileDevice::lseek(int64_t off, int whence) {
  if (!open_)
    return -SysError::eBADF;
  // Sequential mode (manifest): ignore the engine's bogus absolute seeks; only a
  // SEEK_SET 0 resets the cursor. SEEK_END still reports the size (size queries).
  if (seq_) {
    if (whence == 2)
      return static_cast<int64_t>(file_.GetSize()) + off;
    if (whence == 0 && off == 0)
      seqPos_ = 0;
    return static_cast<int64_t>(seqPos_);
  }
  // SEEK_DATA(3)/SEEK_HOLE(4): we expose fully-allocated, hole-less files. The
  // engine uses lseek(fd, 0, SEEK_HOLE) as a file-size query (the only "hole" is
  // at EOF), so this must return the size, not silently fall back to SEEK_SET.
  if (whence == 3 || whence == 4) {
    int64_t sz = static_cast<int64_t>(file_.GetSize());
    if (off < 0 || off > sz) {
      if (std::getenv("DELTA_OPEN_TRACE"))
        std::fprintf(stderr, "[lseek] whence=%d off=%lld sz=%lld -> ENXIO\n",
                     whence, (long long)off, (long long)sz);
      return -SysError::eNXIO;
    }
    int64_t r = (whence == 3) ? off : sz;  // SEEK_DATA: off; SEEK_HOLE: EOF
    file_.Seek(r, utl::seekMode::seek_set);
    if (std::getenv("DELTA_OPEN_TRACE"))
      std::fprintf(stderr, "[lseek] whence=%d off=%lld sz=%lld -> %lld\n", whence,
                   (long long)off, (long long)sz, (long long)r);
    return r;
  }
  utl::seekMode mode = utl::seekMode::seek_set;
  if (whence == 1)
    mode = utl::seekMode::seek_cur;
  else if (whence == 2)
    mode = utl::seekMode::seek_end;
  file_.Seek(off, mode);
  int64_t pos = static_cast<int64_t>(file_.Tell());
  if (std::getenv("DELTA_RDALL")) {
    std::fprintf(stderr, "[lseek] off=%lld whence=%d -> pos=%lld\n", (long long)off,
                 whence, (long long)pos);
    // Non-trivial seek: scan the host stack (guest runs natively) for TRAS .text
    // return addresses to find who computed this offset.
    if (off > 0x10) {
      volatile uint64_t marker = 0;
      auto *sp = reinterpret_cast<uint64_t *>(
          reinterpret_cast<uintptr_t>(&marker) & ~7ull);
      int shown = 0;
      for (int i = 0; i < 1024 && shown < 8; i++) {
        uint64_t v = sp[i];
        if (v >= 0x401000 && v < 0x1500000) {
          std::fprintf(stderr, "  lseek-caller TRAS+%#llx\n",
                       (unsigned long long)(v - 0x400000));
          shown++;
        }
      }
    }
  }
  return pos;
}

// pread: read at an absolute offset without disturbing the file position (the
// guest keeps its own position for sequential reads). Backs a file mmap.
int64_t fileDevice::readAt(void *buf, size_t n, int64_t off) {
  if (!open_)
    return -SysError::eBADF;
  if (seq_) {  // ignore the bogus offset; serve in order from the cursor
    file_.Seek(static_cast<int64_t>(seqPos_), utl::seekMode::seek_set);
    int64_t r = static_cast<int64_t>(file_.Read(buf, n));
    if (r > 0)
      seqPos_ += static_cast<uint64_t>(r);
    return r;
  }
  uint64_t saved = file_.Tell();
  file_.Seek(off, utl::seekMode::seek_set);
  int64_t r = static_cast<int64_t>(file_.Read(buf, n));
  file_.Seek(static_cast<int64_t>(saved), utl::seekMode::seek_set);
  if (r >= 16 && std::getenv("DELTA_FILEREAD_TRACE")) {
    auto *b = static_cast<const uint8_t *>(buf);
    if (b[0] == 'T' && b[1] == 'A' && b[2] == 'F' && b[3] == 'S') {
      uint32_t cc = b[0x0c] | (b[0x0d] << 8) | (b[0x0e] << 16) | (b[0x0f] << 24);
      std::fprintf(stderr, "[preadAt TAFS] off=%lld n=%zu -> %lld count@0xc=%u\n",
                   (long long)off, n, (long long)r, cc);
    }
  }
  return r;
}

int fileDevice::fstat(void *stat) {
  if (!open_)
    return -SysError::eBADF;
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), kSceFileModeReg,
           static_cast<int64_t>(file_.GetSize()));
  return 0;
}
} // namespace krnl
