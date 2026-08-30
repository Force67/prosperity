/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base/logging.h>
#include "base/arch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "file_dev.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kFileReadTrace, "DELTA_FILEREAD_TRACE", false);
DELTA_OPTION(bool, kOpenTrace, "DELTA_OPEN_TRACE", false);
DELTA_OPTION(bool, kRdall, "DELTA_RDALL", false);
}  // namespace

namespace krnl {
void fillStat(SceKernelStat &out, u16 mode, i64 size) {
  std::memset(&out, 0, sizeof(out));
  out.st_mode = mode;
  out.st_size = size;
  out.st_nlink = 1;
  out.st_blksize = 0x4000;
  out.st_blocks = (size + 0x3FFF) / 0x4000;
}

fileDevice::fileDevice(objectTable &objects) : device(objects) {}

bool fileDevice::open(const base::String &hostPath, u32 /*flags*/) {
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

bool fileDevice::openWritable(const base::String &hostPath, bool create,
                              bool truncate) {
  // Probe existence (open-read then close) to pick the fopen mode: an existing
  // file opens rb+ (keep contents), else wb+ when creating.
  utl::File probe(hostPath, utl::fileMode::read);
  const bool exists = probe.Exists() && probe.IsOpen();
  probe.Close();

  utl::fileMode mode;
  if (truncate || (!exists && create))
    mode = utl::fileMode::create;  // wb+ : create/truncate, read+write
  else if (exists)
    mode = utl::fileMode::readWrite;  // rb+ : keep contents, read+write
  else
    return false;  // open-existing for write, but absent and no create

  utl::File f(hostPath, mode);
  if (!f.Exists() || !f.IsOpen())
    return false;
  file_.Reset(f.GetBase());
  open_ = true;
  writable_ = true;
  return true;
}

i64 fileDevice::write(const void *buf, size_t n) {
  if (!open_ || !writable_)
    return -SysError::eBADF;
  // PhysFile::Write returns 1 on a complete fwrite; report bytes written.
  if (!file_.Write(buf, n))
    return 0;
  // Flush every guest write: the emulator is normally SIGKILLed, which discards
  // whatever is still sitting in the stdio buffer. Without this a save only ever
  // reached disk in whole 4 KiB buffer flushes -- every savedata file ended up
  // 0 bytes or truncated mid-record at exactly 4096.
  file_.Flush();
  return static_cast<i64>(n);
}

bool fileDevice::adopt(utl::File &&file) {
  if (!file.Exists())
    return false;
  file_.Reset(file.GetBase());
  open_ = true;
  return true;
}

i64 fileDevice::read(void *buf, size_t n) {
  if (!open_)
    return -SysError::eBADF;
  if (seq_)
    file_.Seek(static_cast<i64>(seqPos_), utl::seekMode::seek_set);
  i64 r = static_cast<i64>(file_.Read(buf, n));
  if (seq_ && r > 0)
    seqPos_ += static_cast<u64>(r);
  if (r >= 16 && kFileReadTrace) {
    auto *b = static_cast<const u8 *>(buf);
    // TAFS manifest? dump the entry_count at +0x0c the game will read back.
    if (b[0] == 'T' && b[1] == 'A' && b[2] == 'F' && b[3] == 'S') {
      u32 cc = b[0x0c] | (b[0x0d] << 8) | (b[0x0e] << 16) | (b[0x0f] << 24);
      BASE_LOGI("fread TAFS", "n={} -> {}  entry_count@0xc={}", n,
                (long long)r, cc);
    }
  }
  return r;
}

i64 fileDevice::lseek(i64 off, int whence) {
  if (!open_)
    return -SysError::eBADF;
  // Sequential mode (manifest): ignore the engine's bogus absolute seeks; only a
  // SEEK_SET 0 resets the cursor. SEEK_END still reports the size (size queries).
  if (seq_) {
    if (whence == 2)
      return static_cast<i64>(file_.GetSize()) + off;
    if (whence == 0 && off == 0)
      seqPos_ = 0;
    return static_cast<i64>(seqPos_);
  }
  // SEEK_DATA(3)/SEEK_HOLE(4): we expose fully-allocated, hole-less files. The
  // engine uses lseek(fd, 0, SEEK_HOLE) as a file-size query (the only "hole" is
  // at EOF), so this must return the size, not silently fall back to SEEK_SET.
  if (whence == 3 || whence == 4) {
    i64 sz = static_cast<i64>(file_.GetSize());
    if (off < 0 || off > sz) {
      if (kOpenTrace)
        BASE_LOGI("lseek", "whence={} off={} sz={} -> ENXIO", whence,
                  (long long)off, (long long)sz);
      return -SysError::eNXIO;
    }
    i64 r = (whence == 3) ? off : sz;  // SEEK_DATA: off; SEEK_HOLE: EOF
    file_.Seek(r, utl::seekMode::seek_set);
    if (kOpenTrace)
      BASE_LOGI("lseek", "whence={} off={} sz={} -> {}", whence,
                (long long)off, (long long)sz, (long long)r);
    return r;
  }
  utl::seekMode mode = utl::seekMode::seek_set;
  if (whence == 1)
    mode = utl::seekMode::seek_cur;
  else if (whence == 2)
    mode = utl::seekMode::seek_end;
  file_.Seek(off, mode);
  i64 pos = static_cast<i64>(file_.Tell());
  if (kRdall) {
    BASE_LOGI("lseek", "off={} whence={} -> pos={}", (long long)off, whence,
              (long long)pos);
    // Non-trivial seek: scan the host stack (guest runs natively) for TRAS .text
    // return addresses to find who computed this offset.
    if (off > 0x10) {
      volatile u64 marker = 0;
      auto *sp = reinterpret_cast<u64 *>(
          reinterpret_cast<uintptr_t>(&marker) & ~7ull);
      int shown = 0;
      for (int i = 0; i < 1024 && shown < 8; i++) {
        u64 v = sp[i];
        if (v >= 0x401000 && v < 0x1500000) {
          BASE_LOGI("lseek", "  lseek-caller TRAS+{:#x}",
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
i64 fileDevice::readAt(void *buf, size_t n, i64 off) {
  if (!open_)
    return -SysError::eBADF;
  if (seq_) {  // ignore the bogus offset; serve in order from the cursor
    file_.Seek(static_cast<i64>(seqPos_), utl::seekMode::seek_set);
    i64 r = static_cast<i64>(file_.Read(buf, n));
    if (r > 0)
      seqPos_ += static_cast<u64>(r);
    return r;
  }
  u64 saved = file_.Tell();
  file_.Seek(off, utl::seekMode::seek_set);
  i64 r = static_cast<i64>(file_.Read(buf, n));
  file_.Seek(static_cast<i64>(saved), utl::seekMode::seek_set);
  if (r >= 16 && kFileReadTrace) {
    auto *b = static_cast<const u8 *>(buf);
    if (b[0] == 'T' && b[1] == 'A' && b[2] == 'F' && b[3] == 'S') {
      u32 cc = b[0x0c] | (b[0x0d] << 8) | (b[0x0e] << 16) | (b[0x0f] << 24);
      BASE_LOGI("preadAt TAFS", "off={} n={} -> {} count@0xc={}",
                (long long)off, n, (long long)r, cc);
    }
  }
  return r;
}

int fileDevice::fstat(void *stat) {
  if (!open_)
    return -SysError::eBADF;
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), kSceFileModeReg,
           static_cast<i64>(file_.GetSize()));
  return 0;
}
} // namespace krnl
