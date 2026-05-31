/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

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
  return static_cast<int64_t>(file_.Read(buf, n));
}

int64_t fileDevice::lseek(int64_t off, int whence) {
  if (!open_)
    return -SysError::eBADF;
  utl::seekMode mode = utl::seekMode::seek_set;
  if (whence == 1)
    mode = utl::seekMode::seek_cur;
  else if (whence == 2)
    mode = utl::seekMode::seek_end;
  file_.Seek(off, mode);
  return static_cast<int64_t>(file_.Tell());
}

int fileDevice::fstat(void *stat) {
  if (!open_)
    return -SysError::eBADF;
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), kSceFileModeReg,
           static_cast<int64_t>(file_.GetSize()));
  return 0;
}
} // namespace krnl
