#pragma once

#include <base.h>
#include <cstdio>

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "kern/ps4/lv2/error_table.h"
#include "kern/object.h"

namespace krnl {
class proc;

class device : public kObject {
public:
  inline device(proc *p) : kObject(p, kObject::oType::device) {}

  virtual bool init(const char *, uint32_t, uint32_t) { return true; }

  // True for host/pfs-backed regular files. Titles read these from async worker
  // threads that can lag behind the opening thread's close+reopen; sys_close
  // defers releasing their fd slot so a still-pending read can't land on a freed
  // (and reused) slot and read another file's bytes. Overridden by fileDevice.
  virtual bool isRegularFile() const { return false; }

  // True for /dev/dmem: an mmap through it maps direct memory at the physical
  // offset in the mmap offset argument, which sceKernelVirtualQuery must report
  // back (titles turn it into a block index in their own heap map).
  virtual bool isDirectMemory() const { return false; }

  // Unknown map/ioctl on a device: soft-fail (and log) instead of trapping, so
  // the boot keeps advancing and we can see what the guest actually wanted.
  virtual uint8_t *map(void *, size_t, uint32_t, uint32_t, size_t) {
    std::printf("[dev] UNHANDLED map on %s\n", name.c_str());
    return reinterpret_cast<uint8_t *>(-1);
  }
  virtual int32_t ioctl(uint32_t command, void *args) {
    std::printf("[dev] UNHANDLED ioctl(%#x) on %s -> 0\n", command,
                name.c_str());
    return 0;
  }

  // File-like operations. Default to "not supported"; real files and char
  // devices override what they implement. Convention follows lv2: >= 0 on
  // success (byte count / offset), negative SysError on failure.
  virtual int64_t read(void *, size_t) { return -SysError::eNODEV; }
  virtual int64_t write(const void *, size_t) { return -SysError::eNODEV; }
  virtual int64_t lseek(int64_t, int) { return -SysError::eNODEV; }
  // Read `n` bytes at absolute `off` into `buf` WITHOUT moving the file position
  // (pread). Used to back a file mmap with the file's content. Returns bytes read
  // (0 at/after EOF), or -1 if this device isn't a readable file.
  virtual int64_t readAt(void *, size_t, int64_t) { return -1; }
  virtual int fstat(void * /*SceKernelStat*/) { return -SysError::eNODEV; }
  // Directory enumeration (FreeBSD dirents). Non-directories aren't one.
  virtual int64_t getdents(void *, size_t) { return -SysError::eNOTDIR; }
};
}