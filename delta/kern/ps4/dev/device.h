#pragma once

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
#include <cstring>
#include <cstdio>

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "kern/lv2/error_table.h"
#include "kern/object.h"

namespace krnl {
class proc;

class device : public kObject {
public:
  inline device(proc *p) : kObject(p, kObject::oType::device) {}

  virtual bool init(const char *, u32, u32) { return true; }

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
  virtual u8 *map(void *, size_t, u32, u32, size_t) {
    BASE_LOGI("dev", "UNHANDLED map on {}", name.c_str());
    return reinterpret_cast<u8 *>(-1);
  }
  virtual i32 ioctl(u32 command, void *args) {
    // Decode the BSD encoding into the log: a bare number tells the next
    // person nothing, and the group/number pair is what identifies which
    // driver's command set an unknown ioctl belongs to.
    BASE_LOGI("dev",
              "UNHANDLED ioctl({:#x}) on '{}' -> 0 (dir={}{} len={:#x} "
              "group={:#x} num={:#x})",
              command, name.c_str(), (command & 0x80000000u) ? "I" : "",
              (command & 0x40000000u) ? "O" : "", (command >> 16) & 0x1fff,
              (command >> 8) & 0xff, command & 0xff);
    // An OUT buffer the guest reads after a soft-succeed would otherwise hold
    // whatever was on its stack.
    if (args && (command & 0x40000000u)) {
      const u32 len = (command >> 16) & 0x1fff;
      if (len)
        std::memset(args, 0, len);
    }
    return 0;
  }

  // File-like operations. Default to "not supported"; real files and char
  // devices override what they implement. Convention follows lv2: >= 0 on
  // success (byte count / offset), negative SysError on failure.
  virtual i64 read(void *, size_t) { return -SysError::eNODEV; }
  virtual i64 write(const void *, size_t) { return -SysError::eNODEV; }
  virtual i64 lseek(i64, int) { return -SysError::eNODEV; }
  // Read `n` bytes at absolute `off` into `buf` WITHOUT moving the file position
  // (pread). Used to back a file mmap with the file's content. Returns bytes read
  // (0 at/after EOF), or -1 if this device isn't a readable file.
  virtual i64 readAt(void *, size_t, i64) { return -1; }
  virtual int fstat(void * /*SceKernelStat*/) { return -SysError::eNODEV; }
  // Directory enumeration (FreeBSD dirents). Non-directories aren't one.
  virtual i64 getdents(void *, size_t) { return -SysError::eNOTDIR; }
};
}