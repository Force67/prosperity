
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <base/strings/string_ref.h>
#include <cstdio>

#include "kern/dev/console_dev.h"
#include "kern/dev/dipsw_dev.h"
#include "kern/dev/dce_dev.h"
#include "kern/dev/dir_dev.h"
#include "kern/dev/dma_dev.h"
#include "kern/dev/file_dev.h"
#include "kern/dev/gc_dev.h"
#include "kern/dev/tty6_dev.h"
#include "kern/proc.h"
#include "kern/vfs.h"
#include "sys_vfs.h"

#include <utl/object_ref.h>

namespace krnl {
// Scan the (guest) stack for the first return address inside any guest module's
// .text and print it as <module>+offset, to pin which guest code issued an open.
// Native backend runs handlers on the guest stack. Gated; for tracing loops.
static void printOpenCaller(const char *path) {
  static const bool on = std::getenv("DELTA_OPEN_CALLER") != nullptr;
  if (!on)
    return;
  auto *proc = proc::getActive();
  if (!proc)
    return;
  auto *sp = reinterpret_cast<uintptr_t *>(__builtin_frame_address(0));
  int printed = 0;
  for (int i = 0; i < 768 && printed < 5; i++) {
    uintptr_t v = sp[i];
    for (auto &m : proc->getModuleList()) {
      auto &mi = m->getInfo();
      auto base = reinterpret_cast<uintptr_t>(mi.textSeg.addr);
      if (base && v >= base && v < base + mi.textSeg.size) {
        std::fprintf(stderr, "[open-caller] %s : %s+%#lx\n", path,
                     mi.name.c_str(), v - base);
        printed++;
        break;
      }
    }
  }
}

static device *make_device(const char *deviceName) {
  base::StringRef xname(deviceName);

  device *dev = nullptr;
  auto *proc = proc::getActive();
  if (xname == "console")
    dev = new consoleDevice(proc);
  if (xname == "deci_tty6")
    dev = new tty6Device(proc);
  if (xname == "gc")
    dev = new gcDevice(proc);
  if (xname == "dce")
    dev = new dceDevice(proc);
  if (xname == "dipsw")
    dev = new dipswDevice(proc);
  /*there are multiple of these*/
  if (xname.find("dmem", 0, 4) != base::StringRef::npos)
    dev = new dmaDevice(proc);

  return dev;
}

int PS4ABI sys_open(const char *path, uint32_t flags, uint32_t mode) {
  if (!path)
    return -SysError::eINVAL;

  static const bool vtrace = std::getenv("DELTA_VFS_TRACE") != nullptr;
  if (vtrace)
    std::fprintf(stderr, "[open] %s flags=%#x mode=%#x\n", path, flags, mode);
  if (std::strstr(path, ".psarc"))
    printOpenCaller(path);

  if (std::strncmp(path, "/dev/", 5) == 0) {
    const char *name = &path[5];

    auto dev = make_device(name);
    if (dev) {

      if (!dev->init(name, flags, mode)) {
        dev->releaseHandle();
        return -SysError::eNXIO;
      }

      return dev->handle();
    }
    // unknown device: fail soft instead of trapping
    return -SysError::eNOENT;
  }

  // Directory: games open one (O_DIRECTORY) then getdents it to find assets.
  // Gated on the flag so a normal file open doesn't pay the full listing scan.
  if (flags & O_DIRECTORY) {
    std::vector<vfs::DirEntry> entries;
    if (vfs::listDir(path, entries)) {
      auto *dir = new dirDevice(proc::getActive(), std::move(entries));
      return dir->handle();
    }
    return -SysError::eNOENT;
  }

  // Regular file: resolve through the VFS (host + virtual mounts).
  utl::File vf = vfs::openRead(path);
  if (!vf.Exists()) {
    if (vtrace)
      std::fprintf(stderr, "[open]   -> ENOENT %s\n", path);
    return -SysError::eNOENT;
  }

  int64_t fsize = vf.GetSize();
  auto *file = new fileDevice(proc::getActive());
  if (!file->adopt(std::move(vf))) {
    file->releaseHandle();
    return -SysError::eNOENT;
  }
  if (vtrace)
    std::fprintf(stderr, "[open]   -> fd=%u size=%lld %s\n", file->handle(),
                 (long long)fsize, path);
  return file->handle();
}

// Resolve an fd (object-table handle) back to the device that backs it.
static device *fdToDevice(uint32_t fd) {
  auto *obj = proc::getActive()->getObjTable().get(fd);
  if (!obj || obj->type() != kObject::oType::device)
    return nullptr;
  return static_cast<device *>(obj);
}

int64_t PS4ABI sys_read(uint32_t fd, void *buf, size_t nbytes) {
  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  return d->read(buf, nbytes);
}

int64_t PS4ABI sys_lseek(uint32_t fd, int64_t offset, int whence) {
  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  return d->lseek(offset, whence);
}

int PS4ABI sys_fstat(uint32_t fd, void *stat) {
  // Zero first: a failed/unsupported fstat must not leave the caller's stat
  // buffer uninitialized. Games read st_size from it without checking the
  // return and then allocate that many bytes (garbage -> bad_alloc).
  if (stat)
    std::memset(stat, 0, sizeof(SceKernelStat));
  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  return d->fstat(stat);
}

int PS4ABI sys_stat(const char *path, void *stat) {
  if (!path || !stat)
    return -SysError::eFAULT;
  int64_t size = 0;
  bool isDir = false;
  if (!vfs::stat(path, size, isDir))
    return -SysError::eNOENT;
  fillStat(*reinterpret_cast<SceKernelStat *>(stat),
           isDir ? kSceFileModeDir : kSceFileModeReg, size);
  return 0;
}

int64_t PS4ABI sys_getdents(uint32_t fd, void *buf, size_t nbytes) {
  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  return d->getdents(buf, nbytes);
}

int PS4ABI sys_close(uint32_t fd) {
  auto *proc = proc::getActive();

  if (proc && fd != -1) {
    proc->getObjTable().release(fd);
    return 0;
  }

  LOG_WARNING("failed to release handle {}", fd);
  return -SysError::eBADF;
}
} // namespace krnl