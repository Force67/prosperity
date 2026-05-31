
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
#include "kern/dev/dma_dev.h"
#include "kern/dev/file_dev.h"
#include "kern/dev/gc_dev.h"
#include "kern/dev/tty6_dev.h"
#include "kern/proc.h"
#include "kern/vfs.h"

#include <utl/object_ref.h>

namespace krnl {
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
    return SysError::eINVAL;

  std::fprintf(stderr, "[open] %s flags=%#x mode=%#x\n", path, flags, mode);

  if (std::strncmp(path, "/dev/", 5) == 0) {
    const char *name = &path[5];

    auto dev = make_device(name);
    if (dev) {

      if (!dev->init(name, flags, mode)) {
        dev->releaseHandle();
        return -1;
      }

      return dev->handle();
    }
    // unknown device: fail soft instead of trapping
    return SysError::eNOENT;
  }

  // Regular file: resolve through the VFS (host + virtual mounts).
  utl::File vf = vfs::openRead(path);
  if (!vf.Exists())
    return SysError::eNOENT;

  auto *file = new fileDevice(proc::getActive());
  if (!file->adopt(std::move(vf))) {
    file->releaseHandle();
    return SysError::eNOENT;
  }
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
  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  return d->fstat(stat);
}

int PS4ABI sys_stat(const char *path, void *stat) {
  int64_t size = 0;
  bool isDir = false;
  if (!vfs::stat(path, size, isDir))
    return SysError::eNOENT;
  fillStat(*reinterpret_cast<SceKernelStat *>(stat),
           isDir ? kSceFileModeDir : kSceFileModeReg, size);
  return 0;
}

int PS4ABI sys_close(uint32_t fd) {
  auto *proc = proc::getActive();

  if (proc && fd != -1) {
    proc->getObjTable().release(fd);
    return 0;
  }

  LOG_WARNING("failed to release handle {}", fd);
  return -1;
}
} // namespace krnl