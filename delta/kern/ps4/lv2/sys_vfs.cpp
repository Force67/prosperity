
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <unistd.h>
#include <base/strings/string_ref.h>
#include <algorithm>
#include <cstdio>
#include <deque>
#include <mutex>

#include "kern/ps4/dev/ajm_dev.h"
#include "kern/ps4/dev/console_dev.h"
#include "kern/ps4/dev/dipsw_dev.h"
#include "kern/ps4/dev/dce_dev.h"
#include "kern/ps4/dev/dir_dev.h"
#include "kern/ps4/dev/dma_dev.h"
#include "kern/ps4/dev/file_dev.h"
#include "kern/ps4/dev/gc_dev.h"
#include "kern/ps4/dev/tty6_dev.h"
#include "kern/proc.h"
#include "kern/crash.h"
#include "kern/vfs.h"
#include "sys_mem.h"
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
  if (xname == "ajm")
    dev = new ajmDevice(proc);
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

  // Writable open (savedata): a create/write flag on a path under a writable
  // host mount goes to a writable fileDevice. Read-only titles never take this
  // (they open /app0, a read-only virtual mount), so it can't affect them.
  const uint32_t accmode = flags & O_ACCMODE;
  const bool writeIntent =
      accmode == O_WRONLY || accmode == O_RDWR || (flags & O_CREAT);
  if (writeIntent) {
    base::String host = vfs::resolveWritable(path);
    if (!host.empty()) {
      auto *file = new fileDevice(proc::getActive());
      if (file->openWritable(host, (flags & O_CREAT) != 0,
                             (flags & O_TRUNC) != 0)) {
        if (vtrace)
          std::fprintf(stderr, "[open]   -> writable fd=%u %s\n",
                       file->handle(), host.c_str());
        return file->handle();
      }
      file->releaseHandle();
      return -SysError::eNOENT;
    }
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
  // SOTTR's TAFS loader reads .manifest.bin with an uninitialised file offset;
  // serve those sequentially so the header (off 0) loads. See setSeqMode().
  if (std::getenv("DELTA_MANIFEST_SEQ") && std::strstr(path, ".manifest.bin"))
    file->setSeqMode();
  // Flag manifest fds so the read-request setter hook (DELTA_RDOFF_FIX) can
  // force their read offset to 0.
  if (std::strstr(path, ".manifest.bin"))
    markManifestFd(file->handle(), true);
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
  if (!d) {
    if (std::getenv("DELTA_RDALL"))
      std::fprintf(stderr, "[rd] fd=%u -> EBADF (no device)\n", fd);
    return -SysError::eBADF;
  }
  int64_t r = d->read(buf, nbytes);
  // DELTA_READ_TRACE: log large reads (asset/texture loads) + their target buffer,
  // to see whether texture data lands in the GPU texture region (0x41x) directly or
  // a staging buffer the game later copies from.
  static const bool rt = std::getenv("DELTA_READ_TRACE") != nullptr;
  if (rt && nbytes >= 0x4000)
    std::fprintf(stderr, "[read] fd=%u buf=%p nbytes=%#zx -> %lld\n", fd, buf, nbytes,
                 (long long)r);
  static const bool ra = std::getenv("DELTA_RDALL") != nullptr;
  if (ra) {
    uint32_t f4 = 0;
    if (buf && r >= 4) f4 = *reinterpret_cast<const uint32_t *>(buf);
    std::fprintf(stderr, "[rd] t=%ld fd=%u nbytes=%#zx -> %lld buf=%p first4=%08x\n", (long)gettid(), fd, nbytes,
                 (long long)r, buf, f4);
  }
  return r;
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
  // shm fds aren't device-backed; size them from the shm backing so a title
  // that fstat()s a shm before mmap'ing it (e.g. libSceAvSetting) gets a real
  // st_size instead of -EBADF + a zero-length map.
  if (size_t sz = shmFstatSize(fd); sz != SIZE_MAX) {
    if (stat) {
      auto *st = static_cast<SceKernelStat *>(stat);
      st->st_size = static_cast<int64_t>(sz);
      st->st_mode = 0x8000;  // S_IFREG
      st->st_blksize = 0x4000;
    }
    return 0;
  }
  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  int r = d->fstat(stat);
  if (std::getenv("DELTA_RDALL") && stat)
    std::fprintf(stderr, "[fstat] fd=%u -> st_size=%lld\n", fd,
                 (long long)static_cast<SceKernelStat *>(stat)->st_size);
  return r;
}

int PS4ABI sys_stat(const char *path, void *stat) {
  if (!path || !stat)
    return -SysError::eFAULT;
  // Zero first, for the reason sys_fstat documents: callers read st_size without
  // checking the return and then size a buffer from it. A missing file must
  // leave st_size = 0, not stack garbage (DOOM read a -1 size and crashed).
  std::memset(stat, 0, sizeof(SceKernelStat));
  int64_t size = 0;
  bool isDir = false;
  if (!vfs::stat(path, size, isDir)) {
    if (std::getenv("DELTA_RDALL"))
      std::fprintf(stderr, "[stat] %s -> ENOENT\n", path);
    return -SysError::eNOENT;
  }
  fillStat(*reinterpret_cast<SceKernelStat *>(stat),
           isDir ? kSceFileModeDir : kSceFileModeReg, size);
  if (std::getenv("DELTA_RDALL"))
    std::fprintf(stderr, "[stat] %s -> size=%lld dir=%d\n", path,
                 (long long)size, (int)isDir);
  return 0;
}

int64_t PS4ABI sys_getdents(uint32_t fd, void *buf, size_t nbytes) {
  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  return d->getdents(buf, nbytes);
}

// Regular-file fd slots are released a bounded number of closes late. Titles
// (e.g. Shadow of the Tomb Raider) open a file, hand its fd to an async I/O
// worker, then immediately close and reopen the next file. If we free the slot
// at once it is reused for the next open, and the worker's still-pending read
// lands on the wrong file -> a garbage archive header -> a huge (~32 GiB)
// entry-table allocation. Keeping the last N closed file slots alive lets the
// lagging read complete against the right file. The window is small; PFS-backed
// files share one host fd, so this does not consume host descriptors. Char
// devices (/dev/gc, ...) are released immediately.
static std::mutex g_deferM;
static std::deque<uint32_t> g_deferred;
static constexpr size_t kDeferredCloseWindow = 256;

int PS4ABI sys_close(uint32_t fd) {
  auto *proc = proc::getActive();

  if (proc && fd != -1) {
    if (std::getenv("DELTA_RDALL"))
      std::fprintf(stderr, "[close] fd=%u\n", fd);
    auto *d = fdToDevice(fd);
    if (d && d->isRegularFile()) {
      uint32_t evict = static_cast<uint32_t>(-1);
      {
        std::lock_guard<std::mutex> lk(g_deferM);
        // A deferred fd keeps its slot pinned, so it can't have been reopened as
        // a different file; a second close of it is a redundant double-close and
        // must not queue a second (wrong) release.
        bool already = std::find(g_deferred.begin(), g_deferred.end(), fd) !=
                       g_deferred.end();
        if (!already) {
          g_deferred.push_back(fd);
          if (g_deferred.size() > kDeferredCloseWindow) {
            evict = g_deferred.front();
            g_deferred.pop_front();
          }
        }
      }
      if (evict != static_cast<uint32_t>(-1))
        proc->getObjTable().release(evict);
      return 0;
    }
    proc->getObjTable().release(fd);
    return 0;
  }

  LOG_WARNING("failed to release handle {}", fd);
  return -SysError::eBADF;
}
} // namespace krnl