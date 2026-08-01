
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <base.h>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <unistd.h>
#include <unordered_map>

#include <logger/logger.h>

#include "error_table.h"
#include "kern/ps4/dev/device.h"
#include "kern/ps4/dev/file_dev.h" // SceKernelStat, fillStat, kSceFileMode*
#include "kern/proc.h"
#include "kern/vfs.h"
#include "sys_vfs.h" // sys_open (sys_openat delegates to it)
#include "sys_vfs_ext.h"
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kQarBuf, "DELTA_QARBUF", false);
DELTA_OPTION(bool, kIoprogress, "DELTA_IOPROGRESS", false);
DELTA_OPTION(bool, kRdall, "DELTA_RDALL", false);
DELTA_OPTION(bool, kVfsTrace, "DELTA_VFS_TRACE", false);
}  // namespace

namespace krnl {

enum { kSeekSet = 0, kSeekCur = 1 };

// The helper in sys_vfs.cpp is file-local static, so keep our own copy.
static device *fdToDevice(uint32_t fd) {
  auto *obj = proc::getActive()->getObjTable().get(fd);
  if (!obj || obj->type() != kObject::oType::device)
    return nullptr;
  return static_cast<device *>(obj);
}

struct sce_iovec {
  void *iov_base;
  size_t iov_len;
};

int PS4ABI sys_access(const char *path, int mode) {
  if (!path)
    return -SysError::eINVAL;
  // We model read-only assets, so existence is the only check we can honour;
  // W_OK/X_OK are accepted for anything that exists.
  int64_t size = 0;
  bool isDir = false;
  if (!vfs::stat(path, size, isDir))
    return -SysError::eNOENT;
  return 0;
}

int PS4ABI sys_faccessat(int fd, const char *path, int mode, int flag) {
  return sys_access(path, mode);
}

// We have no symbolic links in the VFS. The kernel returns EINVAL when the
// target is not a VLNK vnode, so we do too. The PS4 uses symlinks only for
// /app0 -> the PFS mount root and the base system dirs; our VFS resolves
// those directly, so readlink should never reach a callable path.
int PS4ABI sys_readlink(const char *path, char *buf, size_t bufsize) {
  return -SysError::eINVAL;
}

int PS4ABI sys_readlinkat(int fd, const char *path, char *buf, size_t bufsize) {
  return -SysError::eINVAL;
}

// No symlinks, so lstat is plain stat. Zero the buffer first for the reason
// sys_fstat documents: callers read st_size without checking the return.
int PS4ABI sys_lstat(const char *path, void *stat) {
  if (!path)
    return -SysError::eINVAL;
  if (stat)
    std::memset(stat, 0, sizeof(SceKernelStat));
  int64_t size = 0;
  bool isDir = false;
  if (!vfs::stat(path, size, isDir))
    return -SysError::eNOENT;
  fillStat(*reinterpret_cast<SceKernelStat *>(stat),
           isDir ? kSceFileModeDir : kSceFileModeReg, size);
  return 0;
}

int PS4ABI sys_fstatat(int fd, const char *path, void *stat, int flag) {
  return sys_lstat(path, stat);
}

// sys_fcntl: the kernel validates cmd in the non-privileged path against
// bitmask 0x3818 {F_GETFL(3), F_SETFL(4), F_GETLK(11), F_SETLK(12),
// F_SETLKW(13)}. cmds 7/8/9 (OGETLK/OSETLK/OSETLKW) are translated to 11/12/13.
// We don't model fd flags or advisory locks, so GETFD/GETFL report 0 and the
// lock commands accept silently.
int PS4ABI sys_fcntl(uint32_t fd, int cmd, int64_t arg) {
  enum {
    F_DUPFD = 0, F_GETFD = 1, F_SETFD = 2, F_GETFL = 3, F_SETFL = 4,
    F_GETOWN = 5, F_SETOWN = 6,
    F_OGETLK = 7, F_OSETLK = 8, F_OSETLKW = 9,
    F_GETLK = 11, F_SETLK = 12, F_SETLKW = 13,
  };
  // Normalize legacy OGETLK/OSETLK/OSETLKW (7/8/9) to their modern equivalents.
  int ncmd = cmd;
  switch (cmd) {
  case F_OGETLK:  ncmd = F_GETLK;  break;
  case F_OSETLK:  ncmd = F_SETLK;  break;
  case F_OSETLKW: ncmd = F_SETLKW; break;
  }
  switch (ncmd) {
  case F_GETFL:
    // Report O_RDONLY: the device opened with whatever flags the guest passed,
    // but we model read-only access for regular files.
    return 0;
  case F_SETFL:
  case F_GETFD:
  case F_SETFD:
    return 0;
  case F_GETLK:
    // No locks held: return F_UNLCK (type 2) in the caller's flock struct.
    if (arg) {
      auto *fl = reinterpret_cast<int32_t *>(arg);
      fl[0] = 2;  // l_type = F_UNLCK
    }
    return 0;
  case F_SETLK:
  case F_SETLKW:
    return 0;  // advisory lock accepted, not enforced
  case F_DUPFD:
    return -SysError::eOPNOTSUPP;  // no descriptor duplication in the object table
  case F_GETOWN:
  case F_SETOWN:
    return 0;  // no signal delivery so ownership is inert
  default:
    LOG_WARNING("sys_fcntl: unhandled cmd {} on fd {} -> 0", cmd, fd);
    return 0;
  }
}

int PS4ABI sys_dup(uint32_t fd) {
  LOG_WARNING("sys_dup({}) unsupported", fd);
  return -SysError::eOPNOTSUPP;
}

int PS4ABI sys_dup2(uint32_t oldfd, uint32_t newfd) {
  LOG_WARNING("sys_dup2({}, {}) unsupported", oldfd, newfd);
  return -SysError::eOPNOTSUPP;
}

int PS4ABI sys_fsync(uint32_t fd) { return 0; }
int PS4ABI sys_fdatasync(uint32_t fd) { return 0; }

int PS4ABI sys_getcwd(char *buf, size_t size) {
  if (!buf || size == 0)
    return -SysError::eINVAL;
  const char *cwd = "/app0"; // the single working directory we expose
  size_t n = std::strlen(cwd);
  if (n + 1 > size)
    n = size - 1;
  std::memcpy(buf, cwd, n);
  buf[n] = '\0';
  return 0;
}

// pread/pwrite must not disturb the file pointer. Our devices only offer
// seek+read, so snapshot the current offset, do the positioned I/O, then
// restore it. Without the restore a following read() would resume from the
// wrong place.
static bool g_qarFd[8192] = {false};
void markQarFd(uint32_t fd, bool v) {
  if (fd < 8192)
    g_qarFd[fd] = v;
}

void throttleIo(int64_t bytes);

int64_t PS4ABI sys_pread(uint32_t fd, void *buf, size_t nbytes, int64_t offset) {
  auto *d = fdToDevice(fd);
  if (!d) {
    if (kRdall)
      std::fprintf(stderr, "[pread] fd=%u off=%lld -> EBADF (no device)\n", fd, (long long)offset);
    return -SysError::eBADF;
  }
  struct timespec t0;
  if (kQarBuf)
    clock_gettime(CLOCK_MONOTONIC, &t0);
  int64_t saved = d->lseek(0, kSeekCur);
  d->lseek(offset, kSeekSet);
  int64_t r = d->read(buf, nbytes);
  if (saved >= 0)
    d->lseek(saved, kSeekSet);
  long readUs = 0;
  if (kQarBuf) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    readUs = (t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_nsec - t0.tv_nsec) / 1000;
  }
  throttleIo(r);
  if (kRdall) {
    uint32_t f4 = 0;
    if (buf && r >= 4) f4 = *reinterpret_cast<const uint32_t *>(buf);
    std::fprintf(stderr, "[pread] t=%ld fd=%u off=%lld nbytes=%#zx -> %lld buf=%p first4=%08x\n",
                 (long)gettid(), fd, (long long)offset, (size_t)nbytes, (long long)r, buf, f4);
  }
  // DELTA_QARBUF: where does streamed .qar data land? Reports the destination
  // buffer for reads on a *.qar fd, so we can tell whether textures stream into
  // a GPU-mapped region (0x81xx, directly bindable) or a low staging buffer that
  // still needs a copy/commit step the engine never performs.
  if (fd < 8192 && g_qarFd[fd] && kQarBuf) {
    std::fprintf(stderr,
                 "[qarbuf] fd=%u off=%lld nbytes=%#zx -> %lld buf=%p %ldus\n",
                 fd, (long long)offset, (size_t)nbytes, (long long)r, buf,
                 readUs);
  }
  // DELTA_IOPROGRESS: throttled per-fd streaming high-water mark. FOX/FIOS2 streams
  // large world archives via pread; this shows whether that streaming is advancing
  // (offset climbing) or has completed/stalled, without the DELTA_RDALL firehose.
  // At most one line per fd per ~2 s; prints the current + max offset and MB/s since
  // the last line so a long headless load can be tracked to completion.
  if (kIoprogress) {
    // maxOff/lastMax = streaming high-water. nNew/nReread since last line tell
    // whether the FIOS2 streamer is fetching NEW file bytes (nNew climbing,
    // offset+r > previous max) or RE-READING already-covered blocks (nReread) --
    // the latter signals a downstream consume/decompress stage that never drains,
    // so the streamer re-issues the same reads. lastOff catches exact-repeat reads.
    struct FdIo { int64_t maxOff, lastMax, lastOff; long lastMs; long nNew, nReread, nSame; };
    static std::mutex m;
    static std::unordered_map<uint32_t, FdIo> tbl;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    long nowMs = ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    std::lock_guard<std::mutex> lk(m);
    auto &e = tbl[fd];
    int64_t end = offset + (r > 0 ? r : 0);
    if (end > e.maxOff) e.nNew++; else e.nReread++;
    if (offset == e.lastOff) e.nSame++;
    e.lastOff = offset;
    if (end > e.maxOff) e.maxOff = end;
    if (e.lastMs == 0) e.lastMs = nowMs;
    if (nowMs - e.lastMs >= 2000) {
      double mb = (e.maxOff - e.lastMax) / 1048576.0;
      double sec = (nowMs - e.lastMs) / 1000.0;
      std::fprintf(stderr,
                   "[ioprog] fd=%u off=%lld max=%lld (%.1f MB) +%.2f MB/s  new=%ld reread=%ld same=%ld\n",
                   fd, (long long)offset, (long long)e.maxOff,
                   e.maxOff / 1048576.0, sec > 0 ? mb / sec : 0.0,
                   e.nNew, e.nReread, e.nSame);
      e.lastMax = e.maxOff;
      e.lastMs = nowMs;
      e.nNew = e.nReread = e.nSame = 0;
    }
  }
  return r;
}

int64_t PS4ABI sys_pwrite(uint32_t fd, const void *buf, size_t nbytes,
                          int64_t offset) {
  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  int64_t saved = d->lseek(0, kSeekCur);
  d->lseek(offset, kSeekSet);
  int64_t r = d->write(buf, nbytes);
  if (saved >= 0)
    d->lseek(saved, kSeekSet);
  return r;
}

int64_t PS4ABI sys_writev(uint32_t fd, const void *iov, int iovcnt) {
  auto *segs = static_cast<const sce_iovec *>(iov);
  if (!segs || iovcnt < 0)
    return -SysError::eINVAL;

  if (fd == 1 || fd == 2) { // stdout / stderr, like sys_write
    int64_t total = 0;
    for (int i = 0; i < iovcnt; ++i) {
      auto *b = static_cast<const char *>(segs[i].iov_base);
      for (size_t j = 0; j < segs[i].iov_len; ++j)
        std::printf("%c", b[j]);
      total += static_cast<int64_t>(segs[i].iov_len);
    }
    return total;
  }

  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  int64_t total = 0;
  for (int i = 0; i < iovcnt; ++i) {
    int64_t r = d->write(segs[i].iov_base, segs[i].iov_len);
    if (r < 0)
      return r;
    total += r;
  }
  return total;
}

int64_t PS4ABI sys_readv(uint32_t fd, const void *iov, int iovcnt) {
  auto *segs = static_cast<const sce_iovec *>(iov);
  if (!segs || iovcnt < 0)
    return -SysError::eINVAL;

  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  int64_t total = 0;
  for (int i = 0; i < iovcnt; ++i) {
    int64_t r = d->read(segs[i].iov_base, segs[i].iov_len);
    if (r < 0)
      return r;
    total += r;
  }
  return total;
}

// Positional vectored I/O. These were stubbed to return 0, which reads as a
// clean end-of-file to the caller: a title that loads through preadv gets empty
// buffers and no error to notice it by. Offsets advance across the segments and
// the file position is left alone, like pread/pwrite.
int64_t PS4ABI sys_preadv(uint32_t fd, const void *iov, int iovcnt,
                          int64_t offset) {
  auto *segs = static_cast<const sce_iovec *>(iov);
  if (!segs || iovcnt < 0 || offset < 0)
    return -SysError::eINVAL;
  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  const int64_t saved = d->lseek(0, kSeekCur);
  int64_t total = 0;
  for (int i = 0; i < iovcnt; ++i) {
    d->lseek(offset + total, kSeekSet);
    int64_t r = d->read(segs[i].iov_base, segs[i].iov_len);
    if (r < 0) {
      if (saved >= 0)
        d->lseek(saved, kSeekSet);
      return total ? total : r;
    }
    total += r;
    if (static_cast<size_t>(r) < segs[i].iov_len)
      break;  // short read: end of file
  }
  if (saved >= 0)
    d->lseek(saved, kSeekSet);
  return total;
}

int64_t PS4ABI sys_pwritev(uint32_t fd, const void *iov, int iovcnt,
                           int64_t offset) {
  auto *segs = static_cast<const sce_iovec *>(iov);
  if (!segs || iovcnt < 0 || offset < 0)
    return -SysError::eINVAL;
  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  const int64_t saved = d->lseek(0, kSeekCur);
  int64_t total = 0;
  for (int i = 0; i < iovcnt; ++i) {
    d->lseek(offset + total, kSeekSet);
    int64_t r = d->write(segs[i].iov_base, segs[i].iov_len);
    if (r < 0) {
      if (saved >= 0)
        d->lseek(saved, kSeekSet);
      return total ? total : r;
    }
    total += r;
    if (static_cast<size_t>(r) < segs[i].iov_len)
      break;
  }
  if (saved >= 0)
    d->lseek(saved, kSeekSet);
  return total;
}

// We have no pollable fds. Returning 0 (zero ready) immediately would turn a
// timed poll into a busy-spin, so honour the caller's timeout by sleeping it
// first (capped). timeout is in milliseconds; negative means "wait forever",
// which we treat as the cap rather than hanging.
int PS4ABI sys_poll(void *fds, uint32_t nfds, int timeout) {
  int ms = timeout;
  if (ms < 0 || ms > 50)
    ms = 50;
  if (ms > 0)
    ::usleep(static_cast<useconds_t>(ms) * 1000);
  return 0;
}

int PS4ABI sys_select(int nfds, void *readfds, void *writefds, void *exceptfds,
                      void *timeout) {
  return 0; // zero ready descriptors
}

int PS4ABI sys_openat(int fd, const char *path, uint32_t flags, uint32_t mode) {
  return sys_open(path, flags, mode);
}

int PS4ABI sys_chdir(const char *path) { return 0; }
int PS4ABI sys_fchdir(uint32_t fd) { return 0; }

// The host tree stays read-only. We report success so installers and savedata
// setup proceed, but log every call: if a title relies on a file it "created"
// here being readable back, that read returns stale VFS data and this trace is
// the only sign of why.
int PS4ABI sys_unlink(const char *path) {
  // Under a writable mount (savedata, /download0) do the real thing: a title
  // that rewrites a file by unlink+create reads back stale content otherwise.
  if (path && vfs::removeFile(path))
    return 0;
  std::printf("[vfs] unlink('%s') ignored (read-only host)\n",
              path ? path : "(null)");
  return 0;
}
int PS4ABI sys_rmdir(const char *path) {
  std::printf("[vfs] rmdir('%s') ignored (read-only host)\n",
              path ? path : "(null)");
  return 0;
}
int PS4ABI sys_mkdir(const char *path, uint32_t mode) {
  (void)mode;
  // Real directory creation under a writable mount (savedata); otherwise a
  // no-op success as before (the read-only host content the game expects to
  // exist already does).
  if (path && vfs::makeDir(path)) {
    if (kVfsTrace)
      std::fprintf(stderr, "[vfs] mkdir('%s') -> host\n", path);
    return 0;
  }
  return 0;
}
int PS4ABI sys_rename(const char *from, const char *to) {
  base::String hf = from ? vfs::resolveWritable(from) : base::String();
  base::String ht = to ? vfs::resolveWritable(to) : base::String();
  if (!hf.empty() && !ht.empty() && std::rename(hf.c_str(), ht.c_str()) == 0)
    return 0;
  std::printf("[vfs] rename('%s' -> '%s') ignored (read-only host)\n",
              from ? from : "(null)", to ? to : "(null)");
  return 0;
}

// sys_unlinkat (503): flag bit 0x800 (AT_REMOVEDIR) means rmdir, else unlink.
int PS4ABI sys_unlinkat(int fd, const char *path, int flag) {
  if (flag & 0x800)
    return sys_rmdir(path);
  return sys_unlink(path);
}

// sys_mkdirat (496): identical to mkdir (the dirfd is always AT_FDCWD here).
int PS4ABI sys_mkdirat(int fd, const char *path, uint32_t mode) {
  return sys_mkdir(path, mode);
}

// sys_renameat (501): identical to rename (both dirfds are AT_FDCWD here).
int PS4ABI sys_renameat(int fd_old, const char *old, int fd_new,
                        const char *new_) {
  return sys_rename(old, new_);
}

int64_t PS4ABI sys_getdirentries(uint32_t fd, void *buf, size_t nbytes,
                                 int64_t *basep) {
  auto *d = fdToDevice(fd);
  if (!d) {
    if (kVfsTrace)
      std::fprintf(stderr, "[getdirentries] fd=%u BADF\n", fd);
    return -SysError::eBADF;
  }
  // The kernel validates buflen and returns EINVAL on a negative value.
  if (nbytes == 0)
    return -SysError::eINVAL;
  int64_t r = d->getdents(buf, nbytes);
  // On success the kernel writes the next directory seek offset to *basep so
  // the caller can resume a partial enumeration. Use the byte count consumed
  // as the cookie: a subsequent getdirentries at this offset reads the next
  // chunk. Our dirDevice serves all entries on the first call and returns EOF
  // after, so the cookie is simply the total returned.
  if (r >= 0 && basep)
    *basep = r;
  if (kVfsTrace)
    std::fprintf(stderr, "[getdirentries] fd=%u buf=%p n=%zu -> %lld basep=%lld\n",
                 fd, buf, nbytes, (long long)r,
                 basep ? (long long)*basep : -1);
  return r;
}

int PS4ABI sys_closefrom(uint32_t lowfd) { return 0; }

} // namespace krnl
