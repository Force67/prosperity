
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

// EINVAL is the POSIX answer for "not a symbolic link", which is what callers
// probing our (symlink-free) VFS expect.
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

int PS4ABI sys_fcntl(uint32_t fd, int cmd, int64_t arg) {
  enum { F_DUPFD = 0, F_GETFD = 1, F_SETFD = 2, F_GETFL = 3, F_SETFL = 4 };
  switch (cmd) {
  case F_GETFL:
  case F_SETFL:
  case F_GETFD:
  case F_SETFD:
    return 0;
  case F_DUPFD:
    return -SysError::eOPNOTSUPP; // the object table has no descriptor dup
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

int64_t PS4ABI sys_pread(uint32_t fd, void *buf, size_t nbytes, int64_t offset) {
  auto *d = fdToDevice(fd);
  if (!d) {
    if (std::getenv("DELTA_RDALL"))
      std::fprintf(stderr, "[pread] fd=%u off=%lld -> EBADF (no device)\n", fd, (long long)offset);
    return -SysError::eBADF;
  }
  int64_t saved = d->lseek(0, kSeekCur);
  d->lseek(offset, kSeekSet);
  int64_t r = d->read(buf, nbytes);
  if (saved >= 0)
    d->lseek(saved, kSeekSet);
  if (std::getenv("DELTA_RDALL")) {
    uint32_t f4 = 0;
    if (buf && r >= 4) f4 = *reinterpret_cast<const uint32_t *>(buf);
    std::fprintf(stderr, "[pread] t=%ld fd=%u off=%lld nbytes=%#zx -> %lld buf=%p first4=%08x\n",
                 (long)gettid(), fd, (long long)offset, (size_t)nbytes, (long long)r, buf, f4);
  }
  // DELTA_QARBUF: where does streamed .qar data land? Reports the destination
  // buffer for reads on a *.qar fd, so we can tell whether textures stream into
  // a GPU-mapped region (0x81xx, directly bindable) or a low staging buffer that
  // still needs a copy/commit step the engine never performs.
  if (fd < 8192 && g_qarFd[fd] && std::getenv("DELTA_QARBUF")) {
    std::fprintf(stderr,
                 "[qarbuf] fd=%u off=%lld nbytes=%#zx -> %lld buf=%p\n", fd,
                 (long long)offset, (size_t)nbytes, (long long)r, buf);
  }
  // DELTA_IOPROGRESS: throttled per-fd streaming high-water mark. FOX/FIOS2 streams
  // large world archives via pread; this shows whether that streaming is advancing
  // (offset climbing) or has completed/stalled, without the DELTA_RDALL firehose.
  // At most one line per fd per ~2 s; prints the current + max offset and MB/s since
  // the last line so a long headless load can be tracked to completion.
  if (std::getenv("DELTA_IOPROGRESS")) {
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
    if (std::getenv("DELTA_VFS_TRACE"))
      std::fprintf(stderr, "[vfs] mkdir('%s') -> host\n", path);
    return 0;
  }
  return 0;
}
int PS4ABI sys_rename(const char *from, const char *to) {
  std::printf("[vfs] rename('%s' -> '%s') ignored (read-only host)\n",
              from ? from : "(null)", to ? to : "(null)");
  return 0;
}

int64_t PS4ABI sys_getdirentries(uint32_t fd, void *buf, size_t nbytes,
                                 int64_t *basep) {
  auto *d = fdToDevice(fd);
  if (!d)
    return -SysError::eBADF;
  // basep is an in/out seek cookie; the dir device tracks its own cursor, so we
  // leave whatever the caller passed untouched.
  return d->getdents(buf, nbytes);
}

int PS4ABI sys_closefrom(uint32_t lowfd) { return 0; }

} // namespace krnl
