/*
 * PS4Delta : PS4 emulation and research project
 *
 * Prospero file and descriptor syscalls: the FreeBSD 11 calls an Orbis kernel
 * lacks (pipe2, ppoll, futimens, the *at attribute calls, ...).
 */

#include "base/arch.h"
#include "guest_abi.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/ps5/handlers.h"
#include "kern/lv2/sys_signal.h"
#include "kern/lv2/sys_time.h"
#include "kern/lv2/sys_vfs.h"
#include "kern/process.h"
#include "kern/ps4/dev/pipe_dev.h"

namespace kern {
namespace {
constexpr int kAtFdcwd = -100;
constexpr int kAtSymlinkNofollow = 0x200;
constexpr int kONonblock = 0x4;
constexpr int kOCloexec = 0x100000;

// Paths are resolved from the root whatever the directory fd (as sys_openat
// does); an fd that names nothing is still EBADF.
int AtPathExists(int fd, const char* path, int flag) {
  if (flag & ~kAtSymlinkNofollow)
    return -SysError::eINVAL;
  if (fd != kAtFdcwd && GuestFdError(fd) && path && *path != '/')
    return -SysError::eBADF;
  return GuestPathError(path);
}
}  // namespace

int PS4ABI Ps5Chflagsat(int fd, const char* path, u64 /*flags*/, int atflag) {
  return AtPathExists(fd, path, atflag);
}

int PS4ABI Ps5Fchmodat(int fd, const char* path, u32 /*mode*/, int flag) {
  return AtPathExists(fd, path, flag);
}

int PS4ABI
Ps5Fchownat(int fd, const char* path, int /*uid*/, int /*gid*/, int flag) {
  return AtPathExists(fd, path, flag);
}

int PS4ABI Ps5Futimesat(int fd, const char* path, const void* /*times*/) {
  return AtPathExists(fd, path, 0);
}

int PS4ABI Ps5Futimens(int fd, const void* /*times*/) {
  return GuestFdError(fd);
}

int PS4ABI Ps5Utimensat(int fd,
                        const char* path,
                        const void* /*times*/,
                        int flag) {
  return AtPathExists(fd, path, flag);
}

int PS4ABI Ps5Pipe2(int* fds, int flags) {
  if (flags & ~(kONonblock | kOCloexec))
    return -SysError::eINVAL;
  auto* proc = Process::GetActive();
  if (!proc)
    return -SysError::eINVAL;
  if (!fds)
    return -SysError::eFAULT;
  u32 reader, writer;
  CreatePipe(proc->GetObjTable(), reader, writer);
  fds[0] = static_cast<int>(reader);
  fds[1] = static_cast<int>(writer);
  return 0;
}

int PS4ABI Ps5Ppoll(void* fds, u32 nfds, const void* ts, const void* sigmask) {
  int timeout_ms = -1;
  if (const auto* t = static_cast<const sce_timespec*>(ts)) {
    if (t->tv_sec < 0 || t->tv_nsec < 0 || t->tv_nsec >= 1'000'000'000)
      return -SysError::eINVAL;
    const i64 ms = t->tv_sec * 1000 + (t->tv_nsec + 999'999) / 1'000'000;
    timeout_ms = ms > 0x7fffffff ? 0x7fffffff : static_cast<int>(ms);
  }
  if (const int err = BeginWaitMask(sigmask))
    return err;
  const int r = sys_poll(fds, nfds, timeout_ms);
  EndWaitMask(sigmask);
  return r;
}

// openat(AT_FDCWD, ...) that a signal may interrupt; opening never blocks here.
int PS4ABI Ps5Openintr(const char* path, u32 flags, u32 mode) {
  return sys_open(path, flags, mode);
}

}  // namespace kern
