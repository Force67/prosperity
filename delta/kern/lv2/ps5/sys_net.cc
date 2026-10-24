/*
 * PS4Delta : PS4 emulation and research project
 *
 * Prospero socket syscalls that differ from the Orbis ones, and the FreeBSD 11
 * accept4, bindat and connectat.
 */

#include "base/arch.h"
#include "guest_abi.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/ps5/handlers.h"
#include "kern/lv2/sys_net.h"
#include "kern/process.h"
#include "kern/ps4/dev/socket_dev.h"

namespace kern {
namespace {
constexpr int kAtFdcwd = -100;
constexpr int kSockCloexec = 0x10000000;
constexpr int kSockNonblock = 0x20000000;
}  // namespace

int PS4ABI Ps5Accept(int fd, void* addr, u32* addrlen) {
  return Ps5Accept4(fd, addr, addrlen, 0);
}

int PS4ABI Ps5Accept4(int fd, void* /*addr*/, u32* /*addrlen*/, int flags) {
  if (flags & ~(kSockCloexec | kSockNonblock))
    return -SysError::eINVAL;
  if (!FdToSocket(static_cast<u32>(fd)))
    return NotSocket(fd);
  return -SysError::eINVAL;  // not a listening socket
}

// Relative to a directory fd only local sockets can bind or connect; from
// AT_FDCWD these are plain bind and connect.
int PS4ABI Ps5Bindat(int dirfd, int fd, const void* addr, u32 addrlen) {
  if (dirfd == kAtFdcwd)
    return sys_bind(fd, addr, addrlen);
  if (!FdToSocket(static_cast<u32>(fd)))
    return NotSocket(fd);
  return -SysError::eOPNOTSUPP;
}

int PS4ABI Ps5Connectat(int dirfd, int fd, const void* addr, u32 addrlen) {
  if (dirfd == kAtFdcwd)
    return sys_connect(fd, addr, addrlen);
  if (!FdToSocket(static_cast<u32>(fd)))
    return NotSocket(fd);
  return -SysError::eOPNOTSUPP;
}

// Aborts pending operations on a socket; ours never block in the kernel.
int PS4ABI Ps5Netabort(int fd, int /*flags*/) {
  return FdToSocket(static_cast<u32>(fd)) ? 0 : NotSocket(fd);
}

}  // namespace kern
