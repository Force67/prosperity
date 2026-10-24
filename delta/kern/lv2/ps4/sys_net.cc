/*
 * PS4Delta : PS4 emulation and research project
 *
 * Orbis socket syscalls that differ from the Prospero ones: accept and
 * netabort.
 */

#include "base/arch.h"
#include "guest_abi.h"

#include "kern/lv2/error_table.h"
#include "kern/lv2/ps4/handlers.h"
#include "kern/lv2/sys_net.h"
#include "kern/process.h"
#include "kern/ps4/dev/socket_dev.h"

namespace kern {
namespace {
// netabort flag: the descriptor is an epoll (equeue), not a socket.
constexpr int kNetabortEpoll = 0x1000000;
constexpr int kNetabortMask = 0x3;
}  // namespace

int PS4ABI Ps4Accept(int fd, void* /*addr*/, u32* /*addrlen*/) {
  if (!FdToSocket(static_cast<u32>(fd)))
    return NotSocket(fd);
  return -SysError::eINVAL;  // not a listening socket
}

// Aborts pending operations on a socket (or, with the epoll flag, on an
// equeue); ours never block in the kernel. Flags past the two abort bits are
// refused.
int PS4ABI Ps4Netabort(int fd, int flags) {
  if (flags & kNetabortEpoll) {
    auto* proc = Process::GetActive();
    if (!proc || fd < 0)
      return -SysError::eBADF;
    auto obj = proc->GetObjTable().Get(static_cast<u32>(fd));
    if (!obj || obj->type() != Object::Type::kEqueue)
      return -SysError::eBADF;
    return (flags & ~kNetabortEpoll) > kNetabortMask ? -SysError::eINVAL : 0;
  }
  if (!FdToSocket(static_cast<u32>(fd)))
    return NotSocket(fd);
  return flags > kNetabortMask ? -SysError::eINVAL : 0;
}

}  // namespace kern
