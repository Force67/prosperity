// Copyright (C) Force67 2019

#include <base.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/socket.h>

#include "kern/proc.h"
#include "kern/ps4/dev/socket_dev.h"
#include "sys_net.h"
#include "error_table.h"

namespace {
// FreeBSD constants, as the guest passes them.
constexpr int32_t kBsdAfInet = 2;
constexpr int32_t kBsdAfInet6 = 28;
constexpr int32_t kBsdSockDgram = 2;
}  // namespace
#include "kern/crash.h"
#include <cstring>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kNetTrace, "DELTA_NET_TRACE", false);
}  // namespace

namespace krnl {
int PS4ABI sys_netcontrol(uint32_t fd, uint32_t op, void* buffer,
        uint32_t size) {

    if (kNetTrace)
      std::fprintf(stderr, "[netctl] fd=%d op=%#x buf=%p size=%u\n", (int)fd,
                   op, buffer, size);

    if (size > 160)
    return -SysError::eINVAL;

    if (op == 20) {
      *static_cast<uint32_t *>(buffer) = 0xF00D;
      return 0;
    }

    return -SysError::eINVAL;
}

int PS4ABI sys_socketex(const char* name, int32_t domain, int32_t type,
    int32_t protocol) {
    // TOO lazy for now
  return 0;
}

// Datagram sockets get a real host socket: a title that uses one for LAN
// discovery also polls it for readability, and a stub fd it can never read from
// wedges whatever waits on that poll. Everything else (the AF_UNIX sockets the
// guest uses to reach NP/ShellCore, and TCP) is still refused up front, so those
// callers keep falling back to their offline path instead of blocking on a
// service process we do not host.
int PS4ABI sys_socket(int32_t domain, int32_t type, int32_t protocol) {
  const int hostDomain = domain == kBsdAfInet    ? AF_INET
                         : domain == kBsdAfInet6 ? AF_INET6
                                                 : -1;
  if (hostDomain != -1 && type == kBsdSockDgram) {
    int fd = ::socket(hostDomain, SOCK_DGRAM, 0);
    if (fd >= 0) {
      auto *dev = new socketDevice(proc::getActive(), fd, domain);
      std::printf("[net] socket(domain=%d type=%d) -> fd=%u (host %d)\n", domain,
                  type, dev->handle(), fd);
      return static_cast<int>(dev->handle());
    }
  }
  std::printf("[net] socket(domain=%d type=%d proto=%d) -> EAFNOSUPPORT\n",
              domain, type, protocol);
  return -SysError::eAFNOSUPPORT;
}

int PS4ABI sys_bind(int32_t fd, const void *addr, uint32_t addrlen) {
  auto *s = fdToSocket(fd);
  return s ? s->bind(addr, addrlen) : 0;
}

int PS4ABI sys_getsockname(int32_t fd, void *addr, uint32_t *addrlen) {
  auto *s = fdToSocket(fd);
  if (!s) {
    if (addr && addrlen)
      std::memset(addr, 0, *addrlen);
    return -SysError::eBADF;
  }
  return s->getsockname(addr, addrlen);
}

int PS4ABI sys_socketclose(int32_t fd) {
  auto *s = fdToSocket(fd);
  if (s)
    s->releaseHandle();
  return 0;
}

int PS4ABI sys_connect(int32_t fd, const void *addr, uint32_t addrlen) {
  return -SysError::eCONNREFUSED;
}

int PS4ABI sys_recvmsg(int32_t fd, void *msg, int32_t flags) {
  return -SysError::eBADF;
}

// With no network stack every socket fd is invalid; fail sends/receives like a
// closed socket so callers error out. The old null_handler returned 0, which a
// sender reads as "0 bytes sent" and retries forever (Shadow of the Tomb
// Raider's telemetry spun millions of sendto calls and wedged its boot).
int64_t PS4ABI sys_sendto(int32_t fd, const void *buf, size_t len,
                          int32_t flags, const void *to, uint32_t tolen) {
  auto *s = fdToSocket(fd);
  return s ? s->sendto(buf, len, flags, to, tolen) : -SysError::eBADF;
}

int64_t PS4ABI sys_recvfrom(int32_t fd, void *buf, size_t len, int32_t flags,
                            void *from, uint32_t *fromlen) {
  auto *s = fdToSocket(fd);
  return s ? s->recvfrom(buf, len, flags, from, fromlen) : -SysError::eBADF;
}
}