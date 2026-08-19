// Copyright (C) Force67 2019

#include <base.h>
#include "base/arch.h"
#include <base/logging.h>
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
constexpr i32 kBsdAfInet = 2;
constexpr i32 kBsdAfInet6 = 28;
constexpr i32 kBsdSockDgram = 2;
constexpr i32 kSceSockDgramP2p = 6;
}  // namespace
#include "kern/crash.h"
#include <cstring>
#include <utl/mem.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kNetTrace, "DELTA_NET_TRACE", false);

// The reply layout of sceNetGetSockInfo, and the FreeBSD sockaddr_in the guest
// hands back from getsockname (a leading length byte, then the family).
struct sceNetSockaddrIn {
  u8 len, family;
  u16 port;
  u32 addr;
  u8 zero[8];
};

struct sceNetSockInfo {
  char name[32];
  i32 pid;
  i32 s;
  i8 socketType;
  i8 policy;
  i16 localPort;
  u32 localAddr;
  i16 foreignPort;
  u32 foreignAddr;
  u32 recvQueueLength;
  u32 sendQueueLength;
  u8 boundInterface;
  u64 options;
  i32 flags;
  i32 sendBufferSize;
  i32 recvBufferSize;
  i32 error;
  i32 state;
};
}  // namespace

namespace krnl {
int PS4ABI sys_netcontrol(u32 fd, u32 op, void* buffer,
        u32 size) {

    if (kNetTrace)
      BASE_LOGI("netctl", "fd={} op={:#x} buf={:p} size={}", (int)fd, op,
                buffer, size);

    if (size > 160)
    return -SysError::eINVAL;

    if (op == 20) {
      *static_cast<u32 *>(buffer) = 0xF00D;
      return 0;
    }

    return -SysError::eINVAL;
}

// Same policy as sys_socket below. Returning a fake success here hands the
// guest fd 0: Bloodborne's net thread then spins on sceNetGetsockname(0)
// getting EBADF forever instead of taking its offline path.
int PS4ABI sys_socketex(const char* name, i32 domain, i32 type,
    i32 protocol) {
  if (kNetTrace)
    BASE_LOGI("net", "socketex name='{}' domain={} type={} proto={}",
              name ? name : "", domain, type, protocol);
  return sys_socket(domain, type, protocol);
}

// Datagram sockets get a real host socket: a title that uses one for LAN
// discovery also polls it for readability, and a stub fd it can never read from
// wedges whatever waits on that poll. Everything else (the AF_UNIX sockets the
// guest uses to reach NP/ShellCore, and TCP) is still refused up front, so those
// callers keep falling back to their offline path instead of blocking on a
// service process we do not host.
int PS4ABI sys_socket(i32 domain, i32 type, i32 protocol) {
  const int hostDomain = domain == kBsdAfInet    ? AF_INET
                         : domain == kBsdAfInet6 ? AF_INET6
                                                 : -1;
  // SOCK_DGRAM_P2P counts as a datagram: libSceNet's module_start opens one
  // ("SceNetInit") and treats a failure as init failure, after which every net
  // API returns 0x804101c8 and the PS5 Np stack can't start.
  if (hostDomain != -1 && (type == kBsdSockDgram || type == kSceSockDgramP2p)) {
    int fd = ::socket(hostDomain, SOCK_DGRAM, 0);
    if (fd >= 0) {
      auto *dev = new socketDevice(proc::getActive(), fd, domain);
      dev->setName("socket");  // so a diagnostic can say what it landed on
      BASE_LOGI("net", "socket(domain={} type={}) -> fd={} (host {})", domain,
                type, dev->handle(), fd);
      return static_cast<int>(dev->handle());
    }
  }
  BASE_LOGI("net", "socket(domain={} type={} proto={}) -> EAFNOSUPPORT", domain,
            type, protocol);
  return -SysError::eAFNOSUPPORT;
}

// sceNetGetSockInfo(s, info, n, flags): report the kernel's view of a socket.
// The reply is a count of filled entries, so zero entries is a legitimate
// answer; what is not legitimate is the old stub, which returned "0 entries"
// without touching the buffer. libSceNet reads the first entry regardless of
// the count and calls through a pointer it finds there, so GTA:SA's net thread
// jumped into stack garbage the moment it asked.
int PS4ABI sys_netgetsockinfo(i32 fd, void *info, i32 n, i32 flags) {
  if (kNetTrace)
    BASE_LOGI("net", "getsockinfo fd={} info={:p} n={} flags={:#x}", fd, info, n,
              flags);
  if (!info || n <= 0)
    return -SysError::eINVAL;

  const size_t span = sizeof(sceNetSockInfo) * static_cast<size_t>(n);
  if (!utl::isMemoryRangeMapped(info, span))
    return -SysError::eFAULT;
  std::memset(info, 0, span);

  auto *s = fdToSocket(fd);
  if (!s)
    return -SysError::eBADF;

  auto *out = static_cast<sceNetSockInfo *>(info);
  std::snprintf(out->name, sizeof(out->name), "%s", "socket");
  out->pid = 0x1337;  // the pid sys_getpid reports
  out->s = fd;
  out->socketType = 2;  // SOCK_DGRAM: the only kind sys_socket hands out
  out->state = 1;       // bound/open, the only state we model

  u32 addrLen = sizeof(sceNetSockaddrIn);
  sceNetSockaddrIn local{};
  if (s->getsockname(&local, &addrLen) == 0) {
    out->localPort = local.port;
    out->localAddr = local.addr;
  }
  return 1;
}

int PS4ABI sys_bind(i32 fd, const void *addr, u32 addrlen) {
  auto *s = fdToSocket(fd);
  return s ? s->bind(addr, addrlen) : 0;
}

int PS4ABI sys_getsockname(i32 fd, void *addr, u32 *addrlen) {
  auto *s = fdToSocket(fd);
  if (!s) {
    if (addr && addrlen)
      std::memset(addr, 0, *addrlen);
    return -SysError::eBADF;
  }
  return s->getsockname(addr, addrlen);
}

int PS4ABI sys_socketclose(i32 fd) {
  auto *s = fdToSocket(fd);
  if (s)
    s->releaseHandle();
  return 0;
}

int PS4ABI sys_connect(i32 fd, const void *addr, u32 addrlen) {
  return -SysError::eCONNREFUSED;
}

int PS4ABI sys_recvmsg(i32 fd, void *msg, i32 flags) {
  return -SysError::eBADF;
}

// With no network stack every socket fd is invalid; fail sends/receives like a
// closed socket so callers error out. The old null_handler returned 0, which a
// sender reads as "0 bytes sent" and retries forever (Shadow of the Tomb
// Raider's telemetry spun millions of sendto calls and wedged its boot).
i64 PS4ABI sys_sendto(i32 fd, const void *buf, size_t len,
                          i32 flags, const void *to, u32 tolen) {
  auto *s = fdToSocket(fd);
  return s ? s->sendto(buf, len, flags, to, tolen) : -SysError::eBADF;
}

i64 PS4ABI sys_recvfrom(i32 fd, void *buf, size_t len, i32 flags,
                            void *from, u32 *fromlen) {
  auto *s = fdToSocket(fd);
  return s ? s->recvfrom(buf, len, flags, from, fromlen) : -SysError::eBADF;
}
}