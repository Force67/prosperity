/*
 * PS4Delta : PS4 emulation and research project
 *
 * See socket_dev.h. The only real work here is the sockaddr translation: a
 * FreeBSD sockaddr leads with a 1-byte length and a 1-byte family, where Linux
 * has a 2-byte family and no length, so the guest's bytes cannot be handed to
 * the host unchanged.
 */

#include <cerrno>
#include "base/arch.h"
#include <cstring>

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "kern/proc.h"
#include "socket_dev.h"

namespace krnl {
namespace {

// FreeBSD address families, as the guest sees them.
constexpr u8 kBsdAfInet = 2;
constexpr u8 kBsdAfInet6 = 28;

struct BsdSockaddrIn {
  u8 len, family;
  u16 port;
  u32 addr;
  u8 zero[8];
};

struct BsdSockaddrIn6 {
  u8 len, family;
  u16 port;
  u32 flowinfo;
  u8 addr[16];
  u32 scopeId;
};

// Guest -> host. Returns the host length, or 0 if the family isn't one we serve.
socklen_t toHost(const void *guest, u32 guestLen, sockaddr_storage &out) {
  std::memset(&out, 0, sizeof(out));
  if (!guest || guestLen < 2)
    return 0;
  const u8 family = static_cast<const u8 *>(guest)[1];
  if (family == kBsdAfInet && guestLen >= sizeof(BsdSockaddrIn)) {
    const auto *g = static_cast<const BsdSockaddrIn *>(guest);
    auto *h = reinterpret_cast<sockaddr_in *>(&out);
    h->sin_family = AF_INET;
    h->sin_port = g->port;  // already network order on both sides
    h->sin_addr.s_addr = g->addr;
    return sizeof(sockaddr_in);
  }
  if (family == kBsdAfInet6 && guestLen >= sizeof(BsdSockaddrIn6)) {
    const auto *g = static_cast<const BsdSockaddrIn6 *>(guest);
    auto *h = reinterpret_cast<sockaddr_in6 *>(&out);
    h->sin6_family = AF_INET6;
    h->sin6_port = g->port;
    h->sin6_flowinfo = g->flowinfo;
    std::memcpy(&h->sin6_addr, g->addr, sizeof(g->addr));
    h->sin6_scope_id = g->scopeId;
    return sizeof(sockaddr_in6);
  }
  return 0;
}

// Host -> guest. Returns the number of bytes written.
u32 toGuest(const sockaddr_storage &in, void *guest, u32 cap) {
  if (!guest)
    return 0;
  if (in.ss_family == AF_INET && cap >= sizeof(BsdSockaddrIn)) {
    const auto *h = reinterpret_cast<const sockaddr_in *>(&in);
    auto *g = static_cast<BsdSockaddrIn *>(guest);
    std::memset(g, 0, sizeof(*g));
    g->len = sizeof(*g);
    g->family = kBsdAfInet;
    g->port = h->sin_port;
    g->addr = h->sin_addr.s_addr;
    return sizeof(*g);
  }
  if (in.ss_family == AF_INET6 && cap >= sizeof(BsdSockaddrIn6)) {
    const auto *h = reinterpret_cast<const sockaddr_in6 *>(&in);
    auto *g = static_cast<BsdSockaddrIn6 *>(guest);
    std::memset(g, 0, sizeof(*g));
    g->len = sizeof(*g);
    g->family = kBsdAfInet6;
    g->port = h->sin6_port;
    g->flowinfo = h->sin6_flowinfo;
    std::memcpy(g->addr, &h->sin6_addr, sizeof(g->addr));
    g->scopeId = h->sin6_scope_id;
    return sizeof(*g);
  }
  return 0;
}

int fromErrno() { return -errno; }

}  // namespace

socketDevice::socketDevice(proc *p, int hostFd, int guestFamily)
    : device(p), fd_(hostFd), family_(guestFamily) {}

socketDevice::~socketDevice() {
  if (fd_ >= 0)
    ::close(fd_);
}

int socketDevice::bind(const void *guestAddr, u32 len) {
  sockaddr_storage sa;
  socklen_t n = toHost(guestAddr, len, sa);
  if (!n)
    return -SysError::eINVAL;
  return ::bind(fd_, reinterpret_cast<sockaddr *>(&sa), n) == 0 ? 0 : fromErrno();
}

int socketDevice::getsockname(void *guestAddr, u32 *len) {
  sockaddr_storage sa;
  socklen_t n = sizeof(sa);
  if (::getsockname(fd_, reinterpret_cast<sockaddr *>(&sa), &n) != 0)
    return fromErrno();
  u32 wrote = toGuest(sa, guestAddr, len ? *len : 0);
  if (len)
    *len = wrote;
  return 0;
}

i64 socketDevice::sendto(const void *buf, size_t len, int flags,
                             const void *guestAddr, u32 addrLen) {
  sockaddr_storage sa;
  socklen_t n = toHost(guestAddr, addrLen, sa);
  ssize_t r = n ? ::sendto(fd_, buf, len, flags,
                           reinterpret_cast<sockaddr *>(&sa), n)
                : ::send(fd_, buf, len, flags);
  return r >= 0 ? r : fromErrno();
}

i64 socketDevice::recvfrom(void *buf, size_t len, int flags,
                               void *guestAddr, u32 *addrLen) {
  sockaddr_storage sa;
  socklen_t n = sizeof(sa);
  ssize_t r = ::recvfrom(fd_, buf, len, flags,
                         reinterpret_cast<sockaddr *>(&sa), &n);
  if (r < 0)
    return fromErrno();
  if (guestAddr && addrLen)
    *addrLen = toGuest(sa, guestAddr, *addrLen);
  return r;
}

// Sony's own socket controls, in the 'P' ioctl group rather than BSD's. Both
// are 36-byte IOC_IN blocks with no reply, issued once each by libSceNet while
// it configures the socket it opens for its own bring-up. Accepting them is the
// whole contract: there is nothing for us to hand back, and refusing one made
// libSceNet treat its init as failed.
i32 socketDevice::ioctl(u32 command, void *args) {
  switch (command) {
    case 0x802450c8:
    case 0x802450c9:
      return 0;
    default:
      return device::ioctl(command, args);
  }
}

socketDevice *fdToSocket(u32 fd) {
  auto *p = proc::getActive();
  if (!p)
    return nullptr;
  auto *obj = p->getObjTable().get(fd);
  if (!obj || obj->type() != kObject::oType::device)
    return nullptr;
  return dynamic_cast<socketDevice *>(static_cast<device *>(obj));
}

}  // namespace krnl
