/*
 * PS4Delta : PS4 emulation and research project
 *
 * See socket_dev.h. The only real work here is the sockaddr translation: a
 * FreeBSD sockaddr leads with a 1-byte length and a 1-byte family, where Linux
 * has a 2-byte family and no length, so the guest's bytes cannot be handed to
 * the host unchanged.
 */

#include <cerrno>
#include <cstring>
#include "base/arch.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "kern/process.h"
#include "kern/ps4/dev/socket_dev.h"

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
  u32 scope_id;
};

// Guest -> host. Returns the host length, or 0 if the family isn't one we
// serve.
socklen_t ToHost(const void* guest, u32 guest_len, sockaddr_storage& out) {
  std::memset(&out, 0, sizeof(out));
  if (!guest || guest_len < 2)
    return 0;
  const u8 family = static_cast<const u8*>(guest)[1];
  if (family == kBsdAfInet && guest_len >= sizeof(BsdSockaddrIn)) {
    const auto* g = static_cast<const BsdSockaddrIn*>(guest);
    auto* h = reinterpret_cast<sockaddr_in*>(&out);
    h->sin_family = AF_INET;
    h->sin_port = g->port;  // already network order on both sides
    h->sin_addr.s_addr = g->addr;
    return sizeof(sockaddr_in);
  }
  if (family == kBsdAfInet6 && guest_len >= sizeof(BsdSockaddrIn6)) {
    const auto* g = static_cast<const BsdSockaddrIn6*>(guest);
    auto* h = reinterpret_cast<sockaddr_in6*>(&out);
    h->sin6_family = AF_INET6;
    h->sin6_port = g->port;
    h->sin6_flowinfo = g->flowinfo;
    std::memcpy(&h->sin6_addr, g->addr, sizeof(g->addr));
    h->sin6_scope_id = g->scope_id;
    return sizeof(sockaddr_in6);
  }
  return 0;
}

// Host -> guest. Returns the number of bytes written.
u32 ToGuest(const sockaddr_storage& in, void* guest, u32 cap) {
  if (!guest)
    return 0;
  if (in.ss_family == AF_INET && cap >= sizeof(BsdSockaddrIn)) {
    const auto* h = reinterpret_cast<const sockaddr_in*>(&in);
    auto* g = static_cast<BsdSockaddrIn*>(guest);
    std::memset(g, 0, sizeof(*g));
    g->len = sizeof(*g);
    g->family = kBsdAfInet;
    g->port = h->sin_port;
    g->addr = h->sin_addr.s_addr;
    return sizeof(*g);
  }
  if (in.ss_family == AF_INET6 && cap >= sizeof(BsdSockaddrIn6)) {
    const auto* h = reinterpret_cast<const sockaddr_in6*>(&in);
    auto* g = static_cast<BsdSockaddrIn6*>(guest);
    std::memset(g, 0, sizeof(*g));
    g->len = sizeof(*g);
    g->family = kBsdAfInet6;
    g->port = h->sin6_port;
    g->flowinfo = h->sin6_flowinfo;
    std::memcpy(g->addr, &h->sin6_addr, sizeof(g->addr));
    g->scope_id = h->sin6_scope_id;
    return sizeof(*g);
  }
  return 0;
}

int FromErrno() {
  return -errno;
}

}  // namespace

SocketDevice::SocketDevice(ObjectTable& objects, int host_fd, int guest_family)
    : Device(objects), fd_(host_fd), family_(guest_family) {}

SocketDevice::~SocketDevice() {
  if (fd_ >= 0)
    ::close(fd_);
}

int SocketDevice::Bind(const void* guest_addr, u32 len) {
  sockaddr_storage sa;
  socklen_t n = ToHost(guest_addr, len, sa);
  if (!n)
    return -SysError::eINVAL;
  return ::bind(fd_, reinterpret_cast<sockaddr*>(&sa), n) == 0 ? 0
                                                               : FromErrno();
}

int SocketDevice::Getsockname(void* guest_addr, u32* len) {
  sockaddr_storage sa;
  socklen_t n = sizeof(sa);
  if (::getsockname(fd_, reinterpret_cast<sockaddr*>(&sa), &n) != 0)
    return FromErrno();
  u32 wrote = ToGuest(sa, guest_addr, len ? *len : 0);
  if (len)
    *len = wrote;
  return 0;
}

i64 SocketDevice::Sendto(const void* buf,
                         size_t len,
                         int flags,
                         const void* guest_addr,
                         u32 addr_len) {
  sockaddr_storage sa;
  socklen_t n = ToHost(guest_addr, addr_len, sa);
  ssize_t r =
      n ? ::sendto(fd_, buf, len, flags, reinterpret_cast<sockaddr*>(&sa), n)
        : ::send(fd_, buf, len, flags);
  return r >= 0 ? r : FromErrno();
}

i64 SocketDevice::Recvfrom(void* buf,
                           size_t len,
                           int flags,
                           void* guest_addr,
                           u32* addr_len) {
  sockaddr_storage sa;
  socklen_t n = sizeof(sa);
  ssize_t r =
      ::recvfrom(fd_, buf, len, flags, reinterpret_cast<sockaddr*>(&sa), &n);
  if (r < 0)
    return FromErrno();
  if (guest_addr && addr_len)
    *addr_len = ToGuest(sa, guest_addr, *addr_len);
  return r;
}

// Sony's own socket controls, in the 'P' ioctl group rather than BSD's. Both
// are 36-byte IOC_IN blocks with no reply, issued once each by libSceNet while
// it configures the socket it opens for its own bring-up. Accepting them is the
// whole contract: there is nothing for us to hand back, and refusing one made
// libSceNet treat its init as failed.
i32 SocketDevice::Ioctl(u32 command, void* args) {
  switch (command) {
    case 0x802450c8:
    case 0x802450c9:
      return 0;
    default:
      return Device::Ioctl(command, args);
  }
}

SocketDevice* FdToSocket(u32 fd) {
  auto* p = Proc::GetActive();
  if (!p)
    return nullptr;
  auto* obj = p->GetObjTable().Get(fd);
  if (!obj || obj->type() != Object::OType::kDevice)
    return nullptr;
  return dynamic_cast<SocketDevice*>(static_cast<Device*>(obj));
}

}  // namespace krnl
