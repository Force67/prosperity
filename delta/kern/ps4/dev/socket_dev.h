#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * A datagram socket backed by a real host socket. Titles that only ever talked
 * to the system-service processes over AF_UNIX are still refused in sys_socket;
 * this exists for the ones that need a working UDP socket on the local network.
 * Minecraft's NetherNet/WebRTC layer is the case in point: rtc::PhysicalSocket-
 * Server waits on its wakeup socket forever, and every
 * rtc::Thread::BlockingCall into the LAN manager blocks with it, so the game
 * never renders a frame.
 */

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace kern {

class SocketDevice : public Device {
 public:
  SocketDevice(ObjectTable& objects, int host_fd, int guest_family);
  ~SocketDevice();

  // The host fd, for the event queue's readability poll.
  int HostFd() const { return fd_; }

  int Bind(const void* guest_addr, u32 len);
  int Getsockname(void* guest_addr, u32* len);
  i64 Sendto(const void* buf,
             size_t len,
             int flags,
             const void* guest_addr,
             u32 addr_len);
  i64 Recvfrom(void* buf,
               size_t len,
               int flags,
               void* guest_addr,
               u32* addr_len);

  i64 Read(void* buf, size_t len) override {
    return Recvfrom(buf, len, 0, nullptr, nullptr);
  }
  i64 Write(const void* buf, size_t len) override {
    return Sendto(buf, len, 0, nullptr, 0);
  }

  i32 Ioctl(u32 command, void* args) override;

 private:
  int fd_ = -1;
  int family_ = 0;  // the guest's AF_*, needed to rebuild replies
};

// The socket behind an fd, or null when it isn't one.
SocketDevice* FdToSocket(u32 fd);

}  // namespace kern
