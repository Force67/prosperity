#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * A datagram socket backed by a real host socket. Titles that only ever talked
 * to the system-service processes over AF_UNIX are still refused in sys_socket;
 * this exists for the ones that need a working UDP socket on the local network.
 * Minecraft's NetherNet/WebRTC layer is the case in point: rtc::PhysicalSocket-
 * Server waits on its wakeup socket forever, and every rtc::Thread::BlockingCall
 * into the LAN manager blocks with it, so the game never renders a frame.
 */

#include "device.h"

namespace krnl {

class socketDevice : public device {
public:
  socketDevice(proc *p, int hostFd, int guestFamily);
  ~socketDevice();

  // The host fd, for the event queue's readability poll.
  int hostFd() const { return fd_; }

  int bind(const void *guestAddr, uint32_t len);
  int getsockname(void *guestAddr, uint32_t *len);
  int64_t sendto(const void *buf, size_t len, int flags, const void *guestAddr,
                 uint32_t addrLen);
  int64_t recvfrom(void *buf, size_t len, int flags, void *guestAddr,
                   uint32_t *addrLen);

  int64_t read(void *buf, size_t len) override { return recvfrom(buf, len, 0, nullptr, nullptr); }
  int64_t write(const void *buf, size_t len) override { return sendto(buf, len, 0, nullptr, 0); }

private:
  int fd_ = -1;
  int family_ = 0;  // the guest's AF_*, needed to rebuild replies
};

// The socket behind an fd, or null when it isn't one.
socketDevice *fdToSocket(uint32_t fd);

}  // namespace krnl
