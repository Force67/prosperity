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
#include "base/arch.h"

namespace krnl {

class socketDevice : public device {
public:
  socketDevice(proc *p, int hostFd, int guestFamily);
  ~socketDevice();

  // The host fd, for the event queue's readability poll.
  int hostFd() const { return fd_; }

  int bind(const void *guestAddr, u32 len);
  int getsockname(void *guestAddr, u32 *len);
  i64 sendto(const void *buf, size_t len, int flags, const void *guestAddr,
                 u32 addrLen);
  i64 recvfrom(void *buf, size_t len, int flags, void *guestAddr,
                   u32 *addrLen);

  i64 read(void *buf, size_t len) override { return recvfrom(buf, len, 0, nullptr, nullptr); }
  i64 write(const void *buf, size_t len) override { return sendto(buf, len, 0, nullptr, 0); }

  i32 ioctl(u32 command, void *args) override;

private:
  int fd_ = -1;
  int family_ = 0;  // the guest's AF_*, needed to rebuild replies
};

// The socket behind an fd, or null when it isn't one.
socketDevice *fdToSocket(u32 fd);

}  // namespace krnl
