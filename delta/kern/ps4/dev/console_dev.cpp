// Copyright (C) Force67 2019

#include <base.h>
#include <cstdio>
#include <cstring>

#include <utl/options.h>

#include "console_dev.h"
#include "file_dev.h"

namespace {
// The same knob sys_write honours for guest fd 1/2. Console output is the same
// class of chatter and has to answer to it too, or turning it on silences only
// half of what the guest prints.
DELTA_OPTION(bool, kQuietGuest, "DELTA_QUIET_GUEST", false);
}  // namespace

namespace krnl {
consoleDevice::consoleDevice(proc *p) : device(p) {}

bool consoleDevice::init(const char *, uint32_t, uint32_t) { return true; }

// No input source in the emulator: report EOF so a reader loop terminates.
int64_t consoleDevice::read(void *, size_t) { return 0; }

// The point of a console is that its output is visible. Forward to the host
// console; the guest wrote to /dev/console specifically to be seen.
int64_t consoleDevice::write(const void *buf, size_t n) {
  if (!buf || !n)
    return 0;
  if (kQuietGuest)
    return static_cast<int64_t>(n);  // consumed, just not shown
  return static_cast<int64_t>(std::fwrite(buf, 1, n, stdout));
}

int64_t consoleDevice::lseek(int64_t, int) { return 0; }

int consoleDevice::fstat(void *stat) {
  if (!stat)
    return -static_cast<int>(SysError::eFAULT);
  fillStat(*reinterpret_cast<SceKernelStat *>(stat), 0x2000, 0);
  return 0;
}

// The tty ioctl set the kernel's console answers. The getters a caller reads
// from get a zeroed buffer (a default that needs no hardware); the setters
// that would reconfigure a real line are accepted and ignored. Anything else
// soft-fails like the base device, with its output buffer zeroed so the caller
// never reads stack garbage.
int32_t consoleDevice::ioctl(uint32_t cmd, void *data) {
  switch (cmd) {
    case 0x402c7413:  // get termios
      if (data)
        std::memset(data, 0, 0x2c);
      return 0;
    case 0x40087468:  // get window size
      if (data)
        std::memset(data, 0, 8);
      return 0;
    case 0x40047477:  // TIOCGPGRP: foreground process group
    case 0x40047473:  // TIOCOUTQ: bytes still queued for output
    case 0x4004746a:  // TIOCMGET: modem line status
    case 0x4004741a:  // TIOCGETD: line discipline
    case 0x4004667f:  // FIONREAD: bytes available to read
    case 0x4004667b:  // FIOGETOWN: owner receiving SIGIO
    case 0x40046677:  // FIODTYPE: device type bits
      if (data)
        std::memset(data, 0, 4);
      return 0;
    case 0x802c7414:  // TIOCSETA: set termios now
    case 0x802c7415:  // TIOCSETAW: drain, then set
    case 0x802c7416:  // TIOCSETAF: drain and flush input, then set
    case 0x80087467:  // TIOCSWINSZ: set window size
    case 0x80047476:  // TIOCSPGRP: set foreground process group
    case 0x80047462:  // TIOCSETD: set line discipline
    case 0x80047410:  // TIOCFLUSH: discard queued data
    case 0x80017472:  // TIOCSTI: push a byte back into the input queue
    case 0x2000740d:  // TIOCEXCL: claim exclusive use
    case 0x2000740e:  // TIOCNXCL: release it
    case 0x2000745e:  // TIOCDRAIN: wait for output to drain
    case 0x20007461:  // TIOCSCTTY: become the controlling terminal
    case 0x20007465:  // TIOCSTAT: report line status
    case 0x2000746e:  // TIOCSTART: resume output
    case 0x2000746f:  // TIOCSTOP: suspend output
    case 0x2000747a:  // TIOCCBRK: clear break
    case 0x2000747b:  // TIOCSBRK: set break
      return 0;
    default: {
      // Bounded like the other device logs: an unknown ioctl a title issues per
      // frame would otherwise print on every one, and an unbuffered printf on a
      // per-frame path is what once throttled the render loop to a few fps.
      static uint32_t logged = 0;
      if (logged < 32) {
        logged++;
        std::printf("[console] UNHANDLED ioctl(%#x) -> 0\n", cmd);
      }
      if (data && (cmd & 0x40000000u)) {
        const uint32_t len = (cmd >> 16) & 0x1fff;
        if (len)
          std::memset(data, 0, len);
      }
      return 0;
    }
  }
}
} // namespace krnl
