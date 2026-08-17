#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * /dev/random, /dev/urandom and /dev/rng. Without them the open fails with
 * ENOENT, and a guest that seeds from one either runs with no entropy or takes a
 * failure path it was never meant to: Minecraft's V8 opens /dev/urandom while
 * creating the isolate for its gameplay view.
 *
 * /dev/rng answers by ioctl rather than read. libSceSsl's DT_INIT seeds itself
 * through it, and a failure there makes it destroy its own static context mutex;
 * every later scePthreadMutexLock on that slot then reports EINVAL, so sceSslInit
 * fails, libSceNpManager's module start fails behind it, and a title that calls
 * into the Np stack anyway crashes in a library that never initialised.
 */

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

class randomDevice : public device {
public:
  randomDevice(proc *);

  i64 read(void *buf, size_t len) override;
  i64 lseek(i64 off, int whence) override;
  int fstat(void *stat) override;
  i32 ioctl(u32 command, void *args) override;
};
} // namespace krnl
