#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * /dev/random, /dev/urandom and /dev/rng. Without them the open fails with
 * ENOENT, and a guest that seeds from one either runs with no entropy or takes
 * a failure path it was never meant to: Minecraft's V8 opens /dev/urandom while
 * creating the isolate for its gameplay view.
 *
 * /dev/rng answers by ioctl rather than read. libSceSsl's DT_INIT seeds itself
 * through it, and a failure there makes it destroy its own static context
 * mutex; every later scePthreadMutexLock on that slot then reports EINVAL, so
 * sceSslInit fails, libSceNpManager's module start fails behind it, and a title
 * that calls into the Np stack anyway crashes in a library that never
 * initialised.
 */

#include "base/arch.h"
#include "kern/ps4/dev/device.h"

namespace krnl {
class Proc;

class RandomDevice : public Device {
 public:
  RandomDevice(ObjectTable&);

  i64 Read(void* buf, size_t len) override;
  i64 Lseek(i64 off, int whence) override;
  int Fstat(void* stat) override;
  i32 Ioctl(u32 command, void* args) override;
};
}  // namespace krnl
