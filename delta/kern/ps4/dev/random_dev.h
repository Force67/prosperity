#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * /dev/random and /dev/urandom. Without them the open fails with ENOENT, and a
 * guest that seeds from one either runs with no entropy or takes a failure path
 * it was never meant to: Minecraft's V8 opens /dev/urandom while creating the
 * isolate for its gameplay view.
 */

#include "device.h"

namespace krnl {
class proc;

class randomDevice : public device {
public:
  randomDevice(proc *);

  int64_t read(void *buf, size_t len) override;
  int64_t lseek(int64_t off, int whence) override;
  int fstat(void *stat) override;
};
} // namespace krnl
