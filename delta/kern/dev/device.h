#pragma once

#include <base.h>

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "kern/lv2/error_table.h"
#include "kern/object.h"

namespace krnl {
class proc;

class device : public kObject {
public:
  inline device(proc *p) : kObject(p, kObject::oType::device) {}

  virtual bool init(const char *, uint32_t, uint32_t) { return true; }
  virtual uint8_t *map(void *, size_t, uint32_t, uint32_t, size_t) {
    __debugbreak();
    return nullptr;
  }
  virtual int32_t ioctl(uint32_t command, void *args) {
    __debugbreak();
    return -1;
  }
};
}