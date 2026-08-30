
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "object.h"
#include "base/arch.h"
#include "kern/object_table.h"

#include <cstdlib>

#include <logger/logger.h>
#include <utl/options.h>

namespace {
DELTA_OPTION(bool, kObjTrace, "DELTA_OBJ_TRACE", false);
}  // namespace

namespace krnl {
kObject::kObject(objectTable &objects, oType type)
    : otype(type), objects(objects) {
  u32 temp = 0;
  objects.add(this, temp);

  // DELTA_OBJ_TRACE: titles that poll a device re-create its object thousands
  // of times a second (Minecraft: ~14k in 40s), which buried every other line
  // in the log. Off unless asked for.
  if (kObjTrace) {
    static const char *tn[] = {"file", "device", "equeue", "eventflag",
                               "semaphore", "shm"};
    LOG_INFO("assigned handle {} type={}", temp, tn[static_cast<int>(type)]);
  }
}

void kObject::release() {
  if (--refCount == 0)
    delete this;
}

void kObject::retain() { refCount++; }

void kObject::retainHandle() {
  objects.keep(handleCollection[0]);
}

void kObject::releaseHandle() {
  objects.release(handleCollection[0]);
}
} // namespace krnl