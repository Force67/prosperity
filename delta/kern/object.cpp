
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "object.h"
#include "kern/proc.h"
#include "util/object_table.h"

#include <cstdlib>

#include <logger/logger.h>

namespace krnl {
kObject::kObject(proc *process, oType type) : otype(type), process(process) {
  uint32_t temp = 0;
  process->getObjTable().add(this, temp);

  // DELTA_OBJ_TRACE: titles that poll a device re-create its object thousands
  // of times a second (Minecraft: ~14k in 40s), which buried every other line
  // in the log. Off unless asked for.
  static const bool trace = std::getenv("DELTA_OBJ_TRACE") != nullptr;
  if (trace) {
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
  process->getObjTable().keep(handleCollection[0]);
}

void kObject::releaseHandle() {
  process->getObjTable().release(handleCollection[0]);
}
} // namespace krnl