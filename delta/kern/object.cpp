
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

#include <logger/logger.h>

namespace krnl {
kObject::kObject(proc *process, oType type) : otype(type), process(process) {
  uint32_t temp = 0;
  process->getObjTable().add(this, temp);

  LOG_INFO("assigned handle {}", temp);
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