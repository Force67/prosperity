
/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "kern/object.h"
#include "base/arch.h"
#include "kern/object_table.h"

#include <cstdlib>

#include "logger/logger.h"
#include "options/options.h"

namespace {
DELTA_OPTION(bool, kObjTrace, "DELTA_OBJ_TRACE", false);
}  // namespace

namespace kern {
Object::Object(ObjectTable& objects, Type type)
    : otype_(type), objects_(objects) {
  u32 temp = 0;
  objects.Add(this, temp);

  // DELTA_OBJ_TRACE: titles that poll a device re-create its object thousands
  // of times a second (Minecraft: ~14k in 40s), which buried every other line
  // in the log. Off unless asked for.
  if (kObjTrace) {
    static const char* tn[] = {"file",      "device",    "equeue",
                               "eventflag", "semaphore", "shm"};
    LOG_INFO("assigned handle {} type={}", temp, tn[static_cast<int>(type)]);
  }
}

void Object::Release() {
  if (--ref_count_ == 0)
    delete this;
}

void Object::Retain() {
  ref_count_++;
}

void Object::RetainHandle() {
  objects_.Keep(handle_collection_[0]);
}

void Object::ReleaseHandle() {
  objects_.Release(handle_collection_[0]);
}
}  // namespace kern