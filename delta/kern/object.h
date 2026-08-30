#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include <atomic>
#include "base/arch.h"
#include <mutex>
#include <utl/object_ref.h>

#include <base/containers/vector.h>
#include <base/strings/string_ref.h>
#include <base/strings/xstring.h>

namespace krnl {
class proc;
class objectTable;

class kObject {
public:
  using handleList = base::Vector<u32>;

  enum class oType {
    file,
    device,
    equeue,
    eventflag,
    semaphore,
    shm,
  };

  explicit kObject(objectTable &, oType);
  // Must be virtual: derived devices add virtual methods, so without a virtual
  // dtor the kObject subobject sits past the vptr (offset 8) and `delete this`
  // in release() would free an interior pointer (invalid free) and skip the
  // derived destructors (leaking the device's file handle).
  virtual ~kObject() = default;

  void retain();
  void release();
  void retainHandle();
  void releaseHandle();

  oType type() const { return otype; }

  handleList &handles() { return handleCollection; }

  u32 handle() const { return handleCollection[0]; }

  // What diagnostics call this object. Devices are opened by name, so the
  // opener is the only place that knows it.
  void setName(base::StringRef n) { name = base::String(n.data(), n.size()); }
  const base::String &getName() const { return name; }

protected:
  oType otype;
  objectTable &objects;
  base::String name;

private:
  handleList handleCollection;
  std::atomic<i32> refCount;
};

template <typename T> utl::object_ref<T> retain_object(T *ptr) {
  if (ptr)
    ptr->retain();
  return utl::object_ref<T>(ptr);
}
}
