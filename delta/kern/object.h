#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include "kern/object_ref.h"

#include "base/atomic.h"
#include "base/containers/vector.h"
#include "base/strings/string_ref.h"
#include "base/strings/xstring.h"

namespace kern {
class Process;
class ObjectTable;

class Object {
 public:
  using HandleList = base::Vector<u32>;

  enum class Type {
    kFile,
    kDevice,
    kEqueue,
    kEventflag,
    kSemaphore,
    kShm,
  };

  explicit Object(ObjectTable&, Type);
  // Must be virtual: derived devices add virtual methods, so without a virtual
  // dtor the Object subobject sits past the vptr (offset 8) and `delete this`
  // in release() would free an interior pointer (invalid free) and skip the
  // derived destructors (leaking the device's file handle).
  virtual ~Object() = default;

  void Retain();
  void Release();
  void RetainHandle();
  void ReleaseHandle();

  Type type() const { return otype_; }

  HandleList& handles() { return handle_collection_; }

  u32 handle() const { return handle_collection_[0]; }

  // What diagnostics call this object. Devices are opened by name, so the
  // opener is the only place that knows it.
  void SetName(base::StringRef n) { name_ = base::String(n.data(), n.size()); }
  const base::String& GetName() const { return name_; }

 protected:
  Type otype_;
  ObjectTable& objects_;
  base::String name_;

 private:
  HandleList handle_collection_;
  base::Atomic<i32> ref_count_;
};

template <typename T>
kern::ObjectRef<T> RetainObject(T* ptr) {
  if (ptr)
    ptr->retain();
  return kern::ObjectRef<T>(ptr);
}
}  // namespace kern
