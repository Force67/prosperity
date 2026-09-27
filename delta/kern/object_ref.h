#pragma once

// taken from:
// https://github.com/xenia-project/xenia/blob/master/src/xenia/kernel/xobject.h

#include "base/algorithm.h"
#include "base/memory/cxx_lifetime.h"
#include "base/memory/move.h"
#include "base/meta/traits.h"

namespace kern {
template <typename T>
class ObjectRef {
 public:
  ObjectRef() noexcept : value_(nullptr) {}
  ObjectRef(base::nullptr_t) noexcept  // NOLINT(runtime/explicit)
      : value_(nullptr) {}
  ObjectRef& operator=(base::nullptr_t) noexcept {
    Reset();
    return (*this);
  }

  explicit ObjectRef(T* value) noexcept : value_(value) {
    // Assumes retained on call.
  }
  explicit ObjectRef(const ObjectRef& right) noexcept {
    Reset(right.get());
    if (value_)
      value_->Retain();
  }
  template <class V>
    requires(base::ConvertibleTo<V*, T*>)
  ObjectRef(const ObjectRef<V>& right) noexcept {
    Reset(right.get());
    if (value_)
      value_->Retain();
  }

  ObjectRef(ObjectRef&& right) noexcept : value_(right.Release()) {}
  ObjectRef& operator=(ObjectRef&& right) noexcept {
    ObjectRef(base::move(right)).Swap(*this);
    return (*this);
  }
  template <typename V>
  ObjectRef& operator=(ObjectRef<V>&& right) noexcept {
    ObjectRef(base::move(right)).Swap(*this);
    return (*this);
  }

  ObjectRef& operator=(const ObjectRef& right) noexcept {
    ObjectRef(right).Swap(*this);  // NOLINT(runtime/explicit): misrecognized?
    return (*this);
  }
  template <typename V>
  ObjectRef& operator=(const ObjectRef<V>& right) noexcept {
    ObjectRef(right).Swap(*this);  // NOLINT(runtime/explicit): misrecognized?
    return (*this);
  }

  void Swap(ObjectRef& right) noexcept { base::Swap(value_, right.value_); }

  ~ObjectRef() noexcept {
    if (value_) {
      value_->Release();
      value_ = nullptr;
    }
  }

  T& operator*() const { return *get(); }

  T* operator->() const noexcept { return get(); }

  T* get() const noexcept { return value_; }

  template <typename V>
  V* Get() const noexcept {
    return reinterpret_cast<V*>(value_);
  }

  explicit operator bool() const noexcept { return value_ != nullptr; }

  T* Release() noexcept {
    T* value = value_;
    value_ = nullptr;
    return value;
  }

  void Reset() noexcept { ObjectRef().Swap(*this); }

  void Reset(T* value) noexcept { ObjectRef(value).Swap(*this); }

  inline bool operator==(const T* right) noexcept { return value_ == right; }

 private:
  T* value_ = nullptr;
};

template <class _Ty, class... _Types>
  requires(!base::IsArray<_Ty>)
ObjectRef<_Ty> MakeRef(_Types&&... args) {
  return ObjectRef<_Ty>(new _Ty(base::forward<_Types>(args)...));
}
}  // namespace kern