#pragma once

// taken from:
// https://github.com/xenia-project/xenia/blob/master/src/xenia/kernel/xobject.h

#include <base/algorithm.h>
#include <base/memory/cxx_lifetime.h>
#include <base/memory/move.h>
#include <base/meta/traits.h>

namespace utl {
template <typename T> class object_ref {
public:
  object_ref() noexcept : value_(nullptr) {}
  object_ref(base::nullptr_t) noexcept // NOLINT(runtime/explicit)
      : value_(nullptr) {}
  object_ref &operator=(base::nullptr_t) noexcept {
    reset();
    return (*this);
  }

  explicit object_ref(T *value) noexcept : value_(value) {
    // Assumes retained on call.
  }
  explicit object_ref(const object_ref &right) noexcept {
    reset(right.get());
    if (value_)
      value_->retain();
  }
  template <class V>
    requires(base::ConvertibleTo<V *, T *>)
  object_ref(const object_ref<V> &right) noexcept {
    reset(right.get());
    if (value_)
      value_->retain();
  }

  object_ref(object_ref &&right) noexcept : value_(right.release()) {}
  object_ref &operator=(object_ref &&right) noexcept {
    object_ref(base::move(right)).swap(*this);
    return (*this);
  }
  template <typename V> object_ref &operator=(object_ref<V> &&right) noexcept {
    object_ref(base::move(right)).swap(*this);
    return (*this);
  }

  object_ref &operator=(const object_ref &right) noexcept {
    object_ref(right).swap(*this); // NOLINT(runtime/explicit): misrecognized?
    return (*this);
  }
  template <typename V>
  object_ref &operator=(const object_ref<V> &right) noexcept {
    object_ref(right).swap(*this); // NOLINT(runtime/explicit): misrecognized?
    return (*this);
  }

  void swap(object_ref &right) noexcept { base::Swap(value_, right.value_); }

  ~object_ref() noexcept {
    if (value_) {
      value_->release();
      value_ = nullptr;
    }
  }

  T &operator*() const { return *get(); }

  T *operator->() const noexcept { return get(); }

  T *get() const noexcept { return value_; }

  template <typename V> V *get() const noexcept {
    return reinterpret_cast<V *>(value_);
  }

  explicit operator bool() const noexcept { return value_ != nullptr; }

  T *release() noexcept {
    T *value = value_;
    value_ = nullptr;
    return value;
  }

  void reset() noexcept { object_ref().swap(*this); }

  void reset(T *value) noexcept { object_ref(value).swap(*this); }

  inline bool operator==(const T *right) noexcept { return value_ == right; }

private:
  T *value_ = nullptr;
};

template <class _Ty, class... _Types>
  requires(!base::IsArray<_Ty>)
object_ref<_Ty> make_ref(_Types &&... _Args) {
  return object_ref<_Ty>(new _Ty(base::forward<_Types>(_Args)...));
}
}