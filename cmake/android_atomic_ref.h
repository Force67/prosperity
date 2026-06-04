// std::atomic_ref polyfill for libc++ versions that don't ship it yet
// (Android NDK r27/r28 still have __cpp_lib_atomic_ref disabled). FEX requires
// std::atomic_ref; this implements it with the compiler's __atomic builtins,
// which is exactly what a real atomic_ref lowers to, so codegen is identical.
//
// Force-included (-include) into the Android builds only. Guarded on the feature
// macro so it disappears once the toolchain provides the real thing. We include
// <atomic> first so the guard sees the library's own definition if present.
#pragma once
#include <atomic>

#if !defined(__cpp_lib_atomic_ref)

namespace std {

template <class T>
class atomic_ref {
  T* ptr_;
  static constexpr int ord(memory_order o) noexcept { return static_cast<int>(o); }
  // For the single-order compare_exchange overloads the failure order may not
  // be release/acq_rel; weaken it the way the standard specifies.
  static constexpr memory_order fail_ord(memory_order o) noexcept {
    return o == memory_order_acq_rel ? memory_order_acquire
         : o == memory_order_release ? memory_order_relaxed
                                      : o;
  }

public:
  using value_type = T;
  explicit atomic_ref(T& obj) noexcept : ptr_(&obj) {}
  atomic_ref(const atomic_ref&) noexcept = default;
  atomic_ref& operator=(const atomic_ref&) = delete;

  T load(memory_order o = memory_order_seq_cst) const noexcept {
    T tmp;
    __atomic_load(ptr_, &tmp, ord(o));
    return tmp;
  }
  void store(T desired, memory_order o = memory_order_seq_cst) const noexcept {
    __atomic_store(ptr_, &desired, ord(o));
  }
  T exchange(T desired, memory_order o = memory_order_seq_cst) const noexcept {
    T tmp;
    __atomic_exchange(ptr_, &desired, &tmp, ord(o));
    return tmp;
  }
  bool compare_exchange_strong(T& expected, T desired, memory_order success,
                               memory_order failure) const noexcept {
    return __atomic_compare_exchange(ptr_, &expected, &desired, false,
                                     ord(success), ord(failure));
  }
  bool compare_exchange_strong(T& expected, T desired,
                               memory_order o = memory_order_seq_cst) const noexcept {
    return compare_exchange_strong(expected, desired, o, fail_ord(o));
  }
  bool compare_exchange_weak(T& expected, T desired, memory_order success,
                             memory_order failure) const noexcept {
    return __atomic_compare_exchange(ptr_, &expected, &desired, true,
                                     ord(success), ord(failure));
  }
  bool compare_exchange_weak(T& expected, T desired,
                             memory_order o = memory_order_seq_cst) const noexcept {
    return compare_exchange_weak(expected, desired, o, fail_ord(o));
  }
  T fetch_add(T arg, memory_order o = memory_order_seq_cst) const noexcept {
    return __atomic_fetch_add(ptr_, arg, ord(o));
  }
  T fetch_sub(T arg, memory_order o = memory_order_seq_cst) const noexcept {
    return __atomic_fetch_sub(ptr_, arg, ord(o));
  }
  operator T() const noexcept { return load(); }
};

}  // namespace std

#define __cpp_lib_atomic_ref 201806L
#endif  // !__cpp_lib_atomic_ref
