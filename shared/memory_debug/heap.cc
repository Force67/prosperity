#include "memory_debug/memory_debug.h"

#if defined(DELTA_MEMORY_DEBUG)
#include <cstdlib>
#include <limits>
#include <new>
#if defined(__linux__)
#include <malloc.h>
// mimalloc's size-class lookup, when mimalloc is linked (ps4delta, not tests).
extern "C" size_t mi_good_size(size_t size) __attribute__((weak));
#endif

namespace {
// Heap traffic is batched per thread: three shared atomic adds on every
// allocation and free were 9% of shader optimization time. A thread's deltas
// reach the shared counters every kFlushOps operations, once a bucket's live
// delta passes kFlushBytes, when it reads a snapshot, and when it exits.
constexpr u32 kFlushOps = 128;
constexpr i64 kFlushBytes = 1 << 20;

struct Pending {
  i64 live[memory_debug::kBucketCount];
  i64 backing[memory_debug::kBucketCount];
  i64 allocations[memory_debug::kBucketCount];
  u32 ops;
  bool flusher_armed;
  bool dead;

  void Flush() {
    for (u32 i = 0; i < memory_debug::kBucketCount; i++) {
      if (live[i] || backing[i] || allocations[i])
        memory_debug::Change(static_cast<memory_debug::Bucket>(i), live[i],
                             backing[i], allocations[i]);
      live[i] = backing[i] = allocations[i] = 0;
    }
    ops = 0;
  }
};
// Trivially destructible, so the hot path reads it without a TLS init check;
// the flusher below is touched once per thread to register the exit flush.
thread_local Pending t_pending;

struct ExitFlusher {
  ~ExitFlusher() {
    t_pending.Flush();
    t_pending.dead = true;
  }
};
thread_local ExitFlusher t_exit_flusher;

[[gnu::noinline]] void ArmExitFlush() {
  t_pending.flusher_armed = true;
  static_cast<void>(&t_exit_flusher);
}

inline void Account(memory_debug::Bucket bucket,
                    i64 live,
                    i64 backing,
                    i64 allocations) {
  Pending& p = t_pending;
  if (p.dead) {
    memory_debug::Change(bucket, live, backing, allocations);
    return;
  }
  if (!p.flusher_armed)
    ArmExitFlush();
  const u32 i = static_cast<u32>(bucket);
  p.live[i] += live;
  p.backing[i] += backing;
  p.allocations[i] += allocations;
  if (++p.ops >= kFlushOps || p.live[i] >= kFlushBytes ||
      p.live[i] <= -kFlushBytes)
    p.Flush();
}

struct Header {
  void* base;
  size_t size;
  size_t backing;
  memory_debug::Bucket bucket;
};

void* Allocate(size_t size, size_t alignment) {
  alignment = alignment < alignof(Header) ? alignof(Header) : alignment;
  const size_t payload = size ? size : 1;
  if (payload > std::numeric_limits<size_t>::max() - sizeof(Header) ||
      alignment - 1 >
          std::numeric_limits<size_t>::max() - payload - sizeof(Header))
    throw std::bad_alloc();
  const size_t total = payload + sizeof(Header) + alignment - 1;
  void* block = nullptr;
  while (!(block = std::malloc(total))) {
    const auto handler = std::get_new_handler();
    if (!handler)
      throw std::bad_alloc();
    handler();
  }
  const uintptr_t address =
      (reinterpret_cast<uintptr_t>(block) + sizeof(Header) + alignment - 1) &
      ~(alignment - 1);
  const size_t prefix = address - reinterpret_cast<uintptr_t>(block);
#if defined(__linux__)
  // The size class mimalloc serves `total` from needs no pointer lookup.
  const size_t backing =
      (mi_good_size ? mi_good_size(total) : malloc_usable_size(block)) - prefix;
#else
  const size_t backing = total - prefix;
#endif
  auto* header = reinterpret_cast<Header*>(address) - 1;
  ::new (header) Header{block, size, backing, memory_debug::CurrentBucket()};
  Account(header->bucket, size, backing, 1);
  return reinterpret_cast<void*>(address);
}

void Free(void* pointer) noexcept {
  if (!pointer)
    return;
  const auto* header = reinterpret_cast<Header*>(pointer) - 1;
  Account(header->bucket, -static_cast<i64>(header->size),
          -static_cast<i64>(header->backing), -1);
  std::free(header->base);
}
}  // namespace

namespace memory_debug {
void FlushHeapChanges() {
  if (!t_pending.dead)
    t_pending.Flush();
}
}  // namespace memory_debug

void* operator new(size_t size) {
  return Allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}
void* operator new[](size_t size) {
  return Allocate(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}
void* operator new(size_t size, std::align_val_t alignment) {
  return Allocate(size, static_cast<size_t>(alignment));
}
void* operator new[](size_t size, std::align_val_t alignment) {
  return Allocate(size, static_cast<size_t>(alignment));
}
void operator delete(void* pointer) noexcept {
  Free(pointer);
}
void operator delete[](void* pointer) noexcept {
  Free(pointer);
}
void operator delete(void* pointer, size_t) noexcept {
  Free(pointer);
}
void operator delete[](void* pointer, size_t) noexcept {
  Free(pointer);
}
void operator delete(void* pointer, std::align_val_t) noexcept {
  Free(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept {
  Free(pointer);
}
void operator delete(void* pointer, size_t, std::align_val_t) noexcept {
  Free(pointer);
}
void operator delete[](void* pointer, size_t, std::align_val_t) noexcept {
  Free(pointer);
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  Free(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  Free(pointer);
}
void operator delete(void* pointer,
                     std::align_val_t,
                     const std::nothrow_t&) noexcept {
  Free(pointer);
}
void operator delete[](void* pointer,
                       std::align_val_t,
                       const std::nothrow_t&) noexcept {
  Free(pointer);
}
#endif
