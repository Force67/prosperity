#include "memory_debug/memory_debug.h"

#if defined(DELTA_MEMORY_DEBUG)
#include <cstdlib>
#include <limits>
#include <new>
#if defined(__linux__)
#include <malloc.h>
#endif

namespace {
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
  const size_t backing = malloc_usable_size(block) - prefix;
#else
  const size_t backing = total - prefix;
#endif
  auto* header = reinterpret_cast<Header*>(address) - 1;
  ::new (header) Header{block, size, backing, memory_debug::CurrentBucket()};
  memory_debug::Change(header->bucket, size, backing, 1);
  return reinterpret_cast<void*>(address);
}

void Free(void* pointer) noexcept {
  if (!pointer)
    return;
  const auto* header = reinterpret_cast<Header*>(pointer) - 1;
  memory_debug::Change(header->bucket, -static_cast<i64>(header->size),
                       -static_cast<i64>(header->backing), -1);
  std::free(header->base);
}
}  // namespace

namespace memory_debug {
void HeapTrackingLinked() {}
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
