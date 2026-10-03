#include "memory_debug/memory_debug.h"

#if defined(DELTA_MEMORY_DEBUG)
#include <unistd.h>

#include "base/atomic.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"

namespace memory_debug {
namespace {
constexpr u32 kSlots = 8192;
constexpr u32 kProbes = 64;
constexpr uintptr_t kDeleted = 1;
struct Mapping {
  uintptr_t begin = 0;
  uintptr_t end = 0;
  Bucket bucket = Bucket::kRuntime;
};
Mapping g_mappings[kSlots];
base::Mutex g_mutex;
base::Atomic<u64> g_dropped{0};

u32 Hash(uintptr_t address) {
  u64 value = address >> 12;
  value = (value ^ (value >> 33)) * 0xff51afd7ed558ccdull;
  return (value ^ (value >> 33)) & (kSlots - 1);
}

u32 Find(uintptr_t address) {
  const u32 first = Hash(address);
  for (u32 i = 0; i < kProbes; ++i) {
    const u32 slot = (first + i) & (kSlots - 1);
    if (!g_mappings[slot].begin)
      break;
    if (g_mappings[slot].begin == address)
      return slot;
  }
  return kSlots;
}

void Insert(uintptr_t begin, uintptr_t end, Bucket bucket) {
  const u32 first = Hash(begin);
  for (u32 i = 0; i < kProbes; ++i) {
    auto& mapping = g_mappings[(first + i) & (kSlots - 1)];
    if (mapping.begin > kDeleted)
      continue;
    mapping = {begin, end, bucket};
    Change(bucket, end - begin, end - begin, 1);
    return;
  }
  g_dropped.fetch_add(1, base::memory_order_relaxed);
}

void Erase(u32 slot) {
  auto& mapping = g_mappings[slot];
  const i64 size = mapping.end - mapping.begin;
  Change(mapping.bucket, -size, -size, -1);
  mapping.begin = kDeleted;
}

void ForgetRange(uintptr_t begin, uintptr_t end) {
  const u32 exact = Find(begin);
  if (exact != kSlots && g_mappings[exact].end == end) {
    Erase(exact);
    return;
  }
  for (u32 i = 0; i < kSlots; ++i) {
    const Mapping mapping = g_mappings[i];
    if (mapping.begin <= kDeleted || mapping.end <= begin ||
        mapping.begin >= end)
      continue;
    Erase(i);
    if (mapping.begin < begin)
      Insert(mapping.begin, begin, mapping.bucket);
    if (mapping.end > end)
      Insert(end, mapping.end, mapping.bucket);
  }
}

bool Range(void* address, u64& size, uintptr_t& begin, uintptr_t& end) {
  begin = reinterpret_cast<uintptr_t>(address);
  const u64 page = static_cast<u64>(sysconf(_SC_PAGESIZE));
  if (begin <= kDeleted || !size || !page || size > UINTPTR_MAX - begin ||
      page - 1 > UINTPTR_MAX - begin - size)
    return false;
  size = (size + page - 1) & ~(page - 1);
  end = begin + size;
  return true;
}
}  // namespace

void TrackMapping(void* address, u64 size, Bucket bucket, bool replace) {
  uintptr_t begin, end;
  if (!Range(address, size, begin, end))
    return;
  base::LockGuard lock(g_mutex);
  if (replace || Find(begin) != kSlots)
    ForgetRange(begin, end);
  Insert(begin, end, bucket);
}

void ForgetMapping(void* address, u64 size) {
  uintptr_t begin, end;
  if (!Range(address, size, begin, end))
    return;
  base::LockGuard lock(g_mutex);
  ForgetRange(begin, end);
}

u64 DroppedMappings() {
  return g_dropped.load(base::memory_order_relaxed);
}
}  // namespace memory_debug
#endif
