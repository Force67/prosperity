#include "memory_debug/memory_debug.h"

#include "base/atomic.h"

namespace memory_debug {
namespace {
constexpr Description kDescriptions[] = {
    {"Textures", Domain::kGpu, true}, {"Render targets", Domain::kGpu, true},
    {"Buffer mirror", Domain::kGpu},  {"Upload staging", Domain::kGpu},
    {"Readback", Domain::kGpu},       {"Image backing", Domain::kGpu},
    {"Commands", Domain::kHost},      {"JIT / CPU", Domain::kHost},
    {"Shader cache", Domain::kHost},  {"GPU metadata", Domain::kHost},
    {"Thread stacks", Domain::kHost}, {"Media", Domain::kHost},
    {"Runtime / HLE", Domain::kHost}, {"Debugger", Domain::kHost},
};
static_assert(sizeof(kDescriptions) / sizeof(kDescriptions[0]) == kBucketCount);

#if defined(DELTA_MEMORY_DEBUG)
struct Counters {
  base::Atomic<i64> live{0};
  base::Atomic<i64> backing{0};
  base::Atomic<i64> allocations{0};
};
struct alignas(64) Shard {
  Counters buckets[kBucketCount];
  base::Atomic<i64> imported{0};
};
constexpr u32 kShards = 32;
Shard g_shards[kShards];
base::Atomic<u32> g_next_shard{0};
thread_local u32 t_shard = UINT32_MAX;
thread_local Bucket t_bucket = Bucket::kRuntime;

Shard& ThreadShard() {
  if (t_shard == UINT32_MAX)
    t_shard = g_next_shard.fetch_add(1, base::memory_order_relaxed) % kShards;
  return g_shards[t_shard];
}

u64 Nonnegative(i64 value) {
  return value > 0 ? static_cast<u64>(value) : 0;
}
#endif
}  // namespace

const Description& Describe(Bucket bucket) {
  return kDescriptions[static_cast<u32>(bucket)];
}

#if defined(DELTA_MEMORY_DEBUG)
// Keeps the replacement new/delete object linked out of the static library.
void HeapTrackingLinked();
u64 DroppedMappings();

Bucket CurrentBucket() {
  return t_bucket;
}

Bucket SetBucket(Bucket bucket) {
  const Bucket previous = t_bucket;
  t_bucket = bucket;
  return previous;
}

void Change(Bucket bucket, i64 live, i64 backing, i64 allocations) {
  auto& counters = ThreadShard().buckets[static_cast<u32>(bucket)];
  if (live)
    counters.live.fetch_add(live, base::memory_order_relaxed);
  if (backing)
    counters.backing.fetch_add(backing, base::memory_order_relaxed);
  if (allocations)
    counters.allocations.fetch_add(allocations, base::memory_order_relaxed);
}

void ChangeImported(i64 bytes) {
  ThreadShard().imported.fetch_add(bytes, base::memory_order_relaxed);
}

Snapshot ReadSnapshot() {
  HeapTrackingLinked();
  Snapshot result;
  for (u32 i = 0; i < kBucketCount; ++i) {
    i64 live = 0, backing = 0, allocations = 0;
    for (const auto& shard : g_shards) {
      const auto& counters = shard.buckets[i];
      live += counters.live.load(base::memory_order_relaxed);
      backing += counters.backing.load(base::memory_order_relaxed);
      allocations += counters.allocations.load(base::memory_order_relaxed);
    }
    result.buckets[i] = {Nonnegative(live), Nonnegative(backing),
                         Nonnegative(allocations)};
    if (i < kVisibleBuckets) {
      if (kDescriptions[i].domain == Domain::kHost) {
        result.host_live += result.buckets[i].live;
      } else {
        result.gpu_live += result.buckets[i].live;
        result.gpu_backing += result.buckets[i].backing;
      }
    }
  }
  i64 imported = 0;
  for (const auto& shard : g_shards)
    imported += shard.imported.load(base::memory_order_relaxed);
  result.imported = Nonnegative(imported);
  result.dropped_mappings = DroppedMappings();
  return result;
}
#endif
}  // namespace memory_debug
