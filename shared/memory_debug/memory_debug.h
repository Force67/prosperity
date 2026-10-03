#pragma once

#include <cstdint>

#include "base/arch.h"

namespace memory_debug {

enum class Bucket : u8 {
  kTextures,
  kRenderTargets,
  kBufferMirror,
  kUpload,
  kReadback,
  kImagePool,
  kCommands,
  kJit,
  kShaderCache,
  kMetadata,
  kThreadStacks,
  kMedia,
  kRuntime,
  kDebugger,
  kCount,
};

constexpr u32 kBucketCount = static_cast<u32>(Bucket::kCount);
constexpr u32 kVisibleBuckets = static_cast<u32>(Bucket::kDebugger);

enum class Domain { kGpu, kHost };

struct Description {
  const char* name;
  Domain domain;
  bool shared_backing = false;
};

struct Stats {
  u64 live = 0;
  u64 backing = 0;
  u64 allocations = 0;
};

struct Snapshot {
  Stats buckets[kBucketCount];
  u64 host_live = 0;
  u64 gpu_live = 0;
  u64 gpu_backing = 0;
  u64 imported = 0;
  u64 dropped_mappings = 0;
};

const Description& Describe(Bucket bucket);

#if defined(DELTA_MEMORY_DEBUG)
Bucket CurrentBucket();
Bucket SetBucket(Bucket bucket);
// Signed deltas permit frees on a different thread without a pointer registry.
void Change(Bucket bucket, i64 live, i64 backing, i64 allocations);
void ChangeImported(i64 bytes);
Snapshot ReadSnapshot();
void TrackMapping(void* address, u64 size, Bucket bucket, bool replace = false);
void ForgetMapping(void* address, u64 size);
#else
inline Bucket CurrentBucket() {
  return Bucket::kRuntime;
}
inline Bucket SetBucket(Bucket) {
  return Bucket::kRuntime;
}
inline void Change(Bucket, i64, i64, i64) {}
inline void ChangeImported(i64) {}
inline Snapshot ReadSnapshot() {
  return {};
}
inline void TrackMapping(void*, u64, Bucket, bool = false) {}
inline void ForgetMapping(void*, u64) {}
#endif

class Scope {
 public:
  explicit Scope(Bucket bucket) : previous_(SetBucket(bucket)) {}
  ~Scope() { SetBucket(previous_); }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  Bucket previous_;
};

// Lives with the owning resource, so failure paths and cross-thread frees
// retain their allocation tag without a pointer lookup.
class Record {
 public:
  Record() = default;
  ~Record() { Reset(); }
  Record(const Record&) = delete;
  Record& operator=(const Record&) = delete;
  Record(Record&& other) noexcept { Move(other); }
  Record& operator=(Record&& other) noexcept {
    if (this != &other) {
      Reset();
      Move(other);
    }
    return *this;
  }
  void Set(Bucket bucket, u64 live, u64 backing, u64 allocations = 1) {
#if defined(DELTA_MEMORY_DEBUG)
    Reset();
    bucket_ = bucket;
    stats_ = {live, backing, allocations};
    Change(bucket_, live, backing, allocations);
#endif
  }
  void AddLive(u64 bytes) {
#if defined(DELTA_MEMORY_DEBUG)
    stats_.live += bytes;
    Change(bucket_, bytes, 0, 0);
#endif
  }
  void AddBacking(u64 bytes) {
#if defined(DELTA_MEMORY_DEBUG)
    stats_.backing += bytes;
    Change(bucket_, 0, bytes, 0);
#endif
  }
  void Reset() {
#if defined(DELTA_MEMORY_DEBUG)
    Change(bucket_, -static_cast<i64>(stats_.live),
           -static_cast<i64>(stats_.backing),
           -static_cast<i64>(stats_.allocations));
    stats_ = {};
#endif
  }

 private:
  void Move(Record& other) {
#if defined(DELTA_MEMORY_DEBUG)
    bucket_ = other.bucket_;
    stats_ = other.stats_;
    other.stats_ = {};
#endif
  }
#if defined(DELTA_MEMORY_DEBUG)
  Bucket bucket_ = Bucket::kRuntime;
  Stats stats_;
#endif
};

}  // namespace memory_debug
