/*
 * PS4Delta : PS4 emulation and research project
 */

#ifdef DELTA_HAVE_SPIRV_BACKEND

#include "gpu/gcn/spirv/spv_post.h"
#include "gpu/gcn/spirv/spv_opt.h"
#include "base/arch.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gpu_perf.h"
#include "memory_debug/memory_debug.h"
#include "ui/shader_activity.h"

#include "base/logging.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <unistd.h>

#include "base/containers/deque.h"
#include "base/containers/hash_map.h"
#include "base/containers/map.h"
#include "base/containers/vector.h"
#include "base/functional/function.h"
#include "base/math/value_bounds.h"
#include "base/memory/move.h"
#include "base/memory/shared_pointer.h"
#include "base/strings/xstring.h"
#include "base/threading/condition_variable.h"
#include "base/threading/lock_guard.h"
#include "base/threading/mutex.h"
#include "base/threading/thread.h"
#include "options/options.h"
#include "profile/profile.h"

namespace gpu::gcn::spirv {

using gpu::NowNs;

namespace {
DELTA_OPTION(u32, kOptLevel, "DELTA_GPU_SPIRV_OPT", 2);
DELTA_OPTION(u32, kCompilerThreads, "DELTA_GPU_COMPILER_THREADS", 0);
}  // namespace

base::Vector<u32> Optimize(const base::Vector<u32>& spv) {
  base::Vector<u32> out;
  if (kOptLevel == 0 || !OptimizeModule(spv, &out, kOptLevel))
    return spv;
  return out;
}

bool Validate(const base::Vector<u32>& spv, base::String* err) {
  return CheckModule(spv, err);
}

namespace {
// Validate + Optimize over one parse.
bool CheckAndOptimize(const base::Vector<u32>& spv,
                      base::Vector<u32>* out,
                      base::String* err) {
  if (kOptLevel != 0) {
    base::String why;
    const bool ok = OptimizeModule(spv, out, kOptLevel, &why);
    if (!ok && err)
      *err = why;
    return ok;
  }
  *out = spv;
  return Validate(spv, err);
}
}  // namespace

namespace {
DELTA_OPTION(bool, kShaderCache, "DELTA_GPU_SHADER_CACHE", true);
DELTA_OPTION(const char*,
             kShaderCacheDir,
             "DELTA_GPU_SHADER_CACHE_DIR",
             nullptr);
DELTA_OPTION(const char*, kRawDump, "DELTA_GPU_SPV_RAW_DUMP", nullptr);

constexpr u32 kCacheGeneration = 6;

u64 HashWords(const base::Vector<u32>& w) {
  u64 h = 1469598103934665603ull;
  for (u32 x : w) {
    h ^= x;
    h *= 1099511628211ull;
  }
  h ^= kCacheGeneration;
  h *= 1099511628211ull;
  h ^= kOptLevel.get();
  h *= 1099511628211ull;
  return h;
}

const base::String& CacheDir() {
  static const base::String kDir = [] {
    if (!kShaderCache)
      return base::String();
    base::String d;
    if (kShaderCacheDir && *kShaderCacheDir) {
      d = kShaderCacheDir;
    } else if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
      d = base::String(xdg) + "/ps4delta/spirv";
    } else if (const char* home = std::getenv("HOME"); home && *home) {
      d = base::String(home) + "/.cache/ps4delta/spirv";
    } else {
      return base::String();
    }
    for (size_t i = 1; i <= d.size(); i++)
      if (i == d.size() || d[i] == '/')
        ::mkdir(d.substr(0, i).c_str(), 0755);
    return d;
  }();
  return kDir;
}

base::String EntryPath(u64 key) {
  char name[32];
  std::snprintf(name, sizeof(name), "/%016llx.spv",
                static_cast<unsigned long long>(key));
  return CacheDir() + name;
}

bool ReadEntry(u64 key, base::Vector<u32>* out) {
  if (CacheDir().empty())
    return false;
  FILE* f = std::fopen(EntryPath(key).c_str(), "rb");
  if (!f)
    return false;
  std::fseek(f, 0, SEEK_END);
  const long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  bool ok = bytes > 0 && (bytes % 4) == 0;
  if (ok) {
    out->resize(static_cast<size_t>(bytes) / 4);
    ok = std::fread(out->data(), 1, static_cast<size_t>(bytes), f) ==
         static_cast<size_t>(bytes);
  }
  std::fclose(f);
  if (!ok)
    out->clear();
  return ok;
}

void WriteEntry(u64 key, const base::Vector<u32>& spv) {
  if (CacheDir().empty() || spv.empty())
    return;
  const base::String path = EntryPath(key);
  char tmp[512];
  std::snprintf(tmp, sizeof(tmp), "%s.%d.tmp", path.c_str(), (int)::getpid());
  FILE* f = std::fopen(tmp, "wb");
  if (!f)
    return;
  const size_t bytes = spv.size() * 4;
  const bool ok = std::fwrite(spv.data(), 1, bytes, f) == bytes;
  std::fclose(f);
  if (ok)
    ::rename(tmp, path.c_str());
  else
    ::unlink(tmp);
}
}  // namespace

namespace {
struct Finalized {
  bool valid = false;
  base::Vector<u32> spv;
  base::String err;
};

Finalized RunFinalize(const base::Vector<u32>& spv, u64 key) {
  const ui::ShaderCompilation compilation;
  Finalized f;
  f.valid = CheckAndOptimize(spv, &f.spv, &f.err);
  if (f.valid)
    WriteEntry(key, f.spv);
  return f;
}

// A Finalize result several threads may wait on.
class Job {
 public:
  void Set(Finalized f) {
    {
      base::LockGuard<base::Mutex> lock(mutex_);
      result_ = base::move(f);
      done_ = true;
    }
    finished_.NotifyAll();
  }
  const Finalized& Get() {
    base::UniqueLock<base::Mutex> lock(mutex_);
    finished_.Wait(lock, [this] { return done_; });
    return result_;
  }

 private:
  base::Mutex mutex_;
  base::ConditionVariable finished_;
  bool done_ = false;
  Finalized result_;
};
using JobRef = base::SharedPointer<Job>;

class FinalizePool {
 public:
  FinalizePool(u32 threads, const char* name) {
    for (u32 i = 0; i < threads; i++)
      base::SpawnDetachedThread(name, [this] { Work(); });
  }

  void Start(const base::Vector<u32>& spv, u64 key) {
    base::LockGuard<base::Mutex> lock(mutex_);
    if (pending_.count(key))
      return;
    JobRef job = base::MakeShared<Job>();
    pending_.emplace(key, job);
    queue_.push_back([this, job, spv, key] {
      job->Set(RunFinalize(spv, key));
      if (!CacheDir().empty()) {
        base::LockGuard<base::Mutex> lock(mutex_);
        pending_.erase(key);
      }
    });
    ready_.NotifyOne();
  }

  void Then(const base::Vector<u32>& spv,
            u64 key,
            base::Function<void(const base::Vector<u32>&)> then) {
    base::LockGuard<base::Mutex> lock(mutex_);
    const auto it = pending_.find(key);
    JobRef job;
    if (it != pending_.end())
      job = it->second;
    queue_.push_back([spv, key, job, then = base::move(then)] {
      if (job) {
        const Finalized& f = job->Get();
        if (f.valid)
          then(f.spv);
        return;
      }
      base::Vector<u32> out;
      if (ReadEntry(key, &out)) {
        then(out);
        return;
      }
      const Finalized f = RunFinalize(spv, key);
      if (f.valid)
        then(f.spv);
    });
    ready_.NotifyOne();
  }

  void Post(base::Function<void()> job) {
    base::LockGuard<base::Mutex> lock(mutex_);
    queue_.push_back(base::move(job));
    ready_.NotifyOne();
  }

  bool Take(u64 key, JobRef* out) {
    base::LockGuard<base::Mutex> lock(mutex_);
    const auto it = pending_.find(key);
    if (it == pending_.end())
      return false;
    *out = it->second;
    pending_.erase(it);
    return true;
  }

  void Reset() {
    base::UniqueLock<base::Mutex> lock(mutex_);
    idle_.Wait(lock, [this] { return queue_.empty() && active_ == 0; });
    pending_ = {};
    queue_ = {};
  }

 private:
  void Work() {
    const memory_debug::Scope memory_scope(memory_debug::Bucket::kShaderCache);
    for (;;) {
      base::Function<void()> job;
      {
        base::UniqueLock<base::Mutex> lock(mutex_);
        ready_.Wait(lock, [this] { return !queue_.empty(); });
        job = base::move(queue_.front());
        queue_.pop_front();
        ++active_;
      }
      job();
      job = nullptr;
      {
        base::LockGuard<base::Mutex> lock(mutex_);
        --active_;
        idle_.NotifyAll();
      }
    }
  }

  base::Mutex mutex_;
  base::ConditionVariable ready_;
  base::ConditionVariable idle_;
  size_t active_ = 0;
  base::SimpleDeque<base::Function<void()>> queue_;
  base::HashMap<u64, JobRef> pending_;
};

FinalizePool* g_pool = nullptr;
FinalizePool& Pool() {
  static FinalizePool* pool = [] {
    auto* pool = new FinalizePool(
        base::Max(2u, base::GetProcessorCount() / 2), "spv_post");
    g_pool = pool;
    return pool;
  }();
  return *pool;
}

FinalizePool* g_pipeline_pool = nullptr;
FinalizePool& PipelinePool() {
  static FinalizePool* pool = [] {
    const u32 n = kCompilerThreads
                      ? kCompilerThreads.get()
                      : base::Max(2u, base::GetProcessorCount() / 4);
    auto* pool = new FinalizePool(n, "pipe_build");
    g_pipeline_pool = pool;
    return pool;
  }();
  return *pool;
}

thread_local bool t_prefetching = false;
}  // namespace

PrefetchScope::PrefetchScope() {
  t_prefetching = true;
}

PrefetchScope::~PrefetchScope() {
  t_prefetching = false;
}

void ResetPrefetch() {
  if (g_pool)
    g_pool->Reset();
  if (g_pipeline_pool)
    g_pipeline_pool->Reset();
}

void Prefetch(const base::Vector<u32>& spv) {
  const u64 key = HashWords(spv);
  if (!CacheDir().empty() && ::access(EntryPath(key).c_str(), F_OK) == 0)
    return;
  Pool().Start(spv, key);
}

void PrefetchThen(const base::Vector<u32>& spv,
                  base::Function<void(const base::Vector<u32>&)> then) {
  Pool().Then(spv, HashWords(spv), base::move(then));
}

bool Finalize(const base::Vector<u32>& spv,
              base::Vector<u32>* out,
              base::String* err) {
  DELTA_ZONE("spv.finalize");
  if (t_prefetching) {
    Prefetch(spv);
    *out = spv;
    return true;
  }
  const u64 key = HashWords(spv);
  if (const char* dir = kRawDump) {
    char path[512];
    std::snprintf(path, sizeof(path), "%s/%016llx.spv", dir,
                  (unsigned long long)key);
    if (FILE* f = std::fopen(path, "wb")) {
      std::fwrite(spv.data(), 4, spv.size(), f);
      std::fclose(f);
    }
  }
  JobRef job;
  if (Pool().Take(key, &job)) {
    g_stats.spv_miss_n++;
    const u64 t0 = NowNs();
    const Finalized& f = job->Get();
    g_stats.ns_spv_opt += NowNs() - t0;
    if (!f.valid && err)
      *err = f.err;
    *out = f.spv;
    return f.valid;
  }
  if (ReadEntry(key, out)) {
    g_stats.spv_hit_n++;
    return true;
  }
  g_stats.spv_miss_n++;
  const u64 t0 = NowNs();
  const ui::ShaderCompilation compilation;
  const bool valid = CheckAndOptimize(spv, out, err);
  g_stats.ns_spv_opt += NowNs() - t0;
  if (valid)
    WriteEntry(key, *out);
  return valid;
}

}  // namespace gpu::gcn::spirv

#endif

namespace gpu::gcn {
void ResetShaderWork() {
#ifdef DELTA_HAVE_SPIRV_BACKEND
  spirv::ResetPrefetch();
#endif
}

void PostShaderWork(base::Function<void()> job) {
#ifdef DELTA_HAVE_SPIRV_BACKEND
  spirv::PipelinePool().Post(base::move(job));
#else
  job();
#endif
}
}  // namespace gpu::gcn
