/*
 * PS4Delta : PS4 emulation and research project
 *
 * SPIRV-Tools optimize + validate wrapper. See spv_post.h.
 */

#ifdef DELTA_HAVE_SPIRV_BACKEND

#include "gpu/gcn/spirv/spv_post.h"
#include "base/arch.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/gpu_perf.h"

#include "base/logging.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include <spirv-tools/libspirv.h>
#include <spirv-tools/optimizer.hpp>

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
// DELTA_GPU_SPIRV_OPT: 2 = legalize + performance passes, 1 = legalize only,
// 0 = neither. Legalization (mem2reg over the Private-storage register file) is
// not a speed choice: without it the module is a load/store stream over 512
// variables, and whoever compiles it pays. Turning it off moved 22.6 s of
// Shadow of the Colossus's first minute out of spirv-opt and straight into the
// driver, at 626 ms a pipeline. The performance passes on top are the part that
// is optional.
DELTA_OPTION(u32, kOptLevel, "DELTA_GPU_SPIRV_OPT", 2);
}  // namespace

static spv_target_env TargetEnv(const base::Vector<u32>& spv) {
  const u32 version = spv.size() > 1 ? spv[1] : 0;
  return version >= 0x00010600u   ? SPV_ENV_VULKAN_1_3
         : version >= 0x00010400u ? SPV_ENV_VULKAN_1_2
                                  : SPV_ENV_VULKAN_1_1;
}

base::Vector<u32> Optimize(const base::Vector<u32>& spv) {
  if (kOptLevel == 0)
    return spv;
  const auto env = TargetEnv(spv);
  spvtools::Optimizer opt(env);
  opt.SetMessageConsumer([](spv_message_level_t lvl, const char*,
                            const spv_position_t&, const char* msg) {
    if (lvl <= SPV_MSG_WARNING)
      BASE_LOGI("spv-opt", "{}", msg);
  });
  // The register file is two Private arrays. Scalar replacement only splits
  // arrays of up to 100 elements by default, and the VGPR file has 256, so
  // every register access stayed a memory access in the finished shader.
  opt.RegisterPass(spvtools::CreatePrivateToLocalPass());
  opt.RegisterPass(spvtools::CreateScalarReplacementPass(0));
  opt.RegisterLegalizationPasses();
  if (kOptLevel >= 2)
    opt.RegisterPerformancePasses();
  // SPIRV-Tools hands the result back in a std::vector.
  std::vector<u32> out;
  if (!opt.Run(spv.data(), spv.size(), &out) || out.empty())
    return spv;  // keep the valid-but-unoptimized binary on failure
  return base::Vector<u32>(out.data(), out.data() + out.size());
}

bool Validate(const base::Vector<u32>& spv, base::String* err) {
  const auto env = TargetEnv(spv);
  spv_context ctx = spvContextCreate(env);
  spv_diagnostic diag = nullptr;
  spv_const_binary_t bin{spv.data(), spv.size()};
  spv_result_t r = spvValidate(ctx, &bin, &diag);
  bool ok = r == SPV_SUCCESS;
  if (!ok && err && diag)
    *err = diag->error;
  spvDiagnosticDestroy(diag);
  spvContextDestroy(ctx);
  return ok;
}

namespace {
DELTA_OPTION(bool, kShaderCache, "DELTA_GPU_SHADER_CACHE", true);
DELTA_OPTION(const char*,
             kShaderCacheDir,
             "DELTA_GPU_SHADER_CACHE_DIR",
             nullptr);
// DELTA_GPU_SPV_RAW_DUMP=<dir>: every module as it reaches the optimizer, for
// trying pass pipelines offline (spirv-opt) against what we really emit.
DELTA_OPTION(const char*, kRawDump, "DELTA_GPU_SPV_RAW_DUMP", nullptr);

// Bump when anything that changes the optimizer's OUTPUT changes: the pass
// list here, or the SPIRV-Tools version the build links. Entries from an older
// generation are never looked up.
constexpr u32 kCacheGeneration = 2;

u64 HashWords(const base::Vector<u32>& w) {
  u64 h = 1469598103934665603ull;  // FNV-1a
  for (u32 x : w) {
    h ^= x;
    h *= 1099511628211ull;
  }
  h ^= kCacheGeneration;
  h *= 1099511628211ull;
  // The pass set is part of what produced the entry, so it is part of the key:
  // otherwise a run at one DELTA_GPU_SPIRV_OPT level silently serves modules
  // built at another.
  h ^= kOptLevel.get();
  h *= 1099511628211ull;
  return h;
}

// The cache directory, created on first use. Empty means "no cache".
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
    // mkdir -p over the components we own.
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

// Write through a temporary + rename, so a torn file is never observed: two
// processes recompiling the same shader is normal.
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
  Finalized f;
  f.valid = Validate(spv, &f.err);
  if (f.valid) {
    f.spv = Optimize(spv);
    WriteEntry(key, f.spv);
  }
  return f;
}

// Finalize jobs started ahead of the draw that needs them. A level's first
// frame can bring hundreds of new shaders, and optimizing them one after the
// other on the submit thread stalled it for minutes while every other core
// sat idle.
class FinalizePool {
 public:
  FinalizePool() {
    const u32 n = base::Max(2u, std::thread::hardware_concurrency() / 2);
    for (u32 i = 0; i < n; i++)
      base::SpawnDetachedThread("spv_post", [this] { Work(); });
  }

  void Start(const base::Vector<u32>& spv, u64 key) {
    base::LockGuard<base::Mutex> lock(mutex_);
    if (pending_.count(key))
      return;
    auto task = base::MakeShared<std::packaged_task<Finalized()>>(
        [spv, key] { return RunFinalize(spv, key); });
    pending_.emplace(key, task->get_future().share());
    queue_.push_back([this, task, key] {
      (*task)();
      // Once the disk cache holds the result a later Finalize reads it there;
      // a module the prefetch guessed and no draw asked for is not kept.
      if (!CacheDir().empty()) {
        base::LockGuard<base::Mutex> lock(mutex_);
        pending_.erase(key);
      }
    });
    ready_.NotifyOne();
  }

  // Run `then` on a worker with the optimized module, once there is one.
  void Then(const base::Vector<u32>& spv,
            u64 key,
            base::Function<void(const base::Vector<u32>&)> then) {
    base::LockGuard<base::Mutex> lock(mutex_);
    const auto it = pending_.find(key);
    std::shared_future<Finalized> job;
    if (it != pending_.end())
      job = it->second;
    // FIFO: the job this waits on was queued earlier, so some worker already
    // holds it.
    queue_.push_back([spv, key, job, then = base::move(then)] {
      if (job.valid()) {
        if (job.get().valid)
          then(job.get().spv);
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

  // The job for `key`, handed over once: afterwards the disk cache has it.
  bool Take(u64 key, std::shared_future<Finalized>* out) {
    base::LockGuard<base::Mutex> lock(mutex_);
    const auto it = pending_.find(key);
    if (it == pending_.end())
      return false;
    *out = it->second;
    pending_.erase(it);
    return true;
  }

 private:
  void Work() {
    for (;;) {
      base::Function<void()> job;
      {
        base::UniqueLock<base::Mutex> lock(mutex_);
        ready_.Wait(lock, [this] { return !queue_.empty(); });
        job = base::move(queue_.front());
        queue_.pop_front();
      }
      job();
    }
  }

  base::Mutex mutex_;
  base::ConditionVariable ready_;
  base::SimpleDeque<base::Function<void()>> queue_;
  base::HashMap<u64, std::shared_future<Finalized>> pending_;
};

FinalizePool& Pool() {
  static FinalizePool* pool = new FinalizePool;
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
  std::shared_future<Finalized> job;
  if (Pool().Take(key, &job)) {
    g_spv_miss_n++;
    const u64 t0 = NowNs();
    const Finalized& f = job.get();
    g_ns_spv_opt += NowNs() - t0;
    if (!f.valid && err)
      *err = f.err;
    *out = f.spv;
    return f.valid;
  }
  if (ReadEntry(key, out)) {
    g_spv_hit_n++;
    return true;
  }
  g_spv_miss_n++;
  u64 t0 = NowNs();
  const bool valid = Validate(spv, err);
  g_ns_spv_val += NowNs() - t0;
  if (!valid)
    return false;
  t0 = NowNs();
  *out = Optimize(spv);
  g_ns_spv_opt += NowNs() - t0;
  WriteEntry(key, *out);
  return true;
}

}  // namespace gpu::gcn::spirv

#endif  // DELTA_HAVE_SPIRV_BACKEND
