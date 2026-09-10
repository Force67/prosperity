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

#include <base/logging.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <spirv-tools/libspirv.h>
#include <spirv-tools/optimizer.hpp>

#include <sys/stat.h>
#include <unistd.h>

#include <utl/options.h>

namespace gpu::gcn::spirv {

using gpu::NowNs;

namespace {
// DELTA_GPU_SPIRV_OPT: 2 = legalize + performance passes, 1 = legalize only,
// 0 = neither. Legalization (mem2reg over the Private-storage register file) is
// not a speed choice: without it the module is a load/store stream over 512
// variables, and whoever compiles it pays -- turning it off moved 22.6 s of
// Shadow of the Colossus's first minute out of spirv-opt and straight into the
// driver, at 626 ms a pipeline. The performance passes on top are the part that
// is optional.
DELTA_OPTION(u32, kOptLevel, "DELTA_GPU_SPIRV_OPT", 2);
}  // namespace

std::vector<u32> Optimize(const std::vector<u32>& spv) {
  if (kOptLevel == 0)
    return spv;
  const auto env = spv.size() > 1 && spv[1] >= 0x00010400u
                       ? SPV_ENV_VULKAN_1_2 : SPV_ENV_VULKAN_1_1;
  spvtools::Optimizer opt(env);
  opt.SetMessageConsumer([](spv_message_level_t lvl, const char*,
                            const spv_position_t&, const char* msg) {
    if (lvl <= SPV_MSG_WARNING)
      BASE_LOGI("spv-opt", "{}", msg);
  });
  opt.RegisterLegalizationPasses();
  if (kOptLevel >= 2)
    opt.RegisterPerformancePasses();
  std::vector<u32> out;
  if (!opt.Run(spv.data(), spv.size(), &out) || out.empty())
    return spv;  // keep the valid-but-unoptimized binary on failure
  return out;
}

bool Validate(const std::vector<u32>& spv, std::string* err) {
  const auto env = spv.size() > 1 && spv[1] >= 0x00010400u
                       ? SPV_ENV_VULKAN_1_2 : SPV_ENV_VULKAN_1_1;
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

// Bump when anything that changes the optimizer's OUTPUT changes -- the pass
// list here, or the SPIRV-Tools version the build links. Entries from an older
// generation are simply never looked up.
constexpr u32 kCacheGeneration = 1;

u64 HashWords(const std::vector<u32>& w) {
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
const std::string& CacheDir() {
  static const std::string dir = [] {
    if (!kShaderCache)
      return std::string();
    std::string d;
    if (kShaderCacheDir && *kShaderCacheDir) {
      d = kShaderCacheDir;
    } else if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
      d = std::string(xdg) + "/ps4delta/spirv";
    } else if (const char* home = std::getenv("HOME"); home && *home) {
      d = std::string(home) + "/.cache/ps4delta/spirv";
    } else {
      return std::string();
    }
    // mkdir -p over the components we own.
    for (size_t i = 1; i <= d.size(); i++)
      if (i == d.size() || d[i] == '/')
        ::mkdir(d.substr(0, i).c_str(), 0755);
    return d;
  }();
  return dir;
}

std::string EntryPath(u64 key) {
  char name[32];
  std::snprintf(name, sizeof(name), "/%016llx.spv",
                static_cast<unsigned long long>(key));
  return CacheDir() + name;
}

bool ReadEntry(u64 key, std::vector<u32>* out) {
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
void WriteEntry(u64 key, const std::vector<u32>& spv) {
  if (CacheDir().empty() || spv.empty())
    return;
  const std::string path = EntryPath(key);
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

bool Finalize(const std::vector<u32>& spv,
              std::vector<u32>* out,
              std::string* err) {
  const u64 key = HashWords(spv);
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
