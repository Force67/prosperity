/*
 * PS4Delta : PS4 emulation and research project
 *
 * SPIRV-Tools optimize + validate wrapper. See spv_post.h.
 */

#ifdef DELTA_HAVE_SPIRV_BACKEND

#include "gpu/gcn/spirv/spv_post.h"

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

std::vector<uint32_t> Optimize(const std::vector<uint32_t>& spv) {
  spvtools::Optimizer opt(SPV_ENV_VULKAN_1_1);
  opt.SetMessageConsumer([](spv_message_level_t lvl, const char*,
                            const spv_position_t&, const char* msg) {
    if (lvl <= SPV_MSG_WARNING)
      std::fprintf(stderr, "[spv-opt] %s\n", msg);
  });
  // Legalization first: promotes the Private register-file variables to SSA
  // (mem2reg) so the performance passes can actually fold the naive load/store
  // stream the translator emits.
  opt.RegisterLegalizationPasses();
  opt.RegisterPerformancePasses();
  std::vector<uint32_t> out;
  if (!opt.Run(spv.data(), spv.size(), &out) || out.empty())
    return spv;  // keep the valid-but-unoptimized binary on failure
  return out;
}

bool Validate(const std::vector<uint32_t>& spv, std::string* err) {
  spv_context ctx = spvContextCreate(SPV_ENV_VULKAN_1_1);
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
constexpr uint32_t kCacheGeneration = 1;

uint64_t HashWords(const std::vector<uint32_t>& w) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a
  for (uint32_t x : w) {
    h ^= x;
    h *= 1099511628211ull;
  }
  h ^= kCacheGeneration;
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

std::string EntryPath(uint64_t key) {
  char name[32];
  std::snprintf(name, sizeof(name), "/%016llx.spv",
                static_cast<unsigned long long>(key));
  return CacheDir() + name;
}

bool ReadEntry(uint64_t key, std::vector<uint32_t>* out) {
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
void WriteEntry(uint64_t key, const std::vector<uint32_t>& spv) {
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

bool Finalize(const std::vector<uint32_t>& spv,
              std::vector<uint32_t>* out,
              std::string* err) {
  const uint64_t key = HashWords(spv);
  if (ReadEntry(key, out))
    return true;
  if (!Validate(spv, err))
    return false;
  *out = Optimize(spv);
  WriteEntry(key, *out);
  return true;
}

}  // namespace gpu::gcn::spirv

#endif  // DELTA_HAVE_SPIRV_BACKEND
