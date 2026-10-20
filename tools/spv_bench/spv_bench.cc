// Times the recompiler's SPIR-V check + optimize over raw modules dumped
// with DELTA_GPU_SPV_RAW_DUMP, through the same single pass Finalize uses.
// For profiling the optimizer under perf without booting a title.
// Usage: spv_bench <dump_dir> [rounds] [out_dir]
// out_dir receives each optimized module under its input name.
#include <dirent.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "base/containers/vector.h"
#include "base/strings/xstring.h"
#include "gpu/gcn/spirv/spv_opt.h"

namespace {
double Now() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

bool ReadWords(const base::String& path, base::Vector<u32>* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    return false;
  std::fseek(f, 0, SEEK_END);
  const long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  out->resize(static_cast<size_t>(bytes) / 4);
  const bool ok = std::fread(out->data(), 4, out->size(), f) == out->size();
  std::fclose(f);
  return ok && bytes > 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: spv_bench <dump_dir> [rounds]\n");
    return 1;
  }
  const int rounds = argc > 2 ? std::atoi(argv[2]) : 1;
  const char* out_dir = argc > 3 ? argv[3] : nullptr;
  base::Vector<base::String> names;
  base::Vector<base::Vector<u32>> modules;
  DIR* dir = opendir(argv[1]);
  if (!dir) {
    std::printf("cannot open %s\n", argv[1]);
    return 1;
  }
  while (dirent* e = readdir(dir)) {
    const size_t n = std::strlen(e->d_name);
    if (n < 4 || std::strcmp(e->d_name + n - 4, ".spv") != 0)
      continue;
    base::Vector<u32> words;
    if (ReadWords(base::String(argv[1]) + "/" + e->d_name, &words)) {
      modules.push_back(words);
      names.push_back(e->d_name);
    }
  }
  closedir(dir);

  double finalize = 0;
  size_t in_words = 0, out_words = 0;
  u64 hash = 1469598103934665603ull;  // FNV-1a of every output, in order
  for (int r = 0; r < rounds; r++) {
    for (size_t mi = 0; mi < modules.size(); mi++) {
      const auto& m = modules[mi];
      const double t0 = Now();
      base::String err;
      base::Vector<u32> out;
      if (!gpu::gcn::spirv::OptimizeModule(m, &out, 2, &err)) {
        std::printf("invalid module %s: %s\n", names[mi].c_str(), err.c_str());
        continue;
      }
      finalize += Now() - t0;
      if (out_dir && r == 0) {
        const base::String path = base::String(out_dir) + "/" + names[mi];
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
          std::fwrite(out.data(), 4, out.size(), f);
          std::fclose(f);
        }
      }
      in_words += m.size();
      out_words += out.size();
      for (u32 w : out)
        hash = (hash ^ w) * 1099511628211ull;
    }
  }
  std::printf("%zu modules x%d: finalize %.3f s, %zu -> %zu "
              "words, output %016llx\n",
              static_cast<size_t>(modules.size()), rounds, finalize,
              static_cast<size_t>(in_words), static_cast<size_t>(out_words),
              static_cast<unsigned long long>(hash));
  return 0;
}
