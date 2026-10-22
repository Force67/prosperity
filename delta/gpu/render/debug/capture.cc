/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/debug/capture.h"
#include "base/arch.h"

#include <cstdio>
#include <cstdlib>
#include "base/containers/vector.h"
#include "base/logging.h"
#include "options/options.h"

namespace {
DELTA_OPTION(const char*, kDumpDir, "DELTA_GPU_DUMP_DIR", nullptr);
}  // namespace

namespace gpu::render {

DELTA_OPTION(bool, g_dump, "DELTA_GPU_DUMP", false);
DELTA_OPTION(bool, kDeclines, "DELTA_GPU_DECLINES", false);

namespace {
int g_dumped_frames = 0;
}  // namespace

const char* DumpDir() {
  const char* d = kDumpDir;
  return (d && *d) ? d : "/tmp";
}

void WritePpm(const char* path, const u8* bgra, u32 w, u32 h) {
  FILE* f = std::fopen(path, "wb");
  if (!f)
    return;
  std::fprintf(f, "P6\n%u %u\n255\n", w, h);
  base::Vector<u8> row(w * 3);
  for (u32 y = 0; y < h; y++) {
    const u8* in = bgra + u64(y) * w * 4;
    for (u32 x = 0; x < w; x++) {
      row[x * 3 + 0] = in[x * 4 + 2];
      row[x * 3 + 1] = in[x * 4 + 1];
      row[x * 3 + 2] = in[x * 4 + 0];
    }
    std::fwrite(row.data(), 1, row.size(), f);
  }
  std::fclose(f);
}

void DumpPpm(const u8* bgra, u32 w, u32 h) {
  if (g_dumped_frames >= 4)
    return;
  char path[256];
  std::snprintf(path, sizeof(path), "%s/gpu_frame_%d.ppm", DumpDir(),
                g_dumped_frames++);
  WritePpm(path, bgra, w, h);
  BASE_LOGI("gpuvk", "dumped {}", path);
}

}  // namespace gpu::render
