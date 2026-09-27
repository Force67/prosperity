/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/capture.h"
#include "base/arch.h"

#include <cstdio>
#include <cstdlib>
#include <base/logging.h>
#include <options/options.h>

namespace {
DELTA_OPTION(const char*, kDumpDir, "DELTA_GPU_DUMP_DIR", nullptr);
}  // namespace

namespace gpu::render {

DELTA_OPTION(bool, g_dump, "DELTA_GPU_DUMP", false);
DELTA_OPTION(bool, kDeclines, "DELTA_GPU_DECLINES", false);

namespace {
int g_dumped_frames = 0;
}  // namespace

// Directory frame dumps go to. Defaults to /tmp; Android has no /tmp, so the
// runner sets DELTA_GPU_DUMP_DIR to a writable path (e.g. the cwd under
// /data/local/tmp). Returned without a trailing slash.
const char* DumpDir() {
  const char* d = kDumpDir;
  return (d && *d) ? d : "/tmp";
}

void WritePpm(const char* path, const u8* bgra, u32 w, u32 h) {
  FILE* f = std::fopen(path, "wb");
  if (!f)
    return;
  std::fprintf(f, "P6\n%u %u\n255\n", w, h);
  for (u32 i = 0; i < w * h; i++) {
    std::fputc(bgra[i * 4 + 2], f);
    std::fputc(bgra[i * 4 + 1], f);
    std::fputc(bgra[i * 4 + 0], f);
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
