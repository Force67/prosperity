/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_capture.h"

#include <cstdio>
#include <cstdlib>
#include <utl/options.h>

namespace {
DELTA_OPTION(const char*, kDumpDir, "DELTA_GPU_DUMP_DIR", nullptr);
}  // namespace

namespace gpu::vk {

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

void WritePpm(const char* path, const uint8_t* bgra, uint32_t w, uint32_t h) {
  FILE* f = std::fopen(path, "wb");
  if (!f)
    return;
  std::fprintf(f, "P6\n%u %u\n255\n", w, h);
  for (uint32_t i = 0; i < w * h; i++) {
    std::fputc(bgra[i * 4 + 2], f);
    std::fputc(bgra[i * 4 + 1], f);
    std::fputc(bgra[i * 4 + 0], f);
  }
  std::fclose(f);
}

void DumpPpm(const uint8_t* bgra, uint32_t w, uint32_t h) {
  if (g_dumped_frames >= 4)
    return;
  char path[256];
  std::snprintf(path, sizeof(path), "%s/gpu_frame_%d.ppm", DumpDir(),
                g_dumped_frames++);
  WritePpm(path, bgra, w, h);
  std::fprintf(stderr, "[gpuvk] dumped %s\n", path);
}

}  // namespace gpu::vk
