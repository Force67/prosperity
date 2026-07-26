/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Frame captures written to disk: the PPM writers behind the DELTA_GPU_DUMP /
// SNAP / RTDUMP knobs, and the directory they land in.

#include <cstdint>

namespace gpu::vk {

// Directory frame dumps go to. Defaults to /tmp; Android has no /tmp, so the
// runner sets DELTA_GPU_DUMP_DIR to a writable path (e.g. the cwd under
// /data/local/tmp). Returned without a trailing slash.
const char *dumpDir();

void writePpm(const char *path, const uint8_t *bgra, uint32_t w, uint32_t h);
// Rolling numbered dump, capped at a handful of frames per run.
void dumpPpm(const uint8_t *bgra, uint32_t w, uint32_t h);

extern const bool g_dump;      // DELTA_GPU_DUMP
extern const bool g_declines;  // DELTA_GPU_DECLINES

}  // namespace gpu::vk
