/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Frame captures written to disk: the PPM writers behind the DELTA_GPU_DUMP /
// SNAP / RTDUMP knobs, and the directory they land in.

#include <cstdint>
#include <utl/options.h>


namespace gpu::vk {

// Directory frame dumps go to. Defaults to /tmp; Android has no /tmp, so the
// runner sets DELTA_GPU_DUMP_DIR to a writable path (e.g. the cwd under
// /data/local/tmp). Returned without a trailing slash.
const char* DumpDir();

void WritePpm(const char* path, const uint8_t* bgra, uint32_t w, uint32_t h);
// Rolling numbered dump, capped at a handful of frames per run.
void DumpPpm(const uint8_t* bgra, uint32_t w, uint32_t h);

// Read by the frame path too, so these live here instead of being declared
// once per translation unit.
extern base::Option<bool> g_dump;     // DELTA_GPU_DUMP
extern base::Option<bool> kDeclines;  // DELTA_GPU_DECLINES

}  // namespace gpu::vk
