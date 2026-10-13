/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"
#include "options/options.h"

namespace gpu::render {

const char* DumpDir();

void WritePpm(const char* path, const u8* bgra, u32 w, u32 h);
void DumpPpm(const u8* bgra, u32 w, u32 h);

extern base::Option<bool> g_dump;
extern base::Option<bool> kDeclines;

}  // namespace gpu::render
