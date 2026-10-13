/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

namespace gpu::render {

bool WritePngRgba8(const char* path, const u8* rgba, u32 width, u32 height);

bool WritePngRgba16(const char* path, const u16* rgba, u32 width, u32 height);

u64 DeflateBound(u64 len);
u64 ZlibCompress(const u8* data, u64 len, u8* out);

}  // namespace gpu::render
