/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

// Guest index decoding and the host upload policy. Vulkan consumes 16- and
// 32-bit indices directly; guest 8-bit indices are widened to 16-bit.

#include "base/arch.h"

namespace gpu::render {

u32 GuestIndexElementBytes(u32 index_type);
u32 UploadedIndexElementBytes(u32 index_type);
u32 MaxGuestIndex(const void* source, u32 count, u32 index_type);
void CopyGuestIndices(void* destination,
                      const void* source,
                      u32 count,
                      u32 index_type);

}  // namespace gpu::render
