/*
 * PS4Delta : PS4/PS5 emulation and research project
 */
#pragma once

#include "base/arch.h"

namespace gpu::render {

u32 GuestIndexElementBytes(u32 index_type);
u32 UploadedIndexElementBytes(u32 index_type);
u32 MaxGuestIndex(const void* source, u32 count, u32 index_type);
void CopyGuestIndices(void* destination,
                      const void* source,
                      u32 count,
                      u32 index_type);
void CopyGuestQuadIndices(void* destination,
                          const void* source,
                          u32 count,
                          u32 index_type);

}  // namespace gpu::render
