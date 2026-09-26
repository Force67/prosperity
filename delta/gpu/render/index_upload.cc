/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/index_upload.h"
#include "base/arch.h"

#include <algorithm>
#include <cstring>

namespace gpu::render {

u32 GuestIndexElementBytes(u32 index_type) {
  return index_type == 1 ? 4 : index_type == 2 ? 1 : 2;
}

u32 UploadedIndexElementBytes(u32 index_type) {
  return index_type == 1 ? 4 : 2;
}

u32 MaxGuestIndex(const void* source,
                       u32 count,
                       u32 index_type) {
  u32 maximum = 0;
  if (index_type == 1) {
    const auto* indices = static_cast<const u32*>(source);
    for (u32 i = 0; i < count; i++)
      maximum = std::max(maximum, indices[i]);
  } else if (index_type == 2) {
    const auto* indices = static_cast<const u8*>(source);
    for (u32 i = 0; i < count; i++)
      maximum = std::max(maximum, static_cast<u32>(indices[i]));
  } else {
    const auto* indices = static_cast<const u16*>(source);
    for (u32 i = 0; i < count; i++)
      maximum = std::max(maximum, static_cast<u32>(indices[i]));
  }
  return maximum;
}

void CopyGuestIndices(void* destination,
                      const void* source,
                      u32 count,
                      u32 index_type) {
  if (index_type == 1) {
    std::memcpy(destination, source, static_cast<size_t>(count) * 4);
  } else if (index_type == 2) {
    auto* output = static_cast<u16*>(destination);
    const auto* input = static_cast<const u8*>(source);
    for (u32 i = 0; i < count; i++)
      output[i] = input[i];
  } else {
    std::memcpy(destination, source, static_cast<size_t>(count) * 2);
  }
}

}  // namespace gpu::render
