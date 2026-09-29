/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/render/index_upload.h"
#include "base/arch.h"

#include <cstring>
#include "base/math/value_bounds.h"

namespace gpu::render {

u32 GuestIndexElementBytes(u32 index_type) {
  return index_type == 1 ? 4 : index_type == 2 ? 1 : 2;
}

u32 UploadedIndexElementBytes(u32 index_type) {
  return index_type == 1 ? 4 : 2;
}

u32 MaxGuestIndex(const void* source, u32 count, u32 index_type) {
  u32 maximum = 0;
  if (index_type == 1) {
    const auto* indices = static_cast<const u32*>(source);
    for (u32 i = 0; i < count; i++)
      maximum = base::Max(maximum, indices[i]);
  } else if (index_type == 2) {
    const auto* indices = static_cast<const u8*>(source);
    for (u32 i = 0; i < count; i++)
      maximum = base::Max(maximum, static_cast<u32>(indices[i]));
  } else {
    const auto* indices = static_cast<const u16*>(source);
    for (u32 i = 0; i < count; i++)
      maximum = base::Max(maximum, static_cast<u32>(indices[i]));
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

void CopyGuestQuadIndices(void* destination,
                          const void* source,
                          u32 count,
                          u32 index_type) {
  const auto at = [&](u32 i) -> u32 {
    if (index_type == 1)
      return static_cast<const u32*>(source)[i];
    if (index_type == 2)
      return static_cast<const u8*>(source)[i];
    return static_cast<const u16*>(source)[i];
  };
  constexpr u32 kCorner[6] = {0, 1, 2, 0, 2, 3};
  u32 o = 0;
  for (u32 q = 0; q + 4 <= count; q += 4) {
    for (u32 c : kCorner) {
      const u32 index = at(q + c);
      if (index_type == 1)
        static_cast<u32*>(destination)[o++] = index;
      else
        static_cast<u16*>(destination)[o++] = static_cast<u16>(index);
    }
  }
}

}  // namespace gpu::render
