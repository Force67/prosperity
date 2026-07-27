/*
 * PS4Delta : PS4/PS5 emulation and research project
 */

#include "gpu/vulkan/vk_index_upload.h"

#include <algorithm>
#include <cstring>

namespace gpu::vk {

uint32_t GuestIndexElementBytes(uint32_t index_type) {
  return index_type == 1 ? 4 : index_type == 2 ? 1 : 2;
}

uint32_t UploadedIndexElementBytes(uint32_t index_type) {
  return index_type == 1 ? 4 : 2;
}

uint32_t MaxGuestIndex(const void* source,
                       uint32_t count,
                       uint32_t index_type) {
  uint32_t maximum = 0;
  if (index_type == 1) {
    const auto* indices = static_cast<const uint32_t*>(source);
    for (uint32_t i = 0; i < count; i++)
      maximum = std::max(maximum, indices[i]);
  } else if (index_type == 2) {
    const auto* indices = static_cast<const uint8_t*>(source);
    for (uint32_t i = 0; i < count; i++)
      maximum = std::max(maximum, static_cast<uint32_t>(indices[i]));
  } else {
    const auto* indices = static_cast<const uint16_t*>(source);
    for (uint32_t i = 0; i < count; i++)
      maximum = std::max(maximum, static_cast<uint32_t>(indices[i]));
  }
  return maximum;
}

void CopyGuestIndices(void* destination,
                      const void* source,
                      uint32_t count,
                      uint32_t index_type) {
  if (index_type == 1) {
    std::memcpy(destination, source, static_cast<size_t>(count) * 4);
  } else if (index_type == 2) {
    auto* output = static_cast<uint16_t*>(destination);
    const auto* input = static_cast<const uint8_t*>(source);
    for (uint32_t i = 0; i < count; i++)
      output[i] = input[i];
  } else {
    std::memcpy(destination, source, static_cast<size_t>(count) * 2);
  }
}

}  // namespace gpu::vk
